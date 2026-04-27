"""
tests/test_tour.py

Tests for repolix/tour.py and the answer_tour() function in llm.py.

All tests use tmp_path. No real ChromaDB or OpenAI calls are made.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from repolix.tour import (
    compute_inbound_counts,
    identify_entry_points,
    select_tour_chunks,
    build_tour_context,
    generate_tour,
    TOUR_MAX_CHUNKS,
)
from repolix.llm import answer_tour


# ── compute_inbound_counts ────────────────────────────────────────────────────

def test_compute_inbound_counts_basic():
    chunks = [
        {"name": "A", "calls": ["B", "C"]},
        {"name": "B", "calls": ["C"]},
        {"name": "C", "calls": []},
    ]
    counts = compute_inbound_counts(chunks)
    assert counts["B"] == 1
    assert counts["C"] == 2
    assert "A" not in counts


def test_compute_inbound_counts_empty():
    assert compute_inbound_counts([]) == {}


def test_compute_inbound_counts_no_calls():
    chunks = [
        {"name": "isolated", "calls": []},
        {"name": "also_isolated", "calls": []},
    ]
    assert compute_inbound_counts(chunks) == {}


def test_compute_inbound_counts_multiple_callers():
    chunks = [
        {"name": "A", "calls": ["Z"]},
        {"name": "B", "calls": ["Z"]},
        {"name": "C", "calls": ["Z"]},
    ]
    counts = compute_inbound_counts(chunks)
    assert counts["Z"] == 3


# ── identify_entry_points ─────────────────────────────────────────────────────

def test_identify_entry_points_heuristic():
    chunks = [
        {
            "name": "main",
            "file_rel_path": "main.py",
            "calls": ["run"],
            "node_type": "function",
        },
        {
            "name": "helper",
            "file_rel_path": "utils.py",
            "calls": [],
            "node_type": "function",
        },
    ]
    counts = {"run": 1}
    eps = identify_entry_points(chunks, counts)
    assert any(c["name"] == "main" for c in eps)


def test_identify_entry_points_entry_point_file():
    chunks = [
        {
            "name": "some_func",
            "file_rel_path": "cli.py",
            "calls": ["do_thing"],
            "node_type": "function",
        },
    ]
    counts = {}
    eps = identify_entry_points(chunks, counts)
    assert any(c["name"] == "some_func" for c in eps)


def test_identify_entry_points_graph_source():
    chunks = [
        {
            "name": "orphan_entry",
            "file_rel_path": "run.py",
            "calls": ["do_work"],
            "node_type": "function",
        },
        {
            "name": "do_work",
            "file_rel_path": "core.py",
            "calls": [],
            "node_type": "function",
        },
    ]
    counts = {"do_work": 1}
    eps = identify_entry_points(chunks, counts)
    assert any(c["name"] == "orphan_entry" for c in eps)


def test_identify_entry_points_caps_at_3():
    chunks = [
        {"name": "main", "file_rel_path": "main.py", "calls": ["a"], "node_type": "function"},
        {"name": "run", "file_rel_path": "run.py", "calls": ["b"], "node_type": "function"},
        {"name": "start", "file_rel_path": "start.py", "calls": ["c"], "node_type": "function"},
        {"name": "serve", "file_rel_path": "serve.py", "calls": ["d"], "node_type": "function"},
    ]
    counts = {}
    eps = identify_entry_points(chunks, counts)
    assert len(eps) <= 3


def test_identify_entry_points_no_duplicates():
    chunks = [
        {"name": "main", "file_rel_path": "main.py", "calls": ["a"], "node_type": "function"},
        {"name": "main", "file_rel_path": "main.py", "calls": ["b"], "node_type": "function"},
    ]
    counts = {}
    eps = identify_entry_points(chunks, counts)
    names = [c["name"] for c in eps]
    assert names.count("main") == 1


# ── select_tour_chunks ────────────────────────────────────────────────────────

def _make_chunk(name: str, file_rel_path: str, calls: list[str] | None = None) -> dict:
    return {
        "name": name,
        "file_rel_path": file_rel_path,
        "calls": calls or [],
        "node_type": "function",
        "start_line": 1,
        "end_line": 10,
        "docstring": "",
        "parent_class": "",
        "source_text": f"def {name}(): pass",
        "is_truncated": False,
        "file_path": f"/repo/{file_rel_path}",
    }


def test_select_tour_chunks_diversity():
    chunks = [
        _make_chunk(f"func_{i}", f"file_{i % 5}.py")
        for i in range(10)
    ]
    inbound = {c["name"]: 1 for c in chunks}
    entry_points: list[dict] = []

    selected = select_tour_chunks(chunks, inbound, entry_points)

    assert len(selected) <= TOUR_MAX_CHUNKS
    unique_files = {c["file_rel_path"] for c in selected}
    assert len(unique_files) >= 4


def test_select_tour_chunks_always_includes_entry_points():
    entry = _make_chunk("main", "main.py", calls=["do_thing"])
    chunks = [entry] + [_make_chunk(f"f{i}", f"f{i}.py") for i in range(10)]
    inbound: dict[str, int] = {}

    selected = select_tour_chunks(chunks, inbound, [entry])

    assert any(c["name"] == "main" for c in selected)


def test_select_tour_chunks_no_duplicates():
    chunks = [_make_chunk(f"func_{i}", f"file_{i}.py") for i in range(5)]
    inbound = {c["name"]: i for i, c in enumerate(chunks)}

    selected = select_tour_chunks(chunks, inbound, [])

    names = [c["name"] for c in selected]
    assert len(names) == len(set(names))


def test_select_tour_chunks_max_cap():
    chunks = [_make_chunk(f"func_{i}", f"file_{i}.py") for i in range(20)]
    inbound = {c["name"]: i for i, c in enumerate(chunks)}

    selected = select_tour_chunks(chunks, inbound, [])

    assert len(selected) <= TOUR_MAX_CHUNKS


# ── build_tour_context ────────────────────────────────────────────────────────

def test_build_tour_context_contains_repo_name():
    chunks = [_make_chunk("main", "main.py")]
    inbound: dict[str, int] = {}
    context = build_tour_context(chunks, inbound, "/home/user/myrepo")
    assert "myrepo" in context


def test_build_tour_context_contains_chunk_name():
    chunks = [_make_chunk("my_special_func", "utils.py")]
    inbound: dict[str, int] = {}
    context = build_tour_context(chunks, inbound, "/repo")
    assert "my_special_func" in context


def test_build_tour_context_scope_prefix():
    chunks = [_make_chunk("func", "src/core.py")]
    inbound: dict[str, int] = {}
    context = build_tour_context(chunks, inbound, "/repo", path_prefix="src/")
    assert "src/" in context


def test_build_tour_context_source_preview_truncated():
    source = "\n".join(f"line_{i}" for i in range(30))
    chunk = _make_chunk("long_func", "big.py")
    chunk["source_text"] = source
    inbound: dict[str, int] = {}
    context = build_tour_context([chunk], inbound, "/repo")
    assert "more lines" in context


# ── answer_tour ───────────────────────────────────────────────────────────────

def _make_mock_openai(content: str) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_answer_tour_parses_sections():
    briefing_text = (
        "OVERVIEW\n"
        "This repo is a widget factory.\n\n"
        "ENTRY POINTS\n"
        "main() in main.py\n\n"
        "MAJOR MODULES\n"
        "widget.py handles widget creation.\n\n"
        "KEY ABSTRACTIONS\n"
        "Widget class is the core abstraction.\n\n"
        "START HERE\n"
        "Start with main.py, then widget.py.\n"
    )
    mock_client = _make_mock_openai(briefing_text)
    result = answer_tour("some context", mock_client)

    assert result["briefing"] == briefing_text
    sections = result["briefing_sections"]
    assert sections["overview"] is not None
    assert sections["entry_points"] is not None
    assert sections["major_modules"] is not None
    assert sections["key_abstractions"] is not None
    assert sections["start_here"] is not None


def test_answer_tour_fallback_on_missing_sections():
    plain_prose = "This is just plain prose with no headers at all."
    mock_client = _make_mock_openai(plain_prose)
    result = answer_tour("some context", mock_client)

    assert result["briefing"] == plain_prose
    sections = result["briefing_sections"]
    assert sections["overview"] is None
    assert sections["entry_points"] is None
    assert sections["major_modules"] is None
    assert sections["key_abstractions"] is None
    assert sections["start_here"] is None


def test_answer_tour_partial_sections():
    partial = (
        "OVERVIEW\n"
        "Handles payments processing.\n\n"
        "KEY ABSTRACTIONS\n"
        "PaymentProcessor is central.\n"
    )
    mock_client = _make_mock_openai(partial)
    result = answer_tour("context", mock_client)

    sections = result["briefing_sections"]
    assert sections["overview"] is not None
    assert sections["key_abstractions"] is not None
    assert sections["entry_points"] is None
    assert sections["major_modules"] is None
    assert sections["start_here"] is None


# ── generate_tour ─────────────────────────────────────────────────────────────

def test_generate_tour_no_index_returns_error(tmp_path):
    mock_client = MagicMock()
    result = generate_tour(
        store_path=tmp_path / "nonexistent_store",
        repo_path=tmp_path,
        openai_client=mock_client,
    )
    assert result["error"] is not None
    assert result["briefing"] is None
    assert result["chunk_count"] == 0
    mock_client.chat.completions.create.assert_not_called()


def test_generate_tour_zero_embedding_calls(tmp_path):
    """Tour must never call embeddings.create — only chat.completions."""
    store_path = tmp_path / ".repolix"
    store_path.mkdir()
    (store_path / "chroma.sqlite3").touch()

    briefing_text = (
        "OVERVIEW\nA test repo.\n\n"
        "ENTRY POINTS\nmain.py\n\n"
        "MAJOR MODULES\ncore.py\n\n"
        "KEY ABSTRACTIONS\nhelper()\n\n"
        "START HERE\nmain.py\n"
    )
    mock_client = _make_mock_openai(briefing_text)

    chunks_data = {
        "ids": ["id1", "id2"],
        "metadatas": [
            {
                "name": "main",
                "node_type": "function",
                "file_rel_path": "main.py",
                "file_path": str(tmp_path / "main.py"),
                "start_line": 1,
                "end_line": 10,
                "calls": "helper",
                "docstring": "",
                "parent_class": "",
                "source_text": "def main(): pass",
                "is_truncated": False,
            },
            {
                "name": "helper",
                "node_type": "function",
                "file_rel_path": "core.py",
                "file_path": str(tmp_path / "core.py"),
                "start_line": 1,
                "end_line": 5,
                "calls": "",
                "docstring": "",
                "parent_class": "",
                "source_text": "def helper(): pass",
                "is_truncated": False,
            },
        ],
    }

    mock_col = MagicMock()
    mock_col.get.return_value = chunks_data
    mock_db = MagicMock()
    mock_db.get_or_create_collection.return_value = mock_col

    with patch("repolix.tour._get_client", return_value=mock_db):
        result = generate_tour(
            store_path=store_path,
            repo_path=tmp_path,
            openai_client=mock_client,
        )

    assert result["error"] is None
    assert result["chunk_count"] == 2
    assert mock_client.chat.completions.create.call_count == 1
    mock_client.embeddings.create.assert_not_called()


def test_generate_tour_no_chunks_returns_error(tmp_path):
    store_path = tmp_path / ".repolix"
    store_path.mkdir()
    (store_path / "chroma.sqlite3").touch()

    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.get.return_value = {"ids": [], "metadatas": []}
    mock_db = MagicMock()
    mock_db.get_or_create_collection.return_value = mock_col

    with patch("repolix.tour._get_client", return_value=mock_db):
        result = generate_tour(
            store_path=store_path,
            repo_path=tmp_path,
            openai_client=mock_client,
        )

    assert result["error"] is not None
    assert result["briefing"] is None
    mock_client.chat.completions.create.assert_not_called()


def test_generate_tour_path_prefix_scoping(tmp_path):
    store_path = tmp_path / ".repolix"
    store_path.mkdir()
    (store_path / "chroma.sqlite3").touch()

    briefing_text = "OVERVIEW\nScoped.\n\nENTRY POINTS\ncli.py\n\nMAJOR MODULES\ncli.py\n\nKEY ABSTRACTIONS\nmain()\n\nSTART HERE\ncli.py\n"
    mock_client = _make_mock_openai(briefing_text)

    chunks_data = {
        "ids": ["id1", "id2"],
        "metadatas": [
            {
                "name": "main",
                "node_type": "function",
                "file_rel_path": "repolix/cli.py",
                "file_path": str(tmp_path / "repolix/cli.py"),
                "start_line": 1,
                "end_line": 10,
                "calls": "",
                "docstring": "",
                "parent_class": "",
                "source_text": "def main(): pass",
                "is_truncated": False,
            },
            {
                "name": "build_ui",
                "node_type": "function",
                "file_rel_path": "frontend/src/App.tsx",
                "file_path": str(tmp_path / "frontend/src/App.tsx"),
                "start_line": 1,
                "end_line": 20,
                "calls": "",
                "docstring": "",
                "parent_class": "",
                "source_text": "function buildUi() {}",
                "is_truncated": False,
            },
        ],
    }

    mock_col = MagicMock()
    mock_col.get.return_value = chunks_data
    mock_db = MagicMock()
    mock_db.get_or_create_collection.return_value = mock_col

    with patch("repolix.tour._get_client", return_value=mock_db):
        result = generate_tour(
            store_path=store_path,
            repo_path=tmp_path,
            openai_client=mock_client,
            path_prefix="repolix/",
        )

    assert result["error"] is None
    assert result["chunk_count"] == 1
