"""
Tests for repolens/retriever.py.

store.query_chunks is mocked — tests never hit ChromaDB or OpenAI.
We test the re-ranking logic and format_results output directly.
"""

import pytest
from unittest.mock import MagicMock, patch
from repolens.retriever import retrieve, rerank, format_results, RETURN_N


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_result(**kwargs) -> dict:
    """Return a result dict with sensible defaults, overridable via kwargs."""
    defaults = dict(
        source="def authenticate_user(token):\n    return True",
        file_path="/repo/auth.py",
        name="authenticate_user",
        node_type="function_definition",
        start_line=1,
        end_line=2,
        calls=["validate_token"],
        docstring="Validates user credentials.",
        distance=0.2,
    )
    defaults.update(kwargs)
    return defaults


# ── rerank ────────────────────────────────────────────────────────────────────

class TestRerank:

    def test_base_score_is_one_minus_distance(self):
        results = [make_result(distance=0.2)]
        ranked = rerank("anything", results)
        # Base score = 1.0 - 0.2 = 0.8, no token matches
        assert abs(ranked[0]["rerank_score"] - 0.8) < 0.01

    def test_name_match_boosts_score(self):
        no_match = make_result(name="unrelated_function", distance=0.2)
        match = make_result(name="authenticate_user", distance=0.25)
        ranked = rerank("authenticate", [no_match, match])
        # match has worse distance but name boost should push it higher
        assert ranked[0]["name"] == "authenticate_user"

    def test_file_path_match_boosts_score(self):
        no_match = make_result(file_path="/repo/utils.py", distance=0.2)
        match = make_result(file_path="/repo/auth.py", distance=0.25)
        ranked = rerank("auth", [no_match, match])
        assert ranked[0]["file_path"] == "/repo/auth.py"

    def test_docstring_match_boosts_score(self):
        no_doc = make_result(docstring=None, distance=0.2)
        with_doc = make_result(
            docstring="Validates user credentials.",
            distance=0.25,
            name="other_func",
        )
        ranked = rerank("validates", [no_doc, with_doc])
        assert ranked[0]["name"] == "other_func"

    def test_calls_match_boosts_score(self):
        no_calls = make_result(calls=[], distance=0.2, name="func_a")
        with_calls = make_result(
            calls=["validate_token"],
            distance=0.25,
            name="func_b",
        )
        ranked = rerank("validate", [no_calls, with_calls])
        assert ranked[0]["name"] == "func_b"

    def test_rerank_score_added_to_each_result(self):
        results = [make_result(), make_result(name="other")]
        ranked = rerank("query", results)
        assert all("rerank_score" in r for r in ranked)

    def test_results_sorted_descending_by_rerank_score(self):
        results = [make_result(distance=0.5), make_result(distance=0.1)]
        ranked = rerank("query", results)
        scores = [r["rerank_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_results_returns_empty(self):
        assert rerank("query", []) == []

    def test_punctuation_stripped_from_query_tokens(self):
        results = [make_result(name="authenticate_user", distance=0.3)]
        ranked = rerank("authenticate,", results)
        # Should still match despite trailing comma
        assert ranked[0]["rerank_score"] > 0.3


# ── retrieve ──────────────────────────────────────────────────────────────────

class TestRetrieve:

    def test_returns_at_most_return_n_results(self):
        many_results = [make_result(name=f"func_{i}") for i in range(10)]
        with patch("repolens.retriever.query_chunks", return_value=many_results):
            results = retrieve("query", "/fake/db", MagicMock())
        assert len(results) <= RETURN_N

    def test_returns_empty_list_when_no_results(self):
        with patch("repolens.retriever.query_chunks", return_value=[]):
            results = retrieve("query", "/fake/db", MagicMock())
        assert results == []

    def test_results_have_rerank_score(self):
        raw = [make_result(name=f"func_{i}") for i in range(3)]
        with patch("repolens.retriever.query_chunks", return_value=raw):
            results = retrieve("query", "/fake/db", MagicMock())
        assert all("rerank_score" in r for r in results)


# ── format_results ────────────────────────────────────────────────────────────

class TestFormatResults:

    def test_empty_results_returns_no_results_message(self):
        output = format_results([])
        assert "No results found" in output

    def test_output_contains_file_path(self):
        results = [rerank("q", [make_result()])[0]]
        output = format_results(results)
        assert "/repo/auth.py" in output

    def test_output_contains_function_name(self):
        results = [rerank("q", [make_result()])[0]]
        output = format_results(results)
        assert "authenticate_user" in output

    def test_output_contains_line_numbers(self):
        results = [rerank("q", [make_result(start_line=10, end_line=20)])[0]]
        output = format_results(results)
        assert "10" in output
        assert "20" in output

    def test_output_contains_source(self):
        results = [rerank("q", [make_result()])[0]]
        output = format_results(results)
        assert "def authenticate_user" in output

    def test_multiple_results_numbered(self):
        raw = [make_result(name=f"func_{i}", distance=0.1 * i) for i in range(3)]
        ranked = rerank("query", raw)
        output = format_results(ranked)
        assert "Result 1" in output
        assert "Result 2" in output
        assert "Result 3" in output
