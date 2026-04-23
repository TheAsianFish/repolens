"""
Tests for repolix/llm.py.

OpenAI calls are mocked. We test prompt construction, citation
parsing, and the full answer_query pipeline without hitting the API.
"""

import pytest
from unittest.mock import MagicMock
from repolix.llm import (
    build_prompt,
    parse_citations,
    answer_query,
    _strip_citations_block,
    _parse_sections,
    MAX_CONTEXT_CHUNKS,
)


def make_result(**kwargs) -> dict:
    defaults = dict(
        source="def authenticate_user(token):\n    return True",
        file_path="/repo/auth.py",
        file_rel_path="auth.py",
        name="authenticate_user",
        node_type="function_definition",
        start_line=1,
        end_line=2,
        calls=["validate_token"],
        docstring="Validates user credentials.",
        parent_class=None,
        distance=0.2,
        rrf_score=0.02,
        rerank_score=0.32,
        score=0.5,
    )
    defaults.update(kwargs)
    return defaults


def mock_openai(response_text: str) -> MagicMock:
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    client.chat.completions.create.return_value.choices = [choice]
    return client


class TestBuildPrompt:

    def test_labels_chunks_sequentially(self):
        results = [make_result(name=f"func_{i}") for i in range(3)]
        prompt, labeled = build_prompt("query", results)
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt

    def test_labeled_chunks_length_matches_input(self):
        results = [make_result() for _ in range(3)]
        _, labeled = build_prompt("query", results)
        assert len(labeled) == 3

    def test_caps_at_max_context_chunks(self):
        results = [make_result(name=f"func_{i}") for i in range(10)]
        _, labeled = build_prompt("query", results)
        assert len(labeled) == MAX_CONTEXT_CHUNKS

    def test_prompt_contains_query(self):
        results = [make_result()]
        prompt, _ = build_prompt("how does auth work", results)
        assert "how does auth work" in prompt

    def test_prompt_contains_file_path(self):
        results = [make_result(file_rel_path="auth.py")]
        prompt, _ = build_prompt("query", results)
        assert "auth.py" in prompt

    def test_prompt_fallback_rel_when_file_rel_path_empty(self):
        results = [make_result(file_rel_path="", file_path="/repo/auth.py")]
        prompt, labeled = build_prompt("query", results)
        assert "repo/auth.py" in prompt
        assert labeled[0]["file_rel_path"] == "repo/auth.py"

    def test_prompt_contains_line_numbers(self):
        results = [make_result(start_line=10, end_line=20)]
        prompt, _ = build_prompt("query", results)
        assert "10" in prompt
        assert "20" in prompt

    def test_parent_class_shown_in_prompt(self):
        results = [make_result(parent_class="AuthService")]
        prompt, _ = build_prompt("query", results)
        assert "AuthService" in prompt

    def test_empty_results_returns_empty_labeled(self):
        prompt, labeled = build_prompt("query", [])
        assert labeled == []


class TestParseCitations:

    def test_extracts_single_citation(self):
        labeled = [{"label": "[1]", "file_rel_path": "auth.py",
                    "start_line": 1, "end_line": 5,
                    "name": "foo", "parent_class": None}]
        citations = parse_citations("See [1] for details.", labeled)
        assert len(citations) == 1
        assert citations[0]["label"] == "[1]"

    def test_extracts_multiple_citations(self):
        labeled = [
            {"label": "[1]", "file_rel_path": "a.py",
             "start_line": 1, "end_line": 2,
             "name": "foo", "parent_class": None},
            {"label": "[2]", "file_rel_path": "b.py",
             "start_line": 3, "end_line": 4,
             "name": "bar", "parent_class": None},
        ]
        citations = parse_citations("See [1] and [2].", labeled)
        assert len(citations) == 2

    def test_deduplicates_repeated_citations(self):
        labeled = [{"label": "[1]", "file_rel_path": "a.py",
                    "start_line": 1, "end_line": 2,
                    "name": "foo", "parent_class": None}]
        citations = parse_citations("[1] does this, [1] also does that.", labeled)
        assert len(citations) == 1

    def test_ignores_labels_not_in_labeled_chunks(self):
        labeled = [{"label": "[1]", "file_rel_path": "a.py",
                    "start_line": 1, "end_line": 2,
                    "name": "foo", "parent_class": None}]
        citations = parse_citations("See [1] and [9].", labeled)
        assert len(citations) == 1

    def test_no_citations_returns_empty(self):
        citations = parse_citations("No citations here.", [])
        assert citations == []

    def test_citations_ordered_by_label_number(self):
        labeled = [
            {"label": "[1]", "file_rel_path": "a.py",
             "start_line": 1, "end_line": 2,
             "name": "foo", "parent_class": None},
            {"label": "[2]", "file_rel_path": "b.py",
             "start_line": 3, "end_line": 4,
             "name": "bar", "parent_class": None},
        ]
        citations = parse_citations("[2] and [1].", labeled)
        assert citations[0]["label"] == "[1]"
        assert citations[1]["label"] == "[2]"

    def test_parse_citations_normalizes_empty_file_rel_path(self):
        labeled = [{
            "label": "[1]",
            "file_rel_path": "",
            "file_path": "/abs/repo/pkg/mod.py",
            "start_line": 1,
            "end_line": 2,
            "name": "foo",
            "parent_class": None,
        }]
        citations = parse_citations("See [1].", labeled)
        assert len(citations) == 1
        assert citations[0]["file_rel_path"] == "pkg/mod.py"
        assert "file_path" in citations[0]
        assert citations[0]["name"] == "foo"


class TestStripCitationsBlock:

    def test_strips_citations_header(self):
        text = "Auth works via [1].\n\nCITATIONS\n[1] auth.py lines 1-5"
        assert _strip_citations_block(text) == "Auth works via [1]."

    def test_strips_citations_with_colon(self):
        text = "Answer [1].\n\nCITATIONS:\n[1] auth.py lines 1-5"
        assert _strip_citations_block(text) == "Answer [1]."

    def test_strips_citations_case_insensitive(self):
        text = "Answer [1].\n\ncitations\n[1] auth.py"
        assert _strip_citations_block(text) == "Answer [1]."

    def test_no_citations_block_returns_unchanged(self):
        text = "Auth happens in [1]. No citations block here."
        assert _strip_citations_block(text) == text

    def test_empty_string_returns_empty(self):
        assert _strip_citations_block("") == ""

    def test_trailing_whitespace_stripped(self):
        text = "Answer [1].   \n\nCITATIONS\n[1] auth.py"
        assert _strip_citations_block(text) == "Answer [1]."


class TestAnswerQuery:

    def test_returns_answer_text(self):
        client = mock_openai("Auth happens in [1].")
        results = [make_result()]
        output = answer_query("how does auth work", results, client)
        assert "Auth happens in [1]." in output["answer"]

    def test_citations_block_stripped_from_answer(self):
        raw = "Auth happens in [1].\n\nCITATIONS\n[1] auth.py lines 1-2"
        client = mock_openai(raw)
        results = [make_result()]
        output = answer_query("how does auth work", results, client)
        assert "CITATIONS" not in output["answer"]
        assert "Auth happens in [1]." in output["answer"]

    def test_returns_citations(self):
        client = mock_openai("See [1] for details.")
        results = [make_result()]
        output = answer_query("query", results, client)
        assert len(output["citations"]) == 1

    def test_chunks_used_count_is_correct(self):
        client = mock_openai("Answer [1].")
        results = [make_result() for _ in range(3)]
        output = answer_query("query", results, client)
        assert output["chunks_used"] == 3

    def test_empty_results_returns_no_results_message(self):
        client = mock_openai("")
        output = answer_query("query", [], client)
        assert "No relevant code" in output["answer"]
        assert output["citations"] == []
        assert output["chunks_used"] == 0

    def test_openai_called_with_correct_model(self):
        client = mock_openai("Answer.")
        results = [make_result()]
        answer_query("query", results, client)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5.4-mini"

    def test_temperature_is_low(self):
        client = mock_openai("Answer.")
        results = [make_result()]
        answer_query("query", results, client)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] <= 0.1

    def test_max_tokens_is_set(self):
        client = mock_openai("Answer.")
        results = [make_result()]
        answer_query("query", results, client)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert "max_completion_tokens" in call_kwargs
        assert call_kwargs["max_completion_tokens"] == 1024


class TestParseSections:

    def test_parse_sections_full_structure(self):
        text = (
            "**Answer:** Auth is handled in auth.py by authenticate_user [1].\n\n"
            "**How it works:** The function validates a JWT token against a secret key. "
            "It raises AuthError on failure so callers get a clear exception rather than "
            "a boolean, making misuse harder [1].\n\n"
            "**Where to look next:** Check validate_token in token.py for the low-level "
            "verification logic."
        )
        sections = _parse_sections(text)
        assert sections["answer"] is not None
        assert "authenticate_user" in sections["answer"]
        assert sections["how_it_works"] is not None
        assert "JWT" in sections["how_it_works"]
        assert sections["where_to_look"] is not None
        assert "validate_token" in sections["where_to_look"]

    def test_parse_sections_no_where_to_look(self):
        text = (
            "**Answer:** Truncation happens in store.py inside _truncate_source [1].\n\n"
            "**How it works:** It splits the source on newlines and rejoins the first "
            "MAX_LINES, appending a comment marker so readers know the code was cut."
        )
        sections = _parse_sections(text)
        assert sections["answer"] is not None
        assert sections["how_it_works"] is not None
        assert sections["where_to_look"] is None

    def test_parse_sections_fallback(self):
        plain = "This is just plain prose with no headers at all."
        sections = _parse_sections(plain)
        assert sections["answer"] == plain
        assert sections["how_it_works"] is None
        assert sections["where_to_look"] is None


class TestAnswerQueryConfidence:

    def test_answer_query_low_confidence(self):
        client = mock_openai("Should not be called.")
        results = [make_result(score=0.05)]
        output = answer_query("query", results, client)
        assert output["answer"] is None
        assert output["confidence"] == "low"
        assert output["navigation"] is not None
        assert client.chat.completions.create.call_count == 0

    def test_answer_query_medium_confidence_injects_caution(self):
        structured = (
            "**Answer:** Auth lives in auth.py [1].\n\n"
            "**How it works:** It calls validate_token which checks a JWT.\n\n"
            "CITATIONS\n[1] auth.py:1-2 (authenticate_user)"
        )
        client = mock_openai(structured)
        results = [make_result(score=0.25)]
        answer_query("query", results, client)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        system_msg = call_kwargs["messages"][0]["content"]
        assert "Retrieval confidence is moderate" in system_msg

    def test_answer_query_returns_sections(self):
        structured = (
            "**Answer:** Auth lives in auth.py [1].\n\n"
            "**How it works:** It validates a JWT token using a secret key [1].\n\n"
            "CITATIONS\n[1] auth.py:1-2 (authenticate_user)"
        )
        client = mock_openai(structured)
        results = [make_result(score=0.5)]
        output = answer_query("query", results, client)
        assert output["answer_sections"]["answer"] is not None
        assert output["answer_sections"]["how_it_works"] is not None
