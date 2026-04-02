"""
retriever.py

Orchestrates the full retrieval pipeline for a user query.

Milestone 5: basic vector retrieval with metadata re-ranking.
Milestone 6 will add keyword search and merge it with vector results.

Pipeline:
  1. Receive plain English query string
  2. Retrieve top 10 chunks by vector similarity via store.query_chunks
  3. Re-rank using metadata signals
  4. Return top 5 chunks with all metadata
"""

from pathlib import Path
from openai import OpenAI
from repolens.store import query_chunks

# Number of chunks to retrieve from vector search before re-ranking.
# We intentionally over-retrieve so re-ranking has room to work.
# Retrieving 10 and returning 5 gives the re-ranker meaningful signal.
RETRIEVE_N = 10

# Number of chunks to return to the caller after re-ranking.
RETURN_N = 5


def retrieve(
    query: str,
    store_path: str | Path,
    openai_client: OpenAI,
) -> list[dict]:
    """
    Run the full retrieval pipeline for a plain English query.

    Retrieves the top RETRIEVE_N chunks by vector similarity, re-ranks
    them using metadata signals, and returns the top RETURN_N.

    Args:
        query: Plain English question from the user.
        store_path: Path to the ChromaDB persistence directory.
        openai_client: Initialized OpenAI client.

    Returns:
        List of up to RETURN_N result dicts, each containing:
            source, file_path, name, node_type, start_line,
            end_line, calls, docstring, distance, rerank_score.
        Sorted by rerank_score descending.
    """
    # Step 1: vector similarity retrieval
    results = query_chunks(
        query_text=query,
        store_path=store_path,
        openai_client=openai_client,
        n_results=RETRIEVE_N,
    )

    if not results:
        return []

    # Step 2: re-rank
    ranked = rerank(query, results)

    # Step 3: return top N
    return ranked[:RETURN_N]


def rerank(query: str, results: list[dict]) -> list[dict]:
    """
    Re-rank retrieved chunks using metadata signals.

    Vector similarity distance is the primary signal but is imperfect.
    We boost chunks whose name or file path contains query terms,
    and chunks whose docstring contains query terms. This corrects
    cases where an exactly-named function ranks below a tangentially
    related one due to embedding distance noise.

    Scoring (additive):
      - Base score: 1.0 - distance  (higher is more similar)
      - +0.3 if any query token appears in the chunk name
      - +0.2 if any query token appears in the file path stem
      - +0.15 if any query token appears in the docstring
      - +0.1 per query token that appears in the calls list

    All boosts are additive and unbounded above 1.0 — a chunk can
    score above 1.0 if it matches on multiple signals. This is
    intentional: a chunk that is both semantically similar AND
    name-matched should rank clearly above one that is only
    semantically similar.

    Args:
        query: The original plain English query string.
        results: List of result dicts from query_chunks.

    Returns:
        Results sorted by rerank_score descending, with rerank_score
        added to each dict.
    """
    # Normalize query into lowercase tokens for matching.
    # Split on whitespace and strip punctuation so "authentication,"
    # matches "authentication".
    query_tokens = [
        t.strip(".,?!:;\"'()[]{}").lower()
        for t in query.split()
        if t.strip(".,?!:;\"'()[]{}").lower()
    ]

    scored = []
    for result in results:
        # ChromaDB cosine distance: 0.0 = identical, 2.0 = opposite.
        # Convert to similarity: higher is better.
        base_score = 1.0 - result["distance"]

        name = result["name"].lower()
        file_stem = Path(result["file_path"]).stem.lower()
        docstring = (result["docstring"] or "").lower()
        calls = [c.lower() for c in result["calls"]]

        boost = 0.0
        for token in query_tokens:
            if token in name:
                boost += 0.3
            if token in file_stem:
                boost += 0.2
            if token in docstring:
                boost += 0.15
            if any(token in call for call in calls):
                boost += 0.1

        rerank_score = base_score + boost
        scored.append({**result, "rerank_score": rerank_score})

    return sorted(scored, key=lambda r: r["rerank_score"], reverse=True)


def format_results(results: list[dict]) -> str:
    """
    Format retrieval results as a human-readable string for CLI output
    and LLM context construction.

    Each result is formatted as a labeled block showing location,
    chunk name, and source code. This is also the format we will
    pass to the LLM in Milestone 7.

    Args:
        results: List of result dicts from retrieve().

    Returns:
        Multi-line string ready for printing or passing to LLM.
    """
    if not results:
        return "No results found."

    lines: list[str] = []
    for i, result in enumerate(results, 1):
        file_path = result["file_path"]
        name = result["name"]
        start = result["start_line"]
        end = result["end_line"]
        score = result["rerank_score"]
        source = result["source"]

        lines.append(f"── Result {i} ──────────────────────────────")
        lines.append(f"File:     {file_path}")
        lines.append(f"Function: {name}  (lines {start}–{end})")
        lines.append(f"Score:    {score:.3f}")
        lines.append("")
        lines.append(source)
        lines.append("")

    return "\n".join(lines)
