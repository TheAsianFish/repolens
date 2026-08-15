# repolix — Project Context

Paste this file at the top of every new Cursor chat and every new
Claude conversation to restore full project context instantly.
Update this file at the end of every milestone before moving on.

---

## New chat — start here

**Next work: Milestone 24 — repolix 0.3.1 local embeddings via Ollama.**
Do not restyle the UI, do not add tour/trace to the SPA, do not
mix extra CLI commands into that PR. One feature per version.

Latest on PyPI: **0.2.4** (https://pypi.org/project/repolix/0.2.4/).
Package version is **0.3.0** (Ollama generation) — publish when
asked. Working tree should be clean except local `.env` / `.repolix/`.

Keep `__version__` in `repolix/__init__.py` identical to
`pyproject.toml` `[project].version`. 0.2.4 fixed a drift (it was
stuck at 0.2.2 while PyPI was 0.2.3).

Twine: `.env` has `twine=pypi-...` (gitignored). Upload with
`TWINE_USERNAME=__token__` and `TWINE_PASSWORD` from that value.
Never paste tokens into chat. Never `twine upload dist/*` — old
wheels sit in `dist/`; upload only `dist/repolix-X.Y.Z*`.

Git remote still points at `TheAsianFish/repolens.git`; GitHub
redirects to `TheAsianFish/repolix.git`. Push works; update the
remote URL when convenient.

Full plan, resume-claim audit, and Ollama constraints:
"Current development line (0.2.4 → 0.3.x)" below.

---

## What repolix is

A local-first codebase context engine. Point it at any Python or
JavaScript/TypeScript repo, ask plain English questions, get back answers
with exact file and line number citations. Code never leaves the user's machine.
Free and open source. Built for developer tooling.

Published on PyPI as `repolix` (previously developed under the name
`codesight`; renamed before public launch).

### Resume claims (must stay true)

These three bullets are the interview baseline. Do not contradict them
in code or docs. Bullet 3 generation half is true as of 0.3.0. Bullet 3
embeddings half (and "code never leaves") waits on 0.3.1.

- Published a local-first AI developer tool on PyPI that answers
  natural-language questions about Python, JavaScript, and TypeScript
  repos with exact file and line citations. **True as of 0.2.4**,
  except "local-first" still sends source to OpenAI embeddings until
  0.3.1. Generation can stay on-machine via `--provider ollama`.
- Engineered an AST-aware retrieval pipeline with Tree-sitter that
  fuses vector and keyword search via Reciprocal Rank Fusion and
  expands context through call graphs to ground LLM responses. **True.**
  Call graph is static name-level callees, not full program analysis.
- Integrated Ollama inference and SHA-256 incremental indexing,
  eliminating external LLM inference costs while cutting repeat
  embedding spend ∼95% on small incremental updates. **Ollama
  generation is implemented (0.3.0). SHA-256 and the ~95% story
  are true (see measurement below). Local embeddings are 0.3.1.**

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Ecosystem, tooling, pip distribution |
| AST parsing | Tree-sitter | Fast, accurate, multi-language ready |
| Embeddings | text-embedding-3-small (OpenAI) | Current default. 0.3.1 adds local embeddings via Ollama. Switching embed models requires a full re-index. |
| Vector store | ChromaDB (persistent, in-process) | Local-first, no server needed |
| LLM | gpt-5.4-mini (OpenAI) or Ollama | Default OpenAI. 0.3.0 adds `--provider ollama` for generation via `http://localhost:11434/v1`. Retrieval stays useful with `--no-llm`. |
| Web server | FastAPI | Async, simple, automatic validation |
| Frontend | React + TypeScript | SPA served by FastAPI from frontend/dist; dev via Vite at localhost:3000 |
| CLI | Click + Rich | Click handles commands/args; Rich handles styled terminal output |
| Install | pip install repolix | One command |

---

## Architecture decisions — locked in

**AST chunking over line splitting.**
Tree-sitter parses each file into a syntax tree. We split only at
function and class boundaries. Every chunk is a semantically complete
unit. This is the most important decision for retrieval quality.

**Class = one chunk. Methods chunked separately with parent_class.**
_walk_tree descends into class bodies so methods are chunked
individually with parent_class set to the enclosing class name.
This enables disambiguation between similarly named methods on
different classes. Example: AuthService.validate and
UserService.validate are distinct chunks with distinct parent_class.

**Module-level singleton for Parser and Tokenizer.**
_PARSER and _TOKENIZER are module-level in chunker.py. Creating
them per file call was wasteful. They are stateless between calls.

**Enriched embedding text.**
We do not embed raw source alone. We prepend node type and name,
then docstring if present, then source. This improves retrieval
because natural language queries map better to natural language
descriptions than to raw syntax.

**Metadata on every chunk.**
Each chunk carries: file_path, file_rel_path, node_type, name,
source, start_line, end_line, token_count, calls, docstring,
parent_class, is_truncated.
calls is stored in ChromaDB as a comma-joined string because
ChromaDB metadata must be primitive types. Split on read, join
on write. parent_class and docstring stored as empty string when
None — ChromaDB does not accept None metadata values.
is_truncated is stored as bool — True when source was cut at the
300-token cap. Surfaced as [truncated] in CLI citation output.

**Incremental indexing via SHA-256 file hashing.**
Every file gets a SHA-256 hash stored in ChromaDB alongside its
chunks. Re-indexing skips files whose hash has not changed.
Only changed files are re-embedded and re-stored.

**Orphan cleanup on every index run.**
After processing all current files, index_repo compares the full
set of stored hash IDs against the walked file set. Any stored path
not present in the current walk (deleted, renamed, moved) is an
orphan — its chunks and its hash entry are deleted. stats["cleaned"]
records the count. The CLI surfaces this only when cleaned > 0 to
avoid noise on normal runs. This prevents stale code from deleted
files appearing in query results.

**Hybrid search: vector similarity + keyword search, merged via RRF.**
Pure vector search misses exact name matches. Pure keyword search
misses semantic similarity. We do both and merge results using
Reciprocal Rank Fusion (k=60). RRF operates on rank positions not
raw scores, sidestepping the normalization problem entirely.

**Keyword search minimum token length is 2 characters.**
Tokens of length 1 are filtered (match too broadly via $contains).
Tokens of length 2+ are included — this covers common Python
identifiers like os, db, id, fn that were previously silently
dropped by the old > 2 guard.

**Call graph expansion after retrieval.**
After retrieving and ranking top N chunks, we inspect each chunk's
calls list and fetch any called functions not already in results
using exact keyword_search by name. Expansions get rerank_score=0.005
so they never displace primary results — they appear at the end.
Max 3 expansions per query to avoid context overflow.

**Metadata re-ranking as a second pass.**
Top 10 chunks retrieved. Re-ranked using metadata signals:
  +0.3 if query token appears in chunk name
  +0.2 if query token appears in file path stem
  +0.15 if query token appears in docstring
  +0.1 per query token appearing in calls list
Base score is rrf_score. Final score is base + boost.
Top 5 sent to LLM after expansion appended.

**Top 5 chunks max sent to LLM.**
Hard cap. 5 chunks * 300 tokens = 1500 tokens max context.
Safe for gpt-5.4-mini. Prevents context overflow.
Call graph expansions are appended after top 5, up to 3 additional.

**Chunk token cap: 300 tokens.**
Hard cap enforced in chunker.py using tiktoken cl100k_base
encoding — the same encoding gpt-5.4-mini uses internally.
Oversized chunks are truncated, not discarded. is_truncated=True
is set on the Chunk so downstream code and the user can detect it.
Metadata (calls, docstring, name) is extracted from the full AST
node before truncation — only source is cut.

**Line numbers are 1-indexed.**
Tree-sitter uses 0-indexed rows (C convention). We add 1 on
every start_point and end_point extraction. All citations the
user sees are 1-indexed.

**Citations use relative paths.**
file_rel_path is computed at index time relative to repo root
and stored as metadata. All citation output uses file_rel_path
not file_path to avoid exposing absolute paths in user-facing output.
If file_rel_path is missing or blank in stored metadata,
display_rel_path_from_meta() in retriever.py derives a short path
from the last two segments of file_path (never the full absolute
path as the display value). build_prompt, parse_citations, the CLI,
and the API chunk payload all use this helper.

**LLM uses max_completion_tokens, not max_tokens.**
gpt-5.4-mini requires max_completion_tokens. Using max_tokens
raises a 400 BadRequest error. All LLM calls use max_completion_tokens.

**LLM CITATIONS block is stripped before returning answer.**
parse_citations extracts inline citation labels [1], [2] etc. from
the full response text. After parsing, _strip_citations_block removes
everything from the first line starting with "CITATIONS" onward.
The CLI and API render their own formatted citation sections.
This prevents citations appearing twice and saves output tokens.

**Confidence label derived from top rerank_score.**
The CLI footer shows confidence: high/medium/low based on the top
retrieved chunk's rerank_score. Thresholds:
  high   >= 0.4   name or file path matched
  medium >= 0.15  docstring or call graph matched
  low    < 0.15   vector similarity only
This replaces the meaningless "N chunks used" counter.

**Query status messages replace progress bar.**
The CLI prints "Searching..." before retrieval and "Generating
answer..." before the LLM call. A 2-step progress bar was not a
real progress indicator and mislabeled the LLM step as "Retrieving".

**Rich library used for CLI output formatting.**
rich>=13.0.0 added as a dependency. Click handles command parsing;
Rich handles all terminal output: Panel for answers and index
summaries, Rule for citation separators, markup for dim/bold/cyan
styling. Console is created inside each command function (not at
module level) so Click's CliRunner can correctly capture output in
tests — CliRunner patches sys.stdout after module load, so a
module-level Console would hold a stale reference to real stdout.

**FastAPI serves the built React SPA as static files.**
api.py mounts frontend/dist at "/" after all API routes. A catch-all
GET /{full_path:path} route returns the requested file if it exists
in dist (JS, CSS, assets), otherwise returns index.html so React
Router handles client-side routing. Routes registered with @app.get()
take precedence over app.mount() in FastAPI's routing table, so all
API endpoints (/index, /query, /status, /health) are always matched
first. The mount is conditional on frontend/dist existing so the
server starts cleanly without a prior npm run build.
Development still uses bash start.sh (Vite dev server + backend).
The static file serving is for distribution: pip-installed users
have no Node.js runtime dependency.

---

## ChromaDB collections

| Collection | Purpose |
|---|---|
| repolix_chunks | Stores chunk source, embeddings, metadata |
| repolix_hashes | Stores one hash per file for incremental indexing |

ChromaDB persists to .repolix/ inside the indexed repo root.
.repolix/ is gitignored.

Chunk IDs: "{absolute_file_path}:{start_line}"
Hash IDs: "{absolute_file_path}"

---

## Module map

| File | Status | Responsibility |
|---|---|---|
| repolix/walker.py | Complete | Filesystem traversal, file filtering |
| repolix/chunker.py | Complete | AST parsing, chunk + metadata extraction, is_truncated flag |
| repolix/store.py | Complete | Embeddings, ChromaDB storage, retrieval, index_repo orchestrator; lookup_by_exact_name for exact symbol lookup |
| repolix/retriever.py | Complete | Hybrid search, RRF, re-ranking, call graph expansion; display_rel_path_from_meta for safe citation paths |
| repolix/llm.py | Complete | Prompt construction, chat completions (OpenAI or Ollama), citation parsing, CITATIONS block stripping; answer_trace for trace explanations |
| repolix/providers.py | Complete | Provider/model resolution, OpenAI embed client, Ollama LLM client via base_url, token-limit kwargs |
| repolix/tour.py | Complete | Call-graph analysis, entry point detection, chunk selection, context formatting, generate_tour orchestrator |
| repolix/trace.py | Complete | BFS forward trace, backward trace (reverse lookup), format_trace_tree, run_trace orchestrator; lookup_chunk_by_name → lookup_by_exact_name |
| repolix/cli.py | Complete | Click CLI — index, query, tour, and trace commands, confidence label |
| repolix/api.py | Complete | FastAPI backend — /index, /query, /tour, /trace, /status, /health; serves built SPA from frontend/dist |
| frontend/src/ | Complete | React SPA: index + query only. API has /tour and /trace; the SPA does not call them. Heavy inline styles. |
| tests/conftest.py | Complete | Creates minimal frontend/dist stub before TestClient initialises |

Note: repolix/embedder.py was deleted. It was an unimplemented stub;
the embedding logic lives in store.py as _embed_texts and build_embed_text.

---

## Test suite status

| File | Tests | Status |
|---|---|---|
| tests/test_walker.py | 11 | Passing |
| tests/test_chunker.py | 23 | Passing |
| tests/test_store.py | 33 | Passing |
| tests/test_retriever.py | 25 | Passing |
| tests/test_llm.py | 37 | Passing |
| tests/test_cli.py | 14 | Passing |
| tests/test_api.py | 11 | Passing |
| tests/test_tour.py | 24 | Passing |
| tests/test_trace.py | 21 | Passing |
| tests/test_providers.py | 17 | Passing |

Run all tests: pytest tests/ -v
Total: 241 passing

Note: test counts above are approximate. Always trust the actual
pytest output over this table.

---

## Milestone map

| # | Name | Status |
|---|---|---|
| 1 | Project scaffold + walker | Complete |
| 2 | AST chunker | Complete |
| 3 | Metadata extraction | Complete |
| 4 | Embedding pipeline + vector store | Complete |
| 5 | Basic retrieval | Complete |
| 6 | Hybrid search + re-ranking | Complete |
| 7 | LLM integration + citations | Complete |
| 8 | CLI | Complete |
| 9 | FastAPI backend + React frontend | Complete |
| 10 | Polish + ship | Complete |
| 11 | Post-V1 output quality + UX fixes | Complete |
| 12 | Rename codesight → repolix; publish to PyPI as repolix 0.1.0 | Complete |
| 13 | Rich CLI output polish + LLM system prompt update | Complete |
| 14 | repolix 0.1.1 — React UI polish, citation path fixes, loading states | Complete |
| 15 | V2-1: JavaScript and TypeScript indexing support | Complete |
| 16 | repolix 0.2.0 — PyPI minor release shipping JS/TS indexing | Complete |
| 17 | repolix 0.2.1 — Web UI same-origin fetches, CORS localhost/127.0.0.1, VITE_API_URL | Complete |
| 18 | LLM output layer: structured response format, section parsing, confidence gating | Complete |
| 19 | repolix 0.2.2 — tour command: proactive orientation briefing via call-graph analysis | Complete |
| 20 | repolix trace command: BFS call-graph traversal, forward/reverse/explain modes | Complete |
| 21 | repolix 0.2.3 — trace output quality: BUILTIN_NAMES filter + citation test coverage | Complete |
| 22 | repolix 0.2.4 — exact-name lookup for trace (`lookup_by_exact_name`) | Complete |
| 23 | repolix 0.3.0 — Ollama generation provider | Complete |
| 24 | repolix 0.3.1 — local embeddings via Ollama | Planned |
| 25 | repolix 0.3.2 — `repolix status` + richer GET /status | Planned |

V1 shipped as repolix 0.1.0 on PyPI; **0.1.1** followed (UI polish and fixes).
**0.2.2** shipped `repolix tour`. **0.2.3** shipped `repolix trace`.
**0.2.4** shipped exact-name lookup for `trace`. **0.3.0** shipped
Ollama generation. Next is **0.3.1** local embeddings.

---

## Milestone 13 — Rich CLI output + LLM system prompt

| Change | Files |
|---|---|
| Add rich>=13.0.0 to dependencies | pyproject.toml |
| Rewrite index command output: dim header, Rich Panel summary | repolix/cli.py |
| Rewrite query command output: dim status, cyan Answer Panel, Rule + citation list, dim confidence footer | repolix/cli.py |
| Console created inside each command function (not module-level) for CliRunner test compatibility | repolix/cli.py |
| Replace system prompt: direct navigation assistant tone, no hedging language, explicit next-search guidance | repolix/llm.py |
| Update test assertion "Index complete" → "Index Complete" to match Panel title casing | tests/test_cli.py |

---

## Post-V1 fixes (Milestone 11)

These were identified after V1 ship and resolved before V2 work begins.

| Fix | File(s) | Commit |
|---|---|---|
| max_tokens → max_completion_tokens for gpt-5.4-mini | llm.py, test_llm.py | de01083 |
| Delete unimplemented embedder.py stub | — | 96248de |
| Strip duplicate CITATIONS block from LLM answer | llm.py, test_llm.py | 230a44c |
| Surface is_truncated flag on chunked output | chunker.py, store.py, llm.py, cli.py | 041c695 |
| Allow 2-char tokens in keyword search (was > 2, now >= 2) | store.py, test_store.py | e140622 |
| Fix mock_openai_client to return N embeddings per N inputs | test_store.py | e140622 |
| Replace chunks-used footer with confidence label | cli.py, test_cli.py | cf015c8 |
| Replace fake 2-step progress bar with status messages | cli.py | a154c10 |
|| Disable noUnusedLocals/noUnusedParameters to unblock npm run build | frontend/tsconfig.json | 338c1e4 |
|| Serve built React SPA from FastAPI; catch-all route for client-side routing | repolix/api.py | 12a2ddd |
|| Add conftest.py to create minimal frontend/dist stub before TestClient initialises | tests/conftest.py | c1d1e69 |
|| Complete pyproject.toml for PyPI: authors, classifiers, tiktoken, package-data | pyproject.toml | 5514e3c |
|| Add MANIFEST.in for sdist completeness | MANIFEST.in | 56f50e1 |
|| Fix DIST_DIR to resolve from package dir when installed via pip | repolix/api.py | e90854a |

---

## Milestone 12 — Rename to repolix + PyPI launch

| Change | Files |
|---|---|
| Rename package folder codesight/ → repolix/ | all Python sources |
| Update all imports from codesight.* → repolix.* | repolix/*.py, tests/*.py |
| Rename CLI entrypoint codesight → repolix | pyproject.toml, cli.py |
| Rename ChromaDB collections to repolix_chunks / repolix_hashes | repolix/store.py |
| Rename store dir .codesight/ → .repolix/ | cli.py, api.py, tests/ |
| Update frontend: title, api.ts comment, package.json name | frontend/* |
| Rename pyproject.toml package name codesight → repolix | pyproject.toml |
| Update README, CONTEXT, MANIFEST.in, .gitignore, start.sh | docs/config |
| Published repolix 0.1.0 to PyPI | — |

---

## PyPI release sequence

**You cannot “update” a version already on PyPI.** Each upload must use a
**new version number** in `pyproject.toml` (e.g. patch `0.1.1` for fixes only,
or `0.2.0` for a minor release with new behavior). Old wheels/sdists stay
forever on the index. A CHANGELOG file is optional; PyPI shows **README.md**
on the project page. For this repo, keeping **CONTEXT.md** current is enough unless you want a public `CHANGELOG.md`.

Run these steps in order before every release:

  npm run build --prefix frontend
  rm -rf repolix/dist
  cp -R frontend/dist repolix/dist   # rm first; cp -r into an existing dest nests dist/dist
  python -m build
  twine check dist/repolix-X.Y.Z*
  twine upload dist/repolix-X.Y.Z*   # never dist/* — leftover old wheels live there

Keep `repolix/__init__.py` `__version__` in lockstep with pyproject.toml.

On PyPI, use an API token (not your password). Create one at
https://pypi.org/manage/account/token/ scoped to the repolix project.
This repo stores it in `.env` as `twine=pypi-...` (gitignored).
Twine username = `__token__`, password = that value. Never paste
tokens into chat. If a token was pasted in a chat, revoke it and
replace `.env`.

Test on TestPyPI first: twine upload --repository testpypi dist/*

---

## Milestone 15 — V2-1: JavaScript and TypeScript indexing

| Change | Files |
|---|---|
| Add .ts, .tsx, .js, .jsx to ALLOWED_EXTENSIONS | repolix/walker.py |
| Add EXTENSION_TO_LANGUAGE mapping | repolix/chunker.py |
| Replace _PARSER singleton with _PARSER_CACHE dict keyed by language | repolix/chunker.py |
| Add _get_cached_parser(language) helper using tree-sitter-javascript and tree-sitter-typescript | repolix/chunker.py |
| Add _extract_js_calls, _extract_js_name_from_parent, _handle_js_node helpers | repolix/chunker.py |
| Extend _walk_tree to dispatch to Python or JS/TS handlers based on language param | repolix/chunker.py |
| chunk_file returns [] for unknown extensions instead of raising ValueError | repolix/chunker.py |
| Add tree-sitter-javascript and tree-sitter-typescript to dependencies | pyproject.toml |
| Add TestJsChunking test class (12 tests); update test_raises_on_non_python_file | tests/test_chunker.py |
| JS/TS node types chunked: function_declaration, arrow_function, function_expression, class_declaration, method_definition | repolix/chunker.py |
| docstring="" for all JS/TS chunks (JSDoc extraction out of scope for V2-1) | repolix/chunker.py |
| tsx extension maps to separate "tsx" language key to select language_tsx() grammar | repolix/chunker.py |

---

## Milestone 18 — LLM output layer: structured response, section parsing, confidence gating

| Change | Files |
|---|---|
| Replace SYSTEM_PROMPT: senior engineer persona, bold-header structure (Answer/How it works/Where to look next) | repolix/llm.py |
| Add _parse_sections(): splits response on **Header:** boundaries; graceful fallback to full text | repolix/llm.py |
| Add confidence gating to answer_query(): reads results[0]["score"]; skips LLM when score < 0.15 | repolix/llm.py |
| Low-confidence path returns navigation dict with closest_matches and rephrasing suggestions | repolix/llm.py |
| Medium-confidence path (0.15–0.4) appends caution note to system prompt | repolix/llm.py |
| answer_query() now returns answer_sections, confidence, and navigation alongside existing keys | repolix/llm.py |
| Add score=0.5 default to make_result() so existing tests remain high-confidence | tests/test_llm.py |
| Add TestParseSections: full structure, no where_to_look, plain-prose fallback | tests/test_llm.py |
| Add TestAnswerQueryConfidence: low confidence (zero API calls), medium caution injection, sections returned | tests/test_llm.py |

---

## Milestone 19 — tour command: proactive orientation briefing

| Change | Files |
|---|---|
| Add repolix/tour.py: get_all_chunks, compute_inbound_counts, identify_entry_points, select_tour_chunks, build_tour_context, generate_tour | repolix/tour.py |
| Add TOUR_SYSTEM_PROMPT and answer_tour() to llm.py; move import re to module level | repolix/llm.py |
| Add tour CLI command: --path scope, --save flag, Rich panel with 5 sections + Most Referenced footer | repolix/cli.py |
| Add TourRequest, TourResponse Pydantic models and POST /tour endpoint | repolix/api.py |
| Add tests/test_tour.py: 24 tests covering all pipeline functions | tests/test_tour.py |

Key design decisions:
- Phase 1 (local): reads ChromaDB metadata only — no embeddings, no API calls
- Phase 2: single LLM chat completion call with tour-specific prompt
- inbound_counts is a reverse-adjacency count — O(n × avg_calls), no full graph needed
- Two-signal entry point detection: heuristic (file/function name) ranks above graph-source (zero inbound, nonzero outbound)
- Two-pass chunk selection: Pass 1 enforces one-chunk-per-file diversity; Pass 2 fills remaining slots
- build_tour_context accepts _all_chunks for top-function file-path lookup — prevents "unknown" for highly-referenced functions not in the selected 8
- frozenset for ENTRY_POINT_FILES and ENTRY_POINT_FUNCTIONS: O(1) membership test, immutable at module scope
- Lazy import of answer_tour inside generate_tour: defensive against future circular imports between tour.py and llm.py

---

## Milestone 20 — trace command: BFS call-graph traversal

| Change | Files |
|---|---|
| Add repolix/trace.py: lookup_chunk_by_name, forward_trace, backward_trace, format_trace_tree, run_trace | repolix/trace.py |
| Add TRACE_SYSTEM_PROMPT and answer_trace() to llm.py; placed after answer_tour() | repolix/llm.py |
| Add trace CLI command: --depth, --max-nodes, --reverse, --explain; Rich Panel tree + Rule callers section | repolix/cli.py |
| Add TraceRequest, TraceResponse Pydantic models and POST /trace endpoint | repolix/api.py |
| Add tests/test_trace.py: 20 tests covering all pipeline functions | tests/test_trace.py |

Key design decisions:
- forward_trace: BFS over calls edges using lookup_chunk_by_name (same lookup as expand_via_call_graph)
- Cycle detection: already-visited calls recorded inline during expansion (not on dequeue) so child_already_visited is populated correctly before the node is ever re-encountered
- backward_trace: O(n) scan via get_all_chunks from tour.py — no BFS, one level up only
- get_all_chunks import is lazy inside backward_trace (defensive against future circular imports between trace.py and tour.py)
- format_trace_tree: recursive Unicode tree renderer; already-visited shown inline as [already visited]; truncated nodes hint --depth
- run_trace: zero API calls by default; explain=True triggers single answer_trace() LLM call
- Patch target for get_all_chunks in tests is repolix.tour.get_all_chunks (lazy import pattern)
- max_nodes cap test requires explicit high max_depth to prevent depth limit firing before node cap

---

## Milestone 21 — trace output quality fixes

| Change | Files |
|---|---|
| Import BUILTIN_NAMES from repolix.tour at top level in trace.py | repolix/trace.py |
| Add `if call_name in BUILTIN_NAMES: continue` guard before lookup_chunk_by_name in forward_trace calls loop | repolix/trace.py |
| Add test_forward_trace_skips_builtins: asserts get/append/len never appear as tree nodes | tests/test_trace.py |
| Extend test_format_trace_tree_basic: assert `[file_rel_path:start_line]` present on every resolved node | tests/test_trace.py |

Key design decisions:
- BUILTIN_NAMES filter applied before lookup_chunk_by_name to avoid wasted ChromaDB roundtrips for names that will never resolve
- format_trace_tree already rendered citations via display_rel_path_from_meta; test coverage added to lock in the format
- BUILTIN_NAMES imported at module scope (not lazily) — it is a frozenset constant with no circular import risk
- 21 tests passing after changes

---

## Current development line (0.2.4 → 0.3.x)

Resume-driven sequence. One feature per version. CLI commands stay
`index`, `query`, `tour`, `trace` — do not rename. FastAPI stays the
HTTP/SPA backend; the CLI does not go through FastAPI.

**Honesty (current runtime vs product goal).** ChromaDB, Tree-sitter,
and keyword search are local. Indexing still sends enriched chunk text
to OpenAI embeddings. `query` search still embeds the question via
OpenAI, including `--no-llm`. Generation (`query` / `tour` /
`trace --explain`) can use OpenAI or `--provider ollama`.
"Code never leaves the machine" is the 0.3.1 goal, not the default
while embeddings stay on OpenAI.

**Call graph — do not overclaim.** Static, name-only callees extracted
at chunk time from AST call nodes (`foo()` → `foo`; `obj.bar()` → `bar`).
Not resolved through imports, types, or runtime. `tour`, `trace`, and
`expand_via_call_graph` all reuse the same `calls` list. Ambiguous names
pick one chunk (file_path + start_line). Dynamic dispatch, aliases,
`getattr`, JS prototypes: not handled. Interview phrasing: name-level
static callees stored per chunk, used to expand retrieval and walk a tree.

**~95% incremental embedding savings (measured 2026-08-15 on this repo).**
SHA-256 per file in `repolix_hashes`; unchanged files skip embed; orphans
cleaned. Walker found **22** indexable files (tests excluded). Unchanged
re-index: **22 skipped, 0 indexed** (100% of embedding calls skipped).
One changed file would skip 21/22 ≈ **95.5%**. Do not invent a new
experiment number without running it.

**Ollama generation (0.3.0, complete).** Same OpenAI SDK. Provider
`ollama` sets `base_url` to `http://localhost:11434/v1` (override
`REPOLIX_OLLAMA_BASE_URL`). Dummy api_key `ollama`. Chat uses
`max_tokens`; OpenAI chat still uses `max_completion_tokens`.
Default Ollama model: `llama3.2`. Embeddings stay OpenAI until 0.3.1.
Switching embedding models changes vector space — full re-index
required. Do not mix local embeddings into a later unrelated PR.

### Milestone 22 — 0.2.4 (complete)

Exact-name lookup so `repolix trace retrieve` cannot miss a real symbol
because `keyword_search(..., n_results=5)` ranked other documents first.
Shipped on PyPI as 0.2.4. `__version__` aligned with pyproject (was
stuck at 0.2.2).

| Change | Files | Status |
|---|---|---|
| Add `lookup_by_exact_name` (ChromaDB `where={"name": name}`) | `repolix/store.py` | Complete |
| `lookup_chunk_by_name` delegates to it | `repolix/trace.py` | Complete |
| Tests for exact lookup + keyword-cap regression | `tests/test_store.py`, `tests/test_trace.py` | Complete |
| Bump package version; sync `__version__` | `pyproject.toml`, `repolix/__init__.py` | Complete |
| Publish 0.2.4 to PyPI | — | Complete |

`expand_via_call_graph` still uses `keyword_search(n_results=3)` — same
class of bug; fix in a later polish pass, not this release.

Known leftover (not 0.2.4): `trace` tree `[file:line]` citations are
stripped inside Rich `Panel` because brackets are parsed as markup.
Callers below the panel render correctly. Fix with `rich.markup.escape`
on `tree_str` in a later polish pass.

### Milestone 23 — 0.3.0 Ollama generation (complete)

Closes resume bullet 3 as written ("Integrated Ollama inference")
without making embeddings local. OpenAI remains the default provider.
`index` and `query` search still need `OPENAI_API_KEY`. `tour` and
`trace --explain` with `--provider ollama` do not.

| Change | Files | Status |
|---|---|---|
| Provider/model helpers; Ollama client via `base_url` | `repolix/providers.py` | Complete |
| Pass `model`/`provider` into chat completions; `max_tokens` for Ollama | `repolix/llm.py` | Complete |
| `--provider` / `--model` on query, tour, trace | `repolix/cli.py` | Complete |
| Optional `provider`/`model` on query, tour, trace request bodies | `repolix/api.py` | Complete |
| Thread model/provider through tour and trace orchestrators | `repolix/tour.py`, `repolix/trace.py` | Complete |
| Provider tests + CLI/API/LLM coverage | `tests/test_providers.py` and others | Complete |
| Bump 0.3.0; sync `__version__` | `pyproject.toml`, `repolix/__init__.py` | Complete |
| README Ollama section + measured skip counts | `README.md` | Complete |

Flags also read `REPOLIX_LLM_PROVIDER`, `REPOLIX_LLM_MODEL`,
`REPOLIX_OLLAMA_BASE_URL`. Default Ollama model is `llama3.2`.
Do not publish until asked; PyPI 0.2.4 is still the live public
release until 0.3.0 is uploaded.

### Milestone 24 — 0.3.1 local embeddings

Makes "local-first" actually true. Ollama (or equivalent) embeddings
so `index` and vector search need no OpenAI. Re-index required when
switching embed models.

### Milestone 25 — 0.3.2 status

`repolix status`: repo, file/chunk counts, provider, model, index
freshness. Enrich `GET /status` (today it only returns `indexed: bool`).
Demo/interview polish, not a resume headline.

### After 0.3.2 (only if time)

- Use `lookup_by_exact_name` in `expand_via_call_graph`
- Escape Rich markup on `trace` trees
- Tiny retrieval eval on this repo (5–10 questions → expected files)
- MCP as a thin wrapper around `retrieve` / `lookup_by_exact_name` /
  `run_trace` — **0.4.0**, optional and additive. Do not redesign
  around MCP. A working CLI with no MCP must still make sense.
- `query --no-llm` plus `trace` already cover "context package for a
  task". A dedicated `context` command is optional later, not resume-critical.

### UI (do not restyle this cycle)

CLI is first-class. The SPA is a bonus: index, query, structured
answer, citations, confidence, chunk list. Dark theme via CSS
variables in `frontend/src/index.css`; most component styling is
inline. No light mode, no useful mobile layout (45%/55% grid).

`/tour` and `/trace` exist on FastAPI; `frontend/src/api.ts` only
wraps `/status`, `/index`, `/query`. That is a functionality gap,
not a CSS gap.

Do not put a visual redesign or tour/trace UI into 0.3.1. If UI
work happens, it is after 0.3.1, as its own version, functional
first (wire tour/trace, move inline styles to CSS, one mobile
breakpoint). Not a new palette or design system before interviews.

### Explicitly not this cycle

VS Code extension, Slack bot, GitHub webhooks, multi-repo, dependency
graph visualization, persistent query sessions, secret-pattern filter,
smart truncation. Move these to backlog; they are not resume blockers.

---

## V2 Roadmap

- TypeScript / JavaScript support (Tree-sitter parser swap) ✓ Done in V2-1
- `repolix tour` — proactive orientation briefing ✓ Done in V2-2
- `repolix trace` — call graph traversal for any named function ✓ Done in V2-3
- Exact-name lookup for `trace` (0.2.4) ✓
- Ollama generation (0.3.0) ✓
- Local embeddings via Ollama (0.3.1) — planned
- `repolix status` + richer GET /status (0.3.2) — planned

## Backlog (not 0.3.x)

- MCP server wrapping existing retrieve / lookup / trace (0.4.0 candidate)
- VS Code extension wrapper
- Dependency graph visualization
- Secret pattern filter in walker.py
- Smart truncation: preserve head + tail of oversized chunks
- Index-time warning for truncated chunks
- Persistent query sessions
- GitHub webhook integration (re-index on push)
- Multi-repo support
- Slack bot

---

## Conventions

- All file paths stored and compared as absolute resolved strings.
- Citations and user-facing output always use file_rel_path.
- Sorted output everywhere for deterministic behavior across runs.
- Tests are hermetic. Every test uses tmp_path. No test touches
  a real repository on disk.
- OpenAI calls are always mocked in tests. Never hit the network
  in a test.
- mock_openai_client uses side_effect to return one embedding per
  input text — not a fixed return_value — so multi-chunk tests work.
- Run pytest after every change before committing.
- Update CONTEXT.md at the end of every milestone or significant
  change. A milestone is not done until CONTEXT.md reflects it.

---

## Git commit conventions

Commit to GitHub after every meaningful unit of work including:
- A complete prompt execution
- A passing test suite for a new feature
- Any fix to a failing test
- Any update to CONTEXT.md
- Any architectural change

Format: conventional commits — ONE LINE ONLY, no body, no bullet points.
  feat: add call graph expansion to retriever
  fix: use file_rel_path in citation output
  refactor: make Parser singleton at module level
  test: add expansion tests to test_retriever
  docs: update CONTEXT.md for Milestone 11
  chore: add httpx to dev dependencies

Semicolons are allowed to join related items on the same line:
  feat: serve React SPA from FastAPI; add catch-all route for client-side routing

Rules:
- One logical change per commit.
- Commit message is exactly one line. No multi-line messages, no -m body flags.
- Never commit with a vague message like "update" or "fix stuff".
- Always run pytest tests/ -v before committing. Green only.
- Always run git status before git add. Verify .env is absent.
- Never use git add . — use git add <specific files>.
- Push to main after every commit unless instructed otherwise.

Sequence:
  pytest tests/ -v
  git status
  git add <specific files>
  git commit -m "type: description"
  git push origin main

---

## What not to do

- Do not split classes into separate method chunks without setting
  parent_class on each method.
- Do not use dirs = [...] in os.walk — use dirs[:] = [...].
- Do not store lists directly in ChromaDB metadata.
- Do not create a Parser or Tokenizer instance per file call.
- Do not embed raw source without enrichment.
- Do not skip the hash check on re-index unless force=True.
- Do not send more than 5 primary chunks to the LLM.
- Do not use 0-indexed line numbers in any user-facing output.
- Do not commit with .env present in git status output.
- Do not bundle unrelated changes into one commit.
- Do not leave CONTEXT.md outdated after a milestone completes.
- Do not use file_path in user-facing citation output — always
  use file_rel_path.
- Do not use max_tokens with gpt-5.4-mini — use max_completion_tokens.
- Do not use a fixed return_value for mock_openai_client — use
  side_effect so the mock returns the right number of embeddings
  for any batch size.
- Do not filter keyword search tokens with len > 2 — the correct
  guard is len >= 2 to preserve 2-character identifiers like os, db.
- Do not print the LLM's raw response as the answer — strip the
  CITATIONS block first via _strip_citations_block.
- Do not write multi-line git commit messages — one line only; use
  semicolons to join related items if needed.
- Do not add a catch-all GET route before the StaticFiles mount without
  making it file-aware — a plain index.html catch-all will serve HTML
  for JS/CSS requests and break the frontend.
- Do not create a module-level Rich Console — create it inside each
  command function so Click's CliRunner patches sys.stdout before the
  Console is constructed, ensuring test output capture works correctly.
- Do not pass citation label strings like [1] directly into Rich markup
  strings — use rich.markup.escape() to prevent them being interpreted
  as markup tags.
- Do not patch "repolix.store.chunk_file" in tests — chunk_file is
  imported locally inside index_repo, so patch "repolix.chunker.chunk_file".
- Do not patch "repolix.trace.get_all_chunks" — get_all_chunks is lazily
  imported inside backward_trace from repolix.tour, so patch
  "repolix.tour.get_all_chunks".
- Do not rely on BFS dequeue to detect already-visited cycle nodes in
  forward_trace — already-visited calls must be recorded in
  child_already_visited during the expansion phase, not on dequeue, because
  already-visited names are never re-enqueued.
- Do not test max_nodes cap with the default max_depth=3 — the depth limit
  will terminate traversal before the node cap fires on short chains.
- Do not call lookup_chunk_by_name for names in BUILTIN_NAMES — filter them
  with `if call_name in BUILTIN_NAMES: continue` before the lookup to avoid
  wasted ChromaDB roundtrips and spurious tree nodes.
- Do not use keyword_search to look up a chunk by function name —
  keyword_search is substring matching on document text with an n_results
  cap. Common identifiers (retrieve, query, index) appear in many chunks
  and the real function can fall outside the cap. Use lookup_by_exact_name
  (ChromaDB where={"name": name}) instead.
- Do not mix local embeddings into a generation-only change —
  embeddings are 0.3.1. One feature per version.
- Do not rewrite cli/api/store/llm around a new LLM SDK for Ollama —
  keep the OpenAI client and set base_url to Ollama's compatible endpoint.
- Do not claim the call graph resolves definitions, imports, or runtime
  dispatch — it is static name-level callees stored on each chunk.
- Do not add MCP, a VS Code extension, or extra CLI commands until
  0.3.1 makes the local-first claim true.
- Do not pass trace tree_str into a Rich Panel without
  rich.markup.escape() — `[file:line]` is parsed as markup and vanishes.
- Do not restyle the React SPA or add tour/trace screens before 0.3.1.
- Do not let `repolix/__init__.py` `__version__` drift from
  pyproject.toml — bump both in the same commit.
- Do not `twine upload dist/*` — upload only the version just built.
- Do not `cp -r frontend/dist repolix/dist` when dest already exists —
  that nests `repolix/dist/dist`. `rm -rf repolix/dist` first, then
  `cp -R frontend/dist repolix/dist`.
