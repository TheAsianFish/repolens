# codecompass

**Ask plain English questions about any Python codebase. Get answers
with exact file and line citations. Runs entirely on your machine.**
```bash
codecompass index ./myrepo
codecompass query "how does authentication work"
```
Searching...
Generating answer...
── Answer ────────────────────────────────────────────────
authenticate_user() validates credentials by calling validate_token()
[1] which checks expiry and signature. On success it creates a
session via SessionService.create() [2].
── Citations ─────────────────────────────────────────────
[1] auth/validators.py:14-28    (validate_token)
[2] auth/session.py:45-67       (SessionService.create)

[confidence: high · 5 chunks · index: ./myrepo/.codecompass]

Your code never leaves your machine. No server. No accounts beyond
an OpenAI API key.

---

## Why codecompass

Getting dropped into an unfamiliar codebase is painful. Documentation
is outdated. Grep finds strings, not meaning. LLM chatbots hallucinate
file names and function signatures because they have no access to your
actual code.

codecompass indexes your code locally using AST-based chunking — every
retrieved chunk is a complete function or class, never an arbitrary
line slice. It runs entirely on your machine.

---

## How it works

**1. AST chunking**
Tree-sitter parses each file into a syntax tree. codecompass splits only
at function and class boundaries. Every chunk is semantically complete.
Methods are tracked with their parent class for disambiguation.

**2. Hybrid search**
Queries run against OpenAI embeddings (vector search) and exact token
matching (keyword search) simultaneously. Results are merged using
Reciprocal Rank Fusion — a ranking algorithm that rewards consistency
across search methods over dominance in just one.

**3. Call graph expansion**
After initial retrieval, codecompass inspects each retrieved chunk's
call graph and fetches called functions that did not rank highly
enough on their own. This surfaces implementation details that live
one function call away from the entry point.

**4. Metadata re-ranking**
Retrieved chunks are re-ranked using function names, file paths,
docstrings, and call graph signals before being sent to the LLM.

**5. Cited answers**
The top chunks go to gpt-5.4-mini with instructions to synthesize
across all chunks and cite every claim. Citations map back to exact
file paths and line numbers.

---

## Quickstart

### Requirements

- Python 3.11+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

> Node.js is **not required** for end users. The web UI is bundled
> inside the package and served directly by FastAPI.

### Install from PyPI
```bash
pip install codecompass
```

Set your API key:
```bash
export OPENAI_API_KEY=sk-your-key-here
# or add it to a .env file in your working directory
```

### Install from source (development)
```bash
git clone https://github.com/TheAsianFish/codecompass
cd codecompass
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### CLI
```bash
# Index a repository (~$0.02 per 30k lines, one-time)
codecompass index ./path/to/repo

# Ask a question
codecompass query "how does authentication work"

# See raw retrieved chunks without an LLM call
codecompass query "where is UserService defined" --no-llm

# Force re-index all files after a major refactor
codecompass index ./path/to/repo --force
```

### Web UI
```bash
# Start the server (the React UI is bundled — no npm needed)
uvicorn codecompass.api:app --port 8000
# Open http://localhost:8000
```

**For frontend development** (hot reload via Vite):
```bash
# Requires Node.js 18+
cd frontend && npm install && cd ..
bash start.sh
# Backend: http://localhost:8000  Frontend: http://localhost:3000
```

---

## Cost

| Action | Cost |
|---|---|
| Index 30k line repo | ~$0.02 (one-time) |
| Re-index after small change | ~$0.001 (changed files only) |
| Each query | ~$0.001 |

Incremental indexing means re-indexing after a small change costs
almost nothing — only changed files are re-embedded.

---

## Stack

| Layer | Choice |
|---|---|
| AST parsing | Tree-sitter |
| Embeddings | text-embedding-3-small |
| Vector store | ChromaDB (local, no server needed) |
| LLM | gpt-5.4-mini |
| Backend | FastAPI |
| Frontend | React + TypeScript |
| CLI | Click |

---

## Output

Each query produces:

- A prose answer with inline citations `[1]`, `[2]` etc.
- A citations section with exact file paths and line ranges.
  Citations marked `[truncated]` mean the function exceeded the
  300-token chunk cap — the answer is based on a partial view of
  that function.
- A confidence label (`high` / `medium` / `low`) derived from how
  strongly the retrieved chunks matched the query across function
  names, file paths, docstrings, and call graph signals.

---

## Limitations

- Python repos only. TypeScript support planned for V2.
- Best on repos up to ~30k lines.
- Deeply nested functions are included in their parent chunk.
- Large functions (>300 tokens) are truncated at the chunk cap.
  The `[truncated]` marker in citations flags when this occurs.
- Complex cross-file reasoning may require rephrasing the query.
- Architecture-level questions (layer structure, dependency graphs)
  require the V2 dependency graph feature to answer reliably.

---

## Roadmap

**V2** — TypeScript support, VS Code extension, dependency graph

**V3** — GitHub webhook re-indexing, multi-repo, Slack bot

---

## Contributing

Bug reports and pull requests are welcome. Please open an issue
before submitting a large change so we can discuss the approach.

See .github/ISSUE_TEMPLATE/bug_report.md for the bug report format.

---

## License

MIT © 2026 Patrick Chung
