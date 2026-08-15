"""
api.py

FastAPI backend for repolix. Exposes the indexing and query
pipeline over HTTP so the React frontend can consume it.

Endpoints:
  POST /index   — index a repository
  POST /query   — query an indexed repository
  GET  /status  — check if a repo has been indexed
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from repolix import __version__
from repolix.llm import answer_query
from repolix.providers import (
    ProviderError,
    get_llm_client,
    ollama_base_url,
    resolve_model,
    resolve_provider,
)
from repolix.retriever import display_rel_path_from_meta, retrieve
from repolix.store import index_repo

load_dotenv()


def resolve_repo_path(repo_path: str) -> Path:
    """
    Resolve repo_path to an absolute directory.

    If the client sends '.' or whitespace only, it means "the API process
    working directory". When REPOLIX_DEFAULT_REPO is set, '.' and empty
    strings use that path instead — useful if uvicorn was started outside
    the repo (see start.sh, which cds to the project root).
    """
    stripped = repo_path.strip()
    default_root = os.environ.get("REPOLIX_DEFAULT_REPO", "").strip()
    if stripped in (".", "") and default_root:
        return Path(default_root).expanduser().resolve()
    return Path(repo_path).expanduser().resolve()


# When installed via pip, the pre-built React bundle is copied into the
# repolix package directory (repolix/dist/) before building the wheel.
# In development, fall back to the Vite output at frontend/dist/.
_PKG_DIST = Path(__file__).parent / "dist"
_DEV_DIST = Path(__file__).parent.parent / "frontend" / "dist"
DIST_DIR = _PKG_DIST if _PKG_DIST.exists() else _DEV_DIST

app = FastAPI(
    title="repolix",
    description="Local-first codebase context engine",
    version=__version__,
)

# CORS for the Vite dev server and for any host/port mismatch (e.g. user
# opens 127.0.0.1:8000 while fetch used to hardcode localhost:8000).
_LOCAL_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_LOCAL_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_openai_client() -> OpenAI:
    """Create an authenticated OpenAI client from the environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set on the server.",
        )
    return OpenAI(api_key=api_key)


def resolve_generation(provider: str | None, model: str | None) -> tuple[str, str]:
    """Resolve provider/model or raise HTTP 400."""
    try:
        resolved = resolve_provider(provider)
        return resolved, resolve_model(resolved, model)
    except ProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def get_generation_client(provider: str, embed_client: OpenAI | None = None) -> OpenAI:
    """OpenAI client for chat, or Ollama via the same SDK."""
    if provider == "openai":
        return embed_client if embed_client is not None else get_openai_client()
    try:
        return get_llm_client(provider)
    except ProviderError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def ollama_http_error(exc: BaseException) -> HTTPException:
    return HTTPException(
        status_code=502,
        detail=(
            f"Ollama generation failed ({exc}). "
            f"Is Ollama running at {ollama_base_url()}? "
            "Pull a model with: ollama pull llama3.2"
        ),
    )


def get_store_path(repo_path: str) -> Path:
    """Resolve the ChromaDB store path for a given repo."""
    return Path(repo_path).resolve() / ".repolix"


# ── Request / Response Models ─────────────────────────────────────────────────
# Pydantic models define the shape of request bodies and response
# payloads. FastAPI uses them for automatic validation and serialization.
# If a request body doesn't match the model, FastAPI returns a 422
# error before your handler function is ever called.

class IndexRequest(BaseModel):
    repo_path: str
    force: bool = False


class IndexResponse(BaseModel):
    total_files: int
    indexed: int
    skipped: int
    total_chunks: int
    errors: list[str]


class QueryRequest(BaseModel):
    question: str
    repo_path: str
    no_llm: bool = False
    provider: str | None = None
    model: str | None = None


class CitationModel(BaseModel):
    label: str
    file_rel_path: str
    start_line: int
    end_line: int
    name: str
    parent_class: str | None


class ChunkModel(BaseModel):
    source: str
    file_rel_path: str
    name: str
    start_line: int
    end_line: int
    rerank_score: float
    parent_class: str | None


class QueryResponse(BaseModel):
    answer: str | None
    citations: list[CitationModel]
    chunks: list[ChunkModel]
    chunks_used: int


class StatusResponse(BaseModel):
    indexed: bool
    store_path: str
    repo_path: str


class TourRequest(BaseModel):
    repo_path: str
    path_prefix: str | None = None
    provider: str | None = None
    model: str | None = None


class TourResponse(BaseModel):
    briefing: str | None
    briefing_sections: dict | None
    entry_points: list[dict]
    top_functions: list[list]
    chunk_count: int
    error: str | None


class TraceRequest(BaseModel):
    symbol: str
    repo_path: str
    max_depth: int = 3
    max_nodes: int = 20
    include_backward: bool = True
    explain: bool = False
    provider: str | None = None
    model: str | None = None


class TraceResponse(BaseModel):
    symbol: str
    tree_str: str
    backward: list[dict]
    visited_count: int
    truncated: bool
    explanation: str | None
    error: str | None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/index", response_model=IndexResponse)
async def index_endpoint(request: IndexRequest):
    """
    Index a repository.

    Walks the repo, chunks every Python file, embeds the chunks,
    and stores everything in ChromaDB. Skips unchanged files unless
    force=True.
    """
    repo_path = resolve_repo_path(request.repo_path)
    if not repo_path.exists() or not repo_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"repo_path does not exist or is not a directory: {repo_path}",
        )

    store_path = get_store_path(str(repo_path))
    store_path.mkdir(parents=True, exist_ok=True)
    client = get_openai_client()

    stats = index_repo(
        repo_path=repo_path,
        store_path=store_path,
        openai_client=client,
        force=request.force,
        exclude_tests=True,
    )

    return IndexResponse(**stats)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query an indexed repository with a plain English question.

    Returns a structured response with the LLM answer, citations,
    and the raw retrieved chunks for display in the frontend.
    """
    repo_path = resolve_repo_path(request.repo_path)
    store_path = get_store_path(str(repo_path))

    if not (store_path / "chroma.sqlite3").exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No index at {store_path} (missing chroma.sqlite3). "
                f"Resolved repo: {repo_path}. "
                "Run POST /index or `repolix index` for that path. "
                "Note: '.' is the API server working directory unless "
                "REPOLIX_DEFAULT_REPO is set."
            ),
        )

    client = get_openai_client()
    provider, model = resolve_generation(request.provider, request.model)

    results = retrieve(
        query=request.question,
        store_path=store_path,
        openai_client=client,
    )

    chunks = [
        ChunkModel(
            source=r["source"],
            file_rel_path=display_rel_path_from_meta(r),
            name=r["name"],
            start_line=r["start_line"],
            end_line=r["end_line"],
            rerank_score=r.get("rerank_score", 0.0),
            parent_class=r.get("parent_class"),
        )
        for r in results
    ]

    if request.no_llm or not results:
        return QueryResponse(
            answer=None,
            citations=[],
            chunks=chunks,
            chunks_used=0,
        )

    llm_client = get_generation_client(provider, embed_client=client)
    try:
        output = answer_query(
            query=request.question,
            results=results,
            openai_client=llm_client,
            model=model,
            provider=provider,
        )
    except (APIConnectionError, APIStatusError) as exc:
        if provider == "ollama":
            raise ollama_http_error(exc) from exc
        raise

    citations = [CitationModel(**c) for c in output["citations"]]

    return QueryResponse(
        answer=output["answer"],
        citations=citations,
        chunks=chunks,
        chunks_used=output["chunks_used"],
    )


@app.post("/tour", response_model=TourResponse)
def tour_endpoint(req: TourRequest):
    """
    Generate a proactive orientation briefing for an indexed repository.

    Analyzes chunk metadata in ChromaDB (no vector search, no embeddings)
    and produces a structured briefing via a single LLM call.
    """
    from repolix.tour import generate_tour
    repo_path = Path(req.repo_path).resolve()
    store_path = repo_path / ".repolix"
    provider, model = resolve_generation(req.provider, req.model)
    client = get_generation_client(provider)
    try:
        result = generate_tour(
            store_path=store_path,
            repo_path=repo_path,
            openai_client=client,
            path_prefix=req.path_prefix,
            model=model,
            provider=provider,
        )
    except (APIConnectionError, APIStatusError) as exc:
        if provider == "ollama":
            raise ollama_http_error(exc) from exc
        raise
    return TourResponse(**result)


@app.post("/trace", response_model=TraceResponse)
def trace_endpoint(req: TraceRequest):
    """
    Trace the call graph for a named function or class.

    Returns the forward call tree, list of callers, and optionally
    an LLM explanation. Zero API calls unless explain=True.
    """
    from repolix.trace import run_trace
    repo_path = Path(req.repo_path).resolve()
    store_path = repo_path / ".repolix"
    provider, model = resolve_generation(req.provider, req.model)
    client = get_generation_client(provider) if req.explain else None
    try:
        result = run_trace(
            symbol=req.symbol,
            store_path=store_path,
            max_depth=req.max_depth,
            max_nodes=req.max_nodes,
            include_backward=req.include_backward,
            openai_client=client,
            explain=req.explain,
            model=model,
            provider=provider,
        )
    except (APIConnectionError, APIStatusError) as exc:
        if provider == "ollama":
            raise ollama_http_error(exc) from exc
        raise
    return TraceResponse(
        symbol=result["symbol"],
        tree_str=result["tree_str"],
        backward=result["backward"],
        visited_count=result["forward"].get("visited_count", 0),
        truncated=result["forward"].get("truncated", False),
        explanation=result.get("explanation"),
        error=result.get("error"),
    )


@app.get("/status", response_model=StatusResponse)
async def status_endpoint(repo_path: str):
    """
    Check whether a repository has been indexed.

    Returns indexed=True if a ChromaDB store exists at the expected
    path for the given repo. Used by the frontend to show whether
    indexing is needed before querying.
    """
    resolved = resolve_repo_path(repo_path)
    store_path = get_store_path(str(resolved))
    indexed = (store_path / "chroma.sqlite3").exists()

    return StatusResponse(
        indexed=indexed,
        store_path=str(store_path),
        repo_path=str(resolved),
    )


@app.get("/health")
async def health():
    """Health check endpoint. Returns 200 if the server is running."""
    return {"status": "ok", "version": __version__}


# ── SPA catch-all ─────────────────────────────────────────────────────────────
# Must come AFTER all API routes so /index, /query, /status, /health are matched
# first. Serves the requested file if it exists in frontend/dist (JS, CSS,
# assets), otherwise returns index.html so React Router handles client-side
# routing for deep-link paths like /dashboard or /profile.

@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str):
    target = DIST_DIR / full_path
    if target.is_file():
        return FileResponse(str(target))
    index_html = DIST_DIR / "index.html"
    if not index_html.exists():
        raise HTTPException(
            status_code=503,
            detail="Frontend not built. Run: cd frontend && npm run build",
        )
    return HTMLResponse(index_html.read_text())


# Mount static files after all routes. Routes take precedence in FastAPI's
# routing table, so all API paths and the catch-all above are matched first.
# The mount provides explicit static-file serving infrastructure and is used
# when the catch-all delegates to FileResponse for direct asset paths.
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")
