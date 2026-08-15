"""
LLM provider selection for generation.

Embeddings stay on OpenAI in 0.3.0. Generation (query / tour /
trace --explain) can use OpenAI or Ollama. Ollama is reached with
the same OpenAI SDK pointed at the compatible /v1 endpoint.
"""

import os

from openai import OpenAI

PROVIDERS = ("openai", "ollama")
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class ProviderError(ValueError):
    """Invalid provider setting or missing credentials."""


def resolve_provider(value: str | None) -> str:
    """
    Resolve the generation provider.

    Order: explicit value, REPOLIX_LLM_PROVIDER, default openai.
    """
    raw = (value or os.getenv("REPOLIX_LLM_PROVIDER") or DEFAULT_PROVIDER)
    provider = raw.strip().lower()
    if provider not in PROVIDERS:
        raise ProviderError(
            f"Unknown LLM provider {raw!r}. Use openai or ollama."
        )
    return provider


def resolve_model(provider: str, model: str | None) -> str:
    """
    Resolve the generation model name.

    Order: explicit value, REPOLIX_LLM_MODEL, provider default.
    """
    raw = (model or os.getenv("REPOLIX_LLM_MODEL") or "").strip()
    if raw:
        return raw
    if provider == "ollama":
        return DEFAULT_OLLAMA_MODEL
    return DEFAULT_OPENAI_MODEL


def ollama_base_url() -> str:
    """OpenAI-compatible Ollama base URL, always ending in /v1."""
    raw = (
        os.getenv("REPOLIX_OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
    ).strip().rstrip("/")
    if not raw.endswith("/v1"):
        raw = f"{raw}/v1"
    return raw


def get_embed_client() -> OpenAI:
    """
    OpenAI client for embeddings.

    0.3.0 still embeds via OpenAI. Requires OPENAI_API_KEY.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ProviderError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or export it in your shell. "
            "Indexing and query search still use OpenAI embeddings."
        )
    return OpenAI(api_key=api_key)


def get_llm_client(provider: str) -> OpenAI:
    """
    OpenAI-SDK client for chat completions.

    Ollama does not need OPENAI_API_KEY. The SDK still requires
    some api_key string; Ollama ignores it.
    """
    provider = resolve_provider(provider)
    if provider == "ollama":
        return OpenAI(base_url=ollama_base_url(), api_key="ollama")
    return get_embed_client()


def completion_token_kwargs(provider: str, n: int) -> dict:
    """
    Token-limit kwarg for chat.completions.create.

    gpt-5.4-mini requires max_completion_tokens. Ollama's
    OpenAI-compatible endpoint supports max_tokens only.
    """
    if provider == "ollama":
        return {"max_tokens": n}
    return {"max_completion_tokens": n}
