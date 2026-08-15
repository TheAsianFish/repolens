"""
Tests for repolix/providers.py.

Hermetic: no network. OpenAI client construction is local only.
"""

import pytest

from repolix.providers import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    ProviderError,
    completion_token_kwargs,
    get_embed_client,
    get_llm_client,
    ollama_base_url,
    resolve_model,
    resolve_provider,
)


class TestResolveProvider:

    def test_default_is_openai(self, monkeypatch):
        monkeypatch.delenv("REPOLIX_LLM_PROVIDER", raising=False)
        assert resolve_provider(None) == "openai"

    def test_explicit_value_wins(self, monkeypatch):
        monkeypatch.setenv("REPOLIX_LLM_PROVIDER", "openai")
        assert resolve_provider("ollama") == "ollama"

    def test_env_used_when_value_missing(self, monkeypatch):
        monkeypatch.setenv("REPOLIX_LLM_PROVIDER", "ollama")
        assert resolve_provider(None) == "ollama"

    def test_case_insensitive(self):
        assert resolve_provider("Ollama") == "ollama"

    def test_unknown_provider_raises(self):
        with pytest.raises(ProviderError, match="Unknown LLM provider"):
            resolve_provider("anthropic")


class TestResolveModel:

    def test_openai_default(self, monkeypatch):
        monkeypatch.delenv("REPOLIX_LLM_MODEL", raising=False)
        assert resolve_model("openai", None) == DEFAULT_OPENAI_MODEL

    def test_ollama_default(self, monkeypatch):
        monkeypatch.delenv("REPOLIX_LLM_MODEL", raising=False)
        assert resolve_model("ollama", None) == DEFAULT_OLLAMA_MODEL

    def test_explicit_model_wins(self, monkeypatch):
        monkeypatch.setenv("REPOLIX_LLM_MODEL", "ignored")
        assert resolve_model("ollama", "qwen2.5") == "qwen2.5"

    def test_env_model_when_unspecified(self, monkeypatch):
        monkeypatch.setenv("REPOLIX_LLM_MODEL", "mistral")
        assert resolve_model("ollama", None) == "mistral"


class TestOllamaBaseUrl:

    def test_default(self, monkeypatch):
        monkeypatch.delenv("REPOLIX_OLLAMA_BASE_URL", raising=False)
        assert ollama_base_url() == DEFAULT_OLLAMA_BASE_URL

    def test_appends_v1_when_missing(self, monkeypatch):
        monkeypatch.setenv("REPOLIX_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        assert ollama_base_url() == "http://127.0.0.1:11434/v1"

    def test_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv(
            "REPOLIX_OLLAMA_BASE_URL", "http://localhost:11434/v1/"
        )
        assert ollama_base_url() == "http://localhost:11434/v1"


class TestCompletionTokenKwargs:

    def test_openai_uses_max_completion_tokens(self):
        assert completion_token_kwargs("openai", 1024) == {
            "max_completion_tokens": 1024
        }

    def test_ollama_uses_max_tokens(self):
        assert completion_token_kwargs("ollama", 512) == {"max_tokens": 512}


class TestClients:

    def test_embed_client_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
            get_embed_client()

    def test_ollama_llm_client_does_not_need_openai_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("REPOLIX_OLLAMA_BASE_URL", raising=False)
        client = get_llm_client("ollama")
        assert "11434/v1" in str(client.base_url)

    def test_openai_llm_client_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
            get_llm_client("openai")
