"""Minimal LLM interface plus an OpenAI-compatible HTTP client (works with OpenAI, Ollama, vLLM, LM Studio, ...)."""

from __future__ import annotations

import os
from typing import Protocol

import httpx


class LLM(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class LLMError(Exception):
    pass


class OpenAICompatibleLLM:
    """Chat-completions client. Configure with arguments or AGENTGATE_LLM_BASE_URL / _API_KEY / _MODEL."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("AGENTGATE_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("AGENTGATE_LLM_API_KEY", "")
        self.model = model or os.getenv("AGENTGATE_LLM_MODEL", "")
        if not self.model:
            raise LLMError("no model configured (pass model= or set AGENTGATE_LLM_MODEL)")
        self.temperature = temperature
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, messages: list[dict[str, str]]) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {"model": self.model, "messages": messages, "temperature": self.temperature}
        try:
            r = self._client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
            r.raise_for_status()
            return str(r.json()["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise LLMError(f"LLM request failed: {e}") from e
