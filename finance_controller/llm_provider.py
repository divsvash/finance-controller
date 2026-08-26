"""Real LLM provider behind the existing LLMClient protocol.

Uses the OpenAI-compatible Chat Completions API over stdlib urllib
(no SDK dependency added). Transport is injectable so tests stay offline.

Configuration (env vars, never hard-coded secrets):
  FINANCE_LLM_API_KEY   required; clear error if missing
  FINANCE_LLM_MODEL     default "gpt-4o-mini" (cheap default; override)
  FINANCE_LLM_BASE_URL  default "https://api.openai.com/v1"

The provider only transports text: prompt in -> raw JSON string out.
All schema validation remains in llm_investigator.py.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
REQUEST_TIMEOUT_SECONDS = 30


class ProviderError(RuntimeError):
    """Transport-level failure (key missing, auth, timeout, network,
    rate limit, malformed/empty response). Wrapped by llm_investigator
    into LLMInvestigatorError at the boundary."""


class MissingAPIKeyError(ProviderError):
    pass


def _redact(text: str) -> str:
    """Strip any occurrence of the API key from error/log strings."""
    key = os.environ.get("FINANCE_LLM_API_KEY", "")
    return text.replace(key, "<redacted>") if key else text


class OpenAICompatibleClient:
    """Implements finance_controller.llm_investigator.LLMClient."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        transport: Any | None = None,
        timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._api_key = api_key if api_key is not None \
            else os.environ.get("FINANCE_LLM_API_KEY")
        if not self._api_key:
            raise MissingAPIKeyError(
                "FINANCE_LLM_API_KEY is not set. Configure it in your "
                "environment before using the real LLM provider.")
        self._model = model or os.environ.get(
            "FINANCE_LLM_MODEL", DEFAULT_MODEL)
        self._base_url = (base_url or os.environ.get(
            "FINANCE_LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        # Injected transport: callable(method, url, headers, body_bytes,
        # timeout) -> (status:int, body:bytes). Defaults to urllib.
        self._transport = transport or self._urllib_transport
        self._timeout = timeout_seconds

    # -- repr must never leak the key --
    def __repr__(self) -> str:
        return (f"<OpenAICompatibleClient model={self._model!r} "
                f"base_url={self._base_url!r} api_key=<redacted>>")

    @staticmethod
    def _urllib_transport(method, url, headers, body, timeout):
        req = urllib.request.Request(url, data=body, headers=headers,
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except urllib.error.URLError as e:
            raise ProviderError(f"connection failure: {e.reason}") from e
        except TimeoutError as e:
            raise ProviderError(f"request timed out after {timeout}s") from e

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system",
                 "content": "Return only valid JSON per the user's schema."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},  # structured output
            "temperature": 0,
        }
        body = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        status, raw = self._transport(
            "POST", f"{self._base_url}/chat/completions",
            headers, body, self._timeout)

        if status == 401 or status == 403:
            raise ProviderError(
                _redact(f"authentication failed (HTTP {status}): "
                        f"{raw.decode(errors='replace')}"))
        if status == 429:
            raise ProviderError("rate limited by provider (HTTP 429); "
                                "retry later (automatic retries disabled "
                                "by design)")
        if status != 200:
            raise ProviderError(_redact(
                f"provider returned HTTP {status}: "
                f"{raw.decode(errors='replace')[:500]}"))
        try:
            obj = json.loads(raw.decode())
        except (ValueError, UnicodeDecodeError) as e:
            raise ProviderError(f"malformed provider response: {e}") from e
        try:
            content = obj["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"unexpected provider shape: {e}") from e
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("provider returned empty content")
        return content
