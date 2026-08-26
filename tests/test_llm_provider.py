import json
import os

import pytest

from finance_controller.llm_investigator import (
    LLMInvestigatorError, llm_investigate_case)
from finance_controller.llm_provider import (
    MissingAPIKeyError, OpenAICompatibleClient, ProviderError)

# helpers shared from conftest: golden_mismatch_case()


VALID_CONTENT = json.dumps({
    "finding": "f", "explanation": "e", "recommended_action": "a",
    "confidence": "high", "evidence_used": ["internal_amount"],
    "warnings": []})


def ok_body(content):
    return json.dumps(
        {"choices": [{"message": {"content": content}}]}).encode()


def make_client(monkeypatch, transport, key="sk-secret-key-123"):
    monkeypatch.setenv("FINANCE_LLM_API_KEY", key)
    return OpenAICompatibleClient(transport=transport)


# 1 success path returns model text
def test_success_returns_content(monkeypatch):
    captured = {}
    def tr(method, url, headers, body, timeout):
        captured["url"] = url; captured["headers"] = dict(headers)
        captured["body"] = json.loads(body)
        return 200, ok_body(VALID_CONTENT)
    c = make_client(monkeypatch, tr)
    assert c.generate("PROMPT") == VALID_CONTENT
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer sk-secret-key-123"
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert "Never invent facts" in captured["body"]["messages"][1]["content"]


# 2 missing API key fails clearly
def test_missing_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("FINANCE_LLM_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="not set"):
        OpenAICompatibleClient(transport=lambda *a: (200, b""))


# 3 timeout handled
def test_timeout(monkeypatch):
    def tr(*a): raise TimeoutError()
    with pytest.raises(ProviderError, match="timed out"):
        make_client(monkeypatch, tr).generate("x")


# 4 auth failure handled, key redacted
def test_auth_failure_redacts_key(monkeypatch):
    def tr(*a): return 401, b'{"error":"bad key sk-secret-key-123"}'
    with pytest.raises(ProviderError) as ei:
        make_client(monkeypatch, tr).generate("x")
    assert "sk-secret-key-123" not in str(ei.value)


# 5 empty content handled
def test_empty_content(monkeypatch):
    def tr(*a): return 200, ok_body("")
    with pytest.raises(ProviderError, match="empty"):
        make_client(monkeypatch, tr).generate("x")


# 5b rate limit / connection / bad shape
def test_rate_limit_and_connection_and_shape(monkeypatch):
    for tr, match in [
        (lambda *a: (429, b""), "rate limit"),
        (lambda *a: (_ for _ in ()).throw(__import__("urllib").error.URLError("dns")), "connection"),
        (lambda *a: (200, b'{"nope":1}'), "shape"),
        (lambda *a: (500, b"boom"), "HTTP 500"),
    ]:
        with pytest.raises(ProviderError, match=match):
            make_client(monkeypatch, tr).generate("x")


# 6 implements LLMClient protocol
def test_implements_llm_client_protocol():
    from finance_controller.llm_investigator import LLMClient
    assert callable(getattr(OpenAICompatibleClient, "generate"))


# 7 configuration from env
def test_env_config(monkeypatch):
    monkeypatch.setenv("FINANCE_LLM_API_KEY", "k")
    monkeypatch.setenv("FINANCE_LLM_MODEL", "my-model")
    monkeypatch.setenv("FINANCE_LLM_BASE_URL", "https://example.com/v1/")
    c = OpenAICompatibleClient(transport=lambda *a: (200, ok_body(VALID_CONTENT)))
    seen = {}
    def tr(m, url, h, b, t):
        seen.update(json.loads(b)); seen_url = url
        return 200, ok_body(VALID_CONTENT)
    c2 = OpenAICompatibleClient(transport=tr)
    c2.generate("x")
    assert seen["model"] == "my-model"


# 8 key never appears in repr/logs/errors
def test_no_key_leakage(monkeypatch, capsys):
    c = make_client(monkeypatch, lambda *a: (200, ok_body(VALID_CONTENT)))
    r = repr(c)
    assert "sk-secret-key-123" not in r and "<redacted>" in r


# 9 end-to-end through llm_investigate_case validation still applies
def test_investigator_validates_provider_response(monkeypatch):
    bad = json.dumps({"finding": "", "explanation": "e",
                      "recommended_action": "a", "confidence": "high",
                      "evidence_used": [], "warnings": []})
    with pytest.raises(LLMInvestigatorError):
        c = make_client(monkeypatch, lambda *a: (200, ok_body(bad)))
        llm_investigate_case(golden_mismatch_case(), c)

    good = json.dumps({
        "finding": "Amounts differ.", "explanation": "OBSERVED EVIDENCE: ...",
        "recommended_action": "Verify ledgers.", "confidence": "moderate",
        "evidence_used": ["internal_amount"], "warnings": []})
    c = make_client(monkeypatch, lambda *a: (200, ok_body(good)))
    a = llm_investigate_case(golden_mismatch_case(), c)
    assert a.case_id == golden_mismatch_case().case_id
    assert a.risk_level.value == "medium"


# 10 no silent FakeLLMClient fallback
def test_no_fallback_to_fake(monkeypatch):
    monkeypatch.delenv("FINANCE_LLM_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        OpenAICompatibleClient()   # raises; nothing silently substituted
