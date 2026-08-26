import json

import pytest
from fastapi.testclient import TestClient

from finance_controller.api import RUNS_DIR, create_app
from finance_controller.storage import StorageError, load_pipeline_result


class OfflineLLM:
    calls = 0
    def generate(self, prompt):
        OfflineLLM.calls += 1
        return json.dumps({"finding": "f",
                           "explanation": "OBSERVED EVIDENCE: delta.",
                           "recommended_action": "review",
                           "confidence": "moderate",
                           "evidence_used": ["internal_amount"],
                           "warnings": []})


@pytest.fixture
def det_client():
    return TestClient(create_app())


@pytest.fixture
def llm_client_obj():
    OfflineLLM.calls = 0
    return OfflineLLM()


@pytest.fixture
def llm_api(llm_client_obj):
    return TestClient(create_app(llm_client=llm_client_obj))


# health
def test_health(det_client):
    r = det_client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


# deterministic run, 122 cases, assessments present, LLM fields null
def test_deterministic_run(det_client):
    r = det_client.post("/pipeline/run", json={})
    assert r.status_code == 200
    b = r.json()
    assert b["case_count"] == 122
    assert len(b["deterministic_assessments"]) == 122
    assert b["llm_assessments"] is None and b["evaluation"] is None
    first = b["investigation_cases"][0]
    assert first["exception_type"] in {
        "AMOUNT_MISMATCH", "EXTRA_EXTERNAL", "AMBIGUOUS_MATCH",
        "UNRESOLVED_MATCH", "MISSING_EXTERNAL"}
    assert isinstance(first["priority"], str)


# decimals serialized as strings, enums as strings
def test_decimal_and_enum_serialization(det_client):
    b = det_client.post("/pipeline/run",
                        json={"seed": 42}).json()
    case = next(c for c in b["investigation_cases"])
    ev = case["evidence"]
    for k, v in ev.items():
        assert not isinstance(v, float)   # money stays out of binary floats


# LLM mode with injected client
def test_llm_mode(llm_api, llm_client_obj):
    r = llm_api.post("/pipeline/run", json={"run_llm": True})
    assert r.status_code == 200
    b = r.json()
    assert b["llm_assessments"] is not None
    assert len(b["llm_assessments"]) == 122
    assert llm_client_obj.calls == 122


# LLM + evaluation
def test_llm_eval(llm_api):
    r = llm_api.post("/pipeline/run",
                     json={"run_llm": True, "run_evaluation": True})
    assert r.status_code == 200
    ev = r.json()["evaluation"]
    assert ev["summary"]["total_cases"] == 122
    assert ev["summary"]["passed_cases"] == 122


# evaluation without LLM rejected (stable error shape)
def test_eval_without_llm_rejected(det_client):
    r = det_client.post("/pipeline/run", json={"run_evaluation": True})
    assert r.status_code == 422
    e = r.json()["detail"]["error"]
    assert e["type"] == "invalid_configuration"


# missing API key -> clear 400, no traceback/key leakage
def test_missing_key(det_client, monkeypatch):
    monkeypatch.delenv("FINANCE_LLM_API_KEY", raising=False)
    app = create_app()
    c = TestClient(app)
    r = c.post("/pipeline/run", json={"run_llm": True})
    assert r.status_code == 400
    err = json.dumps(r.json())
    assert "MissingAPIKeyError" in err
    assert "Traceback" not in err


# provider error -> 502, secret redacted
def test_provider_error_redacted(monkeypatch):
    monkeypatch.setenv("FINANCE_LLM_API_KEY", "sk-api-secret-42")
    class Boom:
        def generate(self, p): raise RuntimeError("boom sk-api-secret-42")
    c = TestClient(create_app(llm_client=Boom()))
    r = c.post("/pipeline/run", json={"run_llm": True})
    assert r.status_code == 502
    txt = json.dumps(r.json())
    assert "sk-api-secret-42" not in txt
    assert "<redacted>" in txt


# malformed body rejected by validation
def test_malformed_body(det_client):
    r = det_client.post("/pipeline/run", json={"seed": "not-an-int"})
    assert r.status_code == 422


def test_wrong_type_body(det_client):
    r = det_client.post("/pipeline/run", content=b"{oops",
                        headers={"Content-Type": "application/json"})
    assert r.status_code == 422


# /pipeline/save happy path + reload
def test_save_and_reload(tmp_path, monkeypatch, det_client):
    runs = tmp_path / "runs"
    app = create_app(runs_dir=str(runs))
    c = TestClient(app)
    r = c.post("/pipeline/save", json={"run_name": "demo"})
    assert r.status_code == 200
    b = r.json()
    assert b["run_name"] == "demo" and b["case_count"] == 122
    loaded = load_pipeline_result(runs / "demo.json")
    assert loaded.case_count == 122


# path traversal rejected
@pytest.mark.parametrize("name", ["../../secret", "../x", "a/b", ".hidden",
                                  "", "x" * 100])
def test_traversal_rejected(tmp_path, name):
    c = TestClient(create_app(runs_dir=str(tmp_path)))
    r = c.post("/pipeline/save", json={"run_name": name})
    assert r.status_code == 422
    assert "invalid_run_name" in json.dumps(r.json())


# repeated deterministic requests identical on relevant output
def test_repeat_identical(det_client):
    a = det_client.post("/pipeline/run", json={}).json()
    b = det_client.post("/pipeline/run", json={}).json()
    assert a["case_count"] == b["case_count"]
    assert [c["case_id"] for c in a["investigation_cases"]] == \
           [c["case_id"] for c in b["investigation_cases"]]
    assert [x["finding"] for x in a["deterministic_assessments"]] == \
           [x["finding"] for x in b["deterministic_assessments"]]
