"""Manual OFFLINE smoke test of the REST layer. Not run under pytest.
Uses the framework test client; makes no network calls."""
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import finance_controller.api as api_mod
from finance_controller.storage import load_pipeline_result


class _Offline:
    def generate(self, prompt):
        import json
        return json.dumps({"finding": "f", "explanation":
                           "OBSERVED EVIDENCE: delta present.",
                           "recommended_action": "review",
                           "confidence": "high",
                           "evidence_used": ["internal_amount"],
                           "warnings": []})


print("API CHECK (offline)")
print("===================")

client = TestClient(api_mod.create_app())

r = client.get("/health")
assert r.status_code == 200 and r.json() == {"status": "ok"}
print("GET /health             : OK")

r = client.post("/pipeline/run", json={})
assert r.status_code == 200 and r.json()["case_count"] == 122
print("POST /pipeline/run (det): OK, 122 cases")

r = client.post("/pipeline/run",
                json={"run_llm": True, "run_evaluation": True})
# note: production path would need a key; injected below instead
assert r.status_code == 400   # missing key handled cleanly
print("missing-key handling    : OK (400)")

inj = TestClient(api_mod.create_app(llm_client=_Offline()))
r = inj.post("/pipeline/run", json={"run_llm": True, "run_evaluation": True})
assert r.status_code == 200
assert r.json()["evaluation"]["summary"]["passed_cases"] == 122
print("LLM+eval (injected)     : OK, 122/122 pass")

with tempfile.TemporaryDirectory() as td:
    c = TestClient(api_mod.create_app(llm_client=_Offline(),
                                      runs_dir=str(td)))
    r = c.post("/pipeline/save", json={"run_name": "demo"})
    assert r.status_code == 200
    loaded = load_pipeline_result(Path(td) / "demo.json")
    assert loaded.case_count == 122
    print("POST /pipeline/save     : OK, reload verified")

r = inj.post("/pipeline/save", json={"run_name": "../../evil"})
assert r.status_code == 422
print("path-traversal defense  : OK")

print("\nALL API CHECKS PASSED (no network used)")

# direct-input mode (offline)
ds_txns = generate_dataset(seed=3)
txns = ds_txns.transactions[:15]
exts, _ = generate_external_dataset(ds_txns, seed=4)
payload_tx = [{"transaction_id": t.transaction_id,
               "internal_amount": str(t.internal_amount),
               "date": t.date.isoformat()} for t in txns]
payload_ex = [{"external_id": e.external_id,
               "external_amount": str(e.external_amount),
               "date": e.date.isoformat(),
               "counterparty": getattr(e, "counterparty", "")}
              for e in exts][:15]
r = client.post("/pipeline/run", json={"transactions": payload_tx,
                                       "external_records": payload_ex})
assert r.status_code == 200
r2 = client.post("/pipeline/run", json={"transactions": payload_tx})
assert r2.status_code == 422
print("direct-input /pipeline/run : OK (+only-one rejected)")
