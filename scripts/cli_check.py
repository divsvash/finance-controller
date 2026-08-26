"""Manual OFFLINE smoke check of the CLI. Not run under pytest.
Requires no credentials; exercises deterministic mode, injected-client
LLM+eval mode, and persistence round trip."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from finance_controller.storage import load_pipeline_result

PY = sys.executable

print("CLI CHECK (offline)")
print("===================")

# 1 deterministic run
r = subprocess.run([PY, "-m", "finance_controller.cli", "run"],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr
assert "Investigation cases    : 122" in r.stdout
print("deterministic run      : OK")

# 2 persistence via --output
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "cli_out.json"
    r = subprocess.run(
        [PY, "-m", "finance_controller.cli", "run",
         "--output", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    loaded = load_pipeline_result(out)
    assert loaded.case_count == 122
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    print("persistence round trip : OK")

# 3 LLM+evaluation through the injection boundary (offline)
import finance_controller.cli as cli_mod

class _Offline:
    calls = 0
    def generate(self, prompt):
        _Offline.calls += 1
        return json.dumps({"finding": "f", "explanation":
                           "OBSERVED EVIDENCE: delta.",
                           "recommended_action": "review",
                           "confidence": "high",
                           "evidence_used": ["internal_amount"],
                           "warnings": []})

orig = cli_mod._make_llm_client
cli_mod._make_llm_client = lambda: _Offline()
try:
    r = subprocess.run(
        [PY, "-c",
         "import finance_controller.cli as c;"
         "c._make_llm_client=lambda: type('O',(),{'generate':lambda s,p:"
         "'{\"finding\":\"f\",\"explanation\":\"OBSERVED EVIDENCE: x.\","
         "\"recommended_action\":\"r\",\"confidence\":\"high\","
         "\"evidence_used\":[\"internal_amount\"],\"warnings\":[]}'})"
         "();"
         "raise SystemExit(c.main(['run','--llm','--evaluate']))"],
        capture_output=True, text=True)
finally:
    cli_mod._make_llm_client = orig
assert r.returncode == 0, r.stderr
assert "Evaluation             : 122/122 passed" in r.stdout
print("LLM+eval (injected)    : OK")
print("\nALL CLI CHECKS PASSED (no network used)")
