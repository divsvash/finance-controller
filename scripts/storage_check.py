"""Manual persistence verification. Not executed under pytest.
Persists nothing sensitive; writes one temp file and removes it."""
import tempfile
from pathlib import Path

from finance_controller.generator import generate_dataset, generate_external_dataset
from finance_controller.pipeline import run_pipeline
from finance_controller.storage import load_pipeline_result, save_pipeline_result

ds = generate_dataset(seed=42)
exts, _ = generate_external_dataset(ds, seed=99)

result = run_pipeline(list(ds.transactions), exts)   # deterministic-only

with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "pipeline_output.json"
    save_pipeline_result(result, path)
    loaded = load_pipeline_result(path)

print("STORAGE CHECK")
print("=============")
print(f"Transactions             : {len(ds.transactions)}")
print(f"External records         : {len(exts)}")
print(f"Reconciliation results   : {len(loaded.reconciliation_results)}")
print(f"Investigation cases      : {loaded.case_count}")
print(f"Deterministic assessments: "
      f"{len(loaded.deterministic_assessments)}")
print(f"LLM assessments present  : "
      f"{loaded.llm_assessments is not None}")
print(f"Evaluation present       : "
      f"{loaded.evaluation_summary is not None}")
orig = {c.case_id for c in result.investigation_cases}
back = {c.case_id for c in loaded.investigation_cases}
print(f"Case IDs survived        : {len(orig & back)} / {len(orig)}")
assert orig == back
print("OK: round trip verified (temp file removed)")
