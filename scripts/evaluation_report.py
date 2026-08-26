from finance_controller.exceptions import build_investigation_cases
from finance_controller.evaluation import evaluate_batch
from finance_controller.generator import generate_dataset, generate_external_dataset
from finance_controller.investigator import investigate_cases
from finance_controller.llm_client import FakeLLMClient
from finance_controller.llm_investigator import llm_investigate_cases
from finance_controller.reconciliation import reconcile

# NOTE: evaluation layer only -- never part of the financial decision path.

ds = generate_dataset(seed=42)
exts, _ = generate_external_dataset(ds, seed=99)
rs, _ = reconcile(list(ds.transactions), exts)
tb = {t.id: t for t in ds.transactions}
cases = build_investigation_cases(rs, tb, exts)
det = investigate_cases(cases)
llm = llm_investigate_cases(cases, FakeLLMClient())
results, s = evaluate_batch(cases, det, llm)
n = s.total_cases

print("EVALUATION REPORT")
print("=================")
print(f"Cases evaluated: {n}")
print(f"Overall passed: {s.passed_cases} / {n}")
print(f"Overall failed: {s.failed_cases} / {n}\n")
print(f"Risk agreement:      {s.risk_agreements} / {n}")
print(f"Type agreement:      {s.type_agreements} / {n}")
print(f"Valid evidence:      {s.valid_evidence_count} / {n}")
print(f"Uncertainty coverage:{s.uncertainty_present_count} / {n}")
print(f"Prohibited claims:   {s.prohibited_claim_count} / {n}\n")
print(f"Average explanation quality: {s.average_explanation_quality} / 5")
print(f"Average safety score:        {s.average_safety_score} / 5\n")

fails = [r for r in results if not r.overall_pass]
if fails:
    print("FAILURES:")
    for r in fails[:5]:
        print(f"  {r.case_id} ({r.exception_type}): "
              f"{'; '.join(r.failures)}")
else:
    print("No failures.")

det_by = {a.case_id: a for a in det}
llm_by = {a.case_id: a for a in llm}
res_by = {r.case_id: r for r in results}
print("\nREPRESENTATIVE PASSES (LLM interpretation quality vs "
      "deterministic financial truth):")
for r in results[:3]:
    d, l = det_by[r.case_id], llm_by[r.case_id]
    print(f"\n{r.case_id} ({r.exception_type})  quality="
          f"{r.explanation_quality_score}/5 safety={r.safety_score}/5")
    print(f"Deterministic finding: {d.finding}")
    print(f"LLM finding:           {l.finding}")
