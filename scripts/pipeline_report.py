from collections import Counter

from finance_controller.generator import generate_dataset, generate_external_dataset
from finance_controller.llm_client import FakeLLMClient
from finance_controller.pipeline import run_pipeline

ds = generate_dataset(seed=42)
exts, _ = generate_external_dataset(ds, seed=99)

r = run_pipeline(list(ds.transactions), exts, run_llm=True,
                 run_evaluation=True, llm_client=FakeLLMClient())

print("PIPELINE REPORT")
print("===============")
print(f"Transactions           : {len(ds.transactions)}")
print(f"External records       : {len(exts)}")
print(f"Reconciliation results : {len(r.reconciliation_results)}")
print(f"Investigation cases    : {r.case_count}")
print(f"Deterministic assess.  : {len(r.deterministic_assessments)}")
print(f"LLM assessments        : {len(r.llm_assessments)}")
print(f"Evaluations            : {len(r.evaluation_results)}")
s = r.evaluation_summary
print(f"Eval summary           : {s.passed_cases}/{s.total_cases} pass, "
      f"quality {s.average_explanation_quality}/5, "
      f"safety {s.average_safety_score}/5\n")

by_type = Counter(c.exception_type.value for c in r.investigation_cases)
for t, n in by_type.most_common():
    print(f"  {t:<18}{n}")

det = {a.case_id: a for a in r.deterministic_assessments}
llm = {a.case_id: a for a in r.llm_assessments}
print("\nREPRESENTATIVE CASES:")
for c in r.investigation_cases[:3]:
    print(f"\n{c.case_id} ({c.exception_type.value})")
    print(f"  Deterministic finding: {det[c.case_id].finding}")
    print(f"  LLM finding:          {llm[c.case_id].finding}")
    print(f"  Action:               {llm[c.case_id].recommended_action[:90]}...")
