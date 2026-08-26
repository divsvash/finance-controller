from collections import Counter

from finance_controller.exceptions import build_investigation_cases
from finance_controller.generator import generate_dataset, generate_external_dataset
from finance_controller.investigator import investigate_cases
from finance_controller.llm_client import FakeLLMClient
from finance_controller.llm_investigator import llm_investigate_cases
from finance_controller.reconciliation import reconcile

ds = generate_dataset(seed=42)
exts, _ = generate_external_dataset(ds, seed=99)
rs, _ = reconcile(list(ds.transactions), exts)
tb = {t.id: t for t in ds.transactions}
cases = build_investigation_cases(rs, tb, exts)
det = {a.case_id: a for a in investigate_cases(cases)}
llm = {a.case_id: a for a in llm_investigate_cases(cases, FakeLLMClient())}

print("INVESTIGATOR REPORT (deterministic vs LLM-fake)")
print("================================================")
print(f"Cases investigated: {len(llm)}\n")
print("Risk agreement:",
      sum(det[c].risk_level == llm[c].risk_level for c in det), "/", len(det))
print("Type agreement:  ",
      sum(det[c].exception_type == llm[c].exception_type for c in det),
      "/", len(det), "\n")
for cid in list(det)[:4]:
    d, l = det[cid], llm[cid]
    print(f"CASE {cid} ({d.exception_type})")
    print(f"Deterministic finding:\n  {d.finding}")
    print(f"LLM finding:\n  {l.finding}")
    print(f"Deterministic action:\n  {d.recommended_action[:100]}...")
    print(f"LLM action:\n  {l.recommended_action}")
    print(f"Risk: {d.risk_level.value} | Confidence det/llm: "
          f"{d.confidence.value}/{l.confidence.value}\n")
