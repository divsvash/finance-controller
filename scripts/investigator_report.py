from collections import Counter
from decimal import Decimal

from finance_controller.exceptions import build_investigation_cases
from finance_controller.generator import generate_dataset, generate_external_dataset
from finance_controller.investigator import investigate_cases
from finance_controller.reconciliation import reconcile

ds = generate_dataset(seed=42)
exts, _ = generate_external_dataset(ds, seed=99)
rs, _ = reconcile(list(ds.transactions), exts)
tb = {t.id: t for t in ds.transactions}
cases = build_investigation_cases(rs, tb, exts)
assessments = investigate_cases(cases)

by_risk = Counter(a.risk_level.value for a in assessments)

print("INVESTIGATOR REPORT")
print("===================")
print(f"Cases investigated: {len(assessments)}\n")
for r in ("critical", "high", "medium", "low"):
    print(f"{r.upper():<10}{by_risk[r]}")
print()
for a in assessments[:4]:
    c = next(x for x in cases if x.case_id == a.case_id)
    print(f"[{a.risk_level.value.upper()}] {a.case_id}")
    print(f"  type       : {a.exception_type}")
    print(f"  impact     : ₹{c.financial_impact:,.2f}")
    print(f"  finding    : {a.finding}")
    print(f"  explanation: {a.explanation}")
    print(f"  action     : {a.recommended_action}")
    print(f"  confidence : {a.confidence.value}")
    print(f"  evidence   : {', '.join(a.evidence_used)}\n")
