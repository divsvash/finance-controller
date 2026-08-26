from collections import Counter
from finance_controller.exceptions import Priority, build_investigation_cases
from finance_controller.generator import generate_dataset, generate_external_dataset
from finance_controller.reconciliation import reconcile

ds = generate_dataset(seed=42)
exts, _ = generate_external_dataset(ds, seed=99)
rs, rep = reconcile(list(ds.transactions), exts)
tb = {t.id: t for t in ds.transactions}
cases = build_investigation_cases(rs, tb, exts)

by_type = Counter(c.exception_type.name for c in cases)
by_prio = Counter(c.priority.name for c in cases)
impact = sum((c.financial_impact for c in cases), __import__("decimal").Decimal(0))

print("EXCEPTION REPORT")
print("================")
print(f"Total cases:          {len(cases)}")
for k in ["AMOUNT_MISMATCH", "AMBIGUOUS_MATCH", "MISSING_EXTERNAL",
          "EXTRA_EXTERNAL", "UNRESOLVED_MATCH"]:
    print(f"{k + ':':<22}{by_type[k]}")
for p in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
    print(f"{p.title() + ' priority:':<22}{by_prio[p]}")
print(f"Total financial impact: ₹{impact:,.2f}")
print()
for c in cases[:5]:
    print(f"[{c.priority.name}] {c.case_id}  ({c.exception_type.value})")
    print(f"  internal : id={c.internal_transaction_id} "
          f"amount={c.internal_amount} ref={c.payment_ref}")
    print(f"  external : ids={c.external_transaction_ids} "
          f"amounts={c.external_amounts}")
    print(f"  diff     : {c.amount_difference}  reason={c.reconciliation_reason}")
    print(f"  evidence : {c.evidence}\n")
