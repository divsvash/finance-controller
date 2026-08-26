from datetime import date
from statistics import mean

from finance_controller.engine import LiquidityPolicy, compute_cash_position
from finance_controller.generator import generate_dataset
from finance_controller.models import Direction, TxnStatus

txns, obligations = generate_dataset(seed=42)
completed = [t for t in txns if t.status == TxnStatus.COMPLETED]
inflow = sum(t.amount for t in completed if t.direction == Direction.INFLOW)
outflow = sum(t.amount for t in completed if t.direction == Direction.OUTFLOW)

policy = LiquidityPolicy(750_000, 250_000, 70_000)
breakdown = compute_cash_position(policy, txns, obligations,
                                  horizon_days=30, assumed_daily_inflow=7_000,
                                  as_of=date(2025, 5, 1))

ds = generate_dataset(seed=42)
policy = LiquidityPolicy(Decimal("750000.00"), Decimal("250000.00"),
                         Decimal("70000.00"))
breakdown = compute_cash_position(
    policy, list(ds.transactions), list(ds.obligations), as_of=ds.as_of,
    assumed_daily_inflow=Decimal("7000.00"))
print(f"as_of: {ds.as_of} | transactions: {len(ds.transactions)} "
      f"| obligations: {len(ds.obligations)}")
print(breakdown.explain())

print(f"Transactions: {len(txns)} ({len(completed)} completed), "
      f"obligations: {len(obligations)}")
print(f"Total inflow ₹{inflow:,.0f} | outflow ₹{outflow:,.0f} "
      f"| avg txn ₹{mean(t.amount for t in completed):,.0f}")
print()
print(breakdown.explain())
