"""Deterministic synthetic merchant dataset generator (stdlib only).

Semantics:
    historical transactions  ->  as_of snapshot date  ->  future obligations

target_txn_count is EXACT, anomalies included.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal

from .models import Direction, Obligation, ObligationStatus, Transaction, TxnStatus, money

CATEGORIES_IN = ["sales", "subscriptions", "refunds_received"]
CATEGORIES_OUT = [
    "vendor_payment", "payroll", "rent", "utilities",
    "marketing", "software_subscriptions", "taxes",
]


@dataclass(frozen=True)
class SyntheticMerchantDataset:
    """Snapshot container: history up to as_of, obligations due after as_of."""
    transactions: tuple[Transaction, ...]
    obligations: tuple[Obligation, ...]
    as_of: date
    period_start: date
    seed: int

    @property
    def period_days(self) -> int:
        return (self.as_of - self.period_start).days + 1


def generate_dataset(
    seed: int,
    start_date: date = date(2025, 1, 1),
    months: int = 4,
    target_txn_count: int = 1000,
) -> SyntheticMerchantDataset:
    """Generate a deterministic dataset with exactly `target_txn_count`
    transactions (anomalies included within that count).

    Patterns: weekday/weekend revenue differences, recurring monthly
    expenses (rent/payroll/utilities), recurring vendor payments, variable
    revenue, occasional large payments, unexpected expenses, anomalies.
    """
    rng = random.Random(seed)
    txns: list[Transaction] = []

    def add(day, amount, direction, category, source,
        status=TxnStatus.COMPLETED):
        ts = datetime(day.year, day.month, day.day,
                      rng.randint(8, 22), rng.randint(0, 59))
        txns.append(Transaction(
            id=f"txn_{seed}_{len(txns):06d}", timestamp=ts,
            amount=money(str(round(amount, 2))),
            direction=direction, category=category, status=status,
            source=source,
            payment_ref=f"pay_{seed}{len(txns):06d}",   # unique, deterministic
        ))


    end_date = start_date + timedelta(days=30 * months)
    day = start_date
    while day < end_date and len(txns) <= target_txn_count - 9:
        weekend = day.weekday() >= 5
        n_sales = rng.randint(1, 4) + (0 if weekend else rng.randint(1, 2))
        for _ in range(n_sales):
            base = rng.uniform(800, 6000)
            if weekend and rng.random() < 0.3:
                base *= rng.uniform(1.5, 3.0)
            if rng.random() < 0.01:
                base *= rng.uniform(10, 25)
            add(day, base, Direction.INFLOW, "sales", "razorpay_payment")
        if rng.random() < 0.15:
            add(day, rng.uniform(500, 3000), Direction.INFLOW,
                "subscriptions", "razorpay_subscription")
        if day.day == 1:
            add(day, Decimal("85000"), Direction.OUTFLOW, "rent", "bank_debit")
        if day.day == 15:
            add(day, rng.uniform(6000, 11000), Direction.OUTFLOW,
                "utilities", "bank_debit")
        if (day + timedelta(days=1)).day == 1:
            add(day, Decimal("240000"), Direction.OUTFLOW,
                "payroll", "payroll_system")
        if day.weekday() in (1, 4) and rng.random() < 0.7:
            add(day, rng.uniform(4000, 18000), Direction.OUTFLOW,
                "vendor_payment", "razorpay_payout")
        if rng.random() < 0.2:
            add(day, rng.uniform(1000, 9000), Direction.OUTFLOW,
                rng.choice(["marketing", "software_subscriptions"]),
                "razorpay_payout")
        if rng.random() < 0.02:
            add(day, rng.uniform(20000, 60000), Direction.OUTFLOW,
                "unexpected_expense", "bank_debit")
        day += timedelta(days=1)

    # Reconcile to exactly target_txn_count - 8 before the 8 fixed
    # anomalies below, so the final count is EXACT (per docstring).
    # The day-loop's own exit check only fires once per day but can add
    # a variable-sized batch within that day, so it can overshoot OR
    # (e.g. when the fixed day-window empties first) undershoot this
    # boundary. Uses the same "recycle an existing day" approach as the
    # anomaly block immediately below, rather than extending the day
    # window (which finance_controller.generator.generate_external_dataset
    # relies on being exactly 120 days) or inventing new generation rates.
    target_before_anomalies = target_txn_count - 8
    if len(txns) > target_before_anomalies:
        del txns[target_before_anomalies:]
    else:
        while len(txns) < target_before_anomalies:
            t = rng.choice(txns)
            add(t.timestamp.date(), rng.uniform(800, 6000),
                Direction.INFLOW, "sales", "razorpay_payment")

    # Exactly 8 anomalous records, counted inside target_txn_count.
    for _ in range(5):
        t = rng.choice(txns)
        add(t.timestamp.date(), float(t.amount) * rng.uniform(0.8, 1.2),
            Direction.INFLOW, "sales", "razorpay_payment",
            status=rng.choice([TxnStatus.FAILED, TxnStatus.PENDING]))
    for _ in range(3):
        d = start_date + timedelta(days=rng.randint(5, 28 * months - 5))
        add(d, rng.uniform(90_000, 150_000), Direction.OUTFLOW,
            "anomalous_charge", "unknown_source")

    assert len(txns) == target_txn_count, (
        f"Generator produced {len(txns)}, expected exactly {target_txn_count}")

    as_of = end_date - timedelta(days=1)
    return SyntheticMerchantDataset(
        transactions=tuple(txns),
        obligations=_generate_obligations(rng, as_of),
        as_of=as_of,
        period_start=start_date,
        seed=seed,
    )


def _generate_obligations(rng: random.Random, after: date) -> list[Obligation]:
    obs = []
    specs = [
        ("rent_next", "85000.00", 3), ("payroll_next", "240000.00", 12),
        ("vendor_acme", None, 5), ("gst_tax", None, 18),
        ("cloud_invoice", None, 9), ("insurance_premium", "14500.00", 25),
    ]
    for i, (name, fixed, offset_days) in enumerate(specs):
        amt = Decimal(fixed) if fixed else \
            money(str(round(rng.uniform(*_lo_hi(name)), 2)))
        obs.append(Obligation(
            id=f"obligation_{i:03d}",
            due_date=after + timedelta(days=offset_days),
            amount=amt, category=name,
            status=ObligationStatus.SCHEDULED,
            description=f"Recurring/scheduled: {name}",
        ))
    return obs


def _lo_hi(name: str) -> tuple[float, float]:
    return {"vendor_acme": (30_000, 50_000), "gst_tax": (40_000, 80_000),
            "cloud_invoice": (8_000, 15_000)}[name]

DEFAULT_CORRUPTION_RATES = {
    "exact": 0.70, "timestamp_drift": 0.10, "reference_format": 0.05,
    "duplicate": 0.05, "missing": 0.05, "amount_mismatch": 0.05,
    "extra": 0.02,   # injected external-only records (documented)
}


def generate_external_dataset(
    dataset: SyntheticMerchantDataset,
    seed: int,
    rates: dict[str, float] | None = None,
) -> tuple[list[ExternalRecord], dict[str, str | None]]:
    """Deterministically derive an external ledger from known internal
    transactions, injecting realistic reconciliation problems.

    Returns (external_records, ground_truth) where ground_truth maps
    external_id -> true internal id (None for injected extras).
    Ground truth is evaluation-only; do NOT pass it into the engine.
    Rates are fractions of total internal transactions; they sum to <= 1.
    """
    rng = random.Random(seed)
    rates = {**DEFAULT_CORRUPTION_RATES, **(rates or {})}
    n = len(dataset.transactions)
    buckets = {k: int(round(v * n)) for k, v in rates.items()}
    order = ["timestamp_drift", "reference_format", "duplicate",
             "missing", "amount_mismatch"]
    pool = list(dataset.transactions)
    rng.shuffle(pool)                      # deterministic shuffle via seeded rng

    assigned: dict[str, str] = {}          # txn.id -> bucket
    idx = 0
    for b in order:
        for _ in range(buckets[b]):
            assigned[pool[idx].id] = b
            idx += 1
    for t in pool[idx:]:
        assigned[t.id] = "exact"

    ext_records: list[ExternalRecord] = []
    truth: dict[str, str | None] = {}

    def emit(i: int, t: Transaction, *, amount=None, ts_delta_s=0,
             ref_override: str | None = None, source="processor_ledger") \
            -> ExternalRecord:
        ref = ref_override or f"pay_{t.id[-6:].upper()}"
        rec = ExternalRecord(
            id=f"ext_{seed}_{i:06d}",
            timestamp=t.timestamp + timedelta(seconds=ts_delta_s),
            amount=t.amount if amount is None else money(amount),
            direction=t.direction, status=TxnStatus.COMPLETED,
            external_reference=ref, source=source, payment_ref=t.id,
        )
        return rec

    i = 0
    for t in dataset.transactions:
        b = assigned[t.id]
        if b == "missing":
            continue
        elif b == "timestamp_drift":
            delta = rng.choice([60, 120, 180]) * rng.choice([1, -1])
            e = emit(i, t, ts_delta_s=delta); i += 1
        elif b == "reference_format":
            e = emit(i, t, ref_override=f"PAY-{t.id[-6:].upper()}_X"); i += 1
        elif b == "duplicate":
            e1 = emit(i, t); i += 1
            e2 = emit(i, t); i += 1
            ext_records += [e1, e2]; truth[e1.id] = t.id; truth[e2.id] = t.id
            continue
        elif b == "amount_mismatch":
            new_amt = money(str(round(float(t.amount) * 0.985, 2)))
            if new_amt == t.amount:
                new_amt = t.amount - Decimal("0.01")
            e = emit(i, t, amount=new_amt); i += 1
        else:                              # exact
            e = emit(i, t); i += 1
        ext_records.append(e); truth[e.id] = t.id

    for k in range(buckets["extra"]):       # injected external-only noise
        day = dataset.period_start + timedelta(days=rng.randint(0, 119))
        rec = ExternalRecord(
            id=f"ext_{seed}_extra{k:04d}",
            timestamp=datetime(day.year, day.month, day.day,
                               rng.randint(9, 21), rng.randint(0, 59)),
            amount=money(str(round(rng.uniform(500, 40_000), 2))),
            direction=rng.choice([Direction.INFLOW, Direction.OUTFLOW]),
            status=TxnStatus.COMPLETED,
            external_reference=f"ext_only_{k:03d}", source="unknown_source",
        )
        ext_records.append(rec); truth[rec.id] = None

    return ext_records, truth
