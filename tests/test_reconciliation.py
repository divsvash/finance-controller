from datetime import datetime, timedelta
from decimal import Decimal
import pytest

from finance_controller.generator import (
    DEFAULT_CORRUPTION_RATES, generate_dataset, generate_external_dataset)
from finance_controller.models import (
    Decimal, Direction, ExternalRecord, Transaction, TxnStatus, money)
from finance_controller.reconciliation import (
    AMBIGUOUS, MATCHED, UNRESOLVED, evaluate, normalize_reference,
    reconcile)

T0 = datetime(2025, 3, 10, 14, 30)


def txn(tid="t1", amount="1000.00", direction=Direction.INFLOW,
        source="razorpay_payment", ts=T0, status=TxnStatus.COMPLETED,
        payment_ref="pay_ABC123"):
    return Transaction(tid, ts, Decimal(amount), direction, "sales",
                       status, source, payment_ref)


def ext(eid="e1", amount="1000.00", direction=Direction.INFLOW,
        ref="pay_ABC123", ts=T0):
    return ExternalRecord(eid, ts, Decimal(amount), direction,
                          TxnStatus.COMPLETED, ref, "ledger")

# 1 — exact payment reference matches (Stage 1)
def test_exact_payment_reference_matches():
    rs, rep = reconcile([txn(payment_ref="pay_ABC123")],
                        [ext(ref="pay_ABC123")])
    assert rs[0].status == MATCHED
    assert rs[0].reason == "exact_reference"

# 2 drift within tolerance
def test_drift_within_tolerance():
    rs, _ = reconcile([txn()], [ext(ts=T0 + timedelta(minutes=3))])
    assert rs[0].status == MATCHED

# 3 — outside tolerance does NOT match even with identical amount/direction
def test_outside_tolerance_no_stage2_match():
    rs, _ = reconcile([txn()], [ext(ts=T0 + timedelta(hours=2))],
                      ts_tolerance=timedelta(minutes=5))
    assert rs[0].status == UNRESOLVED

# 4 amount mismatch not exact
def test_amount_mismatch_not_matched():
    rs, rep = reconcile([txn()], [ext(amount="985.00")])
    assert rs[0].status == UNRESOLVED and rep.amount_mismatch_count >= 1

# 5 direction mismatch
def test_direction_mismatch():
    rs, _ = reconcile([txn()], [ext(direction=Direction.OUTFLOW)])
    assert rs[0].status == UNRESOLVED

# 6 normalization
@pytest.mark.parametrize("raw,want", [
    ("pay_ABC123", "PAYABC123"), ("PAY-ABC123", "PAYABC123"),
    ("pay_abc123_x", "PAYABC123X"), ("  pay abc ", "PAYABC"),
])
def test_reference_normalization(raw, want):
    assert normalize_reference(raw) == want

# 7 missing external
def test_missing_external_unresolved():
    rs, _ = reconcile([txn()], [])
    assert rs[0].status == UNRESOLVED and rs[0].reason == "no_candidate"

# 8 extra external detected
def test_extra_external_detected():
    _, rep = reconcile([txn()], [ext("e1"), ext("e2", ref="other")])
    assert rep.extra_external_count == 1

# 9 duplicates become ambiguous
def test_duplicate_candidates_ambiguous():
    rs, rep = reconcile([txn()], [ext("e1"), ext("e2")])
    assert rs[0].status == AMBIGUOUS and len(rs[0].candidate_external_ids) == 2

# 10 external cannot match twice
def test_one_external_one_internal():
    t1, t2 = txn("t1"), txn("t2", source="pay_DEF456")
    rs, rep = reconcile([t1, t2], [ext("e1"), ext("e2", ref="pay_DEF456")])
    assert rep.matched_count == 2
    assert len({r.external_id for r in rs}) == 2

# 11 internal consumes at most one external
def test_internal_single_result():
    rs, rep = reconcile([txn("t1")], [ext("e1"), ext("e2")])
    assert len(rs) == 1 and rs[0].status == AMBIGUOUS

# 12/13 determinism
def test_corruption_deterministic():
    ds = generate_dataset(seed=7)
    a, ta = generate_external_dataset(ds, seed=99)
    b, tb = generate_external_dataset(ds, seed=99)
    assert a == b and ta == tb

def test_different_seeds_differ():
    ds = generate_dataset(seed=7)
    a, _ = generate_external_dataset(ds, seed=1)
    b, _ = generate_external_dataset(ds, seed=2)
    assert a != b

# 14–16 metric math
def test_precision_recall_f1():
    gt = {"e1": "t1", "e2": "t2", "e3": None}
    results = [
        __import__("finance_controller.reconciliation",
                   fromlist=["ReconciliationResult"]).ReconciliationResult(
            "t1", MATCHED, "r", "e1"),
        __import__("finance_controller.reconciliation",
                   fromlist=["ReconciliationResult"]).ReconciliationResult(
            "t2", MATCHED, "r", "e3"),   # FP: matched to an extra
        __import__("finance_controller.reconciliation",
                   fromlist=["ReconciliationResult"]).ReconciliationResult(
            "t3", UNRESOLVED, "no_candidate"),
    ]
    m = evaluate(results, gt)
    assert m["true_positive_count"] == 1
    assert m["false_positive_count"] == 1
    assert m["false_negative_count"] == 1   # t2's real partner unmatched
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["f1"] == 0.5

# 17 hand-built golden example
def test_hand_built_example_exact():
    txns = [txn("t1", source="pay_AAA111"),
            txn("t2", "500.00", Direction.OUTFLOW, "pay_BBB222",
                T0.replace(hour=16))]
    exts = [ext("e1", ref="PAY_AAA111"),
            ext("e2", "500.00", Direction.OUTFLOW, "pay_BBB222",
                T0.replace(hour=16, minute=4)),
            ext("e3", "999.00", Direction.INFLOW, "ghost_ref")]
    rs, rep = reconcile(txns, exts, ts_tolerance=timedelta(minutes=5))
    assert rep.matched_count == 2 and rep.unresolved_count == 1
    assert rep.extra_external_count == 1
    m = evaluate(rs, {"e1": "t1", "e2": "t2", "e3": None})
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0

# 18 empty batch
def test_empty_batch():
    rs, rep = reconcile([], [])
    assert rs == [] and rep.total_internal_records == 0
    assert rep.match_rate == 0.0 and rep.unresolved_rate == 0.0

# 19 100+ batch works
def test_large_batch():
    ds = generate_dataset(seed=42, target_txn_count=150)
    exts, gt = generate_external_dataset(ds, seed=1)
    assert len(ds.transactions) >= 100 and len(exts) > 0
    rs, rep = reconcile(list(ds.transactions), exts)
    assert len(rs) == len(ds.transactions)
    assert rep.match_rate > 0.5

# 20 tolerance configurable
def test_tolerance_configurable():
    drifted = ext(ts=T0 + timedelta(minutes=4))
    assert reconcile([txn()], [drifted],
                     ts_tolerance=timedelta(minutes=1))[0][0].status == UNRESOLVED
    assert reconcile([txn()], [drifted])[0][0].status == MATCHED

# 21 — date fallback recovers large drift when ref+amount+date agree
def test_date_fallback_large_drift():
    rs, _ = reconcile([txn()], [ext(ts=T0 + timedelta(hours=6))])
    assert rs[0].status == MATCHED
    assert rs[0].reason == "reference_amount_date"

# 22 pending/failed externals never matched
def test_non_completed_external_never_matches():
    bad = ExternalRecord("e9", T0, Decimal("1000.00"), Direction.INFLOW,
                         TxnStatus.PENDING, "pay_ABC123", "ledger")
    rs, _ = reconcile([txn()], [bad])
    assert rs[0].status == UNRESOLVED

# 23 — corrupted reference recovered purely by amount+direction+timestamp (Stage 2)
def test_corrupted_reference_recovered_by_adt():
    rs, _ = reconcile([txn(payment_ref="pay_ABC123")],
                      [ext(ref="PAY-ABC123_X", ts=T0 + timedelta(minutes=2))])
    assert rs[0].status == MATCHED
    assert rs[0].reason == "amount_direction_timestamp"
    assert rs[0].external_id == "e1"

# 24 — Stage 2 must NOT fire on wrong amount even inside tolerance
def test_wrong_amount_no_stage2():
    rs, _ = reconcile([txn()],
                      [ext(amount="985.00", ts=T0 + timedelta(minutes=1))])
    assert rs[0].status == UNRESOLVED

# 25 — Stage 2 must NOT fire on wrong direction
def test_wrong_direction_no_stage2():
    rs, _ = reconcile([txn()],
                      [ext(direction=Direction.OUTFLOW, ts=T0 + timedelta(minutes=1))])
    assert rs[0].status == UNRESOLVED

# 26 — provenance audit: source stays provider, payment_ref is the ref
def test_source_vs_payment_ref_distinction():
    t = txn(source="razorpay_payment", payment_ref="pay_ZZZ999")
    assert t.source == "razorpay_payment"
    ds = generate_dataset(seed=42, target_txn_count=50)
    assert all(t.payment_ref.startswith("pay_") for t in ds.transactions)
    assert len({t.payment_ref for t in ds.transactions}) == len(ds.transactions)  # unique

# 27 — synthetic corruption actually produces claimed categories
def test_synthetic_categories_present_and_meaningful():
    ds = generate_dataset(seed=42)
    exts, gt = generate_external_dataset(ds, seed=99)
    norm_int = {normalize_reference(t.payment_ref): t.id
                for t in ds.transactions}
    cats = {"exact": 0, "format": 0, "drift": 0, "dup": 0,
            "mismatch": 0, "extra": 0}
    seen_normrefs: dict[str, int] = {}
    for e in exts:
        n = normalize_reference(e.external_reference)
        if e.external_reference.endswith("_X"):
            cats["format"] += 1
            assert n != next(iter(k for k, v in norm_int.items()
                                  if v == e.payment_ref))  # truly corrupted
        elif gt[e.id] is None:
            cats["extra"] += 1
        else:
            t = next(x for x in ds.transactions if x.id == e.payment_ref)
            if abs(e.timestamp - t.timestamp).total_seconds() > 5:
                cats["drift"] += 1
            if e.amount != t.amount:
                cats["mismatch"] += 1
            seen_normrefs[n] = seen_normrefs.get(n, 0) + 1
    dup_ids = {e.id for e in exts if seen_normrefs.get(
        normalize_reference(e.external_reference), 0) > 1}
    cats["dup"] = len(dup_ids)
    assert cats["format"] > 20 and cats["drift"] > 50 and cats["dup"] > 40 \
        and cats["mismatch"] > 30 and cats["extra"] >= 15

# 28 — ground truth validity: bijection sanity on matched pairs
def test_ground_truth_validity():
    ds = generate_dataset(seed=42)
    exts, gt = generate_external_dataset(ds, seed=99)
    int_ids = {t.id for t in ds.transactions}
    for eid, iid in gt.items():
        assert any(e.id == eid for e in exts)
        assert iid is None or iid in int_ids
    rs, rep = reconcile(list(ds.transactions), exts)
    m = evaluate(rs, gt)
    assert m["precision"] <= 1.0 and m["recall"] <= 1.0
    assert rep.matched_count == m["true_positive_count"] + m["false_positive_count"]