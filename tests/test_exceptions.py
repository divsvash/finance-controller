from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from finance_controller.exceptions import (
    ExceptionType, InvestigationCase, Priority, build_investigation_cases)
from finance_controller.generator import generate_dataset, generate_external_dataset
from finance_controller.models import Direction, ExternalRecord, Transaction, TxnStatus, money
from finance_controller.reconciliation import reconcile

T0 = datetime(2025, 3, 10, 14, 30)


def txn(tid="t1", amount="1000.00", direction=Direction.INFLOW,
        ts=T0, payment_ref="pay_ABC123", status=TxnStatus.COMPLETED):
    return Transaction(tid, ts, money(amount), direction, "sales",
                       status, "razorpay_payment", payment_ref)


def ext(eid="e1", amount="1000.00", direction=Direction.INFLOW,
        ref="pay_ABC123", ts=T0):
    return ExternalRecord(eid, ts, money(amount), direction,
                          TxnStatus.COMPLETED, ref, "ledger")


# matched creates no case
def test_matched_creates_no_case():
    rs, _ = reconcile([txn()], [ext()])
    cases = build_investigation_cases(rs, {"t1": txn()}, [ext()])
    assert cases == []


def test_amount_mismatch_case():
    t, e = txn(), ext(amount="9850.00")
    rs, _ = reconcile([t], [e])
    cases = build_investigation_cases(rs, {t.id: t}, [e])
    assert len(cases) == 1
    assert cases[0].exception_type == ExceptionType.AMOUNT_MISMATCH


def test_ambiguous_case():
    t = txn()
    rs, _ = reconcile([t], [ext("e1"), ext("e2")])
    cases = build_investigation_cases(rs, {t.id: t}, [ext("e1"), ext("e2")])
    assert cases[0].exception_type == ExceptionType.AMBIGUOUS_MATCH
    assert set(cases[0].candidate_external_ids) == {"e1", "e2"}


def test_missing_external_case():
    t = txn()
    rs, _ = reconcile([t], [])
    cases = build_investigation_cases(rs, {t.id: t}, [])
    assert cases[0].exception_type == ExceptionType.MISSING_EXTERNAL
    assert cases[0].priority >= Priority.HIGH   # missing money ranks high


def test_unresolved_case_created():
    t = txn(payment_ref="pay_XYZ")
    e = ext("e9", ref="pay_QQQ")     # wrong ref, wrong amount, far time
    rs, _ = reconcile([t], [e])
    cases = build_investigation_cases(rs, {t.id: t}, [e])
    assert cases[0].exception_type in (
        ExceptionType.UNRESOLVED_MATCH, ExceptionType.EXTRA_EXTERNAL)


def test_extra_external_case():
    t, e = txn(), ext(ref="totally_other")
    rs, _ = reconcile([t], [e])          # t unresolved, e extra? depends
    cases = build_investigation_cases(
        rs, {t.id: t}, [e, ext("eX", ref="ghost_only")])
    types = {c.exception_type for c in cases}
    assert ExceptionType.EXTRA_EXTERNAL in types


def test_large_discrepancy_higher_priority():
    small = build_investigation_cases(*mk(mismatch="150.00"))
    large = build_investigation_cases(*mk(mismatch="60000.00"))
    assert large[0].priority > small[0].priority


def mk(mismatch):
    t = txn(amount="10000.00")
    e = ext(amount=mismatch)
    rs, rep = reconcile([t], [e], enable_date_fallback=False)
    return rs, {t.id: t}, [e]


def test_deterministic_ordering():
    args = fixed_scenario()
    c1 = build_investigation_cases(*args)
    c2 = build_investigation_cases(*args)
    assert [c.case_id for c in c1] == [c.case_id for c in c2]


def test_decimal_preserved():
    rs, tb, ex = mk(mismatch="985.55")
    cases = build_investigation_cases(rs, tb, ex)
    assert all(isinstance(c.financial_impact, Decimal) for c in cases)
    assert isinstance(cases[0].external_amounts[0], Decimal)


def test_evidence_preserved():
    t = txn(ts=T0)
    e = ext(amount="985.00", ts=T0 + timedelta(seconds=134))
    rs, _ = reconcile([t], [e], enable_date_fallback=False)
    c = build_investigation_cases(rs, {t.id: t}, [e])[0]
    assert c.internal_timestamp == T0 and c.external_timestamps[0] == e.timestamp
    assert c.evidence["reference_match"] == "true"


def test_every_nonmatched_maps_to_one_case():
    ds = generate_dataset(seed=42)
    exts, gt = generate_external_dataset(ds, seed=99)
    rs, rep = reconcile(list(ds.transactions), exts)
    tb = {t.id: t for t in ds.transactions}
    cases = build_investigation_cases(rs, tb, exts)
    nonmatched = sum(r.status != "MATCHED" for r in rs)
    from_nonmatched = sum(1 for c in cases
                          if c.exception_type != ExceptionType.EXTRA_EXTERNAL)
    extras_expected = rep.extra_external_count
    assert from_nonmatched == nonmatched
    assert sum(1 for c in cases
               if c.exception_type == ExceptionType.EXTRA_EXTERNAL) == extras_expected


def test_no_mutation_of_results():
    ds = generate_dataset(seed=42)
    exts, _ = generate_external_dataset(ds, seed=99)
    rs, _ = reconcile(list(ds.transactions), exts)
    snapshot = [tuple(sorted(vars(r).items())) for r in rs]
    build_investigation_cases(rs, {t.id: t for t in ds.transactions}, exts)
    assert [tuple(sorted(vars(r).items())) for r in rs] == snapshot


def test_empty_input():
    assert build_investigation_cases([], {}, []) == []


def test_multiple_exceptions_deterministic_ids():
    args = fixed_scenario()
    cases = build_investigation_cases(*args)
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))
    prios = [c.priority for c in cases]
    assert prios == sorted(prios, reverse=True)


def fixed_scenario():
    txns = [txn("t1"), txn("t2", payment_ref="pay_DUP"),
            txn("t3", amount="9000.00", payment_ref="pay_MM")]
    exts = [ext("e1"), ext("e1b"), ext("e2"), ext("e2b"),
            ext("e3", amount="850.00"),
            ext("eGhost", ref="ghost")]
    rs, _ = reconcile(txns, exts, enable_date_fallback=False)
    return rs, {t.id: t for t in txns}, exts


# golden example — full inspection
def test_golden_example_full_inspection():
    t = txn("t_g1", amount="9850.00", payment_ref="pay_GoldenRef",
            ts=datetime(2025, 3, 10, 14, 30))
    e = ext("e_g1", amount="9700.00", ref="PAY-GOLDENREF",
            ts=datetime(2025, 3, 10, 14, 32, 14))
    rs, _ = reconcile([t], [e], enable_date_fallback=False)
    cases = build_investigation_cases(rs, {t.id: t}, [e])
    assert len(cases) == 1
    c = cases[0]
    assert c.internal_transaction_id == "t_g1"
    assert c.internal_amount == Decimal("9850.00")
    assert c.payment_ref == "pay_GoldenRef"
    assert c.normalized_reference == "PAYGOLDENREF"
    assert c.external_transaction_ids == ("e_g1",)
    assert c.external_amounts == (Decimal("9700.00"),)
    assert c.time_differences_seconds == (134,)
    assert c.evidence["reference_match"] == "true"
    assert c.evidence["direction_match"] == "true"
    assert c.exception_type == ExceptionType.AMOUNT_MISMATCH
