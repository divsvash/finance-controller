import copy
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from finance_controller.exceptions import (
    ExceptionType, InvestigationCase, Priority, build_investigation_cases)
from finance_controller.generator import generate_dataset, generate_external_dataset
from finance_controller.investigator import (
    Confidence, RiskLevel, investigate_case, investigate_cases)
from finance_controller.models import Direction, ExternalRecord, Transaction, TxnStatus, money
from finance_controller.reconciliation import reconcile

T0 = datetime(2025, 3, 10, 14, 30)


def txn(tid="t1", amount="1000.00", ts=T0, pref="pay_ABC123"):
    return Transaction(tid, ts, money(amount), Direction.INFLOW, "sales",
                       TxnStatus.COMPLETED, "razorpay_payment", pref)


def ext(eid="e1", amount="1000.00", ref="pay_ABC123", ts=T0):
    return ExternalRecord(eid, ts, money(amount), Direction.INFLOW,
                          TxnStatus.COMPLETED, ref, "ledger")


def mkcases():
    txns = [txn("t1"), txn("t2", pref="pay_DUP"),
            txn("t3", amount="9000.00", pref="pay_MM")]
    exts = [ext("e1"), ext("e1b"), ext("e2"), ext("e2b"),
            ext("e3", amount="850.00"), ext("eGhost", ref="ghost")]
    rs, _ = reconcile(txns, exts, enable_date_fallback=False)
    cases = build_investigation_cases(rs, {t.id: t for t in txns}, exts)
    return cases


def first_of(etype):
    return next(c for c in mkcases() if c.exception_type == etype)


# 1 every type gets assessment
def test_every_type_assessed():
    seen = set()
    for c in mkcases():
        a = investigate_case(c)
        assert a.case_id == c.case_id
        seen.add(a.exception_type)
    assert seen == {t.value for t in ExceptionType}


# 2 amount mismatch explains discrepancy
def test_mismatch_explains_amount_discrepancy():
    a = investigate_case(first_of(ExceptionType.AMOUNT_MISMATCH))
    assert "differ" in a.finding or "disagree" in a.finding
    assert "₹8,150.00" in a.explanation   # 9000 - 850 exact decimal text


# 3 ambiguous mentions multiple candidates
def test_ambiguous_mentions_multiple():
    a = investigate_case(first_of(ExceptionType.AMBIGUOUS_MATCH))
    assert "candidates" in a.finding.lower()
    assert "cannot determine" in a.finding


# 4 MISSING_EXTERNAL avoids certainty claims
def test_missing_external_no_certainty():
    a = investigate_case(first_of(ExceptionType.MISSING_EXTERNAL))
    low = (a.explanation + a.finding).lower()
    assert "does not prove" in low
    assert "never occurred" not in low.replace(
        "it may be delayed, mis-referenced", "")


# 5 EXTRA_EXTERNAL not labelled fraudulent
def test_extra_not_fraudulent():
    a = investigate_case(first_of(ExceptionType.EXTRA_EXTERNAL))
    blob = (a.explanation + a.finding + a.recommended_action).lower()
    assert "fraud" not in blob


# 6 unresolved recommends manual investigation
def test_unresolved_manual():
    a = investigate_case(first_of(ExceptionType.UNRESOLVED_MATCH))
    assert "manual" in a.recommended_action.lower()


# 7 priority copied, not recalculated
@pytest.mark.parametrize("prio", list(Priority))
def test_priority_translated_only(prio):
    base = first_of(ExceptionType.UNRESOLVED_MATCH)
    c = InvestigationCase(**{**base.__dict__, "priority": prio})
    assert investigate_case(c).risk_level.name == prio.name


# 8 case_id preserved exactly
def test_case_id_preserved():
    for c in mkcases():
        assert investigate_case(c).case_id == c.case_id


# 9 evidence_used references real fields
def test_evidence_fields_exist_on_case():
    valid = set(InvestigationCase.__dataclass_fields__)
    for c in mkcases():
        a = investigate_case(c)
        for e in a.evidence_used:
            assert e in valid


# 10 determinism
def test_deterministic():
    cs = mkcases()
    assert investigate_cases(cs) == investigate_cases(cs)


# 11 batch preserves order
def test_batch_order():
    cs = mkcases()
    assert [a.case_id for a in investigate_cases(cs)] == [c.case_id for c in cs]


# 12 input not mutated
def test_input_not_mutated():
    cs = mkcases()
    snap = [(c.case_id, dict(c.evidence)) for c in cs]
    investigate_cases(cs)
    assert [(c.case_id, dict(c.evidence)) for c in cs] == snap


# 13 decimals safe, no float artifacts
def test_decimal_rendering_safe():
    c = InvestigationCase(**{
        **first_of(ExceptionType.AMOUNT_MISMATCH).__dict__,
        "internal_amount": Decimal("0.1")})
    a = investigate_case(c)
    assert "0.1000000000000000055511151231257827" not in a.explanation
    assert "₹" in a.explanation


# 14 empty input
def test_empty_input():
    assert investigate_cases([]) == []


# 15 golden example — exact expected assessment
def test_golden_example_exact():
    t = txn("t_g1", amount="9850.00", pref="pay_GoldenRef")
    e = ext("e_g1", amount="9700.00", ref="PAY-GOLDENREF",
            ts=T0 + timedelta(seconds=134))
    rs, _ = reconcile([t], [e], enable_date_fallback=False)
    case = build_investigation_cases(rs, {t.id: t}, [e])[0]
    a = investigate_case(case)
    assert a.case_id == case.case_id
    assert a.exception_type == "AMOUNT_MISMATCH"
    assert a.risk_level is RiskLevel.MEDIUM
    assert a.confidence is Confidence.MODERATE
    assert a.finding.startswith("Internal and external records share")
    assert "₹150.00" in a.explanation
    assert "Do not assume which side is correct" in a.recommended_action
    assert a.warnings == ()
    assert a.evidence_used[0] == "exception_type"


# integration test — full pipeline, one assessment per case
def test_full_pipeline_one_assessment_per_case():
    ds = generate_dataset(seed=42)
    exts, _ = generate_external_dataset(ds, seed=99)
    rs, rep = reconcile(list(ds.transactions), exts)
    cases = build_investigation_cases(rs, {t.id: t for t in ds.transactions},
                                      exts)
    assessments = investigate_cases(cases)
    assert len(assessments) == len(cases)
    assert len({a.case_id for a in assessments}) == len(cases)
    assert all(a.risk_level is not None for a in assessments)


# confidence rule spot-checks
def test_confidence_rules_documented_behavior():
    crit = first_of(ExceptionType.AMOUNT_MISMATCH)
    big = InvestigationCase(**{**crit.__dict__,
                               "financial_impact": Decimal("60000"),
                               "priority": Priority.CRITICAL})
    assert investigate_case(big).confidence is Confidence.HIGH
