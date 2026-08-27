from decimal import Decimal

from finance_controller.controller import (
    ControllerDecision, DecisionType)
from finance_controller.generator import (
    generate_dataset, generate_external_dataset)
from finance_controller.pipeline import run_pipeline
from finance_controller.treasury import TreasurySummary


def _inputs():
    ds = generate_dataset(seed=42)
    exts, _ = generate_external_dataset(ds, seed=99)
    return list(ds.transactions), exts


def _pos():
    from datetime import date
    from finance_controller.treasury import CashPosition
    return CashPosition(date(2025, 6, 30), Decimal("100000"),
                        Decimal("20000"), Decimal("5000"))


def _flows():   # identical to test_pipeline_treasury fixtures
    ...

def _pol():
    ...

def _kw(**extra):   # valid full treasury kwargs
    return {"cash_position": _pos(), "expected_flows": _flows(),
            "treasury_policy": _pol(), **extra}


# controller inactive without treasury inputs even if amount given
def test_no_treasury_no_controller():
    txns, exts = _inputs()
    r = run_pipeline(txns, exts, proposed_amount=Decimal(100))
    assert r.treasury_summary is None
    assert r.controller_decision is None


# controller inactive with treasury but no proposed_amount
def test_treasury_without_amount_no_controller():
    txns, exts = _inputs()
    r = run_pipeline(txns, exts, **_kw())
    assert isinstance(r.treasury_summary, TreasurySummary)
    assert r.controller_decision is None


# active when both present -> ALLOW for small amount
def test_small_amount_allowed():
    txns, exts = _inputs()
    r = run_pipeline(txns, exts, **_kw(proposed_amount=Decimal(10000)))
    d = r.controller_decision
    assert isinstance(d, ControllerDecision)
    assert d.decision_type is DecisionType.ALLOW
    assert d.proposed_amount == Decimal(10000)


# DENY path through pipeline (above governance cap 34500, below 97000)
def test_over_cap_denied_through_pipeline():
    txns, exts = _inputs()
    r = run_pipeline(txns, exts, **_kw(proposed_amount=Decimal(40000)))
    assert r.controller_decision.decision_type is DecisionType.DENY


# result matches direct controller call exactly
def test_matches_direct_evaluate():
    from finance_controller.controller import evaluate_treasury_decision
    txns, exts = _inputs()
    r = run_pipeline(txns, exts, **_kw(proposed_amount=Decimal(10000)))
    direct = evaluate_treasury_decision(
        compute := None or r.treasury_summary,  # summary already computed by pipeline
        _pol(), Decimal(10000))
    assert r.controller_decision == direct


# reconciliation/investigation/assessment invariant to controller activity
def test_upstream_outputs_identical():
    txns, exts = _inputs()
    base = run_pipeline(txns, exts, **_kw())
    plus = run_pipeline(txns, exts, **_kw(proposed_amount=Decimal(10000)))
    assert base.reconciliation_report == plus.reconciliation_report
    assert [c.case_id for c in base.investigation_cases] == \
           [c.case_id for c in plus.investigation_cases]
    assert [a.finding for a in base.deterministic_assessments] == \
           [a.finding for a in plus.deterministic_assessments]
    assert base.treasury_summary == plus.treasury_summary


# negative amount surfaces the domain ValueError (no silent coercion)
def test_negative_amount_raises():
    import pytest
    txns, exts = _inputs()
    with pytest.raises(ValueError, match="negative"):
        run_pipeline(txns, exts, **_kw(proposed_amount=Decimal(-5)))
