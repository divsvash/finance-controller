from datetime import date
from decimal import Decimal

import pytest

from finance_controller.controller import ControllerDecision, DecisionType
from finance_controller.generator import (
    generate_dataset, generate_external_dataset)
from finance_controller.pipeline import run_pipeline
from finance_controller.storage import load_pipeline_result, \
    save_pipeline_result
from finance_controller.treasury import (
    CashPosition, TreasurySummary)


def _inputs():
    ds = generate_dataset(seed=42)
    exts, _ = generate_external_dataset(ds, seed=99)
    return list(ds.transactions), exts


def _kw(amount):
    return {
        "cash_position": CashPosition(date(2025, 6, 30), Decimal("100000"),
                                      Decimal("20000"), Decimal("5000")),
        "expected_flows": [
            ExpectedFlow("f1", FlowDirection.INFLOW, Decimal(15000),
                         date(2025, 7, 10), FlowCategory.RECEIVABLE,
                         Certainty.CONFIRMED),
            ExpectedFlow("f2", FlowDirection.OUTFLOW, Decimal(8000),
                         date(2025, 7, 15), FlowCategory.PAYROLL,
                         Certainty.SCHEDULED)],
        "treasury_policy": ControllerPolicy(Decimal(30000), Decimal("0.10"),
                                            Decimal("0.30"), False),
        "proposed_amount": amount}


# --- ALLOW path, precision amount ---
def test_allow_round_trip(tmp_path):
    txns, exts = _inputs()
    result = run_pipeline(txns, exts, **_kw(Decimal("10000.004")))
    assert isinstance(result.controller_decision, ControllerDecision)
    assert result.controller_decision.decision_type is DecisionType.ALLOW

    path = tmp_path / "res.json"
    save_pipeline_result(result, path)
    loaded = load_pipeline_result(path)

    assert loaded.controller_decision == result.controller_decision   # exact
    d = loaded.controller_decision
    assert type(d) is ControllerDecision
    assert type(d.decision_type) is DecisionType
    assert type(d.proposed_amount) is Decimal
    assert d.proposed_amount == Decimal("10000.004")     # precision survives
    assert type(d.movable_capital_basis) is Decimal
    assert type(d.cap_amount) is Decimal
    assert type(d.reasons) is tuple
    # treasury summary still round-trips
    assert isinstance(loaded.treasury_summary, TreasurySummary)
    assert loaded.treasury_summary == result.treasury_summary


# --- DENY path (over governance cap) ---
def test_deny_round_trip(tmp_path):
    txns, exts = _inputs()
    result = run_pipeline(txns, exts, **_kw(Decimal(40000)))
    assert result.controller_decision.decision_type is DecisionType.DENY
    path = tmp_path / "deny.json"
    save_pipeline_result(result, path)
    loaded = load_pipeline_result(path)
    assert loaded.controller_decision == result.controller_decision


# --- backward compatibility ---
def test_no_treasury_loads(tmp_path):
    txns, exts = _inputs()
    result = run_pipeline(txns, exts)
    assert result.controller_decision is None
    assert result.treasury_summary is None
    path = tmp_path / "plain.json"
    save_pipeline_result(result, path)
    loaded = load_pipeline_result(path)
    assert loaded.controller_decision is None
    assert loaded.treasury_summary is None
    assert loaded.case_count == result.case_count          # unrelated fields intact


def test_treasury_without_amount_loads(tmp_path):
    kw = _kw(None); kw.pop("proposed_amount")
    txns, exts = _inputs()
    result = run_pipeline(txns, exts, **kw)
    assert result.treasury_summary is not None
    assert result.controller_decision is None
    path = tmp_path / "sum.json"
    save_pipeline_result(result, path)
    loaded = load_pipeline_result(path)
    assert loaded.controller_decision is None
    assert loaded.treasury_summary == result.treasury_summary


# --- deterministic equality after round-trip ---
def test_repeat_round_trips_equal(tmp_path):
    txns, exts = _inputs()
    a = run_pipeline(txns, exts, **_kw(Decimal("10000.004")))
    b = run_pipeline(txns, exts, **_kw(Decimal("10000.004")))
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    save_pipeline_result(a, pa); save_pipeline_result(b, pb)
    assert load_pipeline_result(pa) == load_pipeline_result(pb)
