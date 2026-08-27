import copy
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from finance_controller.api import create_app
from finance_controller.controller import (
    ControllerDecision, DecisionType, evaluate_treasury_decision)

TREASURY = {  # identical shapes to tests/test_api_treasury.py fixtures
    "cash_position": {"as_of": "2025-06-30", "opening_balance": "100000",
                      "cleared_inflows": "20000", "cleared_outflows": "5000"},
    "expected_flows": [
        {"flow_id": "f1", "direction": "INFLOW", "amount": "15000",
         "expected_date": "2025-07-10", "category": "RECEIVABLE",
         "certainty": "CONFIRMED"},
        {"flow_id": "f2", "direction": "OUTFLOW", "amount": "8000",
         "expected_date": "2025-07-15", "category": "PAYROLL",
         "certainty": "SCHEDULED"}],
    "treasury_policy": {"minimum_cash_reserve": "30000",
                        "reserve_buffer_pct": "0.10",
                        "max_single_movement_pct": "0.30",
                        "include_forecast_flows": False},
}


@pytest.fixture
def client():
    return TestClient(create_app())


def _run(client, body):
    return client.post("/pipeline/run", json=body)


def test_empty_request_unchanged(client):
    b = _run(client, {}).json()
    assert b["case_count"] == 122
    assert b["treasury_summary"] is None
    assert b["controller_decision"] is None


def test_summary_without_amount(client):
    b = _run(client, TREASURY).json()
    assert isinstance(b["treasury_summary"], dict)
    assert b["controller_decision"] is None


def test_allow_via_string_amount(client):
    body = {**copy.deepcopy(TREASURY), "proposed_amount": "10000"}
    d = _run(client, body).json()["controller_decision"]
    assert d["decision_type"] == "ALLOW"                 # enum as string
    assert isinstance(d["proposed_amount"], str)         # Decimal as string
    assert Decimal(d["proposed_amount"]) == Decimal(10000)


def test_deny_over_governance_cap(client):
    body = {**copy.deepcopy(TREASURY), "proposed_amount": "40000"}
    d = _run(client, body).json()["controller_decision"]
    assert d["decision_type"] == "DENY"
    assert any("governance cap" in r for r in d["reasons"])


def test_deny_above_movable_capital(client):
    body = {**copy.deepcopy(TREASURY),
            "proposed_amount": "99999"}
    d = _run(client, body).json()["controller_decision"]
    assert d["decision_type"] == "DENY"


def test_matches_direct_evaluate(client):
    body = {**copy.deepcopy(TREASURY), "proposed_amount": "10000"}
    resp = _run(client, body).json()
    # reconstruct direct call from API's own summary output
    from datetime import date
    from finance_controller.treasury import (
        CashPosition, ControllerPolicy, TreasurySummary)
    s = TreasurySummary(**resp["treasury_summary"])   # strings -> Decimals? no:
    # committed version builds the domain objects from the literal fixtures
    # (same constructors as test_api_treasury.test_matches_direct_compute)
    direct = evaluate_treasury_decision(s, POLICY_DOMAIN, Decimal(10000))
    assert resp["controller_decision"]["decision_type"] == \
        direct.decision_type.value
    assert Decimal(resp["controller_decision"]["cap_amount"]) == \
        direct.cap_amount


@pytest.mark.parametrize("amount", ["-5", 0.1, True, "NaN", "Infinity"])
def test_invalid_amounts_422(amount, client):
    body = {**copy.deepcopy(TREASURY), "proposed_amount": amount}
    r = _run(client, body)
    assert r.status_code == 422, r.text
    e = r.json()["detail"]["error"]
    assert e["type"] == "invalid_treasury_input"


def test_amount_without_treasury_stays_null(client):
    b = _run(client, {"proposed_amount": "100"}).json()
    assert b["treasury_summary"] is None
    assert b["controller_decision"] is None


def test_partial_treasury_with_amount_still_incomplete_error(client):
    body = {**TREASURY, "proposed_amount": "10000"}
    del body["treasury_policy"]
    r = _run(client, body)
    assert r.status_code == 422
    assert r.json()["detail"]["error"]["type"] == "incomplete_treasury_input"


def test_save_accepts_proposed_amount(client):
    body = {**copy.deepcopy(TREASURY), "proposed_amount": "10000",
            "run_name": "ctrl1"}
    r = client.post("/pipeline/save", json=body)
    assert r.status_code == 200
