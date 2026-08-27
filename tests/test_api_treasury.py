from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from finance_controller.api import create_app
from finance_controller.treasury import compute_treasury_summary

POS = {"as_of": "2025-06-30", "opening_balance": "100000",
       "cleared_inflows": "20000", "cleared_outflows": "5000"}
FLOWS = [
    {"flow_id": "f1", "direction": "INFLOW", "amount": "15000",
     "expected_date": "2025-07-10", "category": "RECEIVABLE",
     "certainty": "CONFIRMED"},
    {"flow_id": "f2", "direction": "OUTFLOW", "amount": "8000",
     "expected_date": "2025-07-15", "category": "PAYROLL",
     "certainty": "SCHEDULED"},
]
POLICY = {"minimum_cash_reserve": "30000", "reserve_buffer_pct": "0.10",
          "max_single_movement_pct": "0.30",
          "include_forecast_flows": False}
TREASURY = {"cash_position": POS, "expected_flows": FLOWS,
            "treasury_policy": POLICY}


@pytest.fixture
def client():
    return TestClient(create_app())


def test_empty_request_unchanged(client):
    r = client.post("/pipeline/run", json={})
    assert r.status_code == 200 and r.json()["case_count"] == 122
    assert r.json()["treasury_summary"] is None


def test_valid_treasury_returns_summary(client):
    b = client.post("/pipeline/run", json=TREASURY).json()
    s = b["treasury_summary"]
    assert s["current_cash"] == "115000"
    assert s["safe_movable_capital"] == "82000"
    assert isinstance(s["safe_movable_capital"], str)   # Decimal->string


def test_serialization_shapes(client):
    s = client.post("/pipeline/run", json=TREASURY).json()["treasury_summary"]
    assert isinstance(s["current_cash"], str)           # money as string
    for fid in ("f1", "f2"):
        assert fid in s["included_flow_ids"]
    assert set(s.keys()) >= {"current_cash", "projected_cash",
                             "reserve_requirement", "safe_movable_capital"}


def test_matches_direct_compute(client):
    from datetime import date
    direct = compute_treasury_summary(
        __import__("finance_controller.treasury", fromlist=["x"])
        .CashPosition(date(2025, 6, 30), Decimal(100000), Decimal(20000),
                      Decimal(5000)),
        [__import__("finance_controller.treasury", fromlist=["x"])
         .ExpectedFlow("f1", __import__("finance_controller.treasury",
         fromlist=["x"]).FlowDirection.INFLOW, Decimal(15000),
         date(2025, 7, 10), __import__("finance_controller.treasury",
         fromlist=["x"]).FlowCategory.RECEIVABLE,
         __import__("finance_controller.treasury",
         fromlist=["x"]).Certainty.CONFIRMED),
         __import__("finance_controller.treasury", fromlist=["x"])
         .ExpectedFlow("f2", __import__("finance_controller.treasury",
         fromlist=["x"]).FlowDirection.OUTFLOW, Decimal(8000),
         date(2025, 7, 15), __import__("finance_controller.treasury",
         fromlist=["x"]).FlowCategory.PAYROLL,
         __import__("finance_controller.treasury", fromlist=["x"])
         .Certainty.SCHEDULED)],
        __import__("finance_controller.treasury", fromlist=["x"])
        .ControllerPolicy(Decimal(30000), Decimal("0.10"), Decimal("0.30"),
                          False))
    api = client.post("/pipeline/run", json=TREASURY).json()["treasury_summary"]
    assert api["safe_movable_capital"] == str(direct.safe_movable_capital)
    assert api["projected_cash"] == str(direct.projected_cash)


def test_partial_inputs_rejected(client):
    body = {"cash_position": POS, "expected_flows": FLOWS}
    r = client.post("/pipeline/run", json=body)
    assert r.status_code == 422
    e = r.json()["detail"]["error"]
    assert e["type"] == "incomplete_treasury_input"


@pytest.mark.parametrize("mutate,key", [
    (lambda d: d.update({"opening_balance": "12.5.6"}), "pos"),
    (lambda d: None, "bad_date_flow"),
    (lambda d: None, "bad_enum_flow"),
    (lambda d: None, "negative_flow"),
    (lambda d: d.update({"reserve_buffer_pct": "1.5"}), "pol"),
])
def test_invalid_inputs_return_422(mutate, key, client):
    import copy, json as _j
    t = copy.deepcopy(TREASURY)
    if key == "pos":
        mutate(t["cash_position"])
    elif key == "pol":
        mutate(t["treasury_policy"])
    elif key == "bad_date_flow":
        t["expected_flows"][0]["expected_date"] = "30-06-2025"
    elif key == "bad_enum_flow":
        t["expected_flows"][0]["direction"] = "SIDEWAYS"
    elif key == "negative_flow":
        t["expected_flows"][0]["amount"] = "-100"
    r = client.post("/pipeline/run", json=t)
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["error"]["type"] == "invalid_treasury_input"


def test_save_endpoint_with_treasury(client):
    r = client.post("/pipeline/save", json={**TREASURY, "run_name": "t1"})
    assert r.status_code == 200
    loaded = client.get("/pipeline/load/t1").json() \
        if hasattr(client, "_") else None  # endpoint existence per current API
    # save must accept treasury inputs without duplicate parsing errors:
    assert r.json().get("run_name") == "t1"
