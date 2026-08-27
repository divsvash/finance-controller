import json
from copy import deepcopy
from decimal import Decimal

import pytest

from finance_controller.cli import main

VALID = {  # same shape as test_cli_treasury fixtures
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


def write(tmp_path, data):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_plain_run_unchanged(capsys):
    rc = main(["run"])
    out = capsys.readouterr().out
    assert rc == 0 and "case_count: 122" in out
    assert "== controller ==" not in out and "== treasury ==" not in out


def test_treasury_without_amount_no_controller(capsys, tmp_path):
    rc = main(["run", "--treasury-input", write(tmp_path, VALID)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "== treasury ==" in out and "== controller ==" not in out


def test_allow_small_amount(capsys, tmp_path):
    rc = main(["run", "--treasury-input", write(tmp_path, VALID),
               "--proposed-amount", "10000"])
    out = capsys.readouterr().out
    assert rc == 0 and "decision_type: ALLOW" in out
    assert "movable_capital_basis: 97000" in out


def test_deny_over_governance_cap(capsys, tmp_path):
    main(["run", "--treasury-input", write(tmp_path, VALID),
          "--proposed-amount", "40000"])
    out = capsys.readouterr().out
    assert "decision_type: DENY" in out
    assert any("governance cap" in ln for ln in out.splitlines())


def test_deny_above_movable_capital(capsys, tmp_path):
    main(["run", "--treasury-input", write(tmp_path, VALID),
          "--proposed-amount", "99999"])
    assert "decision_type: DENY" in capsys.readouterr().out


def test_boundary_exact_movable_capital_allows(capsys, tmp_path):
    main(["run", "--treasury-input", write(tmp_path, VALID),
          "--proposed-amount", "97000"])
    assert "decision_type: ALLOW" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["-5", "NaN", "Infinity", "-Infinity",
                                 "abc", "", "1e999999"])
def test_invalid_amounts_rejected(bad, capsys, tmp_path):
    with pytest.raises(SystemExit) as ei:
        main(["run", "--treasury-input", write(tmp_path, VALID),
              "--proposed-amount", bad])
    assert ei.value.code != 0
    assert capsys.readouterr().err.startswith("error:")


def test_decimal_precision_preserved(capsys, tmp_path):
    main(["run", "--treasury-input", write(tmp_path, VALID),
          "--proposed-amount", "10000.004"])
    out = capsys.readouterr().out
    assert "proposed_amount: 10000.004" in out   # exact str(Decimal)


def test_amount_without_treasury_inactive(capsys):
    rc = main(["run", "--proposed-amount", "100"])
    out = capsys.readouterr().out
    assert rc == 0 and "== controller ==" not in out


def test_output_matches_pipeline_directly(capsys, tmp_path):
    from datetime import date
    from finance_controller.generator import (
        generate_dataset, generate_external_dataset)
    from finance_controller.pipeline import run_pipeline
    from finance_controller.treasury import (
        CashPosition, Certainty, ControllerPolicy, ExpectedFlow,
        FlowCategory, FlowDirection)

    main(["run", "--treasury-input", write(tmp_path, VALID),
          "--proposed-amount", "10000"])
    out = capsys.readouterr().out
    ds = generate_dataset(seed=42)
    exts, _ = generate_external_dataset(ds, seed=99)
    direct = run_pipeline(
        list(ds.transactions), exts,
        cash_position=CashPosition(date(2025, 6, 30), Decimal(100000),
                                   Decimal(20000), Decimal(5000)),
        expected_flows=[ExpectedFlow("f1", FlowDirection.INFLOW,
                                     Decimal(15000), date(2025, 7, 10),
                                     FlowCategory.RECEIVABLE,
                                     Certainty.CONFIRMED),
                        ExpectedFlow("f2", FlowDirection.OUTFLOW,
                                     Decimal(8000), date(2025, 7, 15),
                                     FlowCategory.PAYROLL,
                                     Certainty.SCHEDULED)],
        treasury_policy=ControllerPolicy(Decimal(30000), Decimal("0.10"),
                                         Decimal("0.30"), False),
        proposed_amount=Decimal(10000))
    d = direct.controller_decision
    assert f"decision_type: {d.decision_type.value}" in out
    assert f"movable_capital_basis: {d.movable_capital_basis}" in out
    assert f"cap_amount: {d.cap_amount}" in out
