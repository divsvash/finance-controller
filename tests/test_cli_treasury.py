import json
import pytest

from finance_controller.cli import main   # existing entry point convention
from finance_controller.generator import generate_dataset, generate_external_dataset
from finance_controller.pipeline import run_pipeline
from finance_controller.treasury import compute_treasury_summary

VALID = {
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


def write(tmp_path, data, name="t.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data) if not isinstance(data, str) else data)
    return str(p)


def test_no_treasury_unchanged(capsys, tmp_path):
    rc = main(["run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "case_count: 122" in out
    assert "treasury" not in out.lower()


def test_valid_treasury_summary(capsys, tmp_path):
    rc = main(["run", "--treasury-input", write(tmp_path, VALID)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "current_cash: 115000" in out
    assert "safe_movable_capital: 82000" in out
    assert "obligation_breaches_reserve: False" in out


def test_matches_direct_compute(capsys, tmp_path):
    main(["run", "--treasury-input", write(tmp_path, VALID)])
    out = capsys.readouterr().out
    ds = generate_dataset(seed=42)
    exts, _ = generate_external_dataset(ds, seed=99)
    direct = compute_treasury_summary(*_build_domain(VALID))
    for field in ("current_cash", "expected_net", "projected_cash",
                  "reserve_requirement", "safe_movable_capital"):
        assert f"{field}: {getattr(direct, field)}" in out


def test_missing_file_rejected(capsys):
    with pytest.raises(SystemExit) as ei:
        main(["run", "--treasury-input", "/nonexistent/x.json"])
    assert ei.value.code != 0
    assert "not found" in capsys.readouterr().err


def test_malformed_json_rejected(capsys, tmp_path):
    with pytest.raises(SystemExit):
        main(["run", "--treasury-input",
              write(tmp_path, "{not valid json", name="bad.json")])
    assert "not valid JSON" in capsys.readouterr().err


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("treasury_policy"),                      # partial
    lambda d: d["expected_flows"][0].update(amount="-100"),  # negative
    lambda d: d["expected_flows"][0].update(direction="SIDEWAYS"),  # enum
    lambda d: d["cash_position"].update(as_of="30-06-2025"),  # date
    lambda d: d["cash_position"].update(opening_balance=0.1),  # float
    lambda d: d["treasury_policy"].update(reserve_buffer_pct="1.5"),  # pct>1
])
def test_invalid_inputs_exit_nonzero(mutate, capsys, tmp_path):
    import copy
    d = copy.deepcopy(VALID)
    mutate(d)
    with pytest.raises(SystemExit) as ei:
        main(["run", "--treasury-input", write(tmp_path, d, "inv.json")])
    assert ei.value.code != 0
    assert capsys.readouterr().err.strip()


def test_decimal_display_exact(capsys, tmp_path):
    d = copy.deepcopy(VALID)
    d["cash_position"]["opening_balance"] = "100000.004"
    main(["run", "--treasury-input", write(tmp_path, d, "prec.json")])
    out = capsys.readouterr().out
    assert "current_cash: 115000.004" in out      # exact Decimal text


def test_reconciliation_identical_with_treasury(capsys, tmp_path):
    base = run_pipeline(list(generate_dataset(seed=42).transactions),
                        generate_external_dataset(generate_dataset(seed=42),
                                                  seed=99)[0])
    main(["run", "--treasury-input", write(tmp_path, VALID)])
    out = capsys.readouterr().out
    # CLI's pipeline stages identical: case_count line unchanged
    assert f"case_count: {base.case_count}" in out
