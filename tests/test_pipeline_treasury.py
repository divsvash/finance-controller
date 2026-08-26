import json
from datetime import date
from decimal import Decimal

from finance_controller.generator import (
    generate_dataset, generate_external_dataset)
from finance_controller.pipeline import run_pipeline
from finance_controller.storage import save_pipeline_result, \
    load_pipeline_result
from finance_controller.treasury import (
    CashPosition, Certainty, ControllerPolicy, ExpectedFlow,
    FlowDirection, compute_treasury_summary)


def _inputs():
    ds = generate_dataset(seed=42)
    exts, _ = generate_external_dataset(ds, seed=99)
    return list(ds.transactions), exts


def _pos():
    return CashPosition(date(2025, 6, 30), Decimal("100000"),
                        Decimal("20000"), Decimal("5000"))


def _flows():
    return [
        ExpectedFlow("f1", FlowDirection.INFLOW, Decimal("15000"),
                     date(2025, 7, 10), __import__("finance_controller.treasury",
                     fromlist=["FlowCategory"]).FlowCategory.RECEIVABLE),
        ExpectedFlow("f2", FlowDirection.OUTFLOW, Decimal("8000"),
                     date(2025, 7, 15),
                     __import__("finance_controller.treasury",
                     fromlist=["FlowCategory"]).FlowCategory.PAYROLL,
                     Certainty.SCHEDULED),
    ]


def _pol():
    return ControllerPolicy(Decimal("30000"), Decimal("0.10"),
                            Decimal("0.30"), False)


# 1 existing behavior unchanged with no treasury inputs
def test_no_treasury_inputs_unchanged(tmp_path):
    txns, exts = _inputs()
    r = run_pipeline(txns, exts)
    assert r.treasury_summary is None
    assert r.case_count == 122


# 2 summary appears only when all three supplied
def test_all_inputs_produce_summary():
    txns, exts = _inputs()
    r = run_pipeline(txns, exts, cash_position=_pos(),
                     expected_flows=_flows(), treasury_policy=_pol())
    assert isinstance(r.treasury_summary.__class__.__name__, str)
    s = r.treasury_summary
    assert s.current_cash == Decimal("115000")
    assert s.safe_movable_capital == Decimal("82000")


# 3 summary EXACTLY matches direct call
def test_matches_direct_compute():
    txns, exts = _inputs()
    r = run_pipeline(txns, exts, cash_position=_pos(),
                     expected_flows=_flows(), treasury_policy=_pol())
    direct = compute_treasury_summary(_pos(), _flows(), _pol())
    assert r.treasury_summary == direct


# 4 partial inputs rejected
import pytest

@pytest.mark.parametrize("kwargs", [
    dict(cash_position=None, expected_flows=_flows(), treasury_policy=_pol()),
    dict(cash_position="pos", expected_flows=None, treasury_policy=_pol()),
    dict(cash_position="pos", expected_flows=_flows(), treasury_policy=None),
])
def test_partial_treasury_rejected(kwargs):
    txns, exts = _inputs()
    kw = {k: (_pos() if v == "pos" else v) for k, v in kwargs.items()}
    with pytest.raises(ValueError, match="partial treasury"):
        run_pipeline(txns, exts, **kw)


# 5 reconciliation/investigation outputs identical with & without treasury
def test_reconciliation_untouched_by_treasury():
    txns, exts = _inputs()
    base = run_pipeline(txns, exts)
    plus = run_pipeline(txns, exts, cash_position=_pos(),
                        expected_flows=_flows(), treasury_policy=_pol())
    assert base.reconciliation_report == plus.reconciliation_report
    assert [c.case_id for c in base.investigation_cases] == \
           [c.case_id for c in plus.investigation_cases]
    assert [a.finding for a in base.deterministic_assessments] == \
           [a.finding for a in plus.deterministic_assessments]


# 6 persistence round-trip preserves Decimal/date/enum through the
#    generic storage encoder
def test_storage_round_trip_with_treasury(tmp_path):
    txns, exts = _inputs()
    r = run_pipeline(txns, exts, cash_position=_pos(),
                     expected_flows=_flows(), treasury_policy=_pol())
    p = tmp_path / "run.json"
    save_pipeline_result(r, p)
    loaded = load_pipeline_result(p)
    s = loaded.treasury_summary
    assert isinstance(s.current_cash, Decimal)
    assert s.safe_movable_capital == Decimal("82000")
