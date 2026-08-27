import dataclasses
from datetime import date
from decimal import Decimal

import pytest

from finance_controller.controller import (
    ControllerDecision, DecisionType, evaluate_treasury_decision)
from finance_controller.treasury import (
    CashPosition, Certainty, ControllerPolicy, ExpectedFlow,
    FlowCategory, FlowDirection, check_movement_governance,
    compute_treasury_summary)


def _summary(opening="100000", ci="20000", co="5000",
             flows=None, forecast=False):
    pos = CashPosition(date(2025, 6, 30), Decimal(opening),
                       Decimal(ci), Decimal(co))
    pol = ControllerPolicy(Decimal("30000"), Decimal("0.10"),
                           Decimal("0.30"), forecast)
    fs = flows or [ExpectedFlow("f1", FlowDirection.INFLOW, Decimal(15000),
                                date(2025, 7, 10), FlowCategory.RECEIVABLE,
                                Certainty.CONFIRMED)]
    return compute_treasury_summary(pos, fs, pol), pol


# 1 healthy + small -> ALLOW
def test_allow_small_movement():
    s, p = _summary()
    d = evaluate_treasury_decision(s, p, Decimal("10000"))
    assert d.decision_type is DecisionType.ALLOW
    assert d.movable_capital_basis == Decimal("97000")


# 2 exact equality with safe_movable_capital -> ALLOW (boundary)
def test_boundary_equality_allowed():
    s, p = _summary()
    smc = s.safe_movable_capital          # 97000
    d = evaluate_treasury_decision(s, p, smc)
    assert d.decision_type is DecisionType.ALLOW


# 3 above ceiling -> DENY
def test_above_ceiling_denied():
    s, p = _summary()
    d = evaluate_treasury_decision(s, p, s.safe_movable_capital + Decimal(1))
    assert d.decision_type is DecisionType.DENY


# 4 within capital but over governance cap (30000*0.30=34500... wait:
# current_cash=115000 -> cap=34500; use 40000 which is < 97000 but > 34500)
def test_over_governance_cap_denied():
    s, p = _summary()
    assert Decimal("40000") <= s.safe_movable_capital
    d = evaluate_treasury_decision(s, p, Decimal("40000"))
    assert d.decision_type is DecisionType.DENY
    assert "governance cap" in d.reasons[0]


# 5/6 forced denies
def test_insufficiency_denies_regardless():
    pos = CashPosition(date(2025, 6, 30), Decimal("25000"), Decimal(0),
                       Decimal("30000"))
    pol = ControllerPolicy(Decimal(0), Decimal(0), Decimal("0.30"), False)
    s = compute_treasury_summary(pos, [], pol)   # current=-5000
    for amt in (Decimal(1), Decimal("999999")):
        assert evaluate_treasury_decision(s, pol, amt).decision_type \
            is DecisionType.DENY


def test_reserve_breach_denies():
    s, p = _summary()
    breach = dataclasses.replace(s,
        obligation_breaches_reserve=True, projected_cash=Decimal("-100"))
    d = evaluate_treasury_decision(breach, p, Decimal(1))
    assert d.decision_type is DecisionType.DENY
    assert "breach" in " ".join(d.reasons).lower()


# 7 negative rejected
def test_negative_rejected():
    s, p = _summary()
    with pytest.raises(ValueError):
        evaluate_treasury_decision(s, p, Decimal(-5))


def test_non_decimal_and_bool_rejected():
    s, p = _summary()
    with pytest.raises(ValueError):
        evaluate_treasury_decision(s, p, 100)          # int
    with pytest.raises(ValueError):
        evaluate_treasury_decision(s, p, True)         # bool


# 8 zero valid on healthy state
def test_zero_allows_on_healthy():
    s, p = _summary()
    assert evaluate_treasury_decision(s, p, Decimal(0)).decision_type \
        is DecisionType.ALLOW


# 9 cross-consistency with check_movement_governance
def test_cap_matches_governance_function():
    s, p = _summary()
    g = check_movement_governance(Decimal("10000"),
                                  s.current_cash, p)
    d = evaluate_treasury_decision(s, p, Decimal("10000"))
    assert d.cap_amount == g.cap_amount == Decimal("34500.0")


# 10 determinism
def test_deterministic_repeat():
    s, p = _summary()
    a = evaluate_treasury_decision(s, p, Decimal("10000"))
    b = evaluate_treasury_decision(s, p, Decimal("10000"))
    assert a == b and a.reasons == b.reasons


# 11 immutability
def test_frozen():
    s, p = _summary()
    d = evaluate_treasury_decision(s, p, Decimal(0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.proposed_amount = Decimal(1)


# 12 types preserved
def test_decimals_stay_decimal():
    s, p = _summary()
    d = evaluate_treasury_decision(s, p, Decimal("10000"))
    assert isinstance(d.movable_capital_basis, Decimal)
    assert isinstance(d.cap_amount, Decimal)


# 13 reasons are ordered tuples
def test_reasons_tuple():
    s, p = _summary()
    d = evaluate_treasury_decision(s, p, Decimal("10000"))
    assert isinstance(d.reasons, tuple)
    assert all(isinstance(r, str) for r in d.reasons)


# 14 linked id inert: summaries carry no link info at controller level;
# two structurally identical summaries give identical decisions regardless
# of what linked_transaction_id was on the source flows.
def test_linked_id_cannot_affect_decision():
    mk = lambda link: compute_treasury_summary(
        CashPosition(date(2025, 6, 30), Decimal(100000), Decimal(20000),
                     Decimal(5000)),
        [ExpectedFlow("f1", FlowDirection.INFLOW, Decimal(15000),
                      date(2025, 7, 10), FlowCategory.RECEIVABLE,
                      Certainty.CONFIRMED, link)],
        ControllerPolicy(Decimal("30000"), Decimal("0.10"),
                         Decimal("0.30"), False))
    a = mk(None); b = mk("TXN-77")
    pa = evaluate_treasury_decision(a, ..., Decimal("10000"))
    pb = evaluate_treasury_decision(b, ..., Decimal("10000"))
    assert pa == pb


# 15 round-trip via existing generic storage encoder (no prod changes)
def test_storage_round_trip(tmp_path):
    from finance_controller.storage import save_pipeline_result, \
        load_pipeline_result
    ...
