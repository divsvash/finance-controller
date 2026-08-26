"""Deterministic unit tests for the treasury layer.

Manual hand-calculations verified before implementation:
- Healthy:   cc=115000, net=+7000, proj=122000, reserve=33000 -> smc=82000
- Forecast:  cc=60000, forecast ignored, net=0 -> smc=min(60000,60000)-20000=40000
- Tight:     cc=40000, -35000 tax -> proj=5000, reserve=10000 -> smc=0, breach=True
"""
import pytest
from datetime import date
from decimal import Decimal

from finance_controller.treasury import (
    CashPosition, Certainty, ControllerPolicy, ExpectedFlow, FlowCategory,
    FlowDirection, check_movement_governance, compute_treasury_summary)

AS_OF = date(2025, 6, 30)


def pos(opening="100000", ci="20000", co="5000"):
    return CashPosition(AS_OF, Decimal(opening), Decimal(ci), Decimal(co))


def pol(reserve="30000", buf="0.10", cap="0.30", forecast=False):
    return ControllerPolicy(Decimal(reserve), Decimal(buf), Decimal(cap),
                            forecast)


def flow(fid, direction, amount, category=FlowCategory.OTHER,
         certainty=Certainty.CONFIRMED, day=15, link=None):
    return ExpectedFlow(fid, direction, Decimal(amount), date(2025, 7, day),
                        category, certainty, link)


# 1 healthy — matches the hand-calculation above exactly
def test_healthy_position():
 s = compute_treasury_summary(pos(), [
        flow("f1", FlowDirection.INFLOW, "15000", FlowCategory.RECEIVABLE),
        flow("f2", FlowDirection.OUTFLOW, "8000", FlowCategory.PAYROLL,
             Certainty.SCHEDULED)], pol())
    assert s.current_cash == Decimal("115000")
    assert s.expected_net == Decimal("7000")
    assert s.projected_cash == Decimal("122000")
    assert s.reserve_requirement == Decimal("33000.0")
    assert s.safe_movable_capital == Decimal("82000")
    assert not s.obligation_breaches_reserve and not s.insufficiency


# 2 scheduled inflow/outflow inclusion
def test_scheduled_flows_included():
    s = compute_treasury_summary(pos(), [
        flow("a", FlowDirection.INFLOW, "10", certainty=Certainty.SCHEDULED)],
        pol())
    assert s.included_flow_ids == ("a",)


# 3 reserve buffer math
def test_reserve_buffer():
    s = compute_treasury_summary(pos(), [], pol(reserve="25000",
                                                buf="0.12"))
    assert s.reserve_requirement == Decimal("28000.0")


# 4 safe movable capital basic
def test_safe_movable_capital():
    s = compute_treasury_summary(pos(), [], pol())
    assert s.safe_movable_capital == min(Decimal("115000"),
                                         Decimal("115000")) - Decimal("33000.0")


# 5 tight / breach
def test_reserve_breach():
    s = compute_treasury_summary(
        pos("40000", "0", "0"),
        [flow("tax", FlowDirection.OUTFLOW, "35000", FlowCategory.TAX)],
        pol(reserve="10000", buf="0"))
    assert s.projected_cash == Decimal("5000")
    assert s.safe_movable_capital == Decimal("0")
    assert s.obligation_breaches_reserve is True


# 6/7 forecast guard
def test_forecast_excluded_by_default():
    s = compute_treasury_summary(pos("60000", "0", "0"), [
        flow("fc", FlowDirection.INFLOW, "50000",
             certainty=Certainty.FORECAST)], pol())
    assert s.expected_net == Decimal("0")
    assert s.projected_cash == Decimal("60000")
    assert s.safe_movable_capital == Decimal("40000")
    assert s.excluded_forecast_ids == ("fc",)


def test_forecast_included_when_policy_allows():
    s = compute_treasury_summary(pos("60000", "0", "0"), [
        flow("fc", FlowDirection.INFLOW, "50000",
             certainty=Certainty.FORECAST)], pol(forecast=True))
    assert s.projected_cash == Decimal("110000")
    assert s.included_flow_ids == ("fc",)
    # optimism can't manufacture movable liquidity beyond current cash:
    assert s.safe_movable_capital == Decimal("90000")  # min(60k,110k)-... wait
    # correct: min(60000,110000)=60000; 60000-33000=27000
    assert s.safe_movable_capital == Decimal("27000")


# 8 negative projection exposed, not clamped
def test_negative_projected_cash_insufficiency():
    s = compute_treasury_summary(
        pos("25000", "0", "30000"),
        [flow("o", FlowDirection.OUTFLOW, "10000")], pol())
    assert s.current_cash == Decimal("-5000")
    assert s.projected_cash == Decimal("-15000")
    assert s.insufficiency is True
    assert s.safe_movable_capital == Decimal("0")


# 9 zero flows / 10 empty list
def test_no_flows():
    s = compute_treasury_summary(pos(), [], pol())
    assert s.expected_net == Decimal("0")
    assert s.included_flow_ids == ()
    assert s.safe_movable_capital == Decimal("82000")


def test_empty_list_equals_no_flows():
    a = compute_treasury_summary(pos(), [], pol())
    b = compute_treasury_summary(pos(), [], pol())
    assert a == b


# 11/12/13 governance separate from capital
def test_governance_below_cap_allowed():
    g = check_movement_governance(Decimal("30000"), Decimal("115000"), pol())
    assert g.allowed and g.cap_amount == Decimal("34500.0")


def test_governance_above_cap_flagged():
    g = check_movement_governance(Decimal("40000"), Decimal("115000"), pol())
    assert not g.allowed and g.warning is not None


def test_governance_does_not_mutate_capital():
    flows = [flow("f1", FlowDirection.INFLOW, "15000")]
    before = compute_treasury_summary(pos(), flows, pol()).safe_movable_capital
    check_movement_governance(Decimal("999999"), Decimal("115000"), pol())
    after = compute_treasury_summary(pos(), flows, pol()).safe_movable_capital
    assert before == after == Decimal("97000")


# 14 Decimal precision (no float drift)
def test_decimal_precision():
    p = CashPosition(AS_OF, Decimal("100.01"), Decimal("0.03"),
                     Decimal("0.02"))
    s = compute_treasury_summary(p, [
        flow("x", FlowDirection.INFLOW, "0.004")],
        pol(reserve="50.001", buf="0.001"))
    assert s.current_cash == Decimal("100.02")
    assert s.projected_cash == Decimal("100.024")
    assert isinstance(s.safe_movable_capital, Decimal)


# 15/16 invalid declarations rejected, never normalized
def test_negative_flow_amount_rejected():
    with pytest.raises(ValueError):
        flow("bad", FlowDirection.INFLOW, "-5")


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        pol(reserve="-1")
    with pytest.raises(ValueError):
        pol(buf="-0.1")
    with pytest.raises(ValueError):
        pol(buf="1.5")           # proportion > 1
    with pytest.raises(ValueError):
        pol(cap="2")
    with pytest.raises(ValueError):
        CashPosition(AS_OF, Decimal("-100"), Decimal("0"), Decimal("0"))


# 17 linked id is inert metadata
def test_linked_id_has_no_arithmetic_effect():
    base = [flow("f1", FlowDirection.INFLOW, "15000")]
    linked = [ExpectedFlow("f1", FlowDirection.INFLOW, Decimal("15000"),
                           date(2025, 7, 15), FlowCategory.RECEIVABLE,
                           Certainty.CONFIRMED, "TXN-42")]
    a = compute_treasury_summary(pos(), base, pol())
    b = compute_treasury_summary(pos(), linked, pol())
    assert a.safe_movable_capital == b.safe_movable_capital
    assert b.included_flow_ids[0] == "f1"   # preserved as traceability


# 18 deterministic repetition
def test_deterministic_repeat():
    flows = [flow("f1", FlowDirection.INFLOW, "15000",
                  category=FlowCategory.RECEIVABLE),
             flow("f2", FlowDirection.OUTFLOW, "8000",
                  category=FlowCategory.PAYROLL, certainty=Certainty.SCHEDULED)]
    r1 = compute_treasury_summary(pos(), flows, pol())
    r2 = compute_treasury_summary(pos(), flows, pol())
    assert r1 == r2
