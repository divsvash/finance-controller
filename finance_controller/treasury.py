"""Deterministic Treasury domain layer.

PURE calculation module: no reconciliation, no LLM, no inference.
Every monetary fact must be explicitly DECLARED by the caller
(CashPosition, ExpectedFlow, ControllerPolicy). Nothing is derived
from Transaction/ExternalRecord data. The treasury layer never
auto-executes or mutates a movement.

All arithmetic uses Decimal. Money is NEVER converted to float.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Optional


class FlowDirection(enum.Enum):
    INFLOW = "INFLOW"
    OUTFLOW = "OUTFLOW"


class FlowCategory(enum.Enum):
    RECEIVABLE = "RECEIVABLE"
    PAYABLE = "PAYABLE"
    PAYROLL = "PAYROLL"
    TAX = "TAX"
    OTHER = "OTHER"


class Certainty(enum.Enum):
    CONFIRMED = "CONFIRMED"
    SCHEDULED = "SCHEDULED"
    FORECAST = "FORECAST"


def _require_non_negative(value: Decimal, name: str,
                          allow_zero: bool = True) -> None:
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must not be negative")


@dataclass(frozen=True)
class CashPosition:
    """Declared authoritative position. NOT derived from transactions."""
    as_of: date
    opening_balance: Decimal
    cleared_inflows: Decimal
    cleared_outflows: Decimal

    def __post_init__(self) -> None:
        _require_non_negative(self.opening_balance, "opening_balance")
        _require_non_negative(self.cleared_inflows, "cleared_inflows")
        _require_non_negative(self.cleared_outflows, "cleared_outflows")
        # DECISION (documented in tests): negative opening_balance is
        # rejected. A treasury position represents declared truth about
        # cash held; an overdraft is better modeled as a cleared outflow
        # against a smaller opening balance than as a negative balance,
        # because current_cash itself may legitimately go negative later
        # via flows and we must distinguish those two cases.


@dataclass(frozen=True)
class ExpectedFlow:
    flow_id: str
    direction: FlowDirection
    amount: Decimal
    expected_date: date
    category: FlowCategory
    certainty: Certainty
    # Traceability metadata ONLY. Never used in arithmetic; no lookup.
    linked_transaction_id: Optional[str] = None

    def __post_init__(self) -> None:
        _require_non_negative(self.amount, f"flow {self.flow_id} amount")


@dataclass(frozen=True)
class ControllerPolicy:
    minimum_cash_reserve: Decimal          # absolute floor >= 0
    reserve_buffer_pct: Decimal            # proportion of floor, [0, 1]
    max_single_movement_pct: Decimal       # proportion of cash, [0, 1]
    include_forecast_flows: bool

    def __post_init__(self) -> None:
        _require_non_negative(self.minimum_cash_reserve,
                              "minimum_cash_reserve")
        _require_non_negative(self.reserve_buffer_pct, "reserve_buffer_pct")
        _require_non_negative(self.max_single_movement_pct,
                              "max_single_movement_pct")
        if self.reserve_buffer_pct > 1:
            raise ValueError("reserve_buffer_pct is a proportion "
                             "and must be <= 1")
        if self.max_single_movement_pct > 1:
            raise ValueError("max_single_movement_pct is a proportion "
                             "and must be <= 1")


@dataclass(frozen=True)
class TreasurySummary:
    """Deterministic outputs of the treasury computation."""
    current_cash: Decimal
    expected_net: Decimal
    projected_cash: Decimal
    reserve_requirement: Decimal
    safe_movable_capital: Decimal     # always >= 0 (only clamped output)
    obligation_breaches_reserve: bool
    insufficiency: bool               # current_cash < 0 OR projected < 0
    included_flow_ids: tuple[str, ...] = field(default=())
    excluded_forecast_ids: tuple[str, ...] = field(default=())


def compute_treasury_summary(
    position: CashPosition,
    flows: list[ExpectedFlow],
    policy: ControllerPolicy,
) -> TreasurySummary:
    """Pure deterministic computation. See module rules 1-8."""
    current_cash = (position.opening_balance
                    + position.cleared_inflows
                    - position.cleared_outflows)

    included: list[ExpectedFlow] = []
    excluded_forecast: list[ExpectedFlow] = []
    for f in flows:
        if f.certainty is Certainty.FORECAST and \
                not policy.include_forecast_flows:
            excluded_forecast.append(f)
        else:
            included.append(f)

    expected_net = sum(
        ((f.amount if f.direction is FlowDirection.INFLOW else -f.amount)
         for f in included),
        start=Decimal("0"))

    projected_cash = current_cash + expected_net

    reserve_requirement = policy.minimum_cash_reserve * (
        Decimal("1") + policy.reserve_buffer_pct)

    safe_movable_capital = max(
        Decimal("0"),
        min(current_cash, projected_cash) - reserve_requirement)

    return TreasurySummary(
        current_cash=current_cash,
        expected_net=expected_net,
        projected_cash=projected_cash,
        reserve_requirement=reserve_requirement,
        safe_movable_capital=safe_movable_capital,
        obligation_breaches_reserve=(
            projected_cash < reserve_requirement),
        insufficiency=(current_cash < 0 or projected_cash < 0),
        included_flow_ids=tuple(f.flow_id for f in included),
        excluded_forecast_ids=tuple(f.flow_id for f in excluded_forecast))


@dataclass(frozen=True)
class GovernanceCheck:
    movement_amount: Decimal
    cap_amount: Decimal                 # current_cash * max pct
    allowed: bool
    warning: Optional[str]


def check_movement_governance(
    movement_amount: Decimal,
    current_cash: Decimal,
    policy: ControllerPolicy,
) -> GovernanceCheck:
    """Separate pure governance check (rule 9). Does NOT reduce
    safe_movable_capital; does NOT execute or mutate anything."""
    _require_non_negative(movement_amount, "movement_amount")
    cap = current_cash * policy.max_single_movement_pct
    allowed = movement_amount <= cap
    return GovernanceCheck(
        movement_amount=movement_amount,
        cap_amount=cap,
        allowed=allowed,
        warning=None if allowed else (
            f"movement {movement_amount} exceeds governance cap {cap}"))
