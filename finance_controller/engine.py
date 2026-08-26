"""Deterministic cash-position engine.

as_of is REQUIRED at the public boundary -- never defaults to today.
Expected inflows are an explicit input (temporary deterministic forecast
stub); the engine does not know how they were produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from .models import LiquidityPolicy, Obligation, Transaction, TxnStatus

ZERO = Decimal("0.00")


def current_cash(policy: LiquidityPolicy, txns: list[Transaction],
                 as_of: datetime | date) -> Decimal:
    """Starting balance + completed inflows - completed outflows,
    counting only transactions occurring on or before `as_of`.

    `as_of` may be a date or datetime; dates are treated as end-of-day
    (23:59:59.999999) so any transaction on that day counts.
    """
    cutoff = _to_datetime_cutoff(as_of)
    total = policy.starting_balance
    for t in txns:
        if t.timestamp <= cutoff:
            total += t.signed_amount
    return money(total)


def expected_net_cash_flow(obligations: list[Obligation],
                           horizon_days: int, as_of: date,
                           assumed_daily_inflow: Decimal = ZERO,
                           ) -> tuple[Decimal, Decimal, Decimal]:
    """Returns (expected_inflows, expected_outflows, net) over horizon.

    NOTE: assumed_daily_inflow is a TEMPORARY deterministic forecast stub.
    Future architecture: forecasting engine produces inflows; this engine
    consumes them as an opaque input.
    """
    horizon_end = as_of.fromordinal(as_of.toordinal() + horizon_days)
    inflows = money(assumed_daily_inflow * horizon_days)
    outflows = sum(
        (o.amount for o in obligations
         if o.is_upcoming and as_of <= o.due_date <= horizon_end),
        start=ZERO)
    return inflows, money(outflows), money(inflows - outflows)


@dataclass
class CashPositionBreakdown:
    current_cash: Decimal
    expected_inflows: Decimal
    expected_outflows: Decimal
    minimum_operating_balance: Decimal
    safety_buffer: Decimal
    safe_movable_capital: Decimal = field(init=False)

    def __post_init__(self) -> None:
        net = self.expected_inflows - self.expected_outflows
        required = self.minimum_operating_balance + self.safety_buffer
        self.safe_movable_capital = max(ZERO, self.current_cash + net - required)

    def explain(self) -> str:
        f = lambda v: f"₹{v:>12,.0f}"  # noqa: E731
        return "\n".join([
            f"Current cash:                 {f(self.current_cash)}",
            f"Expected inflows:             {f(self.expected_inflows)}",
            f"Expected obligations:        -{f(self.expected_outflows)}",
            f"Minimum operating balance:    {f(self.minimum_operating_balance)}",
            f"Safety buffer:                {f(self.safety_buffer)}",
            "-" * 48,
            f"Safe movable capital:         {f(self.safe_movable_capital)}",
        ])


def compute_cash_position(policy: LiquidityPolicy, txns: list[Transaction],
                          obligations: list[Obligation], as_of: date,
                          horizon_days: int = 30,
                          expected_inflows: Decimal | None = None,
                          assumed_daily_inflow: Decimal = ZERO,
                          ) -> CashPositionBreakdown:
    """Full deterministic calculation. `as_of` is mandatory."""
    cash = current_cash(policy, txns, as_of)
    inflows = (money(expected_inflows) if expected_inflows is not None
               else money(assumed_daily_inflow * horizon_days))
    _, outflows, _ = expected_net_cash_flow(
        obligations, horizon_days, as_of, ZERO)
    return CashPositionBreakdown(
        current_cash=cash, expected_inflows=inflows,
        expected_outflows=outflows,
        minimum_operating_balance=policy.minimum_operating_balance,
        safety_buffer=policy.safety_buffer)


def _to_datetime_cutoff(as_of: datetime | date) -> datetime:
    from datetime import datetime as dt, time
    if isinstance(as_of, dt):
        return as_of
    return dt.combine(as_of, time(23, 59, 59, 999999))
