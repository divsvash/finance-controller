"""Deterministic treasury controller intelligence layer.

PURE decision function over ALREADY-COMPUTED facts (TreasurySummary).
No side effects, no mutation, no execution of movements, no inference.
Imports treasury.py and stdlib only.

Policy semantics:
- summary.safe_movable_capital is the movement ceiling (already encodes
  the reserve floor via min(current, projected) - reserve).
- policy.max_single_movement_pct applies ONLY through the existing
  check_movement_governance() from treasury.py -- never reimplemented.
- minimum_cash_reserve / reserve_buffer_pct / include_forecast_flows are
  NOT re-applied here; they are already baked into TreasurySummary.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from .treasury import (
    ControllerPolicy, TreasurySummary, check_movement_governance)


class DecisionType(enum.Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"


@dataclass(frozen=True)
class ControllerDecision:
    decision_type: DecisionType
    proposed_amount: Decimal
    movable_capital_basis: Decimal
    cap_amount: Decimal
    reasons: tuple[str, ...]


def _deny(
    reason: str,
    proposed_amount: Decimal,
    movable_capital_basis: Decimal,
    cap_amount: Decimal,
) -> ControllerDecision:
    """Build a DENY decision. Mirrors the field population of the ALLOW
    branch in evaluate_treasury_decision; carries no rule logic itself."""
    return ControllerDecision(
        decision_type=DecisionType.DENY,
        proposed_amount=proposed_amount,
        movable_capital_basis=movable_capital_basis,
        cap_amount=cap_amount,
        reasons=(reason,))


def evaluate_treasury_decision(
    summary: TreasurySummary,
    policy: ControllerPolicy,
    proposed_amount: Decimal,
) -> ControllerDecision:
    """Pure, deterministic. Rules evaluated in exact order 1..6."""
    # rule 1 -- type & sign validation at domain boundary
    if isinstance(proposed_amount, bool):
        raise ValueError("proposed_amount must be a Decimal")
    if not isinstance(proposed_amount, Decimal):
        raise ValueError("proposed_amount must be a Decimal")
    if proposed_amount < 0:
        raise ValueError("proposed_amount must not be negative")
    if not proposed_amount.is_finite():
        raise ValueError("proposed_amount must be finite")

    # governance cap via EXISTING treasury domain function (no reimpl.)
    gov = check_movement_governance(proposed_amount,
                                    summary.current_cash, policy)

    # rule 2 -- insufficiency
    if summary.insufficiency:
        return _deny(
            "projected cash is insufficient: current_cash="
            f"{summary.current_cash}, projected_cash={summary.projected_cash}",
            proposed_amount, summary.safe_movable_capital, gov.cap_amount)
    # rule 3 -- reserve breach
    if summary.obligation_breaches_reserve:
        return _deny(
            f"expected obligations breach required reserve "
            f"{summary.reserve_requirement}: projected_cash="
            f"{summary.projected_cash}",
            proposed_amount, summary.safe_movable_capital, gov.cap_amount)

    # rule 4 -- hard reserve-protection ceiling
    if proposed_amount > summary.safe_movable_capital:
        return _deny(f"proposed amount {proposed_amount} exceeds "
                     f"safe movable capital {summary.safe_movable_capital}",
                     proposed_amount, summary.safe_movable_capital,
                     gov.cap_amount)

    # rule 5 -- single-movement governance cap
    if not gov.allowed:
        return _deny(f"proposed amount {proposed_amount} exceeds "
                     f"single-movement governance cap {gov.cap_amount}",
                     proposed_amount, summary.safe_movable_capital,
                     gov.cap_amount)

    # rule 6 -- allow
    return ControllerDecision(
        decision_type=DecisionType.ALLOW,
        proposed_amount=proposed_amount,
        movable_capital_basis=summary.safe_movable_capital,
        cap_amount=gov.cap_amount,
        reasons=(
            f"proposed amount {proposed_amount} is within safe movable "
            f"capital {summary.safe_movable_capital} and within governance "
            f"cap {gov.cap_amount}",
        ))
