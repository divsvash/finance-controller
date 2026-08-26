"""Normalized financial domain models.

All monetary values are `decimal.Decimal`, constructed from strings or
ints -- never from binary floats.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum


class MoneyError(ValueError):
    """Raised when a monetary value violates domain invariants."""


def money(value: Decimal | int | str) -> Decimal:
    """Normalize a monetary input to a Decimal with 2dp precision.

    Accepts Decimal/int/str only. Floats are rejected to prevent
    binary-float contamination of financial arithmetic.
    """
    if isinstance(value, float):
        raise MoneyError(
            f"Monetary values must not be floats (got {value!r}); "
            "use Decimal, int, or string.")
    d = Decimal(value) if not isinstance(value, Decimal) else value
    return d.quantize(Decimal("0.01"))


class Direction(str, Enum):
    INFLOW = "inflow"
    OUTFLOW = "outflow"


class TxnStatus(str, Enum):
    COMPLETED = "completed"
    PENDING = "pending"
    FAILED = "failed"


class ObligationStatus(str, Enum):
    SCHEDULED = "scheduled"
    PAID = "paid"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Transaction:
    id: str
    timestamp: datetime          # domain-wide convention: instants are datetimes
    amount: Decimal              # non-negative; direction gives sign
    direction: Direction
    category: str
    status: TxnStatus
    source: str

    def __post_init__(self) -> None:
        amt = money(self.amount)
        object.__setattr__(self, "amount", amt)
        if amt < 0:
            raise MoneyError(f"Transaction amount must be non-negative: {amt}")

    @property
    def signed_amount(self) -> Decimal:
        """Signed effect on cash for completed transactions."""
        if self.status != TxnStatus.COMPLETED:
            return Decimal("0.00")
        return self.amount if self.direction == Direction.INFLOW else -self.amount


@dataclass(frozen=True)
class Obligation:
    id: str
    due_date: date
    amount: Decimal              # non-negative expected outflow
    category: str
    status: ObligationStatus = ObligationStatus.SCHEDULED
    description: str = ""

    def __post_init__(self) -> None:
        amt = money(self.amount)
        object.__setattr__(self, "amount", amt)
        if amt < 0:
            raise MoneyError(f"Obligation amount must be non-negative: {amt}")

    @property
    def is_upcoming(self) -> bool:
        return self.status == ObligationStatus.SCHEDULED


@dataclass(frozen=True)
class LiquidityPolicy:
    """Explicit policy inputs -- nothing hidden."""
    starting_balance: Decimal
    minimum_operating_balance: Decimal
    safety_buffer: Decimal

    def __post_init__(self) -> None:
        for name in ("starting_balance", "minimum_operating_balance",
                     "safety_buffer"):
            v = money(getattr(self, name))
            object.__setattr__(self, name, v)
            if v < 0:
                raise MoneyError(f"{name} must be non-negative: {v}")

@dataclass(frozen=True)
class Transaction:
    id: str
    timestamp: datetime
    amount: Decimal              # non-negative; direction gives sign
    direction: Direction
    category: str
    status: TxnStatus
    source: str                  # originating system/provider, e.g. "razorpay_payment"
    payment_ref: str = ""        # processor-side payment reference used for reconciliation

    def __post_init__(self) -> None:
        amt = money(self.amount)
        object.__setattr__(self, "amount", amt)
        if amt < 0:
            raise MoneyError(f"Transaction amount must be non-negative: {amt}")

    @property
    def signed_amount(self) -> Decimal:
        if self.status != TxnStatus.COMPLETED:
            return Decimal("0.00")
        return self.amount if self.direction == Direction.INFLOW else -self.amount
