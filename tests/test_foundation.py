from decimal import Decimal

import pytest

from finance_controller.models import MoneyError


# --- Decimal regression ---
def test_decimal_precision_preserved():
    t = mk_txn(Decimal("0.1"), Direction.INFLOW)  # would be 0.1000000000000000055 as float
    p = LiquidityPolicy(Decimal("500000.00"), Decimal("0"), Decimal("0"))
    cash = current_cash(p, [t], as_of=date(2025, 5, 1))
    assert cash == Decimal("500000.10")
    assert isinstance(cash, Decimal)


def test_float_money_is_rejected():
    with pytest.raises(MoneyError):
        mk_txn(100.50, Direction.INFLOW)
    with pytest.raises(MoneyError):
        LiquidityPolicy(100000.0, Decimal("0"), Decimal("0"))


# --- negative-value validation ---
@pytest.mark.parametrize("factory", [
    lambda: Transaction("t", datetime(2025, 1, 1), Decimal("-1"),
                        Direction.INFLOW, "x", TxnStatus.COMPLETED, "s"),
    lambda: Obligation("o", date(2025, 5, 1), Decimal("-5"), "rent"),
    lambda: LiquidityPolicy(Decimal("-1"), Decimal("0"), Decimal("0")),
    lambda: LiquidityPolicy(Decimal("0"), Decimal("-1"), Decimal("0")),
    lambda: LiquidityPolicy(Decimal("0"), Decimal("0"), Decimal("-1")),
])
def test_negative_amounts_raise(factory):
    with pytest.raises(MoneyError):
        factory()


# --- temporal correctness ---
AS_OF = date(2025, 1, 6)
P500K = LiquidityPolicy(Decimal("500000"), Decimal("200000"), Decimal("50000"))


def dated_txn(amount, d, status=TxnStatus.COMPLETED, direction=Direction.INFLOW):
    return Transaction("t", datetime(2025, 1, d, 12, 0), amount,
                       direction, "x", status, "test")


def test_txn_before_as_of_counts():
    assert current_cash(P500K, [dated_txn(Decimal("100000"), 1)], AS_OF) == \
        Decimal("600000.00")


def test_txn_exactly_on_as_of_counts():
    assert current_cash(P500K, [dated_txn(Decimal("100000"), 6)], AS_OF) == \
        Decimal("600000.00")


def test_txn_after_as_of_excluded():
    assert current_cash(P500K, [dated_txn(Decimal("200000"), 10)], AS_OF) == \
        Decimal("500000.00")


def test_pending_or_failed_after_as_of_excluded():
    txns = [dated_txn(Decimal("100"), 3, TxnStatus.PENDING),
            dated_txn(Decimal("100"), 3, TxnStatus.FAILED),
            dated_txn(Decimal("100"), 10)]
    assert current_cash(P500K, txns, AS_OF) == Decimal("500000.00")


def test_temporal_example_exact():
    txns = [dated_txn(Decimal("100000"), 1, direction=Direction.INFLOW),
            dated_txn(Decimal("50000"), 5, direction=Direction.OUTFLOW),
            dated_txn(Decimal("200000"), 10, direction=Direction.INFLOW)]
    assert current_cash(P500K, txns, AS_OF) == Decimal("550000.00")


# --- no hidden today ---
def test_compute_requires_as_of():
    import inspect
    sig = inspect.signature(compute_cash_position)
    assert sig.parameters["as_of"].default is inspect.Parameter.empty
