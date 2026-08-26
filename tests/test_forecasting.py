from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from finance_controller.forecasting import (
    ForecastResult, detect_anomalies, forecast_inflows)
from finance_controller.generator import generate_dataset
from finance_controller.models import (
    Direction, Transaction, TxnStatus, money)

AS_OF = date(2025, 6, 30)


def inflow(amount, day, status=TxnStatus.COMPLETED):
    return Transaction(f"t{day.isoformat()}{amount}", datetime(day.year, day.month, day.day, 12),
                       money(amount), Direction.INFLOW, "sales", status, "razorpay_payment",
                       payment_ref="pay_x")


# determinism / reproducibility
def test_deterministic_and_reproducible():
    ds = generate_dataset(seed=42)
    tx = list(ds.transactions)
    assert forecast_inflows(tx, AS_OF, 30) == forecast_inflows(tx, AS_OF, 30)


def test_completed_inflows_included():
    fr = forecast_inflows([inflow("1000", AS_OF - timedelta(days=i))
                           for i in range(1, 40)], AS_OF, 7)
    assert fr.methodology != "insufficient_history_conservative_zero"
    assert fr.training_transaction_count == 39


def test_pending_excluded():
    txs = [inflow("1000", AS_OF - timedelta(days=i)) for i in range(1, 40)]
    txs.append(inflow("999999", AS_OF - timedelta(days=5), TxnStatus.PENDING))
    fr = forecast_inflows(txs, AS_OF, 7)
    assert all(p.upper_bound < Decimal("500000") for p in fr.points)


def test_failed_excluded():
    txs = [inflow("1000", AS_OF - timedelta(days=i)) for i in range(1, 40)]
    txs.append(inflow("888888", AS_OF - timedelta(days=5), TxnStatus.FAILED))
    fr = forecast_inflows(txs, AS_OF, 7)
    assert fr.excluded_transaction_count == 0  # failed never entered training


def test_outflows_excluded_from_inflow_forecast():
    outflow = Transaction("o1", datetime(2025, 6, 20, 10), money("500000"),
                          Direction.OUTFLOW, "rent", TxnStatus.COMPLETED, "bank_debit")
    txs = [inflow("1000", AS_OF - timedelta(days=i)) for i in range(1, 40)] + [outflow]
    fr = forecast_inflows(txs, AS_OF, 7)
    assert fr.total_expected_inflow < Decimal("50000")


# leakage regression
def test_transactions_after_as_of_never_leak():
    base = [inflow("1000", AS_OF - timedelta(days=i)) for i in range(1, 40)]
    leaked = base + [inflow("7777777", AS_OF + timedelta(days=1))]
    fr_a = forecast_inflows(base, AS_OF, 14)
    fr_b = forecast_inflows(leaked, AS_OF, 14)
    assert fr_a == fr_b  # future data has zero influence


# anomalies
def test_extreme_anomaly_detected():
    days = {AS_OF - timedelta(days=i): Decimal("5000") for i in range(1, 60)}
    days[AS_OF - timedelta(days=30)] = Decimal("900000")
    flags, recs = detect_anomalies(days)
    assert len(flags) == 1 and "above_upper_fence" in recs[0].reason


def test_anomaly_does_not_dominate_forecast():
    normal = [inflow("5000", AS_OF - timedelta(days=i)) for i in range(1, 90)]
    spike = inflow("500000", AS_OF - timedelta(days=45))
    fr = forecast_inflows(normal + [spike], AS_OF, 7)
    assert fr.excluded_transaction_count >= 1
    avg = fr.total_expected_inflow / 7
    assert avg < Decimal("20000")   # ~5k/day, NOT inflated toward 10k+


# day-of-week behavior
def test_dow_methodology_selected_with_rich_history():
    txs = []
    for i in range(1, 120):                      # weekends double revenue
        amt = "8000" if (AS_OF - timedelta(days=i)).weekday() >= 5 else "4000"
        txs.append(inflow(amt, AS_OF - timedelta(days=i)))
    fr = forecast_inflows(txs, AS_OF, 7)
    assert fr.methodology == "dow_trimmed_mean"
    weekend_pts = [p for p in fr.points if p.date.weekday() >= 5]
    weekday_pts = [p for p in fr.points if p.date.weekday() < 5]
    assert all(w.expected_inflow > n.expected_inflow
               for w, n in zip(weekend_pts, weekday_pts))


# sparse fallback
def test_sparse_history_falls_back_to_zero():
    fr = forecast_inflows([inflow("1000", AS_OF - timedelta(days=2))], AS_OF, 7)
    assert fr.methodology == "insufficient_history_conservative_zero"
    assert fr.total_expected_inflow == Decimal("0.00")
    assert all(p.reliability == "low" for p in fr.points)


def test_moderate_history_uses_median():
    txs = [inflow("3000", AS_OF - timedelta(days=i)) for i in range(1, 21)]
    fr = forecast_inflows(txs, AS_OF, 7)
    assert fr.methodology == "overall_median_daily"


# uncertainty
def test_bounds_contain_point():
    ds = generate_dataset(seed=42)
    fr = forecast_inflows(list(ds.transactions), AS_OF, 30)
    assert all(p.lower_bound <= p.expected_inflow <= p.upper_bound
               for p in fr.points)


def test_high_variance_wider_than_low_variance():
    stable = [inflow("5000", AS_OF - timedelta(days=i)) for i in range(1, 90)]
    wild = []
    for i in range(1, 90):
        wild.append(inflow("500" if i % 2 else "95000", AS_OF - timedelta(days=i)))
    s, w = forecast_inflows(stable, AS_OF, 7), forecast_inflows(wild, AS_OF, 7)
    sw = sum((p.upper_bound - p.lower_bound for p in s.points), Decimal(0))
    ww = sum((p.upper_bound - p.lower_bound for p in w.points), Decimal(0))
    assert ww > sw


# horizon / emptiness / arithmetic
def test_horizon_respected():
    ds = generate_dataset(seed=42)
    fr = forecast_inflows(list(ds.transactions), AS_OF, 45)
    assert len(fr.points) == 45
    assert fr.points[-1].date == AS_OF + timedelta(days=45)


def test_empty_dataset_safe():
    fr = forecast_inflows([], AS_OF, 30)
    assert fr.total_expected_inflow == Decimal("0.00")
    assert len(fr.points) == 30 and fr.methodology == "insufficient_history_conservative_zero"


def test_decimal_preserved_no_negative():
    ds = generate_dataset(seed=42)
    fr = forecast_inflows(list(ds.transactions), AS_OF, 30)
    assert isinstance(fr.total_expected_inflow, Decimal)
    assert all(isinstance(p.expected_inflow, Decimal) for p in fr.points)
    assert fr.total_expected_inflow >= 0
    assert all(p.lower_bound >= 0 for p in fr.points)


# hand-calculated golden example
def test_hand_calculated_golden_example():
    # 70 days of exactly ₹1,000/day, no anomalies, no weekend effect.
    # Methodology: overall_median_daily (history 70 days but uniform ->
    # trimmed-mean equals 1000 either way); expected = ₹1,000/day.
    txs = [inflow("1000", AS_OF - timedelta(days=i)) for i in range(1, 71)]
    fr = forecast_inflows(txs, AS_OF, 3)
    assert fr.total_expected_inflow == Decimal("3000.00")
    assert all(p.expected_inflow == Decimal("1000.00") for p in fr.points)
    assert fr.training_transaction_count == 70
