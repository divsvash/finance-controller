"""Deterministic, explainable inflow forecasting baseline.

No ML, no randomness. Consumes completed historical inflow Transactions,
produces a structured ForecastResult consumable by compute_cash_position
via expected_inflows=... .

Methodology (documented thresholds):

  History length = distinct calendar days between the first qualifying
  transaction and as_of (inclusive).

    history_days >= DOW_MIN_DAYS (56) AND every weekday has >=
    DOW_MIN_OBS_PER_DAY (4) observations:
        -> day-of-week baseline: per-weekday trimmed mean of daily inflows,
           scaled by the overall ratio of total to weekday-mean inflow.
        -> methodology "dow_trimmed_mean"

    history_days >= BASELINE_MIN_DAYS (14):
        -> overall daily median of non-zero inflow days, methodology
           "overall_median_daily"

    else:
        -> conservative fallback: zero expected inflow, methodology
           "insufficient_history_conservative_zero"

Anomalies: daily totals beyond median +/- ANOMALY_IQR_FACTOR * IQR of daily
totals are flagged and excluded from baseline statistics. Detection is
recorded, never silent.

Uncertainty: half-width = UNCERTAINTY_MAD_FACTOR * MAD(daily inflow totals),
clamped to at least MIN_BOUND_FRACTION * point forecast so bounds are never
degenerate. High variance -> wider bounds; stable history -> narrow.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from statistics import median

from .models import Direction, Transaction, TxnStatus

ZERO = Decimal("0.00")

# ---- documented thresholds ----
DOW_MIN_DAYS = 56            # need >= 8 weeks of history for dow effects
DOW_MIN_OBS_PER_DAY = 4      # >= 4 observed occurrences per weekday
BASELINE_MIN_DAYS = 14       # >= 2 weeks before we forecast anything nonzero
ANOMALY_IQR_FACTOR = Decimal("3")     # beyond 3*IQR fences == extreme
TRIM_FRACTION = 0.1          # trim 10% each side for trimmed mean
UNCERTAINTY_MAD_FACTOR = Decimal("2")  # +/- 2*MAD half-width
MIN_BOUND_FRACTION = Decimal("0.25")   # lower bound >= 75%? no: see below


@dataclass(frozen=True)
class ForecastPoint:
    date: date
    expected_inflow: Decimal
    lower_bound: Decimal
    upper_bound: Decimal
    reliability: str            # "high" | "moderate" | "low"


@dataclass(frozen=True)
class AnomalyRecord:
    transaction_id: str
    amount: Decimal
    reason: str                 # explicit detection evidence


@dataclass(frozen=True)
class ForecastResult:
    as_of: date
    horizon_days: int
    points: tuple[ForecastPoint, ...]
    total_expected_inflow: Decimal
    total_lower_bound: Decimal
    total_upper_bound: Decimal
    methodology: str
    training_transaction_count: int
    excluded_transaction_count: int
    anomalies: tuple[AnomalyRecord, ...] = ()
    history_days: int = 0


def _trimmed_mean(values: list[Decimal], fraction: float) -> Decimal:
    """Mean after dropping `fraction` from each tail (deterministic)."""
    if len(values) < 3 or fraction <= 0:
        return sum(values, ZERO) / len(values)
    k = max(1, int(len(values) * fraction))
    core = sorted(values)[k:len(values) - k]
    return sum(core, ZERO) / len(core)


def _mad(values: list[Decimal]) -> Decimal:
    """Median absolute deviation about the median."""
    if not values:
        return ZERO
    med = median(values)
    return median([abs(v - med) for v in values])


def detect_anomalies(
    daily_totals: dict[date, Decimal],
) -> tuple[set[date], list[AnomalyRecord]]:
    """Flag extreme inflow days: outside median +/- ANOMALY_IQR_FACTOR*IQR.

    Deterministic; nothing is deleted here -- callers decide exclusion.
    """
    values = sorted(daily_totals.values())
    flags: set[date] = set()
    records: list[AnomalyRecord] = []
    if len(values) < 8:
        return flags, records
    q1, q3 = values[len(values) // 4], values[(3 * len(values)) // 4]
    iqr = q3 - q1
    if iqr == 0:
        return flags, records
    lo, hi = q1 - ANOMALY_IQR_FACTOR * iqr, q3 + ANOMALY_IQR_FACTOR * iqr
    med = median(values)
    for d, v in sorted(daily_totals.items()):
        if v < lo or v > hi:
            side = "above_upper_fence" if v > hi else "below_lower_fence"
            flags.add(d)
            records.append(AnomalyRecord(
                transaction_id=f"day_{d.isoformat()}",
                amount=v,
                reason=f"daily_total_{side}_median={med}_iqr={iqr}"))
    return flags, records


def _qualifying(txns: list[Transaction], as_of: datetime_like) \
        -> list[Transaction]:
    """Completed inflow transactions strictly usable for training.

    Leakage rule: only transactions with timestamp.date() < as_of OR on
    as_of itself are included (as_of treated as end-of-day snapshot).
    """
    cutoff_end = _end_of(as_of)
    return [t for t in txns
            if t.direction == Direction.INFLOW
            and t.status == TxnStatus.COMPLETED
            and t.timestamp <= cutoff_end]


def forecast_inflows(
    transactions: list[Transaction],
    as_of: date,
    horizon_days: int,
) -> ForecastResult:
    """Produce deterministic daily inflow forecasts for the horizon."""
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    qualifying = _qualifying(transactions, as_of)
    by_day: dict[date, Decimal] = defaultdict(lambda: ZERO)
    for t in qualifying:
        by_day[t.timestamp.date()] += t.amount

    all_days = set(by_day)
    anomaly_flags, anomaly_records = detect_anomalies(by_day)

    excluded_ids: set[str] = set()
    train_by_day = {d: v for d, v in by_day.items()
                    if d not in anomaly_flags}
    for t in qualifying:
        if t.timestamp.date() in anomaly_flags:
            excluded_ids.add(t.id)
    train_txns = [t for t in qualifying if t.id not in excluded_ids]

    first_day = min(all_days) if all_days else as_of
    history_days = (min(as_of, max(all_days, default=as_of)) - first_day).days + 1 \
        if all_days else 0

    totals = list(train_by_day.values())
    active_days = [v for v in totals if v > 0]

    # --- choose methodology ---
    dow_ok = False
    if history_days >= DOW_MIN_DAYS and len(active_days) >= 7 * DOW_MIN_OBS_PER_DAY // 2:
        by_weekday: dict[int, list[Decimal]] = defaultdict(list)
        n_weekday_obs: dict[int, int] = defaultdict(int)
        d = first_day
        while d <= as_of:
            if d in train_by_day:
                by_weekday[d.weekday()].append(train_by_day[d])
            n_weekday_obs[d.weekday()] += 1
            d += timedelta(days=1)
        counts = {wd: len(v) for wd, v in by_weekday.items()}
        dow_ok = bool(counts) and min(counts.values()) >= DOW_MIN_OBS_PER_DAY \
            and len(counts) == 7

    if dow_ok:
        methodology = "dow_trimmed_mean"
        base = {wd: _trimmed_mean(vals, TRIM_FRACTION)
                for wd, vals in by_weekday.items()}
        overall = _trimmed_mean(active_days, TRIM_FRACTION)
        scale = (sum(base.values(), ZERO) / 7 / overall
                 if overall > 0 else Decimal(1))
    elif history_days >= BASELINE_MIN_DAYS and active_days:
        methodology = "overall_median_daily"
        base = None
        overall = median(active_days)
        scale = Decimal(1)
    else:
        methodology = "insufficient_history_conservative_zero"
        overall = ZERO
        scale = Decimal(1)

    variability = (_mad(active_days) if active_days else ZERO)
    reliability = ("high" if methodology == "dow_trimmed_mean"
                   else "moderate" if methodology == "overall_median_daily"
                   else "low")

    points: list[ForecastPoint] = []
    for i in range(1, horizon_days + 1):
        fdate = as_of + timedelta(days=i)
        if dow_ok:
            raw = base[fdate.weekday()]
            point = raw * scale if scale != Decimal(1) else raw
            point = max(ZERO, point.quantize(Decimal("0.01")))
        elif methodology == "overall_median_daily":
            point = max(ZERO, Decimal(str(round(float(overall), 2))))
        else:
            point = ZERO
        half = variability * UNCERTAINTY_MAD_FACTOR
        floor_half = point * MIN_BOUND_FRACTION if point > 0 else ZERO
        half = max(half, floor_half).quantize(Decimal("0.01"))
        points.append(ForecastPoint(
            date=fdate, expected_inflow=point,
            lower_bound=max(ZERO, point - half),
            upper_bound=point + half,
            reliability=reliability))

    return ForecastResult(
        as_of=as_of, horizon_days=horizon_days, points=tuple(points),
        total_expected_inflow=sum((p.expected_inflow for p in points), ZERO),
        total_lower_bound=sum((p.lower_bound for p in points), ZERO),
        total_upper_bound=sum((p.upper_bound for p in points), ZERO),
        methodology=methodology,
        training_transaction_count=len(train_txns),
        excluded_transaction_count=len(excluded_ids),
        anomalies=tuple(anomaly_records),
        history_days=history_days)


# helpers shared with engine-style semantics
from datetime import datetime as _dt, time as _time
from typing import Union as _U

datetime_like = _U[_dt, date]


def _end_of(as_of: datetime_like) -> _dt:
    if isinstance(as_of, _dt):
        return as_of
    return _dt.combine(as_of, _time(23, 59, 59, 999999))
