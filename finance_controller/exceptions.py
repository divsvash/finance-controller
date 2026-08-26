"""Deterministic exception/investigation case layer.

Classifies -> structures -> prioritizes -> explains reconciliation
outcomes as evidence for a future investigator (AI or human).

Hard boundary: this layer NEVER re-guesses matches. It consumes the
deterministic reconciliation output verbatim and adds structure only.

Priority rules (documented constants, no randomness):

  CRITICAL if any of:
      - |amount_difference| >= CRITICAL_AMOUNT_DIFFERENCE (₹50,000)
    or MISSING_EXTERNAL / EXTRA_EXTERNAL with amount
       >= CRITICAL_IMPACT_THRESHOLD (₹50,000)
    or AMBIGUOUS_MATCH where any candidate amount >= same threshold

  HIGH if any of:
      - |amount_difference| >= HIGH_AMOUNT_DIFFERENCE (₹10,000)
    or exception type is AMOUNT_MISMATCH or MISSING_EXTERNAL
       (money is unaccounted-for by definition)

  MEDIUM otherwise, except UNRESOLVED_MATCH with zero candidate evidence
  on tiny amounts (< LOW_VALUE_FLOOR ₹1,000) which is LOW.

Ties broken deterministically by (priority_rank, -financial_impact,
case_id) -- never insertion luck alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum, IntEnum

from .models import ExternalRecord, Transaction
from .reconciliation import (
    AMBIGUOUS, MATCHED, ReconciliationResult, normalize_reference)

ZERO = Decimal("0.00")

# ---- configurable priority thresholds ----
CRITICAL_AMOUNT_DIFFERENCE = Decimal("50000")
HIGH_AMOUNT_DIFFERENCE = Decimal("10000")
CRITICAL_IMPACT_THRESHOLD = Decimal("50000")
LOW_VALUE_FLOOR = Decimal("1000")


class ExceptionType(str, Enum):
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
    MISSING_EXTERNAL = "MISSING_EXTERNAL"
    EXTRA_EXTERNAL = "EXTRA_EXTERNAL"
    UNRESOLVED_MATCH = "UNRESOLVED_MATCH"


class Priority(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class InvestigationCase:
    """Immutable structured evidence. Self-contained: an investigator
    needs nothing beyond these fields."""
    case_id: str                          # deterministic: CASE_<seq>_<type>
    sequence: int                         # stable ordinal within the batch
    exception_type: ExceptionType
    priority: Priority

    internal_transaction_id: str | None
    internal_amount: Decimal | None
    internal_timestamp: datetime | None
    payment_ref: str                      # "" when no internal record
    normalized_reference: str

    external_transaction_ids: tuple[str, ...]
    external_amounts: tuple[Decimal, ...]
    external_timestamps: tuple[datetime, ...]

    amount_difference: Decimal            # external - internal (0 if N/A)
    time_differences_seconds: tuple[int, ...]
    reconciliation_reason: str
    candidate_external_ids: tuple[str, ...]

    financial_impact: Decimal             # absolute exposure of this case
    evidence: dict[str, str]              # explicit key=value facts


def _classify(result: ReconciliationResult) -> ExceptionType:
    r = result.reason
    if r == "multiple_candidates":
        return ExceptionType.AMBIGUOUS_MATCH
    if r == "amount_mismatch_no_candidate":
        return ExceptionType.AMOUNT_MISMATCH
    if r == "no_candidate":
        return ExceptionType.MISSING_EXTERNAL \
            if not result.candidate_external_ids else \
            ExceptionType.UNRESOLVED_MATCH
    # Any matched-but-unconsumed externals are handled separately; a
    # MATCHED result never reaches here.
    raise ValueError(f"Unclassifiable reason: {r!r}")


def _priority(etype: ExceptionType, abs_diff: Decimal,
              impact: Decimal) -> Priority:
    if (abs_diff >= CRITICAL_AMOUNT_DIFFERENCE
            or impact >= CRITICAL_IMPACT_THRESHOLD):
        return Priority.CRITICAL
    if etype in (ExceptionType.AMOUNT_MISMATCH,
                 ExceptionType.MISSING_EXTERNAL):
        return Priority.HIGH if abs_diff >= HIGH_AMOUNT_DIFFERENCE \
            else Priority.MEDIUM
    if etype == ExceptionType.UNRESOLVED_MATCH and abs_diff < LOW_VALUE_FLOOR:
        return Priority.LOW
    return Priority.MEDIUM


def build_investigation_cases(
    results: list[ReconciliationResult],
    transactions_by_id: dict[str, Transaction],
    external_records: list[ExternalRecord],
) -> list[InvestigationCase]:
    """Convert reconciliation output into investigation cases.

    Contract:
      * every non-MATCHED result -> exactly one case
      * MATCHED results -> zero cases
      * unconsumed external records -> one EXTRA_EXTERNAL case each
      * inputs are never mutated
      * deterministic ordering: (priority desc, impact desc, sequence)
    """
    consumed_external_ids = {r.external_id for r in results
                             if r.status == MATCHED and r.external_id}
    ext_by_id = {e.id: e for e in external_records}
    extras = [ext_by_id[eid] for eid in sorted(ext_by_id)
              if eid not in consumed_external_ids]

    raw_cases: list[tuple[int, InvestigationCase]] = []
    seq = 0

    def make(case: InvestigationCase) -> None:
        nonlocal seq
        seq += 1
        raw_cases.append((seq, case))

    for res in results:                       # preserve reconciliation order
        if res.status == MATCHED:
            continue
        t = transactions_by_id.get(res.internal_id)
        int_amt = t.amount if t else None
        int_ts = t.timestamp if t else None
        pref = t.payment_ref if t else ""
        norm = normalize_reference(pref) if pref else ""

        cands = [ext_by_id[cid] for cid in res.candidate_external_ids
                 if cid in ext_by_id]

        if cands:                             # ambiguous / mismatch w/ cand
            ext_ids = tuple(c.id for c in cands)
            ext_amts = tuple(c.amount for c in cands)
            ext_tss = tuple(c.timestamp for c in cands)
        elif res.external_id:                 # defensive: unmatched single
            e = ext_by_id.get(res.external_id)
            ext_ids = (res.external_id,)
            ext_amts = (e.amount,) if e else ()
            ext_tss = (e.timestamp,) if e else ()
        else:
            ext_ids, ext_amts, ext_tss = (), (), ()

        diffs = [a - int_amt for a in ext_amts] if int_amt is not None else []
        abs_diff = max((abs(d) for d in diffs), default=abs(res.amount_difference))
        impact = abs_diff
        etype = _classify(res)
        prio = _priority(etype, abs_diff, impact)

        ev = {
            "reference_match": str(bool(norm and any(
                normalize_reference(e.external_reference) == norm
                for e in cands))),
            "direction_match": str(all(
                e.direction == t.direction for e in cands)) if t and cands
                else "n/a",
            "status_internal": t.status.value if t else "unknown",
            "candidate_count": str(len(cands)),
            "reconciliation_status": res.status,
        }

        make(InvestigationCase(
            case_id=f"CASE_{seq:05d}_{etype.value}", sequence=seq,
            exception_type=etype, priority=prio,
            internal_transaction_id=res.internal_id,
            internal_amount=int_amt, internal_timestamp=int_ts,
            payment_ref=pref, normalized_reference=norm,
            external_transaction_ids=ext_ids,
            external_amounts=ext_amts, external_timestamps=ext_tss,
            amount_difference=res.amount_difference,
            time_differences_seconds=tuple(
                int((e.timestamp - t.timestamp).total_seconds())
                for e in cands) if t else (),
            reconciliation_reason=res.reason,
            candidate_external_ids=res.candidate_external_ids,
            financial_impact=impact.quantize(Decimal("0.01")),
            evidence=ev))

    for e in extras:                          # external-only records
        impact = e.amount
        prio = _priority(ExceptionType.EXTRA_EXTERNAL, ZERO, impact)
        make(InvestigationCase(
            case_id=f"CASE_{seq + 1:05d}_{ExceptionType.EXTRA_EXTERNAL.value}",
            sequence=seq + 1,
            exception_type=ExceptionType.EXTRA_EXTERNAL,
            priority=prio,
            internal_transaction_id=None, internal_amount=None,
            internal_timestamp=None, payment_ref="",
            normalized_reference=normalize_reference(e.external_reference),
            external_transaction_ids=(e.id,),
            external_amounts=(e.amount,),
            external_timestamps=(e.timestamp,),
            amount_difference=e.amount,
            time_differences_seconds=(),
            reconciliation_reason="unconsumed_external_record",
            candidate_external_ids=(),
            financial_impact=impact.quantize(Decimal("0.01")),
            evidence={
                "reference_match": "false",
                "direction_match": "n/a",
                "status_external": e.status.value,
                "source": e.source,
                "reconciliation_status": "UNMATCHED_EXTERNAL",
            }))
        seq += 1

    raw_cases.sort(key=lambda p: (-p[1].priority, -p[1].financial_impact,
                                  p[0]))
    return [c for _, c in raw_cases]
