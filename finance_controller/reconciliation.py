"""Deterministic staged reconciliation engine.

Knows nothing about LLMs, agents, APIs, or storage. Accepts normalized
records, returns structured results + metrics. Correctness prioritized
over recall: vague matches are never forced.

Stage 1  exact_reference          : normalized reference uniquely identifies
                                    exactly one unconsumed external record
                                    with compatible amount/direction.
Stage 2  amount_direction_ts      : unique candidate with identical amount +
                                    direction and |dt| <= ts_tolerance.
Stage 3  reference_amount_date    : same normalized reference + amount +
                                    direction on the same calendar date
                                    (recovers large clock skew).
Anything else is AMBIGUOUS (multiple candidates at any stage) or UNRESOLVED.

One-to-many protection: each external record can be consumed once per run;
each internal record produces exactly one result row.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal

from .models import Direction, ExternalRecord, Transaction, TxnStatus, money

ZERO = Decimal("0.00")
MATCHED = "MATCHED"
AMBIGUOUS = "AMBIGUOUS"
UNRESOLVED = "UNRESOLVED"


def normalize_reference(ref: str) -> str:
    """Uppercase, strip separators/punctuation: 'pay_ABc-123' -> 'PAYABC123'."""
    return "".join(c for c in ref.upper() if c.isalnum())


@dataclass(frozen=True)
class ReconciliationResult:
    internal_id: str
    status: str                       # MATCHED / AMBIGUOUS / UNRESOLVED
    reason: str                       # explicit matching evidence
    external_id: str | None = None
    amount_difference: Decimal = ZERO     # external - internal
    time_difference_seconds: int = 0      # external - internal
    candidate_external_ids: tuple[str, ...] = ()


@dataclass
class ReconciliationReport:
    total_internal_records: int
    total_external_records: int
    matched_count: int
    ambiguous_count: int
    unresolved_count: int
    # Exceptions (external-side anomalies):
    duplicate_external_count: int     # external records sharing a normalized
                                      # reference AND amount AND direction
                                      # with another external record
    extra_external_count: int         # external records left unconsumed
    amount_mismatch_count: int        # external records whose normalized
                                      # reference matches an internal record
                                      # but whose amount differs
    @property
    def match_rate(self) -> float:
        return self.matched_count / max(self.total_internal_records, 1)

    @property
    def ambiguity_rate(self) -> float:
        return self.ambiguous_count / max(self.total_internal_records, 1)

    @property
    def unresolved_rate(self) -> float:
        return self.unresolved_count / max(self.total_internal_records, 1)


def _compatible(ext: ExternalRecord, txn: Transaction,
                exact_amount: bool) -> bool:
    if ext.direction != txn.direction:
        return False
    if ext.status != TxnStatus.COMPLETED or txn.status != TxnStatus.COMPLETED:
        return False
    return ext.amount == txn.amount if exact_amount else True


def reconcile(
    transactions: list[Transaction],
    external_records: list[ExternalRecord],
    ts_tolerance: timedelta = timedelta(minutes=5),
    enable_date_fallback: bool = True,
) -> tuple[list[ReconciliationResult], ReconciliationReport]:
    """Run staged reconciliation. Deterministic: internals processed in
    given order; consumption tracked across all stages."""
    by_normref: dict[str, list[ExternalRecord]] = defaultdict(list)
    for e in external_records:
        by_normref[normalize_reference(e.external_reference)].append(e)

    consumed: set[str] = set()

    # Exception detection (purely descriptive; does NOT affect matching).
    sig_counts: dict[tuple, int] = defaultdict(int)
    for e in external_records:
        sig_counts[(normalize_reference(e.external_reference),
                    e.amount, e.direction)] += 1
    duplicates = {e.id for e in external_records
                  if sig_counts[(normalize_reference(e.external_reference),
                                 e.amount, e.direction)] > 1}

    results: list[ReconciliationResult] = []

    def diff(t: Transaction, e: ExternalRecord) -> tuple[Decimal, int]:
        return (money(e.amount - t.amount),
                int((e.timestamp - t.timestamp).total_seconds()))

    for t in transactions:
        norm = normalize_reference(t.source)   # internal ref lives in `source`
        cands_ref = [e for e in by_normref.get(norm, [])
                     if e.id not in consumed]
        cands_strong = [e for e in cands_ref if _compatible(e, t, True)]

        result: ReconciliationResult | None = None

        # Stage 1 — exact normalized reference
        if len(cands_strong) == 1:
            e = cands_strong[0]
            ad, td = diff(t, e)
            result = ReconciliationResult(
                t.id, MATCHED, "exact_reference", e.id, ad, td)
        elif len(cands_strong) > 1:
            result = ReconciliationResult(
                t.id, AMBIGUOUS, "multiple_candidates",
                candidate_external_ids=tuple(sorted(e.id for e in cands_strong)))

        # Stage 2 — amount + direction + timestamp tolerance over ALL
        # unconsumed externals. No reference requirement whatsoever.
        if result is None:
            near = [e for e in external_records
                    if e.id not in consumed
                    and _compatible(e, t, exact_amount=True)
                    and abs(e.timestamp - t.timestamp) <= ts_tolerance]
            if len(near) == 1:
                e = near[0]
                ad, td = diff(t, e)
                result = ReconciliationResult(
                    t.id, MATCHED, "amount_direction_timestamp",
                    e.id, ad, td)
            elif len(near) > 1:
                result = ReconciliationResult(
                    t.id, AMBIGUOUS, "multiple_candidates",
                    candidate_external_ids=tuple(sorted(e.id for e in near)))

        # Stage 3 — controlled fallback: SAME normalized reference +
        # amount + direction + same calendar date (recovers large skew).
        if result is None and enable_date_fallback:
            same_day = [e for e in cands_ref          # reference-constrained
                        if _compatible(e, t, True)
                        and e.timestamp.date() == t.timestamp.date()]
            if len(same_day) == 1:
                e = same_day[0]
                ad, td = diff(t, e)
                result = ReconciliationResult(
                    t.id, MATCHED, "reference_amount_date", e.id, ad, td)
            elif len(same_day) > 1:
                result = ReconciliationResult(
                    t.id, AMBIGUOUS, "multiple_candidates",
                    candidate_external_ids=tuple(sorted(e.id for e in same_day)))

        if result is None:
            reason = ("amount_mismatch_no_candidate"
                      if cands_ref and not _compatible(cands_ref[0], t, True)
                      else "no_candidate")
            ad = (money(cands_ref[0].amount - t.amount) if cands_ref else ZERO)
            td = (int((cands_ref[0].timestamp - t.timestamp).total_seconds())
                  if cands_ref else 0)
            result = ReconciliationResult(t.id, UNRESOLVED, reason,
                                          None, ad, td)
        else:
            if result.external_id is not None:
                consumed.add(result.external_id)

        results.append(result)

    amount_mismatches = sum(1 for r in results
                            if r.reason == "amount_mismatch_no_candidate")
    report = ReconciliationReport(
        total_internal_records=len(transactions),
        total_external_records=len(external_records),
        matched_count=sum(r.status == MATCHED for r in results),
        ambiguous_count=sum(r.status == AMBIGUOUS for r in results),
        unresolved_count=sum(r.status == UNRESOLVED for r in results),
        duplicate_external_count=len(duplicates),
        extra_external_count=len(external_records) - len(consumed),
        amount_mismatch_count=amount_mismatches,
    )
    return results, report


# ------------------------- Ground truth + evaluation -------------------------

def evaluate(results: list[ReconciliationResult],
             ground_truth: dict[str, str | None]) -> dict[str, float | int]:
    """Score matched results against held-out ground truth.

    ground_truth: external_id -> true internal_id (or None for injected
    extras). Not visible to the engine.

    TP: matched pair where ground_truth[external_id] == internal_id.
    FP: matched pair pointing at the wrong external record.
    FN: internal records having a true counterpart (some external record
        maps back to them) that did not end MATCHED to that counterpart.
    """
    tp = fp = 0
    matched_internal = {r.internal_id: r.external_id
                        for r in results if r.status == MATCHED}
    for r in results:
        if r.status != MATCHED or r.external_id is None:
            continue
        truth = ground_truth.get(r.external_id)
        if truth == r.internal_id:
            tp += 1
        else:
            fp += 1
    true_partners = {iid for iid in ground_truth.values() if iid is not None}
    fn = sum(1 for iid, eid in matched_internal.items()
             if iid in true_partners and ground_truth.get(eid) != iid)
    fn += sum(1 for iid in true_partners if iid not in matched_internal)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "true_positive_count": tp,
        "false_positive_count": fp, "false_negative_count": fn,
    }
