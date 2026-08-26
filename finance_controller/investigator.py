"""Deterministic baseline investigator.

READ-ONLY with respect to financial truth: consumes InvestigationCase
evidence verbatim, produces an explainable assessment. Never re-matches,
never resolves, never modifies anything upstream. The future LLM
investigator must implement this same interface and be benchmarked
against this baseline.

Confidence rule (documented, deterministic -- confidence in the EXPLANATION,
not in any verdict about correctness/fraudulence):

    high     : CRITICAL/HIGH priority AND strong explicit evidence present
               (numeric amount evidence or candidate lists)
    moderate : MEDIUM priority with adequate evidence
    low      : LOW priority, or missing internal/external amounts

Risk level is a pure translation of InvestigationCase.priority.
No second risk model exists here by design.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from .exceptions import ExceptionType, InvestigationCase, Priority


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Confidence(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


@dataclass(frozen=True)
class InvestigationAssessment:
    case_id: str
    exception_type: str
    risk_level: RiskLevel
    finding: str
    explanation: str
    recommended_action: str
    confidence: Confidence
    evidence_used: tuple[str, ...]
    warnings: tuple[str, ...]


_PRIORITY_TO_RISK = {
    Priority.CRITICAL: RiskLevel.CRITICAL,
    Priority.HIGH: RiskLevel.HIGH,
    Priority.MEDIUM: RiskLevel.MEDIUM,
    Priority.LOW: RiskLevel.LOW,
}


def _money(v: Decimal | None) -> str:
    """Float-free rendering of a Decimal monetary value."""
    if v is None:
        return "unknown"
    return f"₹{v:,.2f}"


def _has_strong_evidence(case: InvestigationCase) -> bool:
    """Explicit numeric/list evidence that materially supports findings."""
    has_amounts = (case.internal_amount is not None
                   or bool(case.external_amounts))
    has_candidates = bool(case.candidate_external_ids
                          or case.external_transaction_ids)
    return has_amounts and has_candidates


def investigate_case(case: InvestigationCase) -> InvestigationAssessment:
    """Interpret one case's evidence deterministically. No side effects."""
    risk = _PRIORITY_TO_RISK[case.priority]
    ev: list[str] = ["exception_type", "priority"]
    warnings: list[str] = []

    et = case.exception_type
    abs_diff = case.amount_difference if case.amount_difference >= 0 \
        else -case.amount_difference

    # ---- per-type interpretation (explanation only; no re-matching) ----
    if et == ExceptionType.AMOUNT_MISMATCH:
        finding = ("Internal and external records share identity signals "
                   f"but disagree on amount by {_money(abs_diff)}.")
        explanation = (
            f"Internal transaction {case.internal_transaction_id} "
            f"({_money(case.internal_amount)}, ref={case.payment_ref!r}) "
            f"corresponds to external record(s) "
            f"{list(case.external_transaction_ids)} "
            f"({_money(case.external_amounts[0]) if case.external_amounts else 'unknown'}). "
            f"The absolute discrepancy is {_money(abs_diff)}. "
            "Reconciliation confirmed reference compatibility "
            f"(reference_match={case.evidence.get('reference_match', 'n/a')}, "
            f"time_delta_seconds={list(case.time_differences_seconds)}), so "
            "the mismatch is isolated to the recorded amounts.")
        action = ("Manually verify both source systems: check the provider "
                  "ledger entry, payment-gateway record, and internal ledger "
                  "posting. Do not assume which side is correct without "
                  "source-of-truth confirmation.")
        ev += ["internal_amount", "external_amounts", "amount_difference",
               "payment_ref", "normalized_reference",
               "time_differences_seconds"]

    elif et == ExceptionType.AMBIGUOUS_MATCH:
        n = len(case.candidate_external_ids)
        finding = (f"{n} external candidates plausibly match internal "
                   f"transaction {case.internal_transaction_id}; the system "
                   "cannot determine the correct one.")
        explanation = (
            f"Internal transaction {case.internal_transaction_id} "
            f"({_money(case.internal_amount)}, ref={case.payment_ref!r}) has "
            f"{n} unconsumed external candidates "
            f"{list(case.candidate_external_ids)} with identical "
            "compatibility evidence. Reconciliation deliberately refused to "
            "arbitrate between them to avoid false matches.")
        action = ("Manual review required: inspect each candidate record in "
                  "the external ledger and determine which corresponds to "
                  "this transaction. Check for duplicate ingestion at the "
                  "provider.")
        ev += ["internal_amount", "candidate_external_ids",
               "payment_ref", "normalized_reference"]
        if n > 2:
            warnings.append(
                f"unusually_high_candidate_count:{n}")

    elif et == ExceptionType.MISSING_EXTERNAL:
        finding = ("No compatible external counterpart was found for "
                   f"internal transaction {case.internal_transaction_id}.")
        explanation = (
            f"Internal transaction {case.internal_transaction_id} "
            f"({_money(case.internal_amount)}, ref={case.payment_ref!r}, "
            f"ts={case.internal_timestamp}) has no matching record in the "
            "external dataset under reconciliation's tolerance rules. "
            "This does NOT prove the external event never occurred; it may "
            "be delayed, mis-referenced, or outside matching tolerances.")
        action = ("Check provider/ledger ingestion completeness for this "
                  "date range and verify the payment reference was correctly "
                  "transmitted. Query the provider directly using "
                  f"payment_ref {case.payment_ref!r}.")
        ev += ["internal_amount", "payment_ref", "internal_timestamp",
               "normalized_reference"]

    elif et == ExceptionType.EXTRA_EXTERNAL:
        finding = ("External record(s) were not consumed by any internal "
                   f"match: {list(case.external_transaction_ids)}.")
        explanation = (
            f"External record(s) {list(case.external_transaction_ids)} "
            f"({_money(case.external_amounts[0]) if case.external_amounts else 'unknown'}, "
            f"ref={case.normalized_reference!r}, "
            f"ts={list(case.external_timestamps)}) have no corresponding "
            "internal transaction. Possible causes include an untracked "
            "internal transaction, a duplicate of another external record, "
            "delayed internal ingestion, or an unrelated record.")
        action = ("Determine whether this represents revenue/activity absent "
                  "from the internal ledger. If it duplicates another "
                  "external record, flag it as a duplicate; otherwise create "
                  "or link the missing internal record after verification.")
        ev += ["external_amounts", "external_transaction_ids",
               "normalized_reference", "external_timestamps"]

    else:  # UNRESOLVED_MATCH
        finding = ("Reconciliation could not establish a valid match under "
                   "any configured strategy.")
        explanation = (
            f"Internal transaction {case.internal_transaction_id} "
            f"(ref={case.payment_ref!r}) failed all reconciliation stages: "
            f"reason={case.reconciliation_reason}, candidates="
            f"{list(case.candidate_external_ids)}. No alternative matching "
            "strategy was attempted by design.")
        action = ("Manual investigation: review raw records on both sides "
                  "and reconcile outside the automated pipeline if needed.")
        ev += ["payment_ref", "candidate_external_ids",
               "reconciliation_reason"]

    if case.financial_impact > 0:
        ev.append("financial_impact")

    conf = (Confidence.LOW
            if (risk is RiskLevel.LOW
                or case.internal_amount is None
                or not case.external_amounts)
            else Confidence.MODERATE
            if risk is RiskLevel.MEDIUM
            else Confidence.HIGH if _has_strong_evidence(case)
            else Confidence.MODERATE)

    return InvestigationAssessment(
        case_id=case.case_id,                       # preserved exactly
        exception_type=et.value,
        risk_level=risk,                            # translated, not recomputed
        finding=finding, explanation=explanation,
        recommended_action=action,
        confidence=conf,
        evidence_used=tuple(dict.fromkeys(ev)),     # deduped, order-stable
        warnings=tuple(warnings))


def investigate_cases(
    cases: list[InvestigationCase],
) -> list[InvestigationAssessment]:
    """Batch investigation. Deterministic, order-preserving, non-mutating."""
    return [investigate_case(c) for c in cases]
