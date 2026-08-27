"""Orchestration layer for the finance-controller backend.

Pipeline stages (all pre-existing modules; nothing re-implemented here):

    generate/supplied data -> reconcile -> InvestigationCase(s)
        -> deterministic investigation
        -> [optional] LLM investigation via injected client
        -> [optional] evaluation

TRUST BOUNDARY (unchanged): the deterministic InvestigationCase remains
the sole source of financial truth. LLM output is interpretation only;
identity fields are protected inside llm_investigator.py and the pipeline
never uses LLM output for any financial decision.

No global mutable state. Exceptions propagate; no silent fallbacks.
The real OpenAICompatibleClient is never auto-instantiated here -- the
caller must inject any object implementing generate(prompt)->str.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .exceptions import InvestigationCase, build_investigation_cases
from .evaluation import EvaluationResult, EvaluationSummary, evaluate_batch
from .investigator import (
    InvestigationAssessment, investigate_cases)
from .llm_investigator import LLMClient, llm_investigate_cases
from .models import ExternalRecord, Transaction
from .reconciliation import reconcile


@dataclass(frozen=True)
class PipelineResult:
    reconciliation_results: tuple            # as returned by reconcile()
    reconciliation_report: Any               # second return value of reconcile
    investigation_cases: tuple[InvestigationCase, ...]
    deterministic_assessments: tuple[InvestigationAssessment, ...]
    llm_assessments: Optional[tuple[InvestigationAssessment, ...]] = None
    evaluation_results: Optional[tuple[EvaluationResult, ...]] = None
    evaluation_summary: Optional[EvaluationSummary] = None

    @property
    def case_count(self) -> int:
        return len(self.investigation_cases)


def run_pipeline(
    transactions: Sequence[Transaction],
    external_records: Sequence[ExternalRecord],
    llm_client: LLMClient | None = None,
    run_llm: bool = False,
    run_evaluation: bool = False,
    enable_date_fallback: bool = False,
) -> PipelineResult:
    """Run the full orchestration deterministically.

    Args:
        transactions: caller-supplied transactions (not mutated).
        external_records: caller-supplied external records (not mutated).
        llm_client: required when run_llm is True. Must implement
            generate(prompt)->str. Never auto-created; no fallback to a
            fake client.
        run_llm: enable the optional LLM investigation stage.
        run_evaluation: enable the evaluation harness. Requires LLM mode;
            raises ValueError otherwise.
        enable_date_fallback: passed through to reconcile().

    Raises:
        ValueError: run_evaluation without run_llm, or run_llm without a
            client. All lower-layer exceptions propagate untouched.
    """
    if run_evaluation and not run_llm:
        raise ValueError(
            "run_evaluation requires run_llm=True: there is no LLM "
            "assessment to evaluate.")
    if run_llm and llm_client is None:
        raise ValueError(
            "run_llm=True requires an injected llm_client implementing "
            "LLMClient.generate(prompt). The pipeline does not create "
            "clients itself.")

    txns = list(transactions)
    exts = list(external_records)

    results, report = reconcile(txns, exts,
                                enable_date_fallback=enable_date_fallback)
    cases = build_investigation_cases(results, {t.id: t for t in txns}, exts)

    det_assessments = investigate_cases(cases)

    llm_assessments: Optional[tuple[InvestigationAssessment, ...]] = None
    eval_results: Optional[tuple[EvaluationResult, ...]] = None
    eval_summary: Optional[EvaluationSummary] = None

    if run_llm:
        llm_list = llm_investigate_cases(cases, llm_client)
        llm_assessments = tuple(llm_list)
        if run_evaluation:
            er, es = evaluate_batch(cases, det_assessments, list(llm_list))
            eval_results = tuple(er)
            eval_summary = es

    return PipelineResult(
        reconciliation_results=tuple(results),
        reconciliation_report=report,
        investigation_cases=tuple(cases),
        deterministic_assessments=tuple(det_assessments),
        llm_assessments=llm_assessments,
        evaluation_results=eval_results,
        evaluation_summary=eval_summary)

# --- additive imports ---
from .treasury import (
    CashPosition, ControllerPolicy, ExpectedFlow,
    TreasurySummary, compute_treasury_summary)

# --- PipelineResult gains ONE trailing optional field ---
@dataclass(frozen=True)
class PipelineResult:
    # ...all existing fields unchanged...
    llm_assessments: Optional[...] = None
    evaluation_results: ... = None
    evaluation_summary: ... = None
    treasury_summary: Optional[TreasurySummary] = None   # NEW, default None

def run_pipeline(
    transactions,
    external_records,
    *,
    run_llm: bool = False,
    run_evaluation: bool = False,
    enable_date_fallback: bool = False,
    llm_client=None,
    # --- NEW optional treasury inputs (all-or-none, see below) ---
    cash_position: Optional[CashPosition] = None,
    expected_flows: Optional[list[ExpectedFlow]] = None,
    treasury_policy: Optional[ControllerPolicy] = None,
):

    # ---- treasury (additive; all-or-none validation rule) ----
    supplied = [cash_position is not None,
                expected_flows is not None,
                treasury_policy is not None]
    if any(supplied) and not all(supplied):
        raise ValueError(
            "partial treasury inputs are not allowed: supply ALL of "
            "cash_position, expected_flows, treasury_policy — or none")
    treasury_summary = None
    if all(supplied):
        # Pure deterministic computation over DECLARED inputs only.
        # No inference from transactions/external_records.
        # linked_transaction_id remains inert metadata (not connected
        # to reconciliation results in this task).
        treasury_summary = compute_treasury_summary(
            cash_position, expected_flows, treasury_policy)

# --- additive import ---
from .controller import ControllerDecision, evaluate_treasury_decision

# --- PipelineResult gains ONE trailing optional field ---
    treasury_summary: Optional[TreasurySummary] = None
    controller_decision: Optional[ControllerDecision] = None   # NEW

# --- run_pipeline() gains ONE optional kwarg ---
def run_pipeline(
    transactions,
    external_records,
    *,
    run_llm: bool = False,
    run_evaluation: bool = False,
    enable_date_fallback: bool = False,
    llm_client=None,
    cash_position=None,
    expected_flows=None,
    treasury_policy=None,
    # --- NEW ---
    proposed_amount: Optional[Decimal] = None,
):
