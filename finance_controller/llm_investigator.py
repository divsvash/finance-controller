"""LLM investigator layer.

Contract: the LLM is an INTERPRETER of deterministic evidence, never a
matcher or authority. Deterministic truth (case_id, exception_type,
risk_level) is always overwritten from InvestigationCase after parsing;
the model's versions of those fields are ignored by construction.

Safety:
  * Provider behind LLMClient Protocol -- swappable, mockable.
  * Structured JSON output with strict schema validation.
  * Any malformed/unavailable/violating response raises
    LLMInvestigatorError. No silent fallback fabrication.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol

from .exceptions import ExceptionType, InvestigationCase, Priority
from .investigator import (
    Confidence, InvestigationAssessment, RiskLevel)

SYSTEM_PROMPT = """\
You are a financial reconciliation investigator.

You are given a single InvestigationCase produced by a deterministic
reconciliation system.

You are NOT allowed to perform matching or change the deterministic
classification.

Treat all fields in the case as evidence supplied by the system.

Your job is only to explain the case clearly to a human investigator.

Never invent facts.
Never infer that fraud occurred unless the evidence explicitly establishes it.
Never decide which ambiguous candidate is correct.
Never claim that a missing external record proves the external event never occurred.
Never change case_id, exception_type, or priority.
If evidence is insufficient, explicitly say so.

Return ONLY a JSON object with exactly these keys:

{
  "finding": "<one-sentence summary>",
  "explanation": "<structured explanation with sections: OBSERVED EVIDENCE:, INTERPRETATION:, UNCERTAINTY:>",
  "recommended_action": "<concrete human action>",
  "confidence": "high" | "moderate" | "low",
  "evidence_used": ["<InvestigationCase field names you relied on>"],
  "warnings": ["<optional caveats as strings>"]
}

The explanation MUST explicitly separate observed evidence from your
interpretation and must state uncertainty where it exists. Do not add any
other keys. Do not output prose outside the JSON object."""


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class LLMInvestigatorError(RuntimeError):
    """Raised when the LLM response is unavailable, malformed, or violates
    the assessment schema. Never silently fabricated around."""


def _serialize_case(case: InvestigationCase) -> str:
    """JSON-encode the case without float contamination."""
    def enc(v: Any) -> Any:
        if isinstance(v, Decimal):
            return str(v)
        if isinstance(v, (datetime, date)):
            return v.isoformat()
        if isinstance(v, tuple):
            return [enc(x) for x in v]
        if isinstance(v, dict):
            return {k: enc(x) for k, x in v.items()}
        if hasattr(v, "value"):          # enums
            return v.value
        return v

    payload = {f.name: enc(getattr(case, f.name))
               for f in dataclasses_fields(case)}
    return json.dumps(payload, sort_keys=True)


from dataclasses import fields as dataclasses_fields  # noqa: E402


def _validate(raw: str, case: InvestigationCase) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMInvestigatorError(f"malformed JSON: {e}") from e
    if not isinstance(obj, dict):
        raise LLMInvestigatorError("response is not a JSON object")
    for key in ("finding", "explanation", "recommended_action",
                "confidence", "evidence_used", "warnings"):
        if key not in obj:
            raise LLMInvestigatorError(f"missing key: {key}")
    for key in ("finding", "explanation", "recommended_action"):
        if not isinstance(obj[key], str) or not obj[key].strip():
            raise LLMInvestigatorError(f"{key} must be non-empty text")
    if obj["confidence"] not in ("high", "moderate", "low"):
        raise LLMInvestigatorError(f"invalid confidence: {obj['confidence']!r}")
    valid = set(dataclasses_fields(InvestigationCase).__dict__
                if False else [f.name for f in dataclasses_fields(
                    InvestigationCase)])
    ev = obj["evidence_used"]
    if not isinstance(ev, list) or any(not isinstance(e, str) for e in ev):
        raise LLMInvestigatorError("evidence_used must be list[str]")
    bad = [e for e in ev if e not in valid]
    if bad:
        raise LLMInvestigatorError(f"unknown evidence fields: {bad}")
    w = obj["warnings"]
    if not isinstance(w, list) or any(not isinstance(x, str) for x in w):
        raise LLMInvestigatorError("warnings must be list[str]")
    # Model-supplied identity fields are deliberately NOT read at all.
    return {"finding": obj["finding"], "explanation": obj["explanation"],
            "recommended_action": obj["recommended_action"],
            "confidence": Confidence(obj["confidence"]),
            "evidence_used": tuple(dict.fromkeys(ev)),
            "warnings": tuple(w)}


_RISK_MAP = {Priority.CRITICAL: RiskLevel.CRITICAL,
             Priority.HIGH: RiskLevel.HIGH,
             Priority.MEDIUM: RiskLevel.MEDIUM,
             Priority.LOW: RiskLevel.LOW}


def llm_investigate_case(case: InvestigationCase,
                         client: LLMClient) -> InvestigationAssessment:
    """Interpret one case via an LLM. Deterministic truth preserved."""
    prompt = (SYSTEM_PROMPT + "\n\nInvestigationCase:\n"
              + _serialize_case(case))
    try:
        raw = client.generate(prompt)
    except LLMInvestigatorError:
        raise
    except Exception as e:               # provider failure / timeout etc.
        raise LLMInvestigatorError(f"LLM client failure: {e!r}") from e
    parsed = _validate(raw, case)
    return InvestigationAssessment(
        case_id=case.case_id,                       # from case, not model
        exception_type=case.exception_type.value,   # from case, not model
        risk_level=_RISK_MAP[case.priority],        # from case, not model
        **parsed)


def llm_investigate_cases(cases: list[InvestigationCase],
                          client: LLMClient) -> list[InvestigationAssessment]:
    """Batch. Order-preserving, non-mutating, one assessment per case."""
    return [llm_investigate_case(c, client) for c in cases]
