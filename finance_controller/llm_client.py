"""Provider abstraction for the LLM investigator."""
from __future__ import annotations

import json


class FakeLLMClient:
    """Deterministic offline stand-in: echoes structured, schema-valid
    assessments derived purely from the serialized case in the prompt.
    Reproducible; no network."""

    def generate(self, prompt: str) -> str:
        # Extract the JSON case block appended after our system prompt.
        blob = prompt.split("InvestigationCase:\n", 1)[1]
        case = json.loads(blob)
        et = case["exception_type"]
        diff = case["amount_difference"]
        finding = {
            "AMOUNT_MISMATCH":
                f"Amount discrepancy of ₹{diff} between internal and external records.",
            "AMBIGUOUS_MATCH":
                f"{len(case['candidate_external_ids'])} indistinguishable external candidates.",
            "MISSING_EXTERNAL":
                "No compatible external counterpart located for this internal transaction.",
            "EXTRA_EXTERNAL":
                "External record was not consumed by reconciliation.",
            "UNRESOLVED_MATCH":
                "Reconciliation could not establish a valid match.",
        }[et]
        explanation = (
            f"OBSERVED EVIDENCE: internal_amount={case['internal_amount']}, "
            f"external_amounts={case['external_amounts']}, "
            f"payment_ref={case['payment_ref']!r}. "
            f"INTERPRETATION: the deterministic layer classified this as {et}; "
            f"the discrepancy magnitude is the key fact for human review. "
            f"UNCERTAINTY: which side of the ledger is correct cannot be "
            f"determined from this evidence alone.")
        action = {
            "AMOUNT_MISMATCH": "Verify both ledgers against the payment-gateway source record.",
            "AMBIGUOUS_MATCH": "Manually inspect each candidate external record.",
            "MISSING_EXTERNAL": "Query the provider directly using the payment reference; check ingestion logs.",
            "EXTRA_EXTERNAL": "Determine whether this is untracked revenue, a duplicate, delayed ingestion, or unrelated.",
            "UNRESOLVED_MATCH": "Manual investigation of raw records on both sides required.",
        }[et]
        return json.dumps({
            "finding": finding, "explanation": explanation,
            "recommended_action": action,
            "confidence": "high",
            "evidence_used": ["internal_amount", "external_amounts",
                              "amount_difference", "payment_ref",
                              "candidate_external_ids"][:4],
            "warnings": [],
        })
