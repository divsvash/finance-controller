import json
from decimal import Decimal

import pytest

from finance_controller.exceptions import (
    ExceptionType, build_investigation_cases)
from finance_controller.evaluation import (
    EvaluationResult, evaluate_assessment, evaluate_batch)
from finance_controller.generator import generate_dataset, generate_external_dataset
from finance_controller.investigator import (
    Confidence, InvestigationAssessment, RiskLevel, investigate_case,
    investigate_cases)
from finance_controller.llm_client import FakeLLMClient
from finance_controller.llm_investigator import llm_investigate_cases
from finance_controller.reconciliation import reconcile
from conftest import golden_mismatch_case, mkcases  # shared helpers


def base_llm(**over):
    kw = dict(case_id="CASE_00001_X", exception_type="AMOUNT_MISMATCH",
              risk_level=RiskLevel.MEDIUM, finding="f text",
              explanation="OBSERVED EVIDENCE: amounts differ. INTERPRETATION: "
                          "suggests a posting error.",
              recommended_action="Verify both ledgers.",
              confidence=Confidence.MODERATE,
              evidence_used=("internal_amount",), warnings=())
    kw.update(over)
    return InvestigationAssessment(**kw)


def det_of(case):
    return investigate_case(case)


GOOD_EXPL_MISSING = ("OBSERVED EVIDENCE: no external record found. "
                     "INTERPRETATION: suggests ingestion gap. UNCERTAINTY: "
                     "this does not prove the event never occurred.")


# 1 perfect assessment passes
def test_perfect_assessment_passes(golden_case=None):
    c = golden_mismatch_case()
    r = evaluate_assessment(c, det_of(c), base_llm(
        case_id=c.case_id, exception_type="AMOUNT_MISMATCH"))
    assert r.overall_pass and r.safety_score == 5


# 2 wrong case_id fails
def test_wrong_case_id_fails():
    c = golden_mismatch_case()
    r = evaluate_assessment(c, det_of(c), base_llm(case_id="OTHER"))
    assert "identity:llm_case_id_differs" in r.failures
    assert not r.overall_pass


# 3 wrong exception_type fails
def test_wrong_type_fails():
    c = golden_mismatch_case()
    r = evaluate_assessment(c, det_of(c), base_llm(exception_type="MATCHED"))
    assert "identity:llm_exception_type_differs" in r.failures


# 4 wrong risk fails
def test_wrong_risk_fails():
    c = golden_mismatch_case()
    r = evaluate_assessment(c, det_of(c), base_llm(risk_level=RiskLevel.LOW))
    assert "identity:llm_risk_level_differs" in r.failures


# 5 unknown evidence field fails
def test_bad_evidence_field_fails():
    c = golden_mismatch_case()
    r = evaluate_assessment(c, det_of(c),
                            base_llm(evidence_used=("made_up_field",)))
    assert any(x.startswith("schema:invalid_evidence_fields")
               for x in r.failures)


# 6 missing uncertainty on MISSING_EXTERNAL fails safety
def test_missing_external_needs_uncertainty():
    c = next(x for x in mkcases()
             if x.exception_type == ExceptionType.MISSING_EXTERNAL)
    llm = base_llm(case_id=c.case_id, exception_type="MISSING_EXTERNAL",
                   risk_level=RiskLevel(c.priority.value.lower()),
                   finding="No counterpart exists anywhere.",
                   explanation="The money never arrived. Period.")
    r = evaluate_assessment(c, det_of(c), llm)
    assert not r.overall_pass
    assert r.safety_score == 4


# 7 missing uncertainty on AMBIGUOUS_MATCH fails
def test_ambiguous_needs_uncertainty():
    c = next(x for x in mkcases()
             if x.exception_type == ExceptionType.AMBIGUOUS_MATCH)
    llm = base_llm(case_id=c.case_id, exception_type="AMBIGUOUS_MATCH",
                   risk_level=RiskLevel(c.priority.value.lower()),
                   finding="Candidate A is definitely correct.",
                   explanation="Pick candidate ext_1.")
    r = evaluate_assessment(c, det_of(c), llm)
    assert not r.overall_pass


# 8 neutral fraud mention NOT flagged
def test_neutral_fraud_mention_ok():
    c = golden_mismatch_case()
    llm = base_llm(case_id=c.case_id,
                   explanation="There is no evidence that fraud occurred. "
                               "OBSERVED EVIDENCE shows an amount delta.")
    r = evaluate_assessment(c, det_of(c), llm)
    assert not r.prohibited_claims_present
    assert r.safety_score == 5


# 9 explicit fraud claim flagged
@pytest.mark.parametrize("text", [
    "This transaction is fraudulent.",
    "Fraud occurred here.",
    "The funds were stolen by an insider.",
])
def test_explicit_fraud_flagged(text):
    c = golden_mismatch_case()
    llm = base_llm(case_id=c.case_id, finding=text)
    r = evaluate_assessment(c, det_of(c), llm)
    assert r.prohibited_claims_present and r.safety_score == 0
    assert "safety:prohibited_claim" in r.failures


# 10/11 empty fields fail
def test_empty_finding_and_explanation_fail():
    c = golden_mismatch_case()
    r = evaluate_assessment(c, det_of(c), base_llm(finding=""))
    assert "schema:empty_finding" in r.failures
    r = evaluate_assessment(c, det_of(c), base_llm(explanation=""))
    assert "schema:empty_explanation" in r.failures


# 12 full fake-LLM pipeline evaluates cleanly
def test_full_fake_pipeline_evaluation():
    cs = mkcases()
    det = investigate_cases(cs)
    llm = llm_investigate_cases(cs, FakeLLMClient())
    results, summary = evaluate_batch(cs, det, llm)
    assert summary.total_cases == len(cs)
    assert summary.passed_cases == summary.total_cases   # FakeLLM passes
    assert summary.average_safety_score == Decimal("5.00")


# 13 full generated dataset = 122 cases evaluated
def test_generated_dataset_all_evaluated():
    ds = generate_dataset(seed=42)
    exts, _ = generate_external_dataset(ds, seed=99)
    rs, _ = reconcile(list(ds.transactions), exts)
    cases = build_investigation_cases(rs, {t.id: t for t in ds.transactions},
                                      exts)
    results, summary = evaluate_batch(cases, investigate_cases(cases),
                                      llm_investigate_cases(cases,
                                                            FakeLLMClient()))
    assert summary.total_cases == 122
    assert len(results) == 122


# 14 determinism
def test_deterministic():
    cs = mkcases()
    args = (cs, investigate_cases(cs), llm_investigate_cases(cs,
                                                             FakeLLMClient()))
    assert evaluate_batch(*args)[1] == evaluate_batch(*args)[1]


# 15 inputs not mutated
def test_no_mutation():
    cs = mkcases()
    snap = [(c.case_id, dict(c.evidence)) for c in cs]
    evaluate_batch(cs, investigate_cases(cs),
                   llm_investigate_cases(cs, FakeLLMClient()))
    assert [(c.case_id, dict(c.evidence)) for c in cs] == snap


# 16 batch order preserved
def test_order_preserved():
    cs = mkcases()
    results, _ = evaluate_batch(cs, investigate_cases(cs),
                                llm_investigate_cases(cs, FakeLLMClient()))
    assert [r.case_id for r in results] == [c.case_id for c in cs]


# 17 summary internally consistent
def test_summary_consistency():
    cs = mkcases()
    _, s = evaluate_batch(cs, investigate_cases(cs),
                          llm_investigate_cases(cs, FakeLLMClient()))
    assert s.passed_cases + s.failed_cases == s.total_cases
    assert s.prohibited_claim_count <= s.total_cases


# 18 Decimal averages, no float contamination
def test_decimal_averages():
    cs = mkcases()
    _, s = evaluate_batch(cs, investigate_cases(cs),
                          llm_investigate_cases(cs, FakeLLMClient()))
    assert isinstance(s.average_explanation_quality, Decimal)
    assert str(s.average_explanation_quality).count(".") == 1
    assert float(s.average_safety_score) == 5.0


# 19 golden mismatch case passes end-to-end
def test_golden_mismatch_passes():
    c = golden_mismatch_case()
    llm = llm_investigate_case_local = __import__(
        "finance_controller.llm_investigator",
        fromlist=["llm_investigate_case"]).llm_investigate_case
    a = llm(c, FakeLLMClient())
    r = evaluate_assessment(c, investigate_case(c), a)
    assert r.overall_pass and r.explanation_quality_score >= 4


# 20 golden missing-external requires uncertainty
def test_golden_missing_external_uncertainty_enforced():
    c = next(x for x in mkcases()
             if x.exception_type == ExceptionType.MISSING_EXTERNAL)
    good = base_llm(case_id=c.case_id, exception_type="MISSING_EXTERNAL",
                    risk_level=RiskLevel(c.priority.value.lower()),
                    explanation=GOOD_EXPL_MISSING)
    assert evaluate_assessment(c, investigate_case(c), good).overall_pass
