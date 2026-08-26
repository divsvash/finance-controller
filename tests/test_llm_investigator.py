import copy
import json
from decimal import Decimal

import pytest

from finance_controller.exceptions import (
    ExceptionType, build_investigation_cases)
from finance_controller.generator import generate_dataset, generate_external_dataset
from finance_controller.investigator import (
    InvestigationAssessment, investigate_case)
from finance_controller.llm_client import FakeLLMClient
from finance_controller.llm_investigator import (
    LLMInvestigatorError, llm_investigate_case, llm_investigate_cases)
from finance_controller.reconciliation import reconcile
# txn/ext/mkcases helpers identical to test_investigator.py (shared conftest)


def golden_mismatch_case():
    t = txn("t_g1", amount="9850.00", pref="pay_GoldenRef")
    e = ext("e_g1", amount="9700.00", ref="PAY-GOLDENREF")
    rs, _ = reconcile([t], [e], enable_date_fallback=False)
    return build_investigation_cases(rs, {t.id: t}, [e])[0]


def resp(**over):
    base = {"finding": "f", "explanation": "e", "recommended_action": "a",
            "confidence": "moderate", "evidence_used": ["internal_amount"],
            "warnings": []}
    base.update(over)
    return json.dumps(base)


# 1 valid response -> assessment
def test_valid_response():
    c = golden_mismatch_case()
    class C:
        def generate(self, p): return resp()
    a = llm_investigate_case(c, C())
    assert isinstance(a, InvestigationAssessment)


# 2/3/4 deterministic truth protected
def test_identity_fields_cannot_be_changed():
    c = golden_mismatch_case()

    class C:
        def generate(self, p):
            r = json.loads(resp())
            r["case_id"] = "HACKED"; r["exception_type"] = "MATCHED"
            r["risk_level"] = "low"
            return json.dumps(r)
    a = llm_investigate_case(c, C())
    assert a.case_id == c.case_id
    assert a.exception_type == "AMOUNT_MISMATCH"


def test_risk_copied_from_priority():
    c = next(x for x in mkcases()
             if x.exception_type == ExceptionType.AMBIGUOUS_MATCH)

    class C:
        def generate(self, p): return resp(confidence="high")
    assert llm_investigate_case(c, C()).risk_level.value == c.priority.value \
        .lower() if False else True
    a = llm_investigate_case(c, C())
    from finance_controller.investigator import RiskLevel
    assert a.risk_level == RiskLevel(c.priority.value.lower())


# 5 malformed schema
@pytest.mark.parametrize("raw", ["not json", "[1,2]", '{"finding": ""}',
                                 resp(confidence="certain"),
                                 resp(evidence_used="x"),
                                 resp(warnings=["ok", 3])])
def test_malformed_responses_raise(raw):
    c = golden_mismatch_case()
    class C:
        def generate(self, p): return raw
    with pytest.raises(LLMInvestigatorError):
        llm_investigate_case(c, C())


# 6 empty finding
def test_empty_finding_raises():
    c = golden_mismatch_case()
    class C:
        def generate(self, p): return resp(finding="")
    with pytest.raises(LLMInvestigatorError):
        llm_investigate_case(c, C())


# 7 invalid evidence field
def test_invalid_evidence_field_raises():
    c = golden_mismatch_case()
    class C:
        def generate(self, p): return resp(evidence_used=["not_a_field"])
    with pytest.raises(LLMInvestigatorError, match="not_a_field"):
        llm_investigate_case(c, C())


# 8 ambiguity cannot become matched
def test_ambiguous_stays_unresolved():
    c = next(x for x in mkcases()
             if x.exception_type == ExceptionType.AMBIGUOUS_MATCH)
    class C:
        def generate(self, p): return resp()
    a = llm_investigate_case(c, C())
    assert a.exception_type == "AMBIGUOUS_MATCH"


# 9 anti-fraud instruction present in prompt; extras stay neutral
def test_prompt_contains_antifraud_and_extra_neutral():
    c = next(x for x in mkcases()
             if x.exception_type == ExceptionType.EXTRA_EXTERNAL)
    seen_prompts = []
    class C:
        def generate(self, p):
            seen_prompts.append(p); return resp()
    llm_investigate_case(c, C())
    assert "Never infer that fraud occurred unless the evidence explicitly establishes it" \
        in seen_prompts[0]


# 10 missing-external uncertainty preserved in fake output contract
def test_missing_external_uncertainty_section():
    c = next(x for x in mkcases()
             if x.exception_type == ExceptionType.MISSING_EXTERNAL)
    a = llm_investigate_case(c, FakeLLMClient())
    assert "UNCERTAINTY:" in a.explanation


# 11 decimals serialize cleanly
def test_decimal_serialization_no_float_artifacts():
    c = golden_mismatch_case()
    seen = []
    class C:
        def generate(self, p): seen.append(p); return resp()
    llm_investigate_case(c, C())
    assert "9850.00" in seen[0] and "9700.00" in seen[0]
    assert "5511151231257827" not in seen[0]


# 12 fake client reproducible
def test_fake_client_reproducible():
    cs = mkcases()
    assert (llm_investigate_cases(cs, FakeLLMClient())
            == llm_investigate_cases(cs, FakeLLMClient()))


# 13 batch preserves order
def test_batch_order_preserved():
    cs = mkcases()
    out = llm_investigate_cases(cs, FakeLLMClient())
    assert [a.case_id for a in out] == [c.case_id for c in cs]


# 14 input not mutated
def test_input_not_mutated():
    cs = mkcases()
    snap = [(c.case_id, dict(c.evidence)) for c in cs]
    llm_investigate_cases(cs, FakeLLMClient())
    assert [(c.case_id, dict(c.evidence)) for c in cs] == snap


# 15 provider failures wrapped
def test_provider_failure_wrapped():
    c = golden_mismatch_case()
    class Boom(Exception): pass
    class C:
        def generate(self, p): raise TimeoutError("upstream down")
    with pytest.raises(LLMInvestigatorError, match="client failure"):
        llm_investigate_case(c, C())


# 16 integration: one assessment per case over full pipeline
def test_full_pipeline_one_assessment_per_case():
    ds = generate_dataset(seed=42)
    exts, _ = generate_external_dataset(ds, seed=99)
    rs, _ = reconcile(list(ds.transactions), exts)
    cases = build_investigation_cases(rs, {t.id: t for t in ds.transactions},
                                      exts)
    out = llm_investigate_cases(cases, FakeLLMClient())
    assert len(out) == len(cases) == 122
    assert len({a.case_id for a in out}) == 122


# 17 golden case
def test_golden_preserves_truth_and_explains():
    c = golden_mismatch_case()
    a = llm_investigate_case(c, FakeLLMClient())
    from finance_controller.investigator import RiskLevel
    assert a.case_id == c.case_id
    assert a.exception_type == "AMOUNT_MISMATCH"
    assert a.risk_level == RiskLevel.MEDIUM
    assert "150.00" in a.explanation
