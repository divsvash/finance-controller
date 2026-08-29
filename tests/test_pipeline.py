import pytest

from conftest import mkcases  # existing shared helpers
from finance_controller.exceptions import build_investigation_cases
from finance_controller.generator import generate_dataset, generate_external_dataset
from finance_controller.llm_client import FakeLLMClient
from finance_controller.llm_investigator import LLMInvestigatorError
from finance_controller.pipeline import PipelineResult, run_pipeline


def full_cases():
    ds = generate_dataset(seed=42)
    exts, _ = generate_external_dataset(ds, seed=99)
    return ds, exts


def run_full(**kw):
    ds, exts = full_cases()
    return ds, run_pipeline(list(ds.transactions), exts, **kw)


# 1/2 deterministic-only works with no API key
def test_deterministic_only(monkeypatch):
    monkeypatch.delenv("FINANCE_LLM_API_KEY", raising=False)
    _, r = run_full()
    assert r.llm_assessments is None and r.evaluation_summary is None
    assert len(r.deterministic_assessments) == len(r.investigation_cases)


# 3/4 generated full dataset, 122 cases, one assessment each
def test_full_122_case_pipeline():
    _, r = run_full()
    assert r.case_count == 122
    assert len(r.deterministic_assessments) == 122
    assert len({a.case_id for a in r.deterministic_assessments}) == 122
    assert all(a.case_id == c.case_id for a, c in
               zip(r.deterministic_assessments, r.investigation_cases))


# 5 fake-llm pipeline
def test_fake_llm_pipeline():
    _, r = run_full(run_llm=True, llm_client=FakeLLMClient())
    assert len(r.llm_assessments) == 122
    assert r.evaluation_summary is None


# 6 llm + evaluation
def test_llm_and_evaluation():
    _, r = run_full(run_llm=True, run_evaluation=True,
                    llm_client=FakeLLMClient())
    assert len(r.evaluation_results) == 122
    assert r.evaluation_summary.total_cases == 122
    assert r.evaluation_summary.passed_cases == 122


# 7 llm without evaluation -> evaluation fields None
def test_llm_without_evaluation():
    _, r = run_full(run_llm=True, llm_client=FakeLLMClient())
    assert r.evaluation_results is None and r.evaluation_summary is None


# 8 evaluation without llm rejected clearly
def test_evaluation_requires_llm():
    with pytest.raises(ValueError, match="requires"):
        _, r = full_cases()
        run_pipeline(list(r[0].transactions), r[1], run_evaluation=True)


def test_run_llm_requires_client():
    ds, exts = full_cases()
    with pytest.raises(ValueError, match="injected llm_client"):
        run_pipeline(list(ds.transactions), exts, run_llm=True)


# 9/10 order + one-per-case
def test_order_and_cardinality():
    _, r = run_full(run_llm=True, run_evaluation=True,
                    llm_client=FakeLLMClient())
    ids = [c.case_id for c in r.investigation_cases]
    assert [a.case_id for a in r.deterministic_assessments] == ids
    assert [a.case_id for a in r.llm_assessments] == ids
    assert [e.case_id for e in r.evaluation_results] == ids


# 11 inputs not mutated
def test_inputs_not_mutated():
    cs = mkcases()
    snap_t, snap_e = [(c.case_id, dict(c.evidence)) for c in cs], []
    ds, exts = full_cases()
    t_ids = [(t.id, str(t.amount)) for t in ds.transactions]
    e_ids = [(e.id, str(e.amount)) for e in exts]
    run_pipeline(list(ds.transactions), exts, run_llm=True,
                 run_evaluation=True, llm_client=FakeLLMClient())
    assert [(t.id, str(t.amount)) for t in ds.transactions] == t_ids
    assert [(e.id, str(e.amount)) for e in exts] == e_ids


# 12 repeated runs deterministic
def test_deterministic_repeat():
    _, a = run_full(run_llm=True, run_evaluation=True,
                    llm_client=FakeLLMClient())
    _, b = run_full(run_llm=True, run_evaluation=True,
                    llm_client=FakeLLMClient())
    assert a.deterministic_assessments == b.deterministic_assessments
    assert a.llm_assessments == b.llm_assessments
    assert a.evaluation_summary == b.evaluation_summary


# 13 injected client actually used
def test_injected_client_used():
    calls = []
    # simpler inline spy returning valid schema JSON:
    import json
    class Spy2:
        def generate(self, p):
            calls.append(p)
            return json.dumps({"finding": "f", "explanation": "OBSERVED EVIDENCE: x.",
                               "recommended_action": "review",
                               "confidence": "high",
                               "evidence_used": ["internal_amount"],
                               "warnings": []})
    _, r = full_cases()
    out = run_pipeline(list(_[0].transactions), _[1], run_llm=True,
                       llm_client=Spy2())
    assert len(calls) == 122
    assert out.llm_assessments[0].finding == "f"


# 14 provider errors propagate untouched
def test_provider_error_propagates():
    class Boom:
        def generate(self, p): raise TimeoutError("down")
    ds, exts = full_cases()
    with pytest.raises(LLMInvestigatorError):
        run_pipeline(list(ds.transactions), exts, run_llm=True,
                     llm_client=Boom())


# 15 deterministic truth preserved through LLM stage
def test_truth_preserved():
    _, r = run_full(run_llm=True, llm_client=FakeLLMClient())
    for c, d, l in zip(r.investigation_cases, r.deterministic_assessments,
                       r.llm_assessments):
        assert l.exception_type == c.exception_type.value == d.exception_type
        assert l.risk_level == d.risk_level
        assert l.case_id == c.case_id


# 16 empty dataset
def test_empty_dataset():
    r = run_pipeline([], [])
    assert r.reconciliation_results == () and r.case_count == 0
    assert r.deterministic_assessments == ()
    assert isinstance(r, PipelineResult)


# 17 result structure / optionality
def test_result_structure():
    _, r = run_full()
    assert r.llm_assessments is None
    _, r2 = run_full(run_llm=True, llm_client=FakeLLMClient())
    assert r2.llm_assessments is not None and r2.evaluation_results is None

