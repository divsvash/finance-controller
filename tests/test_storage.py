import json
import tempfile
from pathlib import Path

import pytest

from conftest import mkcases
from finance_controller.generator import generate_dataset, generate_external_dataset
from finance_controller.llm_client import FakeLLMClient
from finance_controller.pipeline import run_pipeline
from finance_controller.storage import SCHEMA_VERSION, StorageError, \
    load_pipeline_result, save_pipeline_result


def full_result(**kw):
    ds = generate_dataset(seed=42)
    exts, _ = generate_external_dataset(ds, seed=99)
    return run_pipeline(list(ds.transactions), exts, **kw)


def roundtrip(tmp_path, r):
    p = tmp_path / "out.json"
    save_pipeline_result(r, p)
    return p, load_pipeline_result(p)


# 1 deterministic-only round trip
def test_det_only_roundtrip(tmp_path):
    p, back = roundtrip(tmp_path, full_result())
    assert len(back.deterministic_assessments) == 122


# 2 full LLM + evaluation round trip
def test_full_roundtrip(tmp_path):
    _, back = roundtrip(tmp_path, full_result(run_llm=True,
                                              run_evaluation=True,
                                              llm_client=FakeLLMClient()))
    s = back.evaluation_summary
    assert s.total_cases == 122 and s.passed_cases == 122
    assert back.evaluation_summary.average_safety_score == \
        __import__("decimal").Decimal("5.00")


# 3 None optionals survive
def test_none_optionals_survive(tmp_path):
    _, back = roundtrip(tmp_path, full_result())
    assert back.llm_assessments is None
    assert back.evaluation_results is None
    assert back.evaluation_summary is None


# 4 semantic equality of important fields
def test_semantic_equality(tmp_path):
    r = full_result(run_llm=True, llm_client=FakeLLMClient())
    _, b = roundtrip(tmp_path, r)
    assert [a.case_id for a in b.deterministic_assessments] == \
           [a.case_id for a in r.deterministic_assessments]
    assert [a.finding for a in b.llm_assessments] == \
           [a.finding for a in r.llm_assessments]
    assert [a.explanation for a in b.llm_assessments] == \
           [a.explanation for a in r.llm_assessments]
    assert [str(a.risk_level.value) for a in b.llm_assessments] == \
           [str(a.risk_level.value) for a in r.llm_assessments]


# 5 human-readable + schema version present
def test_readable_and_schema_version(tmp_path):
    ds = generate_dataset(seed=42)
    exts, _ = generate_external_dataset(ds, seed=99)
    r = run_pipeline(list(ds.transactions)[:10], exts[:10])
    p = tmp_path / "x.json"
    save_pipeline_result(r, p)
    text = p.read_text(encoding="utf-8")
    doc = json.loads(text)
    assert doc["schema_version"] == SCHEMA_VERSION
    assert "\n" in text and text.startswith("{")   # indented JSON
    assert '"finding"' in text                      # inspectable


# 6 missing file fails clearly
def test_missing_file(tmp_path):
    with pytest.raises(StorageError, match="not found"):
        load_pipeline_result(tmp_path / "nope.json")


# 7 malformed JSON fails clearly
def test_malformed_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(StorageError, match="malformed"):
        load_pipeline_result(p)


# 8 missing top-level field fails
def test_missing_top_field(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(StorageError, match="pipeline_result"):
        load_pipeline_result(p)


# 9 unsupported schema version fails
def test_bad_schema_version(tmp_path):
    p = tmp_path / "v.json"
    p.write_text(json.dumps({"schema_version": 99,
                             "pipeline_result": {}}), encoding="utf-8")
    with pytest.raises(StorageError, match="unsupported schema_version"):
        load_pipeline_result(p)


# 10 corrupt nested data fails
@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema_version=2),
    lambda d: d["pipeline_result"].update(__type__="bogus"),
    lambda d: d["pipeline_result"]["fields"]["deterministic_assessments"]
              ["items"][0]["fields"].pop("case_id"),
])
def test_corrupt_nested_fails(tmp_path, mutate):
    p, _ = roundtrip(tmp_path, full_result())
    doc = json.loads(p.read_text(encoding="utf-8"))
    pr = doc["pipeline_result"]
    if "__type__" not in pr:  # unwrap the PipelineResult tag
        pr = next(v for v in pr.values() if isinstance(v, dict))
        doc["pipeline_result"] = pr
    mutate(doc)
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(StorageError):
        load_pipeline_result(p)


# 11 no secrets in persisted JSON
def test_no_secrets_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCE_LLM_API_KEY", "sk-leaky-key-999")
    r = full_result()   # deliberately no client attached anywhere
    p, back = roundtrip(tmp_path, r)
    text = p.read_text(encoding="utf-8")
    assert "sk-" not in text and "api_key" not in text.lower()


# 12 input PipelineResult not mutated
def test_input_not_mutated(tmp_path):
    r = full_result(run_llm=True, run_evaluation=True,
                    llm_client=FakeLLMClient())
    snap_ids = [c.case_id for c in r.investigation_cases]
    snap_findings = [a.finding for a in r.deterministic_assessments]
    roundtrip(tmp_path, r)
    assert [c.case_id for c in r.investigation_cases] == snap_ids
    assert [a.finding for a in r.deterministic_assessments] == snap_findings


# 13 parent dirs created
def test_parent_dirs_created(tmp_path):
    r = full_result()
    deep = tmp_path / "a" / "b" / "c.json"
    save_pipeline_result(r, deep)
    assert load_pipeline_result(deep).case_count == 122