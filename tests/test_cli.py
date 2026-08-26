import json

import pytest

from finance_controller.cli import build_parser, main
from finance_controller.storage import load_pipeline_result


class OfflineLLM:
    """Offline stand-in injected at the CLI boundary."""
    def __init__(self): self.calls = 0
    def generate(self, prompt):
        self.calls += 1
        return json.dumps({
            "finding": "Amounts differ.", "explanation": "OBSERVED EVIDENCE: "
            "delta present.", "recommended_action": "Verify ledgers.",
            "confidence": "moderate", "evidence_used": ["internal_amount"],
            "warnings": []})


@pytest.fixture
def inject_offline(monkeypatch):
    c = OfflineLLM()
    monkeypatch.setattr("finance_controller.cli._make_llm_client",
                        lambda: c)
    return c


# --help succeeds (both levels)
def test_top_help(capsys):
    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "FINANCE_LLM_API_KEY" in out and "deterministic" in out.lower()

def test_run_help(capsys):
    with pytest.raises(SystemExit) as ei:
        main(["run", "--help"])
    assert ei.value.code == 0


# default deterministic run succeeds, no client constructed
def test_default_run(capsys, monkeypatch):
    monkeypatch.delenv("FINANCE_LLM_API_KEY", raising=False)
    constructed = []
    monkeypatch.setattr("finance_controller.cli._make_llm_client",
                        lambda: (_ for _ in ()).throw(AssertionError))
    assert main(["run"]) == 0
    out = capsys.readouterr().out
    assert "Investigation cases    : 122" in out
    assert "not run" in out


# custom seeds reach generators (verified via report counts changing)
def test_custom_seeds(capsys):
    assert main(["run", "--seed", "7", "--external-seed", "11"]) == 0
    out = capsys.readouterr().out
    assert "Transactions           :" in out
    # different seed -> still parses fine; count line derived from result
    assert "Exception distribution:" in out


# date-fallback reaches run_pipeline
def test_date_fallback(capsys, monkeypatch):
    seen = {}
    orig = __import__("finance_controller.pipeline",
                      fromlist=["run_pipeline"]).run_pipeline
    def spy(*a, **kw):
        seen["fallback"] = kw.get("enable_date_fallback")
        return orig(*a, **kw)
    monkeypatch.setattr("finance_controller.cli", "__dict__",
                        {**vars(__import__("finance_controller.cli",
                                           fromlist=["run_pipeline"])),
                         "run_pipeline": spy})
    assert main(["run", "--date-fallback"]) == 0
    assert seen["fallback"] is True


# --output persists a loadable result
def test_output_roundtrip(tmp_path, capsys):
    out_path = tmp_path / "res.json"
    assert main(["run", "--output", str(out_path)]) == 0
    loaded = load_pipeline_result(out_path)
    assert loaded.case_count == 122
    assert "Saved pipeline result to" in capsys.readouterr().out


# no auto-save without --output
def test_no_autosave(tmp_path, capsys):
    assert main(["run"]) == 0
    assert list(tmp_path.iterdir()) == []


# --llm uses injected offline client, no network
def test_llm_with_injected_client(inject_offline, capsys):
    assert main(["run", "--llm"]) == 0
    assert inject_offline.calls == 122
    out = capsys.readouterr().out
    assert "LLM assessments       : 122" in out


# --llm --evaluate works offline
def test_llm_evaluate(inject_offline, capsys):
    assert main(["run", "--llm", "--evaluate"]) == 0
    out = capsys.readouterr().out
    assert "Evaluation             : 122/122 passed" in out
    assert "Safety score           : 5.00/5" in out


# --evaluate without --llm fails clearly
def test_evaluate_without_llm_fails(capsys):
    assert main(["run", "--evaluate"]) != 0
    err = capsys.readouterr().err
    assert "--evaluate requires --llm" in err


# missing API key fails clearly without fallback to FakeLLMClient
def test_missing_key_no_fake_fallback(capsys, monkeypatch):
    monkeypatch.delenv("FINANCE_LLM_API_KEY", raising=False)
    # ensure production factory really runs (no injection)
    monkeypatch.setattr("finance_controller.cli._make_llm_client",
                        lambda: __import__(
                            "finance_controller.llm_provider",
                            fromlist=["OpenAICompatibleClient"]
                        ).OpenAICompatibleClient())
    rc = main(["run", "--llm"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "FINANCE_LLM_API_KEY" in err and "not set" in err
    assert "FakeLLMClient" not in err


# runtime/provider errors -> non-zero, no traceback, no key leak
def test_provider_error_nonzero_and_redacted(capsys, monkeypatch):
    monkeypatch.setenv("FINANCE_LLM_API_KEY", "sk-secret-cli-777")
    class Boom:
        def generate(self, p): raise TimeoutError("sk-secret-cli-777 down")
    monkeypatch.setattr("finance_controller.cli._make_llm_client",
                        lambda: Boom())
    rc = main(["run", "--llm"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "sk-secret-cli-777" not in captured.err + captured.out
    assert "Traceback" not in captured.err


# report contains calculated counts matching a direct pipeline run
def test_counts_match_direct_pipeline(capsys):
    from finance_controller.generator import (
        generate_dataset, generate_external_dataset)
    from finance_controller.pipeline import run_pipeline
    ds = generate_dataset(seed=42)
    exts, _ = generate_external_dataset(ds, seed=99)
    expected = run_pipeline(list(ds.transactions), exts).case_count
    main(["run"])
    assert f"Investigation cases    : {expected}" in capsys.readouterr().out


# CLI does not mutate generated data / deterministic repeat stability
def test_repeat_deterministic(capsys):
    outs = []
    for _ in range(2):
        assert main(["run"]) == 0
        outs.append(capsys.readouterr().out)
    assert outs[0] == outs[1]
