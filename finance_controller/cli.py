"""Command-line interface for the finance-controller backend.

Interface layer ONLY: it composes generate_dataset ->
generate_external_dataset -> run_pipeline -> optional storage. All
financial logic lives in the existing modules; the deterministic
InvestigationCase remains the sole source of financial truth.

Usage:
    python -m finance_controller.cli run [--seed N] [--external-seed N]
        [--llm] [--evaluate] [--output PATH] [--date-fallback]

Environment variables (LLM mode only):
    FINANCE_LLM_API_KEY   required for --llm
    FINANCE_LLM_MODEL     model name   (default: gpt-4o-mini)
    FINANCE_LLM_BASE_URL  API base URL (default: https://api.openai.com/v1)

No flags = deterministic-only; no API key needed, no LLM client is ever
constructed. --llm never falls back to FakeLLMClient.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation

from .generator import generate_dataset, generate_external_dataset
from .llm_client import FakeLLMClient  # noqa: F401  (injection point only)
from .pipeline import run_pipeline
from .treasury import (
    CashPosition, Certainty, ControllerPolicy, ExpectedFlow,
    FlowCategory, FlowDirection)

PROG = "finance-controller"

DESCRIPTION = """\
Run the finance-controller reconciliation pipeline from the terminal.

Modes:
  deterministic (default) : reconcile + investigate cases. No API key,
                            no network access.
  --llm                   : additionally produce LLM interpretations.
                            Requires FINANCE_LLM_API_KEY. Never falls
                            back to a fake client.
  --llm --evaluate        : also evaluate LLM interpretations against
                            the deterministic baseline.

Persistence:
  --output PATH           : save the full PipelineResult as readable
                            JSON (schema_version=1). Nothing is saved
                            unless --output is given.

The LLM NEVER makes financial decisions; identity/risk/type always come
from the deterministic pipeline.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"python -m {__package__ or 'finance_controller'}.cli",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the pipeline on generated data")
    r.add_argument("--seed", type=int, default=42,
                   help="dataset generator seed (default: 42)")
    r.add_argument("--external-seed", dest="external_seed", type=int,
                   default=99,
                   help="external-record generator seed (default: 99)")
    r.add_argument("--llm", action="store_true",
                   help="enable LLM investigation (requires "
                        "FINANCE_LLM_API_KEY)")
    r.add_argument("--evaluate", action="store_true",
                   help="evaluate LLM interpretations (requires --llm)")
    r.add_argument("--output", metavar="PATH", default=None,
                   help="persist PipelineResult to PATH as JSON")
    r.add_argument("--date-fallback", action="store_true",
                   help="allow date-window fallback during matching")
    return p


def _client_factory():
    """Imported lazily so deterministic mode never touches provider code."""
    from .llm_provider import OpenAICompatibleClient
    return OpenAICompatibleClient


def _make_llm_client():
    """Production path: real provider from env config. Raises
    MissingAPIKeyError clearly if FINANCE_LLM_API_KEY is unset."""
    return _client_factory()()


def execute(args, llm_client=None) -> int:
    """Core command body. `llm_client` exists purely as an injection
    point for offline tests; production callers leave it None."""
    if args.evaluate and not args.llm:
        print("error: --evaluate requires --llm "
              "(there is no LLM assessment to evaluate)", file=sys.stderr)
        return 2

    ds = generate_dataset(seed=args.seed)
    exts, _ = generate_external_dataset(ds, seed=args.external_seed)

    try:
        result = run_pipeline(
            list(ds.transactions), exts,
            run_llm=args.llm,
            run_evaluation=args.evaluate,
            enable_date_fallback=args.date_fallback,
            **({"llm_client": llm_client} if args.llm else {}))
    except Exception as e:  # concise user error; no stack trace by default
        msg = str(e).replace(
            __import__("os").environ.get("FINANCE_LLM_API_KEY", "\x00"),
            "<redacted>")
        print(f"error: {type(e).__name__}: {msg}", file=sys.stderr)
        return 1

    n = result.case_count
    dist = Counter(c.exception_type.value for c in result.investigation_cases)
    print("FINANCE CONTROLLER")
    print("==================")
    print(f"Transactions           : {len(ds.transactions)}")
    print(f"External records       : {len(exts)}")
    print(f"Reconciliation results : {len(result.reconciliation_results)}")
    print(f"Investigation cases    : {n}")
    print(f"Deterministic assess.  : "
          f"{len(result.deterministic_assessments)}")
    print()
    print("Exception distribution:")
    for t in sorted(dist):
        print(f"  {t:<18}{dist[t]}")
    print()
    if result.llm_assessments is not None:
        print(f"LLM assessments       : {len(result.llm_assessments)}")
    else:
        print("LLM assessments       : not run")
    s = result.evaluation_summary
    if s is not None:
        print(f"Evaluation             : {s.passed_cases}/{s.total_cases} "
              f"passed")
        print(f"Explanation quality    : "
              f"{s.average_explanation_quality}/5")
        print(f"Safety score           : {s.average_safety_score}/5")
    else:
        print("Evaluation             : not run")

    if args.output:
        from .storage import save_pipeline_result
        try:
            save_pipeline_result(result, args.output)
        except Exception as e:
            print(f"error saving output: {e}", file=sys.stderr)
            return 1
        print(f"\nSaved pipeline result to: {args.output}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    llm_client = None
    if getattr(args, "llm", False):
        try:
            llm_client = _make_llm_client()
        except Exception as e:
            key = __import__("os").environ.get("FINANCE_LLM_API_KEY", "")
            msg = str(e).replace(key, "<redacted>") if key else str(e)
            print(f"error: cannot start LLM mode: {msg}", file=sys.stderr)
            print("hint: export FINANCE_LLM_API_KEY='sk-...' before using "
                  "--llm", file=sys.stderr)
            return 2
    return execute(args, llm_client=llm_client)


if __name__ == "__main__":
    sys.exit(main())

TREASURY_KEYS = ("cash_position", "expected_flows", "treasury_policy")

# ---- treasury input loading (parsing only — NO calculation logic here;
#      all arithmetic stays in treasury.py) ----

def _cli_error(msg: str) -> "NoReturn":
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def _t_dec(value, name):
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(f"{name}: monetary values must be strings or "
                         f"integers, never floats/booleans")
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        raise ValueError(f"{name}: not a Decimal-compatible value")


def _t_date(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name}: must be an ISO YYYY-MM-DD string")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name}: invalid ISO date {value!r}")


def _t_enum(enum_cls, value, name):
    try:
        return enum_cls(value)
    except ValueError:
        raise ValueError(f"{name}: invalid {enum_cls.__name__} {value!r}")


def _load_treasury_inputs(path_str):
    """Returns (cash_position, expected_flows, policy) or (None,)*3.
    All three keys required together; no defaults invented."""
    path = pathlib.Path(path_str)
    if not path.is_file():
        _cli_error(f"treasury input file not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        _cli_error(f"treasury input file is not valid JSON: {e}")
    if not isinstance(data, dict):
        _cli_error("treasury input must be a JSON object")
    present = [k for k in TREASURY_KEYS if k in data]
    missing = [k for k in TREASURY_KEYS if k not in data]
    if present and missing:
        _cli_error(f"partial treasury input: missing {missing} "
                   f"(all of {list(TREASURY_KEYS)} are required together)")
    if not present:
        return None, None, None          # treasury disabled
    try:
        cp = data["cash_position"]
        pos = CashPosition(
            as_of=_t_date(cp["as_of"], "cash_position.as_of"),
            opening_balance=_t_dec(cp["opening_balance"],
                                   "cash_position.opening_balance"),
            cleared_inflows=_t_dec(cp["cleared_inflows"],
                                   "cash_position.cleared_inflows"),
            cleared_outflows=_t_dec(cp["cleared_outflows"],
                                    "cash_position.cleared_outflows"))
        flows = []
        for i, f in enumerate(data["expected_flows"]):
            flows.append(ExpectedFlow(
                flow_id=str(f["flow_id"]),
                direction=_t_enum(FlowDirection, f["direction"],
                                  f"flow[{i}].direction"),
                amount=_t_dec(f["amount"], f"flow[{i}].amount"),
                expected_date=_t_date(f["expected_date"],
                                      f"flow[{i}].expected_date"),
                category=_t_enum(FlowCategory, f.get("category", "OTHER"),
                                 f"flow[{i}].category"),
                certainty=_t_enum(Certainty, f["certainty"],
                                  f"flow[{i}].certainty"),
                linked_transaction_id=f.get("linked_transaction_id")))
        pol_raw = data["treasury_policy"]
        if not isinstance(pol_raw.get("include_forecast_flows"), bool):
            raise ValueError(
                "treasury_policy.include_forecast_flows must be a boolean")
        pol = ControllerPolicy(
            minimum_cash_reserve=_t_dec(pol_raw["minimum_cash_reserve"],
                                        "policy.minimum_cash_reserve"),
            reserve_buffer_pct=_t_dec(pol_raw["reserve_buffer_pct"],
                                      "policy.reserve_buffer_pct"),
            max_single_movement_pct=_t_dec(pol_raw["max_single_movement_pct"],
                                           "policy.max_single_movement_pct"),
            include_forecast_flows=pol_raw["include_forecast_flows"])
    except KeyError as e:
        _cli_error(f"missing required treasury field: {e}")
    except ValueError as e:              # includes domain __post_init__ rules
        _cli_error(f"invalid treasury input: {e}")
    return pos, flows, pol


def _print_treasury_summary(s):
    # Plain labeled lines, consistent with existing CLI output style.
    # str(Decimal) preserves exact precision — never float().
    print("== treasury ==")
    print(f"current_cash: {s.current_cash}")
    print(f"expected_net: {s.expected_net}")
    print(f"projected_cash: {s.projected_cash}")
    print(f"reserve_requirement: {s.reserve_requirement}")
    print(f"safe_movable_capital: {s.safe_movable_capital}")
    print(f"obligation_breaches_reserve: {s.obligation_breaches_reserve}")
    print(f"insufficiency: {s.insufficiency}")

import copy
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from finance_controller.api import create_app
from finance_controller.controller import (
    ControllerDecision, DecisionType, evaluate_treasury_decision)

TREASURY = {  # identical shapes to tests/test_api_treasury.py fixtures
    "cash_position": {"as_of": "2025-06-30", "opening_balance": "100000",
                      "cleared_inflows": "20000", "cleared_outflows": "5000"},
    "expected_flows": [
        {"flow_id": "f1", "direction": "INFLOW", "amount": "15000",
         "expected_date": "2025-07-10", "category": "RECEIVABLE",
         "certainty": "CONFIRMED"},
        {"flow_id": "f2", "direction": "OUTFLOW", "amount": "8000",
         "expected_date": "2025-07-15", "category": "PAYROLL",
         "certainty": "SCHEDULED"}],
    "treasury_policy": {"minimum_cash_reserve": "30000",
                        "reserve_buffer_pct": "0.10",
                        "max_single_movement_pct": "0.30",
                        "include_forecast_flows": False},
}


@pytest.fixture
def client():
    return TestClient(create_app())


def _run(client, body):
    return client.post("/pipeline/run", json=body)


def test_empty_request_unchanged(client):
    b = _run(client, {}).json()
    assert b["case_count"] == 122
    assert b["treasury_summary"] is None
    assert b["controller_decision"] is None


def test_summary_without_amount(client):
    b = _run(client, TREASURY).json()
    assert isinstance(b["treasury_summary"], dict)
    assert b["controller_decision"] is None


def test_allow_via_string_amount(client):
    body = {**copy.deepcopy(TREASURY), "proposed_amount": "10000"}
    d = _run(client, body).json()["controller_decision"]
    assert d["decision_type"] == "ALLOW"                 # enum as string
    assert isinstance(d["proposed_amount"], str)         # Decimal as string
    assert Decimal(d["proposed_amount"]) == Decimal(10000)


def test_deny_over_governance_cap(client):
    body = {**copy.deepcopy(TREASURY), "proposed_amount": "40000"}
    d = _run(client, body).json()["controller_decision"]
    assert d["decision_type"] == "DENY"
    assert any("governance cap" in r for r in d["reasons"])


def test_deny_above_movable_capital(client):
    body = {**copy.deepcopy(TREASURY),
            "proposed_amount": "99999"}
    d = _run(client, body).json()["controller_decision"]
    assert d["decision_type"] == "DENY"


def test_matches_direct_evaluate(client):
    body = {**copy.deepcopy(TREASURY), "proposed_amount": "10000"}
    resp = _run(client, body).json()
    # reconstruct direct call from API's own summary output
    from datetime import date
    from finance_controller.treasury import (
        CashPosition, ControllerPolicy, TreasurySummary)
    s = TreasurySummary(**resp["treasury_summary"])   # strings -> Decimals? no:
    # committed version builds the domain objects from the literal fixtures
    # (same constructors as test_api_treasury.test_matches_direct_compute)
    direct = evaluate_treasury_decision(s, POLICY_DOMAIN, Decimal(10000))
    assert resp["controller_decision"]["decision_type"] == \
        direct.decision_type.value
    assert Decimal(resp["controller_decision"]["cap_amount"]) == \
        direct.cap_amount


@pytest.mark.parametrize("amount", ["-5", 0.1, True, "NaN", "Infinity"])
def test_invalid_amounts_422(amount, client):
    body = {**copy.deepcopy(TREASURY), "proposed_amount": amount}
    r = _run(client, body)
    assert r.status_code == 422, r.text
    e = r.json()["detail"]["error"]
    assert e["type"] == "invalid_treasury_input"


def test_amount_without_treasury_stays_null(client):
    b = _run(client, {"proposed_amount": "100"}).json()
    assert b["treasury_summary"] is None
    assert b["controller_decision"] is None


def test_partial_treasury_with_amount_still_incomplete_error(client):
    body = {**TREASURY, "proposed_amount": "10000"}
    del body["treasury_policy"]
    r = _run(client, body)
    assert r.status_code == 422
    assert r.json()["detail"]["error"]["type"] == "incomplete_treasury_input"


def test_save_accepts_proposed_amount(client):
    body = {**copy.deepcopy(TREASURY), "proposed_amount": "10000",
            "run_name": "ctrl1"}
    r = client.post("/pipeline/save", json=body)
    assert r.status_code == 200

import copy
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from finance_controller.api import create_app
from finance_controller.controller import (
    ControllerDecision, DecisionType, evaluate_treasury_decision)

TREASURY = {  # identical shapes to tests/test_api_treasury.py fixtures
    "cash_position": {"as_of": "2025-06-30", "opening_balance": "100000",
                      "cleared_inflows": "20000", "cleared_outflows": "5000"},
    "expected_flows": [
        {"flow_id": "f1", "direction": "INFLOW", "amount": "15000",
         "expected_date": "2025-07-10", "category": "RECEIVABLE",
         "certainty": "CONFIRMED"},
        {"flow_id": "f2", "direction": "OUTFLOW", "amount": "8000",
         "expected_date": "2025-07-15", "category": "PAYROLL",
         "certainty": "SCHEDULED"}],
    "treasury_policy": {"minimum_cash_reserve": "30000",
                        "reserve_buffer_pct": "0.10",
                        "max_single_movement_pct": "0.30",
                        "include_forecast_flows": False},
}


@pytest.fixture
def client():
    return TestClient(create_app())


def _run(client, body):
    return client.post("/pipeline/run", json=body)


def test_empty_request_unchanged(client):
    b = _run(client, {}).json()
    assert b["case_count"] == 122
    assert b["treasury_summary"] is None
    assert b["controller_decision"] is None


def test_summary_without_amount(client):
    b = _run(client, TREASURY).json()
    assert isinstance(b["treasury_summary"], dict)
    assert b["controller_decision"] is None


def test_allow_via_string_amount(client):
    body = {**copy.deepcopy(TREASURY), "proposed_amount": "10000"}
    d = _run(client, body).json()["controller_decision"]
    assert d["decision_type"] == "ALLOW"                 # enum as string
    assert isinstance(d["proposed_amount"], str)         # Decimal as string
    assert Decimal(d["proposed_amount"]) == Decimal(10000)


def test_deny_over_governance_cap(client):
    body = {**copy.deepcopy(TREASURY), "proposed_amount": "40000"}
    d = _run(client, body).json()["controller_decision"]
    assert d["decision_type"] == "DENY"
    assert any("governance cap" in r for r in d["reasons"])


def test_deny_above_movable_capital(client):
    body = {**copy.deepcopy(TREASURY),
            "proposed_amount": "99999"}
    d = _run(client, body).json()["controller_decision"]
    assert d["decision_type"] == "DENY"


def test_matches_direct_evaluate(client):
    body = {**copy.deepcopy(TREASURY), "proposed_amount": "10000"}
    resp = _run(client, body).json()
    # reconstruct direct call from API's own summary output
    from datetime import date
    from finance_controller.treasury import (
        CashPosition, ControllerPolicy, TreasurySummary)
    s = TreasurySummary(**resp["treasury_summary"])   # strings -> Decimals? no:
    # committed version builds the domain objects from the literal fixtures
    # (same constructors as test_api_treasury.test_matches_direct_compute)
    direct = evaluate_treasury_decision(s, POLICY_DOMAIN, Decimal(10000))
    assert resp["controller_decision"]["decision_type"] == \
        direct.decision_type.value
    assert Decimal(resp["controller_decision"]["cap_amount"]) == \
        direct.cap_amount


@pytest.mark.parametrize("amount", ["-5", 0.1, True, "NaN", "Infinity"])
def test_invalid_amounts_422(amount, client):
    body = {**copy.deepcopy(TREASURY), "proposed_amount": amount}
    r = _run(client, body)
    assert r.status_code == 422, r.text
    e = r.json()["detail"]["error"]
    assert e["type"] == "invalid_treasury_input"


def test_amount_without_treasury_stays_null(client):
    b = _run(client, {"proposed_amount": "100"}).json()
    assert b["treasury_summary"] is None
    assert b["controller_decision"] is None


def test_partial_treasury_with_amount_still_incomplete_error(client):
    body = {**TREASURY, "proposed_amount": "10000"}
    del body["treasury_policy"]
    r = _run(client, body)
    assert r.status_code == 422
    assert r.json()["detail"]["error"]["type"] == "incomplete_treasury_input"


def test_save_accepts_proposed_amount(client):
    body = {**copy.deepcopy(TREASURY), "proposed_amount": "10000",
            "run_name": "ctrl1"}
    r = client.post("/pipeline/save", json=body)
    assert r.status_code == 200
