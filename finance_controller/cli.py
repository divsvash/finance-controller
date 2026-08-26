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
import sys
from collections import Counter

from .generator import generate_dataset, generate_external_dataset
from .llm_client import FakeLLMClient  # noqa: F401  (injection point only)
from .pipeline import run_pipeline

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
