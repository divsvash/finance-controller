"""Manual live check against the real provider. NEVER runs under pytest.

Requires FINANCE_LLM_API_KEY in the environment. Investigates exactly 3
cases end-to-end and prints their assessments.
"""
import os
import sys

from finance_controller.exceptions import build_investigation_cases
from finance_controller.generator import generate_dataset, generate_external_dataset
from finance_controller.llm_provider import MissingAPIKeyError, OpenAICompatibleClient
from finance_controller.llm_investigator import llm_investigate_case
from finance_controller.reconciliation import reconcile

if __name__ == "__main__":
    if not os.environ.get("FINANCE_LLM_API_KEY"):
        print("FINANCE_LLM_API_KEY is not set.\n"
              "Export it first, e.g.:  export FINANCE_LLM_API_KEY='sk-...'\n"
              "Optional: FINANCE_LLM_MODEL (default gpt-4o-mini), "
              "FINANCE_LLM_BASE_URL.")
        sys.exit(1)

    ds = generate_dataset(seed=42)
    exts, _ = generate_external_dataset(ds, seed=99)
    rs, _ = reconcile(list(ds.transactions), exts)
    cases = build_investigation_cases(rs, {t.id: t for t in ds.transactions},
                                      exts)
    picked = cases[:2] + [c for c in cases
                          if c.exception_type.value == "MISSING_EXTERNAL"][:1]
    try:
        client = OpenAICompatibleClient()
    except MissingAPIKeyError as e:
        print(f"cannot start live check: {e}")
        sys.exit(1)

    print(f"LIVE CHECK ({client._model})\n=================")
    for c in picked:
        try:
            a = llm_investigate_case(c, client)
        except Exception as e:
            print(f"[FAIL] {c.case_id}: {type(e).__name__}: {e}\n")
            continue
        print(f"case id      : {a.case_id}")
        print(f"type         : {a.exception_type}")
        print(f"determ. risk : {c.priority.name.lower()}")
        print(f"finding      : {a.finding}")
        print(f"explanation  : {a.explanation}")
        print(f"action       : {a.recommended_action}")
        print(f"confidence   : {a.confidence.value}\n")
