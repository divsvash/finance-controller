"""Shared test helpers.

mkcases() builds the canonical 122-case InvestigationCase fixture set,
using the same seed=42 (transactions) / seed=99 (external records)
convention used identically throughout this test suite, and the same
reconcile() -> build_investigation_cases() sequence already used inside
finance_controller.pipeline.run_pipeline().
"""
from finance_controller.exceptions import (
    ExceptionType, build_investigation_cases)
from finance_controller.generator import (
    generate_dataset, generate_external_dataset)
from finance_controller.reconciliation import reconcile


def mkcases():
    ds = generate_dataset(seed=42)
    exts, _ = generate_external_dataset(ds, seed=99)
    txns = list(ds.transactions)
    results, _ = reconcile(txns, exts)
    return build_investigation_cases(results, {t.id: t for t in txns}, exts)


def golden_mismatch_case():
    return next(c for c in mkcases()
                if c.exception_type == ExceptionType.AMOUNT_MISMATCH)
