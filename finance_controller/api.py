"""REST API layer -- thin interface over existing backend modules.

ARCHITECTURE: this module contains ZERO financial logic. It composes
generate_dataset -> generate_external_dataset -> run_pipeline ->
save_pipeline_result. The deterministic InvestigationCase remains the
sole source of financial truth; LLM output is interpretation only and
is never used for decisions here.

No auth, DB, jobs, caching, retries, or fallback clients. Production
LLM mode constructs OpenAICompatibleClient exactly as cli.py does.
"""
from __future__ import annotations

import re
from dataclasses import fields as dc_fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .generator import generate_dataset, generate_external_dataset
from .pipeline import PipelineResult, run_pipeline
from .storage import RUNS_DIR_DEFAULT, load_pipeline_result, save_pipeline_result
from .models import Transaction, ExternalRecord

RUNS_DIR = RUNS_DIR_DEFAULT          # application-controlled output dir
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


# ---------- request models ----------
class PipelineRequest(BaseModel):
    seed: int = Field(default=42)
    external_seed: int = Field(default=99)
    run_llm: bool = False
    run_evaluation: bool = False
    enable_date_fallback: bool = False


class SaveRequest(PipelineRequest):
    run_name: str = Field(default="run")


# ---------- serialization (interface concern ONLY) ----------
def _jsonable(v):
    if is_dataclass(v) and not isinstance(v, type):
        return {f.name: _jsonable(getattr(v, f.name))
                for f in dc_fields(type(v))}
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, Decimal):
        return str(v)                       # never float-ify money
    if isinstance(v, datetime | date):
        return v.isoformat()
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def serialize_result(r: PipelineResult) -> dict[str, Any]:
    """JSON-safe view of PipelineResult. No dataclass reprs, no new
    financial schema -- field names mirror the domain dataclasses."""
    rec = r.reconciliation_results
    matched = sum(1 for x in rec if getattr(x, "status", None) is not None)
    return {
        "case_count": r.case_count,
        "reconciliation": {
            "result_count": len(rec),
            "report": _jsonable(r.reconciliation_report),
        },
        "investigation_cases": _jsonable(r.investigation_cases),
        "deterministic_assessments": _jsonable(r.deterministic_assessments),
        "llm_assessments": (_jsonable(r.llm_assessments)
                            if r.llm_assessments is not None else None),
        "evaluation": ({
            "results": _jsonable(r.evaluation_results),
            "summary": _jsonable(r.evaluation_summary),
        } if r.evaluation_summary is not None else None),
    }


def error_response(exc_type: str, message: str) -> dict:
    return {"error": {"type": exc_type, "message": message}}


def _sanitize_run_name(name: str) -> str:
    if not _NAME_RE.match(name) or ".." in name or "/" in name or "\\" in name:
        raise ValueError(
            "invalid run_name: use simple filenames like 'demo' "
            "(letters, digits, dot, dash)")
    return name


def create_app(llm_client: Any = None,
               llm_client_factory: Optional[callable] = None,
               runs_dir: str = RUNS_DIR) -> FastAPI:
    """Build the app. Tests inject an offline client via `llm_client`
    (or a factory); production leaves both None so the CLI-style real
    provider is constructed lazily on demand. Never auto-fallbacks."""
    app = FastAPI(title="Finance Controller API",
                  version="1.0.0")

    def make_llm():
        if llm_client is not None:
            return llm_client
        if llm_client_factory is not None:
            return llm_client_factory()
        from .llm_provider import OpenAICompatibleClient   # lazy import
        return OpenAICompatibleClient()   # raises MissingAPIKeyError clearly

    def execute(req: PipelineRequest) -> PipelineResult:
        if req.run_evaluation and not req.run_llm:
            raise HTTPException(status_code=422, detail=error_response(
                "invalid_configuration",
                "run_evaluation=true requires run_llm=true"))
        try:
            ds = generate_dataset(seed=req.seed)
            exts, _ = generate_external_dataset(ds, seed=req.external_seed)
            kwargs = {}
            if req.run_llm:
                kwargs["llm_client"] = make_llm()
            return run_pipeline(list(ds.transactions), exts,
                                run_llm=req.run_llm,
                                run_evaluation=req.run_evaluation,
                                enable_date_fallback=req.enable_date_fallback,
                                **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            # provider/config failures -> 5xx/4xx, no secrets/tracebacks
            msg = str(e)
            import os
            key = os.environ.get("FINANCE_LLM_API_KEY", "")
            if key:
                msg = msg.replace(key, "<redacted>")
            status = 400 if type(e).__name__ == "MissingAPIKeyError" else 502 \
                if req.run_llm else 500
            raise HTTPException(status_code=status, detail=error_response(
                type(e).__name__, msg)) from None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/pipeline/run")
    def pipeline_run(req: PipelineRequest):
        result = execute(req)
        body = serialize_result(result)
        body["llm_run"] = req.run_llm
        body["evaluation_run"] = req.run_evaluation
        return body

    @app.post("/pipeline/save")
    def pipeline_save(req: SaveRequest):
        try:
            name = _sanitize_run_name(req.run_name)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=error_response(
                "invalid_run_name", str(e)))
        result = execute(req)
        from pathlib import Path
        target_dir = Path(runs_dir)
        target = target_dir / f"{name}.json"
        try:
            save_pipeline_result(result, target)
        except Exception as e:
            raise HTTPException(status_code=500, detail=error_response(
                "persistence_error",
                "failed to persist run")) from None
        rel = target.as_posix()
        return {
            "run_name": name,
            "saved_path": rel,           # application-safe relative path
            "case_count": result.case_count,
            "llm_run": req.run_llm,
            "evaluation_run": req.run_evaluation,
        }

    return app


app = create_app()

# --- extended PipelineRequest (defaults unchanged; behavior identical
#     when neither field is supplied) ---
class PipelineRequest(BaseModel):
    seed: int = Field(default=42)
    external_seed: int = Field(default=99)
    run_llm: bool = False
    run_evaluation: bool = False
    enable_date_fallback: bool = False
    # Optional direct input. When BOTH are provided they are parsed with
    # the EXISTING domain dataclasses (no duplicate financial schema) and
    # passed straight to run_pipeline(). Seeds are ignored in that case.
    transactions: Optional[List[Dict[str, Any]]] = None
    external_records: Optional[List[Dict[str, Any]]] = None


class _InputError(Exception):
    """Invalid user-supplied records -> mapped to HTTP 422."""


def _parse_transactions(raw: List[dict]) -> List[Transaction]:
    try:
        return [Transaction(**item) for item in raw]
    except Exception as e:
        raise _InputError(f"invalid transaction record: {e}") from None


def _parse_external(raw: List[dict]) -> List[ExternalRecord]:
    try:
        return [ExternalRecord(**item) for item in raw]
    except Exception as e:
        raise _InputError(f"invalid external_record: {e}") from None


def _resolve_inputs(req: PipelineRequest):
    """Returns (transactions, externals). Generated when neither field
    is supplied; otherwise both must be supplied and valid."""
    if req.transactions is None and req.external_records is None:
        ds = generate_dataset(seed=req.seed)
        exts, _ = generate_external_dataset(ds, seed=req.external_seed)
        return list(ds.transactions), exts
    if (req.transactions is None) != (req.external_records is None):
        raise HTTPException(status_code=422, detail=error_response(
            "incomplete_input",
            "provide BOTH 'transactions' and 'external_records' "
            "(or neither; generated dataset will be used)"))
    if not isinstance(req.transactions, list) or \
       not isinstance(req.external_records, list):
        raise HTTPException(status_code=422, detail=error_response(
            "invalid_input", "transactions/external_records must be arrays"))
    try:
        txns = _parse_transactions(req.transactions)
        exts = _parse_external(req.external_records)
    except _InputError as e:
        raise HTTPException(status_code=422, detail=error_response(
            "invalid_record", str(e)))
    return txns, exts

    # --- additive imports ---
import datetime as _dt
from decimal import Decimal, InvalidOperation
from .treasury import (
    CashPosition, Certainty, ControllerPolicy, ExpectedFlow,
    FlowCategory, FlowDirection, TreasurySummary)

# --- extended PipelineRequest (additive; defaults preserve behavior) ---
class PipelineRequest(BaseModel):
    # ...all existing fields unchanged...
    cash_position: Optional[Dict[str, Any]] = None
    expected_flows: Optional[List[Dict[str, Any]]] = None
    treasury_policy: Optional[Dict[str, Any]] = None


def _dec(value: Any, name: str) -> Decimal:
    """Money arrives as JSON strings/integers — never floats."""
    if isinstance(value, bool):                       # bool is an int subclass
        raise ValueError(f"{name}: boolean is not a monetary value")
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        raise ValueError(f"{name}: not a Decimal-compatible value")


def _iso_date(value: Any, name: str) -> _dt.date:
    if not isinstance(value, str):
        raise ValueError(f"{name}: date must be an ISO YYYY-MM-DD string")
    try:
        return _dt.date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name}: invalid ISO date {value!r}")


def _enum(enum_cls, value: Any, name: str):
    try:
        return enum_cls(value)
    except ValueError:
        raise ValueError(
            f"{name}: invalid {enum_cls.__name__} value {value!r}")


class _TreasuryInputError(Exception):
    pass


def _parse_cash_position(raw: Dict[str, Any]) -> CashPosition:
    try:
        return CashPosition(
            as_of=_iso_date(raw.get("as_of"), "cash_position.as_of"),
            opening_balance=_dec(raw.get("opening_balance"),
                                 "cash_position.opening_balance"),
            cleared_inflows=_dec(raw.get("cleared_inflows"),
                                 "cash_position.cleared_inflows"),
            cleared_outflows=_dec(raw.get("cleared_outflows"),
                                  "cash_position.cleared_outflows"))
    except ValueError as e:
        raise _TreasuryInputError(str(e))


def _parse_expected_flow(raw: Dict[str, Any]) -> ExpectedFlow:
    try:
        return ExpectedFlow(
            flow_id=str(raw["flow_id"]),
            direction=_enum(FlowDirection, raw.get("direction"),
                            f"flow[{raw.get('flow_id')}].direction"),
            amount=_dec(raw.get("amount"), f"flow[{raw.get('flow_id')}].amount"),
            expected_date=_iso_date(raw.get("expected_date"), "expected_date"),
            category=_enum(FlowCategory, raw.get("category", "OTHER"),
                           "category"),
            certainty=_enum(Certainty, raw.get("certainty"),
                            f"flow[{raw.get('flow_id')}].certainty"),
            linked_transaction_id=raw.get("linked_transaction_id"))
    except KeyError as e:
        raise _TreasuryInputError(f"expected flow missing field {e}")
    except ValueError as e:
        raise _TreasuryInputError(str(e))


def _parse_policy(raw: Dict[str, Any]) -> ControllerPolicy:
    include_forecast = raw.get("include_forecast_flows")
    if not isinstance(include_forecast, bool):
        raise _TreasuryInputError(
            "treasury_policy.include_forecast_flows must be a boolean")
    try:
        return ControllerPolicy(
            minimum_cash_reserve=_dec(raw.get("minimum_cash_reserve"),
                                      "policy.minimum_cash_reserve"),
            reserve_buffer_pct=_dec(raw.get("reserve_buffer_pct"),
                                    "policy.reserve_buffer_pct"),
            max_single_movement_pct=_dec(raw.get("max_single_movement_pct"),
                                         "policy.max_single_movement_pct"),
            include_forecast_flows=include_forecast)
    except ValueError as e:   # domain validation (negatives, pct > 1)
        raise _TreasuryInputError(str(e))


def _resolve_treasury_inputs(req: PipelineRequest):
    """All-or-none rule, mirroring run_pipeline()'s validation."""
    supplied = [req.cash_position is not None,
                req.expected_flows is not None,
                req.treasury_policy is not None]
    if any(supplied) and not all(supplied):
        raise HTTPException(status_code=422, detail=error_response(
            "incomplete_treasury_input",
            "supply ALL of cash_position, expected_flows, treasury_policy "
            "— or none (treasury disabled)"))
    if not all(supplied):
        return None, None, None
    try:
        pos = _parse_cash_position(req.cash_position)
        flows = [_parse_expected_flow(f) for f in req.expected_flows]
        pol = _parse_policy(req.treasury_policy)
    except _TreasuryInputError as e:
        raise HTTPException(status_code=422, detail=error_response(
            "invalid_treasury_input", str(e)))
    return pos, flows, pol

def _dec(value: Any, name: str) -> Decimal:
    """Money arrives as strings/integers — never floats."""
    if isinstance(value, bool):                       # bool is an int subclass
        raise ValueError(f"{name}: boolean is not a monetary value")
    if isinstance(value, float):
        raise ValueError(
            f"{name}: float values are not accepted for monetary fields; "
            "use a string or integer")
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        raise ValueError(f"{name}: not a Decimal-compatible value")
