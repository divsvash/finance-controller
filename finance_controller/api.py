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
