"""Persistence layer for PipelineResult (output-only, stdlib only).

ARCHITECTURAL BOUNDARY: persistence is outside the financial decision
path. run_pipeline() never persists anything automatically; callers opt
in explicitly via save_pipeline_result().

Format: human-readable UTF-8 JSON, stable key ordering, explicit
"schema_version". Values that need type fidelity use tagged objects:
    {"__type__": "dataclass:<ClassName>", "fields": {...}}
    {"__type__": "enum:<EnumName>", "value": ...}
    {"__type__": "decimal", "value": "..."}      # string form: no float loss
    {"__type__": "tuple", "items": [...]}
    {"__type__": "date"/"datetime", "value": iso}
Decoding is strictly schema-driven -- no pickle, no eval, no arbitrary
object construction; unknown tags/fields raise StorageError.
"""
from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path

SCHEMA_VERSION = 1

_REGISTRY: dict[str, type] = {
    f"{m.__name__}.{c.__name__}": c
    for m in (
        __import__(n) for n in (
            "finance_controller.models",
            "finance_controller.exceptions",
            "finance_controller.investigator",
            "finance_controller.reconciliation",
            "finance_controller.evaluation",
            "finance_controller.pipeline",
        ))
    for c in vars(m).values()
    if isinstance(c, type) and is_dataclass(c)
}
_ENUM_REGISTRY: dict[str, type] = {
    e.__name__: e
    for mod in ("finance_controller.models", "finance_controller.exceptions",
                "finance_controller.investigator", "finance_controller.evaluation")
    for e in vars(__import__(mod)).values()
    if isinstance(e, type) and issubclass(e, Enum)
}

_REQUIRED_TOP = ("schema_version", "pipeline_result")


class StorageError(RuntimeError):
    """Raised on missing files, malformed JSON/schema, or corrupt data.
    Corrupted payloads are never silently repaired."""


# ---------------- encoding ----------------

def _enc(v):
    if is_dataclass(v) and not isinstance(v, type):
        return {"__type__": f"dataclass:{type(v).__module__.rsplit('.', 1)[-1]}.{type(v).__name__}",
                "fields": {f.name: _enc(getattr(v, f.name))
                           for f in fields(type(v))}}
    if isinstance(v, Enum):
        return {"__type__": f"enum:{type(v).__name__}", "value": v.value}
    if isinstance(v, Decimal):
        return {"__type__": "decimal", "value": str(v)}
    if isinstance(v, datetime):
        return {"__type__": "datetime", "value": v.isoformat()}
    if isinstance(v, date):
        return {"__type__": "date", "value": v.isoformat()}
    if isinstance(v, tuple):
        return {"__type__": "tuple", "items": [_enc(x) for x in v]}
    if isinstance(v, list):
        return [_enc(x) for x in v]
    if isinstance(v, dict):
        return {k: _enc(x) for k, x in sorted(v.items())}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    raise StorageError(f"cannot serialize type {type(v).__name__}")


def save_pipeline_result(result, path) -> None:
    """Persist pipeline OUTPUT only. Never accepts/persists clients,
    keys, transports, or arbitrary Python objects."""
    payload = {"schema_version": SCHEMA_VERSION,
               "pipeline_result": _enc(result)}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                            sort_keys=True), encoding="utf-8")


# ---------------- decoding ----------------

def _dec(v):
    if isinstance(v, dict):
        tag = v.get("__type__")
        if tag is None:
            return {k: _dec(x) for k, x in v.items()}
        kind, _, name = str(tag).partition(":")
        try:
            if kind == "dataclass":
                cls = _REGISTRY[name]
                raw = v["fields"]
                expected = {f.name for f in fields(cls)}
                if set(raw) != expected:
                    raise StorageError(
                        f"{name}: field mismatch {set(raw) ^ expected}")
                return cls(**{k: _dec(x) for k, x in raw.items()})
            if kind == "enum":
                return _ENUM_REGISTRY[name](v["value"])
            if kind == "decimal":
                return Decimal(str(v["value"]))
            if kind in ("date", "datetime"):
                return (datetime.fromisoformat if kind == "datetime"
                        else date.fromisoformat)(v["value"])
            if kind == "tuple":
                return tuple(_dec(x) for x in v["items"])
        except KeyError as e:
            raise StorageError(f"corrupt {tag} payload: missing {e}") from e
        except (ValueError, TypeError) as e:
            raise StorageError(f"corrupt {tag} payload: {e}") from e
        raise StorageError(f"unknown tag {tag!r}")
    if isinstance(v, list):
        return [_dec(x) for x in v]
    return v


def load_pipeline_result(path) -> "PipelineResult":
    p = Path(path)
    if not p.exists():
        raise StorageError(f"file not found: {p}")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise StorageError(f"malformed JSON in {p}: {e}") from e
    except UnicodeDecodeError as e:
        raise StorageError(f"not valid UTF-8 in {p}: {e}") from e
    if not isinstance(doc, dict):
        raise StorageError("top-level document must be a JSON object")
    for req in _REQUIRED_TOP:
        if req not in doc:
            raise StorageError(f"missing top-level field {req!r}")
    if doc["schema_version"] != SCHEMA_VERSION:
        raise StorageError(
            f"unsupported schema_version {doc['schema_version']!r}; "
            f"expected {SCHEMA_VERSION}")
    result = _dec(doc["pipeline_result"])
    if not isinstance(result, __import__(
            "finance_controller.pipeline",
            fromlist=["PipelineResult"]).PipelineResult):
        raise StorageError("payload is not a PipelineResult")
    return result
