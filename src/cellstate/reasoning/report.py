"""Deterministic ScientificReport assembly and atomic JSON persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cellstate.schemas.evidence import EvidenceBundle
from cellstate.schemas.reasoning import (
    CRITIC_PROMPT_VERSION,
    INTERPRETER_PROMPT_VERSION,
    REASONING_SCHEMA_VERSION,
    CriticReport,
    InterpretationReport,
    ScientificReport,
)

from .exceptions import ReasoningValidationError
from .validation import validate_critic, validate_interpretation


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def assemble_scientific_report(
    bundle: EvidenceBundle,
    critic: CriticReport,
    interpretation: InterpretationReport,
    *,
    model_name: str,
    report_paths: dict[str, str],
    created_at_utc: str | None = None,
) -> ScientificReport:
    validate_critic(bundle, critic)
    validate_interpretation(bundle, critic, interpretation)
    timestamp = created_at_utc or utc_now()
    cache_hit = bool(bundle.cache.get("cache_hit", False))
    summary = (
        f"Deterministic execution status: {bundle.execution_status.value}. "
        f"Analysis type: {bundle.analysis_type}. "
        f"Unit of inference: {bundle.unit_of_inference}. "
        f"EvidenceBundle {bundle.bundle_id} contains "
        f"{len(bundle.artifacts)} referenced deterministic artifacts and "
        f"{len(bundle.warnings)} structured warnings."
    )
    return ScientificReport(
        evidence_bundle_id=bundle.bundle_id,
        deterministic_executive_summary=summary,
        critic_report=critic,
        interpretation_report=interpretation,
        provenance={
            "evidence_bundle_schema_version": bundle.schema_version,
            "evidence_bundle_id": bundle.bundle_id,
            "deterministic_execution_status": bundle.execution_status.value,
            "deterministic_restored_from_cache": cache_hit,
            "critic_prompt_version": CRITIC_PROMPT_VERSION,
            "interpreter_prompt_version": INTERPRETER_PROMPT_VERSION,
            "reasoning_schema_version": REASONING_SCHEMA_VERSION,
            "openai_model": model_name,
            "critic_timestamp_utc": critic.created_at_utc,
            "interpreter_timestamp_utc": interpretation.created_at_utc,
            "scientific_report_timestamp_utc": timestamp,
            "report_artifact_paths": dict(report_paths),
        },
        created_at_utc=timestamp,
    )


def write_model_atomic(model: BaseModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(model.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception as exc:
        raise ReasoningValidationError(
            f"Could not persist reasoning report {path.name}."
        ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
