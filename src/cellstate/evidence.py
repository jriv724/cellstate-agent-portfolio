"""EvidenceBundle construction and atomic persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas.evidence import (
    EVIDENCE_BUNDLE_SCHEMA_VERSION,
    EvidenceArtifact,
    EvidenceBundle,
    EvidenceExecutionStatus,
    EvidenceWarning,
)


_ARTIFACT_METADATA = {
    "sample_qc.csv": ("sample_qc", "QC", "text/csv"),
    "eligible_cell_indices.npy": (
        "eligible_cell_indices", "input-derived", "application/x-npy"
    ),
    "design_assessment.json": ("design_assessment", "QC", "application/json"),
    "dataset_by_group.csv": ("dataset_by_group", "QC", "text/csv"),
    "pseudobulk_counts.npz": (
        "pseudobulk_counts", "input-derived", "application/x-npz"
    ),
    "pseudobulk_sample_metadata.csv": (
        "pseudobulk_sample_metadata", "input-derived", "text/csv"
    ),
    "genes.tsv": ("genes", "input-derived", "text/tab-separated-values"),
    "aggregation_summary.json": (
        "aggregation_summary", "descriptive", "application/json"
    ),
    "cache_manifest.json": ("cache_manifest", "manifest", "application/json"),
    "run_status.json": ("run_status", "manifest", "application/json"),
}


def _json_safe(value: Any, *, location: str = "value") -> Any:
    """Return plain JSON data and reject live scientific/runtime objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, location=f"{location}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if hasattr(value, "item") and callable(value.item):
        return _json_safe(value.item(), location=location)
    raise TypeError(
        f"{location} must contain only JSON-safe deterministic summaries; "
        f"received {type(value).__name__}"
    )


def collect_evidence_artifacts(root: Path) -> tuple[EvidenceArtifact, ...]:
    artifacts = []
    for filename, (logical_name, category, media_type) in _ARTIFACT_METADATA.items():
        path = root / filename
        if path.exists():
            artifacts.append(
                EvidenceArtifact(
                    logical_name=logical_name,
                    path=str(path),
                    category=category,
                    media_type=media_type,
                )
            )
    return tuple(artifacts)


def _design_warnings(
    design: Mapping[str, Any],
    *,
    group_a: str,
    group_b: str,
) -> tuple[EvidenceWarning, ...]:
    warnings: list[EvidenceWarning] = []
    counts = design.get("group_sample_counts", {})
    if not design.get("ready", False):
        warnings.append(
            EvidenceWarning(
                code="DESIGN_NOT_READY",
                message="The deterministic design-readiness checks did not pass.",
                severity="error",
                context={"group_sample_counts": counts},
            )
        )
    if design.get("shared_datasets", 0) == 0:
        warnings.append(
            EvidenceWarning(
                code="NO_DATASET_OVERLAP",
                message=(
                    "No dataset contains eligible samples from both comparison "
                    "groups; group and dataset may be confounded."
                ),
                severity="error",
                context={"group_a": group_a, "group_b": group_b},
            )
        )
    if design.get("full_rank") is False:
        warnings.append(
            EvidenceWarning(
                code="NON_ESTIMABLE_DESIGN",
                message="The deterministic design matrix is not full rank.",
                severity="error",
                context={
                    "design_rank": design.get("design_rank"),
                    "design_columns": design.get("design_columns"),
                },
            )
        )
    return tuple(warnings)


def build_evidence_bundle(
    *,
    state: Mapping[str, Any],
    execution_status: EvidenceExecutionStatus | str,
    output_directory: Path,
    design_assessment: Mapping[str, Any],
    deterministic_evidence: Mapping[str, Any] | None = None,
    cache_key: str | None = None,
    cache_directory: Path | None = None,
    cache_hit: bool = False,
    limitations: Sequence[str] = (),
) -> EvidenceBundle:
    """Build a bundle from persisted deterministic summaries only."""
    status = EvidenceExecutionStatus(execution_status)
    design = _json_safe(design_assessment, location="design_assessment")
    evidence = _json_safe(
        deterministic_evidence or {}, location="deterministic_evidence"
    )
    context = _json_safe(
        {
            "cell_state": state.get("cell_type", ""),
            "group_a": state.get("group_a", ""),
            "group_b": state.get("group_b", ""),
            "contrast": (
                f"{state.get('group_a', '')} versus {state.get('group_b', '')}"
            ),
            "contrast_direction": "group_a_minus_group_b",
        },
        location="biological_context",
    )
    warnings = _design_warnings(
        design,
        group_a=str(state.get("group_a", "")),
        group_b=str(state.get("group_b", "")),
    )
    artifacts = collect_evidence_artifacts(output_directory)
    if not artifacts:
        raise ValueError("no deterministic artifacts are available for EvidenceBundle")

    identity_payload = {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "question": state.get("question", ""),
        "analysis_type": state.get("analysis_type", "pseudobulk_de"),
        "context": context,
        "status": status.value,
        "design": design,
        "evidence": evidence,
        "cache_key": cache_key,
    }
    bundle_id = sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    default_limitations = [
        "Cell counts are observations; biological samples are the unit of inference.",
        "The bundle contains deterministic evidence and does not establish causality.",
    ]
    if status is EvidenceExecutionStatus.BLOCKED:
        default_limitations.append(
            "Inferential execution was blocked by deterministic design-readiness checks."
        )
    if evidence.get("integer_like") is False:
        default_limitations.append(
            "The selected count source did not appear integer-like and requires review."
        )

    return EvidenceBundle(
        bundle_id=bundle_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        execution_status=status,
        analysis_question=str(state.get("question") or "Unspecified analysis question"),
        analysis_type=str(state.get("analysis_type") or "pseudobulk_de"),
        biological_context=context,
        unit_of_inference="biological_sample",
        deterministic_evidence=evidence,
        design_assessment=design,
        limitations=tuple(dict.fromkeys([*default_limitations, *limitations])),
        warnings=warnings,
        artifacts=artifacts,
        provenance={
            "producer": "cellstate.evidence.build_evidence_bundle",
            "output_directory": str(output_directory),
            "source": "deterministic_execution",
        },
        cache={
            "cache_key": cache_key,
            "cache_directory": str(cache_directory) if cache_directory else None,
            "cache_hit": cache_hit,
        },
    )


def write_evidence_bundle_atomic(bundle: EvidenceBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(bundle.to_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
