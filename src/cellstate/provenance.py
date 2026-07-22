"""Construction and atomic persistence of analysis provenance."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas.common import AnalysisProvenance, StructuredWarning


def build_provenance(
    *,
    capability_id: str,
    node_version: str,
    cache_schema_version: int,
    source_files: Sequence[str],
    source_locations: Sequence[str],
    input_dataset_signature: str,
    parameters: Mapping[str, Any],
    model_formula: str | None,
    reference_group: str | None,
    covariates: Sequence[str],
    unit_of_inference: str,
    random_seed: int | None,
    software_versions: Mapping[str, str],
    output_paths: Sequence[str],
    warnings: Sequence[StructuredWarning],
) -> AnalysisProvenance:
    return AnalysisProvenance(
        capability_id=capability_id,
        node_version=node_version,
        cache_schema_version=cache_schema_version,
        source_files=tuple(source_files),
        source_locations=tuple(source_locations),
        input_dataset_signature=input_dataset_signature,
        parameters=dict(parameters),
        model_formula=model_formula,
        reference_group=reference_group,
        covariates=tuple(covariates),
        unit_of_inference=unit_of_inference,
        random_seed=random_seed,
        software_versions=dict(software_versions),
        output_paths=tuple(output_paths),
        warnings=tuple(warnings),
        execution_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def write_provenance_atomic(provenance: AnalysisProvenance, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(provenance.to_dict(), sort_keys=True, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
