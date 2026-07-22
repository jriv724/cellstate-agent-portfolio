from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .common import (AnalysisProvenance, ArtifactCategory, ArtifactReference,
                     CapabilityResult, StructuredWarning, capability_status_from_warnings)

DEFAULT_STAGE_ORDER = ("NBM", "MGUS", "SMM", "NDMM", "RRMM", "MM-Remission")


@dataclass(frozen=True)
class AbundanceInput:
    metadata_path: Path
    input_format: Literal["csv", "tsv"] = "csv"
    sample_column: str = "sample"
    patient_column: str | None = None
    dataset_column: str = "dataset"
    stage_column: str = "stage_model_v2"
    cell_state_column: str = "preserved"
    macro_column: str | None = None
    stage_order: tuple[str, ...] = DEFAULT_STAGE_ORDER
    analysis_profile: Literal["standard", "plasma_rbc_removed"] = "standard"
    minimum_samples_per_stage_for_warning: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.metadata_path, Path):
            object.__setattr__(self, "metadata_path", Path(self.metadata_path))
        if self.input_format not in {"csv", "tsv"}:
            raise ValueError("input_format must be csv or tsv")
        names = (self.sample_column, self.dataset_column, self.stage_column, self.cell_state_column)
        if any(not name.strip() for name in names):
            raise ValueError("metadata column names must be nonblank")
        if not self.stage_order or len(set(self.stage_order)) != len(self.stage_order):
            raise ValueError("stage_order must contain unique ordered levels")
        if self.minimum_samples_per_stage_for_warning < 1:
            raise ValueError("minimum_samples_per_stage_for_warning must be >= 1")

    def parameters(self) -> dict[str, object]:
        return {
            "input_format": self.input_format,
            "sample_column": self.sample_column,
            "patient_column": self.patient_column,
            "dataset_column": self.dataset_column,
            "stage_column": self.stage_column,
            "cell_state_column": self.cell_state_column,
            "macro_column": self.macro_column,
            "stage_order": list(self.stage_order),
            "analysis_profile": self.analysis_profile,
            "minimum_samples_per_stage_for_warning": self.minimum_samples_per_stage_for_warning,
        }


@dataclass(frozen=True)
class AbundanceOutput:
    capability_id: str
    node_version: str
    cache_key: str
    cache_hit: bool
    sample_table_path: str
    stage_summary_path: str
    provenance_path: str
    cache_manifest_path: str
    warnings: tuple[StructuredWarning, ...]
    provenance: AnalysisProvenance

    def __post_init__(self) -> None:
        if self.capability_id != "CAP-COMP-001":
            raise ValueError("unexpected capability_id")

    def to_capability_result(self) -> CapabilityResult:
        artifacts = (
            ArtifactReference("sample_table", self.sample_table_path, ArtifactCategory.DESCRIPTIVE, "text/csv"),
            ArtifactReference("stage_summary", self.stage_summary_path, ArtifactCategory.DESCRIPTIVE, "text/csv"),
            ArtifactReference("provenance", self.provenance_path, ArtifactCategory.PROVENANCE, "application/json"),
            ArtifactReference("cache_manifest", self.cache_manifest_path, ArtifactCategory.MANIFEST, "application/json"),
        )
        return CapabilityResult(self.capability_id, self.node_version,
            capability_status_from_warnings(self.warnings), self.cache_key, self.cache_hit,
            artifacts, self.warnings, self.provenance_path, self.cache_manifest_path)
