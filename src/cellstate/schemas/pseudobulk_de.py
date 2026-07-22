"""Typed raw-count pseudobulk construction contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .common import AnalysisProvenance, StructuredWarning


DEFAULT_STATES = (
    "Memory B cell", "Memory CD4 T cell", "Naive CD4 T cell", "Naive CD8 T cell",
    "GZMK CD8 T cell", "GZMB CD8 T cell", "CD16 NK cell",
    "MHCII high CD14 monocyte", "MHCII low CD14 monocyte",
)


@dataclass(frozen=True)
class PseudobulkInput:
    raw_counts_path: Path
    metadata_path: Path
    input_format: Literal["csv", "tsv"] = "csv"
    count_source: Literal["raw_counts"] = "raw_counts"
    cell_id_column: str = "cell_id"
    sample_column: str = "sample"
    patient_column: str | None = None
    dataset_column: str = "dataset"
    stage_column: str = "stage_model_v2"
    cell_state_column: str = "preserved"
    stages: tuple[str, ...] = ("NBM", "SMM", "NDMM")
    cell_states: tuple[str, ...] = DEFAULT_STATES
    minimum_cells_per_sample_state: int = 100
    minimum_library_size_warning: int = 1

    def __post_init__(self) -> None:
        for name in ("raw_counts_path", "metadata_path"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))
        if self.input_format not in {"csv", "tsv"} or self.count_source != "raw_counts":
            raise ValueError("only CSV/TSV raw_counts input is supported")
        if len(set(self.stages)) != len(self.stages) or not self.stages:
            raise ValueError("stages must be unique and nonempty")
        if len(set(self.cell_states)) != len(self.cell_states) or not self.cell_states:
            raise ValueError("cell_states must be unique and nonempty")
        if self.minimum_cells_per_sample_state < 1 or self.minimum_library_size_warning < 0:
            raise ValueError("count thresholds are invalid")

    def parameters(self) -> dict[str, object]:
        return {
            "input_format": self.input_format, "count_source": self.count_source,
            "cell_id_column": self.cell_id_column, "sample_column": self.sample_column,
            "patient_column": self.patient_column, "dataset_column": self.dataset_column,
            "stage_column": self.stage_column, "cell_state_column": self.cell_state_column,
            "stages": list(self.stages), "cell_states": list(self.cell_states),
            "minimum_cells_per_sample_state": self.minimum_cells_per_sample_state,
            "minimum_library_size_warning": self.minimum_library_size_warning,
        }


@dataclass(frozen=True)
class PseudobulkOutput:
    capability_id: str
    node_version: str
    cache_key: str
    cache_hit: bool
    count_matrix_paths: tuple[str, ...]
    sample_metadata_path: str
    qc_path: str
    provenance_path: str
    cache_manifest_path: str
    warnings: tuple[StructuredWarning, ...]
    provenance: AnalysisProvenance

    def __post_init__(self) -> None:
        if self.capability_id != "CAP-DESEQ-001":
            raise ValueError("unexpected pseudobulk capability")

    def to_capability_result(self):
        from .common import ArtifactCategory, ArtifactReference, CapabilityResult, capability_status_from_warnings
        artifacts = [ArtifactReference(f"count_matrix_{i + 1}", path, ArtifactCategory.INPUT_DERIVED, "text/csv")
                     for i, path in enumerate(self.count_matrix_paths)]
        artifacts.extend((ArtifactReference("sample_metadata", self.sample_metadata_path, ArtifactCategory.INPUT_DERIVED, "text/csv"),
                          ArtifactReference("pseudobulk_qc", self.qc_path, ArtifactCategory.QC, "text/csv"),
                          ArtifactReference("provenance", self.provenance_path, ArtifactCategory.PROVENANCE, "application/json"),
                          ArtifactReference("cache_manifest", self.cache_manifest_path, ArtifactCategory.MANIFEST, "application/json")))
        return CapabilityResult(self.capability_id, self.node_version, capability_status_from_warnings(self.warnings),
                                self.cache_key, self.cache_hit, tuple(artifacts), self.warnings,
                                self.provenance_path, self.cache_manifest_path)

# Backward-compatible schema imports; implementation lives in schemas.deseq2.
from .deseq2 import DifferentialExpressionInput, DifferentialExpressionOutput
