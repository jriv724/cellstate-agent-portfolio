"""Typed contracts for cross-sectional progression and paired kinetics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .common import (AnalysisProvenance, ArtifactCategory, ArtifactReference,
                     CapabilityResult, StructuredWarning, capability_status_from_warnings)


@dataclass(frozen=True)
class CrossSectionalProgressionInput:
    table_path: Path
    input_format: Literal["csv", "tsv"] = "csv"
    sample_column: str = "sample"
    dataset_column: str | None = "dataset"
    stage_column: str = "stage_model_v2"
    stratum_column: str = "preserved"
    outcome_column: str = "fraction"
    stage_order: tuple[str, ...] = ("NBM", "MGUS", "SMM", "NDMM", "RRMM", "MM-Remission")
    minimum_samples_per_stage: int = 3
    minimum_stages_passing: int = 3
    minimum_mean_outcome: float = 1e-4
    p_adjust_method: Literal["BH"] = "BH"

    def __post_init__(self) -> None:
        if not isinstance(self.table_path, Path):
            object.__setattr__(self, "table_path", Path(self.table_path))
        if len(self.stage_order) < 2 or len(set(self.stage_order)) != len(self.stage_order):
            raise ValueError("stage_order must contain at least two unique ordered stages")
        if self.minimum_samples_per_stage < 1 or self.minimum_stages_passing < 2:
            raise ValueError("progression replication thresholds are invalid")
        if self.minimum_mean_outcome < 0 or self.p_adjust_method != "BH":
            raise ValueError("source requires nonnegative mean threshold and BH")

    def parameters(self) -> dict[str, object]:
        return {key: (list(value) if isinstance(value, tuple) else value) for key, value in self.__dict__.items() if key != "table_path"}


@dataclass(frozen=True)
class PairedProgressionInput:
    table_path: Path
    input_format: Literal["csv", "tsv"] = "tsv"
    sample_column: str = "sample"
    patient_column: str = "patient_id"
    dataset_column: str | None = None
    class_column: str = "cell_type"
    stage_column: str = "stage_model"
    outcome_column: str = "mean_bm_retrained_age"
    earlier_stage: str = "NDMM"
    later_stage: str = "RRMM"
    duplicate_sample_rule: Literal["mean"] = "mean"
    minimum_pairs_for_test: int = 3
    p_adjust_method: Literal["BH"] = "BH"

    def __post_init__(self) -> None:
        if not isinstance(self.table_path, Path):
            object.__setattr__(self, "table_path", Path(self.table_path))
        if self.earlier_stage == self.later_stage:
            raise ValueError("earlier_stage and later_stage must differ")
        if self.duplicate_sample_rule != "mean" or self.minimum_pairs_for_test < 1 or self.p_adjust_method != "BH":
            raise ValueError("unsupported paired source configuration")

    def parameters(self) -> dict[str, object]:
        return {key: value for key, value in self.__dict__.items() if key != "table_path"}


@dataclass(frozen=True)
class LongitudinalKineticsInput:
    table_path: Path
    input_format: Literal["csv", "tsv"] = "tsv"
    patient_column: str = "patient_id"
    dataset_column: str | None = None
    class_column: str = "cell_type"
    timepoint_column: str = "timepoint"
    outcome_column: str = "mean_bm_retrained_age"
    timepoint_order: tuple[str, ...] = ("S", "D28", "M3")
    contrasts: tuple[tuple[str, str], ...] = (("S", "D28"), ("S", "M3"), ("D28", "M3"))
    minimum_pairs_for_test: int = 3
    p_adjust_method: Literal["BH"] = "BH"

    def __post_init__(self) -> None:
        if not isinstance(self.table_path, Path):
            object.__setattr__(self, "table_path", Path(self.table_path))
        if len(self.timepoint_order) < 2 or len(set(self.timepoint_order)) != len(self.timepoint_order):
            raise ValueError("timepoint_order must contain unique ordered levels")
        positions = {value: index for index, value in enumerate(self.timepoint_order)}
        for earlier, later in self.contrasts:
            if earlier not in positions or later not in positions or positions[earlier] >= positions[later]:
                raise ValueError("contrasts must follow declared timepoint order")
        if self.minimum_pairs_for_test < 1 or self.p_adjust_method != "BH":
            raise ValueError("unsupported longitudinal source configuration")

    def parameters(self) -> dict[str, object]:
        result = {key: value for key, value in self.__dict__.items() if key != "table_path"}
        result["timepoint_order"] = list(self.timepoint_order)
        result["contrasts"] = [list(value) for value in self.contrasts]
        return result


@dataclass(frozen=True)
class ProgressionOutput:
    capability_id: str
    node_version: str
    cache_key: str
    cache_hit: bool
    descriptive_paths: tuple[str, ...]
    inferential_paths: tuple[str, ...]
    provenance_path: str
    cache_manifest_path: str
    warnings: tuple[StructuredWarning, ...]
    provenance: AnalysisProvenance

    def __post_init__(self) -> None:
        if self.capability_id not in {"CAP-COMP-002", "CAP-STAT-003", "CAP-STAT-004"}:
            raise ValueError("unexpected progression capability")

    def to_capability_result(self) -> CapabilityResult:
        artifacts = [ArtifactReference(f"descriptive_{i + 1}", path, ArtifactCategory.DESCRIPTIVE, "text/csv")
                     for i, path in enumerate(self.descriptive_paths)]
        artifacts.extend(ArtifactReference(f"inferential_{i + 1}", path, ArtifactCategory.INFERENTIAL, "text/csv")
                         for i, path in enumerate(self.inferential_paths))
        artifacts.extend((ArtifactReference("provenance", self.provenance_path, ArtifactCategory.PROVENANCE, "application/json"),
                          ArtifactReference("cache_manifest", self.cache_manifest_path, ArtifactCategory.MANIFEST, "application/json")))
        return CapabilityResult(self.capability_id, self.node_version,
            capability_status_from_warnings(self.warnings), self.cache_key, self.cache_hit,
            tuple(artifacts), self.warnings, self.provenance_path, self.cache_manifest_path)
