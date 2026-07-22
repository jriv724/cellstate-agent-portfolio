"""Typed contracts for CAP-LODO-001 AtlasLODO."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from .common import (AnalysisProvenance, ArtifactCategory, ArtifactReference,
                     CapabilityResult, CapabilityStatus, StructuredWarning)

@dataclass(frozen=True)
class AtlasLODOInput:
    source_path: Path
    input_format: Literal["h5ad", "cell_csv", "sample_mean_csv"] = "sample_mean_csv"
    expression_path: Path | None = None
    gene_columns: tuple[str, ...] = ()
    cell_id_column: str = "cell_id"
    sample_column: str = "sample"
    dataset_column: str = "dataset"
    group_column: str = "stage_model_v2"
    cell_state_column: str = "preserved"
    n_cells_column: str = "n_cells"
    reference_group: str = "NBM"
    group_a: str = "SMM"
    group_b: str = "NDMM"
    requested_cell_states: tuple[str, ...] = ()
    minimum_cells_per_sample_state: int = 100
    minimum_reference_samples: int = 10
    minimum_group_a_samples: int = 5
    minimum_group_b_samples: int = 10
    minimum_datasets: int = 3
    minimum_training_samples_per_group: int = 3
    minimum_successful_folds: int = 3
    pseudocount: float = 1.0
    minimum_mean_expression: float = 0.02
    group_a_absolute_effect_cutoff: float = 0.20
    group_b_absolute_effect_cutoff: float = 0.20
    group_a_fdr_cutoff: float = 0.10
    group_b_fdr_cutoff: float = 0.10
    group_a_sign_consistency_cutoff: float = 0.80
    group_b_sign_consistency_cutoff: float = 0.80
    delta_absolute_cutoff: float = 0.25
    delta_fdr_cutoff: float = 0.10
    delta_sign_consistency_cutoff: float = 0.80

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, Path): object.__setattr__(self, "source_path", Path(self.source_path))
        if self.expression_path is not None and not isinstance(self.expression_path, Path):
            object.__setattr__(self, "expression_path", Path(self.expression_path))
        if len({self.reference_group, self.group_a, self.group_b}) != 3:
            raise ValueError("reference_group, group_a, and group_b must be distinct and directionally ordered")
        if self.input_format == "cell_csv" and self.expression_path is None:
            raise ValueError("cell_csv requires expression_path")
        if self.input_format in {"cell_csv", "sample_mean_csv"} and not self.gene_columns:
            raise ValueError("flat-file adapters require explicit gene_columns")
        counts = (self.minimum_cells_per_sample_state, self.minimum_reference_samples,
                  self.minimum_group_a_samples, self.minimum_group_b_samples,
                  self.minimum_datasets, self.minimum_training_samples_per_group,
                  self.minimum_successful_folds)
        if any(value < 1 for value in counts): raise ValueError("replication and cell thresholds must be positive")
        if self.pseudocount != 1.0: raise ValueError("notebook-15 contract requires pseudocount 1.0")
        if self.minimum_mean_expression < 0: raise ValueError("minimum_mean_expression must be nonnegative")
        if any(value <= 0 for value in (self.group_a_absolute_effect_cutoff, self.group_b_absolute_effect_cutoff,
                                        self.delta_absolute_cutoff)): raise ValueError("effect cutoffs must be positive")
        if any(not 0 < value <= 1 for value in (self.group_a_fdr_cutoff, self.group_b_fdr_cutoff,
                self.delta_fdr_cutoff, self.group_a_sign_consistency_cutoff,
                self.group_b_sign_consistency_cutoff, self.delta_sign_consistency_cutoff)):
            raise ValueError("FDR and sign-consistency cutoffs must be in (0,1]")

    def parameters(self) -> dict[str, object]:
        excluded = {"source_path", "expression_path"}
        result = {key: value for key, value in self.__dict__.items() if key not in excluded}
        for key in ("gene_columns", "requested_cell_states"): result[key] = list(result[key])
        result["model_formula"] = "log2(mean expression + 1) ~ 1 + I(group_a) + I(group_b)"
        result["delta_definition"] = "beta_group_a - beta_group_b"
        return result

@dataclass(frozen=True)
class AtlasLODOOutput:
    capability_id: str
    node_version: str
    status: CapabilityStatus
    cache_key: str
    cache_hit: bool
    artifact_paths: tuple[tuple[str, str, str], ...]
    provenance_path: str
    cache_manifest_path: str
    warnings: tuple[StructuredWarning, ...]
    provenance: AnalysisProvenance

    def __post_init__(self) -> None:
        if self.capability_id != "CAP-LODO-001": raise ValueError("unexpected AtlasLODO capability")
        if isinstance(self.status, str): object.__setattr__(self, "status", CapabilityStatus(self.status))

    def to_capability_result(self) -> CapabilityResult:
        artifacts = tuple(ArtifactReference(name, path, category, "text/csv")
                          for name, path, category in self.artifact_paths) + (
            ArtifactReference("provenance", self.provenance_path, ArtifactCategory.PROVENANCE, "application/json"),
            ArtifactReference("cache_manifest", self.cache_manifest_path, ArtifactCategory.MANIFEST, "application/json"),)
        return CapabilityResult(self.capability_id, self.node_version, self.status, self.cache_key,
            self.cache_hit, artifacts, self.warnings, self.provenance_path, self.cache_manifest_path)
