"""Typed contracts for CAP-TF-002 Signed TF Activity Inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .common import (AnalysisProvenance, ArtifactCategory, ArtifactReference,
                     CapabilityResult, CapabilityStatus, StructuredWarning)


@dataclass(frozen=True)
class TFActivityInput:
    signed_feature_program_path: Path
    dorothea_path: Path
    collectri_path: Path
    gene_column: str = "feature_id"
    signed_statistic_column: str = "signed_statistic"
    organism: Literal["human", "mouse"] = "human"
    feature_type: Literal["gene"] = "gene"
    statistic_orientation: Literal[
        "condition_a_minus_condition_b"
    ] = "condition_a_minus_condition_b"
    dorothea_confidence_levels: tuple[str, ...] = ("A", "B", "C")
    minimum_overlapping_targets: int = 5
    minimum_eligible_genes: int = 3
    fdr_cutoff: float = 0.10
    multiple_testing_method: Literal["benjamini-hochberg"] = "benjamini-hochberg"
    correction_family: Literal[
        "resource_x_feature_program"
    ] = "resource_x_feature_program"
    minimum_consensus_resources: int = 2
    ulm_implementation_mode: Literal[
        "documented_decoupler_v2_ulm_formula"
    ] = "documented_decoupler_v2_ulm_formula"

    def __post_init__(self) -> None:
        for name in ("signed_feature_program_path", "dorothea_path", "collectri_path"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))
        levels = tuple(str(value).upper() for value in self.dorothea_confidence_levels)
        if self.organism not in {"human", "mouse"}:
            raise ValueError("unsupported organism")
        if self.feature_type != "gene":
            raise ValueError("unsupported feature type")
        if self.statistic_orientation != "condition_a_minus_condition_b":
            raise ValueError("unsupported statistic orientation")
        if not levels or len(set(levels)) != len(levels):
            raise ValueError("DoRothEA confidence levels must be nonempty and unique")
        if any(level not in {"A", "B", "C", "D", "E"} for level in levels):
            raise ValueError("DoRothEA confidence levels must be values from A-E")
        object.__setattr__(self, "dorothea_confidence_levels", levels)
        if self.minimum_overlapping_targets < 1:
            raise ValueError("minimum_overlapping_targets must be positive")
        if self.minimum_eligible_genes < 3:
            raise ValueError("minimum_eligible_genes must be at least 3")
        if not 0 < self.fdr_cutoff <= 1:
            raise ValueError("fdr_cutoff must be in (0, 1]")
        if self.minimum_consensus_resources < 1:
            raise ValueError("minimum_consensus_resources must be positive")
        if not self.gene_column.strip() or not self.signed_statistic_column.strip():
            raise ValueError("input column names must be nonblank")

    def parameters(self) -> dict[str, object]:
        return {
            "gene_column": self.gene_column,
            "signed_statistic_column": self.signed_statistic_column,
            "organism": self.organism,
            "feature_type": self.feature_type,
            "statistic_orientation": self.statistic_orientation,
            "dorothea_confidence_levels": list(self.dorothea_confidence_levels),
            "minimum_overlapping_targets": self.minimum_overlapping_targets,
            "minimum_eligible_genes": self.minimum_eligible_genes,
            "fdr_cutoff": self.fdr_cutoff,
            "multiple_testing_method": self.multiple_testing_method,
            "correction_family": self.correction_family,
            "minimum_consensus_resources": self.minimum_consensus_resources,
            "ulm_implementation_mode": self.ulm_implementation_mode,
        }


@dataclass(frozen=True)
class TFActivityOutput:
    capability_id: str
    implementation_version: str
    node_version: str
    status: CapabilityStatus
    cache_key: str
    cache_hit: bool
    artifact_paths: tuple[tuple[str, str, str, str], ...]
    provenance_path: str
    cache_manifest_path: str
    warnings: tuple[StructuredWarning, ...]
    provenance: AnalysisProvenance

    def __post_init__(self) -> None:
        if self.capability_id != "CAP-TF-002":
            raise ValueError("unexpected TF activity capability")
        if isinstance(self.status, str):
            object.__setattr__(self, "status", CapabilityStatus(self.status))

    def to_capability_result(self) -> CapabilityResult:
        artifacts = tuple(
            ArtifactReference(name, path, category, media_type)
            for name, path, category, media_type in self.artifact_paths
        ) + (
            ArtifactReference("provenance", self.provenance_path,
                              ArtifactCategory.PROVENANCE, "application/json"),
            ArtifactReference("cache_manifest", self.cache_manifest_path,
                              ArtifactCategory.MANIFEST, "application/json"),
        )
        return CapabilityResult(
            self.capability_id, self.node_version, self.status, self.cache_key,
            self.cache_hit, artifacts, self.warnings, self.provenance_path,
            self.cache_manifest_path,
        )
