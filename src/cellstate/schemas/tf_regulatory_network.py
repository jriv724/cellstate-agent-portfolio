"""Typed contracts for CAP-TF-001 Consensus TF Regulatory Network."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .common import (AnalysisProvenance, ArtifactCategory, ArtifactReference,
                     CapabilityResult, CapabilityStatus, StructuredWarning)


@dataclass(frozen=True)
class TFRegulatoryNetworkInput:
    feature_program_path: Path
    background_universe_path: Path
    dorothea_path: Path
    collectri_path: Path
    feature_column: str = "feature_id"
    background_feature_column: str = "feature_id"
    organism: Literal["human", "mouse"] = "human"
    feature_type: Literal["gene"] = "gene"
    dorothea_confidence_levels: tuple[str, ...] = ("A", "B", "C")
    minimum_regulon_target_count: int = 5
    fdr_cutoff: float = 0.10
    minimum_query_target_overlap: int = 2
    minimum_supporting_databases: int = 2
    multiple_testing_method: Literal["benjamini-hochberg"] = "benjamini-hochberg"
    correction_family: Literal["resource_x_feature_program"] = "resource_x_feature_program"
    background_source: str = "explicit tested feature universe"
    minimum_query_features: int = 1
    upstream_qc_status: str | None = None

    def __post_init__(self) -> None:
        for name in ("feature_program_path", "background_universe_path",
                     "dorothea_path", "collectri_path"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))
        levels = tuple(str(x).upper() for x in self.dorothea_confidence_levels)
        if self.organism not in {"human", "mouse"}:
            raise ValueError("unsupported organism")
        if self.feature_type != "gene":
            raise ValueError("unsupported feature type")
        if not levels or len(set(levels)) != len(levels) or any(x not in {"A", "B", "C", "D", "E"} for x in levels):
            raise ValueError("DoRothEA confidence levels must be unique values from A-E")
        object.__setattr__(self, "dorothea_confidence_levels", levels)
        if self.minimum_regulon_target_count < 1:
            raise ValueError("minimum_regulon_target_count must be positive")
        if self.minimum_query_target_overlap < 1 or self.minimum_query_features < 1:
            raise ValueError("minimum overlap and query feature thresholds must be positive")
        if self.minimum_supporting_databases < 1:
            raise ValueError("minimum_supporting_databases must be positive")
        if not 0 < self.fdr_cutoff <= 1:
            raise ValueError("fdr_cutoff must be in (0,1]")
        if not self.background_source.strip():
            raise ValueError("background_source must be explicit")

    def parameters(self) -> dict[str, object]:
        return {
            "organism": self.organism,
            "feature_type": self.feature_type,
            "dorothea_confidence_levels": list(self.dorothea_confidence_levels),
            "minimum_regulon_target_count": self.minimum_regulon_target_count,
            "fdr_cutoff": self.fdr_cutoff,
            "minimum_query_target_overlap": self.minimum_query_target_overlap,
            "minimum_supporting_databases": self.minimum_supporting_databases,
            "multiple_testing_method": self.multiple_testing_method,
            "correction_family": self.correction_family,
            "background_source": self.background_source,
            "minimum_query_features": self.minimum_query_features,
            "upstream_qc_status": self.upstream_qc_status,
            "feature_column": self.feature_column,
            "background_feature_column": self.background_feature_column,
        }


@dataclass(frozen=True)
class TFRegulatoryNetworkOutput:
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
        if self.capability_id != "CAP-TF-001":
            raise ValueError("unexpected TF regulatory network capability")
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
