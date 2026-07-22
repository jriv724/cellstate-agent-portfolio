"""Contracts for testing predicted immune-age outcomes across groups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .common import (AnalysisProvenance, ArtifactCategory, ArtifactReference,
                     CapabilityResult, StructuredWarning, capability_status_from_warnings)


@dataclass(frozen=True)
class GroupAgeAssociationInput:
    """Request to compare a predicted immune-age outcome across disease or response groups."""
    table_path: Path
    input_format: Literal["csv", "tsv"] = "tsv"
    independent_id_column: str = "sample"
    class_column: str = "cell_type"
    outcome_column: str = "mean_bm_retrained_age"
    group_column: str = "stage_model"
    dataset_column: str | None = None
    group_levels: tuple[str, ...] = ("NBM", "MGUS", "SMM", "NDMM", "RRMM")
    class_levels: tuple[str, ...] = ("B", "CD4T", "CD8T", "MONO", "NK")
    contrasts: tuple[tuple[str, str], ...] = ()
    p_adjust_method: Literal["BH"] = "BH"
    minimum_replicates_per_group: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.table_path, Path):
            object.__setattr__(self, "table_path", Path(self.table_path))
        if self.input_format not in {"csv", "tsv"}:
            raise ValueError("input_format must be csv or tsv")
        if self.p_adjust_method != "BH":
            raise ValueError("source supports BH adjustment only")
        if len(self.group_levels) < 2 or len(set(self.group_levels)) != len(self.group_levels):
            raise ValueError("group_levels must contain at least two unique ordered levels")
        if not self.class_levels or len(set(self.class_levels)) != len(self.class_levels):
            raise ValueError("class_levels must contain unique levels")
        for first, second in self.resolved_contrasts():
            if first == second or first not in self.group_levels or second not in self.group_levels:
                raise ValueError("contrasts must contain distinct declared group levels")
        if self.minimum_replicates_per_group < 1:
            raise ValueError("minimum_replicates_per_group must be >= 1")

    def resolved_contrasts(self) -> tuple[tuple[str, str], ...]:
        if self.contrasts:
            return self.contrasts
        return tuple(
            (self.group_levels[i], self.group_levels[j])
            for i in range(len(self.group_levels))
            for j in range(i + 1, len(self.group_levels))
        )

    def parameters(self) -> dict[str, object]:
        return {
            "input_format": self.input_format,
            "independent_id_column": self.independent_id_column,
            "class_column": self.class_column,
            "outcome_column": self.outcome_column,
            "group_column": self.group_column,
            "dataset_column": self.dataset_column,
            "group_levels": list(self.group_levels),
            "class_levels": list(self.class_levels),
            "contrasts": [list(value) for value in self.resolved_contrasts()],
            "p_adjust_method": self.p_adjust_method,
            "minimum_replicates_per_group": self.minimum_replicates_per_group,
        }


@dataclass(frozen=True)
class OrderedAgeAssociationInput:
    """Request to associate a predicted immune-age outcome with ordered response groups."""
    table_path: Path
    input_format: Literal["csv", "tsv"] = "tsv"
    independent_id_column: str = "patient_id"
    class_column: str = "cell_type"
    outcome_column: str = "mean_bm_retrained_age"
    group_column: str = "response_group"
    score_mapping: tuple[tuple[str, float], ...] = (("CR", 1.0), ("NR", 2.0), ("ER", 3.0))
    class_levels: tuple[str, ...] = ("B", "CD4T", "CD8T", "MONO", "NK")
    p_adjust_method: Literal["BH"] = "BH"
    minimum_represented_levels: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.table_path, Path):
            object.__setattr__(self, "table_path", Path(self.table_path))
        if self.input_format not in {"csv", "tsv"}:
            raise ValueError("input_format must be csv or tsv")
        labels = [label for label, _ in self.score_mapping]
        scores = [score for _, score in self.score_mapping]
        if len(labels) < 2 or len(set(labels)) != len(labels) or len(set(scores)) != len(scores):
            raise ValueError("score_mapping must map unique labels one-to-one to unique scores")
        if self.minimum_represented_levels < 2:
            raise ValueError("minimum_represented_levels must be >= 2")
        if self.p_adjust_method != "BH":
            raise ValueError("source supports BH adjustment only")

    def parameters(self) -> dict[str, object]:
        return {
            "input_format": self.input_format,
            "independent_id_column": self.independent_id_column,
            "class_column": self.class_column,
            "outcome_column": self.outcome_column,
            "group_column": self.group_column,
            "score_mapping": [list(value) for value in self.score_mapping],
            "class_levels": list(self.class_levels),
            "p_adjust_method": self.p_adjust_method,
            "minimum_represented_levels": self.minimum_represented_levels,
        }


@dataclass(frozen=True)
class AgeAssociationOutput:
    """Files, warnings, cache identity, and provenance for predicted immune-age tests."""
    capability_id: str
    node_version: str
    cache_key: str
    cache_hit: bool
    analysis_table_path: str
    statistics_paths: tuple[str, ...]
    provenance_path: str
    cache_manifest_path: str
    warnings: tuple[StructuredWarning, ...]
    provenance: AnalysisProvenance

    def __post_init__(self) -> None:
        if self.capability_id not in {"CAP-STAT-001", "CAP-STAT-002"}:
            raise ValueError("unexpected capability_id")

    def to_capability_result(self) -> CapabilityResult:
        artifacts = [ArtifactReference("analysis_table", self.analysis_table_path,
                     ArtifactCategory.INPUT_DERIVED, "text/csv")]
        artifacts.extend(ArtifactReference(f"statistics_{index + 1}", path,
                         ArtifactCategory.INFERENTIAL, "text/csv")
                         for index, path in enumerate(self.statistics_paths))
        artifacts.extend((
            ArtifactReference("provenance", self.provenance_path, ArtifactCategory.PROVENANCE, "application/json"),
            ArtifactReference("cache_manifest", self.cache_manifest_path, ArtifactCategory.MANIFEST, "application/json"),
        ))
        return CapabilityResult(self.capability_id, self.node_version,
            capability_status_from_warnings(self.warnings), self.cache_key, self.cache_hit,
            tuple(artifacts), self.warnings, self.provenance_path, self.cache_manifest_path)
