"""Typed application models; scientific contracts remain in capability schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .labels import disease_contrast_label, disease_group_label


@dataclass(frozen=True)
class AtlasSummary:
    path: Path
    identity: str
    n_cells: int
    n_genes: int
    metadata_columns: tuple[str, ...]
    groups: tuple[str, ...]
    cell_states: tuple[str, ...]
    sample_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    cell_counts: dict[tuple[str, str], int] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisPlan:
    question: str
    cell_state: str
    group_a: str
    group_b: str
    requested_capabilities: tuple[str, ...]
    reasoning_requested: bool = True
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    feature_program_path: Path | None = None
    background_path: Path | None = None
    confounded_design_policy: str = "block"
    lodo_min_estimable_folds: int = 3
    lodo_min_direction_fraction: float = 0.80
    lodo_min_median_abs_log2fc: float = 0.25
    lodo_full_analysis_fdr: float = 0.05
    lodo_require_two_datasets_per_group: bool = True
    lodo_max_opposite_log2fc: float = 0.0

    def __post_init__(self) -> None:
        if self.confounded_design_policy not in {"block", "exploratory_lodo"}:
            raise ValueError("invalid confounded_design_policy")

    @property
    def contrast(self) -> str:
        return f"{self.group_a} − {self.group_b}"

    @property
    def effect_direction(self) -> str:
        return f"Positive values indicate higher expression in {self.group_a}"

    @property
    def display_contrast(self) -> str:
        return disease_contrast_label(self.group_a, self.group_b)

    @property
    def display_effect_direction(self) -> str:
        return f"Positive values indicate higher expression in {disease_group_label(self.group_a)}"


@dataclass
class ApplicationRunResult:
    plan: AnalysisPlan
    run_dir: Path
    overall_status: str
    elapsed_seconds: float
    adapter: Any | None = None
    de: Any | None = None
    tf: Any | None = None
    tf_status: str = "not_requested"
    tf_message: str = ""
    tf_resources: tuple[str, ...] = ()
    tf_summary: dict[str, Any] = field(default_factory=dict)
    cap_tf_001_status: str = "not_requested"
    evidence_bundle: Any | None = None
    evidence_bundle_path: Path | None = None
    reasoning: Any | None = None
    reasoning_error: str | None = None
