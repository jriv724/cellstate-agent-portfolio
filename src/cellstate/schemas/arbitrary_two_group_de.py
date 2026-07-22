"""Strict contracts for CAP-DESEQ-003 arbitrary two-group pseudobulk DE."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .common import (
    ArtifactReference,
    CapabilityResult,
    CapabilityStatus,
    StructuredWarning,
    WarningSeverity,
)

CAP_DESEQ_003_SCHEMA_VERSION = "1.1.0"
CAP_DESEQ_003_VERSION = "1.1.1"
DETerminalStatus = Literal[
    "completed", "completed_with_warnings", "blocked",
    "insufficient_robustness", "failed",
]


class StrictDEModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArbitraryTwoGroupDEInput(StrictDEModel):
    count_matrix_path: Path
    sample_metadata_path: Path
    group_a: str = Field(min_length=1)
    group_b: str = Field(min_length=1)
    replicate_column: str = Field(default="sample", min_length=1)
    group_column: str = Field(default="group", min_length=1)
    dataset_column: str = Field(default="dataset", min_length=1)
    cell_count_column: str = Field(default="n_cells", min_length=1)
    patient_column: str | None = None
    minimum_cells_per_replicate: Literal[100] = 100
    minimum_replicates_per_group: Literal[3] = 3
    gene_minimum_count: Literal[10] = 10
    gene_minimum_replicates: Literal[3] = 3
    alpha: Literal[0.05] = 0.05
    independent_filtering: Literal[True] = True
    confounded_design_policy: Literal["block", "exploratory_lodo"] = "block"
    lodo_min_estimable_folds: int = Field(default=3, ge=1)
    lodo_min_direction_fraction: float = Field(default=0.80, ge=0, le=1)
    lodo_min_median_abs_log2fc: float = Field(default=0.25, ge=0)
    lodo_full_analysis_fdr: float = Field(default=0.05, gt=0, le=1)
    lodo_require_two_datasets_per_group: bool = True
    lodo_max_opposite_log2fc: float = Field(default=0.0, ge=0)
    rscript_path: Path = Path("/usr/local/bin/Rscript")
    output_directory: Path | None = None
    upstream_capability_id: str = "CAP-DESEQ-001"
    upstream_cache_key: str | None = None
    upstream_provenance_path: Path | None = None
    schema_version: Literal["1.1.0"] = CAP_DESEQ_003_SCHEMA_VERSION

    @field_validator("group_a", "group_b")
    @classmethod
    def strip_groups(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("comparison groups must be nonblank")
        return stripped

    @model_validator(mode="after")
    def validate_contract(self) -> "ArbitraryTwoGroupDEInput":
        if self.group_a == self.group_b:
            raise ValueError("group_a and group_b must be distinct")
        if self.patient_column is not None and not self.patient_column.strip():
            raise ValueError("patient_column must be nonblank when supplied")
        if self.upstream_capability_id != "CAP-DESEQ-001":
            raise ValueError("CAP-DESEQ-003 requires CAP-DESEQ-001 pseudobulk input")
        return self

    def parameters(self) -> dict[str, Any]:
        return {
            "groups": [self.group_a, self.group_b],
            "comparison_direction": "group_a_minus_group_b",
            "replicate_column": self.replicate_column,
            "group_column": self.group_column,
            "dataset_column": self.dataset_column,
            "cell_count_column": self.cell_count_column,
            "patient_column": self.patient_column,
            "minimum_cells_per_replicate": self.minimum_cells_per_replicate,
            "minimum_replicates_per_group": self.minimum_replicates_per_group,
            "gene_minimum_count": self.gene_minimum_count,
            "gene_minimum_replicates": self.gene_minimum_replicates,
            "alpha": self.alpha,
            "independent_filtering": self.independent_filtering,
            "confounded_design_policy": self.confounded_design_policy,
            "lodo_min_estimable_folds": self.lodo_min_estimable_folds,
            "lodo_min_direction_fraction": self.lodo_min_direction_fraction,
            "lodo_min_median_abs_log2fc": self.lodo_min_median_abs_log2fc,
            "lodo_full_analysis_fdr": self.lodo_full_analysis_fdr,
            "lodo_require_two_datasets_per_group": self.lodo_require_two_datasets_per_group,
            "lodo_max_opposite_log2fc": self.lodo_max_opposite_log2fc,
            "rscript_path": str(self.rscript_path),
            "upstream_capability_id": self.upstream_capability_id,
            "upstream_cache_key": self.upstream_cache_key,
            "schema_version": self.schema_version,
            "design_policy": (
                "~ group for one dataset; ~ dataset + group for multiple "
                "datasets only when overlapping and full rank; explicit "
                "exploratory_lodo may use ~ group when adjusted inference is confounded"
            ),
        }


class DEWarning(StrictDEModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: Literal["info", "warning", "error"] = "warning"
    context: dict[str, Any] = Field(default_factory=dict)


class TwoGroupDesignAssessment(StrictDEModel):
    group_replicate_counts: dict[str, int]
    represented_datasets: list[str]
    shared_datasets: list[str]
    design_formula: Literal["~ group", "~ dataset + group"] | None
    design_columns: list[str]
    design_rank: int | None
    design_column_count: int | None
    residual_degrees_of_freedom: int | None
    full_rank: bool
    group_coefficient: str
    estimable: bool
    warnings: list[DEWarning] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)


class DEArtifactReference(StrictDEModel):
    logical_name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    category: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    required: bool = True


class ArbitraryTwoGroupDEOutput(StrictDEModel):
    capability_id: Literal["CAP-DESEQ-003"] = "CAP-DESEQ-003"
    capability_version: Literal["1.1.1"] = CAP_DESEQ_003_VERSION
    schema_version: Literal["1.1.0"] = CAP_DESEQ_003_SCHEMA_VERSION
    terminal_status: DETerminalStatus
    cache_key: str = Field(min_length=1)
    cache_hit: bool = False
    comparison_direction: str = Field(min_length=1)
    evidence_class: Literal[
        "adjusted_inference", "exploratory_unadjusted",
        "exploratory_lodo_conserved",
    ]
    design_assessment: TwoGroupDesignAssessment
    input_gene_count: int = Field(ge=0)
    retained_gene_count: int = Field(ge=0)
    tested_gene_count: int = Field(ge=0)
    significant_gene_count: int = Field(ge=0)
    upregulated_in_group_a_count: int = Field(ge=0)
    upregulated_in_group_b_count: int = Field(ge=0)
    group_replicate_counts: dict[str, int]
    artifacts: list[DEArtifactReference]
    warnings: list[DEWarning] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    provenance_path: str = Field(min_length=1)
    cache_manifest_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_terminal_semantics(self) -> "ArbitraryTwoGroupDEOutput":
        names = {artifact.logical_name for artifact in self.artifacts}
        warning_codes = {warning.code for warning in self.warnings}
        if self.evidence_class != "adjusted_inference" and (
            "EXPLORATORY_CONFOUNDED_DESIGN" not in warning_codes
        ):
            raise ValueError("exploratory evidence requires the confounded-design warning")
        if self.terminal_status in {"completed", "completed_with_warnings"}:
            required_result = (
                "conserved_features" if self.evidence_class == "exploratory_lodo_conserved"
                else "deseq2_results"
            )
            if required_result not in names:
                raise ValueError(f"completed output requires {required_result}")
            if self.tested_gene_count != self.retained_gene_count:
                raise ValueError("every retained gene must appear in complete results")
        elif "deseq2_results" in names or "conserved_features" in names:
            raise ValueError("blocked or failed output cannot claim a DE result table")
        if self.terminal_status == "blocked" and not self.blocking_reasons:
            raise ValueError("blocked output requires blocking reasons")
        if self.cache_hit and self.terminal_status not in {
            "completed", "completed_with_warnings"
        }:
            raise ValueError("blocked or failed output cannot be a cache hit")
        return self

    def to_capability_result(self) -> CapabilityResult:
        status = {
            "completed": CapabilityStatus.COMPLETED,
            "completed_with_warnings": CapabilityStatus.COMPLETED_WITH_WARNINGS,
            "blocked": CapabilityStatus.BLOCKED_SCIENTIFIC_DECISION,
            "insufficient_robustness": CapabilityStatus.NOT_ESTIMABLE,
            "failed": CapabilityStatus.FAILED_EXECUTION,
        }[self.terminal_status]
        artifacts = tuple(
            ArtifactReference(
                artifact.logical_name,
                artifact.path,
                artifact.category,
                artifact.media_type,
                artifact.required,
            )
            for artifact in self.artifacts
        )
        warnings = tuple(
            StructuredWarning(
                warning.code,
                warning.message,
                WarningSeverity(warning.severity),
                warning.context,
            )
            for warning in self.warnings
        )
        return CapabilityResult(
            self.capability_id,
            self.capability_version,
            status,
            self.cache_key,
            self.cache_hit,
            artifacts,
            warnings,
            self.provenance_path,
            self.cache_manifest_path,
        )
