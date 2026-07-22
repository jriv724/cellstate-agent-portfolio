"""Dependency-free typed common schemas.

Pydantic is not installed in the source environment. These frozen dataclasses
validate at construction and expose JSON-compatible dictionaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class WarningSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class CapabilityStatus(str, Enum):
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    NOT_ESTIMABLE = "not_estimable"
    INVALID_INPUT = "invalid_input"
    INSUFFICIENT_REPLICATION = "insufficient_replication"
    BLOCKED_SCIENTIFIC_DECISION = "blocked_scientific_decision"
    FAILED_EXECUTION = "failed_execution"


class ArtifactCategory(str, Enum):
    INPUT_DERIVED = "input-derived"
    DESCRIPTIVE = "descriptive"
    INFERENTIAL = "inferential"
    QC = "QC"
    MODEL = "model"
    PROVENANCE = "provenance"
    MANIFEST = "manifest"
    OPTIONAL = "optional"


@dataclass(frozen=True)
class StructuredWarning:
    code: str
    message: str
    severity: WarningSeverity = WarningSeverity.WARNING
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.severity, str):
            object.__setattr__(self, "severity", WarningSeverity(self.severity))
        if not self.code.strip() or not self.message.strip():
            raise ValueError("warning code and message must be nonblank")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["severity"] = self.severity.value
        return value


@dataclass(frozen=True)
class ArtifactReference:
    logical_name: str
    path: str
    category: ArtifactCategory
    media_type: str | None = None
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.category, str):
            object.__setattr__(self, "category", ArtifactCategory(self.category))
        if not self.logical_name.strip() or not self.path.strip():
            raise ValueError("artifact logical_name and path must be nonblank")


_STATUS_BY_WARNING_CODE = {
    "NON_ESTIMABLE_ADJUSTED_DESIGN": CapabilityStatus.NOT_ESTIMABLE,
    "INSUFFICIENT_REPLICATION": CapabilityStatus.INSUFFICIENT_REPLICATION,
    "INSUFFICIENT_BIOLOGICAL_REPLICATION": CapabilityStatus.INSUFFICIENT_REPLICATION,
    "INSUFFICIENT_GROUPS": CapabilityStatus.INSUFFICIENT_REPLICATION,
    "INSUFFICIENT_PAIRED_REPLICATION": CapabilityStatus.INSUFFICIENT_REPLICATION,
    "INSUFFICIENT_LONGITUDINAL_REPLICATION": CapabilityStatus.INSUFFICIENT_REPLICATION,
    "BLOCKED_SCIENTIFIC_DECISION": CapabilityStatus.BLOCKED_SCIENTIFIC_DECISION,
    "DESEQ2_RUNTIME_FAILURE": CapabilityStatus.FAILED_EXECUTION,
    "DESEQ2_OUTPUT_INCOMPLETE": CapabilityStatus.FAILED_EXECUTION,
}


def capability_status_from_warnings(
    warnings: tuple[StructuredWarning, ...], *, execution_succeeded: bool = True,
    outcome: CapabilityStatus | None = None,
) -> CapabilityStatus:
    """Map known outcomes explicitly; ordinary warnings remain successful."""
    if outcome is not None:
        return CapabilityStatus(outcome)
    if not execution_succeeded:
        return CapabilityStatus.FAILED_EXECUTION
    statuses = {_STATUS_BY_WARNING_CODE.get(item.code) for item in warnings}
    for status in (
        CapabilityStatus.FAILED_EXECUTION, CapabilityStatus.BLOCKED_SCIENTIFIC_DECISION,
        CapabilityStatus.NOT_ESTIMABLE, CapabilityStatus.INSUFFICIENT_REPLICATION,
    ):
        if status in statuses:
            return status
    return CapabilityStatus.COMPLETED_WITH_WARNINGS if warnings else CapabilityStatus.COMPLETED


@dataclass(frozen=True)
class CapabilityResult:
    capability_id: str
    node_version: str
    status: CapabilityStatus
    cache_key: str
    cache_hit: bool
    artifacts: tuple[ArtifactReference, ...]
    warnings: tuple[StructuredWarning, ...]
    provenance_path: str
    cache_manifest_path: str

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            object.__setattr__(self, "status", CapabilityStatus(self.status))
        if not self.capability_id.strip() or not self.node_version.strip() or not self.cache_key.strip():
            raise ValueError("capability result identity fields must be nonblank")
        if self.cache_hit and self.status not in {CapabilityStatus.COMPLETED, CapabilityStatus.COMPLETED_WITH_WARNINGS}:
            raise ValueError("failed or blocked capability results cannot be cache hits")
        self.validate_required_artifacts()

    def validate_required_artifacts(self, *, require_exists: bool = False) -> None:
        required = [artifact for artifact in self.artifacts if artifact.required]
        if not required:
            raise ValueError("capability result must declare at least one required artifact")
        if require_exists:
            missing = [artifact.path for artifact in required if not Path(artifact.path).exists()]
            if missing:
                raise ValueError(f"required artifacts do not exist: {missing}")


@dataclass(frozen=True)
class ResourceRequirements:
    cpu_cores: int = 1
    memory_gb: float = 1.0
    requires_r: bool = False
    requires_gpu: bool = False
    external_packages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cpu_cores < 1 or self.memory_gb <= 0:
            raise ValueError("resource requirements must be positive")


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str
    name: str
    scientific_question: str
    analysis_class: str
    input_schema: str
    output_schema: str
    unit_of_inference: str
    required_metadata_columns: tuple[str, ...]
    accepted_data_representation: tuple[str, ...]
    inferential: bool
    exploratory: bool
    parallelization_axis: str | None
    resources: ResourceRequirements
    upstream_capability_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required = (self.capability_id, self.name, self.scientific_question, self.analysis_class,
                    self.input_schema, self.output_schema, self.unit_of_inference)
        if any(not value.strip() for value in required):
            raise ValueError("capability specification fields must be nonblank")
        if not self.accepted_data_representation:
            raise ValueError("accepted_data_representation must not be empty")


@dataclass(frozen=True)
class AnalysisProvenance:
    capability_id: str
    node_version: str
    cache_schema_version: int
    source_files: tuple[str, ...]
    source_locations: tuple[str, ...]
    input_dataset_signature: str
    parameters: Mapping[str, Any]
    model_formula: str | None
    reference_group: str | None
    covariates: tuple[str, ...]
    unit_of_inference: str
    random_seed: int | None
    software_versions: Mapping[str, str]
    output_paths: tuple[str, ...]
    warnings: tuple[StructuredWarning, ...]
    execution_timestamp_utc: str

    def __post_init__(self) -> None:
        required = {
            "capability_id": self.capability_id,
            "node_version": self.node_version,
            "input_dataset_signature": self.input_dataset_signature,
            "unit_of_inference": self.unit_of_inference,
            "execution_timestamp_utc": self.execution_timestamp_utc,
        }
        blank = [name for name, value in required.items() if not value.strip()]
        if blank:
            raise ValueError(f"provenance fields must be nonblank: {blank}")
        if self.cache_schema_version < 1:
            raise ValueError("cache_schema_version must be >= 1")
        if not self.source_files:
            raise ValueError("source_files must not be empty")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["warnings"] = [warning.to_dict() for warning in self.warnings]
        return value
