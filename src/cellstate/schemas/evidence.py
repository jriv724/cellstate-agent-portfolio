"""Typed boundary between deterministic execution and downstream reasoning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


EVIDENCE_BUNDLE_SCHEMA_VERSION = "1.0.0"


class EvidenceExecutionStatus(str, Enum):
    COMPLETED = "completed"
    COMPLETED_FROM_CACHE = "completed_from_cache"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class EvidenceArtifact:
    """Reference to a deterministic artifact; never the artifact contents."""

    logical_name: str
    path: str
    category: str
    media_type: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.logical_name.strip() or not self.path.strip():
            raise ValueError("evidence artifact identity fields must be nonblank")


@dataclass(frozen=True)
class EvidenceWarning:
    code: str
    message: str
    severity: str = "warning"
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.message.strip():
            raise ValueError("evidence warning code and message must be nonblank")
        if self.severity not in {"info", "warning", "error"}:
            raise ValueError("invalid evidence warning severity")


@dataclass(frozen=True)
class EvidenceBundle:
    """Complete, JSON-safe evidence supplied to downstream AI agents."""

    bundle_id: str
    created_at_utc: str
    execution_status: EvidenceExecutionStatus
    analysis_question: str
    analysis_type: str
    biological_context: Mapping[str, Any]
    unit_of_inference: str
    deterministic_evidence: Mapping[str, Any]
    design_assessment: Mapping[str, Any]
    limitations: tuple[str, ...]
    warnings: tuple[EvidenceWarning, ...]
    artifacts: tuple[EvidenceArtifact, ...]
    provenance: Mapping[str, Any]
    cache: Mapping[str, Any]
    schema_version: str = EVIDENCE_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.execution_status, str):
            object.__setattr__(
                self, "execution_status", EvidenceExecutionStatus(self.execution_status)
            )
        for value, label in (
            (self.bundle_id, "bundle_id"),
            (self.created_at_utc, "created_at_utc"),
            (self.analysis_question, "analysis_question"),
            (self.analysis_type, "analysis_type"),
            (self.unit_of_inference, "unit_of_inference"),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be nonblank")
        if self.schema_version != EVIDENCE_BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported EvidenceBundle schema version")
        if not self.artifacts:
            raise ValueError("EvidenceBundle must reference deterministic artifacts")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["execution_status"] = self.execution_status.value
        result["limitations"] = list(self.limitations)
        result["warnings"] = [asdict(item) for item in self.warnings]
        result["artifacts"] = [asdict(item) for item in self.artifacts]
        return result
