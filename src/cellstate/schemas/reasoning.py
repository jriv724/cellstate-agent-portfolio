"""Strict schemas for downstream OpenAI reasoning reports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


REASONING_SCHEMA_VERSION = "1.0.0"
CRITIC_PROMPT_VERSION = "1.1.0"
INTERPRETER_PROMPT_VERSION = "1.1.0"

AssessmentStatus = Literal["pass", "warning", "fail", "not_assessable"]
Confidence = Literal["high", "moderate", "low", "insufficient"]


class StrictReasoningModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceAssessment(StrictReasoningModel):
    status: AssessmentStatus
    score: int | None = Field(default=None, ge=0, le=10)
    summary: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class CriticReport(StrictReasoningModel):
    schema_version: Literal["1.0.0"] = REASONING_SCHEMA_VERSION
    evidence_bundle_id: str = Field(min_length=1)
    replication_assessment: EvidenceAssessment
    statistical_support_assessment: EvidenceAssessment
    confounding_assessment: EvidenceAssessment
    design_validity_assessment: EvidenceAssessment
    assumption_risk_assessment: EvidenceAssessment
    generalizability_assessment: EvidenceAssessment
    strengths: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommended_follow_up: list[str] = Field(default_factory=list)
    overall_confidence: Confidence
    overall_confidence_score: int | None = Field(default=None, ge=0, le=10)
    reasoning_summary: str = Field(min_length=1)
    created_at_utc: str = Field(min_length=1)


class InterpretationReport(StrictReasoningModel):
    schema_version: Literal["1.0.0"] = REASONING_SCHEMA_VERSION
    evidence_bundle_id: str = Field(min_length=1)
    observations: list[str] = Field(default_factory=list)
    biological_programs: list[str] = Field(default_factory=list)
    candidate_regulators: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    critic_limitations_referenced: list[str] = Field(default_factory=list)
    experimental_follow_up: list[str] = Field(default_factory=list)
    interpretation_confidence: Confidence
    summary: str = Field(min_length=1)
    created_at_utc: str = Field(min_length=1)

    @field_validator("hypotheses")
    @classmethod
    def hypotheses_are_explicitly_labeled(cls, values: list[str]) -> list[str]:
        if any(not value.strip().casefold().startswith("hypothesis:") for value in values):
            raise ValueError("every hypothesis must begin with 'Hypothesis:'")
        return values


class ScientificReport(StrictReasoningModel):
    schema_version: Literal["1.0.0"] = REASONING_SCHEMA_VERSION
    evidence_bundle_id: str = Field(min_length=1)
    deterministic_executive_summary: str = Field(min_length=1)
    critic_report: CriticReport
    interpretation_report: InterpretationReport
    provenance: dict[str, Any]
    created_at_utc: str = Field(min_length=1)
