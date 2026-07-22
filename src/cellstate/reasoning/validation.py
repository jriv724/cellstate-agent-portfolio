"""Cross-report validation using only serialized EvidenceBundle content."""

from __future__ import annotations

from typing import Any, Mapping

from cellstate.schemas.evidence import EvidenceBundle, EvidenceExecutionStatus
from cellstate.schemas.reasoning import CriticReport, InterpretationReport

from .exceptions import ReasoningValidationError


CONFIDENCE_ORDER = {
    "insufficient": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
}


def serialized_bundle(bundle: EvidenceBundle) -> dict[str, Any]:
    return bundle.to_dict()


def _paths(value: Any, prefix: str = "") -> set[str]:
    result: set[str] = set()
    if prefix:
        result.add(prefix)
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(_paths(item, f"{prefix}[{index}]"))
    return result


def valid_evidence_references(payload: dict[str, Any]) -> set[str]:
    from pathlib import Path

    references = _paths(payload)

    for warning in payload.get("warnings", []):
        code = warning["code"]
        references.add(f"warnings.{code}")
        references.add(f"warnings:{code}")

    for artifact in payload.get("artifacts", []):
        logical_name = artifact["logical_name"]

        references.add(logical_name)
        references.add(f"artifacts.{logical_name}")

        if ":" in logical_name and artifact.get("path"):
            capability = logical_name.split(":", 1)[0]
            filename = Path(artifact["path"]).name
            references.add(f"{capability}:{filename}")
            references.add(f"{capability}:{Path(filename).stem}")
            references.add(f"{logical_name}{Path(filename).suffix}")

    # Add explicit warning-code paths at every level of the bundle.
    def add_warning_references(value: Any, prefix: str = "") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)

                if key == "warnings" and isinstance(item, list):
                    for index, warning in enumerate(item):
                        code = (
                            warning.get("code")
                            if isinstance(warning, Mapping)
                            else warning
                        )
                        if code:
                            references.add(f"{child}.{code}")
                            references.add(f"{child}:{code}")
                            references.add(f"{child}[{index}].{code}")
                            references.add(f"{child}[{index}]:{code}")

                add_warning_references(item, child)

        elif isinstance(value, list):
            for index, item in enumerate(value):
                add_warning_references(item, f"{prefix}[{index}]")

    add_warning_references(payload)

    # Ground critic shorthand against canonical EvidenceBundle paths.
    capability_prefix = "deterministic_evidence.capabilities."

    for reference in tuple(references):
        # Accept misplaced deterministic_evidence prefixes only when the
        # corresponding canonical top-level evidence path really exists.
        if (
            reference == "unit_of_inference"
            or reference.startswith("design_assessment.")
        ):
            references.add(f"deterministic_evidence.{reference}")

        if reference.startswith(capability_prefix):
            remainder = reference[len(capability_prefix):]
            capability, separator, field = remainder.partition(".")

            if separator and field:
                references.add(f"{capability}:{field}")
                references.add(
                    f"{capability}:{field.replace('.', ':')}"
                )

        if reference.startswith("design_assessment."):
            field = reference[len("design_assessment."):]
            references.add(f"design_assessment:{field}")
            references.add(
                f"design_assessment:{field.replace('.', ':')}"
            )

    return references


def validate_critic(bundle: EvidenceBundle, critic: CriticReport) -> None:
    if critic.evidence_bundle_id != bundle.bundle_id:
        raise ReasoningValidationError("CriticReport EvidenceBundle identity mismatch.")
    valid_refs = valid_evidence_references(serialized_bundle(bundle))
    assessments = (
        critic.replication_assessment,
        critic.statistical_support_assessment,
        critic.confounding_assessment,
        critic.design_validity_assessment,
        critic.assumption_risk_assessment,
        critic.generalizability_assessment,
    )
    invalid = sorted(
        {
            reference
            for assessment in assessments
            for reference in assessment.evidence_refs
            if reference not in valid_refs
        }
    )
    if invalid:
        raise ReasoningValidationError(
            "CriticReport contains unknown evidence references: " + ", ".join(invalid)
        )


def has_biological_result_evidence(bundle: EvidenceBundle) -> bool:
    if any(artifact.category == "inferential" for artifact in bundle.artifacts):
        return True
    result_keys = {
        "effect_estimates",
        "differential_expression",
        "significant_features",
        "enrichment",
        "tf_activity",
        "regulatory_network",
        "biological_results",
    }
    return bool(result_keys.intersection(bundle.deterministic_evidence))


def requires_insufficient_interpretation(bundle: EvidenceBundle) -> bool:
    return (
        bundle.execution_status is EvidenceExecutionStatus.BLOCKED
        or not has_biological_result_evidence(bundle)
    )


def validate_interpretation(
    bundle: EvidenceBundle,
    critic: CriticReport,
    interpretation: InterpretationReport,
) -> None:
    if critic.evidence_bundle_id != bundle.bundle_id:
        raise ReasoningValidationError("CriticReport EvidenceBundle identity mismatch.")
    if interpretation.evidence_bundle_id != bundle.bundle_id:
        raise ReasoningValidationError(
            "InterpretationReport EvidenceBundle identity mismatch."
        )
    if (
        CONFIDENCE_ORDER[interpretation.interpretation_confidence]
        > CONFIDENCE_ORDER[critic.overall_confidence]
    ):
        raise ReasoningValidationError(
            "Interpretation confidence cannot exceed Critic confidence."
        )
    unknown_limitations = set(interpretation.critic_limitations_referenced).difference(
        critic.limitations
    )
    if unknown_limitations:
        raise ReasoningValidationError(
            "Interpreter referenced limitations not present in CriticReport."
        )
    if critic.limitations and not interpretation.critic_limitations_referenced:
        raise ReasoningValidationError(
            "Interpreter must reference meaningful Critic limitations."
        )
    if requires_insufficient_interpretation(bundle):
        if interpretation.interpretation_confidence != "insufficient":
            raise ReasoningValidationError(
                "Blocked or evidence-insufficient bundles require insufficient confidence."
            )
        biological_claims = (
            interpretation.observations
            or interpretation.biological_programs
            or interpretation.candidate_regulators
            or interpretation.hypotheses
        )
        if biological_claims:
            raise ReasoningValidationError(
                "Blocked or evidence-insufficient bundles cannot contain biological claims."
            )
