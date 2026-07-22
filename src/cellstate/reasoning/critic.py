"""OpenAI Critic constrained to evidence-quality review."""

from __future__ import annotations

from cellstate.schemas.evidence import EvidenceBundle
from cellstate.schemas.reasoning import CriticReport

from .openai_client import OpenAIReasoningClient
from .retry import generate_validated_report
from .validation import (
    serialized_bundle,
    valid_evidence_references,
    validate_critic,
)


CRITIC_SYSTEM_PROMPT = """
You are a skeptical computational-biology methods reviewer.
Evaluate only replication, statistical support, confounding, design validity,
assumptions, generalizability, evidence quality, limitations, confidence, and
recommended follow-up. Mark absent evidence not_assessable. Use only the
serialized EvidenceBundle supplied by the user and cite identifiable bundle
fields, warning codes, artifact logical names, or provenance fields.
Do not interpret disease biology, pathways, programs, regulators, mechanisms,
therapeutic targets, or biological conclusions. Do not provide chain-of-thought.
For exploratory_unadjusted or exploratory_lodo_conserved evidence, retain the
confounded-design warning verbatim in the review and never describe it as
dataset-adjusted inference.
In user-facing prose, use EvidenceBundle display group labels as the primary
labels. Do not replace or reinterpret canonical IDs in structured evidence.
The user payload contains allowed_evidence_references. Every value placed
in an evidence_refs array MUST be copied verbatim from that list. Never invent,
abbreviate, annotate, combine, or append values to an evidence reference.
Return only the required CriticReport schema.
""".strip()


def run_critic(
    bundle: EvidenceBundle,
    *,
    client: OpenAIReasoningClient,
) -> CriticReport:
    bundle_payload = serialized_bundle(bundle)
    allowed_references = sorted(
        valid_evidence_references(bundle_payload)
    )

    return generate_validated_report(
        client=client,
        system_prompt=CRITIC_SYSTEM_PROMPT,
        base_payload={"evidence_bundle": bundle_payload},
        response_model=CriticReport,
        validate=lambda report: validate_critic(bundle, report),
        allowed_catalog_name="allowed_evidence_references",
        allowed_catalog=allowed_references,
        repair_instruction=(
            "Regenerate the entire CriticReport. Every evidence_refs value "
            "must be copied verbatim from allowed_evidence_references."
        ),
    )
