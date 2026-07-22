"""OpenAI Interpreter constrained by deterministic evidence and Critic review."""

from __future__ import annotations

from cellstate.schemas.evidence import EvidenceBundle
from cellstate.schemas.reasoning import CriticReport, InterpretationReport

from .openai_client import OpenAIReasoningClient
from .retry import generate_validated_report
from .validation import serialized_bundle, validate_critic, validate_interpretation


INTERPRETER_SYSTEM_PROMPT = """
You are a translational computational biologist interpreting validated evidence.
Use only the serialized EvidenceBundle and validated CriticReport supplied.
Separate direct observations from explicitly labeled hypotheses. Reference
important Critic limitations, avoid causal overstatement, never contradict
deterministic evidence, and never invent significance, directionality,
enrichment, pathways, regulators, mechanisms, or findings.
You may discuss explicitly supplied exploratory_lodo_conserved features, but
must retain the confounded-design warning and must never call them
dataset-adjusted inference.
In user-facing prose, use EvidenceBundle display group labels as the primary
labels while preserving canonical IDs in cited structured evidence.
If execution was blocked or the bundle contains only QC, design, or pseudobulk
aggregation information, return insufficient confidence and no observations,
biological programs, candidate regulators, or hypotheses. Experimental follow-up
may describe missing data or analyses. Do not provide chain-of-thought.
The user payload contains allowed_critic_limitations. Every value placed
in critic_limitations_referenced MUST be copied verbatim from that list.
Never paraphrase, combine, annotate, shorten, or invent a limitation.
Return only the required InterpretationReport schema.
""".strip()


def run_interpreter(
    bundle: EvidenceBundle,
    critic: CriticReport,
    *,
    client: OpenAIReasoningClient,
) -> InterpretationReport:
    validate_critic(bundle, critic)

    return generate_validated_report(
        client=client,
        system_prompt=INTERPRETER_SYSTEM_PROMPT,
        base_payload={
            "evidence_bundle": serialized_bundle(bundle),
            "critic_report": critic.model_dump(mode="json"),
        },
        response_model=InterpretationReport,
        validate=lambda report: validate_interpretation(bundle, critic, report),
        allowed_catalog_name="allowed_critic_limitations",
        allowed_catalog=list(critic.limitations),
        repair_instruction=(
            "Regenerate the entire InterpretationReport. Every "
            "critic_limitations_referenced value must be copied verbatim "
            "from allowed_critic_limitations."
        ),
    )
