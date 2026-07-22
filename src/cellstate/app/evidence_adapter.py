"""Concise combined evidence construction for application workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from cellstate.schemas.evidence import (
    EvidenceArtifact,
    EvidenceBundle,
    EvidenceExecutionStatus,
    EvidenceWarning,
)

from .models import AnalysisPlan
from .labels import disease_group_label


def _artifact_refs(*outputs: Any) -> tuple[EvidenceArtifact, ...]:
    refs: list[EvidenceArtifact] = []
    for output in outputs:
        if output is None:
            continue
        if hasattr(output, "artifacts"):
            artifacts = output.artifacts
        elif hasattr(output, "to_capability_result"):
            artifacts = output.to_capability_result().artifacts
        else:
            continue
        for artifact in artifacts:
            refs.append(EvidenceArtifact(
                logical_name=f"{output.capability_id}:{artifact.logical_name}",
                path=str(artifact.path),
                category=str(artifact.category),
                media_type=artifact.media_type,
            ))
    return tuple(refs)


def build_combined_evidence_bundle(
    *,
    plan: AnalysisPlan,
    adapter: Any,
    de: Any,
    tf: Any | None,
    tf_status: str,
    tf_message: str,
    cap_tf_001_status: str,
    run_dir: Path,
) -> EvidenceBundle:
    de_summary = {
        "capability_id": de.capability_id,
        "version": de.capability_version,
        "status": de.terminal_status,
        "ordered_contrast": [plan.group_a, plan.group_b],
        "display_ordered_contrast": [
            disease_group_label(plan.group_a), disease_group_label(plan.group_b)
        ],
        "display_contrast": plan.display_contrast,
        "comparison_direction": de.comparison_direction,
        "group_replicate_counts": de.group_replicate_counts,
        "input_gene_count": de.input_gene_count,
        "retained_gene_count": de.retained_gene_count,
        "tested_gene_count": de.tested_gene_count,
        "significant_gene_count": de.significant_gene_count,
        "higher_in_group_a_count": de.upregulated_in_group_a_count,
        "higher_in_group_b_count": de.upregulated_in_group_b_count,
        "design_formula": de.design_assessment.design_formula,
        "inference_class": getattr(de, "evidence_class", "adjusted_inference"),
        "warnings": [warning.model_dump(mode="json") for warning in de.warnings],
        "limitations": [
            "Association does not establish causality.",
            "DESeq2 inference uses independent biological pseudobulk replicates.",
            *(["Group and dataset are not independently identifiable; this is exploratory unadjusted evidence with dataset-level LODO robustness, not dataset-adjusted inference."]
              if getattr(de, "evidence_class", "adjusted_inference") != "adjusted_inference" else []),
        ],
        "provenance_reference": de.provenance_path,
        "cache": {"key": de.cache_key, "hit": de.cache_hit},
    }
    tf_summary: dict[str, Any] = {
        "capability_id": "CAP-TF-002",
        "version": getattr(tf, "node_version", "1.0.0"),
        "status": tf_status,
        "ordered_contrast": [plan.group_a, plan.group_b],
        "display_ordered_contrast": [
            disease_group_label(plan.group_a), disease_group_label(plan.group_b)
        ],
        "message": tf_message,
    }
    if tf is not None:
        tf_summary.update({
            "warnings": [warning.to_dict() for warning in tf.warnings],
            "limitations": [
                "TF activity is model-based support, not direct protein activity or causality."
            ],
            "provenance_reference": tf.provenance_path,
            "cache": {"key": tf.cache_key, "hit": tf.cache_hit},
        })
    evidence = {
        "atlas_qc": {
            "atlas_identity": adapter.atlas_identity,
            "count_source": adapter.count_source,
            "cells_contributing": adapter.n_cells,
            "genes": adapter.n_genes,
            "eligible_replicates": adapter.group_replicate_counts,
            "minimum_cells_per_replicate": 100,
            "adapter_provenance_reference": str(adapter.provenance_path),
        },
        "capabilities": {
            "CAP-DESEQ-003": de_summary,
            "CAP-TF-002": tf_summary,
            "CAP-TF-001": {
                "capability_id": "CAP-TF-001",
                "version": "1.0.0",
                "status": cap_tf_001_status,
                "message": (
                    "Requires an explicit feature program and tested-feature background."
                    if cap_tf_001_status != "not_requested" else "Not requested."
                ),
            },
        },
    }
    artifacts = [
        EvidenceArtifact(
            "atlas_adapter:provenance", str(adapter.provenance_path),
            "provenance", "application/json",
        ),
        *_artifact_refs(de, tf),
    ]
    warnings = [
        EvidenceWarning(w.code, w.message, w.severity, w.context)
        for w in de.warnings
    ]
    if tf is not None:
        warnings.extend(
            EvidenceWarning(w.code, w.message, w.severity.value, w.context)
            for w in tf.warnings
        )
    if de.terminal_status not in {"completed", "completed_with_warnings"}:
        status = EvidenceExecutionStatus.BLOCKED
    else:
        status = (
            EvidenceExecutionStatus.COMPLETED_FROM_CACHE
            if de.cache_hit and (tf is None or tf.cache_hit)
            else EvidenceExecutionStatus.COMPLETED
        )
    identity = {
        "question": plan.question,
        "context": [plan.cell_state, plan.group_a, plan.group_b],
        "evidence": evidence,
    }
    bundle_id = sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()
    return EvidenceBundle(
        bundle_id=bundle_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        execution_status=status,
        analysis_question=plan.question,
        analysis_type="combined_de_tf" if "CAP-TF-002" in plan.requested_capabilities
        else "arbitrary_two_group_de",
        biological_context={
            "cell_state": plan.cell_state,
            "group_a": plan.group_a,
            "group_b": plan.group_b,
            "contrast": plan.contrast,
            "display_group_a": disease_group_label(plan.group_a),
            "display_group_b": disease_group_label(plan.group_b),
            "display_contrast": plan.display_contrast,
            "contrast_direction": "group_a_minus_group_b",
            "inference_class": getattr(de, "evidence_class", "adjusted_inference"),
        },
        unit_of_inference="independent_biological_pseudobulk_replicate",
        deterministic_evidence=evidence,
        design_assessment=de.design_assessment.model_dump(mode="json"),
        limitations=(
            "Deterministic evidence does not establish causality.",
            "CAP-TF-002 is conditional on configured, versioned regulon resources.",
            *(() if getattr(de, "evidence_class", "adjusted_inference") == "adjusted_inference" else (
                "Exploratory conserved features may still reflect systematic dataset effects; they are not dataset-adjusted inference.",
            )),
        ),
        warnings=tuple(warnings),
        artifacts=tuple(artifacts),
        provenance={
            "producer": "cellstate.app.evidence_adapter",
            "run_directory": str(run_dir),
        },
        cache={
            "CAP-DESEQ-003": {"key": de.cache_key, "hit": de.cache_hit},
            "CAP-TF-002": (
                {"key": tf.cache_key, "hit": tf.cache_hit} if tf is not None else None
            ),
        },
    )
