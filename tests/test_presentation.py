import copy
import json
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from cellstate.reasoning.presentation import generate_presentation
from cellstate.schemas.evidence import (
    EvidenceArtifact, EvidenceBundle, EvidenceExecutionStatus, EvidenceWarning,
)
from cellstate.schemas.reasoning import CriticReport, InterpretationReport, ScientificReport


def _write_table(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False, sep="\t" if path.suffix == ".tsv" else ",")
    return path


def _fixture(tmp_path: Path, *, optional: bool = True):
    de = _write_table(tmp_path / "de.csv", [
        {"gene": "A", "log2FoldChange": 1.2, "padj": .001},
        {"gene": "B", "log2FoldChange": -.7, "padj": .2},
    ])
    artifacts = [EvidenceArtifact("CAP-DESEQ-003:full_unadjusted_deseq2_results", str(de), "inferential", "text/csv")]
    if optional:
        tables = {
            "CAP-DESEQ-003:conserved_features": ("conserved.csv", [{"gene": "A", "conserved": True}]),
            "CAP-DESEQ-003:lodo_fold_results": ("lodo.csv", [
                {"omitted_dataset": "D1", "fold_status": "estimable"},
                {"omitted_dataset": "D2", "fold_status": "skipped"},
            ]),
            "CAP-DESEQ-003:input_sample_metadata": ("samples.tsv", [
                {"sample": "S1", "group": "NBM", "dataset": "D1"},
                {"sample": "S2", "group": "NDMM", "dataset": "D2"},
            ]),
            "CAP-TF-002:significant_tf_activity": ("tf.tsv", [
                {"database": "R1", "tf": "TF1", "activity_score": 2.0, "significant": True},
                {"database": "R2", "tf": "TF2", "activity_score": -1.5, "significant": True},
            ]),
            "CAP-TF-002:tf_activity_consensus": ("consensus.tsv", [
                {"tf": "TF1", "median_consensus_activity_score": 1.8, "directional_consensus_status": True},
            ]),
        }
        for logical_name, (filename, rows) in tables.items():
            path = _write_table(tmp_path / filename, rows)
            artifacts.append(EvidenceArtifact(logical_name, str(path), "inferential", "text/tab-separated-values"))
    warning = "EXPLORATORY CONFOUNDED-DESIGN ANALYSIS: group and dataset are not independently identifiable."
    bundle = EvidenceBundle(
        bundle_id="bundle-1", created_at_utc="2026-07-21T00:00:00+00:00",
        execution_status=EvidenceExecutionStatus.COMPLETED,
        analysis_question="Compare NBM and NDMM", analysis_type="combined_de_tf",
        biological_context={"cell_state": "GZMB CD8 T cell", "group_a": "NBM", "group_b": "NDMM",
                            "display_contrast": "Normal Bone Marrow (NBM) − Newly Diagnosed Multiple Myeloma (NDMM)",
                            "inference_class": "exploratory_lodo_conserved"},
        unit_of_inference="biological replicate",
        deterministic_evidence={"capabilities": {"CAP-DESEQ-003": {"tested_gene_count": 2, "significant_gene_count": 1}}},
        design_assessment={"ready": True}, limitations=(warning,),
        warnings=(EvidenceWarning("EXPLORATORY_CONFOUNDED_DESIGN", warning),),
        artifacts=tuple(artifacts), provenance={"pipeline": "test"}, cache={"cache_hit": False},
    )
    critic = CriticReport.model_validate({
        "evidence_bundle_id": bundle.bundle_id,
        **{name: {"status": "warning", "score": 2, "summary": warning, "evidence_refs": []}
           for name in ("replication_assessment", "statistical_support_assessment", "confounding_assessment",
                        "design_validity_assessment", "assumption_risk_assessment", "generalizability_assessment")},
        "strengths": [], "limitations": [warning], "recommended_follow_up": [],
        "overall_confidence": "low", "overall_confidence_score": 2,
        "reasoning_summary": warning, "created_at_utc": "2026-07-21T00:00:00+00:00",
    })
    interpretation = InterpretationReport.model_validate({
        "evidence_bundle_id": bundle.bundle_id, "observations": ["One deterministic feature was significant."],
        "biological_programs": [], "candidate_regulators": [],
        "hypotheses": ["Hypothesis: the reported association should be independently tested."],
        "critic_limitations_referenced": [warning], "experimental_follow_up": [],
        "interpretation_confidence": "low", "summary": "Associative evidence only.",
        "created_at_utc": "2026-07-21T00:00:00+00:00",
    })
    report = ScientificReport(
        evidence_bundle_id=bundle.bundle_id, deterministic_executive_summary="Completed deterministic analysis.",
        critic_report=critic, interpretation_report=interpretation,
        provenance={"openai_model": "test", "report_artifact_paths": {"scientific_report": "dynamic"}},
        created_at_utc="2026-07-21T00:00:00+00:00",
    )
    return bundle, report


def test_complete_presentation_is_dynamic_atomic_and_readable(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    bundle, report = _fixture(source)
    before_bundle, before_report = copy.deepcopy(bundle.to_dict()), report.model_dump(mode="json")
    run_dir = tmp_path / "arbitrary-run-name"
    result = generate_presentation(report, bundle, run_dir)
    assert result.pdf_path == run_dir / "scientific_report.pdf"
    assert result.pdf_path.stat().st_size > 0
    assert len(result.figure_paths) == 10
    assert all(path.parent == run_dir / "figures" and path.stat().st_size > 0 for path in result.figure_paths)
    assert {path.suffix for path in result.figure_paths} == {".png", ".svg"}
    assert not list(run_dir.rglob("*.tmp"))
    reader = PdfReader(result.pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert len(reader.pages) >= 2
    assert "CellState Agent Scientific Report" in text
    assert "EXPLORATORY CONFOUNDED-DESIGN ANALYSIS" in text
    assert "Explicit hypotheses" in text
    assert bundle.to_dict() == before_bundle
    assert report.model_dump(mode="json") == before_report


def test_missing_optional_tables_skip_only_affected_figures(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    bundle, report = _fixture(source, optional=False)
    result = generate_presentation(report, bundle, tmp_path / "run")
    assert {path.name for path in result.figure_paths} == {"de_volcano.png", "de_volcano.svg"}
    assert result.pdf_path and result.pdf_path.exists()
