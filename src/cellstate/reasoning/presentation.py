"""Optional, deterministic presentation of validated scientific reports."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

from cellstate.schemas.evidence import (
    EvidenceArtifact, EvidenceBundle, EvidenceExecutionStatus, EvidenceWarning,
)
from cellstate.schemas.reasoning import ScientificReport


CONFOUNDING_TEXT = "EXPLORATORY CONFOUNDED-DESIGN ANALYSIS"
COLORS = {"neutral": "#9CA3AF", "significant": "#2563EB", "conserved": "#DC2626",
          "positive": "#2563EB", "negative": "#D97706"}


@dataclass(frozen=True)
class PresentationWarning:
    code: str
    message: str
    severity: str = "warning"


@dataclass(frozen=True)
class PresentationResult:
    pdf_path: Path | None
    figure_paths: tuple[Path, ...]
    captions: tuple[str, ...]
    warnings: tuple[PresentationWarning, ...]

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        return ((self.pdf_path,) if self.pdf_path else ()) + self.figure_paths


def _artifacts(bundle: EvidenceBundle) -> dict[str, Path]:
    return {item.logical_name: Path(item.path) for item in bundle.artifacts}


def _find(bundle: EvidenceBundle, *names: str) -> Path | None:
    artifacts = _artifacts(bundle)
    for name in names:
        path = artifacts.get(name)
        if path is not None and path.is_file():
            return path
    return None


def _table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")


def _atomic_figure(fig: Any, destination: Path, *, dpi: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp{destination.suffix}")
    try:
        fig.savefig(temporary, format=destination.suffix[1:], dpi=dpi,
                    bbox_inches="tight", facecolor="white")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _save_pair(fig: Any, figures_dir: Path, stem: str) -> tuple[Path, Path]:
    png, svg = figures_dir / f"{stem}.png", figures_dir / f"{stem}.svg"
    _atomic_figure(fig, png, dpi=300)
    _atomic_figure(fig, svg)
    plt.close(fig)
    return png, svg


def _context(bundle: EvidenceBundle) -> tuple[str, str, bool]:
    context = bundle.biological_context
    contrast = str(context.get("display_contrast") or context.get("contrast") or bundle.analysis_question)
    cell_state = str(context.get("cell_state") or "Cell-state analysis")
    exploratory = context.get("inference_class") in {
        "exploratory_unadjusted", "exploratory_lodo_conserved"
    } or any(CONFOUNDING_TEXT in warning.message.upper() for warning in bundle.warnings)
    return cell_state, contrast, exploratory


def _volcano(bundle: EvidenceBundle, figures_dir: Path) -> tuple[tuple[Path, Path], str] | None:
    source = _find(bundle, "CAP-DESEQ-003:full_unadjusted_deseq2_results",
                   "CAP-DESEQ-003:deseq2_results")
    if source is None:
        return None
    data = _table(source)
    required = {"log2FoldChange", "padj"}
    if not required <= set(data):
        return None
    conserved_path = _find(bundle, "CAP-DESEQ-003:conserved_features")
    conserved: set[str] = set()
    if conserved_path:
        conserved_data = _table(conserved_path)
        if "gene" in conserved_data:
            conserved = set(conserved_data["gene"].dropna().astype(str))
    genes = data["gene"].astype(str) if "gene" in data else data.index.astype(str)
    padj = pd.to_numeric(data["padj"], errors="coerce").clip(lower=1e-300)
    lfc = pd.to_numeric(data["log2FoldChange"], errors="coerce")
    significant = padj.lt(0.05)
    is_conserved = genes.isin(conserved)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.scatter(lfc, -padj.map(math.log10), s=8, c=COLORS["neutral"], alpha=.45,
               linewidths=0, label=f"Other tested ({len(data) - int(significant.sum())})")
    ax.scatter(lfc[significant], -padj[significant].map(math.log10), s=11,
               c=COLORS["significant"], alpha=.65, linewidths=0,
               label=f"FDR < 0.05 ({int(significant.sum())})")
    if conserved:
        ax.scatter(lfc[is_conserved], -padj[is_conserved].map(math.log10), s=18,
                   c=COLORS["conserved"], alpha=.9, linewidths=0,
                   label=f"LODO conserved ({int(is_conserved.sum())})")
    cell_state, contrast, exploratory = _context(bundle)
    ax.axvline(0, color="#374151", lw=.8)
    ax.set(title=f"Differential expression: {cell_state}\n{contrast}",
           xlabel="DESeq2 log2 fold change", ylabel="−log10 adjusted p-value")
    ax.legend(frameon=False, fontsize=8)
    caption = "DESeq2 effect size versus multiple-testing-adjusted significance; positive values follow the configured group A minus group B contrast."
    if exploratory:
        caption += " Exploratory confounded design: group and dataset are not independently identifiable."
    return _save_pair(fig, figures_dir, "de_volcano"), caption


def _lodo(bundle: EvidenceBundle, figures_dir: Path) -> tuple[tuple[Path, Path], str] | None:
    source = _find(bundle, "CAP-DESEQ-003:lodo_fold_results")
    if source is None:
        return None
    data = _table(source)
    if not {"omitted_dataset", "fold_status"} <= set(data):
        return None
    fold = data[["omitted_dataset", "fold_status"]].drop_duplicates().sort_values("omitted_dataset")
    if fold.empty:
        return None
    counts = fold["fold_status"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    colors_by_status = [COLORS["significant"] if str(x) == "estimable" else COLORS["negative"] for x in counts.index]
    ax.bar(counts.index.astype(str), counts.values, color=colors_by_status)
    ax.set(title="Dataset-level leave-one-dataset-out robustness",
           xlabel="Fold eligibility status", ylabel="Number of omitted-dataset folds")
    for index, value in enumerate(counts.values):
        ax.text(index, value, str(value), ha="center", va="bottom")
    caption = f"Eligibility of {len(fold)} dataset-level LODO folds; estimable folds contribute to feature stability summaries. Exploratory confounded design: group and dataset are not independently identifiable."
    return _save_pair(fig, figures_dir, "lodo_robustness"), caption


def _tf_resource(bundle: EvidenceBundle, figures_dir: Path) -> tuple[tuple[Path, Path], str] | None:
    source = _find(bundle, "CAP-TF-002:significant_tf_activity",
                   "CAP-TF-002:tf_activity_by_resource")
    if source is None:
        return None
    data = _table(source)
    if "significant" in data:
        data = data[data["significant"].astype(str).str.casefold().isin({"true", "1"})]
    if data.empty or not {"database", "tf", "activity_score"} <= set(data):
        return None
    data = data.assign(abs_score=pd.to_numeric(data["activity_score"], errors="coerce").abs())
    data = data.sort_values(["database", "abs_score", "tf"], ascending=[True, False, True]).groupby("database", sort=True).head(8)
    labels = data["tf"].astype(str) + " · " + data["database"].astype(str)
    values = pd.to_numeric(data["activity_score"], errors="coerce")
    order = values.abs().sort_values().index
    fig, ax = plt.subplots(figsize=(8.0, max(4.5, len(data) * .27)))
    ax.barh(labels.loc[order], values.loc[order], color=[COLORS["positive"] if x >= 0 else COLORS["negative"] for x in values.loc[order]])
    ax.axvline(0, color="#374151", lw=.8)
    ax.set(title="Significant TF activity by regulon resource", xlabel="Signed TF activity score", ylabel="TF · resource")
    caption = f"Top absolute significant TF activity estimates within each resource ({len(data)} displayed); sign follows the configured contrast orientation."
    return _save_pair(fig, figures_dir, "tf_activity_by_resource"), caption


def _tf_consensus(bundle: EvidenceBundle, figures_dir: Path) -> tuple[tuple[Path, Path], str] | None:
    source = _find(bundle, "CAP-TF-002:tf_activity_consensus")
    if source is None:
        return None
    data = _table(source)
    if data.empty or not {"tf", "median_consensus_activity_score"} <= set(data):
        return None
    if "directional_consensus_status" in data:
        data = data[data["directional_consensus_status"].astype(str).str.casefold().isin({"true", "1"})]
    data = data.assign(score=pd.to_numeric(data["median_consensus_activity_score"], errors="coerce"))
    data = data.sort_values(["score", "tf"], key=lambda x: x.abs() if x.name == "score" else x,
                            ascending=[False, True]).head(15).sort_values("score")
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.0, max(4.2, len(data) * .32)))
    ax.barh(data["tf"].astype(str), data["score"], color=[COLORS["positive"] if x >= 0 else COLORS["negative"] for x in data["score"]])
    ax.axvline(0, color="#374151", lw=.8)
    ax.set(title="Cross-resource TF consensus", xlabel="Median consensus activity score", ylabel="Transcription factor")
    caption = f"Directionally concordant cross-resource TF estimates ({len(data)} displayed); positive and negative scores follow the configured contrast."
    return _save_pair(fig, figures_dir, "tf_cross_resource_consensus"), caption


def _design(bundle: EvidenceBundle, figures_dir: Path) -> tuple[tuple[Path, Path], str] | None:
    source = _find(bundle, "CAP-DESEQ-003:input_sample_metadata")
    if source is None:
        return None
    data = _table(source)
    if not {"group", "dataset"} <= set(data):
        return None
    counts = data.groupby(["dataset", "group"], sort=True).size().unstack(fill_value=0)
    if counts.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.0, max(4.2, len(counts) * .35)))
    counts.plot.barh(stacked=True, ax=ax, color=["#2563EB", "#D97706", "#059669", "#7C3AED"][:len(counts.columns)])
    ax.set(title="Independent replicate representation by dataset", xlabel="Biological pseudobulk replicates", ylabel="Dataset")
    ax.legend(title="Group", frameon=False)
    caption = f"Distribution of {len(data)} independent biological pseudobulk replicates across {len(counts)} datasets and {len(counts.columns)} comparison groups."
    if _context(bundle)[2]:
        caption += " Exploratory confounded design: group and dataset are not independently identifiable."
    return _save_pair(fig, figures_dir, "replicate_dataset_design_qc"), caption


def _atomic_pdf(report: ScientificReport, bundle: EvidenceBundle,
                figures: list[tuple[Path, str]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
                              textColor=colors.HexColor("#123B5D"), spaceAfter=14))
    styles.add(ParagraphStyle(name="FigureCaption", parent=styles["BodyText"],
                              fontSize=8.5, leading=11, textColor=colors.HexColor("#374151"),
                              spaceAfter=10))
    story: list[Any] = []
    cell_state, contrast, exploratory = _context(bundle)
    story += [Paragraph("CellState Agent Scientific Report", styles["ReportTitle"]),
              Paragraph(f"<b>{cell_state}</b><br/>{contrast}", styles["Heading2"]),
              Paragraph(f"Execution status: {bundle.execution_status.value}", styles["BodyText"]), Spacer(1, 8)]
    if exploratory:
        story += [Paragraph(f"<b>{CONFOUNDING_TEXT}:</b> group and dataset are not independently identifiable. Features may still reflect systematic dataset effects.", styles["BodyText"]), Spacer(1, 8)]
    story += [Paragraph("Deterministic executive summary", styles["Heading2"]),
              Paragraph(report.deterministic_executive_summary, styles["BodyText"])]
    metrics = _metric_rows(bundle)
    if metrics:
        table = Table([["Metric", "Value"], *metrics], colWidths=[2.7*inch, 3.6*inch], repeatRows=1)
        table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DCEAF4")),
                                   ("GRID", (0,0), (-1,-1), .25, colors.grey),
                                   ("VALIGN", (0,0), (-1,-1), "TOP"),
                                   ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold")]))
        story += [Spacer(1, 10), table]
    critic, interpretation = report.critic_report, report.interpretation_report
    story += [Spacer(1, 12), Paragraph("Evidence critique", styles["Heading2"]),
              Paragraph(f"Confidence: <b>{critic.overall_confidence}</b>", styles["BodyText"]),
              Paragraph(critic.reasoning_summary, styles["BodyText"]),
              Paragraph("Limitations", styles["Heading3"])]
    story += [Paragraph(f"• {item}", styles["BodyText"]) for item in critic.limitations] or [Paragraph("None reported.", styles["BodyText"])]
    story += [Paragraph("Validated interpretation", styles["Heading2"]), Paragraph(interpretation.summary, styles["BodyText"]),
              Paragraph("Observations", styles["Heading3"])]
    story += [Paragraph(f"• {item}", styles["BodyText"]) for item in interpretation.observations] or [Paragraph("No observations reported.", styles["BodyText"])]
    story += [Paragraph("Explicit hypotheses", styles["Heading3"])]
    story += [Paragraph(f"• {item}", styles["BodyText"]) for item in interpretation.hypotheses] or [Paragraph("No hypotheses reported.", styles["BodyText"])]
    if figures:
        story += [PageBreak(), Paragraph("Figures", styles["Heading1"])]
        for png, caption in figures:
            story += [KeepTogether([Image(str(png), width=6.7*inch, height=4.4*inch, kind="proportional"),
                                    Paragraph(caption, styles["FigureCaption"]), Spacer(1, 12)])]
    story += [PageBreak(), Paragraph("Artifact and provenance appendix", styles["Heading1"]),
              Paragraph(f"EvidenceBundle ID: {bundle.bundle_id}", styles["BodyText"])]
    for key, value in sorted(report.provenance.items()):
        story.append(Paragraph(f"<b>{key}:</b> {value}", styles["BodyText"] if len(str(value)) < 250 else styles["Code"]))
    story += [Paragraph("Referenced deterministic artifacts", styles["Heading2"])]
    for artifact in sorted(bundle.artifacts, key=lambda item: item.logical_name):
        story.extend([CondPageBreak(.45 * inch), KeepTogether([
            Paragraph(f"<b>{artifact.logical_name}</b><br/>{artifact.path}", styles["BodyText"]),
            Spacer(1, 3),
        ])])
    doc = SimpleDocTemplate(str(temporary), pagesize=letter, rightMargin=.65*inch,
                            leftMargin=.65*inch, topMargin=.65*inch, bottomMargin=.65*inch,
                            title="CellState Agent Scientific Report")
    try:
        doc.build(story)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _metric_rows(bundle: EvidenceBundle) -> list[list[str]]:
    evidence = bundle.deterministic_evidence
    rows: list[list[str]] = []
    capabilities = evidence.get("capabilities", {}) if isinstance(evidence, dict) else {}
    for capability, values in sorted(capabilities.items()):
        if not isinstance(values, dict):
            continue
        for key in ("tested_gene_count", "significant_gene_count", "conserved_feature_count",
                    "significant_activity_count", "directional_consensus_count"):
            if key in values:
                rows.append([f"{capability} · {key.replace('_', ' ')}", str(values[key])])
    return rows


def generate_presentation(report: ScientificReport, bundle: EvidenceBundle,
                          run_dir: Path) -> PresentationResult:
    """Generate optional figures/PDF without mutating validated inputs."""
    figures_dir = Path(run_dir) / "figures"
    figure_paths: list[Path] = []
    pdf_figures: list[tuple[Path, str]] = []
    warnings: list[PresentationWarning] = []
    producers: tuple[tuple[str, Callable[..., Any]], ...] = (
        ("de_volcano", _volcano), ("lodo_robustness", _lodo),
        ("tf_activity_by_resource", _tf_resource),
        ("tf_cross_resource_consensus", _tf_consensus),
        ("replicate_dataset_design_qc", _design),
    )
    for name, producer in producers:
        try:
            generated = producer(bundle, figures_dir)
            if generated:
                paths, caption = generated
                figure_paths.extend(paths)
                pdf_figures.append((paths[0], caption))
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            warnings.append(PresentationWarning("PRESENTATION_FIGURE_SKIPPED", f"{name}: {type(exc).__name__}: {exc}"))
    pdf_path = Path(run_dir) / "scientific_report.pdf"
    try:
        _atomic_pdf(report, bundle, pdf_figures, pdf_path)
    except (OSError, ValueError, RuntimeError) as exc:
        warnings.append(PresentationWarning("PRESENTATION_PDF_FAILED", f"{type(exc).__name__}: {exc}"))
        pdf_path = None
    return PresentationResult(pdf_path, tuple(figure_paths),
                              tuple(caption for _, caption in pdf_figures), tuple(warnings))


def load_evidence_bundle(path: Path) -> EvidenceBundle:
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvidenceBundle(
        bundle_id=data["bundle_id"], created_at_utc=data["created_at_utc"],
        execution_status=EvidenceExecutionStatus(data["execution_status"]),
        analysis_question=data["analysis_question"], analysis_type=data["analysis_type"],
        biological_context=data["biological_context"], unit_of_inference=data["unit_of_inference"],
        deterministic_evidence=data["deterministic_evidence"], design_assessment=data["design_assessment"],
        limitations=tuple(data["limitations"]), warnings=tuple(EvidenceWarning(**item) for item in data["warnings"]),
        artifacts=tuple(EvidenceArtifact(**item) for item in data["artifacts"]),
        provenance=data["provenance"], cache=data["cache"], schema_version=data["schema_version"],
    )


def render_existing_report(run_dir: Path) -> PresentationResult:
    """Render presentation from existing validated JSON without model calls."""
    run_dir = Path(run_dir)
    report = ScientificReport.model_validate_json((run_dir / "scientific_report.json").read_text(encoding="utf-8"))
    bundle = load_evidence_bundle(run_dir / "evidence_bundle.json")
    if report.evidence_bundle_id != bundle.bundle_id:
        raise ValueError("ScientificReport and EvidenceBundle identity mismatch")
    return generate_presentation(report, bundle, run_dir)
