"""Deterministic figures derived exclusively from CAP-LODO-001 tables."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..schemas.common import ArtifactCategory, ArtifactReference
from .io import read_table, resolve_table_artifacts, write_csv_atomic
from .schemas import AtlasLODOPlotInput, AtlasLODOPlotOutput
from .styles import (ELIGIBLE_COLOR, GROUP_A_COLOR, GROUP_B_COLOR, INELIGIBLE_COLOR,
                     NEUTRAL_COLOR, REFERENCE_COLOR, atlas_lodo_style)

REQUIRED_ARTIFACTS = ("atlas_summary", "gene_summaries", "cell_state_eligibility", "fold_results")
_FIXED_DATE = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _save_figure(fig, stem: Path, formats: tuple[str, ...], dpi: int) -> tuple[Path, ...]:
    outputs = []
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.{extension}")
        metadata = ({"CreationDate": _FIXED_DATE, "ModDate": _FIXED_DATE, "Creator": "cellstate"}
                    if extension == "pdf" else {"Date": "2000-01-01T00:00:00Z", "Creator": "cellstate"})
        try:
            fig.savefig(temporary, format=extension, dpi=dpi, bbox_inches="tight", metadata=metadata)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        outputs.append(path)
    return tuple(outputs)


def _program_count_source(atlas: pd.DataFrame) -> pd.DataFrame:
    columns = ["cell_state", "group_a_specific_genes", "group_b_specific_genes",
               "group_a_enriched_genes", "group_b_enriched_genes",
               "group_a_label", "group_b_label", "reference_group"]
    return atlas.loc[:, columns].sort_values("cell_state", kind="mergesort").reset_index(drop=True)


def _effect_source(genes: pd.DataFrame) -> pd.DataFrame:
    columns = ["cell_state", "gene", "group_a_effect_median", "group_b_effect_median",
               "delta_median", "group_a_specific", "group_b_specific", "group_a_enriched",
               "group_b_enriched", "label_worthy", "reference_group", "group_a_label", "group_b_label"]
    return genes.loc[:, columns].sort_values(["cell_state", "gene"], kind="mergesort").reset_index(drop=True)


def _eligibility_source(eligibility: pd.DataFrame) -> pd.DataFrame:
    columns = ["cell_state", "reference_sample_count", "group_a_sample_count", "group_b_sample_count",
               "dataset_count", "eligible", "failure_reasons", "reference_group", "group_a_label", "group_b_label"]
    return eligibility.loc[:, columns].sort_values("cell_state", kind="mergesort").reset_index(drop=True)


def _fold_source(folds: pd.DataFrame) -> pd.DataFrame:
    columns = ["cell_state", "gene", "held_out_dataset", "effect_group_a_vs_reference",
               "effect_group_b_vs_reference", "delta_group_a_minus_group_b", "reference_group",
               "group_a_label", "group_b_label"]
    return folds.loc[:, columns].sort_values(["cell_state", "gene", "held_out_dataset"], kind="mergesort").reset_index(drop=True)


def _plot_program_counts(table: pd.DataFrame):
    height = max(2.6, .48 * len(table) + 1.3)
    fig, ax = plt.subplots(figsize=(7.2, height))
    y = np.arange(len(table))
    a = table.group_a_specific_genes.to_numpy(float)
    b = table.group_b_specific_genes.to_numpy(float)
    ax.barh(y, a, color=GROUP_A_COLOR, label="Group A specific")
    ax.barh(y, b, left=a, color=GROUP_B_COLOR, label="Group B specific")
    ax.set_yticks(y, table.cell_state)
    ax.invert_yaxis()
    ax.set_xlabel("Classified genes")
    ax.set_title("AtlasLODO group-specific gene counts")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    return fig


def _plot_effect_landscape(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    colors = np.where(table.group_a_specific, GROUP_A_COLOR,
             np.where(table.group_b_specific, GROUP_B_COLOR, NEUTRAL_COLOR))
    ax.scatter(table.group_a_effect_median, table.group_b_effect_median,
               c=colors, s=24, alpha=.85, linewidths=0)
    ax.axhline(0, color="#D0D0D0", linewidth=.8)
    ax.axvline(0, color="#D0D0D0", linewidth=.8)
    for row in table.loc[table.label_worthy & (table.group_a_specific | table.group_b_specific)].itertuples():
        ax.annotate(row.gene, (row.group_a_effect_median, row.group_b_effect_median),
                    xytext=(3, 3), textcoords="offset points", fontsize=7)
    labels = table.iloc[0] if len(table) else None
    ax.set_xlabel(f"Median effect: {labels.group_a_label} vs {labels.reference_group}" if labels is not None else "Group A effect")
    ax.set_ylabel(f"Median effect: {labels.group_b_label} vs {labels.reference_group}" if labels is not None else "Group B effect")
    ax.set_title("Cross-fold transcriptional effect landscape")
    return fig


def _plot_eligibility(table: pd.DataFrame):
    height = max(2.7, .55 * len(table) + 1.3)
    fig, ax = plt.subplots(figsize=(7.2, height))
    y = np.arange(len(table)); width = .23
    ax.barh(y - width, table.reference_sample_count, height=width, color=REFERENCE_COLOR, label="Reference")
    ax.barh(y, table.group_a_sample_count, height=width, color=GROUP_A_COLOR, label="Group A")
    ax.barh(y + width, table.group_b_sample_count, height=width, color=GROUP_B_COLOR, label="Group B")
    ax.set_yticks(y, table.cell_state)
    ax.invert_yaxis(); ax.set_xlabel("Retained biological samples")
    ax.set_title("AtlasLODO cell-state eligibility")
    for index, eligible in enumerate(table.eligible):
        ax.get_yticklabels()[index].set_color(ELIGIBLE_COLOR if bool(eligible) else INELIGIBLE_COLOR)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    return fig


def _plot_fold_deltas(table: pd.DataFrame):
    genes = sorted(table.gene.astype(str).unique())
    datasets = sorted(table.held_out_dataset.astype(str).unique())
    fig, ax = plt.subplots(figsize=(max(6.2, .55 * len(genes) + 2), max(3.2, .42 * len(datasets) + 2)))
    gene_index = {value: index for index, value in enumerate(genes)}
    dataset_index = {value: index for index, value in enumerate(datasets)}
    x = table.gene.astype(str).map(gene_index); y = table.held_out_dataset.astype(str).map(dataset_index)
    limit = max(float(table.delta_group_a_minus_group_b.abs().max()), .01)
    points = ax.scatter(x, y, c=table.delta_group_a_minus_group_b, cmap="coolwarm",
                        vmin=-limit, vmax=limit, s=35, edgecolors="none")
    ax.set_xticks(range(len(genes)), genes, rotation=45, ha="right")
    ax.set_yticks(range(len(datasets)), datasets)
    ax.set_xlabel("Gene"); ax.set_ylabel("Held-out dataset")
    ax.set_title("AtlasLODO fold-level group-A-minus-group-B effects")
    fig.colorbar(points, ax=ax, label="Delta effect")
    return fig


def plot_atlas_lodo(request: AtlasLODOPlotInput) -> AtlasLODOPlotOutput:
    """Render figures without modifying or recomputing canonical analysis results."""
    paths = resolve_table_artifacts(request.analysis_result, REQUIRED_ARTIFACTS)
    atlas = read_table(paths["atlas_summary"], ("cell_state", "group_a_specific_genes", "group_b_specific_genes",
        "group_a_enriched_genes", "group_b_enriched_genes", "group_a_label", "group_b_label", "reference_group"))
    genes = read_table(paths["gene_summaries"], ("cell_state", "gene", "group_a_effect_median", "group_b_effect_median",
        "delta_median", "group_a_specific", "group_b_specific", "group_a_enriched", "group_b_enriched", "label_worthy",
        "reference_group", "group_a_label", "group_b_label"))
    eligibility = read_table(paths["cell_state_eligibility"], ("cell_state", "reference_sample_count", "group_a_sample_count",
        "group_b_sample_count", "dataset_count", "eligible", "failure_reasons", "reference_group", "group_a_label", "group_b_label"))
    folds = read_table(paths["fold_results"], ("cell_state", "gene", "held_out_dataset", "effect_group_a_vs_reference",
        "effect_group_b_vs_reference", "delta_group_a_minus_group_b", "reference_group", "group_a_label", "group_b_label"))

    definitions = (
        ("atlas_lodo_program_counts", _program_count_source(atlas), _plot_program_counts),
        ("atlas_lodo_effect_landscape", _effect_source(genes), _plot_effect_landscape),
        ("atlas_lodo_eligibility", _eligibility_source(eligibility), _plot_eligibility),
        ("atlas_lodo_fold_deltas", _fold_source(folds), _plot_fold_deltas),
    )
    figures = []; sources = []
    with atlas_lodo_style():
        for name, source, plotter in definitions:
            source_path = request.output_dir / "source_tables" / f"{name}_source.csv"
            write_csv_atomic(source, source_path)
            sources.append(ArtifactReference(f"{name}_source", str(source_path), ArtifactCategory.INPUT_DERIVED, "text/csv"))
            fig = plotter(source)
            try:
                for path in _save_figure(fig, request.output_dir / "figures" / name, request.formats, request.dpi):
                    figures.append(ArtifactReference(f"{name}_{path.suffix[1:]}", str(path),
                        ArtifactCategory.DESCRIPTIVE, {".pdf":"application/pdf", ".svg":"image/svg+xml", ".png":"image/png"}[path.suffix]))
            finally:
                plt.close(fig)
    return AtlasLODOPlotOutput("CAP-LODO-001", request.analysis_result.cache_key, tuple(figures), tuple(sources))
