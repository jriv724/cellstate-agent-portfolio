"""Statistics-free deterministic renderers for completed CAP-TF-001 artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..schemas.common import (ArtifactCategory, ArtifactReference, CapabilityResult,
                              CapabilityStatus)
from .io import resolve_table_artifacts, write_csv_atomic

_FIXED_DATE = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _read_canonical_tsv(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    missing = [column for column in required_columns if column not in table.columns]
    if missing:
        raise ValueError(f"canonical plotting table {path.name} is missing columns: {missing}")
    return table


@dataclass(frozen=True)
class TFRegulatoryNetworkPlotInput:
    analysis_result: CapabilityResult
    output_dir: Path
    formats: tuple[Literal["pdf", "svg", "png"], ...] = ("pdf", "svg", "png")
    dpi: int = 300
    width: float = 11.0
    height: float = 7.0
    title: str | None = None
    maximum_displayed_tfs: int = 30
    maximum_displayed_targets: int = 80
    show_labels: bool = True
    layout_seed: int = 17
    layout_algorithm: Literal["deterministic_layered"] = "deterministic_layered"
    font_size: float = 8.0

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.analysis_result.capability_id != "CAP-TF-001":
            raise ValueError("TF network plotting requires CAP-TF-001")
        if self.analysis_result.status not in {
            CapabilityStatus.COMPLETED, CapabilityStatus.COMPLETED_WITH_WARNINGS,
        }:
            raise ValueError("TF network plotting requires a completed analysis result")
        if not self.formats or len(set(self.formats)) != len(self.formats):
            raise ValueError("formats must be nonempty and unique")
        if self.dpi < 72 or self.width <= 0 or self.height <= 0:
            raise ValueError("plot dimensions and DPI must be positive")
        if self.maximum_displayed_tfs < 1 or self.maximum_displayed_targets < 1:
            raise ValueError("display limits must be positive")


@dataclass(frozen=True)
class TFRegulatoryNetworkPlotOutput:
    upstream_capability_id: str
    upstream_cache_key: str
    figures: tuple[ArtifactReference, ...]
    display_tables: tuple[ArtifactReference, ...]
    plot_provenance: ArtifactReference

    @property
    def artifacts(self) -> tuple[ArtifactReference, ...]:
        return self.figures + self.display_tables + (self.plot_provenance,)


def _save(fig, stem: Path, formats: tuple[str, ...], dpi: int) -> tuple[Path, ...]:
    outputs = []
    for extension in formats:
        path = stem.with_suffix("." + extension)
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


def _display_subset(nodes: pd.DataFrame, edges: pd.DataFrame,
                    request: TFRegulatoryNetworkPlotInput) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    tf_edges = edges.loc[edges.edge_type == "TF_to_target"].copy()
    if tf_edges.empty:
        return nodes.iloc[0:0].copy(), edges.iloc[0:0].copy(), "empty canonical network"
    tf_rank = (tf_edges.groupby("source").agg(
        program_count=("program_id", "nunique"),
        minimum_adjusted_p_value=("minimum_adjusted_p_value", "min"),
        target_count=("target", "nunique"))
        .sort_values(["program_count", "minimum_adjusted_p_value", "target_count"],
                     ascending=[False, True, False], kind="mergesort"))
    keep_tf = set(tf_rank.head(request.maximum_displayed_tfs).index)
    candidate = tf_edges.loc[tf_edges.source.isin(keep_tf)]
    target_rank = (candidate.groupby("target").agg(
        tf_count=("source", "nunique"), program_count=("program_id", "nunique"),
        minimum_adjusted_p_value=("minimum_adjusted_p_value", "min"))
        .sort_values(["tf_count", "program_count", "minimum_adjusted_p_value"],
                     ascending=[False, False, True], kind="mergesort"))
    keep_targets = set(target_rank.head(request.maximum_displayed_targets).index)
    kept_tf_edges = candidate.loc[candidate.target.isin(keep_targets)]
    keep_programs = set(kept_tf_edges.program_id.map(lambda x: f"PROGRAM::{x}"))
    member_edges = edges.loc[
        (edges.edge_type == "target_to_program")
        & edges.source.isin(keep_targets) & edges.target.isin(keep_programs)]
    display_edges = pd.concat([kept_tf_edges, member_edges], ignore_index=True).drop_duplicates("edge_id")
    keep_nodes = set(display_edges.source) | set(display_edges.target)
    display_nodes = nodes.loc[nodes.node_id.isin(keep_nodes)].copy()
    rule = (
        f"Display-only subset: rank TFs by represented programs descending, minimum adjusted p-value ascending, "
        f"target count descending; retain at most {request.maximum_displayed_tfs}. Rank their targets by TF count "
        f"descending, program count descending, minimum adjusted p-value ascending; retain at most "
        f"{request.maximum_displayed_targets}. Omitted nodes remain in canonical artifacts."
    )
    return (display_nodes.sort_values("node_id", kind="mergesort").reset_index(drop=True),
            display_edges.sort_values("edge_id", kind="mergesort").reset_index(drop=True), rule)


def _heatmap_figure(table: pd.DataFrame, request: TFRegulatoryNetworkPlotInput):
    fig, ax = plt.subplots(figsize=(request.width, request.height))
    if table.empty:
        ax.text(.5, .5, "No consensus-supported TFs", ha="center", va="center")
        ax.axis("off")
        return fig
    labels = table.program_id.astype(str) + " | " + table.cell_state.astype(str) + " | " + table.contrast.astype(str)
    source = table.assign(_program_label=labels)
    matrix = source.pivot_table(index="_program_label", columns="tf", values="display_score",
                                aggfunc="max", fill_value=0)
    tf_order = source.groupby("tf").display_score.max().sort_values(ascending=False, kind="mergesort").index
    matrix = matrix.reindex(index=sorted(matrix.index), columns=tf_order[:request.maximum_displayed_tfs])
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Reds", interpolation="nearest")
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=request.font_size)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=60, ha="right", fontsize=request.font_size)
    ax.set_xlabel("Candidate regulator"); ax.set_ylabel("Feature program")
    ax.set_title(request.title or "Consensus TF target enrichment")
    fig.colorbar(image, ax=ax, label="-log10(adjusted p-value) × database support")
    return fig


def _network_figure(nodes: pd.DataFrame, edges: pd.DataFrame,
                    request: TFRegulatoryNetworkPlotInput):
    fig, ax = plt.subplots(figsize=(request.width, request.height))
    if nodes.empty or edges.empty:
        ax.text(.5, .5, "No consensus-supported network", ha="center", va="center")
        ax.axis("off")
        return fig
    layer_order = {"TF": 0, "target_gene": 1, "program": 2}
    positions = {}
    for node_type, group in nodes.groupby("node_type", sort=False):
        ordered = group.sort_values("node_id", kind="mergesort")
        ys = np.linspace(.92, .08, len(ordered)) if len(ordered) > 1 else np.array([.5])
        for node_id, y in zip(ordered.node_id, ys):
            positions[node_id] = (layer_order[node_type], y)
    edge_colors = {"TF_to_target": "#9A9A9A", "target_to_program": "#78B7B2"}
    for row in edges.itertuples():
        x1, y1 = positions[row.source]; x2, y2 = positions[row.target]
        ax.plot([x1, x2], [y1, y2], color=edge_colors[row.edge_type],
                linewidth=.5 + .35 * float(row.edge_weight), alpha=.42, zorder=1)
    colors = {"TF": "#E76F51", "target_gene": "#8078B8", "program": "#2A9D8F"}
    sizes = {"TF": 90, "target_gene": 38, "program": 110}
    for node_type, group in nodes.groupby("node_type", sort=False):
        xy = np.array([positions[x] for x in group.node_id])
        ax.scatter(xy[:, 0], xy[:, 1], s=sizes[node_type], c=colors[node_type],
                   edgecolors="white", linewidths=.5, label=node_type, zorder=3)
        if request.show_labels:
            for row in group.itertuples():
                x, y = positions[row.node_id]
                ax.text(x + .025, y, str(row.label), fontsize=request.font_size,
                        va="center", ha="left", zorder=4)
    ax.set_xlim(-.12, 2.75); ax.set_ylim(0, 1)
    ax.set_xticks([0, 1, 2], ["Candidate TF", "Target gene", "Feature program"])
    ax.set_yticks([])
    ax.set_title(request.title or "TF → target gene → feature program")
    ax.legend(frameon=False, loc="lower center", ncol=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def plot_tf_regulatory_network(
    request: TFRegulatoryNetworkPlotInput,
) -> TFRegulatoryNetworkPlotOutput:
    """Render canonical artifacts without recomputing analysis or membership."""
    paths = resolve_table_artifacts(request.analysis_result, (
        "heatmap_source_table", "tf_target_program_network_nodes",
        "tf_target_program_network_edges"))
    heatmap = _read_canonical_tsv(paths["heatmap_source_table"], (
        "program_id", "feature_set_id", "cell_state", "contrast", "feature_direction",
        "tf", "number_of_supporting_databases", "minimum_adjusted_p_value",
        "maximum_odds_ratio", "consensus_status", "display_score"))
    nodes = _read_canonical_tsv(paths["tf_target_program_network_nodes"], (
        "node_id", "node_type", "label"))
    edges = _read_canonical_tsv(paths["tf_target_program_network_edges"], (
        "edge_id", "source", "target", "edge_type", "program_id",
        "minimum_adjusted_p_value", "edge_weight"))
    display_nodes, display_edges, rule = _display_subset(nodes, edges, request)
    display_dir = request.output_dir / "display_tables"
    node_path = display_dir / "tf_network_display_nodes.csv"
    edge_path = display_dir / "tf_network_display_edges.csv"
    write_csv_atomic(display_nodes, node_path)
    write_csv_atomic(display_edges, edge_path)
    figures = []
    for name, fig in (
        ("consensus_tf_heatmap", _heatmap_figure(heatmap, request)),
        ("tf_target_program_network", _network_figure(display_nodes, display_edges, request)),
    ):
        try:
            for path in _save(fig, request.output_dir / "figures" / name,
                              request.formats, request.dpi):
                media = {".pdf": "application/pdf", ".svg": "image/svg+xml", ".png": "image/png"}[path.suffix]
                figures.append(ArtifactReference(f"{name}_{path.suffix[1:]}", str(path),
                    ArtifactCategory.DESCRIPTIVE, media))
        finally:
            plt.close(fig)
    provenance_path = request.output_dir / "plot_provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    value = {"upstream_capability_id": "CAP-TF-001",
        "upstream_cache_key": request.analysis_result.cache_key,
        "layout_algorithm": request.layout_algorithm, "layout_seed": request.layout_seed,
        "display_selection_rule": rule, "maximum_displayed_tfs": request.maximum_displayed_tfs,
        "maximum_displayed_targets": request.maximum_displayed_targets}
    temporary = provenance_path.with_name(f".{provenance_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, provenance_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    display = (
        ArtifactReference("tf_network_display_nodes", str(node_path), ArtifactCategory.INPUT_DERIVED, "text/csv"),
        ArtifactReference("tf_network_display_edges", str(edge_path), ArtifactCategory.INPUT_DERIVED, "text/csv"),
    )
    plot_provenance = ArtifactReference("tf_network_plot_provenance", str(provenance_path),
                                        ArtifactCategory.PROVENANCE, "application/json")
    return TFRegulatoryNetworkPlotOutput("CAP-TF-001", request.analysis_result.cache_key,
                                         tuple(figures), display, plot_provenance)
