"""Statistics-free deterministic heatmap for completed CAP-TF-002 artifacts."""
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


@dataclass(frozen=True)
class TFActivityPlotInput:
    analysis_result: CapabilityResult
    output_dir: Path
    formats: tuple[Literal["pdf", "svg", "png"], ...] = ("pdf", "svg", "png")
    dpi: int = 300
    width: float = 11.0
    height: float = 7.0
    maximum_displayed_tfs: int = 40
    title: str | None = None
    font_size: float = 8.0

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.analysis_result.capability_id != "CAP-TF-002":
            raise ValueError("TF activity plotting requires CAP-TF-002")
        if self.analysis_result.status not in {
            CapabilityStatus.COMPLETED, CapabilityStatus.COMPLETED_WITH_WARNINGS,
        }:
            raise ValueError("TF activity plotting requires a completed analysis result")
        if not self.formats or len(set(self.formats)) != len(self.formats):
            raise ValueError("formats must be nonempty and unique")
        if self.dpi < 72 or self.width <= 0 or self.height <= 0:
            raise ValueError("plot dimensions and DPI must be positive")
        if self.maximum_displayed_tfs < 1:
            raise ValueError("maximum_displayed_tfs must be positive")


@dataclass(frozen=True)
class TFActivityPlotOutput:
    upstream_capability_id: str
    upstream_cache_key: str
    figures: tuple[ArtifactReference, ...]
    display_tables: tuple[ArtifactReference, ...]
    plot_provenance: ArtifactReference

    @property
    def artifacts(self) -> tuple[ArtifactReference, ...]:
        return self.figures + self.display_tables + (self.plot_provenance,)


def _read_heatmap(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    required = (
        "program_id", "cell_state", "contrast", "tf", "consensus_direction",
        "minimum_adjusted_p_value", "number_of_significant_supporting_resources",
        "directional_consensus_status", "display_activity_score",
    )
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(
            f"canonical plotting table {path.name} is missing columns: {missing}")
    return table


def _display_subset(
    table: pd.DataFrame, maximum: int,
) -> tuple[pd.DataFrame, str]:
    if table.empty:
        return table.copy(), "empty canonical heatmap source"
    rank = table.groupby("tf").agg(
        represented_programs=("program_id", "nunique"),
        minimum_adjusted_p_value=("minimum_adjusted_p_value", "min"),
        strongest_absolute_score=("display_activity_score",
                                  lambda values: values.abs().max()),
    ).sort_values(
        ["represented_programs", "minimum_adjusted_p_value",
         "strongest_absolute_score"],
        ascending=[False, True, False], kind="mergesort")
    keep = set(rank.head(maximum).index)
    display = table.loc[table.tf.isin(keep)].sort_values(
        ["program_id", "tf"], kind="mergesort").reset_index(drop=True)
    rule = (
        "Display-only TF ranking: directional-consensus program count descending, "
        "minimum adjusted p-value ascending, strongest absolute activity score "
        f"descending; retain at most {maximum} TFs. Full canonical tables are unchanged."
    )
    return display, rule


def _figure(table: pd.DataFrame, request: TFActivityPlotInput):
    fig, ax = plt.subplots(figsize=(request.width, request.height))
    if table.empty:
        ax.text(
            .5, .55, "No directional-consensus TF activity",
            ha="center", va="center", fontsize=12)
        ax.text(
            .5, .43,
            "No inferred increased/decreased activity passed the consensus definition",
            ha="center", va="center", fontsize=9)
        ax.axis("off")
        return fig
    source = table.copy()
    source["_program_label"] = (
        source.program_id.astype(str) + " | " + source.cell_state.astype(str)
        + " | " + source.contrast.astype(str))
    matrix = source.pivot_table(
        index="tf", columns="_program_label", values="display_activity_score",
        aggfunc="first")
    tf_order = source.groupby("tf").agg(
        recurrence=("program_id", "nunique"),
        minimum_p=("minimum_adjusted_p_value", "min"),
    ).sort_values(
        ["recurrence", "minimum_p"], ascending=[False, True],
        kind="mergesort").index
    matrix = matrix.reindex(index=tf_order, columns=sorted(matrix.columns))
    values = matrix.to_numpy(float)
    limit = max(float(np.nanmax(np.abs(values))), 1e-12)
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#D9D9D9")
    image = ax.imshow(
        masked, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit,
        interpolation="nearest")
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=request.font_size)
    ax.set_xticks(
        range(len(matrix.columns)), matrix.columns, rotation=55, ha="right",
        fontsize=request.font_size)
    ax.set_ylabel("Transcription factor")
    ax.set_xlabel("Feature program | cell state | declared contrast")
    ax.set_title(request.title or "Directional-consensus signed TF activity")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(
        "Inferred TF activity (negative: decreased; positive: increased)")
    return fig


def _save(fig, stem: Path, formats: tuple[str, ...], dpi: int) -> tuple[Path, ...]:
    outputs = []
    for extension in formats:
        path = stem.with_suffix("." + extension)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.{extension}")
        metadata = (
            {"CreationDate": _FIXED_DATE, "ModDate": _FIXED_DATE,
             "Creator": "cellstate"}
            if extension == "pdf"
            else {"Date": "2000-01-01T00:00:00Z", "Creator": "cellstate"})
        try:
            fig.savefig(
                temporary, format=extension, dpi=dpi, bbox_inches="tight",
                metadata=metadata)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        outputs.append(path)
    return tuple(outputs)


def plot_tf_activity(request: TFActivityPlotInput) -> TFActivityPlotOutput:
    """Render canonical CAP-TF-002 results without recomputing statistics."""
    paths = resolve_table_artifacts(
        request.analysis_result, ("heatmap_source_table",))
    canonical = _read_heatmap(paths["heatmap_source_table"])
    display, rule = _display_subset(canonical, request.maximum_displayed_tfs)
    display_path = request.output_dir / "display_tables" / "tf_activity_heatmap_display.csv"
    write_csv_atomic(display, display_path)
    fig = _figure(display, request)
    figures = []
    try:
        for path in _save(
            fig, request.output_dir / "figures" / "signed_tf_activity_heatmap",
            request.formats, request.dpi,
        ):
            media = {
                ".pdf": "application/pdf", ".svg": "image/svg+xml",
                ".png": "image/png"}[path.suffix]
            figures.append(ArtifactReference(
                f"signed_tf_activity_heatmap_{path.suffix[1:]}", str(path),
                ArtifactCategory.DESCRIPTIVE, media))
    finally:
        plt.close(fig)
    provenance_path = request.output_dir / "plot_provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "upstream_capability_id": "CAP-TF-002",
        "upstream_cache_key": request.analysis_result.cache_key,
        "canonical_source_artifact": str(paths["heatmap_source_table"]),
        "display_selection_rule": rule,
        "maximum_displayed_tfs": request.maximum_displayed_tfs,
        "scientific_recomputation": False,
        "interpretation": (
            "positive means inferred increased TF activity in condition A relative "
            "to condition B; negative means inferred decreased activity"),
    }
    temporary = provenance_path.with_name(
        f".{provenance_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        os.replace(temporary, provenance_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return TFActivityPlotOutput(
        "CAP-TF-002", request.analysis_result.cache_key, tuple(figures),
        (ArtifactReference(
            "tf_activity_heatmap_display", str(display_path),
            ArtifactCategory.INPUT_DERIVED, "text/csv"),),
        ArtifactReference(
            "tf_activity_plot_provenance", str(provenance_path),
            ArtifactCategory.PROVENANCE, "application/json"))
