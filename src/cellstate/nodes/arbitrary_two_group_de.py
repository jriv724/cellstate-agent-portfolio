"""CAP-DESEQ-003 arbitrary two-group sample-level pseudobulk DESeq2."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from ..cache import (
    build_cache_manifest,
    load_complete_manifest,
    make_cache_key,
    write_manifest_atomic,
)
from ..context import AnalysisContext
from ..provenance import build_provenance, write_provenance_atomic
from ..schemas.arbitrary_two_group_de import (
    CAP_DESEQ_003_SCHEMA_VERSION,
    CAP_DESEQ_003_VERSION,
    ArbitraryTwoGroupDEInput,
    ArbitraryTwoGroupDEOutput,
    DEArtifactReference,
    DEWarning,
    TwoGroupDesignAssessment,
)
from ..schemas.common import StructuredWarning, WarningSeverity
from ..validation.estimability import validate_design_estimability
from ..validation.metadata import validate_required_columns


CAPABILITY_ID = "CAP-DESEQ-003"
CAPABILITY_TITLE = "Arbitrary Two-Group Pseudobulk Differential Expression"
NODE_VERSION = CAP_DESEQ_003_VERSION
CACHE_SCHEMA_VERSION = 1
SOURCE_FILES = (
    "src/cellstate/nodes/arbitrary_two_group_de.py",
    "src/cellstate/nodes/arbitrary_two_group_de_fit.R",
)
SOURCE_LOCATIONS = (
    "CAP-DESEQ-003 validated Python input/design adapter",
    "CAP-DESEQ-003 dedicated R/DESeq2 Wald-test worker",
)
EXPLORATORY_CONFOUNDING_WARNING = (
    "EXPLORATORY CONFOUNDED-DESIGN ANALYSIS: group and dataset are not "
    "independently identifiable. Features are conserved across dataset-level "
    "leave-one-out analyses but may still reflect systematic dataset effects."
)


class _ContractBlock(ValueError):
    def __init__(
        self,
        message: str,
        *,
        group_counts: dict[str, int] | None = None,
        datasets: list[str] | None = None,
        input_gene_count: int = 0,
        filter_summary: dict[str, Any] | None = None,
        warnings: tuple[DEWarning, ...] = (),
    ) -> None:
        super().__init__(message)
        self.group_counts = group_counts
        self.datasets = datasets
        self.input_gene_count = input_gene_count
        self.filter_summary = filter_summary
        self.warnings = warnings


@dataclass(frozen=True)
class _PreparedInput:
    counts: pd.DataFrame
    metadata: pd.DataFrame
    assessment: TwoGroupDesignAssessment
    input_gene_count: int
    filter_summary: dict[str, Any]
    warnings: tuple[DEWarning, ...]
    adjusted_design_confounded: bool = False


def _signature(path: Path) -> str:
    if not path.is_file():
        return f"missing:{path}"
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _separator(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_tsv(table: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        table.to_csv(
            temporary, sep="\t", index=index, lineterminator="\n", na_rep="NA"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        table.to_csv(temporary, index=False, lineterminator="\n", na_rep="NA")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _warning(
    code: str,
    message: str,
    severity: str = "warning",
    context: dict[str, Any] | None = None,
) -> DEWarning:
    return DEWarning(
        code=code, message=message, severity=severity, context=context or {}
    )


def _structured(warnings: tuple[DEWarning, ...]) -> tuple[StructuredWarning, ...]:
    return tuple(
        StructuredWarning(
            item.code,
            item.message,
            WarningSeverity(item.severity),
            item.context,
        )
        for item in warnings
    )


def _blocked_assessment(
    request: ArbitraryTwoGroupDEInput,
    reason: str,
    *,
    counts: dict[str, int] | None = None,
    datasets: list[str] | None = None,
    warnings: tuple[DEWarning, ...] = (),
) -> TwoGroupDesignAssessment:
    return TwoGroupDesignAssessment(
        group_replicate_counts=counts or {
            request.group_a: 0,
            request.group_b: 0,
        },
        represented_datasets=datasets or [],
        shared_datasets=[],
        design_formula=None,
        design_columns=[],
        design_rank=None,
        design_column_count=None,
        residual_degrees_of_freedom=None,
        full_rank=False,
        group_coefficient=f"{request.group_a} - {request.group_b}",
        estimable=False,
        warnings=list(warnings),
        blocking_reasons=[reason],
    )


def validate_arbitrary_two_group_inputs(
    request: ArbitraryTwoGroupDEInput,
) -> _PreparedInput:
    """Validate and prepare DESeq2 inputs without performing inference."""
    if not request.count_matrix_path.is_file():
        raise _ContractBlock(f"count matrix is missing: {request.count_matrix_path}")
    if not request.sample_metadata_path.is_file():
        raise _ContractBlock(
            f"sample metadata is missing: {request.sample_metadata_path}"
        )
    with request.count_matrix_path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle, delimiter=_separator(request.count_matrix_path)))
    raw_replicates = [value.strip() for value in header[1:]]
    if (
        any(not value for value in raw_replicates)
        or len(raw_replicates) != len(set(raw_replicates))
    ):
        raise _ContractBlock("count-matrix replicate identifiers must be unique")
    counts = pd.read_csv(
        request.count_matrix_path,
        sep=_separator(request.count_matrix_path),
        index_col=0,
        keep_default_na=False,
    )
    metadata = pd.read_csv(
        request.sample_metadata_path,
        sep=_separator(request.sample_metadata_path),
        keep_default_na=False,
    )
    required = [
        request.replicate_column,
        request.group_column,
        request.dataset_column,
        request.cell_count_column,
    ]
    if request.patient_column:
        required.append(request.patient_column)
    try:
        validate_required_columns(metadata, required)
    except ValueError as exc:
        raise _ContractBlock(str(exc)) from exc

    if counts.empty or counts.shape[1] == 0:
        raise _ContractBlock("count matrix must contain genes and replicates")
    genes = counts.index.astype(str)
    if (genes.str.strip() == "").any() or genes.duplicated().any():
        raise _ContractBlock("gene identifiers must be nonblank and unique")
    replicates = counts.columns.astype(str)
    if (replicates.str.strip() == "").any() or replicates.duplicated().any():
        raise _ContractBlock("count-matrix replicate identifiers must be unique")
    metadata_replicates = metadata[request.replicate_column].astype(str)
    if metadata_replicates.str.strip().eq("").any():
        raise _ContractBlock("metadata replicate identifiers must be nonblank")
    if metadata_replicates.duplicated().any():
        raise _ContractBlock("metadata must contain one row per independent replicate")
    if set(replicates) != set(metadata_replicates):
        raise _ContractBlock("count matrix and sample metadata are misaligned")
    metadata = (
        metadata.set_index(request.replicate_column)
        .loc[list(replicates)]
        .rename_axis(request.replicate_column)
        .reset_index()
    )

    if request.patient_column:
        patients = metadata[request.patient_column].astype(str)
        if patients.str.strip().eq("").any():
            raise _ContractBlock("patient identifiers must be nonblank")
        repeated = patients.value_counts()
        if (repeated > 1).any():
            raise _ContractBlock(
                "multiple pseudobulk replicates map to one patient; aggregate to "
                "the accepted independent unit before CAP-DESEQ-003"
            )

    selected = metadata[request.group_column].isin(
        [request.group_a, request.group_b]
    )
    metadata = metadata.loc[selected].copy()
    if metadata.empty:
        raise _ContractBlock("no metadata rows match the requested groups")
    counts = counts.loc[:, metadata[request.replicate_column].astype(str)]
    cell_counts = pd.to_numeric(
        metadata[request.cell_count_column], errors="coerce"
    )
    if cell_counts.isna().any() or not np.isfinite(cell_counts).all():
        raise _ContractBlock("contributing-cell counts must be finite numeric values")
    eligible = cell_counts.ge(request.minimum_cells_per_replicate)
    excluded = metadata.loc[~eligible, request.replicate_column].astype(str).tolist()
    warnings: list[DEWarning] = []
    if excluded:
        warnings.append(
            _warning(
                "PSEUDOBULK_REPLICATES_BELOW_CELL_THRESHOLD",
                "Replicates below the production 100-cell threshold were excluded.",
                context={
                    "replicates": excluded,
                    "minimum_cells_per_replicate": request.minimum_cells_per_replicate,
                },
            )
        )
    metadata = metadata.loc[eligible].copy().reset_index(drop=True)
    counts = counts.loc[:, metadata[request.replicate_column].astype(str)]

    group_counts = {
        group: int(
            metadata.loc[
                metadata[request.group_column].eq(group),
                request.replicate_column,
            ].nunique()
        )
        for group in (request.group_a, request.group_b)
    }
    insufficient = {
        group: count
        for group, count in group_counts.items()
        if count < request.minimum_replicates_per_group
    }
    if insufficient:
        raise _ContractBlock(
            f"insufficient independent replication after cell QC: {insufficient}",
            group_counts=group_counts,
            datasets=sorted(
                metadata[request.dataset_column].astype(str).unique().tolist()
            ),
            input_gene_count=len(counts),
            warnings=tuple(warnings),
        )

    try:
        numeric = counts.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise _ContractBlock("raw counts must be numeric") from exc
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise _ContractBlock("raw counts must be finite")
    if (values < 0).any():
        raise _ContractBlock("raw counts must be nonnegative")
    if not np.all(values == np.floor(values)):
        raise _ContractBlock("raw counts must be integer-like")
    numeric = numeric.astype(np.int64)
    if (numeric.sum(axis=0) <= 0).any():
        invalid = numeric.columns[numeric.sum(axis=0).le(0)].astype(str).tolist()
        raise _ContractBlock(f"retained replicates have zero library size: {invalid}")

    input_gene_count = len(numeric)
    keep = numeric.ge(request.gene_minimum_count).sum(axis=1).ge(
        request.gene_minimum_replicates
    )
    filtered = numeric.loc[keep].copy()
    filter_summary = {
        "input_gene_count": input_gene_count,
        "retained_gene_count": len(filtered),
        "filtered_gene_count": int(input_gene_count - len(filtered)),
        "minimum_count": request.gene_minimum_count,
        "minimum_qualifying_replicates": request.gene_minimum_replicates,
    }
    if filtered.empty:
        raise _ContractBlock(
            "no genes survive count >= 10 in at least 3 replicates",
            group_counts=group_counts,
            datasets=sorted(
                metadata[request.dataset_column].astype(str).unique().tolist()
            ),
            input_gene_count=input_gene_count,
            filter_summary=filter_summary,
            warnings=tuple(warnings),
        )

    represented = sorted(
        metadata[request.dataset_column].astype(str).unique().tolist()
    )
    shared = sorted(
        dataset
        for dataset, subset in metadata.groupby(request.dataset_column)
        if set(subset[request.group_column]) == {request.group_a, request.group_b}
    )
    group_indicator = metadata[request.group_column].eq(request.group_a).astype(float)
    design_columns = ["Intercept"]
    columns = [np.ones(len(metadata), dtype=float)]
    if len(represented) == 1:
        formula = "~ group"
    else:
        formula = "~ dataset + group"
        if not shared and request.confounded_design_policy == "block":
            raise _ContractBlock(
                "group is completely confounded with dataset; no dataset contains "
                "eligible replicates from both groups",
                group_counts=group_counts,
                datasets=represented,
                input_gene_count=input_gene_count,
                filter_summary=filter_summary,
                warnings=tuple(warnings),
            )
        if not shared:
            formula = "~ group"
            warnings.append(_warning(
                "EXPLORATORY_CONFOUNDED_DESIGN",
                EXPLORATORY_CONFOUNDING_WARNING,
                "warning",
                {"policy": request.confounded_design_policy},
            ))
        else:
            categorical = pd.Categorical(
                metadata[request.dataset_column].astype(str),
                categories=represented,
            )
            for dataset in represented[1:]:
                columns.append((categorical == dataset).astype(float))
                design_columns.append(f"dataset[{dataset}]")
    columns.append(group_indicator.to_numpy())
    design_columns.append(f"group[{request.group_a} vs {request.group_b}]")
    matrix = np.column_stack(columns)
    try:
        estimability = validate_design_estimability(
            matrix, term_names=design_columns
        )
    except ValueError as exc:
        raise _ContractBlock(
            str(exc),
            group_counts=group_counts,
            datasets=represented,
            input_gene_count=input_gene_count,
            filter_summary=filter_summary,
            warnings=tuple(warnings),
        ) from exc

    assessment = TwoGroupDesignAssessment(
        group_replicate_counts=group_counts,
        represented_datasets=represented,
        shared_datasets=shared,
        design_formula=formula,
        design_columns=design_columns,
        design_rank=estimability["rank"],
        design_column_count=estimability["columns"],
        residual_degrees_of_freedom=estimability["residual_df"],
        full_rank=True,
        group_coefficient=f"{request.group_a} - {request.group_b}",
        estimable=True,
        warnings=warnings,
        blocking_reasons=[],
    )
    return _PreparedInput(
        counts=filtered,
        metadata=metadata,
        assessment=assessment,
        input_gene_count=input_gene_count,
        filter_summary=filter_summary,
        warnings=tuple(warnings),
        adjusted_design_confounded=(len(represented) > 1 and not shared),
    )


def validate_deseq2_result_table(
    table: pd.DataFrame,
    *,
    expected_genes: pd.Index,
    request: ArbitraryTwoGroupDEInput,
) -> pd.DataFrame:
    required = [
        "gene",
        "baseMean",
        "log2FoldChange",
        "lfcSE",
        "stat",
        "pvalue",
        "padj",
        "signed_statistic",
    ]
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"DESeq2 result table is missing columns: {missing}")
    if table.gene.astype(str).duplicated().any():
        raise ValueError("DESeq2 result genes must be unique")
    if set(table.gene.astype(str)) != set(expected_genes.astype(str)):
        raise ValueError("DESeq2 results must contain every retained gene exactly once")
    for column in required[1:]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    finite_required = table[["baseMean", "log2FoldChange", "lfcSE", "stat"]]
    if not np.isfinite(finite_required.to_numpy(float)).all():
        raise ValueError("DESeq2 primary estimates must be finite")
    if not np.allclose(
        table["signed_statistic"].to_numpy(float),
        table["stat"].to_numpy(float),
        rtol=0,
        atol=0,
    ):
        raise ValueError("signed_statistic must equal the DESeq2 Wald statistic")
    table["significant"] = table.padj.notna() & table.padj.lt(request.alpha)
    table["direction"] = np.select(
        [
            table.log2FoldChange.gt(0),
            table.log2FoldChange.lt(0),
        ],
        ["higher_in_group_a", "higher_in_group_b"],
        default="no_direction",
    )
    return table.loc[
        :,
        [
            *required,
            "significant",
            "direction",
        ],
    ].sort_values("gene", kind="mergesort").reset_index(drop=True)


def summarize_exploratory_lodo(
    full_results: pd.DataFrame,
    fold_results: pd.DataFrame,
    *,
    eligible_fold_count: int,
    request: ArbitraryTwoGroupDEInput,
) -> pd.DataFrame:
    """Summarize successful fold estimates without treating folds as independent."""
    rows: list[dict[str, Any]] = []
    for full in full_results.sort_values("gene", kind="mergesort").to_dict("records"):
        gene = str(full["gene"])
        folds = fold_results.loc[fold_results.gene.astype(str).eq(gene)].copy()
        folds = folds.loc[np.isfinite(pd.to_numeric(
            folds.get("log2FoldChange", pd.Series(dtype=float)), errors="coerce"
        ))]
        effects = pd.to_numeric(folds.get("log2FoldChange"), errors="coerce")
        n = int(len(effects))
        full_effect = float(full["log2FoldChange"])
        full_sign = int(np.sign(full_effect))
        matching = effects.map(lambda value: int(np.sign(value)) == full_sign) if n else pd.Series(dtype=bool)
        direction_fraction = float(matching.mean()) if n else 0.0
        median = float(effects.median()) if n else np.nan
        opposite_beyond_tolerance = bool(
            ((effects * full_sign) < -request.lodo_max_opposite_log2fc).any()
        ) if n and full_sign else bool((effects.abs() > request.lodo_max_opposite_log2fc).any())
        changes = (effects - full_effect).abs() if n else pd.Series(dtype=float)
        largest = ""
        if n:
            maximum = float(changes.max())
            largest = ";".join(sorted(
                folds.loc[changes.eq(maximum), "omitted_dataset"].astype(str).unique()
            ))
        full_significant = bool(pd.notna(full.get("padj")) and float(full["padj"]) < request.lodo_full_analysis_fdr)
        reasons: list[str] = []
        if not full_significant: reasons.append("not_significant_in_full_unadjusted_analysis")
        if n < request.lodo_min_estimable_folds: reasons.append("insufficient_estimable_folds")
        if direction_fraction < request.lodo_min_direction_fraction: reasons.append("direction_unstable")
        if not n or abs(median) < request.lodo_min_median_abs_log2fc: reasons.append("median_effect_below_threshold")
        if opposite_beyond_tolerance: reasons.append("omitted_dataset_direction_reversal")
        rows.append({
            "gene": gene,
            "full_log2FoldChange": full_effect,
            "full_pvalue": full.get("pvalue"),
            "full_padj": full.get("padj"),
            "eligible_fold_count": eligible_fold_count,
            "estimable_fold_count": n,
            "direction_consistency_fraction": direction_fraction,
            "median_lodo_log2FoldChange": median,
            "minimum_lodo_log2FoldChange": float(effects.min()) if n else np.nan,
            "maximum_lodo_log2FoldChange": float(effects.max()) if n else np.nan,
            "lodo_log2FoldChange_iqr": float(effects.quantile(.75) - effects.quantile(.25)) if n else np.nan,
            "nominal_significant_fold_fraction": float(pd.to_numeric(folds.get("pvalue"), errors="coerce").lt(.05).mean()) if n else 0.0,
            "fdr_significant_fold_fraction": float(pd.to_numeric(folds.get("padj"), errors="coerce").lt(request.lodo_full_analysis_fdr).mean()) if n else 0.0,
            "largest_effect_change_omitted_datasets": largest,
            "conserved": not reasons,
            "exclusion_reason": ";".join(reasons),
        })
    return pd.DataFrame(rows)


def _run_worker(
    request: ArbitraryTwoGroupDEInput,
    counts_path: Path,
    metadata_path: Path,
    result_path: Path,
    runtime_path: Path,
    *,
    design_mode: str,
) -> None:
    worker = Path(__file__).with_name("arbitrary_two_group_de_fit.R")
    subprocess.run([
        str(request.rscript_path), str(worker), str(counts_path),
        str(metadata_path), str(result_path), str(runtime_path),
        request.replicate_column, request.group_column, request.dataset_column,
        request.group_a, request.group_b, design_mode,
    ], check=True, capture_output=True, text=True)


def _artifact(
    logical_name: str,
    path: Path,
    category: str,
    media_type: str,
) -> DEArtifactReference:
    return DEArtifactReference(
        logical_name=logical_name,
        path=str(path),
        category=category,
        media_type=media_type,
    )


def _output_from_summary(
    summary: dict[str, Any],
    assessment: TwoGroupDesignAssessment,
    artifacts: list[DEArtifactReference],
    *,
    cache_key: str,
    cache_hit: bool,
    provenance_path: Path,
    manifest_path: Path,
) -> ArbitraryTwoGroupDEOutput:
    return ArbitraryTwoGroupDEOutput(
        terminal_status=summary["terminal_status"],
        cache_key=cache_key,
        cache_hit=cache_hit,
        comparison_direction=summary["comparison_direction"],
        evidence_class=summary.get("evidence_class", "adjusted_inference"),
        design_assessment=assessment,
        input_gene_count=summary["input_gene_count"],
        retained_gene_count=summary["retained_gene_count"],
        tested_gene_count=summary["tested_gene_count"],
        significant_gene_count=summary["significant_gene_count"],
        upregulated_in_group_a_count=summary["upregulated_in_group_a_count"],
        upregulated_in_group_b_count=summary["upregulated_in_group_b_count"],
        group_replicate_counts=assessment.group_replicate_counts,
        artifacts=artifacts,
        warnings=[DEWarning.model_validate(item) for item in summary["warnings"]],
        blocking_reasons=summary["blocking_reasons"],
        provenance_path=str(provenance_path),
        cache_manifest_path=str(manifest_path),
    )


def run_arbitrary_two_group_de(
    request: ArbitraryTwoGroupDEInput,
    context: AnalysisContext,
) -> ArbitraryTwoGroupDEOutput:
    """Validate, run the dedicated DESeq2 worker, and persist canonical results."""
    signatures = {
        "counts": _signature(request.count_matrix_path),
        "metadata": _signature(request.sample_metadata_path),
        "upstream_provenance": (
            _signature(request.upstream_provenance_path)
            if request.upstream_provenance_path
            else None
        ),
    }
    parameters = {**request.parameters(), "input_signatures": signatures}
    cache_key = make_cache_key(
        capability_id=CAPABILITY_ID,
        node_version=NODE_VERSION,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        dataset_signature=context.dataset_signature,
        parameters=parameters,
    )
    cache_dir = context.cache_root / CAPABILITY_ID.lower() / cache_key
    manifest_path = cache_dir / "cache_manifest.json"
    cached = load_complete_manifest(manifest_path, cache_key)
    if cached:
        named = {Path(path).name: Path(path) for path in cached.output_files}
        assessment = TwoGroupDesignAssessment.model_validate_json(
            named["deseq2_design_assessment.json"].read_text(encoding="utf-8")
        )
        summary = json.loads(
            named["deseq2_summary.json"].read_text(encoding="utf-8")
        )
        artifacts = [
            _artifact(
                "deseq2_results",
                named["deseq2_results.tsv"],
                "inferential",
                "text/tab-separated-values",
            ),
            _artifact(
                "input_counts",
                named["deseq2_input_counts.tsv"],
                "input-derived",
                "text/tab-separated-values",
            ),
            _artifact(
                "design_assessment",
                named["deseq2_design_assessment.json"],
                "QC",
                "application/json",
            ),
            _artifact(
                "input_sample_metadata",
                named["deseq2_input_sample_metadata.tsv"],
                "input-derived",
                "text/tab-separated-values",
            ),
            _artifact(
                "gene_filter_summary",
                named["deseq2_gene_filter_summary.json"],
                "QC",
                "application/json",
            ),
            _artifact(
                "summary",
                named["deseq2_summary.json"],
                "descriptive",
                "application/json",
            ),
            _artifact(
                "runtime_metadata",
                named["deseq2_runtime.tsv"],
                "provenance",
                "text/tab-separated-values",
            ),
            _artifact(
                "provenance",
                named["provenance.json"],
                "provenance",
                "application/json",
            ),
            _artifact(
                "cache_manifest",
                manifest_path,
                "manifest",
                "application/json",
            ),
        ]
        optional_cached = {
            "full_unadjusted_deseq2_results.csv": ("full_unadjusted_deseq2_results", "inferential", "text/csv"),
            "lodo_fold_results.csv": ("lodo_fold_results", "inferential", "text/csv"),
            "lodo_feature_summary.csv": ("lodo_feature_summary", "descriptive", "text/csv"),
            "conserved_features.csv": ("conserved_features", "inferential", "text/csv"),
            "skipped_lodo_folds.csv": ("skipped_lodo_folds", "QC", "text/csv"),
            "exploratory_lodo_summary.json": ("exploratory_lodo_summary", "QC", "application/json"),
        }
        for filename, (logical_name, category, media_type) in optional_cached.items():
            if filename in named:
                artifacts.append(_artifact(logical_name, named[filename], category, media_type))
        return _output_from_summary(
            summary,
            assessment,
            artifacts,
            cache_key=cache_key,
            cache_hit=True,
            provenance_path=named["provenance.json"],
            manifest_path=manifest_path,
        )

    output_base = (
        request.output_directory
        if request.output_directory is not None
        else context.capability_output_dir(CAPABILITY_ID)
    )
    if not output_base.is_absolute():
        raise ValueError("CAP-DESEQ-003 output_directory must be absolute")
    output_dir = output_base / cache_key
    output_dir.mkdir(parents=True, exist_ok=True)
    design_path = output_dir / "deseq2_design_assessment.json"
    metadata_path = output_dir / "deseq2_input_sample_metadata.tsv"
    counts_path = output_dir / "deseq2_input_counts.tsv"
    filter_path = output_dir / "deseq2_gene_filter_summary.json"
    result_path = output_dir / "deseq2_results.tsv"
    runtime_path = output_dir / "deseq2_runtime.tsv"
    full_unadjusted_path = output_dir / "full_unadjusted_deseq2_results.csv"
    fold_results_path = output_dir / "lodo_fold_results.csv"
    feature_summary_path = output_dir / "lodo_feature_summary.csv"
    conserved_path = output_dir / "conserved_features.csv"
    skipped_folds_path = output_dir / "skipped_lodo_folds.csv"
    lodo_summary_path = output_dir / "exploratory_lodo_summary.json"
    summary_path = output_dir / "deseq2_summary.json"
    provenance_path = output_dir / "provenance.json"

    prepared: _PreparedInput | None = None
    terminal_status = "blocked"
    blocking_reasons: list[str] = []
    warnings: tuple[DEWarning, ...] = ()
    input_gene_count = 0
    retained_gene_count = 0
    tested_gene_count = 0
    significant_gene_count = 0
    up_a = 0
    up_b = 0
    assessment: TwoGroupDesignAssessment
    evidence_class = "adjusted_inference"
    lodo_artifacts: list[DEArtifactReference] = []
    try:
        prepared = validate_arbitrary_two_group_inputs(request)
        assessment = prepared.assessment
        warnings = prepared.warnings
        input_gene_count = prepared.input_gene_count
        retained_gene_count = len(prepared.counts)
        _atomic_tsv(prepared.metadata, metadata_path)
        _atomic_tsv(prepared.counts, counts_path, index=True)
        _atomic_json(assessment.model_dump(mode="json"), design_path)
        _atomic_json(prepared.filter_summary, filter_path)
    except _ContractBlock as exc:
        reason = str(exc)
        blocking_reasons = [reason]
        warning = _warning("CAP_DESEQ_003_BLOCKED", reason, "error")
        warnings = (*exc.warnings, warning)
        input_gene_count = exc.input_gene_count
        if exc.filter_summary is not None:
            retained_gene_count = int(
                exc.filter_summary["retained_gene_count"]
            )
        assessment = _blocked_assessment(
            request,
            reason,
            counts=exc.group_counts,
            datasets=exc.datasets,
            warnings=warnings,
        )
        _atomic_json(assessment.model_dump(mode="json"), design_path)
        _atomic_tsv(pd.DataFrame(columns=[
            request.replicate_column,
            request.group_column,
            request.dataset_column,
            request.cell_count_column,
        ]), metadata_path)
        _atomic_json(
            exc.filter_summary or {
                "input_gene_count": 0,
                "retained_gene_count": 0,
                "filtered_gene_count": 0,
                "minimum_count": request.gene_minimum_count,
                "minimum_qualifying_replicates": request.gene_minimum_replicates,
            },
            filter_path,
        )
    else:
        try:
            exploratory = prepared.adjusted_design_confounded
            _run_worker(
                request, counts_path, metadata_path, result_path, runtime_path,
                design_mode="unadjusted_group" if exploratory else "standard",
            )
            raw_result = pd.read_csv(
                result_path, sep="\t", na_values=["NA"], keep_default_na=False
            )
            validated = validate_deseq2_result_table(
                raw_result,
                expected_genes=prepared.counts.index,
                request=request,
            )
            _atomic_tsv(validated, result_path)
            if exploratory:
                evidence_class = "exploratory_unadjusted"
                _atomic_csv(validated, full_unadjusted_path)
                fold_tables: list[pd.DataFrame] = []
                skipped: list[dict[str, Any]] = []
                eligible_fold_count = 0
                datasets = sorted(prepared.metadata[request.dataset_column].astype(str).unique())
                for fold_index, omitted in enumerate(datasets):
                    fold_metadata = prepared.metadata.loc[
                        ~prepared.metadata[request.dataset_column].astype(str).eq(omitted)
                    ].copy()
                    group_counts = {
                        group: int(fold_metadata.loc[
                            fold_metadata[request.group_column].eq(group),
                            request.replicate_column,
                        ].nunique()) for group in (request.group_a, request.group_b)
                    }
                    datasets_by_group = {
                        group: int(fold_metadata.loc[
                            fold_metadata[request.group_column].eq(group),
                            request.dataset_column,
                        ].astype(str).nunique()) for group in (request.group_a, request.group_b)
                    }
                    reasons = []
                    if min(group_counts.values()) < request.minimum_replicates_per_group:
                        reasons.append("insufficient_biological_replicates")
                    if (request.lodo_require_two_datasets_per_group
                            and min(datasets_by_group.values()) < 2):
                        reasons.append("fewer_than_two_datasets_in_a_group")
                    if reasons:
                        skipped.append({
                            "omitted_dataset": omitted, "status": "skipped",
                            "reason": ";".join(reasons),
                            "group_a_replicates": group_counts[request.group_a],
                            "group_b_replicates": group_counts[request.group_b],
                            "group_a_datasets": datasets_by_group[request.group_a],
                            "group_b_datasets": datasets_by_group[request.group_b],
                        })
                        continue
                    eligible_fold_count += 1
                    fold_counts = prepared.counts.loc[:, fold_metadata[request.replicate_column].astype(str)]
                    fold_gene_keep = fold_counts.ge(request.gene_minimum_count).sum(axis=1).ge(
                        request.gene_minimum_replicates
                    )
                    fold_counts = fold_counts.loc[fold_gene_keep].copy()
                    if fold_counts.empty:
                        skipped.append({
                            "omitted_dataset": omitted, "status": "skipped",
                            "reason": "no_genes_survive_fold_count_filter",
                            "group_a_replicates": group_counts[request.group_a],
                            "group_b_replicates": group_counts[request.group_b],
                            "group_a_datasets": datasets_by_group[request.group_a],
                            "group_b_datasets": datasets_by_group[request.group_b],
                        })
                        continue
                    fold_counts_file = output_dir / f".lodo_{fold_index}_counts.tsv"
                    fold_metadata_file = output_dir / f".lodo_{fold_index}_metadata.tsv"
                    fold_result_file = output_dir / f".lodo_{fold_index}_results.tsv"
                    fold_runtime_file = output_dir / f".lodo_{fold_index}_runtime.tsv"
                    _atomic_tsv(fold_counts, fold_counts_file, index=True)
                    _atomic_tsv(fold_metadata, fold_metadata_file)
                    try:
                        _run_worker(
                            request, fold_counts_file, fold_metadata_file,
                            fold_result_file, fold_runtime_file,
                            design_mode="unadjusted_group",
                        )
                        fold = validate_deseq2_result_table(
                            pd.read_csv(fold_result_file, sep="\t", na_values=["NA"], keep_default_na=False),
                            expected_genes=fold_counts.index, request=request,
                        )
                        fold.insert(0, "omitted_dataset", omitted)
                        fold.insert(1, "fold_status", "estimable")
                        fold_tables.append(fold)
                    except (OSError, subprocess.CalledProcessError, ValueError) as fold_exc:
                        skipped.append({
                            "omitted_dataset": omitted, "status": "failed",
                            "reason": "DESEQ2_FOLD_FAILURE",
                            "detail": (getattr(fold_exc, "stderr", None) or str(fold_exc))[-1000:],
                            "group_a_replicates": group_counts[request.group_a],
                            "group_b_replicates": group_counts[request.group_b],
                            "group_a_datasets": datasets_by_group[request.group_a],
                            "group_b_datasets": datasets_by_group[request.group_b],
                        })
                    finally:
                        for temporary in (fold_counts_file, fold_metadata_file, fold_result_file, fold_runtime_file):
                            temporary.unlink(missing_ok=True)
                fold_results = pd.concat(fold_tables, ignore_index=True) if fold_tables else pd.DataFrame(columns=[
                    "omitted_dataset", "fold_status", *validated.columns,
                ])
                skipped_table = pd.DataFrame(skipped, columns=[
                    "omitted_dataset", "status", "reason", "detail",
                    "group_a_replicates", "group_b_replicates",
                    "group_a_datasets", "group_b_datasets",
                ])
                feature_summary = summarize_exploratory_lodo(
                    validated, fold_results, eligible_fold_count=eligible_fold_count,
                    request=request,
                )
                estimable_folds = int(fold_results.omitted_dataset.nunique()) if len(fold_results) else 0
                robustness_sufficient = estimable_folds >= request.lodo_min_estimable_folds
                conserved = feature_summary.loc[feature_summary.conserved].copy() if robustness_sufficient else feature_summary.iloc[0:0].copy()
                _atomic_csv(fold_results, fold_results_path)
                _atomic_csv(feature_summary, feature_summary_path)
                _atomic_csv(conserved, conserved_path)
                _atomic_csv(skipped_table, skipped_folds_path)
                _atomic_json({
                    "confounded_design_policy": request.confounded_design_policy,
                    "design": "exploratory ~ group",
                    "robustness": "dataset-level LODO",
                    "warning": EXPLORATORY_CONFOUNDING_WARNING,
                    "total_folds": len(datasets),
                    "eligible_folds": eligible_fold_count,
                    "estimable_folds": estimable_folds,
                    "skipped_or_failed_folds": len(skipped),
                    "robustness_sufficient": robustness_sufficient,
                    "conserved_feature_count": int(len(conserved)),
                    "thresholds": {key: value for key, value in request.parameters().items() if key.startswith("lodo_")},
                }, lodo_summary_path)
                lodo_artifacts = [
                    _artifact("full_unadjusted_deseq2_results", full_unadjusted_path, "inferential", "text/csv"),
                    _artifact("lodo_fold_results", fold_results_path, "inferential", "text/csv"),
                    _artifact("lodo_feature_summary", feature_summary_path, "descriptive", "text/csv"),
                    _artifact("conserved_features", conserved_path, "inferential", "text/csv"),
                    _artifact("skipped_lodo_folds", skipped_folds_path, "QC", "text/csv"),
                    _artifact("exploratory_lodo_summary", lodo_summary_path, "QC", "application/json"),
                ]
                if not robustness_sufficient:
                    terminal_status = "insufficient_robustness"
                    evidence_class = "exploratory_unadjusted"
                    warnings = (*warnings, _warning(
                        "INSUFFICIENT_LODO_ROBUSTNESS",
                        "Too few dataset-level LODO folds were estimable; no conserved-feature claim is available.",
                        "error", {"estimable_folds": estimable_folds, "required": request.lodo_min_estimable_folds},
                    ))
                    blocking_reasons = ["insufficient estimable dataset-level LODO folds"]
                else:
                    evidence_class = "exploratory_lodo_conserved"
            tested_gene_count = len(validated)
            significant_gene_count = int(validated.significant.sum())
            up_a = int(
                (
                    validated.significant
                    & validated.direction.eq("higher_in_group_a")
                ).sum()
            )
            up_b = int(
                (
                    validated.significant
                    & validated.direction.eq("higher_in_group_b")
                ).sum()
            )
            if terminal_status != "insufficient_robustness":
                terminal_status = "completed_with_warnings" if warnings else "completed"
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            terminal_status = "failed"
            detail = getattr(exc, "stderr", None) or str(exc)
            warnings = (
                *warnings,
                _warning(
                    "CAP_DESEQ_003_RUNTIME_FAILURE",
                    "The R/DESeq2 worker did not produce a valid complete result.",
                    "error",
                    {"detail": detail[-1000:]},
                ),
            )
            if result_path.exists():
                result_path.unlink()

    summary = {
        "capability_id": CAPABILITY_ID,
        "capability_version": NODE_VERSION,
        "schema_version": CAP_DESEQ_003_SCHEMA_VERSION,
        "terminal_status": terminal_status,
        "comparison_direction": f"{request.group_a} - {request.group_b}",
        "evidence_class": evidence_class,
        "positive_effect_interpretation": f"higher in {request.group_a}",
        "negative_effect_interpretation": f"higher in {request.group_b}",
        "input_gene_count": input_gene_count,
        "retained_gene_count": retained_gene_count,
        "tested_gene_count": tested_gene_count,
        "significant_gene_count": significant_gene_count,
        "upregulated_in_group_a_count": up_a,
        "upregulated_in_group_b_count": up_b,
        "group_replicate_counts": assessment.group_replicate_counts,
        "warnings": [item.model_dump(mode="json") for item in warnings],
        "blocking_reasons": blocking_reasons,
    }
    _atomic_json(summary, summary_path)

    artifacts = [
        _artifact("design_assessment", design_path, "QC", "application/json"),
        _artifact(
            "input_sample_metadata",
            metadata_path,
            "input-derived",
            "text/tab-separated-values",
        ),
        _artifact(
            "gene_filter_summary", filter_path, "QC", "application/json"
        ),
        _artifact("summary", summary_path, "descriptive", "application/json"),
    ]
    if terminal_status in {"completed", "completed_with_warnings"}:
        artifacts.insert(
            0,
            _artifact(
                "deseq2_results",
                result_path,
                "inferential",
                "text/tab-separated-values",
            ),
        )
        artifacts.append(
            _artifact(
                "runtime_metadata",
                runtime_path,
                "provenance",
                "text/tab-separated-values",
            )
        )
        artifacts.extend(lodo_artifacts)
        artifacts.append(
            _artifact(
                "input_counts", counts_path, "input-derived",
                "text/tab-separated-values",
            )
        )
    elif terminal_status == "insufficient_robustness":
        artifacts.extend(artifact for artifact in lodo_artifacts if artifact.logical_name != "conserved_features")
        artifacts.append(
            _artifact(
                "input_counts",
                counts_path,
                "input-derived",
                "text/tab-separated-values",
            )
        )

    structured_warnings = _structured(warnings)
    software = {
        **context.software_versions,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    if runtime_path.exists():
        runtime = pd.read_csv(runtime_path, sep="\t").set_index("key").value
        software.update(
            {
                "R": str(runtime.get("R", "unknown")),
                "DESeq2": str(runtime.get("DESeq2", "unknown")),
            }
        )
    provenance_parameters = {
        **parameters,
        **summary,
        "design_formula": assessment.design_formula,
        "design_rank": assessment.design_rank,
        "design_columns": assessment.design_columns,
        "group_reference_level": request.group_b,
        "coefficient_extracted": f"{request.group_a} - {request.group_b}",
        "gene_filter_policy": (
            f"count >= {request.gene_minimum_count} in at least "
            f"{request.gene_minimum_replicates} eligible replicates"
        ),
        "blocking_reasons": blocking_reasons,
        "input_artifact_paths": {
            "counts": str(request.count_matrix_path),
            "metadata": str(request.sample_metadata_path),
            "upstream_provenance": (
                str(request.upstream_provenance_path)
                if request.upstream_provenance_path
                else None
            ),
        },
    }
    provenance = build_provenance(
        capability_id=CAPABILITY_ID,
        node_version=NODE_VERSION,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        source_files=SOURCE_FILES,
        source_locations=SOURCE_LOCATIONS,
        input_dataset_signature=context.dataset_signature,
        parameters=provenance_parameters,
        model_formula=assessment.design_formula,
        reference_group=request.group_b,
        covariates=(
            (request.dataset_column,)
            if assessment.design_formula == "~ dataset + group"
            else ()
        ),
        unit_of_inference="independent biological pseudobulk replicate",
        random_seed=None,
        software_versions=software,
        output_paths=tuple(artifact.path for artifact in artifacts),
        warnings=structured_warnings,
    )
    write_provenance_atomic(provenance, provenance_path)
    artifacts.extend(
        [
            _artifact("provenance", provenance_path, "provenance", "application/json"),
            _artifact(
                "cache_manifest",
                manifest_path,
                "manifest",
                "application/json",
            ),
        ]
    )
    manifest_outputs = [
        artifact.path
        for artifact in artifacts
        if artifact.logical_name != "cache_manifest"
    ]
    manifest = build_cache_manifest(
        cache_key=cache_key,
        capability_id=CAPABILITY_ID,
        node_version=NODE_VERSION,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        input_signature=signatures["counts"],
        source_dataset_signature=context.dataset_signature,
        parameters=parameters,
        output_files=manifest_outputs,
        completion_status=(
            "complete"
            if terminal_status in {"completed", "completed_with_warnings"}
            else "failed"
        ),
        warnings=structured_warnings,
        software_versions=software,
    )
    write_manifest_atomic(manifest, manifest_path)
    return _output_from_summary(
        summary,
        assessment,
        artifacts,
        cache_key=cache_key,
        cache_hit=False,
        provenance_path=provenance_path,
        manifest_path=manifest_path,
    )
