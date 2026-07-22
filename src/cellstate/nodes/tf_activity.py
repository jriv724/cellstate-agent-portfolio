"""Deterministic signed ULM transcription-factor activity inference."""
from __future__ import annotations

from hashlib import sha256
import json
import os
import platform
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t as student_t

from ..cache import (build_cache_manifest, load_complete_manifest, make_cache_key,
                     write_manifest_atomic)
from ..context import AnalysisContext
from ..provenance import build_provenance, write_provenance_atomic
from ..schemas.common import (AnalysisProvenance, ArtifactCategory, CapabilityStatus,
                              StructuredWarning)
from ..schemas.tf_activity import TFActivityInput, TFActivityOutput

CAPABILITY_ID = "CAP-TF-002"
CAPABILITY_TITLE = "Signed TF Activity Inference"
IMPLEMENTATION_VERSION = "1.0.0"
NODE_VERSION = "1.0.0"
CACHE_SCHEMA_VERSION = 1
SOURCE_FILES = ("workflows/legacy/notebooks/20_SMM_regulatory_networks.ipynb",)
SOURCE_LOCATIONS = (
    "cells 18-28: decoupler import/resource inspection, DoRothEA A/B/C, complete score matrix, ULM tmin=5, long results, exploratory recurrence",
)
CORRECTION_FAMILY_DESCRIPTION = (
    "Benjamini-Hochberg across all estimable TF p-values independently within each resource/database x feature program"
)
MODEL_DEFINITION = (
    "y_i = beta_0 + beta_1*x_i + epsilon_i over every eligible program gene; "
    "x_i is the signed TF-target weight or zero for a non-target; activity_score "
    "is the slope t-statistic; df=n_eligible_genes-2; two-sided Student-t p-value"
)
INTERPRETATION_BOUNDARY = (
    "A positive or negative CAP-TF-002 activity score is a model-based inference "
    "from signed gene-level statistics and signed regulon weights. It is not direct "
    "evidence of TF protein activation, binding, phosphorylation, causality, or "
    "therapeutic tractability."
)

PROGRAM_METADATA = (
    "program_id", "feature_set_id", "feature_type", "cell_state", "condition_a",
    "condition_b", "contrast", "contrast_direction", "feature_direction",
    "signed_statistic_name", "statistic_orientation", "source_capability_id",
    "source_capability_version", "source_cache_key", "source_artifact_path",
    "source_artifact_hash", "upstream_analysis_method",
    "upstream_analysis_parameters", "upstream_provenance",
)
OPTIONAL_GENE_COLUMNS = (
    "source_run_id", "effect_size", "standard_error", "raw_p_value",
    "adjusted_p_value", "base_mean", "evidence_label", "annotation_json",
)
ARTIFACTS = (
    ("validated_signed_gene_statistics", "validated_signed_gene_statistics.tsv", ArtifactCategory.INPUT_DERIVED, "text/tab-separated-values"),
    ("excluded_or_unmatched_gene_statistics", "excluded_or_unmatched_gene_statistics.tsv", ArtifactCategory.QC, "text/tab-separated-values"),
    ("program_gene_universe_qc", "program_gene_universe_qc.tsv", ArtifactCategory.QC, "text/tab-separated-values"),
    ("signed_regulon_resource_qc", "signed_regulon_resource_qc.tsv", ArtifactCategory.QC, "text/tab-separated-values"),
    ("resource_program_coverage_qc", "resource_program_coverage_qc.tsv", ArtifactCategory.QC, "text/tab-separated-values"),
    ("normalized_signed_regulon_edges", "normalized_signed_regulon_edges.tsv", ArtifactCategory.INPUT_DERIVED, "text/tab-separated-values"),
    ("tf_activity_estimability", "tf_activity_estimability.tsv", ArtifactCategory.QC, "text/tab-separated-values"),
    ("tf_activity_by_resource", "tf_activity_by_resource.tsv", ArtifactCategory.INFERENTIAL, "text/tab-separated-values"),
    ("significant_tf_activity", "significant_tf_activity.tsv", ArtifactCategory.INFERENTIAL, "text/tab-separated-values"),
    ("tf_activity_consensus", "tf_activity_consensus.tsv", ArtifactCategory.INFERENTIAL, "text/tab-separated-values"),
    ("tf_activity_discordance", "tf_activity_discordance.tsv", ArtifactCategory.QC, "text/tab-separated-values"),
    ("heatmap_source_table", "heatmap_source_table.tsv", ArtifactCategory.DESCRIPTIVE, "text/tab-separated-values"),
    ("qc_summary", "qc_summary.json", ArtifactCategory.QC, "application/json"),
)


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(table: pd.DataFrame) -> str:
    text = table.to_csv(index=False, lineterminator="\n", na_rep="", float_format="%.17g")
    return sha256(text.encode("utf-8")).hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator, keep_default_na=False)


def _atomic_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        table.to_csv(temporary, sep="\t", index=False, lineterminator="\n",
                     float_format="%.17g")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(value: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def normalize_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _verify_source_artifacts(programs: pd.DataFrame) -> None:
    sources = programs[["source_artifact_path", "source_artifact_hash"]].drop_duplicates()
    for row in sources.itertuples(index=False):
        path = Path(row.source_artifact_path)
        if not path.is_file():
            raise ValueError(f"source artifact does not exist: {path}")
        expected = str(row.source_artifact_hash).strip().lower()
        if expected and file_hash(path) != expected:
            raise ValueError(f"source artifact hash mismatch: {path}")


def validate_signed_gene_statistics(
    request: TFActivityInput,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _read_table(request.signed_feature_program_path)
    required = {request.gene_column, request.signed_statistic_column, *PROGRAM_METADATA}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"signed feature program artifact missing required columns: {missing}")
    if raw.empty:
        raise ValueError("signed feature program artifact contains no rows")
    table = raw.copy()
    table["feature_id"] = table[request.gene_column].map(normalize_identifier)
    for column in PROGRAM_METADATA:
        table[column] = table[column].astype(str).str.strip()
    blank = [column for column in PROGRAM_METADATA if table[column].eq("").any()]
    if blank:
        raise ValueError(f"signed feature program metadata must be explicit and non-null: {blank}")
    if table.feature_id.eq("").any():
        raise ValueError("gene identifiers must be non-null")
    if set(table.feature_type.str.lower()) != {request.feature_type}:
        raise ValueError("unsupported or inconsistent feature type")
    if set(table.statistic_orientation) != {request.statistic_orientation}:
        raise ValueError("unsupported or contradictory statistic orientation")
    statistic = pd.to_numeric(table[request.signed_statistic_column], errors="coerce")
    if statistic.isna().any() or not np.isfinite(statistic.to_numpy(float)).all():
        raise ValueError("signed statistics must be numeric and finite")
    table["signed_statistic"] = statistic.astype(float)
    metadata_check = [column for column in PROGRAM_METADATA if column != "program_id"]
    contradictory_programs = []
    for program_id, group in table.groupby("program_id", sort=True):
        if any(group[column].nunique(dropna=False) != 1 for column in metadata_check):
            contradictory_programs.append(program_id)
    if contradictory_programs:
        raise ValueError(f"contradictory program metadata: {contradictory_programs}")
    _verify_source_artifacts(table)

    duplicate_groups = table.groupby(["program_id", "feature_id"], sort=True, dropna=False)
    contradictory_genes = []
    duplicate_indexes = []
    comparison_columns = [
        column for column in table.columns
        if column != request.gene_column
        and not (
            column == request.signed_statistic_column
            and request.signed_statistic_column != "signed_statistic"
        )
    ]
    for keys, group in duplicate_groups:
        if len(group) == 1:
            continue
        canonical = group[comparison_columns].astype(str)
        if any(canonical[column].nunique(dropna=False) != 1 for column in comparison_columns):
            contradictory_genes.append(keys)
        else:
            duplicate_indexes.extend(group.index[1:])
    if contradictory_genes:
        raise ValueError(f"contradictory duplicate program-gene rows: {contradictory_genes}")
    duplicates = table.loc[duplicate_indexes].copy()
    duplicates["exclusion_reason"] = "duplicate_program_gene"
    table = table.drop(index=duplicate_indexes).copy()
    counts = table.groupby("program_id").feature_id.nunique()
    if (counts < request.minimum_eligible_genes).any():
        raise ValueError("one or more programs have insufficient eligible genes")
    columns = ["program_id", "feature_set_id", "feature_id", "feature_type", "cell_state",
        "condition_a", "condition_b", "contrast", "contrast_direction",
        "feature_direction", "signed_statistic", "signed_statistic_name",
        "statistic_orientation", "source_capability_id", "source_capability_version",
        "source_cache_key", "source_artifact_path", "source_artifact_hash",
        "upstream_analysis_method", "upstream_analysis_parameters",
        "upstream_provenance"]
    columns += [column for column in OPTIONAL_GENE_COLUMNS if column in table.columns]
    table = table.loc[:, columns].sort_values(
        ["program_id", "feature_id"], kind="mergesort").reset_index(drop=True)
    if duplicates.empty:
        duplicates = pd.DataFrame(columns=[*columns, "exclusion_reason"])
    return table, duplicates.reset_index(drop=True)


def normalize_signed_resource(
    table: pd.DataFrame, database: str, request: TFActivityInput,
) -> tuple[pd.DataFrame, dict[str, object]]:
    aliases = {"source": "tf", "target": "target_gene", "gene": "target_gene"}
    table = table.rename(columns={
        old: new for old, new in aliases.items() if old in table and new not in table
    }).copy()
    missing = {"tf", "target_gene", "weight"} - set(table.columns)
    if missing:
        raise ValueError(f"malformed {database} signed resource; missing {sorted(missing)}")
    raw_n = len(table)
    table["tf"] = table.tf.map(normalize_identifier)
    table["target_gene"] = table.target_gene.map(normalize_identifier)
    if table.tf.eq("").any() or table.target_gene.eq("").any():
        raise ValueError(f"malformed {database} signed resource; blank TF or target")
    weights = pd.to_numeric(table.weight, errors="coerce")
    if weights.isna().any() or not np.isfinite(weights.to_numpy(float)).all():
        raise ValueError(f"{database} signed resource weights must be numeric and finite")
    if weights.eq(0).any():
        raise ValueError(f"{database} signed resource weights must be nonzero")
    table["weight"] = weights.astype(float)
    if "confidence" not in table:
        table["confidence"] = ""
    table["confidence"] = table.confidence.astype(str).str.upper().str.strip()
    if database == "DoRothEA":
        table = table.loc[
            table.confidence.isin(request.dorothea_confidence_levels)].copy()
    if "organism" in table:
        observed = set(table.organism.astype(str).str.lower().str.strip())
        if observed - {request.organism}:
            raise ValueError(f"{database} signed resource organism does not match request")
    table["database"] = database
    table["organism"] = request.organism
    table["resource_metadata"] = (
        table.resource_metadata.astype(str) if "resource_metadata" in table else ""
    )
    columns = ["database", "tf", "target_gene", "weight", "confidence",
               "organism", "resource_metadata"]
    post_confidence_n = len(table)
    table = table.loc[:, columns].drop_duplicates().sort_values(
        columns, kind="mergesort").reset_index(drop=True)
    exact_removed = post_confidence_n - len(table)
    conflicts = table.groupby(["tf", "target_gene"]).weight.nunique().gt(1)
    if conflicts.any():
        pairs = [f"{tf}->{target}" for tf, target in conflicts[conflicts].index]
        raise ValueError(
            f"conflicting signed TF-target weights in {database}: {pairs}")
    same_weight_duplicates = int(table.duplicated(["tf", "target_gene"]).sum())
    table = table.drop_duplicates(["tf", "target_gene"], keep="first").reset_index(drop=True)
    if table.empty:
        raise ValueError(f"{database} contains no usable signed regulatory edges")
    resource_hash = _content_hash(table)
    table["resource_hash"] = resource_hash
    qc = {
        "resource_name": database,
        "resource_provider": "caller_supplied_local",
        "resource_resolution_mode": "caller_supplied_local",
        "organism": request.organism,
        "confidence_levels": ";".join(request.dorothea_confidence_levels)
        if database == "DoRothEA" else "",
        "raw_row_count": raw_n,
        "post_confidence_row_count": post_confidence_n,
        "normalized_edge_count": len(table),
        "unique_tf_count": table.tf.nunique(),
        "unique_target_count": table.target_gene.nunique(),
        "exact_duplicate_rows_removed": exact_removed,
        "same_weight_duplicate_edges_collapsed": same_weight_duplicates,
        "conflicting_signed_edge_policy": "fail_validation",
        "resolved_resource_table_hash": resource_hash,
    }
    return table, qc


def _bh(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    n = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(n)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def ulm_equivalent(
    signed_statistics: np.ndarray, signed_weights: np.ndarray,
) -> dict[str, float]:
    """Deterministic scalar implementation of the documented decoupler-v2 ULM formula."""
    y = np.asarray(signed_statistics, dtype=float)
    x = np.asarray(signed_weights, dtype=float)
    if y.ndim != 1 or x.ndim != 1 or len(y) != len(x):
        raise ValueError("ULM inputs must be same-length one-dimensional arrays")
    n = len(y)
    df = n - 2
    if df < 1 or np.var(x, ddof=1) <= 0 or np.var(y, ddof=1) <= 0:
        raise ValueError("ULM is not estimable")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    covariance = np.dot(x_centered, y_centered) / (n - 1)
    x_sd = np.std(x, ddof=1)
    y_sd = np.std(y, ddof=1)
    correlation = float(covariance / (x_sd * y_sd))
    correlation = float(np.clip(correlation, -1.0, 1.0))
    epsilon = 2.2e-16
    statistic = correlation * np.sqrt(
        df / ((1.0 - correlation + epsilon) * (1.0 + correlation + epsilon)))
    coefficient = float(covariance / (x_sd ** 2))
    standard_error = float(abs(coefficient / statistic)) if statistic != 0 else float(
        y_sd / (x_sd * np.sqrt(df)))
    p_value = float(student_t.sf(abs(statistic), df) * 2)
    return {
        "activity_score": float(statistic),
        "fitted_coefficient": coefficient,
        "coefficient_standard_error": standard_error,
        "p_value": p_value,
        "degrees_of_freedom": int(df),
        "correlation": correlation,
    }


def calculate_tf_activity(
    programs: pd.DataFrame, resources: pd.DataFrame, resource_qc: pd.DataFrame,
    excluded: pd.DataFrame, request: TFActivityInput,
) -> tuple[dict[str, pd.DataFrame], tuple[StructuredWarning, ...], dict[str, object]]:
    all_targets = set(resources.target_gene)
    resource_qc = resource_qc.copy()
    unmatched = programs.loc[~programs.feature_id.isin(all_targets)].copy()
    if not unmatched.empty:
        unmatched["exclusion_reason"] = "gene_not_in_any_signed_regulon"
        excluded = (
            unmatched.reset_index(drop=True) if excluded.empty
            else pd.concat([excluded, unmatched], ignore_index=True, sort=False)
        )
    estimability_rows, activity_rows = [], []
    meta_columns = ["program_id", "feature_set_id", "cell_state", "condition_a",
        "condition_b", "contrast", "contrast_direction", "feature_direction",
        "statistic_orientation", "signed_statistic_name", "source_capability_id",
        "source_cache_key"]
    for program_id, program in programs.groupby("program_id", sort=True):
        program = program.sort_values("feature_id", kind="mergesort")
        genes = program.feature_id.tolist()
        gene_index = {gene: index for index, gene in enumerate(genes)}
        y = program.signed_statistic.to_numpy(float)
        meta = program.iloc[0]
        for database, resource in resources.groupby("database", sort=True):
            for tf, regulon in resource.groupby("tf", sort=True):
                weights_by_gene = dict(zip(regulon.target_gene, regulon.weight))
                overlap = sorted(set(genes) & set(weights_by_gene))
                x = np.zeros(len(genes), dtype=float)
                for gene in overlap:
                    x[gene_index[gene]] = weights_by_gene[gene]
                target_weights = np.array([weights_by_gene[gene] for gene in overlap])
                predictor_variance = float(np.var(x, ddof=1)) if len(x) > 1 else 0.0
                target_variance = (
                    float(np.var(target_weights, ddof=1))
                    if len(target_weights) > 1 else 0.0
                )
                reasons = []
                if len(overlap) < request.minimum_overlapping_targets:
                    reasons.append("fewer_than_minimum_overlapping_targets")
                if len(y) - 2 < 1:
                    reasons.append("insufficient_residual_degrees_of_freedom")
                if predictor_variance <= 0:
                    reasons.append("zero_variance_signed_weight_predictor")
                if np.var(y, ddof=1) <= 0:
                    reasons.append("zero_variance_signed_statistic")
                estimable = not reasons
                base = {column: meta[column] for column in meta_columns}
                base.update({
                    "database": database, "organism": request.organism, "tf": tf,
                    "number_of_eligible_genes": len(genes),
                    "total_regulon_target_count": regulon.target_gene.nunique(),
                    "number_of_overlapping_targets": len(overlap),
                    "number_of_unique_signed_target_weights": int(
                        len(np.unique(target_weights))) if len(target_weights) else 0,
                    "target_weight_variance": target_variance,
                    "model_predictor_weight_variance": predictor_variance,
                    "degrees_of_freedom": len(y) - 2,
                    "estimable": estimable,
                    "estimability_status": "estimable" if estimable else "not_estimable",
                    "exclusion_reason": ";".join(reasons),
                    "overlapping_target_genes": ";".join(overlap),
                    "resource_hash": regulon.resource_hash.iloc[0],
                })
                estimability_rows.append(base)
                if estimable:
                    result = ulm_equivalent(y, x)
                    activity_rows.append({
                        **base, **result,
                        "model_statistic_name": "slope_t_statistic",
                        "ulm_implementation_mode": request.ulm_implementation_mode,
                    })
    estimability = pd.DataFrame(estimability_rows).sort_values(
        ["database", "program_id", "tf"], kind="mergesort").reset_index(drop=True)
    activity = pd.DataFrame(activity_rows)
    if activity.empty:
        activity = pd.DataFrame(columns=[
            *estimability.columns, "activity_score", "fitted_coefficient",
            "coefficient_standard_error", "p_value", "correlation",
            "model_statistic_name", "ulm_implementation_mode",
            "adjusted_p_value", "significant", "activity_direction",
        ])
    if not activity.empty:
        activity["adjusted_p_value"] = np.nan
        for _, indexes in activity.groupby(
            ["database", "program_id"], sort=True).groups.items():
            activity.loc[indexes, "adjusted_p_value"] = _bh(
                activity.loc[indexes, "p_value"])
        activity["significant"] = activity.adjusted_p_value.le(request.fdr_cutoff)
        activity["activity_direction"] = "not_significant"
        activity.loc[
            activity.significant & activity.activity_score.gt(0),
            "activity_direction"] = "increased"
        activity.loc[
            activity.significant & activity.activity_score.lt(0),
            "activity_direction"] = "decreased"
        activity.loc[
            activity.significant & activity.activity_score.eq(0),
            "activity_direction"] = "unresolved"
        activity = activity.sort_values(
            ["database", "program_id", "adjusted_p_value", "tf"],
            kind="mergesort").reset_index(drop=True)
    significant = (
        activity.loc[activity.significant].copy() if not activity.empty
        else activity.copy()
    )

    consensus_rows, discordance_rows = [], []
    context_columns = ["program_id", "feature_set_id", "cell_state", "condition_a",
        "condition_b", "contrast", "contrast_direction", "feature_direction",
        "statistic_orientation", "signed_statistic_name", "tf",
        "source_capability_id", "source_cache_key"]
    for keys, est_group in estimability.groupby(
        ["program_id", "feature_set_id", "cell_state", "condition_a", "condition_b",
         "contrast", "contrast_direction", "feature_direction",
         "statistic_orientation", "signed_statistic_name", "tf",
         "source_capability_id", "source_cache_key"], sort=True):
        row = dict(zip(context_columns, keys))
        act_group = activity.loc[
            (activity.program_id == row["program_id"]) & (activity.tf == row["tf"])
        ] if not activity.empty else activity
        sig = act_group.loc[act_group.significant] if not act_group.empty else act_group
        directions = sorted(sig.activity_direction.unique()) if not sig.empty else []
        directional_consensus = (
            len(sig) >= request.minimum_consensus_resources
            and len(directions) == 1 and directions[0] in {"increased", "decreased"}
        )
        discordant_direction = len(set(directions) & {"increased", "decreased"}) > 1
        concordant_support = (
            not sig.empty and len(directions) == 1
            and directions[0] in {"increased", "decreased"}
        )
        supporting_scores = sig.activity_score.to_numpy(float)
        reasons = []
        nonestimable_count = int((~est_group.estimable).sum())
        if nonestimable_count and est_group.estimable.any():
            reasons.append("resource_estimability_difference")
        if discordant_direction:
            reasons.append("significant_opposite_directions")
        if len(sig) == 1:
            reasons.append("single_resource_significant_support")
        overlaps = est_group.number_of_overlapping_targets.to_numpy(float)
        positive = overlaps[overlaps > 0]
        if len(positive) > 1 and positive.max() / positive.min() >= 2:
            reasons.append("major_regulon_coverage_difference")
        if est_group.overlapping_target_genes.nunique() > 1:
            reasons.append("inconsistent_target_overlap")
        reasons = sorted(set(reasons))
        row.update({
            "number_of_estimable_resources": int(est_group.estimable.sum()),
            "estimable_resources": ";".join(sorted(
                est_group.loc[est_group.estimable, "database"].unique())),
            "number_of_significant_supporting_resources": len(sig),
            "supporting_resources": ";".join(sorted(sig.database.unique()))
            if not sig.empty else "",
            "resource_activity_scores": json.dumps({
                item.database: float(item.activity_score)
                for item in act_group.sort_values("database").itertuples()
            }, sort_keys=True, separators=(",", ":")),
            "resource_directions": json.dumps({
                item.database: item.activity_direction
                for item in act_group.sort_values("database").itertuples()
            }, sort_keys=True, separators=(",", ":")),
            "minimum_adjusted_p_value": float(act_group.adjusted_p_value.min())
            if not act_group.empty else np.nan,
            "median_consensus_activity_score": float(np.median(supporting_scores))
            if concordant_support else np.nan,
            "mean_consensus_activity_score": float(np.mean(supporting_scores))
            if concordant_support else np.nan,
            "minimum_absolute_supporting_score": float(
                np.min(np.abs(supporting_scores))) if concordant_support else np.nan,
            "strongest_absolute_activity_score": float(
                np.max(np.abs(supporting_scores))) if len(supporting_scores) else np.nan,
            "consensus_direction": directions[0] if directional_consensus else "unresolved",
            "directional_consensus_status": directional_consensus,
            "consensus_threshold_used": request.minimum_consensus_resources,
            "directional_discordance_status": discordant_direction,
            "any_resource_discordance_status": bool(reasons),
            "resource_discordance_reasons": ";".join(reasons),
        })
        consensus_rows.append(row)
        if reasons:
            discordance_rows.append({
                **{column: row[column] for column in context_columns},
                "directional_discordance_status": discordant_direction,
                "any_resource_discordance_status": True,
                "resource_discordance_reasons": ";".join(reasons),
                "resource_estimability": json.dumps({
                    item.database: bool(item.estimable)
                    for item in est_group.sort_values("database").itertuples()
                }, sort_keys=True, separators=(",", ":")),
                "resource_overlap_counts": json.dumps({
                    item.database: int(item.number_of_overlapping_targets)
                    for item in est_group.sort_values("database").itertuples()
                }, sort_keys=True, separators=(",", ":")),
                "resource_activity_scores": row["resource_activity_scores"],
                "resource_directions": row["resource_directions"],
            })
    consensus = pd.DataFrame(consensus_rows)
    if not consensus.empty:
        consensus = consensus.sort_values(
            ["program_id", "directional_consensus_status",
             "minimum_adjusted_p_value", "tf"],
            ascending=[True, False, True, True], kind="mergesort").reset_index(drop=True)
    discordance_columns = [*context_columns, "directional_discordance_status",
        "any_resource_discordance_status", "resource_discordance_reasons",
        "resource_estimability", "resource_overlap_counts",
        "resource_activity_scores", "resource_directions"]
    discordance = pd.DataFrame(discordance_rows, columns=discordance_columns)
    if not discordance.empty:
        discordance = discordance.sort_values(
            ["program_id", "tf"], kind="mergesort").reset_index(drop=True)
    heatmap = consensus.loc[consensus.directional_consensus_status, [
        "program_id", "feature_set_id", "cell_state", "contrast",
        "contrast_direction", "statistic_orientation", "tf",
        "consensus_direction", "minimum_adjusted_p_value",
        "median_consensus_activity_score", "mean_consensus_activity_score",
        "minimum_absolute_supporting_score", "strongest_absolute_activity_score",
        "number_of_significant_supporting_resources",
        "resource_activity_scores", "directional_consensus_status",
    ]].copy() if not consensus.empty else pd.DataFrame(columns=[
        "program_id", "feature_set_id", "cell_state", "contrast",
        "contrast_direction", "statistic_orientation", "tf",
        "consensus_direction", "minimum_adjusted_p_value",
        "median_consensus_activity_score", "mean_consensus_activity_score",
        "minimum_absolute_supporting_score", "strongest_absolute_activity_score",
        "number_of_significant_supporting_resources",
        "resource_activity_scores", "directional_consensus_status"])
    if not heatmap.empty:
        heatmap["display_activity_score"] = heatmap[
            "median_consensus_activity_score"]
        heatmap = heatmap.sort_values(
            ["program_id", "tf"], kind="mergesort").reset_index(drop=True)
    else:
        heatmap["display_activity_score"] = pd.Series(dtype=float)

    warnings = []
    duplicate_count = int(
        excluded.exclusion_reason.eq("duplicate_program_gene").sum())
    unmatched_count = int(
        excluded.exclusion_reason.eq("gene_not_in_any_signed_regulon").sum())
    if duplicate_count:
        warnings.append(StructuredWarning(
            "TF_ACTIVITY_EXACT_DUPLICATES_REMOVED",
            "Exact duplicate program-gene rows were removed deterministically",
            context={"count": duplicate_count}))
    if unmatched_count:
        warnings.append(StructuredWarning(
            "TF_ACTIVITY_GENES_NOT_IN_REGULONS",
            "Eligible genes absent from all signed regulons were retained in the model universe",
            context={"count": unmatched_count}))
    if (~estimability.estimable).any():
        warnings.append(StructuredWarning(
            "TF_ACTIVITY_NON_ESTIMABLE_MODELS",
            "Some resource-program-TF models were non-estimable",
            context={"count": int((~estimability.estimable).sum())}))
    empty_resource_programs = [
        {"database": database, "program_id": program_id}
        for (database, program_id), group in estimability.groupby(
            ["database", "program_id"], sort=True)
        if not group.estimable.any()
    ]
    if empty_resource_programs:
        warnings.append(StructuredWarning(
            "TF_ACTIVITY_NO_ESTIMABLE_TFS_FOR_RESOURCE_PROGRAM",
            "No TF was estimable for one or more resource-program combinations",
            context={"resource_programs": empty_resource_programs}))
    if activity.empty:
        warnings.append(StructuredWarning(
            "TF_ACTIVITY_NO_ESTIMABLE_TFS", "No TF activity models were estimable"))
    elif significant.empty:
        warnings.append(StructuredWarning(
            "TF_ACTIVITY_NO_SIGNIFICANT_ACTIVITY",
            "No estimable TF activity passed the configured FDR threshold"))
    if consensus.empty or not consensus.directional_consensus_status.any():
        warnings.append(StructuredWarning(
            "TF_ACTIVITY_NO_DIRECTIONAL_CONSENSUS",
            "No TF met the cross-resource directional consensus definition"))
    if not discordance.empty:
        warnings.append(StructuredWarning(
            "TF_ACTIVITY_RESOURCE_DISCORDANCE",
            "Resource discordance was preserved as QC evidence",
            context={"count": len(discordance)}))
        single_support = discordance.resource_discordance_reasons.str.contains(
            "single_resource_significant_support", regex=False).sum()
        if single_support:
            warnings.append(StructuredWarning(
                "TF_ACTIVITY_SINGLE_RESOURCE_SUPPORT",
                "Significant activity supported by only one resource was retained without consensus",
                context={"count": int(single_support)}))
    if heatmap.empty:
        warnings.append(StructuredWarning(
            "TF_ACTIVITY_EMPTY_HEATMAP_SOURCE",
            "No directional-consensus TF activity was available for the canonical heatmap"))

    program_qc = []
    for program_id, group in programs.groupby("program_id", sort=True):
        program_qc.append({
            "program_id": program_id,
            "feature_set_id": group.feature_set_id.iloc[0],
            "cell_state": group.cell_state.iloc[0],
            "contrast": group.contrast.iloc[0],
            "statistic_orientation": group.statistic_orientation.iloc[0],
            "signed_statistic_name": group.signed_statistic_name.iloc[0],
            "eligible_gene_count": group.feature_id.nunique(),
            "genes_in_any_signed_regulon": group.feature_id.isin(all_targets).sum(),
            "genes_not_in_any_signed_regulon": (~group.feature_id.isin(all_targets)).sum(),
            "complete_gene_universe_retained": True,
        })
    coverage_rows = []
    for program_id, program in programs.groupby("program_id", sort=True):
        program_genes = set(program.feature_id)
        for database, resource in resources.groupby("database", sort=True):
            model_rows = estimability.loc[
                (estimability.program_id == program_id)
                & (estimability.database == database)]
            overlap_count = len(program_genes & set(resource.target_gene))
            coverage_rows.append({
                "database": database,
                "program_id": program_id,
                "eligible_gene_count": len(program_genes),
                "resource_target_overlap": overlap_count,
                "resource_target_coverage_fraction": overlap_count / len(program_genes),
                "number_of_estimable_tfs": int(model_rows.estimable.sum()),
                "number_of_nonestimable_tfs": int((~model_rows.estimable).sum()),
            })
    resource_program_coverage = pd.DataFrame(coverage_rows).sort_values(
        ["database", "program_id"], kind="mergesort").reset_index(drop=True)
    low_coverage_pairs = resource_program_coverage.loc[
        resource_program_coverage.resource_target_overlap.lt(
            request.minimum_overlapping_targets), ["database", "program_id"]]
    if not low_coverage_pairs.empty:
        warnings.append(StructuredWarning(
            "TF_ACTIVITY_RESOURCE_LOW_TARGET_COVERAGE",
            "One or more signed resource-program pairs have low target coverage",
            context={
                "resource_program_pairs": low_coverage_pairs.to_dict(orient="records"),
                "minimum_overlapping_targets": request.minimum_overlapping_targets,
            }))
    summary = {
        "capability_id": CAPABILITY_ID,
        "implementation_version": IMPLEMENTATION_VERSION,
        "node_version": NODE_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "program_count": programs.program_id.nunique(),
        "eligible_gene_counts": {
            str(key): int(value) for key, value in
            programs.groupby("program_id").feature_id.nunique().items()},
        "duplicate_program_gene_count": duplicate_count,
        "gene_not_in_any_signed_regulon_count": unmatched_count,
        "estimable_model_count": int(estimability.estimable.sum()),
        "nonestimable_model_count": int((~estimability.estimable).sum()),
        "significant_activity_count": len(significant),
        "significant_regulator_count": (
            int(significant["tf"].nunique()) if not significant.empty else 0
        ),
        "directional_consensus_count": int(
            consensus.directional_consensus_status.sum()
        ) if not consensus.empty else 0,
        "directional_consensus_regulators": (
            sorted(
                consensus.loc[
                    consensus["directional_consensus_status"], "tf"
                ].astype(str).unique().tolist()
            )
            if not consensus.empty
            else []
        ),
        "discordance_count": len(discordance),
        "ulm_implementation_mode": request.ulm_implementation_mode,
        "model_definition": MODEL_DEFINITION,
        "correction_family_definition": CORRECTION_FAMILY_DESCRIPTION,
        "consensus_definition": (
            f"at least {request.minimum_consensus_resources} significant resources "
            "with one shared increased or decreased direction"),
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
        "thresholds": request.parameters(),
        "warnings": [warning.to_dict() for warning in warnings],
    }
    tables = {
        "validated_signed_gene_statistics": programs,
        "excluded_or_unmatched_gene_statistics": excluded,
        "program_gene_universe_qc": pd.DataFrame(program_qc),
        "signed_regulon_resource_qc": resource_qc,
        "resource_program_coverage_qc": resource_program_coverage,
        "normalized_signed_regulon_edges": resources,
        "tf_activity_estimability": estimability,
        "tf_activity_by_resource": activity,
        "significant_tf_activity": significant,
        "tf_activity_consensus": consensus,
        "tf_activity_discordance": discordance,
        "heatmap_source_table": heatmap,
    }
    return tables, tuple(warnings), summary


def _compact_program_metadata(programs: pd.DataFrame) -> list[dict[str, object]]:
    records = []
    columns = [column for column in PROGRAM_METADATA if column != "program_id"]
    for program_id, group in programs.groupby("program_id", sort=True):
        record = {
            "program_id": program_id,
            "feature_count": int(group.feature_id.nunique()),
        }
        record.update({column: group.iloc[0][column] for column in columns})
        records.append(record)
    return records


def _load_provenance(path: Path) -> AnalysisProvenance:
    raw = json.loads(path.read_text())
    warnings = tuple(StructuredWarning(
        item["code"], item["message"], item.get("severity", "warning"),
        item.get("context", {})) for item in raw["warnings"])
    return AnalysisProvenance(
        raw["capability_id"], raw["node_version"], raw["cache_schema_version"],
        tuple(raw["source_files"]), tuple(raw["source_locations"]),
        raw["input_dataset_signature"], raw["parameters"], raw["model_formula"],
        raw["reference_group"], tuple(raw["covariates"]), raw["unit_of_inference"],
        raw["random_seed"], raw["software_versions"], tuple(raw["output_paths"]),
        warnings, raw["execution_timestamp_utc"])


def run_tf_activity(
    request: TFActivityInput, context: AnalysisContext,
) -> TFActivityOutput:
    paths = (request.signed_feature_program_path, request.dorothea_path,
             request.collectri_path)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("one or more CAP-TF-002 input artifacts are missing")
    programs, excluded = validate_signed_gene_statistics(request)
    resource_tables, resource_qcs = [], []
    for database, path in (
        ("DoRothEA", request.dorothea_path),
        ("CollecTRI", request.collectri_path),
    ):
        normalized, qc = normalize_signed_resource(_read_table(path), database, request)
        resource_tables.append(normalized)
        resource_qcs.append(qc)
    resources = pd.concat(resource_tables, ignore_index=True).sort_values(
        ["database", "tf", "target_gene"], kind="mergesort").reset_index(drop=True)
    signatures = {str(path): file_hash(path) for path in paths}
    params = {
        **request.parameters(),
        "implementation_version": IMPLEMENTATION_VERSION,
        "input_artifact_hashes": signatures,
        "normalized_signed_gene_statistics_hash": _content_hash(programs),
        "normalized_signed_resource_hashes": {
            item["resource_name"]: item["resolved_resource_table_hash"]
            for item in resource_qcs},
        "compact_program_metadata": _compact_program_metadata(programs),
        "model_definition": MODEL_DEFINITION,
    }
    key = make_cache_key(
        capability_id=CAPABILITY_ID, node_version=NODE_VERSION,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        dataset_signature=context.dataset_signature, parameters=params)
    manifest_path = context.cache_root / CAPABILITY_ID.lower() / key / "manifest.json"
    cached = load_complete_manifest(manifest_path, key)
    if cached:
        named = {Path(path).name: str(path) for path in cached.output_files}
        provenance = _load_provenance(Path(named["provenance.json"]))
        artifacts = tuple(
            (logical, named[filename], category.value, media)
            for logical, filename, category, media in ARTIFACTS)
        status = (
            CapabilityStatus.COMPLETED_WITH_WARNINGS
            if provenance.warnings else CapabilityStatus.COMPLETED)
        return TFActivityOutput(
            CAPABILITY_ID, IMPLEMENTATION_VERSION, NODE_VERSION, status, key,
            True, artifacts, named["provenance.json"], str(manifest_path),
            provenance.warnings, provenance)
    tables, warnings, summary = calculate_tf_activity(
        programs, resources, pd.DataFrame(resource_qcs), excluded, request)
    output_dir = context.capability_output_dir(CAPABILITY_ID) / key
    artifact_paths = []
    for logical, filename, category, media in ARTIFACTS:
        path = output_dir / filename
        if logical == "qc_summary":
            _atomic_json(summary, path)
        else:
            _atomic_table(tables[logical], path)
        artifact_paths.append((logical, str(path), category.value, media))
    software = {
        **context.software_versions,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    provenance_parameters = {
        **params, **summary,
        "input_artifact_paths": [str(path) for path in paths],
        "resource_hashes": params["normalized_signed_resource_hashes"],
        "resource_provider": "caller_supplied_local",
        "resource_resolution_mode": "caller_supplied_local",
        "decoupler_version": None,
        "estimability_rules": [
            "overlapping targets >= configured minimum",
            "complete-program signed-weight predictor variance > 0",
            "signed statistic variance > 0",
            "n_eligible_genes - 2 >= 1",
        ],
        "correction_family_definition": CORRECTION_FAMILY_DESCRIPTION,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    provenance_path = output_dir / "provenance.json"
    provenance = build_provenance(
        capability_id=CAPABILITY_ID, node_version=NODE_VERSION,
        cache_schema_version=CACHE_SCHEMA_VERSION, source_files=SOURCE_FILES,
        source_locations=SOURCE_LOCATIONS,
        input_dataset_signature=context.dataset_signature,
        parameters=provenance_parameters, model_formula=MODEL_DEFINITION,
        reference_group=None, covariates=(),
        unit_of_inference="resource_x_feature_program_x_tf", random_seed=None,
        software_versions=software,
        output_paths=tuple(path for _, path, _, _ in artifact_paths),
        warnings=warnings)
    write_provenance_atomic(provenance, provenance_path)
    outputs = tuple(path for _, path, _, _ in artifact_paths) + (str(provenance_path),)
    manifest = build_cache_manifest(
        cache_key=key, capability_id=CAPABILITY_ID, node_version=NODE_VERSION,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        input_signature=signatures[str(request.signed_feature_program_path)],
        source_dataset_signature=context.dataset_signature, parameters=params,
        output_files=outputs, completion_status="complete", warnings=warnings,
        software_versions=software)
    write_manifest_atomic(manifest, manifest_path)
    return TFActivityOutput(
        CAPABILITY_ID, IMPLEMENTATION_VERSION, NODE_VERSION,
        CapabilityStatus.COMPLETED_WITH_WARNINGS if warnings else CapabilityStatus.COMPLETED,
        key, False, tuple(artifact_paths), str(provenance_path),
        str(manifest_path), warnings, provenance)
