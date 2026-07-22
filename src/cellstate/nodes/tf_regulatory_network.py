"""Deterministic consensus TF-target enrichment for explicit feature programs."""
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
from scipy.stats import fisher_exact

from ..cache import (build_cache_manifest, load_complete_manifest, make_cache_key,
                     write_manifest_atomic)
from ..context import AnalysisContext
from ..provenance import build_provenance, write_provenance_atomic
from ..schemas.common import (AnalysisProvenance, ArtifactCategory, CapabilityStatus,
                              StructuredWarning)
from ..schemas.tf_regulatory_network import (TFRegulatoryNetworkInput,
                                             TFRegulatoryNetworkOutput)

CAPABILITY_ID = "CAP-TF-001"
CAPABILITY_TITLE = "Consensus TF Regulatory Network"
IMPLEMENTATION_VERSION = "1.0.0"
NODE_VERSION = "1.0.0"
CACHE_SCHEMA_VERSION = 1
SOURCE_FILES = ("workflows/legacy/notebooks/20_SMM_regulatory_networks.ipynb",)
SOURCE_LOCATIONS = (
    "cells 29-46: Fisher TF-target enrichment, BH correction, consensus, convergence, heatmap source, and neutral network export",
)
CORRECTION_FAMILY_DESCRIPTION = (
    "Benjamini-Hochberg across all tested TFs independently within each resource/database x feature program"
)

PROGRAM_METADATA = (
    "program_id", "feature_set_id", "feature_type", "cell_state", "condition_a",
    "condition_b", "contrast", "contrast_direction", "feature_direction",
    "source_capability_id", "source_capability_version", "source_cache_key",
    "source_artifact_path", "source_artifact_hash", "feature_selection_method",
    "feature_selection_thresholds", "feature_selection_provenance",
)
OPTIONAL_FEATURE_COLUMNS = (
    "source_run_id", "effect_size", "adjusted_p_value", "raw_p_value",
    "ranking_score", "evidence_label", "annotation_json",
)
ARTIFACTS = (
    ("validated_input_feature_programs", "validated_input_feature_programs.tsv", ArtifactCategory.INPUT_DERIVED, "text/tab-separated-values"),
    ("excluded_or_unmatched_features", "excluded_or_unmatched_features.tsv", ArtifactCategory.QC, "text/tab-separated-values"),
    ("background_universe_qc", "background_universe_qc.tsv", ArtifactCategory.QC, "text/tab-separated-values"),
    ("regulon_resource_qc", "regulon_resource_qc.tsv", ArtifactCategory.QC, "text/tab-separated-values"),
    ("normalized_regulon_edges", "normalized_regulon_edges.tsv", ArtifactCategory.INPUT_DERIVED, "text/tab-separated-values"),
    ("tf_enrichment_by_database", "tf_enrichment_by_database.tsv", ArtifactCategory.INFERENTIAL, "text/tab-separated-values"),
    ("significant_tf_hits", "significant_tf_hits.tsv", ArtifactCategory.INFERENTIAL, "text/tab-separated-values"),
    ("consensus_tf_hits", "consensus_tf_hits.tsv", ArtifactCategory.INFERENTIAL, "text/tab-separated-values"),
    ("tf_regulatory_convergence", "tf_regulatory_convergence.tsv", ArtifactCategory.INFERENTIAL, "text/tab-separated-values"),
    ("tf_target_program_network_nodes", "tf_target_program_network_nodes.tsv", ArtifactCategory.INFERENTIAL, "text/tab-separated-values"),
    ("tf_target_program_network_edges", "tf_target_program_network_edges.tsv", ArtifactCategory.INFERENTIAL, "text/tab-separated-values"),
    ("cytoscape_nodes", "cytoscape_nodes.tsv", ArtifactCategory.OPTIONAL, "text/tab-separated-values"),
    ("cytoscape_edges", "cytoscape_edges.tsv", ArtifactCategory.OPTIONAL, "text/tab-separated-values"),
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
    text = table.to_csv(index=False, lineterminator="\n", na_rep="")
    return sha256(text.encode("utf-8")).hexdigest()


def _atomic_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        table.to_csv(temporary, sep="\t", index=False, lineterminator="\n")
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


def normalize_feature(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator, keep_default_na=False)


def _verify_source_artifacts(programs: pd.DataFrame) -> None:
    for row in programs[["source_artifact_path", "source_artifact_hash"]].drop_duplicates().itertuples(index=False):
        path = Path(row.source_artifact_path)
        if not path.is_file():
            raise ValueError(f"source artifact does not exist: {path}")
        observed = file_hash(path)
        if str(row.source_artifact_hash).strip() and observed != str(row.source_artifact_hash).strip().lower():
            raise ValueError(f"source artifact hash mismatch: {path}")


def validate_feature_programs(request: TFRegulatoryNetworkInput) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = _read_table(request.feature_program_path)
    required = {request.feature_column, *PROGRAM_METADATA}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"feature program artifact missing required columns: {missing}")
    if raw.empty:
        raise ValueError("feature program artifact contains no feature rows")
    programs = raw.copy()
    programs["feature_id"] = programs[request.feature_column].map(normalize_feature)
    for column in PROGRAM_METADATA:
        programs[column] = programs[column].astype(str).str.strip()
    blank_required = [column for column in PROGRAM_METADATA if (programs[column] == "").any()]
    if blank_required:
        raise ValueError(f"feature program metadata must be explicit and non-null: {blank_required}")
    if (programs["feature_id"] == "").any():
        raise ValueError("feature identifiers must be non-null")
    if set(programs.feature_type.str.lower()) != {request.feature_type}:
        raise ValueError("unsupported or inconsistent feature type")
    if programs.program_id.nunique() != len(programs.groupby("program_id", sort=False)):
        raise ValueError("invalid program identifiers")
    metadata_check = [x for x in PROGRAM_METADATA if x != "program_id"]
    contradictory = []
    for program_id, sub in programs.groupby("program_id", sort=True):
        if any(sub[column].nunique(dropna=False) != 1 for column in metadata_check):
            contradictory.append(program_id)
    if contradictory:
        raise ValueError(f"contradictory program metadata: {contradictory}")
    _verify_source_artifacts(programs)
    duplicate_mask = programs.duplicated(["program_id", "feature_id"], keep="first")
    duplicates = programs.loc[duplicate_mask].copy()
    if not duplicates.empty:
        duplicates["exclusion_reason"] = "duplicate_program_feature"
    programs = programs.loc[~duplicate_mask].copy()
    background_raw = _read_table(request.background_universe_path)
    if request.background_feature_column not in background_raw:
        raise ValueError("background universe missing feature column")
    background_raw["feature_id"] = background_raw[request.background_feature_column].map(normalize_feature)
    if (background_raw.feature_id == "").any():
        raise ValueError("background universe contains invalid features")
    background = pd.DataFrame({"feature_id": sorted(background_raw.feature_id.unique())})
    if background.empty:
        raise ValueError("background universe is empty")
    in_background = programs.feature_id.isin(set(background.feature_id))
    excluded_background = programs.loc[~in_background].copy()
    if not excluded_background.empty:
        excluded_background["exclusion_reason"] = "query_feature_not_in_background"
    excluded = pd.concat([duplicates, excluded_background], ignore_index=True, sort=False)
    if "exclusion_reason" not in excluded:
        excluded["exclusion_reason"] = pd.Series(dtype="object")
    programs = programs.loc[in_background].copy()
    counts = programs.groupby("program_id").feature_id.nunique()
    original_ids = set(raw.program_id.astype(str).str.strip())
    missing_programs = original_ids - set(counts.index)
    if missing_programs or (counts < request.minimum_query_features).any():
        raise ValueError("one or more feature programs have insufficient query features after background intersection")
    columns = ["program_id", "feature_set_id", "feature_id", *[x for x in PROGRAM_METADATA if x not in {"program_id", "feature_set_id"}]]
    columns += [x for x in OPTIONAL_FEATURE_COLUMNS if x in programs.columns]
    programs = programs.loc[:, columns].sort_values(["program_id", "feature_id"], kind="mergesort").reset_index(drop=True)
    return programs, excluded.reset_index(drop=True), background


def normalize_resource(table: pd.DataFrame, database: str, request: TFRegulatoryNetworkInput) -> tuple[pd.DataFrame, dict[str, object]]:
    aliases = {"source": "tf", "target": "target_gene", "gene": "target_gene"}
    table = table.rename(columns={key: value for key, value in aliases.items() if key in table and value not in table}).copy()
    missing = {"tf", "target_gene"} - set(table)
    if missing:
        raise ValueError(f"malformed {database} resource table; missing {sorted(missing)}")
    raw_n = len(table)
    table["tf"] = table.tf.map(normalize_feature)
    table["target_gene"] = table.target_gene.map(normalize_feature)
    if (table.tf == "").any() or (table.target_gene == "").any():
        raise ValueError(f"malformed {database} resource table; blank TF or target")
    if "weight" not in table:
        table["weight"] = np.nan
    if "confidence" not in table:
        table["confidence"] = ""
    table["confidence"] = table.confidence.astype(str).str.upper().str.strip()
    if database == "DoRothEA":
        table = table.loc[table.confidence.isin(request.dorothea_confidence_levels)].copy()
    if "organism" in table:
        observed = set(table.organism.astype(str).str.lower())
        if observed - {request.organism}:
            raise ValueError(f"{database} resource organism does not match request")
    table["database"] = database
    table["organism"] = request.organism
    table["resource_metadata"] = table["resource_metadata"].astype(str) if "resource_metadata" in table else ""
    columns = ["database", "tf", "target_gene", "weight", "confidence", "organism", "resource_metadata"]
    before = len(table)
    table = table.loc[:, columns].drop_duplicates().sort_values(columns, kind="mergesort", na_position="last").reset_index(drop=True)
    conflicting = int(table.groupby(["tf", "target_gene"]).size().gt(1).sum())
    resource_hash = _content_hash(table)
    table["resource_hash"] = resource_hash
    qc = {
        "resource_name": database,
        "resource_provider": "decoupler_or_local_injected_table",
        "resource_resolution_mode": "caller_supplied_local",
        "decoupler_version": None,
        "organism": request.organism,
        "confidence_levels": ";".join(request.dorothea_confidence_levels) if database == "DoRothEA" else "",
        "raw_row_count": raw_n, "post_confidence_row_count": before,
        "normalized_row_count": len(table), "unique_tf_count": table.tf.nunique(),
        "unique_target_count": table.target_gene.nunique(),
        "exact_duplicate_rows_removed": before - len(table),
        "conflicting_tf_target_entries_preserved": conflicting,
        "duplicate_edge_handling": "exact normalized rows removed; conflicting weights/directions preserved",
        "resolved_resource_table_hash": resource_hash,
        "resource_version_or_retrieval_metadata": "caller-supplied local table",
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


def calculate_tf_regulatory_network(
    programs: pd.DataFrame, background: pd.DataFrame, resources: pd.DataFrame,
    resource_qc: pd.DataFrame, excluded: pd.DataFrame,
    request: TFRegulatoryNetworkInput,
) -> tuple[dict[str, pd.DataFrame], tuple[StructuredWarning, ...], dict[str, object]]:
    background_set = set(background.feature_id)
    resources_bg = resources.loc[resources.target_gene.isin(background_set)].copy()
    resource_targets = set(resources.target_gene)
    resource_unmatched = programs.loc[~programs.feature_id.isin(resource_targets)].copy()
    if not resource_unmatched.empty:
        resource_unmatched["exclusion_reason"] = "query_feature_not_in_any_regulon"
        excluded = pd.concat([excluded, resource_unmatched], ignore_index=True, sort=False)
    qc = resource_qc.copy()
    overlap_counts = resources_bg.groupby("database").target_gene.nunique()
    qc["background_target_overlap"] = qc.resource_name.map(overlap_counts).fillna(0).astype(int)
    rows = []
    metadata_columns = ["program_id", "feature_set_id", "cell_state", "condition_a",
                        "condition_b", "contrast", "contrast_direction", "feature_direction",
                        "source_capability_id", "source_cache_key"]
    for program_id, group in programs.groupby("program_id", sort=True):
        query = set(group.feature_id)
        meta = group.iloc[0]
        for database, resource in resources_bg.groupby("database", sort=True):
            regulons = resource.groupby("tf", sort=True).target_gene.apply(lambda x: set(x))
            resource_hash = resource.resource_hash.iloc[0]
            for tf, targets in regulons.items():
                if len(targets) < request.minimum_regulon_target_count:
                    continue
                overlap = query & targets
                a = len(overlap)
                b = len(query - targets)
                c = len(targets - query)
                d = len(background_set - query - targets)
                odds, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
                row = {column: meta[column] for column in metadata_columns}
                row.update({
                    "database": database, "organism": request.organism, "tf": tf,
                    "overlap_n": a, "query_n": len(query), "tf_target_n": len(targets),
                    "background_n": len(background_set), "odds_ratio": odds,
                    "p_value": p_value, "passes_minimum_overlap": a >= request.minimum_query_target_overlap,
                    "overlap_genes": ";".join(sorted(overlap)), "resource_hash": resource_hash,
                })
                rows.append(row)
    enrichment = pd.DataFrame(rows)
    if not enrichment.empty:
        enrichment["adjusted_p_value"] = np.nan
        for _, index in enrichment.groupby(["database", "program_id"], sort=True).groups.items():
            enrichment.loc[index, "adjusted_p_value"] = _bh(enrichment.loc[index, "p_value"])
        enrichment["significant"] = (
            enrichment.adjusted_p_value.le(request.fdr_cutoff)
            & enrichment.passes_minimum_overlap
        )
        enrichment = enrichment.sort_values(
            ["database", "program_id", "adjusted_p_value", "tf"],
            kind="mergesort",
        ).reset_index(drop=True)
    significant = enrichment.loc[enrichment.significant].copy() if not enrichment.empty else enrichment.copy()
    consensus_rows = []
    if not significant.empty:
        group_columns = ["program_id", "feature_set_id", "cell_state", "condition_a",
                         "condition_b", "contrast", "contrast_direction",
                         "feature_direction", "tf", "source_capability_id", "source_cache_key"]
        for keys, group in significant.groupby(group_columns, sort=True):
            row = dict(zip(group_columns, keys))
            genes = sorted({x for text in group.overlap_genes for x in str(text).split(";") if x})
            details = {
                item.database: {"overlap_n": int(item.overlap_n), "overlap_genes": item.overlap_genes,
                                "adjusted_p_value": float(item.adjusted_p_value),
                                "odds_ratio": float(item.odds_ratio)}
                for item in group.sort_values("database").itertuples()
            }
            count = group.database.nunique()
            row.update({
                "number_of_supporting_databases": count,
                "supporting_databases": ";".join(sorted(group.database.unique())),
                "maximum_odds_ratio": float(group.odds_ratio.max()),
                "minimum_adjusted_p_value": float(group.adjusted_p_value.min()),
                "total_overlap_count": int(group.overlap_n.sum()),
                "union_overlap_genes": ";".join(genes),
                "per_resource_overlap_details": json.dumps(details, sort_keys=True, separators=(",", ":")),
                "consensus_status": count >= request.minimum_supporting_databases,
                "consensus_threshold_used": request.minimum_supporting_databases,
            })
            consensus_rows.append(row)
    consensus = pd.DataFrame(consensus_rows)
    if not consensus.empty:
        consensus = consensus.sort_values(["program_id", "consensus_status", "tf"],
                                          ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
    supported = consensus.loc[consensus.consensus_status].copy() if not consensus.empty else consensus.copy()
    convergence_columns = (
        "tf", "number_of_represented_programs", "represented_program_ids",
        "number_of_represented_cell_states", "represented_cell_states",
        "represented_contrasts", "mean_number_of_supporting_databases",
        "strongest_odds_ratio", "minimum_adjusted_p_value", "total_overlap",
        "union_target_genes", "convergence_score", "convergence_denominator",
        "convergence_interpretation",
    )
    convergence_rows = []
    program_count = programs.program_id.nunique()
    if program_count > 1 and not supported.empty:
        for tf, group in supported.groupby("tf", sort=True):
            represented = group.program_id.nunique()
            genes = sorted({x for text in group.union_overlap_genes for x in text.split(";") if x})
            convergence_rows.append({
                "tf": tf, "number_of_represented_programs": represented,
                "represented_program_ids": ";".join(sorted(group.program_id.unique())),
                "number_of_represented_cell_states": group.cell_state.nunique(),
                "represented_cell_states": ";".join(sorted(group.cell_state.unique())),
                "represented_contrasts": ";".join(sorted(group.contrast.unique())),
                "mean_number_of_supporting_databases": float(group.number_of_supporting_databases.mean()),
                "strongest_odds_ratio": float(group.maximum_odds_ratio.max()),
                "minimum_adjusted_p_value": float(group.minimum_adjusted_p_value.min()),
                "total_overlap": int(group.total_overlap_count.sum()),
                "union_target_genes": ";".join(genes),
                "convergence_score": represented / program_count,
                "convergence_denominator": program_count,
                "convergence_interpretation": "fraction of supplied feature programs with consensus TF support",
            })
    convergence = pd.DataFrame(convergence_rows, columns=convergence_columns)
    nodes, edges = _build_network(supported, programs, convergence)
    heatmap = supported[["program_id", "feature_set_id", "cell_state", "contrast", "feature_direction",
                         "tf", "number_of_supporting_databases", "minimum_adjusted_p_value",
                         "maximum_odds_ratio", "consensus_status"]].copy() if not supported.empty else pd.DataFrame(
        columns=["program_id", "feature_set_id", "cell_state", "contrast", "feature_direction",
                 "tf", "number_of_supporting_databases", "minimum_adjusted_p_value",
                 "maximum_odds_ratio", "consensus_status"])
    if not heatmap.empty:
        heatmap["display_score"] = -np.log10(heatmap.minimum_adjusted_p_value.clip(lower=1e-300)) * heatmap.number_of_supporting_databases
        heatmap = heatmap.sort_values(["program_id", "tf"], kind="mergesort").reset_index(drop=True)
    warnings = []
    background_excluded = excluded.loc[
        excluded.exclusion_reason.eq("query_feature_not_in_background")
    ] if not excluded.empty else excluded
    if not background_excluded.empty:
        warnings.append(StructuredWarning("TF_QUERY_FEATURES_EXCLUDED",
            "Some query features were absent from the explicit background universe",
            context={"count": len(background_excluded),
                     "features": sorted(background_excluded.feature_id.unique())}))
    if program_count == 1:
        warnings.append(StructuredWarning("TF_CONVERGENCE_NOT_APPLICABLE",
            "Cross-program convergence is not applicable to a single supplied program",
            context={"program_count": 1}))
    if significant.empty:
        warnings.append(StructuredWarning("TF_NO_SIGNIFICANT_HITS",
            "No tested TF passed the configured FDR and overlap thresholds"))
    elif supported.empty:
        warnings.append(StructuredWarning("TF_NO_CONSENSUS_HITS",
            "No TF met the configured cross-resource consensus threshold"))
    if nodes.empty:
        warnings.append(StructuredWarning("TF_EMPTY_NETWORK",
            "No consensus-supported TF-target-program network edges were available"))
    query_counts = programs.groupby("program_id").feature_id.nunique()
    summary = {
        "capability_id": CAPABILITY_ID, "implementation_version": IMPLEMENTATION_VERSION,
        "node_version": NODE_VERSION, "cache_schema_version": CACHE_SCHEMA_VERSION,
        "organism": request.organism, "program_count": program_count,
        "query_feature_counts": {str(k): int(v) for k, v in query_counts.items()},
        "background_size": len(background),
        "excluded_query_feature_count": len(background_excluded),
        "duplicate_program_feature_count": int(
            excluded.exclusion_reason.eq("duplicate_program_feature").sum()),
        "resource_unmatched_query_feature_count": int(
            excluded.exclusion_reason.eq("query_feature_not_in_any_regulon").sum()),
        "resource_target_count_in_background": int(resources_bg.target_gene.nunique()),
        "tested_tf_count": int(enrichment.tf.nunique()) if not enrichment.empty else 0,
        "tested_tf_rows": len(enrichment), "significant_tf_rows": len(significant),
        "consensus_tf_rows": len(supported), "network_node_count": len(nodes),
        "network_edge_count": len(edges), "thresholds": request.parameters(),
        "correction_family_definition": CORRECTION_FAMILY_DESCRIPTION,
        "warnings": [warning.to_dict() for warning in warnings],
    }
    tables = {
        "validated_input_feature_programs": programs,
        "excluded_or_unmatched_features": excluded,
        "background_universe_qc": pd.DataFrame([{
            "background_source": request.background_source,
            "background_artifact_path": str(request.background_universe_path),
            "background_artifact_hash": file_hash(request.background_universe_path),
            "total_background_size": len(_read_table(request.background_universe_path)),
            "background_size_after_normalization": len(background),
            "number_of_query_features_in_background": programs.feature_id.nunique(),
            "number_of_resource_targets_in_background": resources_bg.target_gene.nunique(),
            "number_of_excluded_query_features": len(background_excluded),
            "excluded_feature_identifiers": ";".join(
                sorted(background_excluded.feature_id.unique())
            ) if not background_excluded.empty else "",
        }]),
        "regulon_resource_qc": qc, "normalized_regulon_edges": resources,
        "tf_enrichment_by_database": enrichment, "significant_tf_hits": significant,
        "consensus_tf_hits": consensus, "tf_regulatory_convergence": convergence,
        "tf_target_program_network_nodes": nodes, "tf_target_program_network_edges": edges,
        "cytoscape_nodes": nodes, "cytoscape_edges": edges,
        "heatmap_source_table": heatmap,
    }
    return tables, tuple(warnings), summary


def _build_network(consensus: pd.DataFrame, programs: pd.DataFrame,
                   convergence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_columns = ["node_id", "node_type", "label", "tf", "target_gene", "program_id",
        "feature_set_id", "cell_state", "condition_a", "condition_b", "contrast",
        "direction", "source_capability_id", "source_cache_key", "source_artifact_hash",
        "number_of_supporting_databases", "supporting_databases",
        "minimum_adjusted_p_value", "maximum_odds_ratio", "strongest_odds_ratio",
        "convergence_score",
        "convergence_denominator", "number_of_represented_programs",
        "represented_program_ids", "number_of_represented_cell_states",
        "represented_cell_states", "program_prevalence", "metadata_json"]
    edge_columns = ["edge_id", "source", "target", "edge_type", "interaction", "program_id",
        "feature_set_id", "cell_state", "contrast", "direction",
        "number_of_supporting_databases", "supporting_databases",
        "minimum_adjusted_p_value", "maximum_odds_ratio", "overlap_evidence",
        "resource_support_details", "edge_weight", "source_capability_id", "source_cache_key"]
    if consensus.empty:
        return pd.DataFrame(columns=node_columns), pd.DataFrame(columns=edge_columns)
    program_meta = programs.drop_duplicates("program_id").set_index("program_id")
    node_map: dict[str, dict[str, object]] = {}
    edge_map: dict[str, dict[str, object]] = {}
    program_count = programs.program_id.nunique()
    for tf, group in consensus.groupby("tf", sort=True):
        represented_program_ids = sorted(group.program_id.unique())
        represented_cell_states = sorted(group.cell_state.unique())
        databases = sorted({
            database
            for value in group.supporting_databases
            for database in str(value).split(";")
            if database
        })
        tf_id = f"TF::{tf}"
        node_map[tf_id] = {column: "" for column in node_columns}
        node_map[tf_id].update({
            "node_id": tf_id, "node_type": "TF", "label": tf, "tf": tf,
            "number_of_supporting_databases": len(databases),
            "supporting_databases": ";".join(databases),
            "minimum_adjusted_p_value": float(group.minimum_adjusted_p_value.min()),
            "maximum_odds_ratio": float(group.maximum_odds_ratio.max()),
            "strongest_odds_ratio": float(group.maximum_odds_ratio.max()),
            "number_of_represented_programs": len(represented_program_ids),
            "represented_program_ids": ";".join(represented_program_ids),
            "number_of_represented_cell_states": len(represented_cell_states),
            "represented_cell_states": ";".join(represented_cell_states),
            "convergence_score": len(represented_program_ids) / program_count,
            "convergence_denominator": program_count,
            "program_prevalence": len(represented_program_ids) / program_count,
        })
    for row in consensus.itertuples(index=False):
        tf_id, program_id = f"TF::{row.tf}", f"PROGRAM::{row.program_id}"
        source_meta = program_meta.loc[row.program_id]
        node_map[program_id] = {column: "" for column in node_columns}
        node_map[program_id].update({"node_id": program_id, "node_type": "program",
            "label": f"{row.cell_state} | {row.contrast} | {row.feature_direction}",
            "program_id": row.program_id, "feature_set_id": row.feature_set_id,
            "cell_state": row.cell_state, "condition_a": row.condition_a,
            "condition_b": row.condition_b, "contrast": row.contrast,
            "direction": row.feature_direction, "source_capability_id": row.source_capability_id,
            "source_cache_key": row.source_cache_key,
            "source_artifact_hash": source_meta.source_artifact_hash,
            "metadata_json": json.dumps({"contrast_direction": row.contrast_direction}, sort_keys=True)})
        genes = [x for x in row.union_overlap_genes.split(";") if x]
        for gene in genes:
            gene_id = f"GENE::{gene}"
            node_map.setdefault(gene_id, {column: "" for column in node_columns})
            node_map[gene_id].update({"node_id": gene_id, "node_type": "target_gene",
                                     "label": gene, "target_gene": gene})
            tf_edge_id = f"EDGE::TF::{row.tf}::GENE::{gene}::PROGRAM::{row.program_id}"
            edge_map[tf_edge_id] = {column: "" for column in edge_columns}
            edge_map[tf_edge_id].update({"edge_id": tf_edge_id, "source": tf_id, "target": gene_id,
                "edge_type": "TF_to_target", "interaction": "regulates",
                "program_id": row.program_id, "feature_set_id": row.feature_set_id,
                "cell_state": row.cell_state, "contrast": row.contrast, "direction": row.feature_direction,
                "number_of_supporting_databases": row.number_of_supporting_databases,
                "supporting_databases": row.supporting_databases,
                "minimum_adjusted_p_value": row.minimum_adjusted_p_value,
                "maximum_odds_ratio": row.maximum_odds_ratio, "overlap_evidence": gene,
                "resource_support_details": row.per_resource_overlap_details,
                "edge_weight": row.number_of_supporting_databases,
                "source_capability_id": row.source_capability_id, "source_cache_key": row.source_cache_key})
            member_id = f"EDGE::GENE::{gene}::PROGRAM::{row.program_id}"
            if member_id not in edge_map:
                edge_map[member_id] = {column: "" for column in edge_columns}
                edge_map[member_id].update({"edge_id": member_id, "source": gene_id, "target": program_id,
                    "edge_type": "target_to_program", "interaction": "member_of_regulatory_program",
                    "program_id": row.program_id, "feature_set_id": row.feature_set_id,
                    "cell_state": row.cell_state, "contrast": row.contrast, "direction": row.feature_direction,
                    "overlap_evidence": gene, "edge_weight": 1,
                    "source_capability_id": row.source_capability_id, "source_cache_key": row.source_cache_key})
    nodes = pd.DataFrame(node_map.values(), columns=node_columns).sort_values("node_id", kind="mergesort").reset_index(drop=True)
    edges = pd.DataFrame(edge_map.values(), columns=edge_columns).sort_values("edge_id", kind="mergesort").reset_index(drop=True)
    return nodes, edges


def _compact_feature_program_metadata(programs: pd.DataFrame) -> list[dict[str, object]]:
    metadata_columns = [column for column in PROGRAM_METADATA if column != "program_id"]
    records = []
    for program_id, group in programs.groupby("program_id", sort=True):
        meta = group.iloc[0]
        record = {"program_id": program_id, "feature_count": int(group.feature_id.nunique())}
        record.update({column: meta[column] for column in metadata_columns})
        records.append(record)
    return records


def _load_provenance(path: Path) -> AnalysisProvenance:
    raw = json.loads(path.read_text())
    warnings = tuple(StructuredWarning(x["code"], x["message"], x["severity"], x.get("context", {})) for x in raw["warnings"])
    return AnalysisProvenance(raw["capability_id"], raw["node_version"], raw["cache_schema_version"],
        tuple(raw["source_files"]), tuple(raw["source_locations"]), raw["input_dataset_signature"],
        raw["parameters"], raw["model_formula"], raw["reference_group"], tuple(raw["covariates"]),
        raw["unit_of_inference"], raw["random_seed"], raw["software_versions"],
        tuple(raw["output_paths"]), warnings, raw["execution_timestamp_utc"])


def run_tf_regulatory_network(request: TFRegulatoryNetworkInput,
                              context: AnalysisContext) -> TFRegulatoryNetworkOutput:
    paths = (request.feature_program_path, request.background_universe_path,
             request.dorothea_path, request.collectri_path)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("one or more CAP-TF-001 input artifacts are missing")
    programs, excluded, background = validate_feature_programs(request)
    resource_tables = []
    resource_qc = []
    for database, path in (("DoRothEA", request.dorothea_path), ("CollecTRI", request.collectri_path)):
        normalized, qc = normalize_resource(_read_table(path), database, request)
        resource_tables.append(normalized)
        resource_qc.append(qc)
    resources = pd.concat(resource_tables, ignore_index=True)
    signatures = {str(path): file_hash(path) for path in paths}
    params = {**request.parameters(), "implementation_version": IMPLEMENTATION_VERSION,
        "input_artifact_hashes": signatures, "normalized_feature_program_hash": _content_hash(programs),
        "normalized_background_hash": _content_hash(background),
        "resolved_resource_hashes": {x["resource_name"]: x["resolved_resource_table_hash"] for x in resource_qc},
        "feature_program_metadata": _compact_feature_program_metadata(programs)}
    key = make_cache_key(capability_id=CAPABILITY_ID, node_version=NODE_VERSION,
        cache_schema_version=CACHE_SCHEMA_VERSION, dataset_signature=context.dataset_signature,
        parameters=params)
    manifest_path = context.cache_root / CAPABILITY_ID.lower() / key / "manifest.json"
    cached = load_complete_manifest(manifest_path, key)
    if cached:
        named = {Path(path).name: str(path) for path in cached.output_files}
        provenance = _load_provenance(Path(named["provenance.json"]))
        artifacts = tuple((logical, named[filename], category.value, media)
                          for logical, filename, category, media in ARTIFACTS)
        status = CapabilityStatus.COMPLETED_WITH_WARNINGS if provenance.warnings else CapabilityStatus.COMPLETED
        return TFRegulatoryNetworkOutput(CAPABILITY_ID, IMPLEMENTATION_VERSION, NODE_VERSION,
            status, key, True, artifacts, named["provenance.json"], str(manifest_path),
            provenance.warnings, provenance)
    tables, warnings, summary = calculate_tf_regulatory_network(
        programs, background, resources, pd.DataFrame(resource_qc), excluded, request)
    output_dir = context.capability_output_dir(CAPABILITY_ID) / key
    artifact_paths = []
    for logical, filename, category, media in ARTIFACTS:
        path = output_dir / filename
        if logical == "qc_summary":
            _atomic_json(summary, path)
        else:
            _atomic_table(tables[logical], path)
        artifact_paths.append((logical, str(path), category.value, media))
    software = {**context.software_versions, "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__}
    provenance_parameters = {**params, **summary, "input_artifact_paths": [str(x) for x in paths],
        "resource_names": ["DoRothEA", "CollecTRI"], "resource_hashes": params["resolved_resource_hashes"],
        "resource_resolution_mode": "caller_supplied_local",
        "resource_provider": "decoupler_or_local_injected_table",
        "decoupler_version": None, "plotting_seed": 17}
    provenance_path = output_dir / "provenance.json"
    provenance = build_provenance(capability_id=CAPABILITY_ID, node_version=NODE_VERSION,
        cache_schema_version=CACHE_SCHEMA_VERSION, source_files=SOURCE_FILES,
        source_locations=SOURCE_LOCATIONS, input_dataset_signature=context.dataset_signature,
        parameters=provenance_parameters, model_formula="Fisher exact overrepresentation (alternative='greater')",
        reference_group=None, covariates=(), unit_of_inference="feature_program",
        random_seed=None, software_versions=software,
        output_paths=tuple(path for _, path, _, _ in artifact_paths), warnings=warnings)
    write_provenance_atomic(provenance, provenance_path)
    outputs = tuple(path for _, path, _, _ in artifact_paths) + (str(provenance_path),)
    manifest = build_cache_manifest(cache_key=key, capability_id=CAPABILITY_ID,
        node_version=NODE_VERSION, cache_schema_version=CACHE_SCHEMA_VERSION,
        input_signature=signatures[str(request.feature_program_path)],
        source_dataset_signature=context.dataset_signature, parameters=params,
        output_files=outputs, completion_status="complete", warnings=warnings,
        software_versions=software)
    write_manifest_atomic(manifest, manifest_path)
    return TFRegulatoryNetworkOutput(CAPABILITY_ID, IMPLEMENTATION_VERSION, NODE_VERSION,
        CapabilityStatus.COMPLETED_WITH_WARNINGS if warnings else CapabilityStatus.COMPLETED,
        key, False, tuple(artifact_paths), str(provenance_path), str(manifest_path),
        warnings, provenance)
