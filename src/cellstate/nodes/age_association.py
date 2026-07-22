"""Deterministic sample-level immune-age association capabilities."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from ..cache import build_cache_manifest, load_complete_manifest, make_cache_key, write_manifest_atomic
from ..context import AnalysisContext
from ..provenance import build_provenance, write_provenance_atomic
from ..schemas.age_association import (
    AgeAssociationOutput,
    GroupAgeAssociationInput,
    OrderedAgeAssociationInput,
)
from ..schemas.common import AnalysisProvenance, StructuredWarning, WarningSeverity
from ..validation.metadata import validate_required_columns, validate_sample_metadata
from ..validation.replication import validate_biological_replication

GROUP_CAPABILITY_ID = "CAP-STAT-001"
ORDERED_CAPABILITY_ID = "CAP-STAT-002"
NODE_VERSION = "1.0.0"
CACHE_SCHEMA_VERSION = 1
GROUP_SOURCES = (
    "scripts/02_run_pbmc_clock.R", "scripts/02_run_pbmc_clock_CD8T.R",
    "scripts/05_apply_bm_retrained_cell_split_multi_celltypes.R",
    "scripts/09_idecell_response_bm_retrained_age.R",
    "scripts/10_apply_ciim_pbmc_to_idecell_response.R",
)
GROUP_LOCATIONS = (
    "02_run_pbmc_clock.R:lines 592-613",
    "02_run_pbmc_clock_CD8T.R:lines 592-613",
    "05_apply_bm_retrained_cell_split_multi_celltypes.R:lines 450-587",
    "09_idecell_response_bm_retrained_age.R:response-group stats block lines 470-553",
    "10_apply_ciim_pbmc_to_idecell_response.R:response-group stats block lines 430-514",
)
ORDERED_SOURCES = (
    "scripts/09_idecell_response_bm_retrained_age.R",
    "scripts/10_apply_ciim_pbmc_to_idecell_response.R",
)
ORDERED_LOCATIONS = (
    "09_idecell_response_bm_retrained_age.R:lines 512-553",
    "10_apply_ciim_pbmc_to_idecell_response.R:lines 473-514",
)


def _file_signature(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        table.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _warning(value: dict[str, Any]) -> StructuredWarning:
    return StructuredWarning(value["code"], value["message"], value.get("severity", "warning"), value.get("context", {}))


def _load_provenance(path: Path) -> AnalysisProvenance:
    value = json.loads(path.read_text())
    return AnalysisProvenance(
        capability_id=value["capability_id"], node_version=value["node_version"],
        cache_schema_version=value["cache_schema_version"], source_files=tuple(value["source_files"]),
        source_locations=tuple(value["source_locations"]), input_dataset_signature=value["input_dataset_signature"],
        parameters=value["parameters"], model_formula=value["model_formula"],
        reference_group=value["reference_group"], covariates=tuple(value["covariates"]),
        unit_of_inference=value["unit_of_inference"], random_seed=value["random_seed"],
        software_versions=value["software_versions"], output_paths=tuple(value["output_paths"]),
        warnings=tuple(_warning(item) for item in value["warnings"]),
        execution_timestamp_utc=value["execution_timestamp_utc"],
    )


def _bh(values: Sequence[float]) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result
    raw = np.asarray(values, dtype=float)[finite]
    order = np.argsort(raw, kind="mergesort")
    ranked = raw[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty(len(ranked), dtype=float)
    restored[order] = np.minimum(adjusted, 1.0)
    result[finite] = restored
    return result


def _prepare(
    table: pd.DataFrame, *, independent_id: str, class_column: str, outcome_column: str,
    group_column: str, class_levels: Sequence[str], group_levels: Sequence[str],
    dataset_column: str | None = None,
) -> tuple[pd.DataFrame, list[StructuredWarning]]:
    required = [independent_id, class_column, outcome_column, group_column]
    if dataset_column:
        required.append(dataset_column)
    validate_required_columns(table, required)
    warnings = validate_sample_metadata(
        table, sample_column=independent_id, dataset_column=dataset_column, group_column=group_column,
    )
    selected = table[required].copy()
    missing = selected[[independent_id, class_column, outcome_column, group_column]].isna().any(axis=1)
    numeric = pd.to_numeric(selected[outcome_column], errors="coerce")
    nonnumeric = numeric.isna() & selected[outcome_column].notna()
    if nonnumeric.any():
        raise ValueError(f"{outcome_column} contains nonnumeric values")
    selected[outcome_column] = numeric
    if not np.isfinite(selected.loc[~missing, outcome_column]).all():
        raise ValueError(f"{outcome_column} contains nonfinite values")
    if missing.any():
        warnings.append(StructuredWarning(
            "ROWS_DROPPED_MISSING_ANALYSIS_VALUES", "Rows missing ID, class, outcome, or group were excluded",
            context={"count": int(missing.sum())},
        ))
    selected = selected.loc[~missing].copy()
    retained = selected[class_column].isin(class_levels) & selected[group_column].isin(group_levels)
    if (~retained).any():
        warnings.append(StructuredWarning(
            "ROWS_OUTSIDE_DECLARED_LEVELS", "Rows outside declared class or group levels were excluded",
            WarningSeverity.INFO, {"count": int((~retained).sum())},
        ))
    selected = selected.loc[retained].copy()
    if selected.empty:
        raise ValueError("empty analysis population after source filters")
    duplicate = selected.duplicated([independent_id, class_column], keep=False)
    if duplicate.any():
        raise ValueError("independent ID and class must identify one analysis row")
    return selected, warnings


def calculate_group_age_association(
    table: pd.DataFrame, request: GroupAgeAssociationInput,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[StructuredWarning]]:
    analysis, warnings = _prepare(
        table, independent_id=request.independent_id_column, class_column=request.class_column,
        outcome_column=request.outcome_column, group_column=request.group_column,
        class_levels=request.class_levels, group_levels=request.group_levels,
        dataset_column=request.dataset_column,
    )
    summaries, omnibus, pairwise = [], [], []
    for class_name in request.class_levels:
        subset = analysis.loc[analysis[request.class_column] == class_name]
        if subset.empty:
            warnings.append(StructuredWarning("EMPTY_CLASS", "A declared cell class has no retained rows", context={"class": class_name}))
            continue
        represented = [level for level in request.group_levels if bool((subset[request.group_column] == level).any())]
        if len(represented) < 2:
            warnings.append(StructuredWarning("INSUFFICIENT_GROUPS", "Class has fewer than two represented groups", WarningSeverity.ERROR, {"class": class_name}))
            continue
        replicate_counts = {
            level: int(subset.loc[subset[request.group_column] == level, request.independent_id_column].nunique())
            for level in represented
        }
        below_two = {level: count for level, count in replicate_counts.items() if count < 2}
        if below_two:
            warnings.append(StructuredWarning(
                "LOW_BIOLOGICAL_REPLICATION",
                "A represented group has fewer than two biological samples; source tests are retained but inference is unreliable",
                WarningSeverity.ERROR,
                {"class": class_name, "counts": below_two, "warning_threshold": 2},
            ))
        warnings.extend(validate_biological_replication(
            subset, sample_column=request.independent_id_column, group_column=request.group_column,
            groups=represented, minimum_samples_per_group=request.minimum_replicates_per_group,
            dataset_column=request.dataset_column,
        ))
        for level in represented:
            values = subset.loc[subset[request.group_column] == level, request.outcome_column]
            summaries.append({request.class_column: class_name, request.group_column: level,
                              "n": len(values), "mean": values.mean(), "median": values.median(),
                              "sd": values.std(ddof=1)})
        arrays = [subset.loc[subset[request.group_column] == level, request.outcome_column].to_numpy() for level in represented]
        if np.unique(np.concatenate(arrays)).size == 1:
            statistic, p_value = 0.0, 1.0
            warnings.append(StructuredWarning("CONSTANT_OUTCOME", "All outcomes are identical within a class", context={"class": class_name}))
        else:
            statistic, p_value = stats.kruskal(*arrays)
        omnibus.append({request.class_column: class_name, "n": len(subset), "groups": len(represented),
                        "statistic": statistic, "p_value": p_value})
        class_pairs = []
        for group1, group2 in request.resolved_contrasts():
            if group1 not in represented or group2 not in represented:
                continue
            first = subset.loc[subset[request.group_column] == group1, request.outcome_column].to_numpy()
            second = subset.loc[subset[request.group_column] == group2, request.outcome_column].to_numpy()
            test = stats.mannwhitneyu(first, second, alternative="two-sided", method="auto", use_continuity=True)
            class_pairs.append({request.class_column: class_name, "group1": group1, "group2": group2,
                                "n1": len(first), "n2": len(second), "statistic": test.statistic,
                                "p_value": test.pvalue})
        adjusted = _bh([row["p_value"] for row in class_pairs])
        for row, value in zip(class_pairs, adjusted):
            row["p_adjusted"] = value
        pairwise.extend(class_pairs)
        if subset[request.outcome_column].duplicated().any():
            warnings.append(StructuredWarning("TIED_OUTCOMES", "Tied immune-age outcomes are present", WarningSeverity.INFO, {"class": class_name}))
    return analysis, pd.DataFrame(summaries), pd.DataFrame(omnibus), pd.DataFrame(pairwise), warnings


def calculate_ordered_age_association(
    table: pd.DataFrame, request: OrderedAgeAssociationInput,
) -> tuple[pd.DataFrame, pd.DataFrame, list[StructuredWarning]]:
    levels = tuple(label for label, _ in request.score_mapping)
    analysis, warnings = _prepare(
        table, independent_id=request.independent_id_column, class_column=request.class_column,
        outcome_column=request.outcome_column, group_column=request.group_column,
        class_levels=request.class_levels, group_levels=levels,
    )
    score_map = dict(request.score_mapping)
    analysis["response_score"] = analysis[request.group_column].map(score_map).astype(float)
    results = []
    for class_name in request.class_levels:
        subset = analysis.loc[analysis[request.class_column] == class_name]
        if subset.empty:
            warnings.append(StructuredWarning("EMPTY_CLASS", "A declared cell class has no retained rows", context={"class": class_name}))
            continue
        represented = subset[request.group_column].nunique()
        if represented < request.minimum_represented_levels:
            warnings.append(StructuredWarning(
                "MISSING_ORDERED_LEVELS", "Fewer than the declared minimum ordered levels are represented",
                WarningSeverity.ERROR, {"class": class_name, "represented": int(represented)},
            ))
            continue
        if subset[request.outcome_column].nunique() < 2:
            rho, p_value = np.nan, np.nan
            warnings.append(StructuredWarning("CONSTANT_OUTCOME", "Spearman association is undefined for constant outcomes", WarningSeverity.ERROR, {"class": class_name}))
        else:
            result = stats.spearmanr(subset["response_score"], subset[request.outcome_column], alternative="two-sided")
            rho, p_value = result.statistic, result.pvalue
        results.append({request.class_column: class_name, "n": len(subset), "represented_levels": int(represented),
                        "spearman_rho": rho, "p_value": p_value})
        if subset[request.outcome_column].duplicated().any():
            warnings.append(StructuredWarning("TIED_OUTCOMES", "Tied immune-age outcomes are present", WarningSeverity.INFO, {"class": class_name}))
    result_table = pd.DataFrame(results)
    if not result_table.empty:
        result_table["p_adjusted"] = _bh(result_table["p_value"].to_numpy())
        result_table = result_table.sort_values("p_adjusted", na_position="last").reset_index(drop=True)
    warnings.append(StructuredWarning(
        "ORDINAL_SPACING_ASSUMPTION", "Spearman analysis uses the explicitly supplied numeric response scores",
        WarningSeverity.INFO, {"mapping": dict(request.score_mapping)},
    ))
    return analysis, result_table, warnings


def _run(request: Any, context: AnalysisContext, capability_id: str, calculator: Any,
         source_files: Sequence[str], source_locations: Sequence[str], filenames: Sequence[str],
         formula: str) -> AgeAssociationOutput:
    if not request.table_path.exists():
        raise FileNotFoundError(request.table_path)
    input_signature = _file_signature(request.table_path)
    parameters = request.parameters()
    cache_key = make_cache_key(
        capability_id=capability_id, node_version=NODE_VERSION, cache_schema_version=CACHE_SCHEMA_VERSION,
        dataset_signature=context.dataset_signature, parameters={**parameters, "input_signature": input_signature},
    )
    cache_dir = context.cache_root / capability_id.lower() / cache_key
    manifest_path = cache_dir / "manifest.json"
    cached = load_complete_manifest(manifest_path, cache_key)
    if cached:
        named = {Path(path).name: path for path in cached.output_files}
        provenance = _load_provenance(Path(named["provenance.json"]))
        stats_paths = tuple(named[name] for name in filenames[1:])
        return AgeAssociationOutput(capability_id, NODE_VERSION, cache_key, True, named[filenames[0]],
                                    stats_paths, named["provenance.json"], str(manifest_path),
                                    provenance.warnings, provenance)
    table = pd.read_csv(request.table_path, sep="\t" if request.input_format == "tsv" else ",", low_memory=False)
    calculated = calculator(table, request)
    warnings = calculated[-1]
    tables = calculated[:-1]
    output_dir = context.capability_output_dir(capability_id) / cache_key
    paths = tuple(output_dir / name for name in filenames)
    for value, path in zip(tables, paths):
        _atomic_csv(value, path)
    provenance_path = output_dir / "provenance.json"
    software = {**context.software_versions, "python": platform.python_version(),
                "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    provenance = build_provenance(
        capability_id=capability_id, node_version=NODE_VERSION, cache_schema_version=CACHE_SCHEMA_VERSION,
        source_files=source_files, source_locations=source_locations,
        input_dataset_signature=context.dataset_signature,
        parameters={**parameters, "input_signature": input_signature, "rows_input": len(table),
                    "rows_analyzed": len(tables[0]), "statistical_backend": "scipy"},
        model_formula=formula, reference_group=None, covariates=(), unit_of_inference="biological_sample_or_patient",
        random_seed=None, software_versions=software, output_paths=tuple(str(path) for path in paths), warnings=warnings,
    )
    write_provenance_atomic(provenance, provenance_path)
    outputs = (*paths, provenance_path)
    manifest = build_cache_manifest(
        cache_key=cache_key, capability_id=capability_id, node_version=NODE_VERSION,
        cache_schema_version=CACHE_SCHEMA_VERSION, input_signature=input_signature,
        source_dataset_signature=context.dataset_signature, parameters=parameters,
        output_files=tuple(str(path) for path in outputs), completion_status="complete",
        warnings=warnings, software_versions=software,
    )
    write_manifest_atomic(manifest, manifest_path)
    return AgeAssociationOutput(capability_id, NODE_VERSION, cache_key, False, str(paths[0]),
                                tuple(str(path) for path in paths[1:]), str(provenance_path),
                                str(manifest_path), tuple(warnings), provenance)


def run_group_age_association(request: GroupAgeAssociationInput, context: AnalysisContext) -> AgeAssociationOutput:
    return _run(request, context, GROUP_CAPABILITY_ID, calculate_group_age_association,
                GROUP_SOURCES, GROUP_LOCATIONS,
                ("analysis_set.csv", "group_summaries.csv", "kruskal_wallis.csv", "pairwise_wilcoxon_bh.csv"),
                f"{request.outcome_column} ~ {request.group_column}")


def run_ordered_age_association(request: OrderedAgeAssociationInput, context: AnalysisContext) -> AgeAssociationOutput:
    return _run(request, context, ORDERED_CAPABILITY_ID, calculate_ordered_age_association,
                ORDERED_SOURCES, ORDERED_LOCATIONS,
                ("analysis_set.csv", "spearman_bh.csv"),
                f"cor({request.outcome_column}, response_score, method='spearman', exact=FALSE)")
