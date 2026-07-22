"""Cross-sectional disease progression and paired longitudinal kinetics."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from ..cache import build_cache_manifest, load_complete_manifest, make_cache_key, write_manifest_atomic
from ..context import AnalysisContext
from ..provenance import build_provenance, write_provenance_atomic
from ..schemas.common import AnalysisProvenance, StructuredWarning, WarningSeverity
from ..schemas.progression import (
    CrossSectionalProgressionInput, LongitudinalKineticsInput,
    PairedProgressionInput, ProgressionOutput,
)
from ..utilities.design import ordered_indicator_design
from ..validation.estimability import validate_design_estimability
from ..validation.metadata import validate_required_columns, validate_sample_metadata

NODE_VERSION = "1.0.0"
CACHE_SCHEMA_VERSION = 1
CROSS_ID = "CAP-COMP-002"
PAIRED_ID = "CAP-STAT-003"
LONGITUDINAL_ID = "CAP-STAT-004"
CROSS_SOURCES = ("scripts/Cell_type_kinetics.ipynb", "scripts/13_generate_cell_type_kinetic_tables.ipynb")
CROSS_LOCATIONS = (
    "Cell_type_kinetics.ipynb:cell 6c9a35cc trend block",
    "Cell_type_kinetics.ipynb:cell f65c44ce trend block",
    "13_generate_cell_type_kinetic_tables.ipynb:cell e58b6355 lines 2894-2941",
)
PAIRED_SOURCES = ("scripts/07_paired_ndmm_rrmm_bm_retrained_change.R",)
PAIRED_LOCATIONS = ("07_paired_ndmm_rrmm_bm_retrained_change.R:lines 35-139",)
LONG_SOURCES = ("scripts/12_novartis_bm_immune_age_kinetics.R",)
LONG_LOCATIONS = ("12_novartis_bm_immune_age_kinetics.R:lines 382-474",)


def _signature(path: Path) -> str:
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


def _bh(values: Sequence[float]) -> np.ndarray:
    result = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result
    raw = np.asarray(values, dtype=float)[finite]
    order = np.argsort(raw, kind="mergesort")
    ranked = raw[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty(len(ranked))
    restored[order] = np.minimum(adjusted, 1.0)
    result[finite] = restored
    return result


def _load_provenance(path: Path) -> AnalysisProvenance:
    raw = json.loads(path.read_text())
    warnings = tuple(StructuredWarning(x["code"], x["message"], x["severity"], x.get("context", {})) for x in raw["warnings"])
    return AnalysisProvenance(
        raw["capability_id"], raw["node_version"], raw["cache_schema_version"], tuple(raw["source_files"]),
        tuple(raw["source_locations"]), raw["input_dataset_signature"], raw["parameters"], raw["model_formula"],
        raw["reference_group"], tuple(raw["covariates"]), raw["unit_of_inference"], raw["random_seed"],
        raw["software_versions"], tuple(raw["output_paths"]), warnings, raw["execution_timestamp_utc"],
    )


def _dataset_stage_warnings(table: pd.DataFrame, stage: str, dataset: str | None) -> list[StructuredWarning]:
    if dataset is None:
        return []
    presence = pd.crosstab(table[stage], table[dataset])
    warnings = []
    exclusive = [str(value) for value in presence if int((presence[value] > 0).sum()) == 1]
    if exclusive:
        warnings.append(StructuredWarning("DATASET_STAGE_CONFOUNDING", "Datasets contain only one represented stage/timepoint", context={"datasets": exclusive}))
    if len(presence.index) > 1 and not any(bool((presence[value] > 0).all()) for value in presence):
        warnings.append(StructuredWarning("NO_DATASET_STAGE_OVERLAP", "No dataset contains every represented stage/timepoint", WarningSeverity.ERROR, {"levels": [str(x) for x in presence.index]}))
    return warnings


def calculate_cross_sectional_progression(table: pd.DataFrame, request: CrossSectionalProgressionInput):
    required = [request.sample_column, request.stage_column, request.stratum_column, request.outcome_column]
    if request.dataset_column:
        required.append(request.dataset_column)
    validate_required_columns(table, required)
    warnings = validate_sample_metadata(table, sample_column=request.sample_column,
                                        dataset_column=request.dataset_column, group_column=request.stage_column)
    data = table[required].copy()
    data[request.outcome_column] = pd.to_numeric(data[request.outcome_column], errors="raise")
    data = data.dropna(subset=[request.sample_column, request.stage_column, request.stratum_column, request.outcome_column])
    data = data.loc[data[request.stage_column].isin(request.stage_order)].copy()
    if data.empty:
        raise ValueError("empty cross-sectional progression analysis set")
    if data.duplicated([request.sample_column, request.stratum_column]).any():
        raise ValueError("sample and stratum must identify one cross-sectional row")
    if not np.isfinite(data[request.outcome_column]).all():
        raise ValueError("outcome contains nonfinite values")
    warnings.extend(_dataset_stage_warnings(data.drop_duplicates(request.sample_column), request.stage_column, request.dataset_column))
    stage_map = {value: index for index, value in enumerate(request.stage_order)}
    data["stage_num"] = data[request.stage_column].map(stage_map).astype(float)
    rows = []
    for stratum, subset in data.groupby(request.stratum_column, sort=False):
        counts = subset.groupby(request.stage_column, observed=True)[request.sample_column].nunique()
        passing = int((counts >= request.minimum_samples_per_stage).sum())
        mean_value = float(subset[request.outcome_column].mean())
        eligible = passing >= request.minimum_stages_passing and mean_value >= request.minimum_mean_outcome
        row = {request.stratum_column: stratum, "n_samples": subset[request.sample_column].nunique(),
               "n_stages_passing": passing, "mean_outcome": mean_value, "eligible": eligible,
               "slope": np.nan, "intercept": np.nan, "r_value": np.nan, "p_value": np.nan, "std_err": np.nan}
        if eligible:
            design = np.column_stack([np.ones(len(subset)), subset["stage_num"].to_numpy()])
            validate_design_estimability(design, term_names=("Intercept", "stage_num"))
            fit = stats.linregress(subset["stage_num"], subset[request.outcome_column])
            row.update(slope=fit.slope, intercept=fit.intercept, r_value=fit.rvalue,
                       p_value=fit.pvalue, std_err=fit.stderr)
        else:
            warnings.append(StructuredWarning("INELIGIBLE_CROSS_SECTIONAL_TREND", "Stratum failed source trend eligibility filters", WarningSeverity.INFO,
                                              {"stratum": str(stratum), "stages_passing": passing, "mean_outcome": mean_value}))
        rows.append(row)
    results = pd.DataFrame(rows)
    results["p_adjusted"] = _bh(results["p_value"].to_numpy())
    return data, results, warnings


def _paired_wilcoxon(later: np.ndarray, earlier: np.ndarray) -> float:
    delta = later - earlier
    if np.allclose(delta, 0):
        return np.nan
    return float(stats.wilcoxon(later, earlier, alternative="two-sided", method="auto").pvalue)


def calculate_paired_progression(table: pd.DataFrame, request: PairedProgressionInput):
    required = [request.sample_column, request.patient_column, request.class_column, request.stage_column, request.outcome_column]
    if request.dataset_column:
        required.append(request.dataset_column)
    validate_required_columns(table, required)
    warnings = validate_sample_metadata(table, sample_column=request.sample_column, patient_column=request.patient_column,
                                        dataset_column=request.dataset_column, group_column=request.stage_column)
    data = table[required].dropna(subset=[request.sample_column, request.patient_column, request.class_column,
                                         request.stage_column, request.outcome_column]).copy()
    if data[request.patient_column].astype(str).str.strip().eq("").any():
        raise ValueError("patient identifiers must be nonblank")
    data = data.loc[data[request.stage_column].isin((request.earlier_stage, request.later_stage))]
    data[request.outcome_column] = pd.to_numeric(data[request.outcome_column], errors="raise")
    if data.duplicated([request.sample_column, request.class_column]).any():
        raise ValueError("sample and class must identify one source row")
    if data.empty:
        raise ValueError("empty paired progression analysis set")
    warnings.extend(_dataset_stage_warnings(data.drop_duplicates(request.sample_column), request.stage_column, request.dataset_column))
    stage_design, stage_terms = ordered_indicator_design(
        data[request.stage_column], levels=(request.earlier_stage, request.later_stage),
        reference_level=request.earlier_stage)
    validate_design_estimability(stage_design, term_names=stage_terms)
    collapsed = data.groupby([request.patient_column, request.class_column, request.stage_column], as_index=False).agg(
        outcome=(request.outcome_column, "mean"), n_samples=(request.sample_column, "nunique"))
    if (collapsed.n_samples > 1).any():
        warnings.append(StructuredWarning("MULTIPLE_SAMPLES_MEAN_COLLAPSED", "Multiple samples were averaged within patient, stage, and class as in source",
                                          context={"groups": int((collapsed.n_samples > 1).sum())}))
    wide = collapsed.pivot(index=[request.patient_column, request.class_column], columns=request.stage_column,
                           values=["outcome", "n_samples"]).reset_index()
    wide.columns = ["_".join(str(x) for x in value if str(x)) if isinstance(value, tuple) else str(value) for value in wide.columns]
    earlier_col, later_col = f"outcome_{request.earlier_stage}", f"outcome_{request.later_stage}"
    incomplete = wide[[earlier_col, later_col]].isna().any(axis=1)
    if incomplete.any():
        warnings.append(StructuredWarning("INCOMPLETE_PAIRS_DROPPED", "Patients lacking one side of the contrast were excluded", context={"count": int(incomplete.sum())}))
    pairs = wide.loc[~incomplete].copy()
    pairs["delta_later_minus_earlier"] = pairs[later_col] - pairs[earlier_col]
    if not pairs.empty:
        paired_design, paired_terms = ordered_indicator_design(
            [request.earlier_stage, request.later_stage] * len(pairs),
            levels=(request.earlier_stage, request.later_stage), reference_level=request.earlier_stage)
        validate_design_estimability(paired_design, term_names=paired_terms)
    stats_rows = []
    for class_name, subset in pairs.groupby(request.class_column):
        n = len(subset)
        if n < request.minimum_pairs_for_test:
            warnings.append(StructuredWarning("INSUFFICIENT_PAIRED_REPLICATION", "Too few complete patient pairs for source tests", WarningSeverity.ERROR,
                                              {"class": class_name, "n_pairs": n, "minimum": request.minimum_pairs_for_test}))
            wilcox_p = t_p = np.nan
        else:
            later, earlier = subset[later_col].to_numpy(), subset[earlier_col].to_numpy()
            wilcox_p = _paired_wilcoxon(later, earlier)
            delta = later - earlier
            if np.isclose(np.std(delta, ddof=1), 0):
                t_p = np.nan
                warnings.append(StructuredWarning(
                    "PAIRED_T_NON_ESTIMABLE", "Paired t-test variance is zero; no t-test p-value was returned",
                    WarningSeverity.ERROR, {"class": class_name}))
            else:
                t_p = float(stats.ttest_rel(later, earlier).pvalue)
        stats_rows.append({request.class_column: class_name, "n_pairs": n,
                           "mean_delta": subset["delta_later_minus_earlier"].mean(),
                           "median_delta": subset["delta_later_minus_earlier"].median(),
                           "sd_delta": subset["delta_later_minus_earlier"].std(ddof=1),
                           "wilcox_p": wilcox_p, "paired_t_p": t_p})
    results = pd.DataFrame(stats_rows, columns=[request.class_column, "n_pairs", "mean_delta", "median_delta",
                                               "sd_delta", "wilcox_p", "paired_t_p"])
    results["wilcox_p_adjusted"] = _bh(results["wilcox_p"].to_numpy())
    results["paired_t_p_adjusted"] = _bh(results["paired_t_p"].to_numpy())
    return data, collapsed, pairs, results, warnings


def calculate_longitudinal_kinetics(table: pd.DataFrame, request: LongitudinalKineticsInput):
    required = [request.patient_column, request.class_column, request.timepoint_column, request.outcome_column]
    if request.dataset_column:
        required.append(request.dataset_column)
    validate_required_columns(table, required)
    data = table[required].dropna(subset=[request.patient_column, request.class_column, request.timepoint_column, request.outcome_column]).copy()
    if data[request.patient_column].astype(str).str.strip().eq("").any():
        raise ValueError("patient identifiers must be nonblank")
    data = data.loc[data[request.timepoint_column].isin(request.timepoint_order)]
    data[request.outcome_column] = pd.to_numeric(data[request.outcome_column], errors="raise")
    if data.duplicated([request.patient_column, request.class_column, request.timepoint_column]).any():
        raise ValueError("patient, class, and timepoint must identify one longitudinal row")
    if data.empty:
        raise ValueError("empty longitudinal kinetics analysis set")
    if request.dataset_column:
        counts = data.groupby(request.patient_column)[request.dataset_column].nunique()
        if (counts > 1).any():
            raise ValueError("dataset is inconsistent within patient")
    warnings = _dataset_stage_warnings(data, request.timepoint_column, request.dataset_column)
    design, terms = ordered_indicator_design(data[request.timepoint_column], levels=request.timepoint_order,
                                             reference_level=request.timepoint_order[0])
    validate_design_estimability(design, term_names=terms)
    wide = data.pivot(index=[request.patient_column, request.class_column], columns=request.timepoint_column,
                      values=request.outcome_column).reset_index()
    results, deltas = [], wide.copy()
    for earlier, later in request.contrasts:
        delta_name = f"delta_{later}_minus_{earlier}"
        deltas[delta_name] = deltas[later] - deltas[earlier]
        for class_name, subset in wide.groupby(request.class_column):
            complete = subset[[earlier, later]].dropna()
            n = len(complete)
            if n < len(subset):
                warnings.append(StructuredWarning("INCOMPLETE_LONGITUDINAL_PAIRS", "Incomplete patients were excluded contrast-by-contrast",
                                                  context={"class": class_name, "contrast": [earlier, later], "dropped": len(subset) - n}))
            if n < request.minimum_pairs_for_test:
                warnings.append(StructuredWarning("INSUFFICIENT_LONGITUDINAL_REPLICATION", "Too few complete pairs for source Wilcoxon test", WarningSeverity.ERROR,
                                                  {"class": class_name, "contrast": [earlier, later], "n_pairs": n}))
                p_value = np.nan
            else:
                p_value = _paired_wilcoxon(complete[later].to_numpy(), complete[earlier].to_numpy())
            delta = complete[later] - complete[earlier]
            results.append({request.class_column: class_name, "earlier": earlier, "later": later, "n_pairs": n,
                            "mean_delta": delta.mean(), "median_delta": delta.median(), "sd_delta": delta.std(ddof=1),
                            "wilcox_p": p_value})
    inferential = pd.DataFrame(results)
    inferential["wilcox_p_adjusted"] = np.nan
    for contrast, index in inferential.groupby(["earlier", "later"]).groups.items():
        inferential.loc[index, "wilcox_p_adjusted"] = _bh(inferential.loc[index, "wilcox_p"].to_numpy())
    return data, deltas, inferential, warnings


def _run(request: Any, context: AnalysisContext, capability: str, structure: str, calculator: Callable,
         sources: Sequence[str], locations: Sequence[str], descriptive_names: Sequence[str], inferential_names: Sequence[str],
         formula: str, reference: str | None) -> ProgressionOutput:
    if not request.table_path.exists():
        raise FileNotFoundError(request.table_path)
    signature = _signature(request.table_path)
    parameters = {**request.parameters(), "analysis_structure": structure}
    key = make_cache_key(capability_id=capability, node_version=NODE_VERSION, cache_schema_version=CACHE_SCHEMA_VERSION,
                         dataset_signature=context.dataset_signature, parameters={**parameters, "input_signature": signature})
    manifest_path = context.cache_root / capability.lower() / key / "manifest.json"
    cached = load_complete_manifest(manifest_path, key)
    if cached:
        named = {Path(path).name: path for path in cached.output_files}
        provenance = _load_provenance(Path(named["provenance.json"]))
        return ProgressionOutput(capability, NODE_VERSION, key, True,
                                 tuple(named[x] for x in descriptive_names), tuple(named[x] for x in inferential_names),
                                 named["provenance.json"], str(manifest_path), provenance.warnings, provenance)
    table = pd.read_csv(request.table_path, sep="\t" if request.input_format == "tsv" else ",", low_memory=False)
    calculated = calculator(table, request)
    warnings, tables = calculated[-1], calculated[:-1]
    names = (*descriptive_names, *inferential_names)
    if len(names) != len(tables):
        raise RuntimeError("progression output contract mismatch")
    output_dir = context.capability_output_dir(capability) / key
    paths = tuple(output_dir / name for name in names)
    for value, path in zip(tables, paths):
        _atomic_csv(value, path)
    provenance_path = output_dir / "provenance.json"
    software = {**context.software_versions, "python": platform.python_version(), "pandas": pd.__version__,
                "numpy": np.__version__, "scipy": scipy.__version__}
    provenance = build_provenance(
        capability_id=capability, node_version=NODE_VERSION, cache_schema_version=CACHE_SCHEMA_VERSION,
        source_files=sources, source_locations=locations, input_dataset_signature=context.dataset_signature,
        parameters={**parameters, "input_signature": signature, "rows_input": len(table), "rows_analysis": len(tables[0])},
        model_formula=formula, reference_group=reference, covariates=(), unit_of_inference="biological_sample_or_paired_patient",
        random_seed=None, software_versions=software, output_paths=tuple(str(x) for x in paths), warnings=warnings)
    write_provenance_atomic(provenance, provenance_path)
    manifest = build_cache_manifest(cache_key=key, capability_id=capability, node_version=NODE_VERSION,
                                    cache_schema_version=CACHE_SCHEMA_VERSION, input_signature=signature,
                                    source_dataset_signature=context.dataset_signature, parameters=parameters,
                                    output_files=tuple(str(x) for x in (*paths, provenance_path)), completion_status="complete",
                                    warnings=warnings, software_versions=software)
    write_manifest_atomic(manifest, manifest_path)
    return ProgressionOutput(capability, NODE_VERSION, key, False,
                             tuple(str(x) for x in paths[:len(descriptive_names)]),
                             tuple(str(x) for x in paths[len(descriptive_names):]), str(provenance_path),
                             str(manifest_path), tuple(warnings), provenance)


def run_cross_sectional_progression(request: CrossSectionalProgressionInput, context: AnalysisContext) -> ProgressionOutput:
    return _run(request, context, CROSS_ID, "cross_sectional", calculate_cross_sectional_progression,
                CROSS_SOURCES, CROSS_LOCATIONS, ("analysis_set.csv",), ("linear_stage_trends_bh.csv",),
                f"{request.outcome_column} ~ stage_num", request.stage_order[0])


def run_paired_progression(request: PairedProgressionInput, context: AnalysisContext) -> ProgressionOutput:
    return _run(request, context, PAIRED_ID, "paired_ndmm_rrmm", calculate_paired_progression,
                PAIRED_SOURCES, PAIRED_LOCATIONS, ("analysis_set.csv", "patient_stage_means.csv", "paired_deltas.csv"),
                ("paired_wilcoxon_t_bh.csv",),
                f"delta = {request.later_stage} - {request.earlier_stage}; paired tests", request.earlier_stage)


def run_longitudinal_kinetics(request: LongitudinalKineticsInput, context: AnalysisContext) -> ProgressionOutput:
    return _run(request, context, LONGITUDINAL_ID, "longitudinal_treatment", calculate_longitudinal_kinetics,
                LONG_SOURCES, LONG_LOCATIONS, ("analysis_set.csv", "patient_timepoint_deltas.csv"),
                ("complete_pair_wilcoxon_bh.csv",), "paired later vs earlier by declared contrast", request.timepoint_order[0])
