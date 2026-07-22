"""CAP-COMP-001 deterministic sample-level cell-state abundance."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd

from ..cache import build_cache_manifest, load_complete_manifest, make_cache_key, write_manifest_atomic
from ..context import AnalysisContext
from ..provenance import build_provenance, write_provenance_atomic
from ..schemas.abundance import AbundanceInput, AbundanceOutput
from ..schemas.common import AnalysisProvenance, StructuredWarning, WarningSeverity
from ..validation.metadata import validate_required_columns, validate_sample_metadata

CAPABILITY_ID = "CAP-COMP-001"
NODE_VERSION = "1.0.0"
CACHE_SCHEMA_VERSION = 1
SOURCE_FILES = (
    "scripts/Cell_type_kinetics.ipynb",
    "scripts/13_generate_cell_type_kinetic_tables.ipynb",
)
SOURCE_LOCATIONS = (
    "Cell_type_kinetics.ipynb:cell 6c9a35cc",
    "Cell_type_kinetics.ipynb:cell f65c44ce",
    "13_generate_cell_type_kinetic_tables.ipynb:cell 880fac3c",
    "13_generate_cell_type_kinetic_tables.ipynb:cell e58b6355",
)
EXCLUDE_PATTERN = (
    "plasma|plasmablast|erythro|erythroid|erythroblast|"
    "red blood|rbc|hemoglobin|hbb|hba"
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
    return StructuredWarning(
        value["code"], value["message"], value.get("severity", "warning"),
        value.get("context", {}),
    )


def _load_provenance(path: Path) -> AnalysisProvenance:
    value = json.loads(path.read_text())
    return AnalysisProvenance(
        capability_id=value["capability_id"],
        node_version=value["node_version"],
        cache_schema_version=value["cache_schema_version"],
        source_files=tuple(value["source_files"]),
        source_locations=tuple(value["source_locations"]),
        input_dataset_signature=value["input_dataset_signature"],
        parameters=value["parameters"],
        model_formula=value["model_formula"],
        reference_group=value["reference_group"],
        covariates=tuple(value["covariates"]),
        unit_of_inference=value["unit_of_inference"],
        random_seed=value["random_seed"],
        software_versions=value["software_versions"],
        output_paths=tuple(value["output_paths"]),
        warnings=tuple(_warning(item) for item in value["warnings"]),
        execution_timestamp_utc=value["execution_timestamp_utc"],
    )


def _dataset_warnings(samples: pd.DataFrame, request: AbundanceInput) -> list[StructuredWarning]:
    presence = pd.crosstab(samples[request.stage_column], samples[request.dataset_column])
    warnings: list[StructuredWarning] = []
    exclusive = [str(column) for column in presence if int((presence[column] > 0).sum()) == 1]
    if exclusive:
        warnings.append(StructuredWarning(
            "DATASET_STAGE_EXCLUSIVE",
            "One or more datasets contain samples from only one retained stage",
            context={"datasets": exclusive},
        ))
    if len(presence.index) > 1 and not any(bool((presence[column] > 0).all()) for column in presence):
        warnings.append(StructuredWarning(
            "NO_DATASET_OVERLAP_ACROSS_STAGES",
            "No dataset contains every retained stage; downstream trends may be confounded",
            WarningSeverity.ERROR,
            {"stages": [str(value) for value in presence.index]},
        ))
    return warnings


def calculate_sample_abundance(
    metadata: pd.DataFrame,
    request: AbundanceInput,
) -> tuple[pd.DataFrame, pd.DataFrame, list[StructuredWarning], dict[str, int]]:
    """Calculate source-compatible sample fractions and stage summaries."""
    required = [request.sample_column, request.dataset_column, request.stage_column, request.cell_state_column]
    if request.patient_column:
        required.append(request.patient_column)
    if request.macro_column:
        required.append(request.macro_column)
    validate_required_columns(metadata, required)
    warnings = validate_sample_metadata(
        metadata,
        sample_column=request.sample_column,
        patient_column=request.patient_column,
        dataset_column=request.dataset_column,
        group_column=request.stage_column,
    )
    obs = metadata[list(dict.fromkeys(required))].copy()
    missing = obs[[request.sample_column, request.stage_column, request.cell_state_column]].isna().any(axis=1)
    if missing.any():
        warnings.append(StructuredWarning(
            "CELLS_DROPPED_MISSING_METADATA",
            "Cells missing sample, stage, or cell-state metadata were excluded",
            context={"count": int(missing.sum())},
        ))
    obs = obs.loc[~missing].copy()
    keep_stage = obs[request.stage_column].isin(request.stage_order)
    if (~keep_stage).any():
        warnings.append(StructuredWarning(
            "CELLS_OUTSIDE_STAGE_ORDER",
            "Cells outside the declared stage order were excluded",
            WarningSeverity.INFO,
            {"count": int((~keep_stage).sum())},
        ))
    obs = obs.loc[keep_stage].copy()

    profile_excluded = 0
    if request.analysis_profile == "plasma_rbc_removed":
        excluded = obs[request.cell_state_column].astype(str).str.contains(
            EXCLUDE_PATTERN, case=False, na=False, regex=True
        )
        if request.macro_column:
            excluded |= obs[request.macro_column].astype(str).str.contains(
                EXCLUDE_PATTERN, case=False, na=False, regex=True
            )
        profile_excluded = int(excluded.sum())
        obs = obs.loc[~excluded].copy()
        warnings.append(StructuredWarning(
            "PLASMA_RBC_DENOMINATOR_EXCLUSION",
            "Plasma/RBC-like annotations were removed from numerator and denominator",
            WarningSeverity.INFO,
            {"count": profile_excluded, "pattern": EXCLUDE_PATTERN},
        ))
    if obs.empty:
        raise ValueError("empty target population after source filters")
    obs[request.stage_column] = pd.Categorical(
        obs[request.stage_column], categories=request.stage_order, ordered=True
    )

    sample_columns = [request.sample_column, request.stage_column, request.dataset_column]
    if request.patient_column:
        sample_columns.append(request.patient_column)
    samples = obs[sample_columns].drop_duplicates()
    if samples[request.sample_column].duplicated().any():
        raise ValueError("sample metadata is not one row per biological sample")
    warnings.extend(_dataset_warnings(samples, request))
    stage_n = samples.groupby(request.stage_column, observed=True)[request.sample_column].nunique()
    low = {
        str(stage): int(stage_n.get(stage, 0))
        for stage in request.stage_order
        if 0 < int(stage_n.get(stage, 0)) < request.minimum_samples_per_stage_for_warning
    }
    if low:
        warnings.append(StructuredWarning(
            "LOW_STAGE_REPLICATION",
            "Retained stages are below the descriptive replication threshold",
            context={"counts": low, "threshold": request.minimum_samples_per_stage_for_warning},
        ))

    counts = (
        obs.groupby([request.sample_column, request.stage_column, request.cell_state_column], observed=True)
        .size().reset_index(name="n_cells")
    )
    totals = (
        obs.groupby([request.sample_column, request.stage_column], observed=True)
        .size().reset_index(name="sample_total")
    )
    observed = counts.merge(totals, on=[request.sample_column, request.stage_column], validate="many_to_one")
    observed["fraction"] = observed["n_cells"] / observed["sample_total"]
    states = sorted(obs[request.cell_state_column].astype(str).unique())
    complete = (
        samples.assign(_key=1)
        .merge(pd.DataFrame({request.cell_state_column: states, "_key": 1}), on="_key")
        .drop(columns="_key")
    )
    fractions = complete.merge(
        observed,
        on=[request.sample_column, request.stage_column, request.cell_state_column],
        how="left",
        validate="one_to_one",
    )
    fractions["n_cells"] = fractions["n_cells"].fillna(0).astype(int)
    sample_totals = totals.set_index(request.sample_column)["sample_total"]
    fractions["sample_total"] = fractions[request.sample_column].map(sample_totals).astype(int)
    fractions["fraction"] = fractions["fraction"].fillna(0.0)

    macro_map: dict[Any, Any] = {}
    if request.macro_column:
        pairs = obs[[request.cell_state_column, request.macro_column]].dropna().drop_duplicates()
        tied = []
        for state, subset in pairs.groupby(request.cell_state_column, sort=False):
            frequency = subset[request.macro_column].value_counts()
            if len(frequency) > 1 and frequency.iloc[0] == frequency.iloc[1]:
                tied.append(str(state))
            macro_map[state] = frequency.index[0]
        fractions[request.macro_column] = fractions[request.cell_state_column].map(macro_map)
        if tied:
            warnings.append(StructuredWarning(
                "MACRO_MAPPING_TIE",
                "Source first-mode macro mapping encountered a tie",
                context={"cell_states": tied},
            ))
    if not np.allclose(fractions.groupby(request.sample_column)["fraction"].sum(), 1.0):
        raise ValueError("sample fractions do not sum to one")

    summary = (
        fractions.groupby([request.stage_column, request.cell_state_column], observed=True)
        .agg(
            mean_fraction=("fraction", "mean"),
            sd_fraction=("fraction", "std"),
            sem_fraction=("fraction", lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum())),
            n_samples=("fraction", "size"),
        ).reset_index()
    )
    summary["ci95"] = 1.96 * summary["sem_fraction"]
    summary["ci_low"] = (summary["mean_fraction"] - summary["ci95"]).clip(lower=0)
    summary["ci_high"] = summary["mean_fraction"] + summary["ci95"]
    if request.macro_column:
        summary[request.macro_column] = summary[request.cell_state_column].map(macro_map)
    report = {
        "cells_input": len(metadata),
        "cells_after_filters": len(obs),
        "cells_excluded_missing": int(missing.sum()),
        "cells_excluded_stage": int((~keep_stage).sum()),
        "cells_excluded_profile": profile_excluded,
        "biological_samples": samples[request.sample_column].nunique(),
        "cell_states": len(states),
    }
    return fractions, summary, warnings, report


def run_cell_state_abundance(request: AbundanceInput, context: AnalysisContext) -> AbundanceOutput:
    if not request.metadata_path.exists():
        raise FileNotFoundError(request.metadata_path)
    input_signature = _file_signature(request.metadata_path)
    parameters = request.parameters()
    cache_key = make_cache_key(
        capability_id=CAPABILITY_ID,
        node_version=NODE_VERSION,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        dataset_signature=context.dataset_signature,
        parameters={**parameters, "input_signature": input_signature},
    )
    cache_dir = context.cache_root / CAPABILITY_ID.lower() / cache_key
    manifest_path = cache_dir / "manifest.json"
    cached = load_complete_manifest(manifest_path, cache_key)
    if cached:
        named = {Path(path).name: path for path in cached.output_files}
        provenance = _load_provenance(Path(named["provenance.json"]))
        return AbundanceOutput(
            CAPABILITY_ID, NODE_VERSION, cache_key, True,
            named["sample_level_cell_type_fractions.csv"],
            named["cell_type_stage_fraction_summary.csv"],
            named["provenance.json"], str(manifest_path),
            provenance.warnings, provenance,
        )

    separator = "\t" if request.input_format == "tsv" else ","
    metadata = pd.read_csv(request.metadata_path, sep=separator, low_memory=False)
    fractions, summary, warnings, counts = calculate_sample_abundance(metadata, request)
    output_dir = context.capability_output_dir(CAPABILITY_ID) / cache_key
    sample_path = output_dir / "sample_level_cell_type_fractions.csv"
    summary_path = output_dir / "cell_type_stage_fraction_summary.csv"
    provenance_path = output_dir / "provenance.json"
    _atomic_csv(fractions, sample_path)
    _atomic_csv(summary, summary_path)
    software = {
        **context.software_versions,
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
    }
    provenance = build_provenance(
        capability_id=CAPABILITY_ID, node_version=NODE_VERSION,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        source_files=SOURCE_FILES, source_locations=SOURCE_LOCATIONS,
        input_dataset_signature=context.dataset_signature,
        parameters={**parameters, "input_signature": input_signature, "counts": counts,
                    "denominator": "all retained cells within biological sample"},
        model_formula=None, reference_group=None, covariates=(),
        unit_of_inference="biological_sample", random_seed=None,
        software_versions=software,
        output_paths=(str(sample_path), str(summary_path)), warnings=warnings,
    )
    write_provenance_atomic(provenance, provenance_path)
    manifest = build_cache_manifest(
        cache_key=cache_key, capability_id=CAPABILITY_ID, node_version=NODE_VERSION,
        cache_schema_version=CACHE_SCHEMA_VERSION, input_signature=input_signature,
        source_dataset_signature=context.dataset_signature, parameters=parameters,
        output_files=(str(sample_path), str(summary_path), str(provenance_path)),
        completion_status="complete", warnings=warnings, software_versions=software,
    )
    write_manifest_atomic(manifest, manifest_path)
    return AbundanceOutput(
        CAPABILITY_ID, NODE_VERSION, cache_key, False,
        str(sample_path), str(summary_path), str(provenance_path),
        str(manifest_path), tuple(warnings), provenance,
    )
