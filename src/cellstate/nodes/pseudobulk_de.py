"""Raw integer sample/state pseudobulk construction for DESeq2 inputs."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import subprocess  # legacy patch point for CAP-DESEQ-002 callers
from typing import Any

import numpy as np
import pandas as pd

from ..cache import build_cache_manifest, load_complete_manifest, make_cache_key, write_manifest_atomic
from ..context import AnalysisContext
from ..provenance import build_provenance, write_provenance_atomic
from ..schemas.common import AnalysisProvenance, StructuredWarning, WarningSeverity
from ..schemas.pseudobulk_de import PseudobulkInput, PseudobulkOutput
from ..validation.metadata import validate_required_columns, validate_sample_metadata

CAPABILITY_ID = "CAP-DESEQ-001"
NODE_VERSION = "1.0.0"
CACHE_SCHEMA_VERSION = 1
SOURCE_FILES = ("scripts/19A_build_DESeq2_pseudobulk_counts.ipynb",)
SOURCE_LOCATIONS = (
    "19A_build_DESeq2_pseudobulk_counts.ipynb:cell d3e0f2c4",
    "19A_build_DESeq2_pseudobulk_counts.ipynb:cell 1c157458",
    "19A_build_DESeq2_pseudobulk_counts.ipynb:cell e165f7d4 build_celltype_pseudobulk_counts",
    "19A_build_DESeq2_pseudobulk_counts.ipynb:cells 626cb0f0, 9d41139a, 3b021ae9, a6f4c41b",
)


def _file_signature(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(table: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        table.to_csv(temporary, index=index)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_name(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", str(value))).strip("_")


def _load_provenance(path: Path) -> AnalysisProvenance:
    raw = json.loads(path.read_text())
    warnings = tuple(StructuredWarning(x["code"], x["message"], x["severity"], x.get("context", {})) for x in raw["warnings"])
    return AnalysisProvenance(
        raw["capability_id"], raw["node_version"], raw["cache_schema_version"], tuple(raw["source_files"]),
        tuple(raw["source_locations"]), raw["input_dataset_signature"], raw["parameters"], raw["model_formula"],
        raw["reference_group"], tuple(raw["covariates"]), raw["unit_of_inference"], raw["random_seed"],
        raw["software_versions"], tuple(raw["output_paths"]), warnings, raw["execution_timestamp_utc"],
    )


def _dataset_warnings(metadata: pd.DataFrame, request: PseudobulkInput, state: str) -> list[StructuredWarning]:
    presence = pd.crosstab(metadata[request.stage_column], metadata[request.dataset_column])
    warnings = []
    exclusive = [str(dataset) for dataset in presence if int((presence[dataset] > 0).sum()) == 1]
    if exclusive:
        warnings.append(StructuredWarning(
            "DATASET_GROUP_CONFOUNDING", "Datasets contain samples from only one retained disease group",
            context={"cell_state": state, "datasets": exclusive}))
    if len(presence.index) > 1 and not any(bool((presence[dataset] > 0).all()) for dataset in presence):
        warnings.append(StructuredWarning(
            "NO_DATASET_GROUP_OVERLAP", "No dataset contains all retained disease groups",
            WarningSeverity.ERROR, {"cell_state": state, "groups": [str(x) for x in presence.index]}))
    return warnings


def construct_raw_pseudobulk(
    raw_counts: pd.DataFrame, metadata: pd.DataFrame, request: PseudobulkInput,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, list[StructuredWarning]]:
    """Sum cell-level raw integer counts within biological sample and cell state."""
    validate_required_columns(raw_counts, [request.cell_id_column])
    required = [request.cell_id_column, request.sample_column, request.dataset_column,
                request.stage_column, request.cell_state_column]
    if request.patient_column:
        required.append(request.patient_column)
    validate_required_columns(metadata, required)
    if raw_counts[request.cell_id_column].duplicated().any() or metadata[request.cell_id_column].duplicated().any():
        raise ValueError("cell identifiers must be unique in counts and metadata")
    count_ids = raw_counts[request.cell_id_column].astype(str)
    metadata_ids = metadata[request.cell_id_column].astype(str)
    if set(count_ids) != set(metadata_ids):
        raise ValueError("raw-count rows and metadata cell identifiers are misaligned")
    genes = [column for column in raw_counts.columns if column != request.cell_id_column]
    if not genes:
        raise ValueError("raw-count matrix contains no genes")
    if len(genes) != len(set(genes)):
        raise ValueError("raw-count matrix contains duplicate gene names")
    numeric = raw_counts[genes].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("raw counts contain nonfinite values")
    if (values < 0).any():
        raise ValueError("raw counts contain negative values")
    if not np.all(values == np.floor(values)):
        raise ValueError("raw counts must be integer values")
    numeric = numeric.astype(np.int64)
    numeric.index = count_ids
    obs = metadata.copy()
    obs.index = metadata_ids
    obs = obs.loc[count_ids]
    warnings = validate_sample_metadata(
        obs, sample_column=request.sample_column, patient_column=request.patient_column,
        dataset_column=request.dataset_column, group_column=request.stage_column)
    missing = obs[required[1:]].isna().any(axis=1)
    if missing.any():
        warnings.append(StructuredWarning("CELLS_DROPPED_MISSING_METADATA", "Cells missing pseudobulk metadata were excluded", context={"count": int(missing.sum())}))
    obs = obs.loc[~missing]
    numeric = numeric.loc[obs.index]
    stage_keep = obs[request.stage_column].isin(request.stages)
    state_keep = obs[request.cell_state_column].isin(request.cell_states)
    target_count = int((stage_keep & state_keep).sum())
    if target_count == 0:
        raise ValueError("empty target population for requested stages and cell states")
    obs = obs.loc[stage_keep & state_keep]
    numeric = numeric.loc[obs.index]
    matrices: dict[str, pd.DataFrame] = {}
    sample_rows, qc_rows = [], []
    for state in request.cell_states:
        state_obs = obs.loc[obs[request.cell_state_column] == state]
        if state_obs.empty:
            warnings.append(StructuredWarning("EMPTY_CELL_STATE", "Requested cell state is absent after source filters", WarningSeverity.ERROR, {"cell_state": state}))
            continue
        kept_samples = []
        vectors = []
        state_meta = []
        for sample, subset in state_obs.groupby(request.sample_column, observed=True, sort=False):
            n_cells = len(subset)
            vector = numeric.loc[subset.index].sum(axis=0)
            library_size = int(vector.sum())
            retained = n_cells >= request.minimum_cells_per_sample_state
            reason = "retained" if retained else "below_minimum_cells"
            qc_rows.append({"cell_state": state, request.sample_column: sample, "n_cells": n_cells,
                            "library_size": library_size, "retained": retained, "reason": reason})
            if not retained:
                continue
            if library_size < request.minimum_library_size_warning:
                warnings.append(StructuredWarning("LOW_LIBRARY_SIZE", "Retained pseudobulk library is below warning threshold",
                                                  context={"cell_state": state, "sample": str(sample), "library_size": library_size,
                                                           "threshold": request.minimum_library_size_warning}))
            row = subset.iloc[0]
            kept_samples.append(str(sample))
            vectors.append(vector.to_numpy(dtype=np.int64))
            entry = {request.sample_column: sample, request.dataset_column: row[request.dataset_column],
                     request.stage_column: row[request.stage_column], "cell_state": state,
                     "n_cells": n_cells, "library_size": library_size}
            if request.patient_column:
                entry[request.patient_column] = row[request.patient_column]
            state_meta.append(entry)
        if not vectors:
            warnings.append(StructuredWarning("NO_USABLE_SAMPLES", "No samples pass the source minimum-cell filter", WarningSeverity.ERROR, {"cell_state": state}))
            continue
        matrix = pd.DataFrame(np.vstack(vectors).T, index=genes, columns=kept_samples, dtype=np.int64)
        matrix.index.name = "gene"
        all_zero = matrix.index[matrix.sum(axis=1) == 0].astype(str).tolist()
        if all_zero:
            warnings.append(StructuredWarning("ALL_ZERO_GENES", "Genes with zero total count are retained for downstream DESeq2 handling",
                                              context={"cell_state": state, "count": len(all_zero), "examples": all_zero[:10]}))
        matrices[state] = matrix
        state_metadata = pd.DataFrame(state_meta)
        sample_rows.extend(state_meta)
        warnings.extend(_dataset_warnings(state_metadata, request, state))
    if not matrices:
        raise ValueError("empty pseudobulk matrices after minimum-cell filtering")
    return matrices, pd.DataFrame(sample_rows), pd.DataFrame(qc_rows), warnings


def run_pseudobulk_construction(request: PseudobulkInput, context: AnalysisContext) -> PseudobulkOutput:
    for path in (request.raw_counts_path, request.metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"missing raw pseudobulk input: {path}")
    count_signature = _file_signature(request.raw_counts_path)
    metadata_signature = _file_signature(request.metadata_path)
    parameters = request.parameters()
    key = make_cache_key(
        capability_id=CAPABILITY_ID, node_version=NODE_VERSION, cache_schema_version=CACHE_SCHEMA_VERSION,
        dataset_signature=context.dataset_signature,
        parameters={**parameters, "raw_counts_signature": count_signature, "metadata_signature": metadata_signature})
    manifest_path = context.cache_root / CAPABILITY_ID.lower() / key / "manifest.json"
    cached = load_complete_manifest(manifest_path, key)
    if cached:
        paths = [Path(path) for path in cached.output_files]
        named = {path.name: str(path) for path in paths}
        provenance = _load_provenance(Path(named["provenance.json"]))
        count_paths = tuple(str(path) for path in paths if path.name.startswith("counts_") and path.suffix == ".csv")
        return PseudobulkOutput(CAPABILITY_ID, NODE_VERSION, key, True, count_paths,
                                named["sample_metadata.csv"], named["pseudobulk_qc.csv"], named["provenance.json"],
                                str(manifest_path), provenance.warnings, provenance)
    separator = "\t" if request.input_format == "tsv" else ","
    raw_counts = pd.read_csv(request.raw_counts_path, sep=separator, low_memory=False)
    metadata = pd.read_csv(request.metadata_path, sep=separator, low_memory=False)
    matrices, sample_metadata, qc, warnings = construct_raw_pseudobulk(raw_counts, metadata, request)
    output_dir = context.capability_output_dir(CAPABILITY_ID) / key
    count_paths = []
    for state, matrix in matrices.items():
        path = output_dir / f"counts_{_safe_name(state)}.csv"
        _atomic_csv(matrix, path, index=True)
        count_paths.append(path)
    metadata_path = output_dir / "sample_metadata.csv"
    qc_path = output_dir / "pseudobulk_qc.csv"
    provenance_path = output_dir / "provenance.json"
    _atomic_csv(sample_metadata, metadata_path)
    _atomic_csv(qc, qc_path)
    software = {**context.software_versions, "python": platform.python_version(),
                "pandas": pd.__version__, "numpy": np.__version__}
    provenance = build_provenance(
        capability_id=CAPABILITY_ID, node_version=NODE_VERSION, cache_schema_version=CACHE_SCHEMA_VERSION,
        source_files=SOURCE_FILES, source_locations=SOURCE_LOCATIONS,
        input_dataset_signature=context.dataset_signature,
        parameters={**parameters, "raw_counts_signature": count_signature, "metadata_signature": metadata_signature,
                    "cells_input": len(raw_counts), "samples_retained": len(sample_metadata),
                    "aggregation": "sum raw integer counts by biological sample and selected cell state"},
        model_formula=None, reference_group=None, covariates=(), unit_of_inference="biological_sample",
        random_seed=None, software_versions=software,
        output_paths=tuple(str(path) for path in (*count_paths, metadata_path, qc_path)), warnings=warnings)
    write_provenance_atomic(provenance, provenance_path)
    outputs = (*count_paths, metadata_path, qc_path, provenance_path)
    manifest = build_cache_manifest(
        cache_key=key, capability_id=CAPABILITY_ID, node_version=NODE_VERSION,
        cache_schema_version=CACHE_SCHEMA_VERSION, input_signature=count_signature,
        source_dataset_signature=context.dataset_signature, parameters=parameters,
        output_files=tuple(str(path) for path in outputs), completion_status="complete",
        warnings=warnings, software_versions=software)
    write_manifest_atomic(manifest, manifest_path)
    return PseudobulkOutput(CAPABILITY_ID, NODE_VERSION, key, False, tuple(str(path) for path in count_paths),
                            str(metadata_path), str(qc_path), str(provenance_path), str(manifest_path),
                            tuple(warnings), provenance)



# Backward-compatible call sites; CAP-DESEQ-002 implementation lives in nodes.deseq2.
def validate_deseq2_inputs(*args, **kwargs):
    from .deseq2 import validate_deseq2_inputs as implementation
    return implementation(*args, **kwargs)

def run_deseq2_differential_expression(*args, **kwargs):
    from .deseq2 import run_deseq2_differential_expression as implementation
    return implementation(*args, **kwargs)
