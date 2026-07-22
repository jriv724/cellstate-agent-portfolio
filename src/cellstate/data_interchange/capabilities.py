"""CAP-DATA-001/002/004 deterministic core.

The module deliberately accepts protocol-like objects instead of importing
scanpy.  AnnData objects satisfy the required attributes, while tests can use
small fixtures without loading the research environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import re
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread, mmwrite


CACHE_VERSIONS = {
    "CAP-DATA-001": "cap-data-001-v1",
    "CAP-DATA-002": "cap-data-002-v1",
    "CAP-DATA-003": "cap-data-003-v1",
    "CAP-DATA-004": "cap-data-004-v1",
}

SOURCE_MAP = {
    "CAP-DATA-001": ["scripts/00_export_metadata_from_anndata.py:top-level export blocks"],
    "CAP-DATA-002": [
        "scripts/08_export_idecell_nbm_to_mtx.R:export_seurat_mtx",
        "scripts/09_IDE-cell_scmatrix.ipynb:cells 5eeddf65,9db5792b",
    ],
    "CAP-DATA-003": ["scripts/01_convert_h5ad_to_seurat_rebuild_metadata.R:top-level reconstruction"],
    "CAP-DATA-004": [
        "scripts/11_build_novartis_trial_dataset.ipynb:cells "
        "bd9f17a1,03a3e689,f32b7784,65d61583,41e96034,e1e73cd4,"
        "35183532,36446913,b1095f04"
    ],
}

DEFAULT_OBS_COLUMNS = (
    "sample", "sample_id", "dataset", "stage", "collection_event",
    "mm_group", "stage3", "macro_cell_type", "preserved",
    "celltype_collapsed", "age_years", "sex", "disease", "cohort",
)

MISSING_TOKENS = ("nan", "None", "<NA>")
NOVARTIS_TIMEPOINTS = ("S", "D28", "M3")
NOVARTIS_CLOCK_LABELS = {
    "B": ("Naive B cell", "Memory B cell", "PreB cell", "ProB cell", "Cycling preB cell"),
    "CD4T": ("Memory CD4 T cell", "Treg cell", "Naive CD4 T cell"),
    "CD8T": ("GZMB CD8 T cell", "GZMK CD8 T cell", "Naive CD8 T cell", "MAIT cell"),
    "MONO": ("MHCII high CD14 monocyte", "MHCII low CD14 monocyte", "CD16 monocyte"),
    "NK": ("CD56 NK cell", "CD16 NK cell", "Cycling T/NK cell"),
}


class AnnDataLike(Protocol):
    X: Any
    obs: pd.DataFrame
    obs_names: Sequence[Any]
    var_names: Sequence[Any]
    layers: Mapping[str, Any]


@dataclass(frozen=True)
class StructuredWarning:
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityResult:
    capability_id: str
    outputs: Mapping[str, Any]
    warnings: list[StructuredWarning]
    provenance: Mapping[str, Any]
    cache_key: str


@dataclass(frozen=True)
class MatrixTriplet:
    matrix_path: Path
    features_path: Path
    barcodes_path: Path
    metadata_path: Path | None
    n_features: int
    n_cells: int


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def make_cache_key(capability_id: str, inputs: Mapping[str, Any]) -> str:
    payload = {
        "capability_id": capability_id,
        "cache_version": CACHE_VERSIONS[capability_id],
        "inputs": _jsonable(inputs),
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _provenance(capability_id: str, parameters: Mapping[str, Any], counts: Mapping[str, int]) -> dict[str, Any]:
    return {
        "capability_id": capability_id,
        "cache_version": CACHE_VERSIONS[capability_id],
        "source_map": SOURCE_MAP[capability_id],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "parameters": _jsonable(parameters),
        "counts": _jsonable(counts),
    }


def _require_unique(values: Sequence[Any], label: str) -> list[str]:
    result = [str(v) for v in values]
    if any(v == "" or v.lower() == "nan" for v in result):
        raise ValueError(f"{label} contains missing/blank identifiers")
    duplicates = pd.Index(result)[pd.Index(result).duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"{label} contains duplicate identifiers: {duplicates[:5]}")
    return result


def _matrix_checksum(matrix: Any) -> str:
    x = sparse.csr_matrix(matrix)
    digest = sha256()
    for array in (x.data, x.indices, x.indptr, np.asarray(x.shape, dtype=np.int64)):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def export_anndata_obs(
    adata: AnnDataLike,
    output_dir: Path,
    columns: Sequence[str] = DEFAULT_OBS_COLUMNS,
    strict: bool = False,
    formats: Sequence[Literal["tsv", "csv"]] = ("tsv", "csv"),
    basename: str = "cell_metadata",
) -> CapabilityResult:
    """CAP-DATA-001: export observation metadata with explicit schema reporting."""
    barcodes = _require_unique(adata.obs_names, "obs_names")
    available = [c for c in columns if c in adata.obs.columns]
    missing = [c for c in columns if c not in adata.obs.columns]
    if strict and missing:
        raise ValueError(f"Missing required observation columns: {missing}")
    md = adata.obs.loc[:, available].copy()
    md.index = barcodes
    md.index.name = "cell_barcode"
    warnings: list[StructuredWarning] = []
    if missing:
        warnings.append(StructuredWarning("OBS_COLUMNS_MISSING", "Requested columns were absent", context={"columns": missing}))
    replacements: dict[str, int] = {}
    for col in md.columns:
        if isinstance(md[col].dtype, pd.CategoricalDtype) or md[col].dtype == object:
            md[col] = md[col].astype("string")
        mask = md[col].astype("string").isin(MISSING_TOKENS)
        replacements[col] = int(mask.sum())
        md.loc[mask, col] = pd.NA
    if "age_years" in md:
        before = int(md["age_years"].notna().sum())
        md["age_years"] = pd.to_numeric(md["age_years"], errors="coerce")
        lost = before - int(md["age_years"].notna().sum())
        if lost:
            warnings.append(StructuredWarning("AGE_COERCED_TO_MISSING", "Non-numeric ages were coerced", context={"count": lost}))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for fmt in formats:
        path = output_dir / f"{basename}.{fmt}"
        md.to_csv(path, sep="\t" if fmt == "tsv" else ",", index=True)
        paths[fmt] = str(path)
    params = {"columns": list(columns), "strict": strict, "formats": list(formats), "basename": basename}
    cache_inputs = {**params, "barcodes": barcodes, "metadata_hash": sha256(pd.util.hash_pandas_object(md, index=True).values.tobytes()).hexdigest()}
    provenance = _provenance("CAP-DATA-001", {**params, "available": available, "missing": missing, "token_replacements": replacements}, {"rows": len(md), "columns": len(md.columns)})
    return CapabilityResult("CAP-DATA-001", {"paths": paths, "metadata": md}, warnings, provenance, make_cache_key("CAP-DATA-001", cache_inputs))


def export_matrix_triplet(
    matrix: Any,
    features: Sequence[Any],
    barcodes: Sequence[Any],
    output_dir: Path,
    prefix: str = "",
    metadata: pd.DataFrame | None = None,
    require_raw_counts: bool = True,
) -> CapabilityResult:
    """CAP-DATA-002: serialize a feature-by-cell sparse matrix."""
    x = sparse.csr_matrix(matrix)
    feature_ids = _require_unique(features, "features")
    barcode_ids = _require_unique(barcodes, "barcodes")
    if x.shape != (len(feature_ids), len(barcode_ids)):
        raise ValueError(f"Matrix shape {x.shape} != identifiers {(len(feature_ids), len(barcode_ids))}")
    if not np.all(np.isfinite(x.data)):
        raise ValueError("Matrix contains non-finite values")
    if np.any(x.data < 0):
        raise ValueError("Matrix contains negative values")
    integer_like = bool(np.allclose(x.data, np.round(x.data)))
    if require_raw_counts and not integer_like:
        raise ValueError("Raw-count export requires integer-like values")
    warnings: list[StructuredWarning] = []
    if not integer_like:
        warnings.append(StructuredWarning("NONINTEGER_MATRIX", "Exported matrix is not raw integer counts"))
    aligned_metadata = None
    if metadata is not None:
        if not metadata.index.is_unique:
            raise ValueError("Metadata index is not unique")
        missing = [b for b in barcode_ids if b not in metadata.index.astype(str)]
        if missing:
            raise ValueError(f"Metadata missing barcodes: {missing[:5]}")
        lookup = metadata.copy()
        lookup.index = lookup.index.astype(str)
        aligned_metadata = lookup.loc[barcode_ids].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""
    matrix_path = output_dir / f"{stem}counts.mtx"
    features_path = output_dir / f"{stem}genes.tsv"
    barcodes_path = output_dir / f"{stem}barcodes.tsv"
    metadata_path = output_dir / f"{stem}metadata.tsv" if aligned_metadata is not None else None
    mmwrite(matrix_path, x.tocoo())
    pd.Series(feature_ids).to_csv(features_path, sep="\t", index=False, header=False)
    pd.Series(barcode_ids).to_csv(barcodes_path, sep="\t", index=False, header=False)
    if metadata_path:
        aligned_metadata.to_csv(metadata_path, sep="\t", index=True)
    validate_matrix_triplet(matrix_path, features_path, barcodes_path, metadata_path, require_raw_counts)
    triplet = MatrixTriplet(matrix_path, features_path, barcodes_path, metadata_path, x.shape[0], x.shape[1])
    params = {"prefix": prefix, "require_raw_counts": require_raw_counts, "include_metadata": metadata is not None}
    cache_inputs = {**params, "matrix_checksum": _matrix_checksum(x), "features": feature_ids, "barcodes": barcode_ids}
    provenance = _provenance("CAP-DATA-002", {**params, "orientation": "features_by_cells", "integer_like": integer_like}, {"features": x.shape[0], "cells": x.shape[1], "nonzero": x.nnz})
    return CapabilityResult("CAP-DATA-002", {"triplet": triplet}, warnings, provenance, make_cache_key("CAP-DATA-002", cache_inputs))


def validate_matrix_triplet(
    matrix_path: Path,
    features_path: Path,
    barcodes_path: Path,
    metadata_path: Path | None = None,
    require_raw_counts: bool = True,
) -> MatrixTriplet:
    matrix = mmread(matrix_path).tocsr()
    features = pd.read_csv(features_path, sep="\t", header=None, dtype=str)[0].tolist()
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None, dtype=str)[0].tolist()
    _require_unique(features, "features")
    _require_unique(barcodes, "barcodes")
    if matrix.shape != (len(features), len(barcodes)):
        raise ValueError("Triplet dimensions do not match feature/barcode files")
    if np.any(matrix.data < 0) or not np.all(np.isfinite(matrix.data)):
        raise ValueError("Matrix values must be finite and nonnegative")
    if require_raw_counts and not np.allclose(matrix.data, np.round(matrix.data)):
        raise ValueError("Matrix is not integer-like raw counts")
    if metadata_path:
        metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)
        if metadata.index.astype(str).tolist() != barcodes:
            raise ValueError("Metadata index order does not exactly match barcodes")
    return MatrixTriplet(Path(matrix_path), Path(features_path), Path(barcodes_path), Path(metadata_path) if metadata_path else None, matrix.shape[0], matrix.shape[1])


def clean_sample(value: Any) -> str:
    return re.sub(r"\s*\(.*?\)", "", str(value))


def patient_from_sample(value: Any) -> str | None:
    match = re.match(r"(N\d+)", clean_sample(value))
    return match.group(1) if match else None


def build_novartis_cell_metadata(barcodes: Sequence[Any], sample_record: Mapping[str, Any]) -> pd.DataFrame:
    """Reproduce notebook cell f32b7784 metadata and barcode construction."""
    required = ("s_id", "sample", "timepoint", "source_map")
    missing = [c for c in required if c not in sample_record]
    if missing:
        raise ValueError(f"Sample record missing: {missing}")
    if sample_record["timepoint"] not in NOVARTIS_TIMEPOINTS:
        raise ValueError(f"Unexpected timepoint: {sample_record['timepoint']}")
    sample_clean = clean_sample(sample_record["sample"])
    patient = sample_record.get("patient_id") or patient_from_sample(sample_clean)
    if patient is None:
        raise ValueError("Cannot derive patient_id from sample")
    raw = _require_unique(barcodes, "raw barcodes")
    rebuilt = [f"{b}--{sample_clean}" for b in raw]
    return pd.DataFrame({
        "s_id": sample_record["s_id"], "sample": sample_record["sample"],
        "sample_clean": sample_clean, "patient_id": patient,
        "timepoint": sample_record["timepoint"], "source_map": sample_record["source_map"],
        "dataset": "Novartis_CART_raw", "barcode_orig_rebuilt": raw,
    }, index=pd.Index(rebuilt, name="cell_barcode"))


def _normalise_label(series: pd.Series) -> pd.Series:
    result = series.astype(object).astype(str)
    return result.mask(result.isin(("nan", "NaN", "None", "<NA>")), "")


def merge_label_donors(
    raw_obs: pd.DataFrame,
    processed_obs: pd.DataFrame,
    atlas_obs: pd.DataFrame,
) -> CapabilityResult:
    """Reproduce notebook cell 35183532 donor precedence and rescue."""
    raw_required = {"sample_clean"}
    proc_required = {"sample", "predicted_cell_type"}
    atlas_required = {"barcode_orig", "sample", "S_id", "predicted_cell_type"}
    for label, frame, required in (("raw", raw_obs, raw_required), ("processed", processed_obs, proc_required), ("atlas", atlas_obs, atlas_required)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} metadata missing columns: {missing}")
    out = raw_obs.copy()
    raw_core = out.index.astype(str).str.replace(r"--.*$", "", regex=True)
    raw_key = raw_core + "__" + out["sample_clean"].astype(str)
    proc = processed_obs.copy()
    proc_core = proc.index.astype(str).str.replace(r"-\d+-(Sampling|day28|Month3)_CART$", "", regex=True)
    proc_sample = proc["sample"].astype(str).str.replace(r"\s*\(.*?\)", "", regex=True)
    proc_key = proc_core + "__" + proc_sample
    if proc_key.duplicated().any():
        raise ValueError("Processed donor has duplicate normalized join keys")
    proc.index = proc_key
    out["proc_predicted_cell_type"] = raw_key.map(proc["predicted_cell_type"])
    atlas = atlas_obs.copy()
    atlas_sample = atlas["sample"].astype(str).str.replace(r"\s*\(.*?\)", "", regex=True)
    atlas_key = atlas["barcode_orig"].astype(str) + "__" + atlas_sample
    duplicate_atlas = int(atlas_key.duplicated().sum())
    atlas.index = atlas_key
    atlas = atlas.loc[~atlas.index.duplicated(keep="first")]
    out["atlas_predicted_cell_type"] = raw_key.map(atlas["predicted_cell_type"].astype(str))
    final = _normalise_label(out["proc_predicted_cell_type"])
    rescue = _normalise_label(out["atlas_predicted_cell_type"])
    fill = final.eq("") & rescue.ne("")
    final.loc[fill] = rescue.loc[fill]
    out["final_predicted_cell_type"] = final
    out["clock_celltype"] = pd.NA
    for clock, labels in NOVARTIS_CLOCK_LABELS.items():
        out.loc[final.isin(labels), "clock_celltype"] = clock
    warnings: list[StructuredWarning] = []
    if duplicate_atlas:
        warnings.append(StructuredWarning("ATLAS_DUPLICATE_JOIN_KEYS", "Atlas duplicate join keys kept first", context={"count": duplicate_atlas}))
    unmatched = int(final.eq("").sum())
    if unmatched:
        warnings.append(StructuredWarning("UNLABELED_CELLS", "Cells remain unlabeled after atlas rescue", context={"count": unmatched}))
    params = {"processed_precedence": True, "atlas_duplicate_policy": "keep_first", "missing_tokens": ["nan", "NaN", "None", "<NA>"]}
    hashes = {name: sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest() for name, frame in (("raw", raw_obs), ("processed", processed_obs), ("atlas", atlas_obs))}
    provenance = _provenance("CAP-DATA-004", {**params, "rescued": int(fill.sum())}, {"cells": len(out), "unlabeled": unmatched})
    return CapabilityResult("CAP-DATA-004", {"metadata": out}, warnings, provenance, make_cache_key("CAP-DATA-004", {**hashes, **params}))


def qc_filter_counts(
    counts_cells_by_genes: Any,
    obs: pd.DataFrame,
    min_counts: int = 300,
    min_genes: int = 100,
    max_counts: int = 100_000,
) -> CapabilityResult:
    """Reproduce notebook cell 36446913 QC thresholds."""
    x = sparse.csr_matrix(counts_cells_by_genes)
    if x.shape[0] != len(obs):
        raise ValueError("Count rows must equal observation rows")
    if np.any(x.data < 0) or not np.allclose(x.data, np.round(x.data)):
        raise ValueError("Novartis input must be nonnegative integer-like raw counts")
    n_counts = np.asarray(x.sum(axis=1)).ravel()
    n_genes = np.asarray((x > 0).sum(axis=1)).ravel()
    keep = (n_counts >= min_counts) & (n_genes >= min_genes) & (n_counts <= max_counts)
    filtered_obs = obs.loc[keep].copy()
    filtered_obs["n_counts_rebuilt"] = n_counts[keep]
    filtered_obs["n_genes_rebuilt"] = n_genes[keep]
    params = {"min_counts": min_counts, "min_genes": min_genes, "max_counts": max_counts}
    warnings = [StructuredWarning("QC_CELLS_REMOVED", "Cells failed source QC thresholds", severity="info", context={"count": int((~keep).sum())})] if (~keep).any() else []
    provenance = _provenance("CAP-DATA-004", params, {"cells_before": len(obs), "cells_after": int(keep.sum())})
    cache_inputs = {**params, "matrix_checksum": _matrix_checksum(x), "obs_index": obs.index.astype(str).tolist()}
    return CapabilityResult("CAP-DATA-004", {"counts": x[keep], "metadata": filtered_obs, "keep_mask": keep}, warnings, provenance, make_cache_key("CAP-DATA-004", cache_inputs))
