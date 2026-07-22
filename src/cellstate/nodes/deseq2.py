"""CAP-DESEQ-002 DESeq2 validation and thin R execution wrapper."""
from __future__ import annotations
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from ..cache import build_cache_manifest, load_complete_manifest, make_cache_key, write_manifest_atomic
from ..context import AnalysisContext
from ..provenance import build_provenance, write_provenance_atomic
from ..schemas.common import StructuredWarning, WarningSeverity
from ..schemas.deseq2 import DifferentialExpressionInput, DifferentialExpressionOutput
from ..utilities.design import ordered_indicator_design
from ..validation.estimability import validate_design_estimability
from ..validation.metadata import validate_required_columns
from .pseudobulk_de import _atomic_csv, _file_signature, _load_provenance, CACHE_SCHEMA_VERSION, NODE_VERSION
from . import pseudobulk_de as _legacy_module

CAPABILITY_ID = "CAP-DESEQ-002"

def validate_deseq2_inputs(counts: pd.DataFrame, metadata: pd.DataFrame, request: DifferentialExpressionInput):
    required = [request.sample_column, request.dataset_column, request.stage_column, request.cell_state_column]
    validate_required_columns(metadata, required)
    subset = metadata.loc[metadata[request.cell_state_column] == request.cell_state].copy()
    if subset.empty: raise ValueError("empty target cell-state metadata")
    if subset[request.sample_column].duplicated().any(): raise ValueError("biological samples must be unique for a cell state")
    if counts.index.duplicated().any() or counts.columns.duplicated().any(): raise ValueError("genes and biological-sample columns must be unique")
    if set(counts.columns.astype(str)) != set(subset[request.sample_column].astype(str)):
        raise ValueError("count matrix and sample metadata are misaligned")
    values = counts.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.all(values == np.floor(values)):
        raise ValueError("DESeq2 counts must be finite nonnegative integers")
    if not set(subset[request.stage_column]).issubset({"NBM", "SMM", "NDMM"}): raise ValueError("invalid stage outside NBM/SMM/NDMM")
    by_group = subset.groupby(request.stage_column)[request.sample_column].nunique()
    insufficient = {g: int(by_group.get(g, 0)) for g in ("NBM", "SMM", "NDMM") if int(by_group.get(g, 0)) < request.minimum_samples_per_group}
    if insufficient: raise ValueError(f"insufficient biological replication: {insufficient}")
    subset = subset.set_index(request.sample_column).loc[counts.columns].reset_index()
    stage_design, stage_terms = ordered_indicator_design(subset[request.stage_column], levels=("NBM", "SMM", "NDMM"), reference_level="NBM")
    dataset_dummies = pd.get_dummies(subset[request.dataset_column], drop_first=True, dtype=float)
    design = np.column_stack([stage_design[:, :1], dataset_dummies.to_numpy(), stage_design[:, 1:]])
    terms = ("Intercept", *[f"dataset[{x}]" for x in dataset_dummies.columns], *stage_terms[1:])
    try:
        estimability = validate_design_estimability(design, term_names=terms)
    except ValueError as error:
        warning = StructuredWarning("NON_ESTIMABLE_ADJUSTED_DESIGN", str(error), WarningSeverity.ERROR,
                                    {"formula": "~ dataset + stage", "cell_state": request.cell_state})
        return subset, counts.iloc[0:0], [warning], None
    presence = pd.crosstab(subset[request.stage_column], subset[request.dataset_column]); warnings = []
    exclusive = [str(x) for x in presence if int((presence[x] > 0).sum()) == 1]
    if exclusive: warnings.append(StructuredWarning("DATASET_GROUP_CONFOUNDING", "Datasets contain only one stage", context={"datasets": exclusive}))
    filtered = counts.loc[(counts >= request.gene_minimum_count).sum(axis=1) >= request.gene_minimum_samples]
    if filtered.empty: raise ValueError("no genes remain after approved count prefilter")
    return subset, filtered, warnings, estimability

def run_deseq2_differential_expression(request: DifferentialExpressionInput, context: AnalysisContext) -> DifferentialExpressionOutput:
    for path in (request.count_matrix_path, request.sample_metadata_path):
        if not path.exists(): raise FileNotFoundError(path)
    count_sig, meta_sig = _file_signature(request.count_matrix_path), _file_signature(request.sample_metadata_path)
    parameters = request.parameters()
    key = make_cache_key(capability_id=CAPABILITY_ID, node_version=NODE_VERSION, cache_schema_version=CACHE_SCHEMA_VERSION,
                         dataset_signature=context.dataset_signature, parameters={**parameters, "count_signature": count_sig, "metadata_signature": meta_sig})
    manifest_path = context.cache_root / CAPABILITY_ID.lower() / key / "manifest.json"
    cached = load_complete_manifest(manifest_path, key)
    if cached:
        paths = [Path(x) for x in cached.output_files]; named = {x.name: str(x) for x in paths}; provenance = _load_provenance(Path(named["provenance.json"]))
        return DifferentialExpressionOutput(CAPABILITY_ID, NODE_VERSION, key, True,
            tuple(str(x) for x in paths if x.name.startswith("unshrunk_")), tuple(str(x) for x in paths if x.name.startswith("apeglm_")),
            named["model_qc.csv"], named["provenance.json"], str(manifest_path), provenance.warnings, provenance)
    counts = pd.read_csv(request.count_matrix_path, index_col=0); metadata = pd.read_csv(request.sample_metadata_path)
    subset, filtered, warnings, estimability = validate_deseq2_inputs(counts, metadata, request)
    output_dir = context.capability_output_dir(CAPABILITY_ID) / key; output_dir.mkdir(parents=True, exist_ok=True)
    qc_path = output_dir / "model_qc.csv"; result_paths = []; shrinkage_paths = []
    if estimability is None:
        _atomic_csv(pd.DataFrame([{"status": "not_fitted", "reason": warnings[-1].message}]), qc_path)
    else:
        worker = Path(__file__).with_name("deseq2_fit.R")
        command = [str(request.rscript_path), str(worker), str(request.count_matrix_path), str(request.sample_metadata_path),
                   str(output_dir), request.sample_column, request.dataset_column, request.stage_column, request.cell_state_column, request.cell_state]
        try: _legacy_module.subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, _legacy_module.subprocess.CalledProcessError) as error:
            detail = getattr(error, "stderr", None) or str(error)
            warnings.append(StructuredWarning("DESEQ2_RUNTIME_FAILURE", "R/DESeq2 worker could not complete", WarningSeverity.ERROR, {"detail": detail[-1000:]}))
            _atomic_csv(pd.DataFrame([{"status": "runtime_failure", "reason": detail[-1000:]}]), qc_path)
        else:
            result_paths = [output_dir / "unshrunk_SMM_vs_NBM.csv", output_dir / "unshrunk_NDMM_vs_NBM.csv"]
            if not all(path.exists() for path in result_paths) or not qc_path.exists():
                warnings.append(StructuredWarning("DESEQ2_OUTPUT_INCOMPLETE", "R worker returned without every required unshrunk result and QC file", WarningSeverity.ERROR)); result_paths = []
                if not qc_path.exists(): _atomic_csv(pd.DataFrame([{"status": "incomplete_output"}]), qc_path)
            shrinkage_paths = sorted(output_dir.glob("apeglm_*_vs_NBM.csv"))
            if request.request_apeglm and len(shrinkage_paths) != 2:
                warnings.append(StructuredWarning("APeglm_UNAVAILABLE", "Unshrunk results retained; apeglm shrinkage was unavailable or coefficient was not identifiable"))
    provenance_path = output_dir / "provenance.json"
    software = {**context.software_versions, "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}
    provenance = build_provenance(capability_id=CAPABILITY_ID, node_version=NODE_VERSION, cache_schema_version=CACHE_SCHEMA_VERSION,
        source_files=("scripts/19B_DESeq2_SMM_validation.R",), source_locations=("approved CAP-DESEQ-002 contract; R worker cellstate/nodes/deseq2_fit.R",),
        input_dataset_signature=context.dataset_signature, parameters={**parameters, "count_signature": count_sig, "metadata_signature": meta_sig,
        "genes_after_prefilter": len(filtered), "design_estimability": estimability}, model_formula="~ dataset + stage", reference_group="NBM",
        covariates=(request.dataset_column,), unit_of_inference="biological_sample", random_seed=None, software_versions=software,
        output_paths=tuple(str(x) for x in (*result_paths, *shrinkage_paths, qc_path)), warnings=warnings)
    write_provenance_atomic(provenance, provenance_path); outputs = (*result_paths, *shrinkage_paths, qc_path, provenance_path)
    manifest = build_cache_manifest(cache_key=key, capability_id=CAPABILITY_ID, node_version=NODE_VERSION, cache_schema_version=CACHE_SCHEMA_VERSION,
        input_signature=count_sig, source_dataset_signature=context.dataset_signature, parameters=parameters, output_files=tuple(str(x) for x in outputs),
        completion_status="complete" if len(result_paths) == 2 else "failed", warnings=warnings, software_versions=software)
    write_manifest_atomic(manifest, manifest_path)
    return DifferentialExpressionOutput(CAPABILITY_ID, NODE_VERSION, key, False, tuple(str(x) for x in result_paths), tuple(str(x) for x in shrinkage_paths),
                                        str(qc_path), str(provenance_path), str(manifest_path), tuple(warnings), provenance)
