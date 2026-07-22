"""Deterministic AtlasLODO sample-mean pseudobulk and vectorized OLS."""
from __future__ import annotations
from hashlib import sha256
import json, os, platform, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse, stats
from ..cache import build_cache_manifest, load_complete_manifest, make_cache_key, write_manifest_atomic
from ..context import AnalysisContext
from ..provenance import build_provenance, write_provenance_atomic
from ..schemas.atlas_lodo import AtlasLODOInput, AtlasLODOOutput
from ..schemas.common import (AnalysisProvenance, ArtifactCategory, CapabilityStatus,
                              StructuredWarning, WarningSeverity)
from ..validation.metadata import validate_required_columns

CAPABILITY_ID = "CAP-LODO-001"
NODE_VERSION = "1.0.0"
CACHE_SCHEMA_VERSION = 1
SOURCE_FILES = ("scripts/15_lodo_precursor_specific_features_9celltypes.ipynb",)
SOURCE_LOCATIONS = ("notebook 15 cells 3-11: backed-safe pseudobulk, eligibility, vectorized OLS, LODO summaries, labels, atlas outputs",)
MODEL_FORMULA = "log2(mean expression + 1) ~ 1 + I(group_a) + I(group_b)"
EXCLUDE_LABEL_PATTERNS = (r"^IGH",r"^IGK",r"^IGL",r"^MT-",r"^RPL",r"^RPS",r"^MIR",r"^LINC",
                          r"^AL\d",r"^AC\d",r"^AP\d",r"^SNHG",r"^RN7",r"^RNU",
                          r"^TRAV",r"^TRBV",r"^TRGV",r"^TRDV")
TABLES = (
 ("cell_state_eligibility","cell_state_eligibility.csv",ArtifactCategory.DESCRIPTIVE),
 ("pseudobulk_metadata","pseudobulk_sample_metadata.csv",ArtifactCategory.INPUT_DERIVED),
 ("fold_results","fold_level_model_results.csv",ArtifactCategory.INFERENTIAL),
 ("gene_summaries","gene_level_cross_fold_summary.csv",ArtifactCategory.INFERENTIAL),
 ("skipped_folds","skipped_folds.csv",ArtifactCategory.QC),
 ("failed_cell_states","failed_cell_states.csv",ArtifactCategory.QC),
 ("atlas_summary","atlas_cell_state_summary.csv",ArtifactCategory.DESCRIPTIVE),
 ("group_a_specific","group_a_specific_genes.csv",ArtifactCategory.INFERENTIAL),
 ("label_group_a_specific","label_worthy_group_a_specific_genes.csv",ArtifactCategory.INFERENTIAL),
 ("group_b_specific","group_b_specific_genes.csv",ArtifactCategory.INFERENTIAL),
 ("label_group_b_specific","label_worthy_group_b_specific_genes.csv",ArtifactCategory.INFERENTIAL),
 ("group_a_enriched","group_a_enriched_genes.csv",ArtifactCategory.INFERENTIAL),
 ("group_b_enriched","group_b_enriched_genes.csv",ArtifactCategory.INFERENTIAL),
 ("model_qc","model_qc_summary.csv",ArtifactCategory.QC),
)

def _signature(path: Path) -> str:
    digest=sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def _atomic_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try: table.to_csv(temporary,index=False); os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()

def _load_provenance(path: Path) -> AnalysisProvenance:
    raw=json.loads(path.read_text()); warnings=tuple(StructuredWarning(x["code"],x["message"],x["severity"],x.get("context",{})) for x in raw["warnings"])
    return AnalysisProvenance(raw["capability_id"],raw["node_version"],raw["cache_schema_version"],tuple(raw["source_files"]),
        tuple(raw["source_locations"]),raw["input_dataset_signature"],raw["parameters"],raw["model_formula"],raw["reference_group"],
        tuple(raw["covariates"]),raw["unit_of_inference"],raw["random_seed"],raw["software_versions"],tuple(raw["output_paths"]),warnings,raw["execution_timestamp_utc"])

def _bh(pvalues: np.ndarray) -> np.ndarray:
    values=np.asarray(pvalues,float); clean=np.nan_to_num(values,nan=1.0,posinf=1.0,neginf=1.0); n=len(clean)
    order=np.argsort(clean,kind="mergesort"); ranked=clean[order]*n/np.arange(1,n+1); adjusted=np.minimum.accumulate(ranked[::-1])[::-1]
    result=np.empty(n); result[order]=np.clip(adjusted,0,1); return result

def is_label_worthy_gene(gene: str) -> bool:
    value=str(gene).upper(); return not any(re.search(pattern,value) for pattern in EXCLUDE_LABEL_PATTERNS)

def build_sample_means_from_cells(metadata: pd.DataFrame, expression: pd.DataFrame,
                                  genes: tuple[str,...], request: AtlasLODOInput) -> tuple[pd.DataFrame,pd.DataFrame,tuple[str,...]]:
    """Pure flat-cell adapter; row positions are explicit and assigned before filtering."""
    required=[request.cell_id_column,request.sample_column,request.dataset_column,request.group_column,request.cell_state_column]
    validate_required_columns(metadata,required); validate_required_columns(expression,[request.cell_id_column,*genes])
    if metadata[request.cell_id_column].duplicated().any() or expression[request.cell_id_column].duplicated().any(): raise ValueError("cell identifiers must be unique")
    if set(metadata[request.cell_id_column].astype(str))!=set(expression[request.cell_id_column].astype(str)): raise ValueError("cell metadata and expression identifiers are misaligned")
    expr=expression.set_index(expression[request.cell_id_column].astype(str)).loc[metadata[request.cell_id_column].astype(str),list(genes)].reset_index(drop=True)
    values=expr.apply(pd.to_numeric,errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any(): raise ValueError("mean-expression input must be finite and nonnegative")
    obs=metadata.copy(); obs["_original_position"]=np.arange(len(obs)); observed=tuple(sorted(obs[request.cell_state_column].dropna().astype(str).unique()))
    keep=obs[required[1:]].notna().all(axis=1)&obs[request.group_column].isin([request.reference_group,request.group_a,request.group_b])
    obs=obs.loc[keep]; expr=expr.loc[keep]
    meta_rows=[]; means=[]
    for (state,sample),sub in obs.groupby([request.cell_state_column,request.sample_column],observed=True,sort=True):
        if sub[request.dataset_column].nunique()!=1 or sub[request.group_column].nunique()!=1: raise ValueError("dataset and group must be consistent within sample and cell state")
        if len(sub)<request.minimum_cells_per_sample_state: continue
        positions=sub.index.to_numpy(); meta_rows.append({request.sample_column:sample,request.dataset_column:sub[request.dataset_column].iloc[0],
            request.group_column:sub[request.group_column].iloc[0],request.cell_state_column:state,request.n_cells_column:len(sub)})
        means.append(expr.loc[positions,list(genes)].mean(axis=0).to_numpy(float))
    return pd.DataFrame(meta_rows),pd.DataFrame(means,columns=genes),observed

def build_sample_means_backed(adata, request: AtlasLODOInput) -> tuple[pd.DataFrame,pd.DataFrame,tuple[str,...]]:
    """Backed-safe adapter: assign original AnnData positions before any filter."""
    required=[request.sample_column,request.dataset_column,request.group_column,request.cell_state_column]
    validate_required_columns(adata.obs,required); genes=tuple(map(str,adata.var_names)); obs=adata.obs[required].copy(); obs["_original_position"]=np.arange(adata.n_obs)
    observed=tuple(sorted(obs[request.cell_state_column].dropna().astype(str).unique())); keep=obs[required].notna().all(axis=1)&obs[request.group_column].isin([request.reference_group,request.group_a,request.group_b]); obs=obs.loc[keep]
    rows=[]; means=[]
    for (state,sample),sub in obs.groupby([request.cell_state_column,request.sample_column],observed=True,sort=True):
        if sub[request.dataset_column].nunique()!=1 or sub[request.group_column].nunique()!=1: raise ValueError("dataset and group must be consistent within sample and cell state")
        if len(sub)<request.minimum_cells_per_sample_state: continue
        positions=sub["_original_position"].to_numpy(); matrix=adata.X[positions,:]
        mean=np.asarray(matrix.mean(axis=0)).ravel() if sparse.issparse(matrix) else np.asarray(matrix).mean(axis=0)
        rows.append({request.sample_column:sample,request.dataset_column:sub[request.dataset_column].iloc[0],request.group_column:sub[request.group_column].iloc[0],request.cell_state_column:state,request.n_cells_column:len(sub)}); means.append(mean)
    return pd.DataFrame(rows),pd.DataFrame(means,columns=genes),observed

def load_sample_means(request: AtlasLODOInput) -> tuple[pd.DataFrame,pd.DataFrame,tuple[str,...]]:
    if request.input_format=="h5ad":
        try: import anndata as ad
        except ImportError as error: raise RuntimeError("anndata is required for h5ad input") from error
        source=ad.read_h5ad(request.source_path,backed="r")
        try: return build_sample_means_backed(source,request)
        finally:
            if getattr(source,"file",None) is not None: source.file.close()
    if request.input_format=="cell_csv":
        return build_sample_means_from_cells(pd.read_csv(request.source_path),pd.read_csv(request.expression_path),request.gene_columns,request)
    table=pd.read_csv(request.source_path); required=[request.sample_column,request.dataset_column,request.group_column,request.cell_state_column,request.n_cells_column,*request.gene_columns]
    validate_required_columns(table,required); observed=tuple(sorted(table[request.cell_state_column].dropna().astype(str).unique()))
    table=table.loc[table[request.group_column].isin([request.reference_group,request.group_a,request.group_b])].copy()
    if table.duplicated([request.sample_column,request.cell_state_column]).any(): raise ValueError("sample_mean_csv requires unique sample and cell-state rows")
    for _,sub in table.groupby([request.sample_column,request.cell_state_column]):
        if sub[request.dataset_column].nunique()!=1 or sub[request.group_column].nunique()!=1: raise ValueError("inconsistent sample metadata")
    values=table[list(request.gene_columns)].apply(pd.to_numeric,errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any(): raise ValueError("sample means must be finite and nonnegative")
    keep=table[request.n_cells_column]>=request.minimum_cells_per_sample_state
    meta=table.loc[keep,[request.sample_column,request.dataset_column,request.group_column,request.cell_state_column,request.n_cells_column]].reset_index(drop=True)
    expr=table.loc[keep,list(request.gene_columns)].reset_index(drop=True).astype(float)
    return meta,expr,observed

def cell_state_eligibility(meta: pd.DataFrame, observed_states: tuple[str,...], request: AtlasLODOInput) -> pd.DataFrame:
    rows=[]
    for state in observed_states:
        sub=meta.loc[meta[request.cell_state_column].astype(str)==state]; counts=sub[request.group_column].value_counts()
        nr=int(counts.get(request.reference_group,0)); na=int(counts.get(request.group_a,0)); nb=int(counts.get(request.group_b,0)); nd=int(sub[request.dataset_column].nunique())
        reasons=[]
        if nr<request.minimum_reference_samples: reasons.append("low_reference_samples")
        if na<request.minimum_group_a_samples: reasons.append("low_group_a_samples")
        if nb<request.minimum_group_b_samples: reasons.append("low_group_b_samples")
        if nd<request.minimum_datasets: reasons.append("low_dataset_count")
        rows.append({"cell_state":state,"reference_sample_count":nr,"group_a_sample_count":na,"group_b_sample_count":nb,"dataset_count":nd,
            "median_cells_per_retained_sample":float(sub[request.n_cells_column].median()) if len(sub) else np.nan,
            "minimum_cells_per_retained_sample":int(sub[request.n_cells_column].min()) if len(sub) else np.nan,
            "eligible":not reasons,"failure_reasons":";".join(reasons) if reasons else "pass","reference_group":request.reference_group,
            "group_a_label":request.group_a,"group_b_label":request.group_b})
    return pd.DataFrame(rows).sort_values(["eligible","cell_state"],ascending=[False,True]).reset_index(drop=True)

def fit_fold_ols(expr: pd.DataFrame, meta: pd.DataFrame, request: AtlasLODOInput) -> pd.DataFrame:
    groups=meta[request.group_column]; xa=(groups==request.group_a).astype(float).to_numpy(); xb=(groups==request.group_b).astype(float).to_numpy(); X=np.column_stack([np.ones(len(meta)),xa,xb])
    if len(meta)-3<=0 or np.linalg.matrix_rank(X)!=3: raise ValueError("singular_or_invalid_design")
    y=np.log2(expr.to_numpy(float)+request.pseudocount); inv=np.linalg.inv(X.T@X); beta=inv@X.T@y; residual=y-X@beta; df=len(meta)-3; sigma2=np.sum(residual**2,axis=0)/df
    se=np.sqrt(np.outer(np.diag(inv),sigma2)); contrast=np.array([0,1,-1.]); delta=beta[1]-beta[2]; sed=np.sqrt((contrast@inv@contrast)*sigma2)
    def tests(effect,standard):
        t=np.divide(effect,standard,out=np.full_like(effect,np.nan),where=standard!=0); t=np.where((standard==0)&(effect!=0),np.sign(effect)*np.inf,t); p=2*stats.t.sf(np.abs(t),df=df); return p,_bh(p)
    pa,fa=tests(beta[1],se[1]); pb,fb=tests(beta[2],se[2]); pdlt,fdlt=tests(delta,sed)
    return pd.DataFrame({"gene":expr.columns,"effect_group_a_vs_reference":beta[1],"se_group_a_vs_reference":se[1],"p_group_a_vs_reference":pa,"fdr_group_a_vs_reference":fa,
        "effect_group_b_vs_reference":beta[2],"se_group_b_vs_reference":se[2],"p_group_b_vs_reference":pb,"fdr_group_b_vs_reference":fb,
        "delta_group_a_minus_group_b":delta,"se_delta":sed,"p_delta":pdlt,"fdr_delta":fdlt,"mean_expression":expr.mean(axis=0).to_numpy(),
        "reference_sample_count":int((groups==request.reference_group).sum()),"group_a_sample_count":int((groups==request.group_a).sum()),"group_b_sample_count":int((groups==request.group_b).sum())})

def summarize_folds(folds: pd.DataFrame, request: AtlasLODOInput) -> pd.DataFrame:
    if folds.empty: return pd.DataFrame()
    agg=folds.groupby(["cell_state","gene"],sort=True).agg(
        group_a_effect_median=("effect_group_a_vs_reference","median"),group_a_effect_mean=("effect_group_a_vs_reference","mean"),group_a_effect_sd=("effect_group_a_vs_reference","std"),group_a_fdr_max=("fdr_group_a_vs_reference","max"),group_a_fdr_median=("fdr_group_a_vs_reference","median"),
        group_b_effect_median=("effect_group_b_vs_reference","median"),group_b_effect_mean=("effect_group_b_vs_reference","mean"),group_b_effect_sd=("effect_group_b_vs_reference","std"),group_b_fdr_min=("fdr_group_b_vs_reference","min"),group_b_fdr_median=("fdr_group_b_vs_reference","median"),
        delta_median=("delta_group_a_minus_group_b","median"),delta_mean=("delta_group_a_minus_group_b","mean"),delta_sd=("delta_group_a_minus_group_b","std"),delta_fdr_max=("fdr_delta","max"),delta_fdr_median=("fdr_delta","median"),median_mean_expression=("mean_expression","median"),successful_lodo_folds=("held_out_dataset","nunique")).reset_index()
    for prefix,column in (("group_a","effect_group_a_vs_reference"),("group_b","effect_group_b_vs_reference"),("delta","delta_group_a_minus_group_b")):
        signs=folds.assign(_pos=folds[column]>0,_neg=folds[column]<0).groupby(["cell_state","gene"]).agg(**{f"{prefix}_positive_fold_fraction":("_pos","mean"),f"{prefix}_negative_fold_fraction":("_neg","mean")}).reset_index(); agg=agg.merge(signs,on=["cell_state","gene"])
        median_col=f"{prefix}_effect_median" if prefix!="delta" else "delta_median"; agg[f"{prefix}_sign_consistency"]=np.where(agg[median_col]>=0,agg[f"{prefix}_positive_fold_fraction"],agg[f"{prefix}_negative_fold_fraction"])
    folds_ok=agg.successful_lodo_folds>=request.minimum_successful_folds; expr_ok=agg.median_mean_expression>=request.minimum_mean_expression
    agg["group_a_changed"]=folds_ok&expr_ok&(agg.group_a_fdr_median<request.group_a_fdr_cutoff)&(agg.group_a_sign_consistency>=request.group_a_sign_consistency_cutoff)&(agg.group_a_effect_median.abs()>=request.group_a_absolute_effect_cutoff)
    agg["group_b_changed"]=folds_ok&expr_ok&(agg.group_b_fdr_median<request.group_b_fdr_cutoff)&(agg.group_b_sign_consistency>=request.group_b_sign_consistency_cutoff)&(agg.group_b_effect_median.abs()>=request.group_b_absolute_effect_cutoff)
    agg["delta_changed"]=folds_ok&(agg.delta_fdr_median<request.delta_fdr_cutoff)&(agg.delta_sign_consistency>=request.delta_sign_consistency_cutoff)&(agg.delta_median.abs()>=request.delta_absolute_cutoff)
    agg["group_a_specific"]=agg.group_a_changed&agg.delta_changed; agg["group_b_specific"]=agg.group_b_changed&agg.delta_changed
    agg["group_a_specific_class"]=np.where(agg.group_a_specific,np.where(agg.group_a_effect_median>0,"group_a_specific_up","group_a_specific_down"),"not_specific")
    agg["group_b_specific_class"]=np.where(agg.group_b_specific,np.where(agg.group_b_effect_median>0,"group_b_specific_up","group_b_specific_down"),"not_specific")
    common=folds_ok&(agg.delta_fdr_median<request.delta_fdr_cutoff)&(agg.delta_sign_consistency>=request.delta_sign_consistency_cutoff)
    agg["group_a_enriched"]=common&(agg.delta_median>=request.delta_absolute_cutoff); agg["group_b_enriched"]=common&(agg.delta_median<=-request.delta_absolute_cutoff)
    agg["maximum_absolute_group_effect"]=np.maximum(agg.group_a_effect_median.abs(),agg.group_b_effect_median.abs()); agg["label_worthy"]=agg.gene.map(is_label_worthy_gene); agg["excluded_from_labels"]=~agg.label_worthy
    agg["reference_group"]=request.reference_group; agg["group_a_label"]=request.group_a; agg["group_b_label"]=request.group_b
    return agg

def calculate_atlas_lodo(meta: pd.DataFrame, expr: pd.DataFrame, observed_states: tuple[str,...], request: AtlasLODOInput) -> tuple[dict[str,pd.DataFrame],tuple[StructuredWarning,...],CapabilityStatus]:
    if len(meta)!=len(expr): raise ValueError("pseudobulk metadata and expression rows are misaligned")
    eligibility=cell_state_eligibility(meta,observed_states,request); selected=set(request.requested_cell_states or eligibility.cell_state); eligible=eligibility.loc[eligibility.eligible&eligibility.cell_state.isin(selected),"cell_state"].tolist()
    fold_tables=[]; skipped=[]; failed=[]
    for state in eligible:
        mask=meta[request.cell_state_column].astype(str)==state; smeta=meta.loc[mask].reset_index(drop=True); sexpr=expr.loc[mask.to_numpy()].reset_index(drop=True)
        successful=0
        for held_out in sorted(smeta[request.dataset_column].astype(str).unique()):
            train=smeta[request.dataset_column].astype(str)!=held_out; tm=smeta.loc[train].reset_index(drop=True); te=sexpr.loc[train.to_numpy()].reset_index(drop=True); counts=tm[request.group_column].value_counts()
            missing=[name for name,label in (("reference",request.reference_group),("group_a",request.group_a),("group_b",request.group_b)) if int(counts.get(label,0))<request.minimum_training_samples_per_group]
            if missing:
                skipped.append({"cell_state":state,"held_out_dataset":held_out,"reason":"insufficient_training_replication:"+",".join(missing),"reference_training_samples":int(counts.get(request.reference_group,0)),"group_a_training_samples":int(counts.get(request.group_a,0)),"group_b_training_samples":int(counts.get(request.group_b,0))}); continue
            try: result=fit_fold_ols(te,tm,request)
            except (ValueError,np.linalg.LinAlgError) as error:
                skipped.append({"cell_state":state,"held_out_dataset":held_out,"reason":str(error),"reference_training_samples":int(counts.get(request.reference_group,0)),"group_a_training_samples":int(counts.get(request.group_a,0)),"group_b_training_samples":int(counts.get(request.group_b,0))}); continue
            result["cell_state"]=state; result["held_out_dataset"]=held_out; result["reference_group"]=request.reference_group; result["group_a_label"]=request.group_a; result["group_b_label"]=request.group_b; fold_tables.append(result); successful+=1
        if successful==0: failed.append({"cell_state":state,"reason":"no_estimable_folds"})
        elif successful<request.minimum_successful_folds: failed.append({"cell_state":state,"reason":"fewer_than_minimum_successful_folds"})
    folds=pd.concat(fold_tables,ignore_index=True) if fold_tables else pd.DataFrame(); genes=summarize_folds(folds,request)
    skipped_df=pd.DataFrame(skipped,columns=["cell_state","held_out_dataset","reason","reference_training_samples","group_a_training_samples","group_b_training_samples"]); failed_df=pd.DataFrame(failed,columns=["cell_state","reason"])
    summaries=[]
    for state,sub in genes.groupby("cell_state") if not genes.empty else []:
        elig=eligibility.loc[eligibility.cell_state==state].iloc[0]
        summaries.append({"cell_state":state,"genes_tested":sub.gene.nunique(),"group_a_specific_genes":int(sub.group_a_specific.sum()),"group_a_specific_up_genes":int((sub.group_a_specific_class=="group_a_specific_up").sum()),"group_a_specific_down_genes":int((sub.group_a_specific_class=="group_a_specific_down").sum()),"group_b_specific_genes":int(sub.group_b_specific.sum()),"group_b_specific_up_genes":int((sub.group_b_specific_class=="group_b_specific_up").sum()),"group_b_specific_down_genes":int((sub.group_b_specific_class=="group_b_specific_down").sum()),"group_a_enriched_genes":int(sub.group_a_enriched.sum()),"group_b_enriched_genes":int(sub.group_b_enriched.sum()),"label_worthy_group_a_specific_genes":int((sub.group_a_specific&sub.label_worthy).sum()),"label_worthy_group_b_specific_genes":int((sub.group_b_specific&sub.label_worthy).sum()),"median_absolute_delta":float(sub.delta_median.abs().median()),"maximum_absolute_delta":float(sub.delta_median.abs().max()),"median_fold_count":float(sub.successful_lodo_folds.median()),**elig.to_dict()})
    atlas=pd.DataFrame(summaries); warnings=[]
    if skipped: warnings.append(StructuredWarning("ATLAS_LODO_SKIPPED_FOLDS","One or more LODO folds were skipped",context={"count":len(skipped)}))
    if failed: warnings.append(StructuredWarning("ATLAS_LODO_FAILED_CELL_STATES","One or more eligible cell states lacked complete fold output",context={"count":len(failed)}))
    if not eligible:
        warnings.append(StructuredWarning("ATLAS_LODO_INSUFFICIENT_REPLICATION","No cell state met global eligibility",WarningSeverity.ERROR)); status=CapabilityStatus.INSUFFICIENT_REPLICATION
    elif folds.empty:
        warnings.append(StructuredWarning("ATLAS_LODO_NOT_ESTIMABLE","No eligible cell state had an estimable fold",WarningSeverity.ERROR)); status=CapabilityStatus.NOT_ESTIMABLE
    else: status=CapabilityStatus.COMPLETED_WITH_WARNINGS if warnings else CapabilityStatus.COMPLETED
    focused=lambda column: genes.loc[genes[column]].sort_values(["cell_state","delta_median"],ascending=[True,False]).reset_index(drop=True) if not genes.empty else genes.copy()
    tables={"cell_state_eligibility":eligibility,"pseudobulk_metadata":meta,"fold_results":folds,"gene_summaries":genes,"skipped_folds":skipped_df,"failed_cell_states":failed_df,"atlas_summary":atlas,
        "group_a_specific":focused("group_a_specific"),"label_group_a_specific":genes.loc[genes.group_a_specific&genes.label_worthy].copy() if not genes.empty else genes.copy(),"group_b_specific":focused("group_b_specific"),"label_group_b_specific":genes.loc[genes.group_b_specific&genes.label_worthy].copy() if not genes.empty else genes.copy(),"group_a_enriched":focused("group_a_enriched"),"group_b_enriched":focused("group_b_enriched")}
    tables["model_qc"]=pd.DataFrame([{"observed_cell_states":len(observed_states),"eligible_cell_states":len(eligible),"failed_cell_states":len(failed),"successful_folds":int(folds.held_out_dataset.count())//max(len(expr.columns),1) if not folds.empty else 0,"skipped_folds":len(skipped),"genes_tested":int(genes.gene.nunique()) if not genes.empty else 0,"reference_group":request.reference_group,"group_a_label":request.group_a,"group_b_label":request.group_b,"model_formula":MODEL_FORMULA}])
    return tables,tuple(warnings),status

def run_atlas_lodo(request: AtlasLODOInput, context: AnalysisContext) -> AtlasLODOOutput:
    paths=[request.source_path]+([request.expression_path] if request.expression_path else [])
    if any(not path.exists() for path in paths): raise FileNotFoundError("missing AtlasLODO input")
    signatures={path.name:_signature(path) for path in paths}; params=request.parameters(); key=make_cache_key(capability_id=CAPABILITY_ID,node_version=NODE_VERSION,cache_schema_version=CACHE_SCHEMA_VERSION,dataset_signature=context.dataset_signature,parameters={**params,"input_signatures":signatures})
    manifest_path=context.cache_root/CAPABILITY_ID.lower()/key/"manifest.json"; cached=load_complete_manifest(manifest_path,key)
    if cached:
        named={Path(path).name:str(path) for path in cached.output_files}; provenance=_load_provenance(Path(named["provenance.json"])); artifact_paths=tuple((logical,named[filename],category.value) for logical,filename,category in TABLES)
        status=CapabilityStatus.COMPLETED_WITH_WARNINGS if provenance.warnings else CapabilityStatus.COMPLETED
        return AtlasLODOOutput(CAPABILITY_ID,NODE_VERSION,status,key,True,artifact_paths,named["provenance.json"],str(manifest_path),provenance.warnings,provenance)
    meta,expr,observed=load_sample_means(request); tables,warnings,status=calculate_atlas_lodo(meta,expr,observed,request); output_dir=context.capability_output_dir(CAPABILITY_ID)/key; artifacts=[]
    for logical,filename,category in TABLES: path=output_dir/filename; _atomic_csv(tables[logical],path); artifacts.append((logical,str(path),category.value))
    provenance_path=output_dir/"provenance.json"; software={**context.software_versions,"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__}
    provenance=build_provenance(capability_id=CAPABILITY_ID,node_version=NODE_VERSION,cache_schema_version=CACHE_SCHEMA_VERSION,source_files=SOURCE_FILES,source_locations=SOURCE_LOCATIONS,input_dataset_signature=context.dataset_signature,
        parameters={**params,"input_signatures":signatures,"pseudobulk_definition":"sample-level mean expression within sample x cell state","group_a_coefficient":f"{request.group_a} versus {request.reference_group}","group_b_coefficient":f"{request.group_b} versus {request.reference_group}","delta_definition":f"beta_{request.group_a} - beta_{request.group_b}","successful_fold_count":int(tables["model_qc"].iloc[0].successful_folds),"skipped_fold_count":len(tables["skipped_folds"]),"eligible_cell_state_count":int(tables["model_qc"].iloc[0].eligible_cell_states),"failed_cell_state_count":len(tables["failed_cell_states"])},
        model_formula=MODEL_FORMULA,reference_group=request.reference_group,covariates=(),unit_of_inference="biological_sample",random_seed=None,software_versions=software,output_paths=tuple(path for _,path,_ in artifacts),warnings=warnings); write_provenance_atomic(provenance,provenance_path)
    outputs=tuple(path for _,path,_ in artifacts)+(str(provenance_path),); completion="complete" if status in {CapabilityStatus.COMPLETED,CapabilityStatus.COMPLETED_WITH_WARNINGS} else "failed"
    manifest=build_cache_manifest(cache_key=key,capability_id=CAPABILITY_ID,node_version=NODE_VERSION,cache_schema_version=CACHE_SCHEMA_VERSION,input_signature=signatures[request.source_path.name],source_dataset_signature=context.dataset_signature,parameters=params,output_files=outputs,completion_status=completion,warnings=warnings,software_versions=software); write_manifest_atomic(manifest,manifest_path)
    return AtlasLODOOutput(CAPABILITY_ID,NODE_VERSION,status,key,False,tuple(artifacts),str(provenance_path),str(manifest_path),warnings,provenance)
