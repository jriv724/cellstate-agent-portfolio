"""Application orchestration over existing deterministic capability runners."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import pandas as pd

from cellstate.context import AnalysisContext
from cellstate.evidence import write_evidence_bundle_atomic
from cellstate.nodes.arbitrary_two_group_de import run_arbitrary_two_group_de
from cellstate.nodes.tf_activity import run_tf_activity
from cellstate.reasoning import ReasoningEngine, ReasoningError
from cellstate.schemas.arbitrary_two_group_de import ArbitraryTwoGroupDEInput
from cellstate.schemas.tf_activity import TFActivityInput

from .capability_registry import build_capability_registry
from .evidence_adapter import build_combined_evidence_bundle
from .models import AnalysisPlan, ApplicationRunResult


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_tsv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        table.to_csv(temporary, sep="\t", index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def signed_program_from_de(de: Any, plan: AnalysisPlan, path: Path) -> Path:
    exploratory = getattr(de, "evidence_class", "adjusted_inference") == "exploratory_lodo_conserved"
    logical_name = "conserved_features" if exploratory else "deseq2_results"
    results_path = Path(next(
        artifact.path for artifact in de.artifacts
        if artifact.logical_name == logical_name
    ))
    table = pd.read_csv(results_path, sep="," if exploratory else "\t")
    if exploratory:
        table = table.rename(columns={"full_log2FoldChange": "signed_statistic"})
    required = {"gene", "signed_statistic"}
    if not required.issubset(table.columns):
        raise ValueError("CAP-DESEQ-003 result lacks complete signed statistics.")
    source_hash = _file_hash(results_path)
    program = pd.DataFrame({
        "program_id": [f"{plan.cell_state}:{plan.group_a}-{plan.group_b}"] * len(table),
        "feature_set_id": [
            "exploratory_lodo_conserved_features" if exploratory
            else "complete_deseq2_wald_statistics"
        ] * len(table),
        "feature_id": table["gene"].astype(str),
        "feature_type": ["gene"] * len(table),
        "cell_state": [plan.cell_state] * len(table),
        "condition_a": [plan.group_a] * len(table),
        "condition_b": [plan.group_b] * len(table),
        "contrast": [plan.contrast] * len(table),
        "contrast_direction": ["group_a_minus_group_b"] * len(table),
        "feature_direction": ["signed_statistic"] * len(table),
        "signed_statistic": table["signed_statistic"],
        "signed_statistic_name": [
            "full_unadjusted_DESeq2_log2FoldChange" if exploratory
            else "DESeq2_Wald_statistic"
        ] * len(table),
        "statistic_orientation": ["condition_a_minus_condition_b"] * len(table),
        "source_capability_id": [de.capability_id] * len(table),
        "source_capability_version": [de.capability_version] * len(table),
        "source_cache_key": [de.cache_key] * len(table),
        "source_artifact_path": [str(results_path)] * len(table),
        "source_artifact_hash": [source_hash] * len(table),
        "upstream_analysis_method": [
            "exploratory_unadjusted_DESeq2_dataset_LODO" if exploratory
            else "DESeq2_Wald_test"
        ] * len(table),
        "upstream_analysis_parameters": [
            json.dumps({"design_formula": de.design_assessment.design_formula},
                       sort_keys=True)
        ] * len(table),
        "upstream_provenance": [de.provenance_path] * len(table),
    })
    _atomic_tsv(program, path)
    return path


class CellStateOrchestrator:
    def __init__(
        self,
        *,
        adapter: Any,
        output_root: Path,
        cache_root: Path,
        de_runner: Callable[..., Any] = run_arbitrary_two_group_de,
        tf_runner: Callable[..., Any] = run_tf_activity,
        reasoning_engine: Any | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.adapter = adapter
        self.output_root = Path(output_root).resolve()
        self.cache_root = Path(cache_root).resolve()
        self.de_runner = de_runner
        self.tf_runner = tf_runner
        self.reasoning_engine = reasoning_engine
        self.progress = progress or (lambda _: None)

    def _run_dir(self, plan: AnalysisPlan) -> Path:
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        safe = f"{plan.cell_state}-{plan.group_a}-vs-{plan.group_b}"
        safe = "".join(char if char.isalnum() else "_" for char in safe)
        path = self.output_root / f"{stamp}_{safe}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def run(self, plan: AnalysisPlan) -> ApplicationRunResult:
        started = perf_counter()
        run_dir = self._run_dir(plan)
        result = ApplicationRunResult(plan, run_dir, "running", 0.0)
        try:
            self.progress("Preparing atlas subset")
            self.progress("Building production pseudobulk inputs")
            adapter_key = self.adapter.cache_key(
                plan.cell_state, plan.group_a, plan.group_b
            )
            adapter = self.adapter.build(
                cell_state=plan.cell_state, group_a=plan.group_a,
                group_b=plan.group_b,
                output_dir=self.cache_root / "atlas-adapter" / adapter_key,
            )
            result.adapter = adapter
            self.progress("Validating independent replicates")
            self.progress("Assessing design")
            context = AnalysisContext(
                self.output_root, self.cache_root, adapter.atlas_identity
            )
            self.progress("Running DESeq2")
            de = self.de_runner(ArbitraryTwoGroupDEInput(
                count_matrix_path=adapter.count_matrix_path,
                sample_metadata_path=adapter.sample_metadata_path,
                group_a=plan.group_a,
                group_b=plan.group_b,
                upstream_cache_key=f"atlas-adapter:{adapter.atlas_identity}",
                upstream_provenance_path=adapter.provenance_path,
                confounded_design_policy=plan.confounded_design_policy,
                lodo_min_estimable_folds=plan.lodo_min_estimable_folds,
                lodo_min_direction_fraction=plan.lodo_min_direction_fraction,
                lodo_min_median_abs_log2fc=plan.lodo_min_median_abs_log2fc,
                lodo_full_analysis_fdr=plan.lodo_full_analysis_fdr,
                lodo_require_two_datasets_per_group=plan.lodo_require_two_datasets_per_group,
                lodo_max_opposite_log2fc=plan.lodo_max_opposite_log2fc,
            ), context)
            result.de = de
            self.progress("Validating signed statistics")
            if de.terminal_status not in {"completed", "completed_with_warnings"}:
                result.overall_status = de.terminal_status
                result.tf_status = "blocked_by_dependency"
                result.tf_message = "CAP-TF-002 requires completed CAP-DESEQ-003 evidence."
                bundle = build_combined_evidence_bundle(
                    plan=plan, adapter=adapter, de=de, tf=None,
                    tf_status=result.tf_status, tf_message=result.tf_message,
                    cap_tf_001_status=result.cap_tf_001_status,
                    run_dir=run_dir,
                )
                bundle_path = run_dir / "evidence_bundle.json"
                write_evidence_bundle_atomic(bundle, bundle_path)
                result.evidence_bundle = bundle
                result.evidence_bundle_path = bundle_path
                if plan.reasoning_requested:
                    try:
                        engine = self.reasoning_engine or ReasoningEngine(
                            progress_callback=self.progress
                        )
                        result.reasoning = engine.run(bundle, run_dir)
                    except ReasoningError as exc:
                        result.reasoning_error = str(exc)
                    except Exception as exc:
                        result.reasoning_error = f"{type(exc).__name__}: {exc}"
                return result

            if "CAP-TF-002" in plan.requested_capabilities:
                registry = build_capability_registry()["CAP-TF-002"]
                result.tf_resources = tuple(path.name for path in registry.resources)
                if not registry.executable:
                    result.tf_status = "configuration_required"
                    result.tf_message = registry.detail
                else:
                    signed_path = signed_program_from_de(
                        de, plan, run_dir / "complete_signed_statistics.tsv"
                    )
                    try:
                        tf = self.tf_runner(TFActivityInput(
                            signed_feature_program_path=signed_path,
                            dorothea_path=registry.resources[0],
                            collectri_path=registry.resources[1],
                        ), context)
                        result.tf = tf
                        result.tf_status = tf.status.value
                        result.tf_message = "Signed TF activity completed."
                        qc_path = next(
                            Path(path) for name, path, _, _ in tf.artifact_paths
                            if name == "qc_summary"
                        )
                        result.tf_summary = json.loads(qc_path.read_text())
                    except Exception as exc:
                        result.tf_status = "failed"
                        result.tf_message = f"{type(exc).__name__}: {exc}"

            if "CAP-TF-001" in plan.requested_capabilities:
                result.cap_tf_001_status = (
                    "ready" if plan.feature_program_path and plan.background_path
                    else "blocked_missing_explicit_inputs"
                )

            bundle = build_combined_evidence_bundle(
                plan=plan, adapter=adapter, de=de, tf=result.tf,
                tf_status=result.tf_status, tf_message=result.tf_message,
                cap_tf_001_status=result.cap_tf_001_status, run_dir=run_dir,
            )
            bundle_path = run_dir / "evidence_bundle.json"
            write_evidence_bundle_atomic(bundle, bundle_path)
            result.evidence_bundle = bundle
            result.evidence_bundle_path = bundle_path
            result.overall_status = de.terminal_status
            self.progress("Writing DE artifacts")
            if plan.reasoning_requested:
                try:
                    engine = self.reasoning_engine or ReasoningEngine(
                        progress_callback=self.progress
                    )
                    result.reasoning = engine.run(bundle, run_dir)
                except ReasoningError as exc:
                    result.reasoning_error = str(exc)
                except Exception as exc:
                    result.reasoning_error = f"{type(exc).__name__}: {exc}"
            return result
        finally:
            result.elapsed_seconds = perf_counter() - started
