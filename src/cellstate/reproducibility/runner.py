"""Focused runner for committed representative regression cases."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
from ..context import AnalysisContext
from ..nodes.abundance import run_cell_state_abundance
from ..nodes.deseq2 import run_deseq2_differential_expression
from ..nodes.progression import run_paired_progression
from ..schemas.abundance import AbundanceInput
from ..schemas.deseq2 import DifferentialExpressionInput
from ..schemas.progression import PairedProgressionInput
from .comparison import compare_artifact
from .schemas import ExpectedArtifact, ReproducibilityCase, ReproducibilityResult

DEFAULT_FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "reproducibility"

def load_cases(fixture_root: Path = DEFAULT_FIXTURE_ROOT) -> tuple[ReproducibilityCase, ...]:
    raw = json.loads((fixture_root / "metadata" / "cases.json").read_text(encoding="utf-8"))
    cases = []
    for item in raw["cases"]:
        artifacts = tuple(ExpectedArtifact(
            value["logical_name"], fixture_root / value["expected_path"], value["actual_filename"],
            value.get("comparison", "csv"), value.get("absolute_tolerance", 1e-12),
            value.get("relative_tolerance", 1e-12)) for value in item.get("expected_artifacts", ()))
        cases.append(ReproducibilityCase(item["case_id"], item["capability_id"],
            tuple(fixture_root / value for value in item["input_paths"]), artifacts,
            item["dataset_signature"], item["expected_status"], tuple(item.get("expected_warning_codes", ())),
            item.get("expected_error_substring")))
    return tuple(cases)

def _execute(case: ReproducibilityCase, context: AnalysisContext):
    if case.capability_id == "CAP-COMP-001":
        request = AbundanceInput(case.input_paths[0], patient_column="patient", stage_order=("NBM", "SMM"),
                                 minimum_samples_per_stage_for_warning=2)
        return run_cell_state_abundance(request, context)
    if case.capability_id == "CAP-STAT-003":
        request = PairedProgressionInput(case.input_paths[0], patient_column="patient", class_column="class",
            stage_column="stage", outcome_column="age", dataset_column="dataset")
        return run_paired_progression(request, context)
    if case.capability_id == "CAP-DESEQ-002":
        request = DifferentialExpressionInput(case.input_paths[0], case.input_paths[1], "B")
        return run_deseq2_differential_expression(request, context)
    raise ValueError(f"capability is outside this harness phase: {case.capability_id}")

def run_case(case: ReproducibilityCase, work_root: Path) -> ReproducibilityResult:
    work_root = work_root.resolve()
    context = AnalysisContext(work_root / "outputs", work_root / "cache", case.dataset_signature)
    output = _execute(case, context)
    envelope = output.to_capability_result()
    manifest = json.loads(Path(output.cache_manifest_path).read_text(encoding="utf-8"))
    actual_by_name = {Path(artifact.path).name: Path(artifact.path) for artifact in envelope.artifacts}
    comparisons = tuple(compare_artifact(actual_by_name[item.actual_filename], item)
                        for item in case.expected_artifacts)
    warning_codes = tuple(sorted(warning.code for warning in output.warnings))
    status_ok = envelope.status.value == case.expected_status
    warnings_ok = set(case.expected_warning_codes).issubset(warning_codes)
    checks = list(comparisons)
    from .schemas import ArtifactComparison
    checks.append(ArtifactComparison("execution_status", status_ok,
        f"actual={envelope.status.value}; expected={case.expected_status}"))
    checks.append(ArtifactComparison("warning_codes", warnings_ok,
        f"actual={warning_codes}; expected subset={case.expected_warning_codes}"))
    if case.expected_error_substring:
        details = tuple(str(warning.context.get("detail", "")) for warning in output.warnings)
        detail_ok = any(case.expected_error_substring in detail for detail in details)
        checks.append(ArtifactComparison("external_failure_detail", detail_ok,
            f"expected substring={case.expected_error_substring!r}"))
    if envelope.status.value not in {"completed", "completed_with_warnings"}:
        safe = not output.cache_hit and manifest["completion_status"] != "complete"
        checks.append(ArtifactComparison("terminal_cache_safety", safe,
            f"cache_hit={output.cache_hit}; manifest={manifest['completion_status']}"))
    return ReproducibilityResult(case.case_id, case.capability_id, envelope.status.value,
        output.cache_key, output.cache_hit, tuple(checks), warning_codes, manifest["completion_status"])

def run_harness(work_root: Path, cases: Iterable[ReproducibilityCase] | None = None) -> tuple[ReproducibilityResult, ...]:
    selected = tuple(cases) if cases is not None else load_cases()
    return tuple(run_case(case, work_root / case.case_id) for case in selected)
