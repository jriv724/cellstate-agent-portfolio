"""Production terminal entry point for CellState Agent."""

from __future__ import annotations

from hashlib import sha256
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import anndata as ad

from cellstate.adapters import AtlasPseudobulkAdapter
from cellstate.app.capability_registry import build_capability_registry
from cellstate.app.models import AnalysisPlan, AtlasSummary
from cellstate.app.labels import disease_group_label
from cellstate.app.orchestrator import CellStateOrchestrator
from cellstate.app.planner import SemanticPlanner
from cellstate.app.terminal_ui import TerminalUI
from cellstate.reasoning import ReasoningEngine, ReasoningError


def configured_atlas_path() -> Path:
    """Return the explicitly configured atlas without embedding private paths."""
    value = os.getenv("CELLSTATE_ATLAS_PATH", "").strip()
    if not value:
        raise RuntimeError(
            "CELLSTATE_ATLAS_PATH is required; set it to a readable atlas .h5ad file."
        )
    path = Path(value).expanduser()
    if not path.is_file() or not os.access(path, os.R_OK):
        raise RuntimeError("CELLSTATE_ATLAS_PATH does not identify a readable file.")
    return path.resolve()


OUTPUT_ROOT = Path(os.getenv("CELLSTATE_OUTPUT_DIR", "./outputs")).resolve()
CACHE_ROOT = Path(os.getenv("CELLSTATE_CACHE_DIR", "./analysis_cache")).resolve()
COUNT_SOURCE = os.getenv("CELLSTATE_COUNT_SOURCE", "auto")
CELL_TYPE_COLUMN = "preserved"
GROUP_COLUMN = "stage_model_v2"
SAMPLE_COLUMN = "sample"
DATASET_COLUMN = "dataset"
MIN_CELLS_PER_SAMPLE = 100
MIN_SAMPLES_PER_GROUP = 3
SEMANTIC_CACHE_PATH = Path(os.getenv(
    "CELLSTATE_SEMANTIC_CACHE", "./semantic_cache.json"
))
GEMINI_MODEL = os.getenv("CELLSTATE_GEMINI_MODEL", "gemini-3.5-flash")
CONFOUNDED_DESIGN_POLICY = os.getenv("CELLSTATE_CONFOUNDED_DESIGN_POLICY", "block")
LODO_MIN_ESTIMABLE_FOLDS = int(os.getenv("CELLSTATE_LODO_MIN_ESTIMABLE_FOLDS", "3"))
LODO_MIN_DIRECTION_FRACTION = float(os.getenv("CELLSTATE_LODO_MIN_DIRECTION_FRACTION", "0.80"))
LODO_MIN_MEDIAN_ABS_LOG2FC = float(os.getenv("CELLSTATE_LODO_MIN_MEDIAN_ABS_LOG2FC", "0.25"))
LODO_FULL_ANALYSIS_FDR = float(os.getenv("CELLSTATE_LODO_FULL_ANALYSIS_FDR", "0.05"))
LODO_REQUIRE_TWO_DATASETS_PER_GROUP = os.getenv(
    "CELLSTATE_LODO_REQUIRE_TWO_DATASETS_PER_GROUP", "true"
).casefold() in {"1", "true", "yes"}
LODO_MAX_OPPOSITE_LOG2FC = float(os.getenv("CELLSTATE_LODO_MAX_OPPOSITE_LOG2FC", "0.0"))


def _atlas_identity(path: Path) -> str:
    stat = path.stat()
    return sha256(
        f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()


def load_atlas_summary(
    path: Path | None = None,
    *,
    reader: Any = ad.read_h5ad,
) -> AtlasSummary:
    path = configured_atlas_path() if path is None else path
    atlas = reader(path, backed="r")
    try:
        required = {CELL_TYPE_COLUMN, GROUP_COLUMN, SAMPLE_COLUMN, DATASET_COLUMN}
        missing = required - set(atlas.obs.columns)
        if missing:
            raise ValueError("Atlas metadata is missing: " + ", ".join(sorted(missing)))
        obs = atlas.obs
        states = tuple(sorted(obs[CELL_TYPE_COLUMN].dropna().astype(str).unique()))
        groups = tuple(sorted(obs[GROUP_COLUMN].dropna().astype(str).unique()))
        selected = obs[[
            CELL_TYPE_COLUMN, GROUP_COLUMN, SAMPLE_COLUMN
        ]].dropna().copy()
        counts = selected.groupby(
            [CELL_TYPE_COLUMN, GROUP_COLUMN, SAMPLE_COLUMN], observed=True
        ).size().rename("n_cells").reset_index()
        eligible = counts.loc[counts.n_cells.ge(MIN_CELLS_PER_SAMPLE)]
        sample_counts = {
            (str(state), str(group)): int(frame[SAMPLE_COLUMN].nunique())
            for (state, group), frame in eligible.groupby(
                [CELL_TYPE_COLUMN, GROUP_COLUMN], observed=True
            )
        }
        cell_counts = {
            (str(state), str(group)): int(value)
            for (state, group), value in selected.groupby(
                [CELL_TYPE_COLUMN, GROUP_COLUMN], observed=True
            ).size().items()
        }
        return AtlasSummary(
            path=path,
            identity=_atlas_identity(path),
            n_cells=int(atlas.n_obs),
            n_genes=int(atlas.n_vars),
            metadata_columns=tuple(str(column) for column in obs.columns),
            groups=groups,
            cell_states=states,
            sample_counts=sample_counts,
            cell_counts=cell_counts,
        )
    finally:
        file_handle = getattr(atlas, "file", None)
        if file_handle is not None:
            file_handle.close()


def runtime_status(atlas: AtlasSummary) -> dict[str, str]:
    rscript = Path("/usr/local/bin/Rscript")
    r_status = "ready" if rscript.is_file() else "unavailable"
    deseq2 = "unavailable"
    if rscript.is_file():
        probe = subprocess.run(
            [str(rscript), "-e",
             "quit(status=if(requireNamespace('DESeq2',quietly=TRUE)) 0 else 2)"],
            capture_output=True, text=True,
        )
        deseq2 = "ready" if probe.returncode == 0 else "unavailable"
    tf = build_capability_registry()["CAP-TF-002"]
    return {
        "Atlas": "ready" if atlas.path.is_file() else "unavailable",
        "Python": sys.version.split()[0],
        "R": r_status,
        "DESeq2": deseq2,
        "Gemini": "configured" if os.getenv("GEMINI_API_KEY") else "not configured",
        "OpenAI": "configured" if os.getenv("OPENAI_API_KEY") else "not configured",
        "TF resources": tf.status,
    }


def parse_with_gemini(question: str, atlas: AtlasSummary) -> AnalysisPlan:
    """Resolve only ambiguous intake; canonical requests remain local."""
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError(
            "The request is ambiguous and Gemini is not configured; please "
            "name one atlas cell state and two ordered groups."
        )
    from google import genai

    client = genai.Client()
    prompt = (
        "Return only JSON with keys cell_state, group_a, group_b. Preserve the "
        "user's group order exactly and choose only from the supplied values. "
        f"Cell states: {json.dumps(atlas.cell_states)}. "
        f"Canonical groups: {json.dumps(atlas.groups)}. "
        f"Display labels: {json.dumps({group: disease_group_label(group) for group in atlas.groups})}. "
        f"Request: {json.dumps(question)}"
    )
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = (response.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    payload = json.loads(text)
    cell_state = str(payload["cell_state"])
    group_a = str(payload["group_a"])
    group_b = str(payload["group_b"])
    if cell_state not in atlas.cell_states or group_a not in atlas.groups \
            or group_b not in atlas.groups or group_a == group_b:
        raise ValueError("Gemini intake returned values outside the atlas contract.")
    return AnalysisPlan(
        question, cell_state, group_a, group_b,
        SemanticPlanner.requested_capabilities(question), True,
        assumptions=("Ambiguous request resolved by configured Gemini intake.",),
    )


def run_reasoning_cli(bundle: Any, run_dir: Path) -> Path | None:
    """Run downstream reasoning without changing deterministic run status."""
    engine = ReasoningEngine()
    try:
        result = engine.run(bundle, run_dir)
    except ReasoningError as exc:
        print(
            "Deterministic execution completed, but OpenAI reasoning was unavailable: "
            f"{exc}"
        )
        print(f"Evidence bundle preserved at: {run_dir / 'evidence_bundle.json'}")
        return None
    return result.scientific_report_path


def execute_analysis(
    plan: AnalysisPlan,
    orchestrator: CellStateOrchestrator,
):
    """Execute an approved plan through connected production capabilities."""
    return orchestrator.run(plan)


def display_result(ui: TerminalUI, result: Any) -> None:
    ui.final(result)
    if result.overall_status == "blocked":
        reasons = "; ".join(result.de.blocking_reasons) if result.de else "Design blocked."
        ui.error("scientifically blocked", reasons, result.run_dir)
    elif result.overall_status == "failed":
        ui.error("runtime", "Deterministic capability execution failed.", result.run_dir)
    if result.tf_status == "configuration_required":
        ui.error("missing configuration", result.tf_message, result.run_dir)
    elif result.tf_status == "failed":
        ui.error("runtime", result.tf_message, result.run_dir)
    if result.reasoning_error:
        ui.error("reasoning", result.reasoning_error, result.run_dir)


def _help(ui: TerminalUI) -> None:
    ui.print(
        "Commands: help, available, options, options --details, capability <ID>, atlas, status, last, run, "
        "do it, continue, quit\n"
        "Enter a scientific comparison to create a plan. Execution always "
        "requires explicit approval."
    )


def main() -> None:
    ui = TerminalUI(force_plain=not sys.stdout.isatty())
    try:
        atlas = load_atlas_summary()
    except Exception as exc:
        ui.error("runtime", f"Atlas startup failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
    ui.startup(atlas)
    _help(ui)
    planner = SemanticPlanner(
        atlas, gemini_parser=parse_with_gemini,
        semantic_cache_path=SEMANTIC_CACHE_PATH,
    )
    adapter = AtlasPseudobulkAdapter(
        atlas.path,
        count_source=COUNT_SOURCE,
        cell_state_column=CELL_TYPE_COLUMN,
        group_column=GROUP_COLUMN,
        sample_column=SAMPLE_COLUMN,
        dataset_column=DATASET_COLUMN,
    )
    orchestrator = CellStateOrchestrator(
        adapter=adapter,
        output_root=OUTPUT_ROOT,
        cache_root=CACHE_ROOT,
        progress=ui.progress,
    )
    pending: AnalysisPlan | None = None
    last_result = None
    continuation = {"run", "do it", "continue"}
    while True:
        try:
            question = input("CellState Agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            ui.print("\nGoodbye.")
            return
        command = question.casefold()
        if command in {"quit", "exit"}:
            ui.print("Goodbye.")
            return
        if command == "help":
            _help(ui)
            continue
        if command == "available":
            ui.capabilities(connected_only=True)
            continue
        if command == "options":
            ui.capabilities()
            continue
        if command == "options --details":
            ui.capabilities(details=True)
            continue
        if command.startswith("capability "):
            ui.capability(question.split(maxsplit=1)[1])
            continue
        if command == "atlas":
            ui.atlas(atlas)
            continue
        if command == "status":
            ui.status(runtime_status(atlas))
            continue
        if command == "last":
            if last_result is None:
                ui.print("No completed or attempted run is available.")
            else:
                ui.final(last_result)
            continue
        if command in continuation:
            if pending is None:
                ui.print("There is no pending plan.")
                continue
            try:
                last_result = execute_analysis(pending, orchestrator)
                display_result(ui, last_result)
            except Exception as exc:
                ui.error("runtime", f"{type(exc).__name__}: {exc}")
            pending = None
            continue
        if not question:
            continue
        try:
            plan = planner.parse(question)
            plan = replace(
                plan,
                confounded_design_policy=CONFOUNDED_DESIGN_POLICY,
                lodo_min_estimable_folds=LODO_MIN_ESTIMABLE_FOLDS,
                lodo_min_direction_fraction=LODO_MIN_DIRECTION_FRACTION,
                lodo_min_median_abs_log2fc=LODO_MIN_MEDIAN_ABS_LOG2FC,
                lodo_full_analysis_fdr=LODO_FULL_ANALYSIS_FDR,
                lodo_require_two_datasets_per_group=LODO_REQUIRE_TWO_DATASETS_PER_GROUP,
                lodo_max_opposite_log2fc=LODO_MAX_OPPOSITE_LOG2FC,
            )
            ui.plan(plan, atlas)
            approval = input("Approve this plan? [y/n/later]: ").strip().casefold()
            if approval in {"y", "yes"}:
                last_result = execute_analysis(plan, orchestrator)
                display_result(ui, last_result)
            elif approval in {"later", "l"}:
                pending = plan
                ui.print("Plan saved. Type 'run', 'do it', or 'continue' to approve.")
            else:
                ui.print("Analysis cancelled.")
        except ValueError as exc:
            ui.error("validation", str(exc))
        except Exception as exc:
            ui.error("runtime", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
