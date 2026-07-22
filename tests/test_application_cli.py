"""Focused production application tests without a full atlas or API calls."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from rich.console import Console

from cellstate.adapters import AtlasPseudobulkAdapter
from cellstate.app.capability_registry import (
    all_capability_rows,
    build_capability_registry,
)
from cellstate.app.evidence_adapter import build_combined_evidence_bundle
from cellstate.app.models import AnalysisPlan, ApplicationRunResult, AtlasSummary
from cellstate.app.labels import DISEASE_GROUP_LABELS, disease_group_label
from cellstate.app.orchestrator import CellStateOrchestrator, signed_program_from_de
from cellstate.app.planner import SemanticPlanner
from cellstate.app.terminal_ui import TerminalUI
from cellstate.reasoning import ReasoningError
from cellstate.schemas.common import (
    ArtifactCategory,
    ArtifactReference,
    CapabilityResult,
    CapabilityStatus,
)


def atlas_summary() -> AtlasSummary:
    return AtlasSummary(
        Path("/atlas.h5ad"), "atlas-v1", 3041619, 20000,
        ("sample", "dataset", "stage_model_v2", "preserved"),
        ("NBM", "NDMM", "RRMM"),
        ("GZMB CD8 T cell", "Memory B cell"),
        {("GZMB CD8 T cell", "NDMM"): 3,
         ("GZMB CD8 T cell", "RRMM"): 3},
    )


class PlainCollector:
    def __init__(self):
        self.values: list[str] = []

    def print(self, value: str):
        self.values.append(value)


class ApplicationRegistryPlannerTests(unittest.TestCase):
    def test_registry_accounts_for_every_production_node(self):
        rows = all_capability_rows()
        modules = {row.node_module for row in rows}
        self.assertEqual(modules, {
            "pseudobulk_de", "deseq2", "arbitrary_two_group_de",
            "tf_activity", "tf_regulatory_network", "abundance",
            "progression", "age_association", "atlas_lodo",
        })
        by_id = {row.capability_id: row for row in rows}
        self.assertEqual(by_id["CAP-DESEQ-003"].status, "connected")
        self.assertFalse(by_id["CAP-TF-001"].executable)
        self.assertEqual(
            by_id["CAP-TF-001"].status, "requires explicit caller inputs")

    def test_tf_status_reflects_resource_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                build_capability_registry()["CAP-TF-002"].status,
                "connected but configuration required",
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            doro, coll = root / "d.tsv", root / "c.tsv"
            doro.write_text("x\n")
            coll.write_text("x\n")
            with patch.dict(os.environ, {
                "CELLSTATE_DOROTHEA_PATH": str(doro),
                "CELLSTATE_COLLECTRI_PATH": str(coll),
            }, clear=True):
                self.assertEqual(
                    build_capability_registry()["CAP-TF-002"].status,
                    "invalid resource configuration",
                )

    def test_local_semantics_default_and_combined_preserve_direction(self):
        planner = SemanticPlanner(atlas_summary())
        simple = planner.parse(
            "Compare GZMB CD8 T cell between NDMM and RRMM.")
        self.assertEqual(simple.requested_capabilities, ("CAP-DESEQ-003",))
        self.assertEqual((simple.group_a, simple.group_b), ("NDMM", "RRMM"))
        combined = planner.parse(
            "Compare GZMB CD8 T cells between NDMM and RRMM. Run "
            "differential expression and signed TF activity, then critique "
            "and interpret the combined evidence.")
        self.assertEqual(
            combined.requested_capabilities, ("CAP-DESEQ-003", "CAP-TF-002"))
        reversed_plan = planner.parse(
            "Compare GZMB CD8 T cell between RRMM and NDMM.")
        self.assertEqual(
            (reversed_plan.group_a, reversed_plan.group_b), ("RRMM", "NDMM"))

    def test_disease_labels_are_presentation_only(self):
        self.assertEqual(DISEASE_GROUP_LABELS, {
            "NBM": "Normal Bone Marrow (NBM)",
            "MGUS": "Monoclonal Gammopathy of Undetermined Significance (MGUS)",
            "SMM": "Smoldering Multiple Myeloma (SMM)",
            "NDMM": "Newly Diagnosed Multiple Myeloma (NDMM)",
            "RRMM": "Relapsed/Refractory Multiple Myeloma (RRMM)",
            "MM-Remission": "Multiple Myeloma in Remission (MM-Remission)",
        })
        plan = AnalysisPlan(
            "question", "GZMB CD8 T cell", "NBM", "NDMM",
            ("CAP-DESEQ-003",),
        )
        self.assertEqual((plan.group_a, plan.group_b), ("NBM", "NDMM"))
        self.assertEqual(plan.contrast, "NBM − NDMM")
        self.assertEqual(
            plan.display_contrast,
            "Normal Bone Marrow (NBM) − Newly Diagnosed Multiple Myeloma (NDMM)",
        )

    def test_rich_and_plain_ui_render(self):
        rich_console = Console(record=True, width=100)
        TerminalUI(rich_console).startup(atlas_summary())
        self.assertIn("CELLSTATE AGENT", rich_console.export_text())
        collector = PlainCollector()
        TerminalUI(collector, force_plain=True).startup(atlas_summary())
        self.assertIn("Available now: ✓ CAP-DESEQ-003", "\n".join(collector.values))

    def test_compact_capabilities_are_grouped_and_hide_schema_details(self):
        collector = PlainCollector()
        ui = TerminalUI(collector, force_plain=True)
        ui.capabilities()
        rendered = "\n".join(collector.values)
        for heading in (
            "Available now", "Configuration required",
            "Advanced / explicit-input workflows",
            "Backend capabilities not yet exposed",
        ):
            self.assertIn(heading, rendered)
        self.assertIn("✓ CAP-DESEQ-003", rendered)
        self.assertIn("◐ CAP-TF-002", rendered)
        self.assertIn("ⓘ CAP-TF-001", rendered)
        self.assertIn("• CAP-DESEQ-001", rendered)
        self.assertNotIn("Canonical inputs", rendered)
        self.assertNotIn("Remaining work", rendered)
        self.assertNotIn(" | ", rendered)

    def test_detailed_options_and_capability_view_show_canonical_details(self):
        collector = PlainCollector()
        ui = TerminalUI(collector, force_plain=True)
        ui.capabilities(details=True)
        detailed = "\n".join(collector.values)
        self.assertIn("Canonical inputs:", detailed)
        self.assertIn("Remaining work:", detailed)
        self.assertIn("Status: connected", detailed)

        collector.values.clear()
        self.assertTrue(ui.capability("cap-deseq-003"))
        single = "\n".join(collector.values)
        self.assertIn("CAP-DESEQ-003", single)
        self.assertIn("Status: ✓ connected", single)
        self.assertIn("Canonical inputs:", single)
        self.assertNotIn("CAP-TF-002", single)

    def test_rich_compact_rendering_and_atlas_summary(self):
        console = Console(record=True, width=110)
        ui = TerminalUI(console)
        ui.capabilities()
        ui.atlas(atlas_summary())
        rendered = console.export_text()
        self.assertIn("Available now", rendered)
        self.assertIn("Configuration required", rendered)
        self.assertIn("Advanced / explicit-input workflows", rendered)
        self.assertIn("Backend capabilities not yet exposed", rendered)
        self.assertNotIn("Canonical inputs", rendered)
        self.assertIn("3,041,619 cells × 20,000 genes", rendered)
        self.assertIn("4 columns", rendered)
        self.assertNotIn("stage_model_v2", rendered)
        self.assertIn("Normal Bone Marrow (NBM)", rendered)
        self.assertIn("Newly Diagnosed Multiple Myeloma (NDMM)", rendered)

    def test_plan_and_warning_render_expanded_groups_without_mutation(self):
        collector = PlainCollector()
        ui = TerminalUI(collector, force_plain=True)
        plan = AnalysisPlan(
            "question", "GZMB CD8 T cell", "NBM", "NDMM",
            ("CAP-DESEQ-003",),
        )
        ui.plan(plan, atlas_summary())
        ui.error("validation", "NBM and NDMM require review")
        rendered = "\n".join(collector.values)
        self.assertIn(
            "Contrast: Normal Bone Marrow (NBM) − Newly Diagnosed Multiple Myeloma (NDMM)",
            rendered,
        )
        self.assertIn(
            "Normal Bone Marrow (NBM) and Newly Diagnosed Multiple Myeloma (NDMM) require review",
            rendered,
        )
        self.assertEqual((plan.group_a, plan.group_b), ("NBM", "NDMM"))

    def test_results_dashboard_uses_display_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = AnalysisPlan(
                "question", "GZMB CD8 T cell", "NDMM", "RRMM",
                ("CAP-DESEQ-003",),
            )
            result = ApplicationRunResult(plan, root, "completed", 1.0)
            result.de = fake_de(root)
            collector = PlainCollector()
            TerminalUI(collector, force_plain=True).final(result)
            rendered = "\n".join(collector.values)
            self.assertIn(
                "Contrast: Newly Diagnosed Multiple Myeloma (NDMM) − Relapsed/Refractory Multiple Myeloma (RRMM)",
                rendered,
            )
            self.assertIn("Higher in Newly Diagnosed Multiple Myeloma (NDMM)", rendered)
            self.assertIn("Higher in Relapsed/Refractory Multiple Myeloma (RRMM)", rendered)


class AtlasAdapterTests(unittest.TestCase):
    def test_adapter_uses_100_cells_and_writes_aligned_integer_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = [f"A{i}" for i in range(3)] + [f"B{i}" for i in range(3)]
            sample_values = np.repeat(samples, 105)
            groups = np.repeat(["NDMM"] * 3 + ["RRMM"] * 3, 105)
            obs = pd.DataFrame({
                "sample": sample_values,
                "dataset": np.tile(np.repeat(["D1", "D2", "D3"], 105), 2),
                "stage_model_v2": groups,
                "preserved": ["GZMB CD8 T cell"] * len(sample_values),
            }, index=[f"cell_{i}" for i in range(len(sample_values))])
            counts = np.arange(len(sample_values) * 8, dtype=np.int64)
            counts = counts.reshape(len(sample_values), 8) % 20
            atlas = ad.AnnData(X=counts, obs=obs)
            atlas.layers["counts"] = counts
            atlas.var_names = [f"gene_{i}" for i in range(8)]
            path = root / "atlas.h5ad"
            atlas.write_h5ad(path)
            adapter = AtlasPseudobulkAdapter(path, chunk_size=37)
            result = adapter.build(
                cell_state="GZMB CD8 T cell", group_a="NDMM",
                group_b="RRMM", output_dir=root / "adapter",
            )
            matrix = pd.read_csv(
                result.count_matrix_path, sep="\t", index_col=0)
            metadata = pd.read_csv(result.sample_metadata_path, sep="\t")
            self.assertEqual(matrix.columns.tolist(), metadata["sample"].tolist())
            self.assertEqual(result.group_replicate_counts, {"NDMM": 3, "RRMM": 3})
            self.assertEqual(AtlasPseudobulkAdapter.minimum_cells_per_sample_state, 100)
            self.assertTrue(np.issubdtype(matrix.to_numpy().dtype, np.integer))
            provenance = json.loads(result.provenance_path.read_text())
            self.assertEqual(provenance["minimum_cells_per_sample_state"], 100)
            self.assertEqual(provenance["count_source"], "layers/counts")
            restored = adapter.build(
                cell_state="GZMB CD8 T cell", group_a="NDMM",
                group_b="RRMM", output_dir=root / "adapter",
            )
            self.assertTrue(restored.cache_hit)


def fake_de(root: Path, *, status: str = "completed", significant: int = 0):
    table_path = root / "deseq2_results.tsv"
    pd.DataFrame({
        "gene": ["G1", "G2", "G3"],
        "signed_statistic": [2.0, -1.0, 0.5],
        "log2FoldChange": [1.0, -0.5, 0.1],
    }).to_csv(table_path, sep="\t", index=False)
    provenance = root / "de_provenance.json"
    provenance.write_text("{}")
    artifact = SimpleNamespace(
        logical_name="deseq2_results", path=str(table_path),
        category="inferential", media_type="text/tab-separated-values",
    )
    design = SimpleNamespace(
        design_formula="~ group",
        model_dump=lambda mode=None: {
            "group_replicate_counts": {"NDMM": 3, "RRMM": 3},
            "represented_datasets": ["D1"],
            "shared_datasets": ["D1"],
            "design_formula": "~ group",
            "design_columns": ["Intercept", "group"],
            "design_rank": 2,
            "design_column_count": 2,
            "residual_degrees_of_freedom": 4,
            "full_rank": True,
            "group_coefficient": "groupNDMM",
            "estimable": True,
            "warnings": [],
            "blocking_reasons": [],
        },
    )
    return SimpleNamespace(
        capability_id="CAP-DESEQ-003", capability_version="1.0.0",
        terminal_status=status, cache_key="de-key", cache_hit=False,
        comparison_direction="group_a_minus_group_b",
        evidence_class="adjusted_inference",
        design_assessment=design, input_gene_count=3, retained_gene_count=3,
        tested_gene_count=3, significant_gene_count=significant,
        upregulated_in_group_a_count=1, upregulated_in_group_b_count=1,
        group_replicate_counts={"NDMM": 3, "RRMM": 3},
        artifacts=[artifact], warnings=[], blocking_reasons=[],
        provenance_path=str(provenance), cache_manifest_path=str(root / "manifest.json"),
    )


class OrchestrationEvidenceTests(unittest.TestCase):
    def test_signed_program_uses_complete_wald_vector(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            de = fake_de(root, significant=0)
            plan = AnalysisPlan(
                "question", "GZMB CD8 T cell", "NDMM", "RRMM",
                ("CAP-DESEQ-003", "CAP-TF-002"),
            )
            path = signed_program_from_de(de, plan, root / "signed.tsv")
            table = pd.read_csv(path, sep="\t")
            self.assertEqual(len(table), de.tested_gene_count)
            self.assertEqual(table.signed_statistic.tolist(), [2.0, -1.0, 0.5])

    def test_exploratory_tf_program_uses_only_conserved_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            de = fake_de(root)
            conserved = root / "conserved_features.csv"
            pd.DataFrame({
                "gene": ["G1"], "full_log2FoldChange": [1.25],
                "conserved": [True],
            }).to_csv(conserved, index=False)
            de.evidence_class = "exploratory_lodo_conserved"
            de.artifacts.append(SimpleNamespace(
                logical_name="conserved_features", path=str(conserved),
                category="inferential", media_type="text/csv",
            ))
            plan = AnalysisPlan(
                "question", "GZMB CD8 T cell", "NDMM", "RRMM",
                ("CAP-DESEQ-003", "CAP-TF-002"),
                confounded_design_policy="exploratory_lodo",
            )
            table = pd.read_csv(signed_program_from_de(de, plan, root / "signed.tsv"), sep="\t")
            self.assertEqual(table.feature_id.tolist(), ["G1"])
            self.assertEqual(table.signed_statistic.tolist(), [1.25])
            self.assertEqual(table.feature_set_id.iloc[0], "exploratory_lodo_conserved_features")

    def test_combined_bundle_is_concise_and_reference_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            de = fake_de(root)
            adapter = SimpleNamespace(
                atlas_identity="atlas", count_source="layers/counts",
                n_cells=630, n_genes=3,
                group_replicate_counts={"NDMM": 3, "RRMM": 3},
                provenance_path=root / "adapter.json",
            )
            adapter.provenance_path.write_text("{}")
            plan = AnalysisPlan(
                "question", "GZMB CD8 T cell", "NDMM", "RRMM",
                ("CAP-DESEQ-003",),
            )
            bundle = build_combined_evidence_bundle(
                plan=plan, adapter=adapter, de=de, tf=None,
                tf_status="not_requested", tf_message="",
                cap_tf_001_status="not_requested", run_dir=root,
            )
            serialized = bundle.to_dict()
            self.assertEqual(
                serialized["deterministic_evidence"]["capabilities"]
                ["CAP-DESEQ-003"]["significant_gene_count"], 0)
            self.assertNotIn("table", json.dumps(serialized).casefold())
            self.assertTrue(all("path" in item for item in serialized["artifacts"]))
            context = serialized["biological_context"]
            self.assertEqual(context["group_a"], "NDMM")
            self.assertEqual(context["group_b"], "RRMM")
            self.assertEqual(context["display_group_a"], disease_group_label("NDMM"))
            self.assertEqual(context["display_group_b"], disease_group_label("RRMM"))

    def test_terminal_plan_displays_exploratory_design_warning(self):
        collector = PlainCollector()
        plan = AnalysisPlan(
            "question", "GZMB CD8 T cell", "NDMM", "RRMM",
            ("CAP-DESEQ-003",), confounded_design_policy="exploratory_lodo",
        )
        TerminalUI(collector, force_plain=True).plan(plan, atlas_summary())
        rendered = "\n".join(collector.values)
        self.assertIn("Design: exploratory ~ group", rendered)
        self.assertIn("Robustness: dataset-level LODO", rendered)
        self.assertIn("Confounding policy: exploratory_lodo", rendered)
        self.assertIn("Warning: group and dataset are not independently identifiable", rendered)

    def test_blocked_de_prevents_tf_and_reasoning_failure_preserves_bundle(self):
        class Adapter:
            def cache_key(self, *args):
                return "adapter-key"

            def build(self, **kwargs):
                root = Path(kwargs["output_dir"])
                root.mkdir(parents=True)
                for name in ("counts.tsv", "metadata.tsv", "provenance.json"):
                    (root / name).write_text("{}" if name.endswith("json") else "x\n")
                return SimpleNamespace(
                    count_matrix_path=root / "counts.tsv",
                    sample_metadata_path=root / "metadata.tsv",
                    provenance_path=root / "provenance.json",
                    atlas_identity="atlas", count_source="layers/counts",
                    n_cells=600, n_genes=3,
                    group_replicate_counts={"NDMM": 3, "RRMM": 3},
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tf_calls = []
            reasoning_calls = []
            blocked = fake_de(root, status="blocked")
            orchestrator = CellStateOrchestrator(
                adapter=Adapter(), output_root=root / "out",
                cache_root=root / "cache",
                de_runner=lambda request, context: blocked,
                tf_runner=lambda *args: tf_calls.append(args),
                reasoning_engine=SimpleNamespace(
                    run=lambda *args: reasoning_calls.append(args)),
            )
            result = orchestrator.run(AnalysisPlan(
                "question", "GZMB CD8 T cell", "NDMM", "RRMM",
                ("CAP-DESEQ-003", "CAP-TF-002"),
            ))
            self.assertEqual(result.overall_status, "blocked")
            self.assertEqual(result.tf_status, "blocked_by_dependency")
            self.assertFalse(tf_calls)
            self.assertEqual(len(reasoning_calls), 1)
            self.assertEqual(
                reasoning_calls[0][0].execution_status.value, "blocked")
            self.assertTrue(result.evidence_bundle_path.is_file())

    def test_connected_nodes_share_adapter_and_reasoning_receives_combined_bundle(self):
        class Adapter:
            calls = 0

            def cache_key(self, *args):
                return "adapter-key"

            def build(self, **kwargs):
                self.calls += 1
                root = Path(kwargs["output_dir"])
                root.mkdir(parents=True, exist_ok=True)
                for name in ("counts.tsv", "metadata.tsv", "provenance.json"):
                    (root / name).write_text(
                        "{}" if name.endswith("json") else "x\n")
                return SimpleNamespace(
                    count_matrix_path=root / "counts.tsv",
                    sample_metadata_path=root / "metadata.tsv",
                    provenance_path=root / "provenance.json",
                    atlas_identity="atlas", count_source="layers/counts",
                    n_cells=600, n_genes=3, cache_hit=False,
                    group_replicate_counts={"NDMM": 3, "RRMM": 3},
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            doro, coll = root / "dorothea.tsv", root / "collectri.tsv"
            resource = pd.DataFrame({
                "source": ["TF"] * 5,
                "target": [f"G{i}" for i in range(5)],
                "weight": [1, -1, 1, -1, 1],
                "organism": ["human"] * 5,
            })
            resource.assign(confidence="A").to_csv(doro, sep="\t", index=False)
            resource.to_csv(coll, sep="\t", index=False)
            de_calls, tf_rows, tf_requests, reasoning_bundles = [], [], [], []

            def de_runner(request, context):
                de_calls.append(request)
                return fake_de(root)

            def tf_runner(request, context):
                tf_requests.append(request)
                signed = pd.read_csv(
                    request.signed_feature_program_path, sep="\t")
                tf_rows.append(len(signed))
                qc = root / "tf_qc.json"
                qc.write_text(json.dumps({
                    "estimable_model_count": 4,
                    "directional_consensus_count": 1,
                }))
                provenance = root / "tf_provenance.json"
                provenance.write_text("{}")
                result = CapabilityResult(
                    "CAP-TF-002", "1.0.0", CapabilityStatus.COMPLETED,
                    "tf-key", False,
                    (ArtifactReference(
                        "qc_summary", str(qc), ArtifactCategory.QC,
                        "application/json"),),
                    (), str(provenance), str(root / "tf_manifest.json"),
                )
                return SimpleNamespace(
                    capability_id="CAP-TF-002", node_version="1.0.0",
                    status=CapabilityStatus.COMPLETED, cache_key="tf-key",
                    cache_hit=False,
                    artifact_paths=(("qc_summary", str(qc), "QC",
                                     "application/json"),),
                    provenance_path=str(provenance),
                    cache_manifest_path=str(root / "tf_manifest.json"),
                    warnings=(), to_capability_result=lambda: result,
                )

            class FailingReasoning:
                def run(self, bundle, run_dir):
                    reasoning_bundles.append(bundle)
                    raise ReasoningError("offline")

            adapter = Adapter()
            with patch.dict(os.environ, {
                "CELLSTATE_DOROTHEA_PATH": str(doro),
                "CELLSTATE_COLLECTRI_PATH": str(coll),
            }, clear=True):
                result = CellStateOrchestrator(
                    adapter=adapter, output_root=root / "out",
                    cache_root=root / "cache", de_runner=de_runner,
                    tf_runner=tf_runner, reasoning_engine=FailingReasoning(),
                ).run(AnalysisPlan(
                    "question", "GZMB CD8 T cell", "NDMM", "RRMM",
                    ("CAP-DESEQ-003", "CAP-TF-002"),
                ))
            self.assertEqual(adapter.calls, 1)
            self.assertEqual(len(de_calls), 1)
            self.assertEqual((de_calls[0].group_a, de_calls[0].group_b),
                             ("NDMM", "RRMM"))
            self.assertEqual(tf_requests[0].dorothea_path, doro)
            self.assertEqual(tf_requests[0].collectri_path, coll)
            self.assertEqual(tf_rows, [3])
            self.assertEqual(len(reasoning_bundles), 1)
            capabilities = reasoning_bundles[0].deterministic_evidence[
                "capabilities"]
            self.assertIn("CAP-DESEQ-003", capabilities)
            self.assertIn("CAP-TF-002", capabilities)
            self.assertEqual(result.reasoning_error, "offline")
            self.assertTrue(result.evidence_bundle_path.is_file())
            self.assertTrue(Path(result.de.provenance_path).is_file())


if __name__ == "__main__":
    unittest.main()
