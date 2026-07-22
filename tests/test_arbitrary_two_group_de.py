"""Contract and orchestration tests for CAP-DESEQ-003 (no real R required)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from pydantic import ValidationError

from cellstate.capability_specs import CAPABILITY_SPECS_BY_ID
from cellstate.context import AnalysisContext
from cellstate.nodes.arbitrary_two_group_de import (
    run_arbitrary_two_group_de,
    summarize_exploratory_lodo,
    validate_arbitrary_two_group_inputs,
    validate_deseq2_result_table,
)
from cellstate.schemas.arbitrary_two_group_de import ArbitraryTwoGroupDEInput


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = [f"A{i}" for i in range(1, 4)] + [f"B{i}" for i in range(1, 4)]
    counts = pd.DataFrame(
        {
            sample: [100 if sample.startswith("A") else 20,
                     20 if sample.startswith("A") else 100, 30, 1]
            for sample in samples
        },
        index=["higher_a", "higher_b", "null", "filtered"],
    )
    metadata = pd.DataFrame(
        {
            "sample": samples,
            "group": ["A"] * 3 + ["B"] * 3,
            "dataset": ["D1"] * 6,
            "n_cells": [100] * 6,
        }
    )
    return counts, metadata


class ArbitraryTwoGroupDETests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.counts_path = self.root / "counts.tsv"
        self.metadata_path = self.root / "metadata.tsv"
        counts, metadata = _tables()
        counts.to_csv(self.counts_path, sep="\t")
        metadata.to_csv(self.metadata_path, sep="\t", index=False)
        self.context = AnalysisContext(
            (self.root / "out").resolve(),
            (self.root / "cache").resolve(),
            "cap-deseq-003-fixture",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def request(self, **changes: object) -> ArbitraryTwoGroupDEInput:
        values = {
            "count_matrix_path": self.counts_path,
            "sample_metadata_path": self.metadata_path,
            "group_a": "A",
            "group_b": "B",
            "rscript_path": self.root / "Rscript",
        }
        values.update(changes)
        return ArbitraryTwoGroupDEInput(**values)

    def test_registry_addition_does_not_change_existing_contracts(self) -> None:
        new = CAPABILITY_SPECS_BY_ID["CAP-DESEQ-003"]
        self.assertEqual(new.upstream_capability_dependencies, ("CAP-DESEQ-001",))
        self.assertEqual(new.unit_of_inference,
                         "independent_biological_pseudobulk_replicate")
        old_1 = CAPABILITY_SPECS_BY_ID["CAP-DESEQ-001"]
        old_2 = CAPABILITY_SPECS_BY_ID["CAP-DESEQ-002"]
        self.assertEqual(old_1.analysis_class, "pseudobulk_construction")
        self.assertEqual(old_1.upstream_capability_dependencies, ())
        self.assertEqual(old_2.required_metadata_columns,
                         ("sample", "dataset", "stage_model_v2", "cell_state"))
        self.assertEqual(old_2.upstream_capability_dependencies,
                         ("CAP-DESEQ-001",))

    def test_schema_fixes_production_thresholds_and_order(self) -> None:
        request = self.request()
        self.assertEqual(request.minimum_cells_per_replicate, 100)
        self.assertEqual(request.minimum_replicates_per_group, 3)
        self.assertEqual(request.parameters()["groups"], ["A", "B"])
        self.assertEqual(request.confounded_design_policy, "block")
        with self.assertRaises(ValidationError):
            self.request(minimum_cells_per_replicate=20)

    def test_validation_prefilter_and_one_dataset_design(self) -> None:
        prepared = validate_arbitrary_two_group_inputs(self.request())
        self.assertEqual(list(prepared.counts.index),
                         ["higher_a", "higher_b", "null"])
        self.assertEqual(prepared.assessment.design_formula, "~ group")
        self.assertEqual(prepared.assessment.group_replicate_counts,
                         {"A": 3, "B": 3})
        self.assertEqual(prepared.filter_summary["filtered_gene_count"], 1)

    def test_multi_dataset_design_requires_overlap(self) -> None:
        metadata = pd.read_csv(self.metadata_path, sep="\t")
        metadata["dataset"] = ["D1", "D1", "D2", "D1", "D2", "D2"]
        metadata.to_csv(self.metadata_path, sep="\t", index=False)
        prepared = validate_arbitrary_two_group_inputs(self.request())
        self.assertEqual(prepared.assessment.design_formula, "~ dataset + group")
        self.assertEqual(prepared.assessment.shared_datasets, ["D1", "D2"])

        metadata["dataset"] = ["D1"] * 3 + ["D2"] * 3
        metadata.to_csv(self.metadata_path, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "confounded"):
            validate_arbitrary_two_group_inputs(self.request())

    def test_invalid_counts_and_identifiers_are_rejected(self) -> None:
        counts = pd.read_csv(
            self.counts_path, sep="\t", index_col=0, keep_default_na=False)
        cases = [("nonnegative", -1), ("integer-like", 1.5)]
        for expected, value in cases:
            changed = counts.astype(float)
            changed.loc["higher_a", "A1"] = value
            changed.to_csv(self.counts_path, sep="\t")
            with self.assertRaisesRegex(ValueError, expected):
                validate_arbitrary_two_group_inputs(self.request())
        counts.to_csv(self.counts_path, sep="\t")

        duplicate = counts.copy()
        duplicate.index = ["dup", "dup", "other", "low"]
        duplicate.to_csv(self.counts_path, sep="\t")
        with self.assertRaisesRegex(ValueError, "gene identifiers"):
            validate_arbitrary_two_group_inputs(self.request())

    def test_replication_cell_qc_alignment_and_empty_filter_block(self) -> None:
        metadata = pd.read_csv(self.metadata_path, sep="\t")
        metadata.loc[0, "n_cells"] = 99
        metadata.to_csv(self.metadata_path, sep="\t", index=False)
        blocked = run_arbitrary_two_group_de(self.request(), self.context)
        self.assertEqual(blocked.terminal_status, "blocked")
        self.assertIn("insufficient independent replication",
                      blocked.blocking_reasons[0])

        _, metadata = _tables()
        metadata.loc[0, "sample"] = "misaligned"
        metadata.to_csv(self.metadata_path, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "misaligned"):
            validate_arbitrary_two_group_inputs(self.request())

        counts, metadata = _tables()
        counts.loc[:, :] = 1
        counts.to_csv(self.counts_path, sep="\t")
        metadata.to_csv(self.metadata_path, sep="\t", index=False)
        blocked = run_arbitrary_two_group_de(self.request(), self.context)
        self.assertEqual(blocked.terminal_status, "blocked")
        self.assertIn("no genes survive", blocked.blocking_reasons[0])

    def test_duplicate_metadata_replicates_are_rejected(self) -> None:
        metadata = pd.read_csv(self.metadata_path, sep="\t")
        metadata.loc[1, "sample"] = metadata.loc[0, "sample"]
        metadata.to_csv(self.metadata_path, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "one row per"):
            validate_arbitrary_two_group_inputs(self.request())

    def test_duplicate_count_replicates_are_rejected_before_pandas_mangling(self) -> None:
        self.counts_path.write_text(
            "gene\tA1\tA1\nhigher_a\t20\t30\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "replicate identifiers"):
            validate_arbitrary_two_group_inputs(self.request())

    def test_rank_deficient_design_blocks_before_runtime(self) -> None:
        with patch(
            "cellstate.nodes.arbitrary_two_group_de.validate_design_estimability",
            side_effect=ValueError("design matrix is rank deficient"),
        ):
            output = run_arbitrary_two_group_de(self.request(), self.context)
        self.assertEqual(output.terminal_status, "blocked")
        self.assertIn("rank deficient", output.blocking_reasons[0])

    def test_result_validation_preserves_complete_signed_statistics_and_null_padj(self) -> None:
        table = pd.DataFrame(
            {
                "gene": ["higher_a", "higher_b", "null"],
                "baseMean": [60, 60, 30],
                "log2FoldChange": [2.0, -2.0, 0.0],
                "lfcSE": [0.3, 0.3, 0.4],
                "stat": [6.0, -6.0, 0.0],
                "pvalue": [1e-6, 1e-6, 1.0],
                "padj": [0.001, 0.001, None],
                "signed_statistic": [6.0, -6.0, 0.0],
            }
        )
        result = validate_deseq2_result_table(
            table, expected_genes=pd.Index(["higher_a", "higher_b", "null"]),
            request=self.request(),
        )
        self.assertEqual(len(result), 3)
        self.assertTrue(result.signed_statistic.equals(result.stat))
        self.assertTrue(pd.isna(result.set_index("gene").loc["null", "padj"]))
        self.assertEqual(result.set_index("gene").loc["higher_a", "direction"],
                         "higher_in_group_a")

    def _fake_r(self, command: list[str], **_: object) -> None:
        counts = pd.read_csv(
            command[2], sep="\t", index_col=0, keep_default_na=False)
        rows = []
        for gene in counts.index:
            effect = {"higher_a": 2.0, "higher_b": -2.0}.get(gene, 0.0)
            stat = effect * 2
            rows.append({
                "gene": gene, "baseMean": 30.0, "log2FoldChange": effect,
                "lfcSE": 0.5, "stat": stat, "pvalue": 0.5, "padj": None,
                "signed_statistic": stat,
            })
        pd.DataFrame(rows).to_csv(command[4], sep="\t", index=False)
        pd.DataFrame({"key": ["R", "DESeq2"],
                      "value": ["test", "test"]}).to_csv(
                          command[5], sep="\t", index=False)

    def test_completed_zero_significance_artifacts_provenance_and_cache(self) -> None:
        with patch(
            "cellstate.nodes.arbitrary_two_group_de.subprocess.run",
            side_effect=self._fake_r,
        ):
            first = run_arbitrary_two_group_de(self.request(), self.context)
            second = run_arbitrary_two_group_de(self.request(), self.context)
        self.assertEqual(first.terminal_status, "completed")
        self.assertEqual(first.significant_gene_count, 0)
        self.assertEqual(first.tested_gene_count, first.retained_gene_count)
        self.assertTrue(second.cache_hit)
        paths = {artifact.logical_name: Path(artifact.path)
                 for artifact in first.artifacts}
        self.assertTrue(all(path.exists() for path in paths.values()))
        provenance = json.loads(Path(first.provenance_path).read_text())
        self.assertEqual(provenance["capability_id"], "CAP-DESEQ-003")
        self.assertEqual(provenance["reference_group"], "B")
        self.assertEqual(provenance["parameters"]["coefficient_extracted"], "A - B")

    def test_cache_identity_is_direction_sensitive(self) -> None:
        with patch(
            "cellstate.nodes.arbitrary_two_group_de.subprocess.run",
            side_effect=self._fake_r,
        ):
            ab = run_arbitrary_two_group_de(self.request(), self.context)
            ba = run_arbitrary_two_group_de(
                self.request(group_a="B", group_b="A"), self.context)
        self.assertNotEqual(ab.cache_key, ba.cache_key)
        self.assertEqual(ba.comparison_direction, "B - A")

    def test_conserved_feature_summary_rejects_instability_and_full_nonsignificance(self) -> None:
        full = pd.DataFrame({
            "gene": ["stable", "unstable", "not_full_sig"],
            "log2FoldChange": [1.0, 1.0, 1.0],
            "pvalue": [0.001, 0.001, 0.2],
            "padj": [0.01, 0.01, 0.2],
        })
        fold = pd.DataFrame([
            {"gene": gene, "omitted_dataset": dataset,
             "log2FoldChange": effect, "pvalue": p, "padj": padj}
            for gene, effects in {
                "stable": [0.8, 1.1, 0.9],
                "unstable": [0.8, -0.6, 1.0],
                "not_full_sig": [0.9, 1.0, 1.1],
            }.items()
            for dataset, effect, p, padj in zip(
                ["D1", "D2", "D3"], effects, [.01, .2, .03], [.03, .4, .08])
        ])
        summary = summarize_exploratory_lodo(
            full, fold, eligible_fold_count=3,
            request=self.request(confounded_design_policy="exploratory_lodo"),
        ).set_index("gene")
        self.assertTrue(summary.loc["stable", "conserved"])
        self.assertFalse(summary.loc["unstable", "conserved"])
        self.assertIn("direction_unstable", summary.loc["unstable", "exclusion_reason"])
        self.assertFalse(summary.loc["not_full_sig", "conserved"])
        self.assertIn("not_significant_in_full", summary.loc["not_full_sig", "exclusion_reason"])
        self.assertAlmostEqual(summary.loc["stable", "nominal_significant_fold_fraction"], 2 / 3)

    def _write_confounded_six_dataset_fixture(self) -> None:
        samples, groups, datasets = [], [], []
        for group in ("A", "B"):
            for dataset_number in range(1, 4):
                dataset = f"{group}D{dataset_number}"
                for replicate in range(3):
                    samples.append(f"{dataset}_{replicate}")
                    groups.append(group)
                    datasets.append(dataset)
        counts = pd.DataFrame({sample: [100, 60, 30] for sample in samples},
                              index=["stable", "unstable", "not_full_sig"])
        metadata = pd.DataFrame({"sample": samples, "group": groups,
                                 "dataset": datasets, "n_cells": 100})
        counts.to_csv(self.counts_path, sep="\t")
        metadata.to_csv(self.metadata_path, sep="\t", index=False)

    def _fake_exploratory_r(self, command: list[str], **_: object) -> None:
        metadata = pd.read_csv(command[3], sep="\t")
        omitted = ({f"AD{i}" for i in range(1, 4)} | {f"BD{i}" for i in range(1, 4)}) - set(metadata.dataset)
        omitted_name = next(iter(omitted), "full")
        unstable = -0.8 if omitted_name == "AD1" else 0.8
        rows = [
            ("stable", 1.0, .001, .01),
            ("unstable", unstable, .001, .01),
            ("not_full_sig", 1.0, .2, .2),
        ]
        pd.DataFrame([{ "gene": gene, "baseMean": 50, "log2FoldChange": effect,
            "lfcSE": .2, "stat": effect / .2, "pvalue": p, "padj": padj,
            "signed_statistic": effect / .2} for gene, effect, p, padj in rows]
        ).to_csv(command[4], sep="\t", index=False)
        pd.DataFrame({"key": ["R", "DESeq2"], "value": ["test", "test"]}).to_csv(
            command[5], sep="\t", index=False)

    def test_explicit_exploratory_lodo_runs_and_cache_varies_with_policy_thresholds(self) -> None:
        self._write_confounded_six_dataset_fixture()
        block = run_arbitrary_two_group_de(self.request(), self.context)
        self.assertEqual(block.terminal_status, "blocked")
        request = self.request(confounded_design_policy="exploratory_lodo")
        with patch("cellstate.nodes.arbitrary_two_group_de.subprocess.run",
                   side_effect=self._fake_exploratory_r):
            result = run_arbitrary_two_group_de(request, self.context)
            changed = run_arbitrary_two_group_de(
                self.request(confounded_design_policy="exploratory_lodo",
                             lodo_min_median_abs_log2fc=.5), self.context)
        self.assertEqual(result.terminal_status, "completed_with_warnings")
        self.assertEqual(result.evidence_class, "exploratory_lodo_conserved")
        names = {item.logical_name for item in result.artifacts}
        self.assertTrue({"full_unadjusted_deseq2_results", "lodo_fold_results",
                         "lodo_feature_summary", "conserved_features",
                         "skipped_lodo_folds", "exploratory_lodo_summary"} <= names)
        conserved_path = Path(next(a.path for a in result.artifacts if a.logical_name == "conserved_features"))
        self.assertEqual(pd.read_csv(conserved_path).gene.tolist(), ["stable"])
        self.assertIn("EXPLORATORY_CONFOUNDED_DESIGN", {w.code for w in result.warnings})
        provenance = json.loads(Path(result.provenance_path).read_text())
        self.assertEqual(provenance["parameters"]["confounded_design_policy"], "exploratory_lodo")
        self.assertNotEqual(result.cache_key, block.cache_key)
        self.assertNotEqual(result.cache_key, changed.cache_key)

    def test_insufficient_lodo_folds_produce_no_conserved_claim(self) -> None:
        self._write_confounded_six_dataset_fixture()
        request = self.request(confounded_design_policy="exploratory_lodo",
                               lodo_min_estimable_folds=7)
        with patch("cellstate.nodes.arbitrary_two_group_de.subprocess.run",
                   side_effect=self._fake_exploratory_r):
            result = run_arbitrary_two_group_de(request, self.context)
        self.assertEqual(result.terminal_status, "insufficient_robustness")
        self.assertNotIn("conserved_features", {a.logical_name for a in result.artifacts})
        self.assertIn("INSUFFICIENT_LODO_ROBUSTNESS", {w.code for w in result.warnings})


if __name__ == "__main__":
    unittest.main()
