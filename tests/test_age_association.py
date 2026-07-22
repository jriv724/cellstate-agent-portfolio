from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cellstate.context import AnalysisContext
from cellstate.nodes.age_association import (
    _bh,
    calculate_group_age_association,
    calculate_ordered_age_association,
    run_group_age_association,
    run_ordered_age_association,
)
from cellstate.schemas.age_association import GroupAgeAssociationInput, OrderedAgeAssociationInput

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "age_association"


def group_fixture() -> pd.DataFrame:
    rows = []
    values = {
        "NBM": [20.0, 21.0, 22.0],
        "SMM": [40.0, 41.0, 42.0],
        "NDMM": [60.0, 61.0, 62.0],
    }
    for class_name, offset in (("B", 0.0), ("CD8T", 5.0)):
        for group, ages in values.items():
            for index, age in enumerate(ages):
                rows.append({"sample": f"{group}-{index}", "dataset": "d1", "cell_type": class_name,
                             "stage_model": group, "immune_age": age + offset})
    return pd.DataFrame(rows)


def group_request(path: Path, **overrides) -> GroupAgeAssociationInput:
    values = {"table_path": path, "outcome_column": "immune_age",
              "group_levels": ("NBM", "SMM", "NDMM"), "class_levels": ("B", "CD8T"),
              "minimum_replicates_per_group": 2}
    values.update(overrides)
    return GroupAgeAssociationInput(**values)


def ordered_request(path: Path, **overrides) -> OrderedAgeAssociationInput:
    values = {"table_path": path, "outcome_column": "immune_age", "class_levels": ("B", "CD8T")}
    values.update(overrides)
    return OrderedAgeAssociationInput(**values)


class AgeAssociationTests(unittest.TestCase):
    def test_group_source_filters_summaries_and_direction(self):
        analysis, summary, omnibus, pairwise, _ = calculate_group_age_association(
            group_fixture(), group_request(Path("unused.tsv")))
        self.assertEqual(len(analysis), 18)
        self.assertAlmostEqual(summary.query("cell_type == 'B' and stage_model == 'NBM'").iloc[0]["mean"], 21.0)
        row = pairwise.query("cell_type == 'B' and group1 == 'NBM' and group2 == 'SMM'").iloc[0]
        self.assertEqual((row.group1, row.group2), ("NBM", "SMM"))
        self.assertEqual(row.statistic, 0.0)
        self.assertTrue((omnibus.p_value < 0.05).all())

    def test_bh_is_within_class_and_mathematically_controlled(self):
        adjusted = _bh([0.01, 0.04, 0.03, np.nan])
        np.testing.assert_allclose(adjusted[:3], [0.03, 0.04, 0.04])
        self.assertTrue(np.isnan(adjusted[3]))

    def test_group_missing_column_and_duplicate_unit_fail(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            calculate_group_age_association(group_fixture().drop(columns="immune_age"), group_request(Path("unused.tsv")))
        duplicate = pd.concat([group_fixture(), group_fixture().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "one analysis row"):
            calculate_group_age_association(duplicate, group_request(Path("unused.tsv")))

    def test_group_dataset_confounding_warning(self):
        data = group_fixture()
        data["dataset"] = data["stage_model"].map({"NBM": "d1", "SMM": "d2", "NDMM": "d3"})
        warnings = calculate_group_age_association(
            data, group_request(Path("unused.tsv"), dataset_column="dataset"))[-1]
        codes = {warning.code for warning in warnings}
        self.assertIn("DATASET_GROUP_CONFOUNDING", codes)
        self.assertIn("NO_DATASET_OVERLAP", codes)

    def test_group_valid_without_dataset_column(self):
        data = group_fixture().drop(columns="dataset")
        analysis, _, omnibus, pairwise, warnings = calculate_group_age_association(
            data, group_request(Path("unused.tsv")))
        self.assertEqual(len(analysis), len(data))
        self.assertEqual(len(omnibus), 2)
        self.assertEqual(len(pairwise), 6)
        self.assertNotIn("NO_DATASET_OVERLAP", {warning.code for warning in warnings})

    def test_fewer_than_two_samples_warns_but_source_test_is_retained(self):
        data = group_fixture().query("not (cell_type == 'B' and stage_model == 'NBM' and sample != 'NBM-0')")
        _, _, omnibus, pairwise, warnings = calculate_group_age_association(
            data, group_request(Path("unused.tsv"), minimum_replicates_per_group=1))
        low = [warning for warning in warnings if warning.code == "LOW_BIOLOGICAL_REPLICATION"]
        self.assertEqual(len(low), 1)
        self.assertEqual(low[0].severity.value, "error")
        self.assertEqual(low[0].context["counts"], {"NBM": 1})
        self.assertIn("B", set(omnibus.cell_type))
        self.assertTrue(((pairwise.cell_type == "B") & (pairwise.group1 == "NBM")).any())

    def test_fixed_rstatix_regression_fixture(self):
        source = FIXTURE_DIR / "rstatix_cd8t_stage_input.tsv"
        data = pd.read_csv(source, sep="\t")
        expected_kw = pd.read_csv(FIXTURE_DIR / "rstatix_cd8t_stage_kruskal.tsv", sep="\t")
        expected_pairs = pd.read_csv(FIXTURE_DIR / "rstatix_cd8t_stage_pairwise.tsv", sep="\t")
        request = GroupAgeAssociationInput(
            source, outcome_column="mean_pbmc_cd8t_age", class_levels=("CD8T",), dataset_column=None)
        _, _, observed_kw, observed_pairs, _ = calculate_group_age_association(data, request)
        self.assertAlmostEqual(observed_kw.iloc[0].statistic, expected_kw.iloc[0].statistic, places=10)
        self.assertAlmostEqual(observed_kw.iloc[0].p_value, expected_kw.iloc[0].p, delta=5e-6)
        merged = observed_pairs.merge(expected_pairs, on=["group1", "group2", "n1", "n2"], suffixes=("_scipy", "_r"))
        self.assertEqual(len(merged), len(expected_pairs))
        np.testing.assert_allclose(merged.statistic_scipy, merged.statistic_r, rtol=0, atol=0)
        # The source R table stores raw p-values with as few as two decimals (for example 0.79).
        np.testing.assert_allclose(merged.p_value, merged.p, rtol=0, atol=5e-3)
        np.testing.assert_allclose(merged.p_adjusted, merged["p.adj"], rtol=0, atol=1e-3)

    def test_group_constant_outcome_is_structured(self):
        data = group_fixture()
        data["immune_age"] = 10.0
        _, _, omnibus, _, warnings = calculate_group_age_association(data, group_request(Path("unused.tsv")))
        self.assertTrue((omnibus.p_value == 1.0).all())
        self.assertIn("CONSTANT_OUTCOME", {warning.code for warning in warnings})

    def test_ordered_spearman_preserves_cr_nr_er_direction(self):
        analysis, results, warnings = calculate_ordered_age_association(
            group_fixture().rename(columns={"sample": "patient_id", "stage_model": "response_group"})
            .replace({"response_group": {"NBM": "CR", "SMM": "NR", "NDMM": "ER"}}),
            ordered_request(Path("unused.tsv")),
        )
        self.assertEqual(dict(zip(analysis.response_group, analysis.response_score))["CR"], 1.0)
        self.assertTrue((results.spearman_rho > 0.9).all())
        self.assertIn("ORDINAL_SPACING_ASSUMPTION", {warning.code for warning in warnings})

    def test_ordered_alternative_mapping_reverses_direction(self):
        data = group_fixture().rename(columns={"sample": "patient_id", "stage_model": "response_group"})
        data = data.replace({"response_group": {"NBM": "CR", "SMM": "NR", "NDMM": "ER"}})
        _, result, _ = calculate_ordered_age_association(
            data, ordered_request(Path("unused.tsv"), score_mapping=(("CR", 3.0), ("NR", 2.0), ("ER", 1.0))))
        self.assertTrue((result.spearman_rho < -0.9).all())

    def test_ordered_missing_levels_warns_without_inventing_result(self):
        data = group_fixture().query("stage_model != 'NDMM'").rename(
            columns={"sample": "patient_id", "stage_model": "response_group"})
        data = data.replace({"response_group": {"NBM": "CR", "SMM": "NR"}})
        _, result, warnings = calculate_ordered_age_association(data, ordered_request(Path("unused.tsv")))
        self.assertTrue(result.empty)
        self.assertIn("MISSING_ORDERED_LEVELS", {warning.code for warning in warnings})

    def test_group_cache_provenance_and_parameter_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ages.tsv"
            group_fixture().to_csv(source, sep="\t", index=False)
            context = AnalysisContext(root / "out", root / "cache", "age-fixture-v1")
            first = run_group_age_association(group_request(source), context)
            second = run_group_age_association(group_request(source), context)
            reversed_result = run_group_age_association(
                group_request(source, contrasts=(("SMM", "NBM"),)), context)
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertNotEqual(first.cache_key, reversed_result.cache_key)
            self.assertEqual(first.provenance.unit_of_inference, "biological_sample_or_patient")
            self.assertEqual(first.provenance.model_formula, "immune_age ~ stage_model")
            self.assertEqual(first.provenance.covariates, ())
            manifest = json.loads(Path(first.cache_manifest_path).read_text())
            self.assertEqual(manifest["completion_status"], "complete")

    def test_ordered_cache_is_deterministic_and_mapping_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "response.tsv"
            data = group_fixture().rename(columns={"sample": "patient_id", "stage_model": "response_group"})
            data.replace({"response_group": {"NBM": "CR", "SMM": "NR", "NDMM": "ER"}}).to_csv(source, sep="\t", index=False)
            context = AnalysisContext(root / "out", root / "cache", "response-fixture-v1")
            first = run_ordered_age_association(ordered_request(source), context)
            second = run_ordered_age_association(ordered_request(source), context)
            reverse = run_ordered_age_association(
                ordered_request(source, score_mapping=(("CR", 3.0), ("NR", 2.0), ("ER", 1.0))), context)
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertNotEqual(first.cache_key, reverse.cache_key)
            pd.testing.assert_frame_equal(pd.read_csv(first.statistics_paths[0]), pd.read_csv(second.statistics_paths[0]))


if __name__ == "__main__":
    unittest.main()
