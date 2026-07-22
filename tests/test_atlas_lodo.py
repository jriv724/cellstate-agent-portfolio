import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cellstate.cache import load_complete_manifest
from cellstate.capability_specs import CAPABILITY_SPECS_BY_ID
from cellstate.context import AnalysisContext
from cellstate.nodes.atlas_lodo import (
    CAPABILITY_ID, MODEL_FORMULA, SOURCE_FILES, _bh,
    build_sample_means_backed, build_sample_means_from_cells,
    calculate_atlas_lodo, cell_state_eligibility, fit_fold_ols,
    is_label_worthy_gene, load_sample_means, run_atlas_lodo,
)
from cellstate.schemas.atlas_lodo import AtlasLODOInput
from cellstate.schemas.common import CapabilityStatus


FIXTURE = Path(__file__).parent / "fixtures/reproducibility/inputs/atlas_lodo_sample_means.csv"
GENES = ("A_UP", "A_DOWN", "B_UP", "B_DOWN", "A_ENRICH", "B_ENRICH",
         "SHARED", "NULL", "LOW_EXPR", "RPL_LABEL")


def request(**changes):
    return AtlasLODOInput(FIXTURE, gene_columns=GENES, **changes)


class FakeBacked:
    def __init__(self):
        self.obs = pd.DataFrame({
            "sample": ["drop", "s1", "s1", "s2", "s2"],
            "dataset": ["D0", "D1", "D1", "D2", "D2"],
            "stage_model_v2": ["OTHER", "NBM", "NBM", "SMM", "SMM"],
            "preserved": ["C", "C", "C", "C", "C"],
        }, index=[11, 20, 31, 42, 55])
        self.X = np.array([[999.], [1.], [3.], [5.], [7.]])
        self.var_names = ["G"]
        self.n_obs = 5


class AtlasLODOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.req = request()
        cls.meta, cls.expr, cls.observed = load_sample_means(cls.req)
        cls.tables, cls.warnings, cls.status = calculate_atlas_lodo(
            cls.meta, cls.expr, cls.observed, cls.req)
        cls.summary = cls.tables["gene_summaries"].set_index("gene")

    def test_backed_safe_positions_and_exact_sample_mean(self):
        q = AtlasLODOInput(Path("unused.h5ad"), input_format="h5ad",
                           minimum_cells_per_sample_state=2,
                           minimum_reference_samples=1, minimum_group_a_samples=1,
                           minimum_group_b_samples=1, minimum_datasets=1)
        meta, expr, observed = build_sample_means_backed(FakeBacked(), q)
        self.assertEqual(observed, ("C",))
        self.assertEqual(meta["sample"].tolist(), ["s1", "s2"])
        np.testing.assert_allclose(expr["G"], [2., 6.])

        cells = pd.DataFrame({"cell_id": ["a", "b", "c"], "sample": ["s", "s", "t"],
            "dataset": ["D", "D", "D"], "stage_model_v2": ["NBM", "NBM", "SMM"],
            "preserved": ["C", "C", "C"]})
        values = pd.DataFrame({"cell_id": ["c", "a", "b"], "G": [9., 1., 3.]})
        flat_q = AtlasLODOInput(Path("m.csv"), input_format="cell_csv", expression_path=Path("x.csv"),
            gene_columns=("G",), minimum_cells_per_sample_state=2, minimum_reference_samples=1,
            minimum_group_a_samples=1, minimum_group_b_samples=1, minimum_datasets=1)
        m, e, _ = build_sample_means_from_cells(cells, values, ("G",), flat_q)
        self.assertEqual(m["sample"].tolist(), ["s"])
        self.assertEqual(e.loc[0, "G"], 2.)

    def test_threshold_and_global_eligibility_all_failure_reasons(self):
        self.assertEqual(len(self.meta), 26)
        eligibility = self.tables["cell_state_eligibility"].set_index("cell_state")
        self.assertTrue(bool(eligibility.loc["EligibleState", "eligible"]))
        self.assertEqual(eligibility.loc["IneligibleState", "failure_reasons"],
            "low_reference_samples;low_group_a_samples;low_group_b_samples;low_dataset_count")
        table = pd.read_csv(FIXTURE)
        self.assertNotIn("BELOW", self.meta["sample"].tolist())
        self.assertEqual(int(table.n_cells.min()), 100)

    def test_fold_membership_directions_bh_and_fold_count(self):
        folds = self.tables["fold_results"]
        self.assertEqual(set(folds.held_out_dataset), {"D1", "D2", "D3"})
        self.assertEqual(set(self.summary.successful_lodo_folds), {3})
        self.assertGreater(self.summary.loc["A_UP", "group_a_effect_median"], 0)
        self.assertGreater(self.summary.loc["B_UP", "group_b_effect_median"], 0)
        self.assertGreater(self.summary.loc["A_UP", "delta_median"], 0)
        self.assertLess(self.summary.loc["B_UP", "delta_median"], 0)
        a_up_folds = folds.loc[folds.gene == "A_UP", "effect_group_a_vs_reference"]
        self.assertAlmostEqual(self.summary.loc["A_UP", "group_a_effect_median"],
                               float(a_up_folds.median()), places=14)
        np.testing.assert_allclose(_bh(np.array([.01, .04, .03])), [.03, .04, .04])
        for _, fold in folds.groupby(["cell_state", "held_out_dataset"]):
            expected = _bh(fold.p_group_a_vs_reference.to_numpy())
            np.testing.assert_allclose(fold.fdr_group_a_vs_reference, expected)

    def test_cross_fold_classifications_expression_and_labels(self):
        self.assertEqual(self.status, CapabilityStatus.COMPLETED)
        self.assertEqual(self.summary.loc["A_UP", "group_a_specific_class"], "group_a_specific_up")
        self.assertEqual(self.summary.loc["A_DOWN", "group_a_specific_class"], "group_a_specific_down")
        self.assertEqual(self.summary.loc["B_UP", "group_b_specific_class"], "group_b_specific_up")
        self.assertEqual(self.summary.loc["B_DOWN", "group_b_specific_class"], "group_b_specific_down")
        self.assertTrue(bool(self.summary.loc["A_ENRICH", "group_a_enriched"]))
        self.assertTrue(bool(self.summary.loc["B_ENRICH", "group_b_enriched"]))
        self.assertFalse(bool(self.summary.loc["LOW_EXPR", "group_a_changed"]))
        self.assertFalse(bool(self.summary.loc["RPL_LABEL", "label_worthy"]))
        self.assertIn("RPL_LABEL", self.summary.index)
        self.assertTrue(is_label_worthy_gene("A_UP"))
        self.assertFalse(is_label_worthy_gene("IGHV1"))
        self.assertGreaterEqual(self.summary.loc["A_UP", "group_a_sign_consistency"], .8)

    def test_swap_group_direction_changes_coefficients_and_delta(self):
        swapped = request(group_a="NDMM", group_b="SMM",
                          minimum_group_a_samples=10, minimum_group_b_samples=5)
        meta, expr, observed = load_sample_means(swapped)
        tables, _, _ = calculate_atlas_lodo(meta, expr, observed, swapped)
        summary = tables["gene_summaries"].set_index("gene")
        self.assertAlmostEqual(summary.loc["A_UP", "delta_median"],
                               -self.summary.loc["A_UP", "delta_median"], places=12)
        self.assertAlmostEqual(summary.loc["A_UP", "group_b_effect_median"],
                               self.summary.loc["A_UP", "group_a_effect_median"], places=12)

    def test_skipped_folds_singular_design_and_terminal_status(self):
        strict = replace(self.req, minimum_training_samples_per_group=5)
        tables, warnings, status = calculate_atlas_lodo(self.meta, self.expr, self.observed, strict)
        self.assertEqual(status, CapabilityStatus.NOT_ESTIMABLE)
        self.assertEqual(len(tables["skipped_folds"]), 3)
        self.assertTrue(all("insufficient_training_replication" in x for x in tables["skipped_folds"].reason))
        with self.assertRaisesRegex(ValueError, "singular_or_invalid_design"):
            fit_fold_ols(pd.DataFrame({"G": [1., 2., 3., 4.]}),
                pd.DataFrame({"stage_model_v2": ["NBM"] * 4}), self.req)
        no_eligible = replace(self.req, minimum_group_a_samples=6)
        _, warnings, status = calculate_atlas_lodo(self.meta, self.expr, self.observed, no_eligible)
        self.assertEqual(status, CapabilityStatus.INSUFFICIENT_REPLICATION)
        self.assertTrue(any(w.code == "ATLAS_LODO_INSUFFICIENT_REPLICATION" for w in warnings))

    def test_atlas_summary_and_compatibility_mapping(self):
        atlas = self.tables["atlas_summary"].set_index("cell_state")
        self.assertEqual(int(atlas.loc["EligibleState", "genes_tested"]), len(GENES))
        self.assertEqual(atlas.loc["EligibleState", "reference_group"], "NBM")
        self.assertEqual((self.req.reference_group, self.req.group_a, self.req.group_b), ("NBM", "SMM", "NDMM"))
        self.assertTrue(bool(self.summary.loc["A_ENRICH", "group_a_enriched"]))  # historical smm_enriched
        self.assertTrue(bool(self.summary.loc["B_ENRICH", "group_b_enriched"]))  # historical ndmm_enriched
        self.assertTrue(bool(self.summary.loc["A_UP", "group_a_specific"]))       # historical smm_specific_lm

    def test_runner_cache_provenance_manifest_and_common_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = AnalysisContext((root / "outputs").resolve(), (root / "cache").resolve(), "fixture-v1")
            first = run_atlas_lodo(self.req, context)
            second = run_atlas_lodo(self.req, context)
            self.assertEqual(first.status, CapabilityStatus.COMPLETED)
            self.assertFalse(first.cache_hit); self.assertTrue(second.cache_hit)
            self.assertEqual(first.cache_key, second.cache_key)
            common = first.to_capability_result(); common.validate_required_artifacts(require_exists=True)
            self.assertEqual(common.capability_id, CAPABILITY_ID)
            provenance = json.loads(Path(first.provenance_path).read_text())
            self.assertEqual(provenance["model_formula"], MODEL_FORMULA)
            self.assertEqual(provenance["unit_of_inference"], "biological_sample")
            self.assertEqual(provenance["reference_group"], "NBM")
            self.assertIn("delta_definition", provenance["parameters"])
            self.assertEqual(provenance["parameters"]["minimum_cells_per_sample_state"], 100)
            self.assertEqual(provenance["parameters"]["successful_fold_count"], 3)
            self.assertEqual(provenance["parameters"]["skipped_fold_count"], 0)
            manifest = json.loads(Path(first.cache_manifest_path).read_text())
            self.assertEqual(manifest["completion_status"], "complete")
            self.assertTrue(all(Path(path).exists() for path in manifest["output_files"]))
            changed = run_atlas_lodo(replace(self.req, delta_absolute_cutoff=.30), context)
            swapped = run_atlas_lodo(request(group_a="NDMM", group_b="SMM",
                minimum_group_a_samples=10, minimum_group_b_samples=5), context)
            self.assertNotEqual(first.cache_key, changed.cache_key)
            self.assertNotEqual(first.cache_key, swapped.cache_key)

            changed_input = root / "changed.csv"
            changed_table = pd.read_csv(FIXTURE)
            changed_table.loc[0, "NULL"] += .01
            changed_table.to_csv(changed_input, index=False)
            input_changed = run_atlas_lodo(replace(self.req, source_path=changed_input), context)
            self.assertNotEqual(first.cache_key, input_changed.cache_key)

            terminal_request = replace(self.req, minimum_group_a_samples=6)
            failed1 = run_atlas_lodo(terminal_request, context)
            failed2 = run_atlas_lodo(terminal_request, context)
            self.assertFalse(failed1.cache_hit); self.assertFalse(failed2.cache_hit)
            self.assertIsNone(load_complete_manifest(Path(failed1.cache_manifest_path), failed1.cache_key))

    def test_determinism_spec_and_source_mapping(self):
        tables2, _, _ = calculate_atlas_lodo(self.meta, self.expr, self.observed, self.req)
        pd.testing.assert_frame_equal(self.tables["fold_results"], tables2["fold_results"])
        spec = CAPABILITY_SPECS_BY_ID[CAPABILITY_ID]
        self.assertEqual(spec.unit_of_inference, "biological_sample")
        self.assertEqual(spec.parallelization_axis, "cell_state")
        self.assertIn("15_lodo_precursor_specific_features_9celltypes.ipynb", SOURCE_FILES[0])
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            request(group_a="NBM")


if __name__ == "__main__":
    unittest.main()
