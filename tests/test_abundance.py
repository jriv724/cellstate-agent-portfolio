from pathlib import Path
import json
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cellstate.context import AnalysisContext
from cellstate.nodes.abundance import calculate_sample_abundance, run_cell_state_abundance
from cellstate.schemas.abundance import AbundanceInput


def fixture_metadata() -> pd.DataFrame:
    rows = []
    values = {
        "n1": ("NBM", "d1", ["B", "B", "T", "Plasma"]),
        "n2": ("NBM", "d2", ["B", "T", "T", "T"]),
        "s1": ("SMM", "d1", ["B", "B", "B", "T"]),
        "s2": ("SMM", "d2", ["B", "T", "Plasma", "Plasma"]),
    }
    for sample, (stage, dataset, states) in values.items():
        for state in states:
            rows.append({"sample": sample, "patient": f"p-{sample}", "dataset": dataset,
                         "stage_model_v2": stage, "preserved": state,
                         "macro": "Lymphoid" if state in {"B", "T"} else "Plasma"})
    return pd.DataFrame(rows)


def request_for(path: Path, **overrides) -> AbundanceInput:
    values = {"metadata_path": path, "patient_column": "patient", "stage_order": ("NBM", "SMM"),
              "minimum_samples_per_stage_for_warning": 2}
    values.update(overrides)
    return AbundanceInput(**values)


class AbundanceTests(unittest.TestCase):
    def test_source_fraction_zero_completion_and_stage_summary(self):
        fractions, summary, warnings, report = calculate_sample_abundance(fixture_metadata(), request_for(Path("unused.csv")))
        self.assertEqual(len(fractions), 12)
        self.assertAlmostEqual(fractions.query("sample == 'n1' and preserved == 'B'").iloc[0].fraction, 0.5)
        self.assertEqual(fractions.query("sample == 'n2' and preserved == 'Plasma'").iloc[0].fraction, 0.0)
        self.assertTrue((fractions.groupby("sample").fraction.sum() == 1.0).all())
        nbm_b = summary.query("stage_model_v2 == 'NBM' and preserved == 'B'").iloc[0]
        self.assertAlmostEqual(nbm_b.mean_fraction, 0.375)
        self.assertEqual(nbm_b.n_samples, 2)
        self.assertEqual(report["biological_samples"], 4)
        self.assertNotIn("LOW_STAGE_REPLICATION", {warning.code for warning in warnings})

    def test_plasma_rbc_profile_changes_numerator_and_denominator(self):
        fractions, _, warnings, report = calculate_sample_abundance(
            fixture_metadata(), request_for(Path("unused.csv"), analysis_profile="plasma_rbc_removed"))
        self.assertEqual(set(fractions.preserved), {"B", "T"})
        self.assertAlmostEqual(fractions.query("sample == 'n1' and preserved == 'B'").iloc[0].fraction, 2 / 3)
        self.assertEqual(report["cells_excluded_profile"], 3)
        self.assertIn("PLASMA_RBC_DENOMINATOR_EXCLUSION", {warning.code for warning in warnings})

    def test_macro_taxonomy_is_explicit_and_source_mode_mapped(self):
        fractions, summary, _, _ = calculate_sample_abundance(
            fixture_metadata(), request_for(Path("unused.csv"), macro_column="macro"))
        self.assertIn("macro", fractions)
        self.assertIn("macro", summary)
        self.assertEqual(set(fractions.query("preserved == 'B'").macro), {"Lymphoid"})

    def test_missing_required_column_fails(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            calculate_sample_abundance(fixture_metadata().drop(columns="dataset"), request_for(Path("unused.csv")))

    def test_empty_target_population_fails(self):
        with self.assertRaisesRegex(ValueError, "empty target population"):
            calculate_sample_abundance(fixture_metadata(), request_for(Path("unused.csv"), stage_order=("NDMM",)))

    def test_sample_metadata_conflict_fails(self):
        data = fixture_metadata()
        data.loc[0, "stage_model_v2"] = "SMM"
        with self.assertRaisesRegex(ValueError, "inconsistent within biological sample"):
            calculate_sample_abundance(data, request_for(Path("unused.csv")))

    def test_dataset_confounding_and_low_replication_are_structured(self):
        data = fixture_metadata().query("sample in ['n1', 's2']").copy()
        warnings = calculate_sample_abundance(
            data, request_for(Path("unused.csv"), minimum_samples_per_stage_for_warning=2))[2]
        codes = {warning.code for warning in warnings}
        self.assertIn("LOW_STAGE_REPLICATION", codes)
        self.assertIn("NO_DATASET_OVERLAP_ACROSS_STAGES", codes)
        self.assertIn("DATASET_STAGE_EXCLUSIVE", codes)

    def test_cache_hit_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "obs.csv"
            fixture_metadata().to_csv(source, index=False)
            context = AnalysisContext(root / "outputs", root / "cache", "fixture-atlas-v1")
            first = run_cell_state_abundance(request_for(source), context)
            second = run_cell_state_abundance(request_for(source), context)
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.cache_key, second.cache_key)
            self.assertEqual(first.provenance.unit_of_inference, "biological_sample")
            self.assertIsNone(first.provenance.model_formula)
            self.assertIsNone(first.provenance.reference_group)
            self.assertEqual(first.provenance.covariates, ())
            manifest = json.loads(Path(first.cache_manifest_path).read_text())
            self.assertEqual(manifest["completion_status"], "complete")
            self.assertTrue(all(Path(path).exists() for path in manifest["output_files"]))

    def test_cache_key_changes_with_profile_and_input_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "obs.csv"
            fixture_metadata().to_csv(source, index=False)
            context = AnalysisContext(root / "outputs", root / "cache", "fixture-atlas-v1")
            standard = run_cell_state_abundance(request_for(source), context)
            filtered = run_cell_state_abundance(request_for(source, analysis_profile="plasma_rbc_removed"), context)
            self.assertNotEqual(standard.cache_key, filtered.cache_key)
            fixture_metadata().iloc[:-1].to_csv(source, index=False)
            changed = run_cell_state_abundance(request_for(source), context)
            self.assertNotEqual(standard.cache_key, changed.cache_key)

    def test_deterministic_tables(self):
        request = request_for(Path("unused.csv"))
        first = calculate_sample_abundance(fixture_metadata(), request)[:2]
        second = calculate_sample_abundance(fixture_metadata(), request)[:2]
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])


if __name__ == "__main__":
    unittest.main()
