from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cellstate.context import AnalysisContext
from cellstate.nodes.progression import (
    calculate_cross_sectional_progression, calculate_longitudinal_kinetics,
    calculate_paired_progression, run_longitudinal_kinetics, run_paired_progression,
)
from cellstate.schemas.progression import (
    CrossSectionalProgressionInput, LongitudinalKineticsInput, PairedProgressionInput,
)


def cross_data():
    rows = []
    for state, sign in (("B", 1), ("T", -1)):
        for stage_index, stage in enumerate(("NBM", "SMM", "NDMM")):
            for replicate in range(3):
                rows.append({"sample": f"{stage}-{replicate}", "dataset": "d1", "stage": stage,
                             "state": state, "fraction": 0.2 + sign * 0.03 * stage_index + replicate * 0.001})
    return pd.DataFrame(rows)


def cross_request(path=Path("unused.csv"), **overrides):
    values = dict(table_path=path, stage_column="stage", stratum_column="state",
                  stage_order=("NBM", "SMM", "NDMM"), minimum_samples_per_stage=3,
                  minimum_stages_passing=3, minimum_mean_outcome=0)
    values.update(overrides)
    return CrossSectionalProgressionInput(**values)


def paired_data():
    rows = []
    for cell_class, offset in (("B", 0), ("T", 10)):
        for patient in range(4):
            rows.extend([
                {"sample": f"p{patient}-n", "patient": f"p{patient}", "dataset": "d1", "class": cell_class,
                 "stage": "NDMM", "age": 40 + offset + patient},
                {"sample": f"p{patient}-r", "patient": f"p{patient}", "dataset": "d1", "class": cell_class,
                 "stage": "RRMM", "age": 42 + offset + patient},
            ])
    return pd.DataFrame(rows)


def paired_request(path=Path("unused.tsv"), **overrides):
    values = dict(table_path=path, patient_column="patient", class_column="class", stage_column="stage",
                  outcome_column="age", dataset_column="dataset")
    values.update(overrides)
    return PairedProgressionInput(**values)


def longitudinal_data():
    rows = []
    for cell_class, offset in (("B", 0), ("T", 5)):
        for patient in range(4):
            for timepoint, change in (("S", 0), ("D28", 1), ("M3", 3)):
                rows.append({"patient": f"p{patient}", "dataset": "trial", "class": cell_class,
                             "time": timepoint, "age": 30 + offset + patient + change})
    return pd.DataFrame(rows)


def longitudinal_request(path=Path("unused.tsv"), **overrides):
    values = dict(table_path=path, patient_column="patient", class_column="class",
                  timepoint_column="time", outcome_column="age", dataset_column="dataset")
    values.update(overrides)
    return LongitudinalKineticsInput(**values)


class ProgressionTests(unittest.TestCase):
    def test_valid_cross_sectional_progression(self):
        analysis, results, warnings = calculate_cross_sectional_progression(cross_data(), cross_request())
        self.assertEqual(len(analysis), 18)
        self.assertGreater(results.query("state == 'B'").iloc[0].slope, 0)
        self.assertLess(results.query("state == 'T'").iloc[0].slope, 0)
        self.assertTrue(results.eligible.all())
        self.assertFalse(any(w.severity.value == "error" for w in warnings))

    def test_cross_sectional_direction_and_stage_order_are_sensitive(self):
        forward = calculate_cross_sectional_progression(cross_data(), cross_request())[1]
        reverse = calculate_cross_sectional_progression(
            cross_data(), cross_request(stage_order=("NDMM", "SMM", "NBM")))[1]
        self.assertAlmostEqual(forward.query("state == 'B'").iloc[0].slope,
                               -reverse.query("state == 'B'").iloc[0].slope)

    def test_cross_sectional_dataset_stage_confounding(self):
        data = cross_data()
        data["dataset"] = data.stage.map({"NBM": "d1", "SMM": "d2", "NDMM": "d3"})
        warnings = calculate_cross_sectional_progression(data, cross_request())[-1]
        self.assertIn("DATASET_STAGE_CONFOUNDING", {w.code for w in warnings})
        self.assertIn("NO_DATASET_STAGE_OVERLAP", {w.code for w in warnings})

    def test_valid_paired_progression_and_direction(self):
        _, collapsed, pairs, results, _ = calculate_paired_progression(paired_data(), paired_request())
        self.assertEqual(len(collapsed), 16)
        self.assertTrue((pairs.delta_later_minus_earlier == 2).all())
        self.assertTrue((results.n_pairs == 4).all())
        reversed_result = calculate_paired_progression(
            paired_data(), paired_request(earlier_stage="RRMM", later_stage="NDMM"))[2]
        self.assertTrue((reversed_result.delta_later_minus_earlier == -2).all())

    def test_paired_source_mean_collapse_and_invalid_duplicate_sample(self):
        data = paired_data()
        extra = data.iloc[[0]].copy()
        extra["sample"] = "p0-n-second"
        extra["age"] = 44
        _, collapsed, _, _, warnings = calculate_paired_progression(pd.concat([data, extra]), paired_request())
        row = collapsed.query("patient == 'p0' and `class` == 'B' and stage == 'NDMM'").iloc[0]
        self.assertEqual(row.outcome, 42)
        self.assertIn("MULTIPLE_SAMPLES_MEAN_COLLAPSED", {w.code for w in warnings})
        with self.assertRaisesRegex(ValueError, "one source row"):
            calculate_paired_progression(pd.concat([data, data.iloc[[0]]]), paired_request())

    def test_incomplete_and_insufficient_pairs(self):
        data = paired_data().query("not (patient == 'p3' and stage == 'RRMM')")
        _, _, _, results, warnings = calculate_paired_progression(
            data, paired_request(minimum_pairs_for_test=4))
        self.assertTrue(results.wilcox_p.isna().all())
        codes = {w.code for w in warnings}
        self.assertIn("INCOMPLETE_PAIRS_DROPPED", codes)
        self.assertIn("INSUFFICIENT_PAIRED_REPLICATION", codes)

    def test_valid_longitudinal_kinetics(self):
        analysis, deltas, results, warnings = calculate_longitudinal_kinetics(
            longitudinal_data(), longitudinal_request())
        self.assertEqual(len(analysis), 24)
        self.assertTrue((deltas.delta_M3_minus_S == 3).all())
        self.assertEqual(set(zip(results.earlier, results.later)), {("S", "D28"), ("S", "M3"), ("D28", "M3")})
        self.assertFalse(any(w.severity.value == "error" for w in warnings))

    def test_longitudinal_duplicate_and_incomplete_rows(self):
        data = longitudinal_data()
        with self.assertRaisesRegex(ValueError, "identify one longitudinal row"):
            calculate_longitudinal_kinetics(pd.concat([data, data.iloc[[0]]]), longitudinal_request())
        incomplete = data.query("not (patient == 'p3' and time == 'M3')")
        warnings = calculate_longitudinal_kinetics(incomplete, longitudinal_request())[-1]
        self.assertIn("INCOMPLETE_LONGITUDINAL_PAIRS", {w.code for w in warnings})

    def test_incorrect_timepoint_order_and_nonestimable_design(self):
        with self.assertRaisesRegex(ValueError, "follow declared timepoint order"):
            longitudinal_request(contrasts=(("M3", "S"),))
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            calculate_longitudinal_kinetics(
                longitudinal_data().query("time != 'M3'"), longitudinal_request())

    def test_missing_required_columns(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            calculate_cross_sectional_progression(cross_data().drop(columns="fraction"), cross_request())
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            calculate_paired_progression(paired_data().drop(columns="patient"), paired_request())

    def test_cache_key_provenance_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "paired.tsv"
            paired_data().to_csv(source, sep="\t", index=False)
            context = AnalysisContext(root / "out", root / "cache", "progression-fixture")
            first = run_paired_progression(paired_request(source), context)
            second = run_paired_progression(paired_request(source), context)
            reverse = run_paired_progression(
                paired_request(source, earlier_stage="RRMM", later_stage="NDMM"), context)
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertNotEqual(first.cache_key, reverse.cache_key)
            self.assertEqual(first.provenance.reference_group, "NDMM")
            self.assertEqual(first.provenance.covariates, ())
            self.assertIsNone(first.provenance.random_seed)
            self.assertEqual(json.loads(Path(first.cache_manifest_path).read_text())["completion_status"], "complete")

    def test_paired_and_longitudinal_capability_cache_and_provenance_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired_source = root / "paired.tsv"
            longitudinal_source = root / "longitudinal.tsv"
            paired_data().to_csv(paired_source, sep="\t", index=False)
            longitudinal_data().to_csv(longitudinal_source, sep="\t", index=False)
            context = AnalysisContext(root / "out", root / "cache", "progression-namespaces")
            paired = run_paired_progression(paired_request(paired_source), context)
            longitudinal = run_longitudinal_kinetics(longitudinal_request(longitudinal_source), context)
            longitudinal_hit = run_longitudinal_kinetics(longitudinal_request(longitudinal_source), context)
            self.assertEqual(paired.capability_id, "CAP-STAT-003")
            self.assertEqual(longitudinal.capability_id, "CAP-STAT-004")
            self.assertEqual(paired.provenance.capability_id, "CAP-STAT-003")
            self.assertEqual(longitudinal.provenance.capability_id, "CAP-STAT-004")
            self.assertEqual(Path(paired.cache_manifest_path).parent.parent.name, "cap-stat-003")
            self.assertEqual(Path(longitudinal.cache_manifest_path).parent.parent.name, "cap-stat-004")
            self.assertNotEqual(paired.cache_key, longitudinal.cache_key)
            self.assertTrue(longitudinal_hit.cache_hit)
            self.assertEqual(longitudinal.cache_key, longitudinal_hit.cache_key)


if __name__ == "__main__":
    unittest.main()
