from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cellstate.context import AnalysisContext
from cellstate.nodes.pseudobulk_de import (
    construct_raw_pseudobulk, run_deseq2_differential_expression,
    run_pseudobulk_construction, validate_deseq2_inputs,
)
from cellstate.schemas.pseudobulk_de import DifferentialExpressionInput, PseudobulkInput


def fixture_tables():
    counts, metadata = [], []
    for stage in ("NBM", "SMM"):
        for replicate in range(3):
            sample = f"{stage}-{replicate}"
            for cell in range(2):
                cell_id = f"{sample}-c{cell}"
                counts.append({"cell_id": cell_id, "g1": 1 + (stage == "SMM"), "g2": cell, "zero": 0})
                metadata.append({"cell_id": cell_id, "sample": sample, "patient": f"p-{sample}",
                                 "dataset": "d1", "stage": stage, "state": "B"})
    return pd.DataFrame(counts), pd.DataFrame(metadata)


def request(counts=Path("counts.csv"), metadata=Path("metadata.csv"), **overrides):
    values = dict(raw_counts_path=counts, metadata_path=metadata, patient_column="patient",
                  stage_column="stage", cell_state_column="state", stages=("NBM", "SMM"),
                  cell_states=("B",), minimum_cells_per_sample_state=2)
    values.update(overrides)
    return PseudobulkInput(**values)


class PseudobulkTests(unittest.TestCase):
    def test_valid_raw_count_pseudobulk_construction(self):
        counts, metadata = fixture_tables()
        matrices, samples, qc, warnings = construct_raw_pseudobulk(counts, metadata, request())
        matrix = matrices["B"]
        self.assertEqual(matrix.shape, (3, 6))
        self.assertEqual(matrix.loc["g1", "NBM-0"], 2)
        self.assertEqual(matrix.loc["g1", "SMM-0"], 4)
        self.assertTrue((samples.n_cells == 2).all())
        self.assertTrue(qc.retained.all())
        self.assertIn("ALL_ZERO_GENES", {warning.code for warning in warnings})

    def test_missing_metadata_and_raw_count_file(self):
        counts, metadata = fixture_tables()
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            construct_raw_pseudobulk(counts, metadata.drop(columns="stage"), request())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError, "missing raw"):
                run_pseudobulk_construction(request(root / "absent.csv", root / "meta.csv"),
                                            AnalysisContext(root / "out", root / "cache", "fixture"))

    def test_noninteger_negative_and_missing_counts_fail(self):
        counts, metadata = fixture_tables()
        noninteger = counts.copy()
        noninteger["g1"] = noninteger["g1"].astype(float)
        noninteger.loc[0, "g1"] = 1.5
        with self.assertRaisesRegex(ValueError, "integer"):
            construct_raw_pseudobulk(noninteger, metadata, request())
        negative = counts.copy()
        negative.loc[0, "g1"] = -1
        with self.assertRaisesRegex(ValueError, "negative"):
            construct_raw_pseudobulk(negative, metadata, request())
        missing = counts.copy()
        missing.loc[0, "g1"] = None
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            construct_raw_pseudobulk(missing, metadata, request())

    def test_duplicate_and_inconsistent_metadata_fail(self):
        counts, metadata = fixture_tables()
        duplicate = pd.concat([metadata, metadata.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "cell identifiers"):
            construct_raw_pseudobulk(counts, duplicate, request())
        inconsistent = metadata.copy()
        inconsistent.loc[0, "dataset"] = "d2"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            construct_raw_pseudobulk(counts, inconsistent, request())

    def test_empty_target_and_empty_after_cell_filter(self):
        counts, metadata = fixture_tables()
        with self.assertRaisesRegex(ValueError, "empty target population"):
            construct_raw_pseudobulk(counts, metadata, request(cell_states=("T",)))
        with self.assertRaisesRegex(ValueError, "empty pseudobulk"):
            construct_raw_pseudobulk(counts, metadata, request(minimum_cells_per_sample_state=3))

    def test_low_cell_qc_and_low_library_warning(self):
        counts, metadata = fixture_tables()
        counts.loc[:, ["g1", "g2", "zero"]] = 0
        matrices, _, qc, warnings = construct_raw_pseudobulk(
            counts, metadata, request(minimum_cells_per_sample_state=1, minimum_library_size_warning=1))
        self.assertEqual(int(matrices["B"].to_numpy().sum()), 0)
        self.assertTrue(qc.retained.all())
        codes = {warning.code for warning in warnings}
        self.assertIn("LOW_LIBRARY_SIZE", codes)
        self.assertIn("ALL_ZERO_GENES", codes)
        original_counts, original_metadata = fixture_tables()
        drop_cell = original_metadata.query("sample == 'NBM-0'").iloc[0].cell_id
        original_counts = original_counts.query("cell_id != @drop_cell")
        original_metadata = original_metadata.query("cell_id != @drop_cell")
        _, _, low_qc, _ = construct_raw_pseudobulk(original_counts, original_metadata, request())
        self.assertFalse(low_qc.query("sample == 'NBM-0'").iloc[0].retained)
        self.assertEqual(low_qc.query("sample == 'NBM-0'").iloc[0].reason, "below_minimum_cells")

    def test_dataset_group_confounding_warning(self):
        counts, metadata = fixture_tables()
        metadata["dataset"] = metadata.stage.map({"NBM": "d1", "SMM": "d2"})
        warnings = construct_raw_pseudobulk(counts, metadata, request())[-1]
        self.assertIn("DATASET_GROUP_CONFOUNDING", {warning.code for warning in warnings})
        self.assertIn("NO_DATASET_GROUP_OVERLAP", {warning.code for warning in warnings})

    def test_deterministic_results(self):
        counts, metadata = fixture_tables()
        first = construct_raw_pseudobulk(counts, metadata, request())
        second = construct_raw_pseudobulk(counts, metadata, request())
        pd.testing.assert_frame_equal(first[0]["B"], second[0]["B"])
        pd.testing.assert_frame_equal(first[1], second[1])
        pd.testing.assert_frame_equal(first[2], second[2])

    def test_cache_key_and_provenance_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counts_path, metadata_path = root / "counts.csv", root / "metadata.csv"
            counts, metadata = fixture_tables()
            counts.to_csv(counts_path, index=False)
            metadata.to_csv(metadata_path, index=False)
            context = AnalysisContext(root / "out", root / "cache", "raw-count-fixture")
            first = run_pseudobulk_construction(request(counts_path, metadata_path), context)
            second = run_pseudobulk_construction(request(counts_path, metadata_path), context)
            changed = run_pseudobulk_construction(
                request(counts_path, metadata_path, minimum_cells_per_sample_state=1), context)
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertNotEqual(first.cache_key, changed.cache_key)
            self.assertEqual(first.provenance.capability_id, "CAP-DESEQ-001")
            self.assertEqual(first.provenance.unit_of_inference, "biological_sample")
            self.assertIsNone(first.provenance.model_formula)
            self.assertTrue(all(Path(path).exists() for path in first.count_matrix_paths))

    def _de_fixture(self, root):
        samples = [f"{stage}{i}" for stage in ("NBM", "SMM", "NDMM") for i in range(3)]
        counts = pd.DataFrame({sample: [20 + (sample.startswith("SMM") * 20), 11, 1] for sample in samples},
                              index=["effect", "kept", "filtered"])
        metadata = pd.DataFrame({"sample": samples, "dataset": ["d1", "d2", "d1"] * 3,
                                 "stage_model_v2": [stage for stage in ("NBM", "SMM", "NDMM") for _ in range(3)],
                                 "cell_state": "B"})
        counts_path, metadata_path = root / "pb.csv", root / "meta.csv"
        counts.to_csv(counts_path); metadata.to_csv(metadata_path, index=False)
        return counts, metadata, DifferentialExpressionInput(counts_path, metadata_path, "B")

    def test_deseq2_contract_direction_filter_and_estimability(self):
        with tempfile.TemporaryDirectory() as tmp:
            counts, metadata, de_request = self._de_fixture(Path(tmp))
            aligned, filtered, warnings, estimability = validate_deseq2_inputs(counts, metadata, de_request)
            self.assertEqual(list(filtered.index), ["effect", "kept"])
            self.assertEqual(estimability["rank"], estimability["columns"])
            self.assertEqual(de_request.contrasts, (("SMM", "NBM"), ("NDMM", "NBM")))
            with self.assertRaisesRegex(ValueError, "approved directional"):
                DifferentialExpressionInput(de_request.count_matrix_path, de_request.sample_metadata_path, "B",
                                            contrasts=(("NBM", "SMM"), ("NBM", "NDMM")))

    def test_deseq2_replication_alignment_and_confounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            counts, metadata, de_request = self._de_fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "misaligned"):
                validate_deseq2_inputs(counts.iloc[:, :-1], metadata, de_request)
            with self.assertRaisesRegex(ValueError, "insufficient biological replication"):
                validate_deseq2_inputs(counts.iloc[:, :-1], metadata.iloc[:-1], de_request)
            confounded = metadata.copy()
            confounded["dataset"] = confounded["stage_model_v2"]
            _, _, warnings, estimability = validate_deseq2_inputs(counts, confounded, de_request)
            self.assertIsNone(estimability)
            self.assertEqual(warnings[0].code, "NON_ESTIMABLE_ADJUSTED_DESIGN")
            self.assertEqual(warnings[0].severity.value, "error")

    def test_deseq2_wrapper_cache_provenance_and_apeglm_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, de_request = self._de_fixture(root)
            context = AnalysisContext(root / "out", root / "cache", "de-fixture")
            def fake_r(command, **kwargs):
                output = Path(command[4]); output.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"gene": ["effect"], "log2FoldChange": [1.0], "padj": [0.01]}).to_csv(output / "unshrunk_SMM_vs_NBM.csv", index=False)
                pd.DataFrame({"gene": ["effect"], "log2FoldChange": [0.5], "padj": [0.02]}).to_csv(output / "unshrunk_NDMM_vs_NBM.csv", index=False)
                pd.DataFrame({"metric": ["status"], "value": ["complete"]}).to_csv(output / "model_qc.csv", index=False)
                return object()
            with patch("cellstate.nodes.pseudobulk_de.subprocess.run", side_effect=fake_r):
                first = run_deseq2_differential_expression(de_request, context)
            second = run_deseq2_differential_expression(de_request, context)
            self.assertFalse(first.cache_hit); self.assertTrue(second.cache_hit)
            self.assertEqual(len(first.result_paths), 2); self.assertEqual(first.shrinkage_paths, ())
            self.assertIn("APeglm_UNAVAILABLE", {warning.code for warning in first.warnings})
            self.assertEqual(first.provenance.capability_id, "CAP-DESEQ-002")
            self.assertEqual(first.provenance.model_formula, "~ dataset + stage")
            self.assertEqual(first.provenance.reference_group, "NBM")


if __name__ == "__main__":
    unittest.main()
