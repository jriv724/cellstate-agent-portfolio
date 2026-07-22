from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellstate.data_interchange import (
    build_novartis_cell_metadata,
    clean_sample,
    export_anndata_obs,
    export_matrix_triplet,
    merge_label_donors,
    patient_from_sample,
    qc_filter_counts,
    validate_matrix_triplet,
)


class FakeAnnData:
    def __init__(self):
        self.X = sparse.csr_matrix([[1, 0], [0, 2]])
        self.layers = {"counts": self.X}
        self.obs_names = pd.Index(["c1", "c2"])
        self.var_names = pd.Index(["g1", "g2"])
        self.obs = pd.DataFrame(
            {
                "sample": pd.Categorical(["s1", "s2"]),
                "age_years": ["42", "bad"],
                "stage": ["nan", "NBM"],
            },
            index=self.obs_names,
        )


class DataInterchangeTests(unittest.TestCase):
    def test_obs_export_is_deterministic_and_structured(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_anndata_obs(
                FakeAnnData(), Path(tmp), columns=("sample", "age_years", "stage", "absent")
            )
            self.assertEqual(result.capability_id, "CAP-DATA-001")
            self.assertEqual(result.outputs["metadata"].index.name, "cell_barcode")
            self.assertTrue(pd.isna(result.outputs["metadata"].loc["c1", "stage"]))
            self.assertTrue(pd.isna(result.outputs["metadata"].loc["c2", "age_years"]))
            self.assertEqual(
                {w.code for w in result.warnings},
                {"OBS_COLUMNS_MISSING", "AGE_COERCED_TO_MISSING"},
            )
            second = export_anndata_obs(
                FakeAnnData(), Path(tmp), columns=("sample", "age_years", "stage", "absent")
            )
            self.assertEqual(result.cache_key, second.cache_key)

    def test_obs_export_strict_missing_column_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Missing required"):
                export_anndata_obs(FakeAnnData(), Path(tmp), columns=("absent",), strict=True)

    def test_triplet_round_trip_and_metadata_alignment(self):
        matrix = sparse.csr_matrix([[1, 0], [2, 3]])
        metadata = pd.DataFrame({"sample": ["s2", "s1"]}, index=["c2", "c1"])
        with tempfile.TemporaryDirectory() as tmp:
            result = export_matrix_triplet(
                matrix, ["g1", "g2"], ["c1", "c2"], Path(tmp), metadata=metadata
            )
            triplet = result.outputs["triplet"]
            checked = validate_matrix_triplet(
                triplet.matrix_path, triplet.features_path, triplet.barcodes_path,
                triplet.metadata_path,
            )
            self.assertEqual((checked.n_features, checked.n_cells), (2, 2))
            aligned = pd.read_csv(triplet.metadata_path, sep="\t", index_col=0)
            self.assertEqual(aligned.index.tolist(), ["c1", "c2"])

    def test_triplet_rejects_noninteger_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "integer-like"):
                export_matrix_triplet([[1.5]], ["g"], ["c"], Path(tmp))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                export_matrix_triplet([[1], [2]], ["g", "g"], ["c"], Path(tmp))

    def test_source_sample_cleaning_and_patient_parsing(self):
        self.assertEqual(clean_sample("N11D28 (2%)"), "N11D28")
        self.assertEqual(patient_from_sample("N11D28 (2%)"), "N11")
        self.assertIsNone(patient_from_sample("unknown"))

    def test_novartis_metadata_rebuild(self):
        result = build_novartis_cell_metadata(
            ["AA-1", "BB-1"],
            {"s_id": "S3018", "sample": "N11D28 (2%)", "timepoint": "D28", "source_map": "D28"},
        )
        self.assertEqual(result.index.tolist(), ["AA-1--N11D28", "BB-1--N11D28"])
        self.assertEqual(result["patient_id"].unique().tolist(), ["N11"])

    def test_label_precedence_and_atlas_rescue(self):
        raw = pd.DataFrame(
            {"sample_clean": ["N11D28", "N11D28"]},
            index=["AA-1--N11D28", "BB-1--N11D28"],
        )
        processed = pd.DataFrame(
            {"sample": ["N11D28 (2%)", "N11D28 (2%)"], "predicted_cell_type": ["Naive B cell", ""]},
            index=["AA-1-1-day28_CART", "BB-1-1-day28_CART"],
        )
        atlas = pd.DataFrame(
            {
                "barcode_orig": ["AA-1", "BB-1"], "sample": ["N11D28", "N11D28"],
                "S_id": ["S3018", "S3018"],
                "predicted_cell_type": ["CD56 NK cell", "Memory CD4 T cell"],
            }
        )
        result = merge_label_donors(raw, processed, atlas)
        md = result.outputs["metadata"]
        self.assertEqual(md["final_predicted_cell_type"].tolist(), ["Naive B cell", "Memory CD4 T cell"])
        self.assertEqual(md["clock_celltype"].tolist(), ["B", "CD4T"])

    def test_qc_threshold_boundaries(self):
        matrix = sparse.csr_matrix(
            [
                [150, 150],       # exactly 300 counts, 2 genes
                [100001, 0],      # above maximum
                [299, 0],         # below minimum
            ]
        )
        obs = pd.DataFrame(index=["keep", "high", "low"])
        result = qc_filter_counts(matrix, obs, min_counts=300, min_genes=2)
        self.assertEqual(result.outputs["metadata"].index.tolist(), ["keep"])
        self.assertEqual(result.outputs["keep_mask"].tolist(), [True, False, False])
        self.assertEqual(result.warnings[0].code, "QC_CELLS_REMOVED")


if __name__ == "__main__":
    unittest.main()
