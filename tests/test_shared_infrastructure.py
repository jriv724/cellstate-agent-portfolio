from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cellstate.cache import (
    build_cache_manifest,
    load_complete_manifest,
    make_cache_key,
    write_manifest_atomic,
)
from cellstate.context import AnalysisContext
from cellstate.provenance import build_provenance, write_provenance_atomic
from cellstate.schemas.common import StructuredWarning, WarningSeverity
from cellstate.utilities.design import ordered_indicator_design
from cellstate.validation.estimability import validate_design_estimability
from cellstate.validation.metadata import validate_required_columns, validate_sample_metadata
from cellstate.validation.replication import validate_biological_replication


class SharedInfrastructureTests(unittest.TestCase):
    def test_directional_cache_key(self):
        common = dict(
            capability_id="CAP-X",
            node_version="1.0.0",
            cache_schema_version=1,
            dataset_signature="dataset-1",
        )
        ab = make_cache_key(**common, parameters={"groups": ["A", "B"], "reference": "A"})
        ba = make_cache_key(**common, parameters={"groups": ["B", "A"], "reference": "B"})
        self.assertNotEqual(ab, ba)
        self.assertEqual(ab, make_cache_key(**common, parameters={"groups": ["A", "B"], "reference": "A"}))

    def test_cache_manifest_requires_complete_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "result.tsv"
            output.write_text("x\n", encoding="utf-8")
            manifest = build_cache_manifest(
                cache_key="key", capability_id="CAP-X", node_version="1.0.0",
                cache_schema_version=1, input_signature="input",
                source_dataset_signature="dataset", parameters={},
                output_files=[str(output)], completion_status="complete",
                warnings=[], software_versions={"python": "test"},
            )
            path = root / "manifest.json"
            write_manifest_atomic(manifest, path)
            self.assertIsNotNone(load_complete_manifest(path, "key"))
            output.unlink()
            self.assertIsNone(load_complete_manifest(path, "key"))

    def test_incomplete_manifest_is_not_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            manifest = build_cache_manifest(
                cache_key="key", capability_id="CAP-X", node_version="1.0.0",
                cache_schema_version=1, input_signature="input",
                source_dataset_signature="dataset", parameters={},
                output_files=[], completion_status="incomplete",
                warnings=[], software_versions={},
            )
            write_manifest_atomic(manifest, path)
            self.assertIsNone(load_complete_manifest(path, "key"))

    def test_context_requires_absolute_paths_and_signature(self):
        with self.assertRaises(ValueError):
            AnalysisContext(Path("relative"), Path("/tmp/cache"), "dataset")
        context = AnalysisContext(Path("/tmp/out"), Path("/tmp/cache"), "dataset")
        self.assertEqual(context.capability_output_dir("CAP-X"), Path("/tmp/out/cap-x"))

    def test_metadata_validation(self):
        table = pd.DataFrame({"sample": ["s1", "s1"], "dataset": ["d1", "d2"]})
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_sample_metadata(table, sample_column="sample", dataset_column="dataset")
        with self.assertRaisesRegex(ValueError, "missing required"):
            validate_required_columns(table, ["group"])

    def test_replication_and_dataset_confounding_warning(self):
        table = pd.DataFrame({
            "sample": ["a1", "a2", "b1", "b2"],
            "group": ["A", "A", "B", "B"],
            "dataset": ["d1", "d1", "d2", "d2"],
        })
        warnings = validate_biological_replication(
            table, sample_column="sample", group_column="group",
            groups=["A", "B"], minimum_samples_per_group=2,
            dataset_column="dataset",
        )
        self.assertEqual(
            {warning.code for warning in warnings},
            {"DATASET_GROUP_CONFOUNDING", "NO_DATASET_OVERLAP"},
        )

    def test_replication_rejects_identical_groups_and_low_n(self):
        table = pd.DataFrame({"sample": ["s1", "s2"], "group": ["A", "B"]})
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_biological_replication(
                table, sample_column="sample", group_column="group",
                groups=["A", "A"], minimum_samples_per_group=1,
            )
        with self.assertRaisesRegex(ValueError, "too few"):
            validate_biological_replication(
                table, sample_column="sample", group_column="group",
                groups=["A", "B"], minimum_samples_per_group=2,
            )

    def test_design_direction_reference_and_estimability(self):
        matrix, names = ordered_indicator_design(
            ["NBM", "SMM", "NDMM", "NBM"],
            levels=["NBM", "SMM", "NDMM"], reference_level="NBM",
        )
        self.assertEqual(names, ("Intercept", "group[SMM]", "group[NDMM]"))
        self.assertEqual(matrix[0].tolist(), [1.0, 0.0, 0.0])
        result = validate_design_estimability(matrix, term_names=names)
        self.assertEqual(result["rank"], 3)
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            validate_design_estimability(
                np.column_stack([matrix, matrix[:, 1]]),
                term_names=(*names, "duplicate"),
            )

    def test_atomic_provenance_contains_required_fields(self):
        warning = StructuredWarning("TEST", "test warning", WarningSeverity.INFO)
        provenance = build_provenance(
            capability_id="CAP-X", node_version="1.0.0", cache_schema_version=1,
            source_files=["scripts/source.py"], source_locations=["function"],
            input_dataset_signature="dataset", parameters={"x": 1},
            model_formula="y ~ group", reference_group="A",
            covariates=["age"], unit_of_inference="sample", random_seed=123,
            software_versions={"python": "test"}, output_paths=["result.tsv"],
            warnings=[warning],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "provenance.json"
            write_provenance_atomic(provenance, path)
            value = json.loads(path.read_text())
            self.assertEqual(value["unit_of_inference"], "sample")
            self.assertEqual(value["warnings"][0]["code"], "TEST")


if __name__ == "__main__":
    unittest.main()
