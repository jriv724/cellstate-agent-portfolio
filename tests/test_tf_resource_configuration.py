from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from cellstate.app.capability_registry import build_capability_registry
from cellstate.app.tf_resource_validation import validate_tf_resource


def write_resource(path: Path, *, dorothea: bool = False) -> None:
    table = pd.DataFrame({
        "source": ["TF1"] * 5,
        "target": [f"G{i}" for i in range(5)],
        "weight": [1, -1, 1, -1, 1],
        "organism": ["human"] * 5,
    })
    if dorothea:
        table["confidence"] = ["A", "B", "C", "D", "A"]
    table.to_csv(path, sep="\t", index=False)


class TFResourceConfigurationTests(unittest.TestCase):
    def test_missing_and_nonexistent_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            row = build_capability_registry()["CAP-TF-002"]
        self.assertEqual(row.status, "connected but configuration required")
        self.assertFalse(row.executable)
        with patch.dict(os.environ, {
            "CELLSTATE_DOROTHEA_PATH": "/missing/d.tsv",
            "CELLSTATE_COLLECTRI_PATH": "/missing/c.tsv",
        }, clear=True):
            row = build_capability_registry()["CAP-TF-002"]
        self.assertEqual(row.status, "connected but configuration required")

    def test_valid_malformed_weight_conflict_and_confidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dorothea, collectri = root / "d.tsv", root / "c.tsv"
            write_resource(dorothea, dorothea=True)
            write_resource(collectri)
            d = validate_tf_resource(dorothea, "DoRothEA")
            c = validate_tf_resource(collectri, "CollecTRI")
            self.assertTrue(d.valid and c.valid)
            self.assertEqual(d.confidence_levels, ("A", "B", "C", "D"))
            self.assertEqual(d.normalized_edge_count, 4)
            with patch.dict(os.environ, {
                "CELLSTATE_DOROTHEA_PATH": str(dorothea),
                "CELLSTATE_COLLECTRI_PATH": str(collectri),
            }, clear=True):
                row = build_capability_registry()["CAP-TF-002"]
            self.assertEqual(row.status, "connected")
            self.assertTrue(row.executable)

            malformed = root / "malformed.tsv"
            pd.DataFrame({"source": ["TF"], "weight": [1]}).to_csv(
                malformed, sep="\t", index=False)
            self.assertIn("missing", validate_tf_resource(
                malformed, "CollecTRI").error)
            nonnumeric = root / "nonnumeric.tsv"
            pd.DataFrame({"source": ["TF"], "target": ["G"],
                          "weight": ["bad"]}).to_csv(
                              nonnumeric, sep="\t", index=False)
            invalid = validate_tf_resource(nonnumeric, "CollecTRI")
            self.assertFalse(invalid.valid)
            self.assertEqual(invalid.invalid_or_nonnumeric_weights, 1)
            conflict = root / "conflict.tsv"
            pd.DataFrame({"source": ["TF", "TF"], "target": ["G", "G"],
                          "weight": [1, -1]}).to_csv(
                              conflict, sep="\t", index=False)
            invalid = validate_tf_resource(conflict, "CollecTRI")
            self.assertFalse(invalid.valid)
            self.assertEqual(invalid.conflicting_edges, 1)

    def test_launcher_loads_local_environment_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / "resources.env"
            env_file.write_text(
                "CELLSTATE_DOROTHEA_PATH=/validated/dorothea.tsv\n"
                "CELLSTATE_COLLECTRI_PATH=/validated/collectri.tsv\n")
            key = root / "key"
            key.write_text("test-key\n")
            capture = root / "environment.txt"
            python = root / "python"
            python.write_text("#!/bin/sh\nenv > \"$CAPTURE_ENV\"\n")
            python.chmod(0o700)
            environment = {
                **os.environ,
                "CELLSTATE_ENV_FILE": str(env_file),
                "CELLSTATE_OPENAI_KEY_FILE": str(key),
                "CELLSTATE_PYTHON_BIN": str(python),
                "CAPTURE_ENV": str(capture),
            }
            subprocess.run(["bash", "run-cellstate-agent.sh"], check=True,
                           env=environment, capture_output=True, text=True)
            loaded = capture.read_text()
            self.assertIn("CELLSTATE_DOROTHEA_PATH=/validated/dorothea.tsv", loaded)
            self.assertIn("CELLSTATE_COLLECTRI_PATH=/validated/collectri.tsv", loaded)


if __name__ == "__main__":
    unittest.main()
