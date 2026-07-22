from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cellstate.evidence import build_evidence_bundle, write_evidence_bundle_atomic
from cellstate.schemas.evidence import (
    EVIDENCE_BUNDLE_SCHEMA_VERSION,
    EvidenceExecutionStatus,
)


class EvidenceBundleTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "question": "Compare NDMM with NBM in CD8 T cells",
            "analysis_type": "pseudobulk_de",
            "cell_type": "CD8 T",
            "group_a": "NDMM",
            "group_b": "NBM",
        }
        self.ready_design = {
            "group_sample_counts": {"NDMM": 4, "NBM": 5},
            "shared_datasets": 2,
            "design_rank": 3,
            "design_columns": 3,
            "full_rank": True,
            "ready": True,
        }

    @staticmethod
    def _write_artifact(root: Path, name: str = "design_assessment.json"):
        (root / name).write_text("{}\n", encoding="utf-8")

    def test_completed_bundle_is_json_safe_and_directional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_artifact(root)
            bundle = build_evidence_bundle(
                state=self.state,
                execution_status=EvidenceExecutionStatus.COMPLETED,
                output_directory=root,
                design_assessment=self.ready_design,
                deterministic_evidence={
                    "integer_like": True,
                    "n_samples": 9,
                    "n_genes": 100,
                },
            )
            payload = bundle.to_dict()
            json.dumps(payload)
            self.assertEqual(payload["schema_version"], EVIDENCE_BUNDLE_SCHEMA_VERSION)
            self.assertEqual(payload["execution_status"], "completed")
            self.assertEqual(
                payload["biological_context"]["contrast_direction"],
                "group_a_minus_group_b",
            )
            self.assertEqual(payload["unit_of_inference"], "biological_sample")
            self.assertEqual(payload["warnings"], [])

    def test_bundle_identity_is_stable_and_direction_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_artifact(root)
            kwargs = dict(
                execution_status="completed",
                output_directory=root,
                design_assessment=self.ready_design,
                deterministic_evidence={"n_samples": 9},
            )
            first = build_evidence_bundle(state=self.state, **kwargs)
            second = build_evidence_bundle(state=self.state, **kwargs)
            reversed_state = {
                **self.state,
                "group_a": "NBM",
                "group_b": "NDMM",
            }
            reversed_bundle = build_evidence_bundle(state=reversed_state, **kwargs)
            self.assertEqual(first.bundle_id, second.bundle_id)
            self.assertNotEqual(first.bundle_id, reversed_bundle.bundle_id)
            self.assertNotEqual(first.created_at_utc, "")

    def test_blocked_bundle_preserves_design_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_artifact(root)
            blocked_design = {
                "group_sample_counts": {"NDMM": 2, "NBM": 4},
                "shared_datasets": 0,
                "design_rank": 2,
                "design_columns": 3,
                "full_rank": False,
                "ready": False,
            }
            bundle = build_evidence_bundle(
                state=self.state,
                execution_status="blocked",
                output_directory=root,
                design_assessment=blocked_design,
            )
            self.assertEqual(bundle.execution_status, EvidenceExecutionStatus.BLOCKED)
            self.assertEqual(
                {warning.code for warning in bundle.warnings},
                {"DESIGN_NOT_READY", "NO_DATASET_OVERLAP", "NON_ESTIMABLE_DESIGN"},
            )
            self.assertTrue(all(warning.severity == "error" for warning in bundle.warnings))
            self.assertEqual(bundle.deterministic_evidence, {})

    def test_cache_restore_metadata_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_artifact(root, "aggregation_summary.json")
            bundle = build_evidence_bundle(
                state=self.state,
                execution_status="completed_from_cache",
                output_directory=root,
                design_assessment=self.ready_design,
                deterministic_evidence={"n_samples": 9},
                cache_key="cache-key",
                cache_directory=root / "cache",
                cache_hit=True,
            )
            self.assertEqual(bundle.cache["cache_key"], "cache-key")
            self.assertTrue(bundle.cache["cache_hit"])

    def test_live_or_tabular_objects_are_rejected(self):
        class FakeDataFrame:
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_artifact(root)
            with self.assertRaisesRegex(TypeError, "JSON-safe deterministic summaries"):
                build_evidence_bundle(
                    state=self.state,
                    execution_status="completed",
                    output_directory=root,
                    design_assessment=self.ready_design,
                    deterministic_evidence={"table": FakeDataFrame()},
                )

    def test_atomic_writer_replaces_existing_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_artifact(root)
            bundle = build_evidence_bundle(
                state=self.state,
                execution_status="completed",
                output_directory=root,
                design_assessment=self.ready_design,
            )
            destination = root / "evidence_bundle.json"
            destination.write_text("incomplete", encoding="utf-8")
            write_evidence_bundle_atomic(bundle, destination)
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(payload["bundle_id"], bundle.bundle_id)
            self.assertFalse(any(root.glob(".evidence_bundle.json.*.tmp")))

    def test_bundle_requires_a_persisted_deterministic_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "no deterministic artifacts"):
                build_evidence_bundle(
                    state=self.state,
                    execution_status="completed",
                    output_directory=Path(tmp),
                    design_assessment=self.ready_design,
                )


if __name__ == "__main__":
    unittest.main()
