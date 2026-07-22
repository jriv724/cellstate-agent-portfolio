from pathlib import Path
import tempfile
import unittest

from cellstate.reproducibility.comparison import compare_csv
from cellstate.reproducibility.runner import load_cases, run_case, run_harness

class ReproducibilityHarnessTests(unittest.TestCase):
    def test_committed_python_regressions_match(self):
        cases = tuple(case for case in load_cases() if case.capability_id != "CAP-DESEQ-002")
        with tempfile.TemporaryDirectory() as tmp:
            results = run_harness(Path(tmp), cases)
        self.assertEqual({item.capability_id for item in results}, {"CAP-COMP-001", "CAP-STAT-003"})
        self.assertTrue(all(item.passed for item in results), [item.comparisons for item in results])
        self.assertEqual({item.status for item in results}, {"completed", "completed_with_warnings"})
        self.assertTrue(all(item.manifest_completion_status == "complete" for item in results))

    def test_repeat_run_is_cache_hit_with_identical_key_and_tables(self):
        case = next(item for item in load_cases() if item.capability_id == "CAP-COMP-001")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); first = run_case(case, root); second = run_case(case, root)
        self.assertFalse(first.cache_hit); self.assertTrue(second.cache_hit)
        self.assertEqual(first.cache_key, second.cache_key)
        self.assertTrue(first.passed and second.passed)

    def test_comparison_detects_scientific_table_drift(self):
        case = next(item for item in load_cases() if item.capability_id == "CAP-COMP-001")
        expected = case.expected_artifacts[0]
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / "changed.csv"
            text = expected.expected_path.read_text().replace("n1,NBM,d1,p-n1,B,2,4,0.5", "n1,NBM,d1,p-n1,B,2,4,0.6")
            changed.write_text(text)
            result = compare_csv(changed, expected)
        self.assertFalse(result.matched)

    def test_deseq2_external_failure_is_never_cacheable(self):
        case = next(item for item in load_cases() if item.capability_id == "CAP-DESEQ-002")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); first = run_case(case, root); second = run_case(case, root)
        self.assertTrue(first.passed and second.passed)
        self.assertEqual(first.status, "failed_execution")
        self.assertEqual(first.manifest_completion_status, "failed")
        self.assertFalse(first.cache_hit); self.assertFalse(second.cache_hit)
        self.assertEqual(first.cache_key, second.cache_key)
        self.assertIn("DESEQ2_RUNTIME_FAILURE", first.warning_codes)

    def test_case_metadata_is_fixed_and_scoped(self):
        cases = load_cases()
        self.assertEqual(len(cases), 3)
        self.assertEqual({case.capability_id for case in cases}, {"CAP-COMP-001", "CAP-STAT-003", "CAP-DESEQ-002"})
        self.assertTrue(all(path.exists() for case in cases for path in case.input_paths))
        self.assertTrue(all(item.expected_path.exists() for case in cases for item in case.expected_artifacts))

if __name__ == "__main__": unittest.main()
