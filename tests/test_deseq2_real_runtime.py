"""Conditional numerical regression through the real CAP-DESEQ-002 R worker."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import pandas as pd
import cellstate

from cellstate.context import AnalysisContext
from cellstate.nodes.deseq2 import run_deseq2_differential_expression
from cellstate.schemas.deseq2 import DifferentialExpressionInput

RSCRIPT = Path("/usr/local/bin/Rscript")
FIXTURE_ROOT = Path(cellstate.__file__).resolve().parent / "reproducibility" / "staging" / "CAP-DESEQ-002"

def _runtime_skip_reason() -> str | None:
    if not RSCRIPT.is_file() or not RSCRIPT.stat().st_mode & 0o111:
        return f"real DESeq2 regression skipped: {RSCRIPT} is not executable"
    probe = subprocess.run([str(RSCRIPT), "-e", "quit(status=if(requireNamespace('DESeq2',quietly=TRUE)) 0 else 2)"],
                           capture_output=True, text=True)
    if probe.returncode:
        detail = (probe.stderr or probe.stdout).strip().replace("\n", " ")
        return f"real DESeq2 regression skipped: R/DESeq2 prerequisite failed ({detail})"
    return None

@unittest.skipIf(_runtime_skip_reason() is not None, _runtime_skip_reason() or "")
class RealDESeq2RegressionTests(unittest.TestCase):
    def test_real_worker_numerics_provenance_and_cache_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counts_path = root / "counts.csv"; metadata_path = root / "metadata.csv"
            shutil.copyfile(FIXTURE_ROOT / "fixture_counts.csv", counts_path)
            shutil.copyfile(FIXTURE_ROOT / "fixture_metadata.csv", metadata_path)
            context = AnalysisContext(root / "outputs", root / "cache", "real-deseq2-regression-v1")
            request = DifferentialExpressionInput(counts_path, metadata_path, "B", rscript_path=RSCRIPT)

            first = run_deseq2_differential_expression(request, context)
            envelope = first.to_capability_result()
            self.assertEqual(envelope.status.value, "completed_with_warnings")
            self.assertEqual({warning.code for warning in first.warnings}, {"APeglm_UNAVAILABLE"})
            self.assertEqual(len(first.result_paths), 2)
            self.assertEqual(first.shrinkage_paths, ())
            self.assertTrue(Path(first.qc_path).exists())

            tables = {Path(path).name: pd.read_csv(path) for path in first.result_paths}
            smm = tables["unshrunk_SMM_vs_NBM.csv"].set_index("gene")
            ndmm = tables["unshrunk_NDMM_vs_NBM.csv"].set_index("gene")
            self.assertGreater(smm.loc["SMM_effect", "log2FoldChange"], 0)
            self.assertGreater(ndmm.loc["NDMM_effect", "log2FoldChange"], 0)
            self.assertLess(abs(smm.loc["null_gene", "log2FoldChange"]), 0.5)
            self.assertLess(abs(ndmm.loc["null_gene", "log2FoldChange"]), 0.5)
            self.assertNotIn("low_count_filtered", smm.index)
            self.assertNotIn("low_count_filtered", ndmm.index)
            for table in tables.values():
                self.assertTrue(table["padj"].dropna().between(0, 1).all())

            provenance = json.loads(Path(first.provenance_path).read_text())
            self.assertEqual(provenance["capability_id"], "CAP-DESEQ-002")
            self.assertEqual(provenance["model_formula"], "~ dataset + stage")
            self.assertEqual(provenance["reference_group"], "NBM")
            self.assertEqual(provenance["covariates"], ["dataset"])
            self.assertEqual(provenance["unit_of_inference"], "biological_sample")
            self.assertEqual(provenance["parameters"]["contrasts"], [["SMM", "NBM"], ["NDMM", "NBM"]])
            self.assertEqual(provenance["parameters"]["genes_after_prefilter"], 39)
            self.assertEqual(provenance["parameters"]["rscript_path"], str(RSCRIPT))

            manifest = json.loads(Path(first.cache_manifest_path).read_text())
            self.assertEqual(manifest["completion_status"], "complete")
            self.assertTrue(all(Path(path).exists() for path in manifest["output_files"]))
            second = run_deseq2_differential_expression(request, context)
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.cache_key, second.cache_key)

            changed_counts = pd.read_csv(counts_path, index_col=0)
            changed_counts.loc["SMM_effect", "SMM1"] += 1
            changed_path = root / "counts_changed.csv"; changed_counts.to_csv(changed_path)
            changed = run_deseq2_differential_expression(
                DifferentialExpressionInput(changed_path, metadata_path, "B", rscript_path=RSCRIPT), context)
            self.assertNotEqual(first.cache_key, changed.cache_key)

            missing_r = root / "missing-Rscript"
            failed_request = DifferentialExpressionInput(counts_path, metadata_path, "B", rscript_path=missing_r)
            failed_first = run_deseq2_differential_expression(failed_request, context)
            failed_second = run_deseq2_differential_expression(failed_request, context)
            self.assertEqual(failed_first.to_capability_result().status.value, "failed_execution")
            self.assertFalse(failed_first.cache_hit)
            self.assertFalse(failed_second.cache_hit)
            self.assertEqual(json.loads(Path(failed_first.cache_manifest_path).read_text())["completion_status"], "failed")

if __name__ == "__main__":
    unittest.main()
