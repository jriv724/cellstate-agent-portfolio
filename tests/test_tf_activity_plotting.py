import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cellstate.context import AnalysisContext
from cellstate.nodes.tf_activity import file_hash, run_tf_activity
from cellstate.plotting.tf_activity import TFActivityPlotInput, plot_tf_activity
from cellstate.schemas.common import (
    ArtifactCategory, ArtifactReference, CapabilityResult, CapabilityStatus)
from cellstate.schemas.tf_activity import TFActivityInput
from tests.test_tf_activity import FIXTURE, program_rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TFActivityPlottingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        source = self.root / "source.tsv"
        source.write_text("upstream\n")
        program = self.root / "program.tsv"
        pd.DataFrame(program_rows(source, file_hash(source))).to_csv(
            program, sep="\t", index=False)
        request = TFActivityInput(
            program, FIXTURE / "dorothea.tsv", FIXTURE / "collectri.tsv")
        context = AnalysisContext(
            (self.root / "analysis").resolve(), (self.root / "cache").resolve(),
            "plot-fixture")
        self.analysis = run_tf_activity(request, context).to_capability_result()

    def tearDown(self):
        self.tmp.cleanup()

    def test_plot_consumes_canonical_artifact_without_mutation(self):
        canonical = next(
            artifact.path for artifact in self.analysis.artifacts
            if artifact.logical_name == "heatmap_source_table")
        before = digest(canonical)
        output = plot_tf_activity(TFActivityPlotInput(
            self.analysis, self.root / "plots", formats=("png",),
            maximum_displayed_tfs=1))
        self.assertEqual(before, digest(canonical))
        self.assertEqual(len(output.figures), 1)
        display = pd.read_csv(output.display_tables[0].path)
        self.assertLessEqual(display.tf.nunique(), 1)
        self.assertTrue((display.display_activity_score.abs() > 0).all())
        pd.testing.assert_series_equal(
            display.display_activity_score,
            display.median_consensus_activity_score,
            check_names=False)
        self.assertTrue(Path(output.plot_provenance.path).is_file())

    def test_repeated_png_rendering_is_deterministic(self):
        first = plot_tf_activity(TFActivityPlotInput(
            self.analysis, self.root / "first", formats=("png",)))
        second = plot_tf_activity(TFActivityPlotInput(
            self.analysis, self.root / "second", formats=("png",)))
        self.assertEqual(digest(first.figures[0].path),
                         digest(second.figures[0].path))
        self.assertEqual(digest(first.display_tables[0].path),
                         digest(second.display_tables[0].path))

    def test_empty_state_and_missing_artifact(self):
        empty = self.root / "empty.tsv"
        pd.DataFrame(columns=[
            "program_id", "cell_state", "contrast", "tf", "consensus_direction",
            "minimum_adjusted_p_value",
            "number_of_significant_supporting_resources",
            "directional_consensus_status", "display_activity_score",
        ]).to_csv(empty, sep="\t", index=False)
        result = CapabilityResult(
            "CAP-TF-002", "1.0.0", CapabilityStatus.COMPLETED, "empty", False,
            (ArtifactReference(
                "heatmap_source_table", str(empty), ArtifactCategory.DESCRIPTIVE,
                "text/tab-separated-values"),), (), "prov.json", "manifest.json")
        output = plot_tf_activity(TFActivityPlotInput(
            result, self.root / "empty-plot", formats=("png",)))
        self.assertTrue(Path(output.figures[0].path).is_file())
        missing = CapabilityResult(
            "CAP-TF-002", "1.0.0", CapabilityStatus.COMPLETED, "missing", False,
            (ArtifactReference(
                "other", str(empty), ArtifactCategory.DESCRIPTIVE),),
            (), "prov.json", "manifest.json")
        with self.assertRaisesRegex(ValueError, "missing required artifacts"):
            plot_tf_activity(TFActivityPlotInput(
                missing, self.root / "missing-plot", formats=("png",)))


if __name__ == "__main__":
    unittest.main()
