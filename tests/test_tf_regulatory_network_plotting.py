import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cellstate.context import AnalysisContext
from cellstate.nodes.tf_regulatory_network import file_hash, run_tf_regulatory_network
from cellstate.plotting import TFRegulatoryNetworkPlotInput, plot_tf_regulatory_network
from cellstate.schemas.tf_regulatory_network import TFRegulatoryNetworkInput
from tests.test_tf_regulatory_network import FIXTURE, program_rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TFRegulatoryNetworkPlottingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        source = root / "source.tsv"; source.write_text("feature_id\nG1\nG2\nG3\nG4\nG5\n")
        program = root / "program.tsv"
        pd.DataFrame(program_rows(source, file_hash(source))).to_csv(program, sep="\t", index=False)
        request = TFRegulatoryNetworkInput(program, FIXTURE / "background.tsv",
            FIXTURE / "dorothea.tsv", FIXTURE / "collectri.tsv")
        context = AnalysisContext((root / "out").resolve(), (root / "cache").resolve(), "plot")
        self.analysis = run_tf_regulatory_network(request, context).to_capability_result()
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def test_plot_artifacts_empty_handling_and_display_rule(self):
        output = plot_tf_regulatory_network(TFRegulatoryNetworkPlotInput(
            self.analysis, self.root / "plots", formats=("png",),
            maximum_displayed_tfs=1, maximum_displayed_targets=2))
        self.assertEqual(len(output.figures), 2)
        self.assertTrue(all(Path(x.path).is_file() for x in output.artifacts))
        nodes = pd.read_csv(output.display_tables[0].path)
        canonical_path = next(x.path for x in self.analysis.artifacts
                              if x.logical_name == "tf_target_program_network_nodes")
        canonical = pd.read_csv(canonical_path, sep="\t")
        self.assertLessEqual(len(nodes), len(canonical))
        self.assertIn("Omitted nodes remain", Path(output.plot_provenance.path).read_text())

    def test_repeated_rendering_deterministic_and_sources_immutable(self):
        before = {x.path: digest(x.path) for x in self.analysis.artifacts if Path(x.path).is_file()}
        one = plot_tf_regulatory_network(TFRegulatoryNetworkPlotInput(
            self.analysis, self.root / "one", formats=("png",)))
        two = plot_tf_regulatory_network(TFRegulatoryNetworkPlotInput(
            self.analysis, self.root / "two", formats=("png",)))
        self.assertEqual({Path(x.path).name: digest(x.path) for x in one.artifacts},
                         {Path(x.path).name: digest(x.path) for x in two.artifacts})
        after = {x.path: digest(x.path) for x in self.analysis.artifacts if Path(x.path).is_file()}
        self.assertEqual(before, after)

    def test_plotter_requires_canonical_artifacts(self):
        incomplete = self.analysis.__class__(
            self.analysis.capability_id, self.analysis.node_version, self.analysis.status,
            self.analysis.cache_key, False, self.analysis.artifacts[:-5],
            self.analysis.warnings, self.analysis.provenance_path,
            self.analysis.cache_manifest_path)
        with self.assertRaisesRegex(ValueError, "missing required artifacts"):
            plot_tf_regulatory_network(TFRegulatoryNetworkPlotInput(
                incomplete, self.root / "bad", formats=("png",)))


if __name__ == "__main__":
    unittest.main()
