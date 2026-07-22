import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import cellstate

from cellstate.plotting import AtlasLODOPlotInput, plot_atlas_lodo
from cellstate.schemas.common import (ArtifactCategory, ArtifactReference,
    CapabilityResult, CapabilityStatus)


STAGING = Path(cellstate.__file__).resolve().parent / "reproducibility/staging/CAP-LODO-001"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_result():
    output_dirs = sorted((STAGING / "outputs/cap-lodo-001").glob("*/"))
    if not output_dirs:
        raise RuntimeError("committed CAP-LODO-001 staging outputs are missing")
    root = output_dirs[0]
    mapping = {
        "atlas_summary": "atlas_cell_state_summary.csv",
        "gene_summaries": "gene_level_cross_fold_summary.csv",
        "cell_state_eligibility": "cell_state_eligibility.csv",
        "fold_results": "fold_level_model_results.csv",
    }
    artifacts = tuple(ArtifactReference(name, str(root / filename), ArtifactCategory.INFERENTIAL, "text/csv")
                      for name, filename in mapping.items())
    manifest = next((STAGING / "cache/cap-lodo-001").glob("*/manifest.json"))
    return CapabilityResult("CAP-LODO-001", "1.0.0", CapabilityStatus.COMPLETED,
        root.name, False, artifacts, (), str(root / "provenance.json"), str(manifest))


class AtlasLODOPlottingTests(unittest.TestCase):
    def test_figures_and_source_tables_generate(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = plot_atlas_lodo(AtlasLODOPlotInput(fixture_result(), Path(tmp)))
            self.assertEqual(len(result.figures), 12)
            self.assertEqual(len(result.source_tables), 4)
            self.assertTrue(all(Path(item.path).is_file() and Path(item.path).stat().st_size > 0
                                for item in result.artifacts))
            self.assertEqual({Path(item.path).suffix for item in result.figures}, {".pdf", ".svg", ".png"})

    def test_repeated_rendering_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = plot_atlas_lodo(AtlasLODOPlotInput(fixture_result(), root / "one"))
            second = plot_atlas_lodo(AtlasLODOPlotInput(fixture_result(), root / "two"))
            first_hashes = {Path(x.path).name: digest(x.path) for x in first.artifacts}
            second_hashes = {Path(x.path).name: digest(x.path) for x in second.artifacts}
            self.assertEqual(first_hashes, second_hashes)

    def test_plotting_does_not_change_analysis_artifacts(self):
        analysis = fixture_result()
        before = {item.path: digest(item.path) for item in analysis.artifacts}
        with tempfile.TemporaryDirectory() as tmp:
            plot_atlas_lodo(AtlasLODOPlotInput(analysis, Path(tmp)))
        after = {item.path: digest(item.path) for item in analysis.artifacts}
        self.assertEqual(before, after)

    def test_source_tables_are_presentational_subsets(self):
        analysis = fixture_result()
        canonical = pd.read_csv(next(Path(x.path) for x in analysis.artifacts if x.logical_name == "gene_summaries"))
        with tempfile.TemporaryDirectory() as tmp:
            output = plot_atlas_lodo(AtlasLODOPlotInput(analysis, Path(tmp), formats=("png",)))
            source_path = next(Path(x.path) for x in output.source_tables if x.logical_name == "atlas_lodo_effect_landscape_source")
            source = pd.read_csv(source_path)
            expected = canonical.sort_values(["cell_state", "gene"], kind="mergesort").reset_index(drop=True)
            pd.testing.assert_series_equal(source["delta_median"], expected["delta_median"], check_names=False)
            pd.testing.assert_series_equal(source["group_a_specific"], expected["group_a_specific"], check_names=False)

    def test_missing_artifact_has_clear_error(self):
        valid = fixture_result()
        incomplete = CapabilityResult(valid.capability_id, valid.node_version, valid.status,
            valid.cache_key, False, valid.artifacts[:-1], (), valid.provenance_path, valid.cache_manifest_path)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "missing required artifacts.*fold_results"):
                plot_atlas_lodo(AtlasLODOPlotInput(incomplete, Path(tmp)))


if __name__ == "__main__":
    unittest.main()
