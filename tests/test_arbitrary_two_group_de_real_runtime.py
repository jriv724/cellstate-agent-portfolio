"""Conditional real-DESeq2 directionality test for CAP-DESEQ-003."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from cellstate.context import AnalysisContext
from cellstate.nodes.arbitrary_two_group_de import run_arbitrary_two_group_de
from cellstate.schemas.arbitrary_two_group_de import ArbitraryTwoGroupDEInput


RSCRIPT = Path("/usr/local/bin/Rscript")


def _skip_reason() -> str | None:
    if not RSCRIPT.is_file():
        return "Rscript is unavailable"
    probe = subprocess.run(
        [str(RSCRIPT), "-e",
         "quit(status=if(requireNamespace('DESeq2',quietly=TRUE)) 0 else 2)"],
        capture_output=True, text=True,
    )
    return None if probe.returncode == 0 else "R/DESeq2 is unavailable"


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class RealArbitraryTwoGroupDETests(unittest.TestCase):
    def test_reversing_groups_reverses_actual_deseq2_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = [f"A{i}" for i in range(1, 4)] + [
                f"B{i}" for i in range(1, 4)]
            rng = np.random.default_rng(20260718)
            genes = [
                "elevated_in_a",
                "elevated_in_b",
                *[f"background_{index:02d}" for index in range(1, 59)],
            ]
            means = np.geomspace(18.0, 1200.0, num=len(genes))
            dispersions = np.geomspace(0.02, 0.8, num=len(genes))
            matrix = np.empty((len(genes), len(samples)), dtype=np.int64)
            for gene_index, (mean, dispersion) in enumerate(
                    zip(means, dispersions, strict=True)):
                size = 1.0 / dispersion
                sample_means = mean * np.array(
                    [0.82, 1.00, 1.18, 0.88, 1.04, 1.14])
                if gene_index == 0:
                    sample_means[:3] *= 12.0
                elif gene_index == 1:
                    sample_means[3:] *= 12.0
                probabilities = size / (size + sample_means)
                matrix[gene_index, :] = rng.negative_binomial(
                    size, probabilities)
            counts = pd.DataFrame(matrix, index=genes, columns=samples)
            metadata = pd.DataFrame({
                "sample": samples,
                "group": ["A"] * 3 + ["B"] * 3,
                "dataset": ["D1"] * 6,
                "n_cells": [100] * 6,
            })
            counts_path = root / "counts.tsv"
            metadata_path = root / "metadata.tsv"
            counts.to_csv(counts_path, sep="\t")
            metadata.to_csv(metadata_path, sep="\t", index=False)
            context = AnalysisContext(
                root / "out", root / "cache", "real-cap-deseq-003")

            def run(a: str, b: str):
                return run_arbitrary_two_group_de(
                    ArbitraryTwoGroupDEInput(
                        count_matrix_path=counts_path,
                        sample_metadata_path=metadata_path,
                        group_a=a, group_b=b, rscript_path=RSCRIPT,
                    ),
                    context,
                )

            ab = run("A", "B")
            ba = run("B", "A")
            self.assertIn(ab.terminal_status, {"completed",
                                               "completed_with_warnings"})
            self.assertIn(ba.terminal_status, {"completed",
                                               "completed_with_warnings"})
            ab_path = next(Path(x.path) for x in ab.artifacts
                           if x.logical_name == "deseq2_results")
            ba_path = next(Path(x.path) for x in ba.artifacts
                           if x.logical_name == "deseq2_results")
            ab_results = pd.read_csv(ab_path, sep="\t").set_index("gene")
            ba_results = pd.read_csv(ba_path, sep="\t").set_index("gene")
            ab_effect = ab_results.loc["elevated_in_a", "log2FoldChange"]
            ba_effect = ba_results.loc["elevated_in_a", "log2FoldChange"]
            self.assertGreater(ab_effect, 0)
            self.assertLess(ba_effect, 0)
            self.assertAlmostEqual(ab_effect, -ba_effect, delta=1e-6)
            self.assertGreater(
                ab_results.loc["elevated_in_a", "stat"], 0)
            self.assertLess(
                ba_results.loc["elevated_in_a", "stat"], 0)


if __name__ == "__main__":
    unittest.main()
