"""Deterministic table comparison used by committed regression cases."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .schemas import ArtifactComparison, ExpectedArtifact

def compare_csv(actual_path: Path, expected: ExpectedArtifact) -> ArtifactComparison:
    if not actual_path.exists():
        return ArtifactComparison(expected.logical_name, False, f"missing actual artifact: {actual_path}")
    if not expected.expected_path.exists():
        return ArtifactComparison(expected.logical_name, False, f"missing committed expectation: {expected.expected_path}")
    actual = pd.read_csv(actual_path)
    reference = pd.read_csv(expected.expected_path)
    try:
        pd.testing.assert_frame_equal(actual, reference, check_like=False, check_dtype=True,
            check_exact=False, atol=expected.absolute_tolerance, rtol=expected.relative_tolerance)
    except AssertionError as error:
        return ArtifactComparison(expected.logical_name, False, str(error))
    return ArtifactComparison(expected.logical_name, True, "matched committed table")

def compare_artifact(actual_path: Path, expected: ExpectedArtifact) -> ArtifactComparison:
    if expected.comparison == "csv":
        return compare_csv(actual_path, expected)
    raise ValueError(f"unsupported comparison: {expected.comparison}")
