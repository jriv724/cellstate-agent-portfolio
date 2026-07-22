"""Typed, dependency-free reproducibility harness contracts."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass(frozen=True)
class ExpectedArtifact:
    logical_name: str
    expected_path: Path
    actual_filename: str
    comparison: Literal["csv"] = "csv"
    absolute_tolerance: float = 1e-12
    relative_tolerance: float = 1e-12

@dataclass(frozen=True)
class ReproducibilityCase:
    case_id: str
    capability_id: str
    input_paths: tuple[Path, ...]
    expected_artifacts: tuple[ExpectedArtifact, ...]
    dataset_signature: str
    expected_status: str
    expected_warning_codes: tuple[str, ...] = ()
    expected_error_substring: str | None = None

@dataclass(frozen=True)
class ArtifactComparison:
    logical_name: str
    matched: bool
    message: str

@dataclass(frozen=True)
class ReproducibilityResult:
    case_id: str
    capability_id: str
    status: str
    cache_key: str
    cache_hit: bool
    comparisons: tuple[ArtifactComparison, ...]
    warning_codes: tuple[str, ...]
    manifest_completion_status: str

    @property
    def passed(self) -> bool:
        return all(item.matched for item in self.comparisons)
