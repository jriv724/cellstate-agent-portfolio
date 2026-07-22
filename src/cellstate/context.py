"""Explicit runtime context passed to deterministic analysis functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class AnalysisContext:
    output_root: Path
    cache_root: Path
    dataset_signature: str
    software_versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_signature.strip():
            raise ValueError("dataset_signature must be nonblank")
        if not self.output_root.is_absolute() or not self.cache_root.is_absolute():
            raise ValueError("output_root and cache_root must be absolute")

    def capability_output_dir(self, capability_id: str) -> Path:
        if not capability_id.strip():
            raise ValueError("capability_id must be nonblank")
        return self.output_root / capability_id.lower()
