"""Typed plotting contracts that consume completed capability results."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..schemas.common import ArtifactReference, CapabilityResult, CapabilityStatus


@dataclass(frozen=True)
class AtlasLODOPlotInput:
    analysis_result: CapabilityResult
    output_dir: Path
    formats: tuple[Literal["pdf", "svg", "png"], ...] = ("pdf", "svg", "png")
    dpi: int = 300

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.analysis_result.capability_id != "CAP-LODO-001":
            raise ValueError("AtlasLODO plotting requires a CAP-LODO-001 CapabilityResult")
        if self.analysis_result.status not in {
            CapabilityStatus.COMPLETED, CapabilityStatus.COMPLETED_WITH_WARNINGS,
        }:
            raise ValueError("AtlasLODO plotting requires a completed analysis result")
        if not self.formats or len(set(self.formats)) != len(self.formats):
            raise ValueError("formats must be nonempty and unique")
        if self.dpi < 72:
            raise ValueError("dpi must be at least 72")


@dataclass(frozen=True)
class AtlasLODOPlotOutput:
    upstream_capability_id: str
    upstream_cache_key: str
    figures: tuple[ArtifactReference, ...]
    source_tables: tuple[ArtifactReference, ...]

    @property
    def artifacts(self) -> tuple[ArtifactReference, ...]:
        return self.figures + self.source_tables
