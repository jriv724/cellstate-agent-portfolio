"""Typed CAP-DESEQ-002 fitting contracts."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from .common import (AnalysisProvenance, ArtifactCategory, ArtifactReference, CapabilityResult,
                     StructuredWarning, capability_status_from_warnings)

@dataclass(frozen=True)
class DifferentialExpressionInput:
    count_matrix_path: Path
    sample_metadata_path: Path
    cell_state: str
    sample_column: str = "sample"
    dataset_column: str = "dataset"
    stage_column: str = "stage_model_v2"
    cell_state_column: str = "cell_state"
    reference_level: Literal["NBM"] = "NBM"
    contrasts: tuple[tuple[str, str], ...] = (("SMM", "NBM"), ("NDMM", "NBM"))
    minimum_samples_per_group: int = 3
    gene_minimum_count: int = 10
    gene_minimum_samples: int = 3
    alpha: float = 0.05
    independent_filtering: Literal[True] = True
    request_apeglm: bool = True
    rscript_path: Path = Path("/opt/R/4.4.2/bin/Rscript")

    def __post_init__(self) -> None:
        for name in ("count_matrix_path", "sample_metadata_path", "rscript_path"):
            value = getattr(self, name)
            if not isinstance(value, Path): object.__setattr__(self, name, Path(value))
        if self.reference_level != "NBM" or self.contrasts != (("SMM", "NBM"), ("NDMM", "NBM")):
            raise ValueError("approved directional contrasts are SMM vs NBM and NDMM vs NBM")
        if self.minimum_samples_per_group != 3: raise ValueError("approved contract requires three samples per group")
        if self.gene_minimum_count != 10 or self.gene_minimum_samples != 3:
            raise ValueError("approved gene filter is count >=10 in at least three samples")
        if self.alpha != 0.05 or self.independent_filtering is not True:
            raise ValueError("approved results contract requires alpha=0.05 and independent filtering")

    def parameters(self) -> dict[str, object]:
        return {"cell_state": self.cell_state, "sample_column": self.sample_column,
                "dataset_column": self.dataset_column, "stage_column": self.stage_column,
                "cell_state_column": self.cell_state_column, "reference_level": self.reference_level,
                "contrasts": [list(x) for x in self.contrasts], "minimum_samples_per_group": self.minimum_samples_per_group,
                "gene_minimum_count": self.gene_minimum_count, "gene_minimum_samples": self.gene_minimum_samples,
                "alpha": self.alpha, "independent_filtering": self.independent_filtering,
                "request_apeglm": self.request_apeglm, "rscript_path": str(self.rscript_path),
                "design_formula": "~ dataset + stage"}

@dataclass(frozen=True)
class DifferentialExpressionOutput:
    capability_id: str
    node_version: str
    cache_key: str
    cache_hit: bool
    result_paths: tuple[str, ...]
    shrinkage_paths: tuple[str, ...]
    qc_path: str
    provenance_path: str
    cache_manifest_path: str
    warnings: tuple[StructuredWarning, ...]
    provenance: AnalysisProvenance

    def __post_init__(self) -> None:
        if self.capability_id != "CAP-DESEQ-002": raise ValueError("unexpected differential-expression capability")

    def to_capability_result(self) -> CapabilityResult:
        status = capability_status_from_warnings(self.warnings)
        artifacts = [ArtifactReference(f"unshrunk_{i + 1}", path, ArtifactCategory.INFERENTIAL, "text/csv")
                     for i, path in enumerate(self.result_paths)]
        artifacts.extend(ArtifactReference(f"shrunken_{i + 1}", path, ArtifactCategory.OPTIONAL, "text/csv", False)
                         for i, path in enumerate(self.shrinkage_paths))
        artifacts.extend((ArtifactReference("model_qc", self.qc_path, ArtifactCategory.QC, "text/csv"),
                          ArtifactReference("provenance", self.provenance_path, ArtifactCategory.PROVENANCE, "application/json"),
                          ArtifactReference("cache_manifest", self.cache_manifest_path, ArtifactCategory.MANIFEST, "application/json")))
        return CapabilityResult(self.capability_id, self.node_version, status, self.cache_key, self.cache_hit,
                                tuple(artifacts), self.warnings, self.provenance_path, self.cache_manifest_path)
