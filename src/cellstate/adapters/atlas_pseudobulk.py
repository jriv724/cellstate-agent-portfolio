"""Backed, chunked atlas adapter for CAP-DESEQ-003 inputs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Callable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class AtlasPseudobulkResult:
    count_matrix_path: Path
    sample_metadata_path: Path
    qc_path: Path
    provenance_path: Path
    count_source: str
    atlas_identity: str
    n_cells: int
    n_genes: int
    group_replicate_counts: dict[str, int]
    cache_hit: bool = False


class AtlasPseudobulkAdapter:
    minimum_cells_per_sample_state = 100

    def __init__(
        self,
        atlas_path: Path,
        *,
        reader: Callable[..., Any] = ad.read_h5ad,
        chunk_size: int = 5000,
        count_source: str = "auto",
        cell_state_column: str = "preserved",
        group_column: str = "stage_model_v2",
        sample_column: str = "sample",
        dataset_column: str = "dataset",
    ) -> None:
        self.atlas_path = Path(atlas_path)
        self.reader = reader
        self.chunk_size = chunk_size
        self.count_source = count_source
        self.cell_state_column = cell_state_column
        self.group_column = group_column
        self.sample_column = sample_column
        self.dataset_column = dataset_column

    def _identity(self) -> str:
        stat = self.atlas_path.stat()
        payload = f"{self.atlas_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        return sha256(payload.encode()).hexdigest()

    def cache_key(self, cell_state: str, group_a: str, group_b: str) -> str:
        payload = {
            "adapter_version": "1.0.0",
            "atlas_identity": self._identity(),
            "cell_state": cell_state,
            "ordered_groups": [group_a, group_b],
            "minimum_cells_per_sample_state": self.minimum_cells_per_sample_state,
            "count_source_policy": self.count_source,
            "columns": [
                self.cell_state_column, self.group_column,
                self.sample_column, self.dataset_column,
            ],
        }
        return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _source(self, atlas: Any) -> str:
        if self.count_source != "auto":
            source = self.count_source
        elif "counts" in atlas.layers:
            source = "layers/counts"
        elif getattr(atlas, "raw", None) is not None:
            source = "raw.X"
        else:
            source = "X"
        if source not in {"layers/counts", "raw.X", "X"}:
            raise ValueError(f"Unsupported count source: {source}")
        return source

    @staticmethod
    def _atomic_table(table: pd.DataFrame, path: Path, *, index: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            table.to_csv(temporary, sep="\t", index=index)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_json(value: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def build(
        self,
        *,
        cell_state: str,
        group_a: str,
        group_b: str,
        output_dir: Path,
    ) -> AtlasPseudobulkResult:
        counts_path = output_dir / "adapter_counts.tsv"
        metadata_path = output_dir / "adapter_sample_metadata.tsv"
        qc_path = output_dir / "adapter_sample_qc.tsv"
        provenance_path = output_dir / "adapter_provenance.json"
        if all(path.is_file() for path in (
            counts_path, metadata_path, qc_path, provenance_path
        )):
            provenance = json.loads(provenance_path.read_text())
            if (
                provenance.get("adapter_cache_key")
                == self.cache_key(cell_state, group_a, group_b)
                and provenance.get("ordered_groups") == [group_a, group_b]
                and provenance.get("minimum_cells_per_sample_state") == 100
            ):
                counts = pd.read_csv(counts_path, sep="\t", index_col=0)
                metadata = pd.read_csv(metadata_path, sep="\t")
                if counts.columns.tolist() != metadata["sample"].astype(str).tolist():
                    raise ValueError("Cached adapter count and metadata artifacts are misaligned.")
                grouped = metadata.groupby("group", observed=True).size().to_dict()
                return AtlasPseudobulkResult(
                    counts_path, metadata_path, qc_path, provenance_path,
                    str(provenance["count_source"]),
                    str(provenance["atlas_identity"]),
                    int(provenance["cells_contributing"]), int(counts.shape[0]),
                    {group_a: int(grouped.get(group_a, 0)),
                     group_b: int(grouped.get(group_b, 0))},
                    True,
                )
        atlas = self.reader(self.atlas_path, backed="r")
        try:
            required = {
                self.cell_state_column, self.group_column, self.sample_column,
                self.dataset_column,
            }
            missing = sorted(required - set(atlas.obs.columns))
            if missing:
                raise ValueError("Atlas metadata is missing: " + ", ".join(missing))
            obs = atlas.obs
            mask = (
                obs[self.cell_state_column].astype(str).eq(cell_state)
                & obs[self.group_column].astype(str).isin([group_a, group_b])
            )
            positions = np.flatnonzero(mask.to_numpy())
            if not positions.size:
                raise ValueError("No atlas cells match the selected comparison.")
            selected = obs.iloc[positions][[
                self.sample_column, self.group_column, self.dataset_column
            ]].copy()
            if selected.isna().any(axis=None):
                raise ValueError("Selected atlas cells contain missing sample/group/dataset metadata.")
            consistency = selected.groupby(self.sample_column, observed=True).agg(
                groups=(self.group_column, "nunique"),
                datasets=(self.dataset_column, "nunique"),
            )
            if consistency[["groups", "datasets"]].gt(1).any(axis=None):
                raise ValueError("Sample identifiers map to conflicting group or dataset metadata.")
            qc = selected.groupby(
                [self.sample_column, self.group_column, self.dataset_column],
                observed=True, sort=False,
            ).size().rename("n_cells").reset_index()
            qc["eligible"] = qc.n_cells.ge(self.minimum_cells_per_sample_state)
            eligible = qc.loc[qc.eligible].copy()
            eligible_samples = set(eligible[self.sample_column].astype(str))
            keep = selected[self.sample_column].astype(str).isin(eligible_samples).to_numpy()
            positions = positions[keep]
            selected = selected.iloc[np.flatnonzero(keep)]
            if not positions.size:
                raise ValueError("No samples pass the production 100-cell threshold.")

            source = self._source(atlas)
            genes = np.asarray(
                atlas.raw.var_names if source == "raw.X" else atlas.var_names,
                dtype=str,
            )
            if len(set(genes)) != len(genes):
                raise ValueError("Count source contains duplicate gene identifiers.")
            samples = sorted(eligible_samples)
            sample_index = {sample: index for index, sample in enumerate(samples)}
            summed = np.zeros((len(genes), len(samples)), dtype=np.int64)
            cell_samples = selected[self.sample_column].astype(str).to_numpy()
            for start in range(0, len(positions), self.chunk_size):
                stop = min(start + self.chunk_size, len(positions))
                chunk_positions = positions[start:stop]
                if source == "layers/counts":
                    matrix = atlas[chunk_positions, :].layers["counts"]
                elif source == "raw.X":
                    matrix = atlas.raw[chunk_positions, :].X
                else:
                    matrix = atlas[chunk_positions, :].X
                matrix = matrix.tocsr() if sp.issparse(matrix) else np.asarray(matrix)
                values = matrix.data if sp.issparse(matrix) else matrix.ravel()
                if (
                    not np.isfinite(values).all()
                    or (values < 0).any()
                    or not np.allclose(values, np.round(values), atol=1e-8)
                ):
                    raise ValueError(
                        "Selected count source must be finite, nonnegative, integer-valued raw counts."
                    )
                for local_row, sample in enumerate(cell_samples[start:stop]):
                    row = matrix.getrow(local_row).toarray().ravel() if sp.issparse(matrix) else matrix[local_row]
                    summed[:, sample_index[sample]] += np.asarray(row, dtype=np.int64)
        finally:
            file_handle = getattr(atlas, "file", None)
            if file_handle is not None:
                file_handle.close()

        output_dir.mkdir(parents=True, exist_ok=True)
        counts = pd.DataFrame(summed, index=genes, columns=samples)
        counts.index.name = "gene"
        metadata = eligible.rename(columns={
            self.sample_column: "sample",
            self.group_column: "group",
            self.dataset_column: "dataset",
        })[["sample", "group", "dataset", "n_cells"]].copy()
        metadata["sample"] = metadata["sample"].astype(str)
        metadata = metadata.set_index("sample").loc[samples].reset_index()
        self._atomic_table(counts, counts_path, index=True)
        self._atomic_table(metadata, metadata_path, index=False)
        self._atomic_table(qc, qc_path, index=False)
        atlas_identity = self._identity()
        provenance = {
            "adapter": "cellstate.adapters.AtlasPseudobulkAdapter",
            "adapter_cache_key": self.cache_key(cell_state, group_a, group_b),
            "atlas_path": str(self.atlas_path.resolve()),
            "atlas_identity": atlas_identity,
            "count_source": source,
            "cell_state": cell_state,
            "ordered_groups": [group_a, group_b],
            "comparison_direction": "group_a_minus_group_b",
            "minimum_cells_per_sample_state": self.minimum_cells_per_sample_state,
            "independent_unit": self.sample_column,
            "dataset_column": self.dataset_column,
            "cells_contributing": int(len(positions)),
            "genes": int(len(genes)),
            "samples": int(len(samples)),
            "outputs": [str(counts_path), str(metadata_path), str(qc_path)],
        }
        self._atomic_json(provenance, provenance_path)
        counts_by_group = metadata.groupby("group", observed=True).size().to_dict()
        return AtlasPseudobulkResult(
            counts_path, metadata_path, qc_path, provenance_path, source,
            atlas_identity, len(positions), len(genes),
            {group_a: int(counts_by_group.get(group_a, 0)),
             group_b: int(counts_by_group.get(group_b, 0))},
        )
