"""Lightweight cached readiness validation for CAP-TF-002 resources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from cellstate.nodes.tf_activity import _read_table, normalize_signed_resource
from cellstate.schemas.tf_activity import TFActivityInput


@dataclass(frozen=True)
class TFResourceValidation:
    database: str
    path: str
    exists: bool
    valid: bool
    error: str | None
    row_count: int
    unique_tf_count: int
    unique_target_count: int
    invalid_or_nonnumeric_weights: int
    zero_weights: int
    exact_duplicate_rows: int
    duplicate_edges: int
    conflicting_edges: int
    confidence_levels: tuple[str, ...]
    normalized_edge_count: int
    normalized_content_hash: str | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["confidence_levels"] = list(self.confidence_levels)
        return value


def _request(dorothea: Path, collectri: Path) -> TFActivityInput:
    return TFActivityInput(Path("unused-signed-program.tsv"), dorothea, collectri)


@lru_cache(maxsize=32)
def _validate_cached(
    path_text: str, size: int, modified_ns: int, database: str,
) -> TFResourceValidation:
    path = Path(path_text)
    try:
        raw = _read_table(path)
    except Exception as exc:
        return TFResourceValidation(database, path_text, True, False, str(exc),
                                    0, 0, 0, 0, 0, 0, 0, 0, (), 0, None)
    aliases = {"source": "tf", "target": "target_gene", "gene": "target_gene"}
    diagnostic = raw.rename(columns={
        old: new for old, new in aliases.items() if old in raw and new not in raw
    }).copy()
    weights = pd.to_numeric(diagnostic.get("weight", pd.Series(dtype=object)),
                            errors="coerce")
    invalid_weights = int(weights.isna().sum()) if "weight" in diagnostic else len(raw)
    zero_weights = int(weights.eq(0).sum()) if "weight" in diagnostic else 0
    exact_duplicates = int(raw.duplicated().sum())
    duplicate_edges = 0
    conflicts = 0
    if {"tf", "target_gene", "weight"}.issubset(diagnostic.columns):
        keys = diagnostic.assign(_weight=weights).groupby(
            ["tf", "target_gene"], dropna=False, sort=True
        )
        duplicate_edges = int(sum(max(0, len(group) - 1) for _, group in keys))
        conflicts = int(sum(group._weight.dropna().nunique() > 1 for _, group in keys))
    confidence = tuple(sorted(
        value for value in diagnostic.get("confidence", pd.Series(dtype=str))
        .astype(str).str.upper().str.strip().unique() if value
    ))
    request = _request(path if database == "DoRothEA" else Path("unused.tsv"),
                       path if database == "CollecTRI" else Path("unused.tsv"))
    try:
        normalized, qc = normalize_signed_resource(raw, database, request)
        return TFResourceValidation(
            database, path_text, True, True, None, len(raw),
            int(normalized.tf.nunique()), int(normalized.target_gene.nunique()),
            invalid_weights, zero_weights, exact_duplicates, duplicate_edges,
            conflicts, confidence, int(qc["normalized_edge_count"]),
            str(qc["resolved_resource_table_hash"]),
        )
    except Exception as exc:
        return TFResourceValidation(
            database, path_text, True, False, str(exc), len(raw),
            int(diagnostic.tf.nunique()) if "tf" in diagnostic else 0,
            int(diagnostic.target_gene.nunique()) if "target_gene" in diagnostic else 0,
            invalid_weights, zero_weights, exact_duplicates, duplicate_edges,
            conflicts, confidence, 0, None,
        )


def validate_tf_resource(path: Path | None, database: str) -> TFResourceValidation:
    if path is None:
        return TFResourceValidation(database, "", False, False,
                                    "resource path is not configured",
                                    0, 0, 0, 0, 0, 0, 0, 0, (), 0, None)
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return TFResourceValidation(database, str(resolved), False, False,
                                    "resource file does not exist",
                                    0, 0, 0, 0, 0, 0, 0, 0, (), 0, None)
    stat = resolved.stat()
    return _validate_cached(str(resolved), stat.st_size, stat.st_mtime_ns, database)


def validate_tf_resource_pair(
    dorothea: Path | None, collectri: Path | None,
) -> tuple[TFResourceValidation, TFResourceValidation]:
    return (validate_tf_resource(dorothea, "DoRothEA"),
            validate_tf_resource(collectri, "CollecTRI"))
