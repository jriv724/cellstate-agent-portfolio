"""Metadata validation shared by multiple planned nodes."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..schemas.common import StructuredWarning, WarningSeverity


def validate_required_columns(table: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def validate_sample_metadata(
    table: pd.DataFrame,
    *,
    sample_column: str,
    patient_column: str | None = None,
    dataset_column: str | None = None,
    group_column: str | None = None,
) -> list[StructuredWarning]:
    required = [sample_column]
    required.extend(
        column
        for column in (patient_column, dataset_column, group_column)
        if column is not None
    )
    validate_required_columns(table, required)
    if table[sample_column].isna().any() or table[sample_column].astype(str).str.strip().eq("").any():
        raise ValueError(f"{sample_column} contains missing or blank identifiers")

    warnings: list[StructuredWarning] = []
    for column in (patient_column, dataset_column, group_column):
        if column is None:
            continue
        counts = table.groupby(sample_column, dropna=False)[column].nunique(dropna=False)
        conflicts = counts[counts > 1]
        if not conflicts.empty:
            raise ValueError(
                f"{column} is inconsistent within biological sample: "
                f"{conflicts.index.astype(str).tolist()[:5]}"
            )
    duplicate_rows = int(table.duplicated().sum())
    if duplicate_rows:
        warnings.append(
            StructuredWarning(
                code="DUPLICATE_METADATA_ROWS",
                message="Exact duplicate metadata rows are present",
                severity=WarningSeverity.WARNING,
                context={"count": duplicate_rows},
            )
        )
    return warnings
