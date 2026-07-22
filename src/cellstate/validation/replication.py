"""Biological-replicate and dataset-overlap checks."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..schemas.common import StructuredWarning, WarningSeverity
from .metadata import validate_required_columns


def validate_biological_replication(
    table: pd.DataFrame,
    *,
    sample_column: str,
    group_column: str,
    groups: Sequence[str],
    minimum_samples_per_group: int,
    dataset_column: str | None = None,
) -> list[StructuredWarning]:
    if minimum_samples_per_group < 1:
        raise ValueError("minimum_samples_per_group must be >= 1")
    if len(groups) < 2 or len(set(groups)) != len(groups):
        raise ValueError("comparison groups must contain at least two distinct levels")
    required = [sample_column, group_column]
    if dataset_column:
        required.append(dataset_column)
    validate_required_columns(table, required)

    sample_rows = table.loc[table[group_column].isin(groups), required].drop_duplicates()
    group_per_sample = sample_rows.groupby(sample_column)[group_column].nunique()
    if (group_per_sample > 1).any():
        raise ValueError("a biological sample appears in multiple comparison groups")
    counts = sample_rows.groupby(group_column)[sample_column].nunique()
    insufficient = {
        group: int(counts.get(group, 0))
        for group in groups
        if int(counts.get(group, 0)) < minimum_samples_per_group
    }
    if insufficient:
        raise ValueError(f"too few biological replicates: {insufficient}")

    warnings: list[StructuredWarning] = []
    if dataset_column:
        presence = pd.crosstab(sample_rows[group_column], sample_rows[dataset_column])
        exclusive = [
            str(dataset)
            for dataset in presence.columns
            if int((presence[dataset] > 0).sum()) == 1
        ]
        if exclusive:
            warnings.append(
                StructuredWarning(
                    code="DATASET_GROUP_CONFOUNDING",
                    message="One or more datasets contain only one comparison group",
                    severity=WarningSeverity.WARNING,
                    context={"datasets": exclusive},
                )
            )
        common = [
            str(dataset)
            for dataset in presence.columns
            if bool((presence[dataset] > 0).all())
        ]
        if not common:
            warnings.append(
                StructuredWarning(
                    code="NO_DATASET_OVERLAP",
                    message="No dataset contains every comparison group",
                    severity=WarningSeverity.ERROR,
                    context={"groups": list(groups)},
                )
            )
    return warnings
