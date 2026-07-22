"""Small deterministic design helpers; no statistical fitting lives here."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def ordered_indicator_design(
    groups: Sequence[str],
    *,
    levels: Sequence[str],
    reference_level: str,
    include_intercept: bool = True,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if reference_level not in levels:
        raise ValueError("reference_level must occur in levels")
    if len(set(levels)) != len(levels):
        raise ValueError("levels must be unique and ordered")
    unknown = sorted(set(groups) - set(levels))
    if unknown:
        raise ValueError(f"unknown group levels: {unknown}")
    categorical = pd.Categorical(groups, categories=list(levels), ordered=True)
    columns: list[np.ndarray] = []
    names: list[str] = []
    if include_intercept:
        columns.append(np.ones(len(groups), dtype=float))
        names.append("Intercept")
    for level in levels:
        if level == reference_level:
            continue
        columns.append((categorical == level).astype(float))
        names.append(f"group[{level}]")
    matrix = np.column_stack(columns) if columns else np.empty((len(groups), 0))
    return matrix, tuple(names)
