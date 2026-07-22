"""Design-matrix rank and estimability checks."""

from __future__ import annotations

import numpy as np


def validate_design_estimability(
    design_matrix: np.ndarray,
    *,
    term_names: list[str] | tuple[str, ...],
    minimum_residual_degrees_of_freedom: int = 1,
) -> dict[str, int]:
    matrix = np.asarray(design_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("design_matrix must be two-dimensional")
    if matrix.shape[1] != len(term_names):
        raise ValueError("term_names length must equal design columns")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("design_matrix contains non-finite values")
    rank = int(np.linalg.matrix_rank(matrix))
    if rank < matrix.shape[1]:
        raise ValueError(
            f"design matrix is rank deficient: rank {rank}, columns {matrix.shape[1]}"
        )
    residual_df = int(matrix.shape[0] - rank)
    if residual_df < minimum_residual_degrees_of_freedom:
        raise ValueError(
            f"insufficient residual degrees of freedom: {residual_df}"
        )
    return {"rows": matrix.shape[0], "columns": matrix.shape[1], "rank": rank, "residual_df": residual_df}
