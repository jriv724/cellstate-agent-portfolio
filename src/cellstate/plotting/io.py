"""Read-only canonical-artifact resolution and deterministic writes."""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from ..schemas.common import CapabilityResult


def resolve_table_artifacts(result: CapabilityResult, required: tuple[str, ...]) -> dict[str, Path]:
    indexed = {artifact.logical_name: Path(artifact.path) for artifact in result.artifacts}
    absent = [name for name in required if name not in indexed]
    if absent:
        raise ValueError(f"CAP-LODO-001 plotting is missing required artifacts: {absent}")
    missing = [str(indexed[name]) for name in required if not indexed[name].is_file()]
    if missing:
        raise FileNotFoundError(f"CAP-LODO-001 plotting artifacts do not exist: {missing}")
    return {name: indexed[name] for name in required}


def read_table(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    table = pd.read_csv(path)
    absent = [column for column in required_columns if column not in table.columns]
    if absent:
        raise ValueError(f"canonical plotting table {path.name} is missing columns: {absent}")
    return table


def write_csv_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        table.to_csv(temporary, index=False, lineterminator="\n", float_format="%.15g")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
