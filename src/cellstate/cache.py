"""Directional cache keys and completion-safe cache manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas.common import (CapabilityStatus, StructuredWarning,
                             capability_status_from_warnings)


def _canonical(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def make_cache_key(
    *,
    capability_id: str,
    node_version: str,
    cache_schema_version: int,
    dataset_signature: str,
    parameters: Mapping[str, Any],
) -> str:
    """Hash parameters without sorting sequence values.

    Mapping keys are canonicalized, but comparison group lists retain their
    declared order so A-vs-B and B-vs-A cannot collide.
    """
    payload = {
        "capability_id": capability_id,
        "node_version": node_version,
        "cache_schema_version": cache_schema_version,
        "dataset_signature": dataset_signature,
        "parameters": _canonical(parameters),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheManifest:
    cache_key: str
    capability_id: str
    node_version: str
    cache_schema_version: int
    creation_timestamp_utc: str
    input_signature: str
    source_dataset_signature: str
    parameters: Mapping[str, Any]
    output_files: tuple[str, ...]
    completion_status: str
    warnings: tuple[StructuredWarning, ...]
    software_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.completion_status not in {"complete", "failed", "incomplete"}:
            raise ValueError("invalid completion_status")
        if not self.cache_key or not self.capability_id or not self.node_version:
            raise ValueError("cache identity fields must be nonblank")
        if self.completion_status == "complete" and capability_status_from_warnings(self.warnings) not in {
            CapabilityStatus.COMPLETED, CapabilityStatus.COMPLETED_WITH_WARNINGS,
        }:
            raise ValueError("terminal capability results cannot have complete manifests")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["warnings"] = [warning.to_dict() for warning in self.warnings]
        return result


def build_cache_manifest(
    *,
    cache_key: str,
    capability_id: str,
    node_version: str,
    cache_schema_version: int,
    input_signature: str,
    source_dataset_signature: str,
    parameters: Mapping[str, Any],
    output_files: Sequence[str],
    completion_status: str,
    warnings: Sequence[StructuredWarning],
    software_versions: Mapping[str, str],
) -> CacheManifest:
    warnings = tuple(warnings)
    result_status = capability_status_from_warnings(warnings)
    if completion_status == "complete" and result_status not in {
        CapabilityStatus.COMPLETED, CapabilityStatus.COMPLETED_WITH_WARNINGS,
    }:
        completion_status = "failed"
    return CacheManifest(
        cache_key=cache_key,
        capability_id=capability_id,
        node_version=node_version,
        cache_schema_version=cache_schema_version,
        creation_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        input_signature=input_signature,
        source_dataset_signature=source_dataset_signature,
        parameters=dict(parameters),
        output_files=tuple(output_files),
        completion_status=completion_status,
        warnings=warnings,
        software_versions=dict(software_versions),
    )


def write_manifest_atomic(manifest: CacheManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest.to_dict(), sort_keys=True, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_complete_manifest(path: Path, expected_cache_key: str) -> CacheManifest | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("cache_key") != expected_cache_key or raw.get("completion_status") != "complete":
        return None
    outputs = tuple(raw.get("output_files", ()))
    if not all(Path(output).exists() for output in outputs):
        return None
    warnings = tuple(
        StructuredWarning(
            code=item["code"],
            message=item["message"],
            severity=item.get("severity", "warning"),
            context=item.get("context", {}),
        )
        for item in raw.get("warnings", ())
    )
    if capability_status_from_warnings(warnings) not in {
        CapabilityStatus.COMPLETED, CapabilityStatus.COMPLETED_WITH_WARNINGS,
    }:
        return None
    return CacheManifest(
        cache_key=raw["cache_key"],
        capability_id=raw["capability_id"],
        node_version=raw["node_version"],
        cache_schema_version=raw["cache_schema_version"],
        creation_timestamp_utc=raw["creation_timestamp_utc"],
        input_signature=raw["input_signature"],
        source_dataset_signature=raw["source_dataset_signature"],
        parameters=raw["parameters"],
        output_files=outputs,
        completion_status=raw["completion_status"],
        warnings=warnings,
        software_versions=raw["software_versions"],
    )
