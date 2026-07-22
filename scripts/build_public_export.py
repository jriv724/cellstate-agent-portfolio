#!/usr/bin/env python3
"""Fail-closed deterministic public export builder (dry-run by default)."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib

DEFAULT_LIMIT = 20 * 1024 * 1024
KNOWN_REJECTED_DESTINATION_NAME = "cellstate-agent-public"
FORBIDDEN_PARTS = {".git", "__pycache__", ".pytest_cache", "outputs", "analysis_cache", "demo_archive", "staging"}
FORBIDDEN_PREFIXES = ("outputs_backup_", "semantic_cache")
FORBIDDEN_EXTENSIONS = {".h5ad", ".h5", ".hdf5", ".loom", ".rds", ".rdata", ".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".zip", ".tar", ".gz", ".mp4", ".mov"}
PRIVATE_PATTERNS = (
    re.compile(b"/" + b"samurlab1/", re.I),
    re.compile(b"/" + rb"homes[0-9]*/", re.I),
    re.compile(rb"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    re.compile(rb"(?:OPENAI|GEMINI|ANTHROPIC)_API_KEY\s*=\s*[^\s#]+", re.I),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
)

class ExportError(RuntimeError):
    pass

def _relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ExportError(f"unsafe relative path: {value!r}")
    if ".git" in pure.parts:
        raise ExportError(".git is prohibited")
    return Path(*pure.parts)

def _prohibited(path: Path) -> bool:
    return (any(p in FORBIDDEN_PARTS or p.startswith(FORBIDDEN_PREFIXES) for p in path.parts)
            or path.suffix.casefold() in FORBIDDEN_EXTENSIONS
            or path.name in {"app.py", "app_backup.py", "validator_test.py", "planner_test.py", "==="}
            or ".before_" in path.name or path.name.endswith(".bak") or path.name.endswith(".pre_index_alias_fix"))

def load_inventory(source: Path, manifest_path: Path) -> tuple[list[dict], list[dict]]:
    source = source.resolve(strict=True)
    data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    mappings: dict[str, dict] = {}
    for entry in data.get("include", []):
        if "source" in entry:
            matches = [_relative(entry["source"])]
        else:
            root = _relative(entry["source_root"])
            matches = sorted(p.relative_to(source) for p in source.glob(entry["pattern"]) if p.is_file())
            if not matches:
                raise ExportError(f"approved pattern matched no files: {entry['pattern']}")
        for rel in matches:
            if _prohibited(rel):
                raise ExportError(f"prohibited manifest path: {rel.as_posix()}")
            src = source / rel
            resolved = src.resolve(strict=True)
            if not resolved.is_relative_to(source) or src.is_symlink():
                raise ExportError(f"source escape or symlink rejected: {rel.as_posix()}")
            if "source" in entry:
                dest = _relative(entry["destination"])
            else:
                root = _relative(entry["source_root"])
                dest = _relative(entry["destination_root"]) / rel.relative_to(root)
            key = dest.as_posix()
            if key in mappings and mappings[key]["source"] != rel:
                raise ExportError(f"destination collision: {key}")
            item = dict(entry, source=rel, destination=dest)
            mappings[key] = item
    return [mappings[k] for k in sorted(mappings)], data.get("blocker", [])

def validate(source: Path, entries: list[dict]) -> tuple[list[dict], Counter, int]:
    result, categories, total = [], Counter(), 0
    for entry in entries:
        rel, dest = entry["source"], entry["destination"]
        path = source / rel
        size = path.stat().st_size
        limit = int(entry.get("maximum_bytes", DEFAULT_LIMIT))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if size > limit:
            if entry.get("expected_sha256") != digest or entry.get("expected_bytes") != size or entry.get("review_status") != "approved":
                raise ExportError(f"file exceeds approved size policy: {rel.as_posix()}")
        expected = entry.get("expected_sha256")
        if expected and expected != digest:
            raise ExportError(f"checksum mismatch: {rel.as_posix()}")
        if entry.get("text", True):
            blob = path.read_bytes()
            try: blob.decode("utf-8")
            except UnicodeDecodeError as exc: raise ExportError(f"non-UTF-8 text: {rel.as_posix()}") from exc
            if any(pattern.search(blob) for pattern in PRIVATE_PATTERNS):
                raise ExportError(f"sensitive or private content pattern detected in: {rel.as_posix()}")
        result.append({"source": rel, "destination": dest, "category": entry["category"], "size": size, "sha256": digest})
        categories[entry["category"]] += 1; total += size
    return result, categories, total

def _validate_destination(source: Path, destination: Path, write: bool) -> Path:
    dest = destination.absolute()
    if dest.name == KNOWN_REJECTED_DESTINATION_NAME:
        raise ExportError("the known inaccessible candidate destination is explicitly rejected")
    if dest == source or dest.is_relative_to(source):
        raise ExportError("destination must be outside the private source")
    if write and (dest.exists() or dest.parent.resolve(strict=True) == source):
        raise ExportError("write destination must be fresh, absent, and outside the source")
    return dest

def run(source: Path, manifest: Path, destination: Path, write: bool = False) -> dict:
    source = source.resolve(strict=True)
    destination = _validate_destination(source, destination, write)
    entries, blockers = load_inventory(source, manifest)
    inventory, categories, total = validate(source, entries)
    if write:
        destination.mkdir(parents=False)
        for item in inventory:
            target = destination / item["destination"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / item["source"], target, follow_symlinks=False)
        reread, _, _ = validate(destination, [{"source": x["destination"], "destination": x["destination"], "category": x["category"], "expected_sha256": x["sha256"]} for x in inventory])
        if len(reread) != len(inventory): raise ExportError("written export verification failed")
    return {"inventory": inventory, "categories": dict(sorted(categories.items())), "total_files": len(inventory), "total_bytes": total, "blockers": blockers, "write": write}

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    manifest = args.manifest or args.source / "public_export/manifest.toml"
    try: report = run(args.source, manifest, args.destination, args.write)
    except ExportError as exc:
        print(f"PUBLIC EXPORT REJECTED: {exc}", file=sys.stderr); return 2
    print(f"mode: {'write' if report['write'] else 'dry-run (no destination created)'}")
    print(f"source: {args.source.resolve()}")
    print(f"destination: {args.destination.absolute()}")
    for item in report["inventory"]: print(f"INCLUDE {item['source']} -> {item['destination']} {item['sha256']} {item['size']}")
    print(f"categories: {report['categories']}")
    print(f"total files: {report['total_files']}"); print(f"total bytes: {report['total_bytes']}")
    print("excluded classes: generated outputs, caches, staging, private history, biological/model binaries, obsolete entrypoints, unapproved assets")
    print("scan status: passed; manifest closed and deterministic: yes")
    for blocker in report["blockers"]: print(f"BLOCKER {blocker['kind']}: {blocker['message']}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
