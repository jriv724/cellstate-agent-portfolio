from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_public_export import ExportError, load_inventory, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "public_export/manifest.toml"

def test_repository_manifest_is_closed_and_excludes_obsolete_material():
    entries, blockers = load_inventory(ROOT, MANIFEST)
    sources = {x["source"].as_posix() for x in entries}
    assert "app.py" not in sources and "app_backup.py" not in sources
    assert "src/cellstate/data_interchange/reconstruct_seurat.R" not in sources
    assert "src/cellstate/data_interchange/capabilities.py" in sources
    assert any(x["kind"] == "license" for x in blockers)

def test_dry_run_does_not_create_destination(tmp_path):
    destination = tmp_path / "absent"
    report = run(ROOT, MANIFEST, destination)
    assert report["total_files"] > 40
    assert not destination.exists()

def _repo(tmp_path, content="safe\n"):
    root = tmp_path / "source"; root.mkdir(); (root / "safe.txt").write_text(content)
    return root

def _manifest(tmp_path, body):
    path = tmp_path / "manifest.toml"; path.write_text("version=1\n" + body)
    return path

@pytest.mark.parametrize("source,destination", [("../escape", "x"), ("safe.txt", "../escape"), (".git/config", "x")])
def test_path_traversal_and_git_rejected(tmp_path, source, destination):
    root = _repo(tmp_path); manifest = _manifest(tmp_path, f'[[include]]\nsource="{source}"\ndestination="{destination}"\ncategory="test"\ntext=true\n')
    with pytest.raises(ExportError): load_inventory(root, manifest)

def test_escaping_symlink_rejected(tmp_path):
    root = _repo(tmp_path); outside = tmp_path / "outside"; outside.write_text("safe")
    (root / "link.txt").symlink_to(outside)
    manifest = _manifest(tmp_path, '[[include]]\nsource="link.txt"\ndestination="link.txt"\ncategory="test"\ntext=true\n')
    with pytest.raises(ExportError): load_inventory(root, manifest)

@pytest.mark.parametrize("name", ["data.h5ad", "weights.ckpt", "movie.mp4"])
def test_forbidden_extensions_rejected(tmp_path, name):
    root = _repo(tmp_path); (root / name).write_bytes(b"x")
    manifest = _manifest(tmp_path, f'[[include]]\nsource="{name}"\ndestination="{name}"\ncategory="test"\ntext=false\n')
    with pytest.raises(ExportError): load_inventory(root, manifest)

def test_size_and_checksum_enforcement(tmp_path):
    root = _repo(tmp_path, "12345")
    manifest = _manifest(tmp_path, '[[include]]\nsource="safe.txt"\ndestination="safe.txt"\ncategory="test"\ntext=true\nmaximum_bytes=2\n')
    with pytest.raises(ExportError): run(root, manifest, tmp_path / "dest")
    digest = hashlib.sha256(b"12345").hexdigest()
    approved = _manifest(tmp_path, f'[[include]]\nsource="safe.txt"\ndestination="safe.txt"\ncategory="asset"\ntext=true\nmaximum_bytes=2\nexpected_bytes=5\nexpected_sha256="{digest}"\nreview_status="approved"\n')
    assert run(root, approved, tmp_path / "approved")["total_files"] == 1

@pytest.mark.parametrize(
    "content",
    [
        "/" + "samu" + "rlab1/private/path\n",
        "OPEN" + "AI_API_" + "KEY=not-a-real-secret-value\n",
        "-----BEGIN " + "PRI" + "VATE KEY-----\n",
    ],
)
def test_private_and_credential_patterns_rejected_without_value(tmp_path, content, capsys):
    root = _repo(tmp_path, content); manifest = _manifest(tmp_path, '[[include]]\nsource="safe.txt"\ndestination="safe.txt"\ncategory="test"\ntext=true\n')
    with pytest.raises(ExportError) as error: run(root, manifest, tmp_path / "dest")
    assert content.strip() not in str(error.value)

def test_destination_inside_source_rejected(tmp_path):
    root = _repo(tmp_path); manifest = _manifest(tmp_path, '[[include]]\nsource="safe.txt"\ndestination="safe.txt"\ncategory="test"\ntext=true\n')
    with pytest.raises(ExportError): run(root, manifest, root / "export")
