#!/usr/bin/env python3
"""Prepare versioned CAP-TF-002 resources from trusted package/API sources."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlopen

import pandas as pd

from cellstate.app.tf_resource_validation import validate_tf_resource_pair


COLLECTRI_URL = (
    "https://omnipathdb.org/interactions?datasets=collectri&organisms=9606"
    "&format=tsv&genesymbols=1"
)
DOROTHEA_VERSION = "1.18.0"


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_dorothea(destination: Path, rscript: str) -> str:
    expression = (
        "suppressPackageStartupMessages(library(dorothea)); data(dorothea_hs); "
        "x<-dorothea_hs[dorothea_hs$confidence %in% c('A','B','C'),]; "
        "x<-data.frame(source=x$tf,target=x$target,weight=x$mor,"
        "confidence=x$confidence,organism='human'); "
        f"write.table(x,{json.dumps(str(destination))},sep='\\t',quote=FALSE,row.names=FALSE); "
        "cat(as.character(packageVersion('dorothea')))"
    )
    result = subprocess.run([rscript, "-e", expression], check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def prepare_collectri(raw_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, sep="\t", keep_default_na=False)
    required = {"source", "source_genesymbol", "target_genesymbol",
                "is_stimulation", "is_inhibition"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"official CollecTRI table missing columns: {missing}")
    table = raw.copy()
    complex_mask = table.source.astype(str).str.contains("COMPLEX", regex=False)
    symbol = table.source_genesymbol.astype(str)
    table.loc[complex_mask & symbol.str.contains("JUN|FOS", regex=True),
              "source_genesymbol"] = "AP1"
    table.loc[complex_mask & symbol.str.contains("REL|NFKB", regex=True),
              "source_genesymbol"] = "NFKB"
    table = table.loc[~complex_mask | table.source_genesymbol.isin(["AP1", "NFKB"])]
    table = table.drop_duplicates(["source_genesymbol", "target_genesymbol"],
                                  keep="first")
    stimulation = table.is_stimulation.astype(str).str.casefold().isin({"true", "1"})
    return pd.DataFrame({
        "source": table.source_genesymbol,
        "target": table.target_genesymbol,
        "weight": stimulation.map({True: 1, False: -1}),
        "organism": "human",
    }).sort_values(["source", "target"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("resources/tf_activity"))
    parser.add_argument("--collectri-source", type=Path)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dorothea_path = output / f"dorothea_human_abc_v{DOROTHEA_VERSION}.tsv"
    collectri_path = output / "collectri_human_omnipath_2026-07-21.tsv"
    with tempfile.TemporaryDirectory() as temporary:
        raw_collectri = args.collectri_source or Path(temporary) / "collectri.tsv"
        if args.collectri_source is None:
            with urlopen(COLLECTRI_URL) as response:
                raw_collectri.write_bytes(response.read())
        package_version = export_dorothea(dorothea_path, args.rscript)
        collectri = prepare_collectri(raw_collectri)
        collectri.to_csv(collectri_path, sep="\t", index=False,
                         lineterminator="\n")
        validations = validate_tf_resource_pair(dorothea_path, collectri_path)
        if not all(item.valid for item in validations):
            raise RuntimeError("prepared resources failed CAP-TF-002 validation: " +
                               "; ".join(str(item.error) for item in validations))
        metadata = {
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "preparation_script": "scripts/prepare_tf_resources.py",
            "resources": {
                "DoRothEA": {
                    "path": str(dorothea_path),
                    "source": "Bioconductor dorothea package data(dorothea_hs)",
                    "package_version": package_version,
                    "confidence_levels": ["A", "B", "C"],
                    "file_sha256": file_hash(dorothea_path),
                    "validation": validations[0].to_dict(),
                },
                "CollecTRI": {
                    "path": str(collectri_path),
                    "source": "OmniPath official interactions API",
                    "source_url": COLLECTRI_URL,
                    "raw_source_sha256": file_hash(raw_collectri),
                    "transformation": "decoupleR get_collectri human, split_complexes=FALSE semantics",
                    "file_sha256": file_hash(collectri_path),
                    "validation": validations[1].to_dict(),
                },
            },
        }
        (output / "tf_resources_2026-07-21.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    print(dorothea_path)
    print(collectri_path)


if __name__ == "__main__":
    main()
