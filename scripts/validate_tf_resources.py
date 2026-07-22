#!/usr/bin/env python3
"""Validate configured CAP-TF-002 resources through its real normalizer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cellstate.app.tf_resource_validation import validate_tf_resource_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dorothea", type=Path,
                        default=os.getenv("CELLSTATE_DOROTHEA_PATH"))
    parser.add_argument("--collectri", type=Path,
                        default=os.getenv("CELLSTATE_COLLECTRI_PATH"))
    args = parser.parse_args()
    results = validate_tf_resource_pair(args.dorothea, args.collectri)
    print(json.dumps({item.database: item.to_dict() for item in results},
                     sort_keys=True, indent=2))
    if not all(item.valid for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
