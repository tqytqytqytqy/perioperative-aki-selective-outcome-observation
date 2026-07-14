#!/usr/bin/env python3
"""Verify a v3.2 SHA-256 manifest without modifying package files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest = pd.read_csv(args.manifest)
    failures = []
    for row in manifest.itertuples(index=False):
        path = root / row.relative_path
        if not path.is_file():
            failures.append(f"missing::{row.relative_path}")
            continue
        if int(path.stat().st_size) != int(row.size_bytes):
            failures.append(f"size::{row.relative_path}")
            continue
        if sha256(path) != row.sha256:
            failures.append(f"sha256::{row.relative_path}")
    print(
        f"manifest={args.manifest}; files={len(manifest)}; "
        f"verified={len(manifest) - len(failures)}; failures={len(failures)}"
    )
    for failure in failures:
        print(failure)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
