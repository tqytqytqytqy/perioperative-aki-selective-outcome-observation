#!/usr/bin/env python3
"""Build the deterministic SHA-256 manifest for the v3.2.3 public package."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release_manifest_sha256_v32.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path != OUTPUT
        and ".git" not in relative.parts
        and "__pycache__" not in relative.parts
        and path.suffix != ".pyc"
        and path.name != ".DS_Store"
    )


def main() -> int:
    rows = []
    for path in sorted(path for path in ROOT.rglob("*") if included(path)):
        rows.append(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "size_bytes", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Built {OUTPUT} with {len(rows)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
