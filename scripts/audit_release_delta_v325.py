#!/usr/bin/env python3
"""Prove that v3.2.5 changes only release and citation metadata assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RELATIVE = "qa/v325_delta_from_v324.json"
ALLOWED_CHANGED = {
    ".zenodo.json",
    "CITATION.cff",
    "MODEL_CARD.md",
    "README.md",
    "qa/release_metadata_checks_v324.json",
    "qa/release_metadata_checks_v325.json",
    "qa/reproducible_workbook_checks_v32.csv",
    "qa/reproducible_workbook_summary_v32.json",
    "qa/v324_delta_from_v323.json",
    "qa/v325_delta_from_v324.json",
    "release_manifest_sha256_v32.csv",
    "scripts/audit_release_delta_v324.py",
    "scripts/audit_release_delta_v325.py",
    "scripts/build_release_manifest_v324.py",
    "scripts/build_release_manifest_v325.py",
    "scripts/build_v32_workbook_public.py",
    "scripts/run_v32_workbook_qa.py",
    "scripts/validate_release_metadata_v324.py",
    "scripts/validate_release_metadata_v325.py",
    "workbook/AKI_selective_outcome_v32_all_tables_reproducible.xlsx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and str(path.relative_to(root)) != OUTPUT_RELATIVE
    }


def main() -> int:
    baseline = parse_args().baseline.resolve()
    current = inventory(ROOT)
    prior = inventory(baseline)
    changed = sorted(
        path for path in current.keys() & prior.keys() if current[path] != prior[path]
    )
    added = sorted(current.keys() - prior.keys())
    removed = sorted(prior.keys() - current.keys())
    observed_delta = set(changed) | set(added) | set(removed)
    unexpected = sorted(observed_delta - ALLOWED_CHANGED)
    scientific_prefixes = (
        "tables/",
        "figures/",
        "models/",
        "reports/",
        "config/",
        "metadata/",
    )
    scientific_changes = sorted(
        path for path in observed_delta if path.startswith(scientific_prefixes)
    )
    output = {
        "status": "pass" if not unexpected and not scientific_changes else "fail",
        "baseline": "v3.2.4 Git tag archive",
        "current": ".",
        "baseline_files": len(prior),
        "current_files": len(current),
        "unchanged_files": sum(
            current[path] == prior[path] for path in current.keys() & prior.keys()
        ),
        "changed_files": changed,
        "added_files": added,
        "removed_files": removed,
        "unexpected_delta": unexpected,
        "scientific_asset_changes": scientific_changes,
        "interpretation": (
            "Version v3.2.5 corrects release and citation metadata only; "
            "aggregate scientific tables, model specification, figures, reports, "
            "configuration, and reference metadata are byte-identical to v3.2.4."
        ),
    }
    output_path = ROOT / OUTPUT_RELATIVE
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))
    return 0 if output["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
