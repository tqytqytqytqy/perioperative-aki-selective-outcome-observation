#!/usr/bin/env python3
"""Prove that v3.2.3 changes only release and author metadata assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_CHANGED = {
    ".zenodo.json",
    "CITATION.cff",
    "LICENSE-CODE",
    "MODEL_CARD.md",
    "README.md",
    "qa/release_metadata_checks_v323.json",
    "qa/reproducible_workbook_checks_v32.csv",
    "qa/reproducible_workbook_summary_v32.json",
    "qa/v323_delta_from_v322.json",
    "release_manifest_sha256_v32.csv",
    "scripts/audit_release_delta_v323.py",
    "scripts/author_metadata_v32.py",
    "scripts/build_release_manifest_v323.py",
    "scripts/build_v32_workbook_public.py",
    "scripts/run_v32_workbook_qa.py",
    "scripts/validate_release_metadata_v323.py",
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
    excluded = {"qa/v323_delta_from_v322.json"}
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and str(path.relative_to(root)) not in excluded
    }


def main() -> int:
    args = parse_args()
    baseline = args.baseline.resolve()
    current = inventory(ROOT)
    prior = inventory(baseline)
    changed = sorted(
        path for path in current.keys() & prior.keys() if current[path] != prior[path]
    )
    added = sorted(current.keys() - prior.keys())
    removed = sorted(prior.keys() - current.keys())
    observed_delta = set(changed) | set(added) | set(removed)
    unexpected = sorted(observed_delta - ALLOWED_CHANGED)
    unchanged = sorted(path for path in current.keys() & prior.keys() if current[path] == prior[path])
    scientific_prefixes = ("tables/", "figures/", "models/", "reports/", "config/", "metadata/")
    scientific_changes = sorted(
        path for path in observed_delta if path.startswith(scientific_prefixes)
    )
    output = {
        "status": "pass" if not unexpected and not removed and not scientific_changes else "fail",
        "baseline": "v3.2.2 extracted release root",
        "current": ".",
        "baseline_files": len(prior),
        "current_files": len(current),
        "unchanged_files": len(unchanged),
        "changed_files": changed,
        "added_files": added,
        "removed_files": removed,
        "unexpected_delta": unexpected,
        "scientific_asset_changes": scientific_changes,
        "interpretation": (
            "Version v3.2.3 changes release and author metadata only; aggregate scientific "
            "assets, model specification, figures, and reports are byte-identical to v3.2.2."
        ),
    }
    output_path = ROOT / "qa" / "v323_delta_from_v322.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if output["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
