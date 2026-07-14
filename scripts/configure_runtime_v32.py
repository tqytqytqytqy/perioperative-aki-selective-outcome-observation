#!/usr/bin/env python3
"""Create a private runtime configuration without changing the public template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspire-archive", required=True, type=Path)
    parser.add_argument("--mover-patient-information", required=True, type=Path)
    parser.add_argument("--mover-patient-labs", required=True, type=Path)
    parser.add_argument("--vitaldb-cases", required=True, type=Path)
    parser.add_argument("--vitaldb-labs", required=True, type=Path)
    parser.add_argument("--vitaldb-cache", type=Path)
    parser.add_argument(
        "--public-config",
        type=Path,
        default=ROOT / "repository_release" / "analysis_config_public_v32.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "runtime_config_v32.local.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "inspire_archive": args.inspire_archive,
        "mover_patient_information": args.mover_patient_information,
        "mover_patient_labs": args.mover_patient_labs,
        "vitaldb_cases": args.vitaldb_cases,
        "vitaldb_labs": args.vitaldb_labs,
    }
    if args.vitaldb_cache is not None:
        paths["vitaldb_analysis_cache_verification_only"] = args.vitaldb_cache
    missing = [str(path) for path in paths.values() if not path.expanduser().exists()]
    if missing:
        raise FileNotFoundError("Missing authorized input files: " + "; ".join(missing))

    config = json.loads(args.public_config.read_text(encoding="utf-8"))
    config["analysis_status"] = (
        "scientific_analysis_complete_submission_no_go_pending_author_gates"
    )
    config["raw_data"] = {
        key: str(path.expanduser().resolve()) for key, path in paths.items()
    }
    config.pop("data_access", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
