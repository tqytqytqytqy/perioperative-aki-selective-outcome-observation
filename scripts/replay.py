#!/usr/bin/env python3
"""Supported local replay entrypoints. No downloads, uploads, or publishing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parents[1]
MARKER = ".scientific-replay.json"


def validate_workdir(path: Path, package: Path = PACKAGE) -> Path:
    path, package = path.expanduser().resolve(), package.resolve()
    if path == package or package in path.parents or path in package.parents:
        raise ValueError("Replay workspace must be separate from the public package and its ancestors.")
    return path


def read_runtime(path: Path) -> dict:
    path = path.expanduser().resolve()
    config = json.loads(path.read_text())
    config["raw_data"] = {
        key: str((path.parent / Path(value).expanduser()).resolve())
        for key, value in config["raw_data"].items()
    }
    return config


def require_files(work: Path, names: list[str]) -> None:
    missing = [name for name in names if not (work / name).is_file()]
    if missing:
        raise ValueError("Missing private replay prerequisites: " + ", ".join(missing))


def initialize(args) -> None:
    work = validate_workdir(args.workdir)
    if work.exists() and any(work.iterdir()):
        raise ValueError("Initialization requires a new or empty workspace; nothing is overwritten.")
    config = read_runtime(args.config)
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    work.chmod(0o700)
    for name in ["scripts", "config", "tables", "models", "figures", "reports",
                 "qa", "private", "audit", "data/processed", "logs", "manifest"]:
        (work / name).mkdir(parents=True, exist_ok=True)
    for source in (PACKAGE / "scripts").glob("*.py"):
        if source.name not in {"replay.py", "verify_public.py", "public_disclosure.py"}:
            shutil.copy2(source, work / "scripts" / source.name)
    # A static variable-definition table is needed by the original finalizer.
    mapping = "48_raw_to_analysis_variable_map_v32.csv"
    shutil.copy2(PACKAGE / "tables" / mapping, work / "tables" / mapping)
    for name in ["analysis_config_v32.json", "analysis_config.json"]:
        (work / "config" / name).write_text(json.dumps(config, indent=2) + "\n")
    (work / MARKER).write_text(json.dumps({"scientific_release": "3.3.0",
                                         "private_workspace": True}) + "\n")
    print("Initialized private replay workspace. No raw files were read and no analysis was run.")


def run_child(work: Path, name: str, options=()) -> None:
    env = dict(os.environ)
    env["V32_CONFIG_PATH"] = str(work / "config/analysis_config_v32.json")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        env[key] = "1"
    subprocess.run([sys.executable, str(work / "scripts" / name), *options],
                   cwd=work, env=env, check=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Initialize a separate empty private workspace.")
    init.add_argument("--workdir", type=Path, required=True)
    init.add_argument("--config", type=Path, required=True)
    verify = sub.add_parser("verify", help="Read-only verification of shipped aggregate outputs.")
    verify.add_argument("--report", type=Path)
    for command in ["etl", "analysis", "supplemental", "finalize", "raw-audit",
                    "ties", "retained-correction", "spotcheck"]:
        p = sub.add_parser(command)
        p.add_argument("--workdir", type=Path, required=True)
        p.add_argument("--execute", action="store_true", required=True,
                       help="Explicitly run computations in the private workspace.")
        if command == "analysis":
            p.add_argument("--bootstrap-replicates", type=int, default=1000)
            p.add_argument("--jobs", type=int, default=6)
            p.add_argument("--point-only", action="store_true")
            p.add_argument("--resume", action="store_true",
                           help="Explicitly permit existing bootstrap checkpoints.")
    args = parser.parse_args(argv)
    if args.command == "init":
        initialize(args)
        return 0
    if args.command == "verify":
        from verify_public import verify_package
        report = verify_package(PACKAGE)
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    work = validate_workdir(args.workdir)
    require_files(work, [MARKER, "config/analysis_config_v32.json"])
    cohorts = ["data/processed/" + f for f in [
        "inspire_rebuilt_v32.parquet", "mover_rebuilt_v32.parquet",
        "vitaldb_supportive_v32.parquet"]]
    if args.command == "etl":
        cfg = json.loads((work / "config/analysis_config_v32.json").read_text())
        if not all(Path(p).is_file() for p in cfg["raw_data"].values()):
            raise ValueError("At least one configured raw input is unavailable. Nothing downloaded.")
        for name in ["prepare_inspire_v32.py", "prepare_mover_v32.py", "prepare_vitaldb_v32.py"]:
            run_child(work, name)
    elif args.command == "retained-correction":
        require_files(work, ["data/processed/inspire_original_v325.parquet",
                             "data/processed/mover_original_v325.parquet"])
        run_child(work, "aki_r1_audit.py", ["--execute"])
        # The correction script expects these names even when a backup already exists.
        for stem in ["inspire", "mover"]:
            dest = work / f"data/processed/{stem}_rebuilt_v32.parquet"
            if not dest.exists():
                shutil.copy2(work / f"data/processed/{stem}_original_v325.parquet", dest)
        run_child(work, "prepare_r1_corrected.py", ["--execute"])
        run_child(work, "prepare_vitaldb_v32.py")
    else:
        require_files(work, cohorts)
        if args.command == "analysis":
            checkpoints = list((work / "data/processed").glob("*.partial.*"))
            if checkpoints and not args.resume:
                raise ValueError("Existing checkpoints detected; choose --resume or a new workspace.")
            options = ["--bootstrap-replicates", str(args.bootstrap_replicates),
                       "--jobs", str(args.jobs)]
            if args.point_only:
                options.append("--point-only")
            run_child(work, "run_v32_analysis.py", options)
        elif args.command == "supplemental":
            require_files(work, ["models/source_logistic_spline_ipw_v32.joblib",
                                 "models/model_specification_v32.json",
                                 "tables/51_mover_deterministic_coarsening_v32.csv"])
            run_child(work, "run_r1_supplemental.py", ["--execute"])
        elif args.command == "finalize":
            require_files(work, ["data/processed/canonical_bootstrap_distribution_v32.parquet",
                                 "data/processed/mnar_chain_bootstrap_v32.parquet"])
            run_child(work, "finalize_r1_analysis.py", ["--execute"])
        elif args.command == "raw-audit":
            require_files(work, ["data/processed/inspire_original_v325.parquet",
                                 "data/processed/mover_original_v325.parquet",
                                 "data/processed/vitaldb_supportive_v2.parquet"])
            run_child(work, "independent_raw_audit.py", ["--execute"])
        elif args.command == "ties":
            require_files(work, ["private/record_level_evidence_PRIVATE.csv",
                                 "tables/31_canonical_primary_bootstrap_v32.csv"])
            run_child(work, "tie_sensitivity.py", ["--execute"])
        elif args.command == "spotcheck":
            require_files(work, ["models/source_logistic_spline_ipw_v32.joblib",
                                 "data/processed/canonical_bootstrap_distribution_v32.parquet",
                                 "data/processed/mnar_chain_bootstrap_v32.parquet"])
            run_child(work, "spotcheck_aki_r1.py", ["--execute"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
