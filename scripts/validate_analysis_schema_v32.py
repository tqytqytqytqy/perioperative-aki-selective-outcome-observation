#!/usr/bin/env python3
"""Validate the v3.2 rebuilt cohorts and their analysis contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REQUIRED = [
    "dataset",
    "patient_key",
    "case_key",
    "age",
    "sex_male",
    "asa",
    "duration_h",
    "baseline_cr",
    "tested_7d",
    "aki",
]


def eligible(frame: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    mask = frame["has_baseline_cr"].fillna(False).astype(bool)
    mask &= frame["baseline_cr_under4"].fillna(False).astype(bool)
    if "first_eligible" in frame:
        mask &= frame["first_eligible"].fillna(False).astype(bool)
    if year is not None:
        mask &= frame["year"].eq(year)
    return frame.loc[mask].copy()


def summary(frame: pd.DataFrame, year: int | None = None) -> dict[str, int]:
    part = eligible(frame, year)
    observed = part["tested_7d"].fillna(False).astype(bool) & part["outcome_operational"].notna()
    return {
        "eligible": len(part),
        "observed": int(observed.sum()),
        "unobserved": int((~observed).sum()),
        "observed_events": int(part.loc[observed, "outcome_operational"].sum()),
    }


def main() -> int:
    paths = {
        "INSPIRE": DATA / "inspire_rebuilt_v32.parquet",
        "MOVER": DATA / "mover_rebuilt_v32.parquet",
        "VitalDB": DATA / "vitaldb_supportive_v32.parquet",
    }
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    checks: list[dict[str, object]] = []
    for name, frame in frames.items():
        missing = sorted(set(REQUIRED) - set(frame.columns))
        checks.append({"check": f"{name}_required_columns", "passed": not missing, "detail": ";".join(missing)})
        checks.append({"check": f"{name}_unique_case_key", "passed": frame["case_key"].is_unique, "detail": str(frame["case_key"].nunique())})
        checks.append({"check": f"{name}_finite_duration", "passed": np.isfinite(frame["duration_h"].dropna()).all(), "detail": str(len(frame))})
    actual = {
        "INSPIRE": summary(frames["INSPIRE"]),
        "MOVER 2021": summary(frames["MOVER"], 2021),
        "MOVER 2022": summary(frames["MOVER"], 2022),
    }
    expected = {
        "INSPIRE": {"eligible": 33396, "observed": 24874, "unobserved": 8522, "observed_events": 1671},
        "MOVER 2021": {"eligible": 2802, "observed": 2212, "unobserved": 590, "observed_events": 245},
        "MOVER 2022": {"eligible": 2587, "observed": 2033, "unobserved": 554, "observed_events": 224},
    }
    for phase in expected:
        checks.append({"check": f"{phase}_locked_counts", "passed": actual[phase] == expected[phase], "detail": json.dumps(actual[phase])})
    mover = frames["MOVER"]
    update = eligible(mover, 2021)
    target = eligible(mover, 2022)
    overlap = len(set(update["patient_key"]) & set(target["patient_key"]))
    checks.append({"check": "MOVER_2021_2022_patient_overlap_zero", "passed": overlap == 0, "detail": str(overlap)})
    result = pd.DataFrame(checks)
    (ROOT / "qa").mkdir(parents=True, exist_ok=True)
    result.to_csv(ROOT / "qa" / "analysis_schema_checks_v32.csv", index=False)
    payload = {"status": "pass" if result["passed"].all() else "fail", "checks": len(result), "actual_counts": actual}
    (ROOT / "qa" / "analysis_schema_summary_v32.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not result["passed"].all():
        raise RuntimeError(result.loc[~result["passed"]].to_json(orient="records"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
