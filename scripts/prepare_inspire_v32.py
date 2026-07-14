#!/usr/bin/env python3
"""Rebuild the v3.2 INSPIRE cohort from the authorized release archive."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

import etl_common_v32 as base
import etl_outcome_representations_v32 as representation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(
    os.environ.get("V32_CONFIG_PATH", ROOT / "config" / "analysis_config_v32.json")
).expanduser().resolve()
OUTPUT = ROOT / "data" / "processed" / "inspire_rebuilt_v32.parquet"


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    base.ensure_dirs()
    cohort, audit = base.build_inspire(config)
    levels, release_table = representation.inspire_release_levels(config)
    for column in ["baseline_cr", "cr_max_48h", "cr_max_7d"]:
        lower, upper = representation.interval_bounds_from_levels(
            cohort[column], levels, 0.05, 30.0
        )
        cohort[f"{column}_lower"] = lower
        cohort[f"{column}_upper"] = upper
    cohort["outcome_operational"] = cohort["aki"]
    cohort["outcome_definite"], cohort["outcome_possible"] = representation.classify_interval_aki(
        cohort,
        "baseline_cr_lower",
        "baseline_cr_upper",
        "cr_max_48h_lower",
        "cr_max_48h_upper",
        "cr_max_7d_lower",
        "cr_max_7d_upper",
    )
    cohort["baseline_cr_coarse"] = cohort["baseline_cr"]
    cohort["outcome_coarsened_operational"] = cohort["outcome_operational"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(OUTPUT, index=False)
    release_table.to_csv(ROOT / "tables" / "50_inspire_creatinine_release_levels_v32.csv", index=False)
    (ROOT / "reports" / "inspire_etl_audit_v32.json").write_text(
        json.dumps({**audit, "output_rows": len(cohort)}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "rows": len(cohort), **audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
