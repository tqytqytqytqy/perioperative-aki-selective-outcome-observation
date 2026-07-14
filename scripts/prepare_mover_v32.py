#!/usr/bin/env python3
"""Rebuild the v3.2 MOVER cohort from the authorized EPIC EMR files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import etl_common_v32 as base
import etl_outcome_representations_v32 as representation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(
    os.environ.get("V32_CONFIG_PATH", ROOT / "config" / "analysis_config_v32.json")
).expanduser().resolve()
OUTPUT = ROOT / "data" / "processed" / "mover_rebuilt_v32.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-strict-creatinine", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    base.ensure_dirs()
    labs, lab_audit = base.extract_mover_creatinine(
        config, reuse=args.reuse_strict_creatinine
    )
    cohort, cohort_audit = base.build_mover(config, labs)
    cohort = representation.attach_mover_surgery_category(cohort, config)
    edges, representatives = representation.percentile_release_parameters(labs["cr"])
    rows = []
    for index, representative_value in enumerate(representatives):
        rows.append(
            {
                "bin_index": index + 1,
                "lower_cutpoint_mg_dl": np.nan if index == 0 else edges[index - 1],
                "upper_cutpoint_mg_dl": np.nan if index == len(representatives) - 1 else edges[index],
                "released_representative_mg_dl": representative_value,
            }
        )
    for column in ["baseline_cr", "cr_max_48h", "cr_max_7d"]:
        coarse = representation.apply_percentile_release(cohort[column], edges, representatives)
        cohort[f"{column}_coarse"] = coarse
        lower, upper = representation.interval_bounds_from_levels(
            coarse, representatives, 0.05, 30.0
        )
        cohort[f"{column}_coarse_lower"] = lower
        cohort[f"{column}_coarse_upper"] = upper
    cohort["outcome_operational"] = cohort["aki"]
    cohort["outcome_coarsened_operational"] = representation.classify_exact_aki(
        cohort, "baseline_cr_coarse", "cr_max_48h_coarse", "cr_max_7d_coarse"
    )
    cohort["outcome_definite"], cohort["outcome_possible"] = representation.classify_interval_aki(
        cohort,
        "baseline_cr_coarse_lower",
        "baseline_cr_coarse_upper",
        "cr_max_48h_coarse_lower",
        "cr_max_48h_coarse_upper",
        "cr_max_7d_coarse_lower",
        "cr_max_7d_coarse_upper",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(OUTPUT, index=False)
    pd.DataFrame(rows).to_csv(ROOT / "tables" / "51_mover_deterministic_coarsening_v32.csv", index=False)
    audit = {"cohort": cohort_audit, "strict_creatinine": lab_audit, "output_rows": len(cohort)}
    (ROOT / "reports" / "mover_etl_audit_v32.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), **audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
