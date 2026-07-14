#!/usr/bin/env python3
"""Rebuild the v3.2 VitalDB supportive cohort from official API tables."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import etl_common_v32 as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(
    os.environ.get("V32_CONFIG_PATH", ROOT / "config" / "analysis_config_v32.json")
).expanduser().resolve()
OUTPUT = ROOT / "data" / "processed" / "vitaldb_supportive_v32.parquet"


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    base.ensure_dirs()
    cases = pd.read_csv(
        Path(config["raw_data"]["vitaldb_cases"]), encoding="utf-8-sig", low_memory=False
    )
    labs = pd.read_csv(
        Path(config["raw_data"]["vitaldb_labs"]), encoding="utf-8-sig", low_memory=False
    )
    creatinine = labs.loc[
        labs["name"].fillna("").astype(str).str.strip().str.casefold().eq("cr")
    ].copy()
    creatinine["result"] = pd.to_numeric(creatinine["result"], errors="coerce")
    creatinine = creatinine.loc[creatinine["result"].notna()].merge(
        cases[["caseid", "opstart", "opend", "preop_cr"]], on="caseid", how="left"
    )

    pre = creatinine.loc[creatinine["dt"] < creatinine["opstart"]]
    pre = pre.sort_values(["caseid", "dt"]).groupby("caseid", sort=False).tail(1)
    post48 = creatinine.loc[
        (creatinine["dt"] > creatinine["opend"])
        & (creatinine["dt"] <= creatinine["opend"] + 2 * 86400)
    ]
    post7 = creatinine.loc[
        (creatinine["dt"] > creatinine["opend"])
        & (creatinine["dt"] <= creatinine["opend"] + 7 * 86400)
    ]
    derived = cases.set_index("caseid").copy()
    derived["baseline_cr_lab"] = pre.set_index("caseid")["result"]
    derived["baseline_cr"] = pd.to_numeric(
        derived["preop_cr"], errors="coerce"
    ).combine_first(derived["baseline_cr_lab"])
    derived["cr_max_48h"] = post48.groupby("caseid")["result"].max()
    derived["cr_max_7d"] = post7.groupby("caseid")["result"].max()
    derived["postop_cr_7d_count"] = (
        post7.groupby("caseid")["result"].count().reindex(derived.index).fillna(0)
    )
    derived["tested_7d"] = derived["postop_cr_7d_count"].gt(0)
    derived["has_baseline_cr"] = derived["baseline_cr"].notna() & derived[
        "baseline_cr"
    ].gt(0)
    derived["baseline_cr_under4"] = derived["baseline_cr"].lt(4.0)
    derived["aki"] = (
        (derived["cr_max_48h"] - derived["baseline_cr"] >= 0.3)
        | (derived["cr_max_7d"] / derived["baseline_cr"] >= 1.5)
    ).astype(float)
    derived.loc[
        ~(derived["has_baseline_cr"] & derived["tested_7d"]), "aki"
    ] = np.nan
    derived["dataset"] = "VitalDB"
    derived["patient_key"] = [
        base.stable_key("VITALDB_CASE", caseid) for caseid in derived.index
    ]
    derived["case_key"] = derived["patient_key"]
    derived["age"] = pd.to_numeric(derived["age"], errors="coerce")
    derived["sex_male"] = (
        derived["sex"].fillna("").astype(str).str.upper().str.startswith("M").astype(float)
    )
    derived["asa"] = pd.to_numeric(derived["asa"], errors="coerce")
    derived["duration_h"] = (derived["opend"] - derived["opstart"]) / 3600.0
    derived["adult_general_duration"] = (
        derived["age"].ge(18)
        & derived["ane_type"].fillna("").astype(str).str.contains("General", case=False)
        & derived["opend"].gt(derived["opstart"])
        & derived["duration_h"].ge(0.5)
    )
    derived["aki_48h"] = np.nan
    cohort = derived[
        [
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
            "aki_48h",
            "adult_general_duration",
            "has_baseline_cr",
            "baseline_cr_under4",
            "postop_cr_7d_count",
        ]
    ].reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(OUTPUT, index=False)
    observed = (
        cohort["has_baseline_cr"].fillna(False).astype(bool)
        & cohort["baseline_cr_under4"].fillna(False).astype(bool)
        & cohort["tested_7d"].fillna(False).astype(bool)
        & cohort["aki"].notna()
    )
    audit = {
        "output": str(OUTPUT),
        "retained_cache_rows": len(cohort),
        "available_analysis_ready_observed_n": int(observed.sum()),
        "observed_events": int(cohort.loc[observed, "aki"].sum()),
        "official_released_case_denominator_reconstructed": True,
        "source_compatible_all_eligible_denominator_used": False,
        "raw_cases_rows": len(cases),
        "raw_labs_rows": len(labs),
        "valid_creatinine_rows": len(creatinine),
        "postoperative_creatinine_values_above_30_mg_dl": int(
            (
                (creatinine["dt"] > creatinine["opend"])
                & (creatinine["dt"] <= creatinine["opend"] + 7 * 86400)
                & creatinine["result"].gt(30)
            ).sum()
        ),
        "creatinine_range_note": "Numeric VitalDB API values were retained to reproduce the locked supportive cache; two postoperative values exceeded 30 mg/dL.",
        "source_note": "official VitalDB API cases and labs tables retained on 2026-06-17",
    }
    (ROOT / "reports" / "vitaldb_etl_audit_v32.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
