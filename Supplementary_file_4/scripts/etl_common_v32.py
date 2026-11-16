"""Unchanged R2 calculation functions extracted from authoritative sources.
Path configuration and orchestration are supplied by portable.py, not this module.
Historical report/author metadata generators and legacy entry points are excluded.
"""
from __future__ import annotations


import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from thresholds_r1 import ge_threshold


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "analysis_config_v32.json"
DATA_DIR = ROOT / "local_outputs" / "data"
MODEL_DIR = ROOT / "local_outputs" / "models"
TABLE_DIR = ROOT / "local_outputs" / "tables"
FIGURE_DIR = ROOT / "local_outputs" / "figures"
REPORT_DIR = ROOT / "local_outputs" / "reports"
QA_DIR = ROOT / "local_outputs" / "qa"
TMP_DIR = ROOT / "local_outputs" / "tmp"
MANIFEST_DIR = ROOT / "local_outputs" / "manifest"
LOG_DIR = ROOT / "local_outputs" / "logs"
EXACT = ROOT / "local_outputs" / "diagnostics"


def stable_key(dataset: str, value: Any) -> str:
    text = f"{dataset}|AKI_LOCAL_EVIDENCE_REDO_2_0|{value}".encode("utf-8")
    return hashlib.sha256(text).hexdigest()[:20]

def find_zip_member(archive: zipfile.ZipFile, basename: str) -> str:
    matches = [name for name in archive.namelist() if Path(name).name == basename]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {basename} member, found {len(matches)}")
    return matches[0]

def read_zip_gzip_csv(zip_path: Path, basename: str, **kwargs: Any) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        member = find_zip_member(archive, basename)
        with archive.open(member) as compressed:
            with gzip.GzipFile(fileobj=compressed) as stream:
                return pd.read_csv(stream, low_memory=False, **kwargs)

def iter_zip_gzip_csv(
    zip_path: Path,
    basename: str,
    usecols: Sequence[str] | None = None,
    chunksize: int = 500_000,
) -> Iterable[pd.DataFrame]:
    archive = zipfile.ZipFile(zip_path)
    member = find_zip_member(archive, basename)
    compressed = archive.open(member)
    stream = gzip.GzipFile(fileobj=compressed)
    try:
        for chunk in pd.read_csv(stream, usecols=usecols, chunksize=chunksize, low_memory=False):
            yield chunk
    finally:
        stream.close()
        compressed.close()
        archive.close()

def finite_min(base: pd.Series, candidates: Sequence[pd.Series]) -> pd.Series:
    values = [pd.to_numeric(base, errors="coerce").to_numpy(dtype=float)]
    values.extend(pd.to_numeric(item, errors="coerce").to_numpy(dtype=float) for item in candidates)
    matrix = np.vstack(values)
    matrix[~np.isfinite(matrix)] = np.inf
    result = np.min(matrix, axis=0)
    result[np.isinf(result)] = np.nan
    return pd.Series(result, index=base.index)

def attach_creatinine_outcomes_relative(
    cases: pd.DataFrame,
    labs: pd.DataFrame,
) -> pd.DataFrame:
    cohort = cases.copy()
    rel = labs.merge(
        cohort[["case_raw", "patient_raw", "an_start_num", "an_end_num", "window_48h_end", "window_7d_end"]],
        left_on="patient_raw",
        right_on="patient_raw",
        how="inner",
    )
    pre = rel[
        (rel["lab_time"] < rel["an_start_num"])
        & (rel["lab_time"] >= rel["an_start_num"] - 7 * 1440)
    ]
    if not pre.empty:
        idx = pre.groupby("case_raw", sort=False)["lab_time"].idxmax()
        cohort["baseline_cr"] = cohort["case_raw"].map(pre.loc[idx].set_index("case_raw")["cr"])
    else:
        cohort["baseline_cr"] = np.nan
    post48 = rel[
        (rel["lab_time"] > rel["an_end_num"])
        & (rel["lab_time"] <= rel["window_48h_end"])
    ]
    post7 = rel[
        (rel["lab_time"] > rel["an_end_num"])
        & (rel["lab_time"] <= rel["window_7d_end"])
    ]
    cohort["cr_max_48h"] = cohort["case_raw"].map(post48.groupby("case_raw")["cr"].max())
    cohort["cr_max_7d"] = cohort["case_raw"].map(post7.groupby("case_raw")["cr"].max())
    cohort["postop_cr_48h_count"] = cohort["case_raw"].map(post48.groupby("case_raw")["cr"].size()).fillna(0).astype(int)
    cohort["postop_cr_7d_count"] = cohort["case_raw"].map(post7.groupby("case_raw")["cr"].size()).fillna(0).astype(int)
    return finalize_outcomes(cohort)

def finalize_outcomes(cohort: pd.DataFrame) -> pd.DataFrame:
    out = cohort.copy()
    out["has_baseline_cr"] = out["baseline_cr"].notna()
    out["baseline_cr_under4"] = out["baseline_cr"].lt(4.0)
    out["tested_7d"] = out["postop_cr_7d_count"].gt(0)
    observed = out["has_baseline_cr"] & out["tested_7d"]
    aki_7d = (
        (out["cr_max_48h"].notna() & ge_threshold(out["cr_max_48h"] - out["baseline_cr"], 0.3))
        | (out["cr_max_7d"].notna() & ge_threshold(out["cr_max_7d"], 1.5 * out["baseline_cr"]))
    )
    aki_48h = (
        out["cr_max_48h"].notna()
        & (
            ge_threshold(out["cr_max_48h"] - out["baseline_cr"], 0.3)
            | ge_threshold(out["cr_max_48h"], 1.5 * out["baseline_cr"])
        )
    )
    out["aki"] = np.where(observed, aki_7d.astype(float), np.nan)
    out["aki_48h"] = np.where(observed, aki_48h.astype(float), np.nan)
    return out

def build_inspire(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    zip_path = Path(config["raw_data"]["inspire_archive"])
    ops = read_zip_gzip_csv(zip_path, "operations.csv.gz")
    ops.columns = [str(c).strip() for c in ops.columns]
    numeric = [
        "age",
        "asa",
        "anstart_time",
        "anend_time",
        "discharge_time",
        "inhosp_death_time",
        "cpbon_time",
    ]
    for col in numeric:
        if col in ops:
            ops[col] = pd.to_numeric(ops[col], errors="coerce")
    ops["patient_raw"] = ops["subject_id"].astype(str)
    ops["case_raw"] = ops["op_id"].astype(str)
    ops["age"] = pd.to_numeric(ops["age"], errors="coerce")
    ops["asa"] = pd.to_numeric(ops["asa"], errors="coerce")
    ops["sex_male"] = ops["sex"].fillna("").astype(str).str.upper().str.startswith("M").astype(float)
    ops["an_start_num"] = pd.to_numeric(ops["anstart_time"], errors="coerce")
    ops["an_end_num"] = pd.to_numeric(ops["anend_time"], errors="coerce")
    ops["duration_h"] = (ops["an_end_num"] - ops["an_start_num"]) / 60.0
    dept = ops["department"].fillna("").astype(str).str.upper()
    pcs = ops["icd10_pcs"].fillna("").astype(str).str.upper()
    ops["cardiac_flag"] = dept.eq("CTS") | ops["cpbon_time"].notna() | pcs.str.contains(r"(^|[,;\s])02[A-Z0-9]", regex=True)
    ops["obstetric_flag"] = dept.eq("OG") | pcs.str.contains(r"(?:^|[,;\s])1(?:[A-Z0-9]{4}|[A-Z0-9]{6})(?=$|[,;\s])", regex=True)
    ops["clinical_eligible"] = (
        ops["age"].ge(18)
        & ops["antype"].fillna("").astype(str).str.casefold().eq("general")
        & ops["an_start_num"].notna()
        & ops["an_end_num"].notna()
        & ops["duration_h"].between(0.5, 24.0, inclusive="both")
        & ~ops["cardiac_flag"]
        & ~ops["obstetric_flag"]
    )
    eligible = ops.loc[ops["clinical_eligible"]].copy().sort_values(["patient_raw", "an_start_num", "case_raw"])
    eligible["first_eligible"] = eligible.groupby("patient_raw", sort=False).cumcount().eq(0)
    base_48 = eligible["an_end_num"] + 48 * 60
    base_7d = eligible["an_end_num"] + 7 * 24 * 60
    discharge = pd.to_numeric(eligible["discharge_time"], errors="coerce")
    death = pd.to_numeric(eligible.get("inhosp_death_time", np.nan), errors="coerce")
    valid_discharge = discharge.where(discharge > eligible["an_end_num"])
    valid_death = death.where(death > eligible["an_end_num"])
    eligible["window_48h_end"] = finite_min(base_48, [valid_discharge, valid_death])
    eligible["window_7d_end"] = finite_min(base_7d, [valid_discharge, valid_death])

    lab_parts = []
    scanned = 0
    for chunk in iter_zip_gzip_csv(
        zip_path,
        "labs.csv.gz",
        usecols=["subject_id", "chart_time", "item_name", "value"],
    ):
        scanned += len(chunk)
        mask = chunk["item_name"].fillna("").astype(str).str.strip().str.casefold().eq("creatinine")
        sub = chunk.loc[mask, ["subject_id", "chart_time", "value"]].copy()
        if sub.empty:
            continue
        sub["patient_raw"] = sub["subject_id"].astype(str)
        sub["lab_time"] = pd.to_numeric(sub["chart_time"], errors="coerce")
        sub["cr"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub[sub["lab_time"].notna() & sub["cr"].between(0.05, 30.0, inclusive="both")]
        lab_parts.append(sub[["patient_raw", "lab_time", "cr"]])
    labs = pd.concat(lab_parts, ignore_index=True)
    cohort = attach_creatinine_outcomes_relative(eligible, labs)
    cohort["dataset"] = "INSPIRE"
    cohort["patient_key"] = cohort["patient_raw"].map(lambda x: stable_key("INSPIRE_PATIENT", x))
    cohort["case_key"] = cohort["case_raw"].map(lambda x: stable_key("INSPIRE_CASE", x))
    keep = [
        "dataset",
        "patient_key",
        "case_key",
        "age",
        "sex_male",
        "asa",
        "duration_h",
        "baseline_cr",
        "cr_max_48h",
        "cr_max_7d",
        "postop_cr_48h_count",
        "postop_cr_7d_count",
        "has_baseline_cr",
        "baseline_cr_under4",
        "tested_7d",
        "aki",
        "aki_48h",
        "first_eligible",
        "cardiac_flag",
        "obstetric_flag",
        "an_start_num",
    ]
    cohort = cohort[keep].reset_index(drop=True)
    cohort.to_parquet(DATA_DIR / "inspire_rebuilt_cohort.parquet", index=False)
    audit = {
        "operations_total": int(len(ops)),
        "clinical_eligible_operations": int(len(eligible)),
        "first_clinically_eligible_operations": int(cohort["first_eligible"].sum()),
        "labs_rows_scanned": int(scanned),
        "valid_creatinine_rows": int(len(labs)),
    }
    return cohort, audit
