#!/usr/bin/env python3
"""Raw-cohort reconstruction helpers retained for the v3.2 audit."""

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

matplotlib.use("Agg")
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

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*feature names.*")

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(
    os.environ.get("V32_CONFIG_PATH", ROOT / "config" / "analysis_config_v32.json")
).expanduser().resolve()
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
TABLE_DIR = ROOT / "tables"
FIGURE_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"
QA_DIR = ROOT / "qa"
MANIFEST_DIR = ROOT / "manifest"
LOG_DIR = ROOT / "logs"

FEATURES = ["age", "sex_male", "duration_h", "baseline_cr"]
CONTINUOUS = ["age", "duration_h", "baseline_cr"]
BINARY = ["sex_male"]

CARDIAC_RE = re.compile(
    r"CABG|CORONARY\s+ARTERY\s+BYPASS|CATHETERIZATION,?\s*HEART|"
    r"CARDIAC|CARDIOPULMONARY\s+BYPASS|HEART\s+TRANSPLANT|"
    r"VENTRICULAR\s+ASSIST|VALVE\s+(REPAIR|REPLACEMENT)|"
    r"AORTIC\s+VALVE|MITRAL\s+VALVE|TRICUSPID\s+VALVE|"
    r"TRANSESOPHAGEAL\s+ECHOCARDIOGRAM|PACEMAKER",
    re.I,
)
OBSTETRIC_RE = re.compile(
    r"CESAREAN|C-SECTION|DELIVERY|OBSTETRIC|LABOR|PLACENTA|"
    r"ECTOPIC\s+PREGNANCY|ABORTION|DILATION\s+AND\s+EVACUATION|"
    r"DILATION\s+AND\s+CURETTAGE",
    re.I,
)
MOVER_BLOOD_CREATININE_LOINC = {"2160-0", "38483-4"}


@dataclass(frozen=True)
class Bounds:
    oe: tuple[float, float]
    slope: tuple[float, float]


def ensure_dirs() -> None:
    for path in [DATA_DIR, MODEL_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR, QA_DIR, MANIFEST_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    out.to_csv(path, index=False)
    return path


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def source_file_audit(config: dict[str, Any], skip_hash: bool = False) -> pd.DataFrame:
    rows = []
    volume_root = Path("/", "Volumes")
    mounted = sorted(str(p) for p in volume_root.iterdir()) if volume_root.exists() else []
    configured_paths = [Path(value).expanduser() for value in config["raw_data"].values()]
    external_paths = [path for path in configured_paths if volume_root in path.parents]
    for role, raw_path in config["raw_data"].items():
        path = Path(raw_path)
        stat = path.stat() if path.exists() else None
        rows.append(
            {
                "source_role": role,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": stat.st_size if stat else np.nan,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat() if stat else "",
                "sha256": "SKIPPED" if skip_hash and path.exists() else (sha256_file(path) if path.exists() else ""),
                "storage": "external_volume" if volume_root in path.parents else "local_copy",
            }
        )
    table = pd.DataFrame(rows)
    write_table(table, "01_source_file_audit.csv")
    write_json(
        {
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "mounted_volumes": mounted,
            "external_data_volume_expected_but_not_mounted": bool(external_paths)
            and not all(path.exists() for path in external_paths),
            "analysis_used_local_raw_or_analysis_ready_copies": True,
        },
        MANIFEST_DIR / "storage_audit.json",
    )
    return table


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


def attach_creatinine_outcomes_datetime(
    cases: pd.DataFrame,
    labs: pd.DataFrame,
) -> pd.DataFrame:
    cohort = cases.copy()
    rel = labs.merge(
        cohort[["case_raw", "an_start", "an_end", "window_48h_end", "window_7d_end"]],
        left_on="case_raw",
        right_on="case_raw",
        how="inner",
    )
    pre = rel[
        (rel["lab_time"] < rel["an_start"])
        & (rel["lab_time"] >= rel["an_start"] - pd.Timedelta(days=7))
    ]
    if not pre.empty:
        idx = pre.groupby("case_raw", sort=False)["lab_time"].idxmax()
        cohort["baseline_cr"] = cohort["case_raw"].map(pre.loc[idx].set_index("case_raw")["cr"])
    else:
        cohort["baseline_cr"] = np.nan
    post48 = rel[
        (rel["lab_time"] > rel["an_end"])
        & (rel["lab_time"] <= rel["window_48h_end"])
    ]
    post7 = rel[
        (rel["lab_time"] > rel["an_end"])
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
        (out["cr_max_48h"].notna() & ((out["cr_max_48h"] - out["baseline_cr"]) >= 0.3))
        | (out["cr_max_7d"].notna() & (out["cr_max_7d"] >= 1.5 * out["baseline_cr"]))
    )
    aki_48h = (
        out["cr_max_48h"].notna()
        & (
            ((out["cr_max_48h"] - out["baseline_cr"]) >= 0.3)
            | (out["cr_max_48h"] >= 1.5 * out["baseline_cr"])
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
    ops["obstetric_flag"] = dept.eq("OG") | pcs.str.contains(r"(^|[,;\s])1[A-Z0-9]{6}", regex=True)
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


def extract_mover_creatinine(config: dict[str, Any], reuse: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    out_path = DATA_DIR / "mover_strict_blood_creatinine.parquet"
    audit_path = MANIFEST_DIR / "mover_creatinine_extraction_audit.json"
    if reuse and out_path.exists() and audit_path.exists():
        return pd.read_parquet(out_path), json.loads(audit_path.read_text(encoding="utf-8"))
    raw_path = Path(config["raw_data"]["mover_patient_labs"])
    usecols = [
        "LOG_ID",
        "Lab Code",
        "Lab Name",
        "Observation Value",
        "Measurement Units",
        "Collection Datetime",
    ]
    parts = []
    scanned = 0
    name_match = 0
    unit_match = 0
    loinc_match = 0
    for chunk in pd.read_csv(raw_path, usecols=usecols, dtype=str, chunksize=500_000, low_memory=False):
        scanned += len(chunk)
        names = chunk["Lab Name"].fillna("").str.strip().str.casefold()
        name_mask = names.eq("creatinine")
        name_match += int(name_mask.sum())
        units = chunk["Measurement Units"].fillna("").str.replace(" ", "", regex=False).str.upper()
        unit_mask = units.eq("MG/DL")
        unit_match += int((name_mask & unit_mask).sum())
        loinc = chunk["Lab Code"].fillna("").str.strip()
        code_mask = loinc.isin(MOVER_BLOOD_CREATININE_LOINC)
        keep = name_mask & unit_mask & code_mask
        loinc_match += int(keep.sum())
        sub = chunk.loc[keep, ["LOG_ID", "Lab Code", "Observation Value", "Collection Datetime"]].copy()
        if sub.empty:
            continue
        cleaned = sub["Observation Value"].str.extract(r"([-+]?[0-9]*\.?[0-9]+)", expand=False)
        sub["cr"] = pd.to_numeric(cleaned, errors="coerce")
        sub["lab_time"] = pd.to_datetime(sub["Collection Datetime"], errors="coerce")
        sub = sub[sub["cr"].between(0.05, 30.0, inclusive="both") & sub["lab_time"].notna()]
        sub["case_raw"] = sub["LOG_ID"].astype(str)
        parts.append(sub[["case_raw", "Lab Code", "cr", "lab_time"]])
    labs = pd.concat(parts, ignore_index=True)
    labs.to_parquet(out_path, index=False)
    code_counts = labs.groupby("Lab Code", dropna=False).size().reset_index(name="n")
    write_table(code_counts, "02_mover_creatinine_code_counts.csv")
    audit = {
        "raw_rows_scanned": int(scanned),
        "exact_creatinine_name_rows": int(name_match),
        "name_and_mg_dl_rows": int(unit_match),
        "blood_loinc_rows_before_value_filter": int(loinc_match),
        "valid_strict_blood_creatinine_rows": int(len(labs)),
        "unique_operations": int(labs["case_raw"].nunique()),
    }
    write_json(audit, audit_path)
    return labs, audit


def build_mover(config: dict[str, Any], labs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    info_path = Path(config["raw_data"]["mover_patient_information"])
    info = pd.read_csv(info_path, dtype=str, low_memory=False)
    raw_rows = len(info)
    duplicate_mask = info["LOG_ID"].duplicated(keep=False)
    duplicate_rows = info.loc[duplicate_mask].copy()
    deduplication_fields = [
        "MRN",
        "BIRTH_DATE",
        "ASA_RATING_C",
        "SEX",
        "AN_START_DATETIME",
        "AN_STOP_DATETIME",
        "HOSP_DISCH_TIME",
        "PRIMARY_PROCEDURE_NM",
        "PRIMARY_ANES_TYPE_NM",
    ]
    conflict_ids = []
    for log_id, group in duplicate_rows.groupby("LOG_ID", sort=False, dropna=False):
        if any(group[field].nunique(dropna=False) > 1 for field in deduplication_fields):
            conflict_ids.append(log_id)
    exact_duplicate_ids = set(duplicate_rows["LOG_ID"].dropna().unique()) - set(conflict_ids)
    info = info.loc[~info["LOG_ID"].isin(conflict_ids)].drop_duplicates("LOG_ID", keep="first").copy()
    info["case_raw"] = info["LOG_ID"].astype(str)
    info["patient_raw"] = info["MRN"].fillna(info["LOG_ID"]).astype(str)
    info["age"] = pd.to_numeric(info["BIRTH_DATE"], errors="coerce")
    info["asa"] = pd.to_numeric(info["ASA_RATING_C"], errors="coerce")
    info["sex_male"] = info["SEX"].fillna("").str.casefold().eq("male").astype(float)
    info["an_start"] = pd.to_datetime(info["AN_START_DATETIME"], errors="coerce")
    info["an_end"] = pd.to_datetime(info["AN_STOP_DATETIME"], errors="coerce")
    info["discharge"] = pd.to_datetime(info["HOSP_DISCH_TIME"], errors="coerce")
    info["duration_h"] = (info["an_end"] - info["an_start"]).dt.total_seconds() / 3600.0
    procedure = info["PRIMARY_PROCEDURE_NM"].fillna("").astype(str)
    info["cardiac_flag"] = procedure.str.contains(CARDIAC_RE, regex=True)
    info["obstetric_flag"] = procedure.str.contains(OBSTETRIC_RE, regex=True)
    info["clinical_eligible"] = (
        info["age"].ge(18)
        & info["PRIMARY_ANES_TYPE_NM"].fillna("").str.strip().str.casefold().eq("general")
        & info["an_start"].notna()
        & info["an_end"].notna()
        & info["duration_h"].between(0.5, 24.0, inclusive="both")
        & ~info["cardiac_flag"]
        & ~info["obstetric_flag"]
    )
    eligible = info.loc[info["clinical_eligible"]].copy().sort_values(["patient_raw", "an_start", "case_raw"])
    eligible["first_eligible"] = eligible.groupby("patient_raw", sort=False).cumcount().eq(0)
    eligible["year"] = eligible["an_start"].dt.year.astype("Int64")
    eligible["quarter"] = eligible["an_start"].dt.quarter.astype("Int64")
    base_48 = eligible["an_end"] + pd.Timedelta(hours=48)
    base_7d = eligible["an_end"] + pd.Timedelta(days=7)
    valid_discharge = eligible["discharge"].where(eligible["discharge"] > eligible["an_end"])
    eligible["window_48h_end"] = pd.concat([base_48, valid_discharge], axis=1).min(axis=1)
    eligible["window_7d_end"] = pd.concat([base_7d, valid_discharge], axis=1).min(axis=1)
    cohort = attach_creatinine_outcomes_datetime(eligible, labs)
    cohort["dataset"] = "MOVER"
    cohort["patient_key"] = cohort["patient_raw"].map(lambda x: stable_key("MOVER_PATIENT", x))
    cohort["case_key"] = cohort["case_raw"].map(lambda x: stable_key("MOVER_CASE", x))
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
        "an_start",
        "year",
        "quarter",
    ]
    cohort = cohort[keep].reset_index(drop=True)
    cohort.to_parquet(DATA_DIR / "mover_rebuilt_cohort.parquet", index=False)
    audit = {
        "operations_total_raw": int(raw_rows),
        "duplicate_operation_ids": int(duplicate_rows["LOG_ID"].nunique()),
        "exact_duplicate_operation_ids_collapsed": int(len(exact_duplicate_ids)),
        "conflicting_duplicate_operation_ids_excluded": int(len(conflict_ids)),
        "operations_after_duplicate_resolution": int(len(info)),
        "clinical_eligible_operations": int(len(eligible)),
        "first_clinically_eligible_operations": int(cohort["first_eligible"].sum()),
        "cardiac_excluded": int(info["cardiac_flag"].sum()),
        "obstetric_excluded": int(info["obstetric_flag"].sum()),
    }
    return cohort, audit


def load_vitaldb(config: dict[str, Any]) -> pd.DataFrame:
    raw = pd.read_parquet(Path(config["raw_data"]["vitaldb_analysis_cache"]))
    cohort = raw.copy()
    cohort["dataset"] = "VitalDB"
    cohort["patient_key"] = cohort["caseid"].map(lambda x: stable_key("VITALDB_CASE", x))
    cohort["case_key"] = cohort["patient_key"]
    cohort["tested_7d"] = cohort["has_postop_cr_7d"].fillna(False).astype(bool)
    if "aki_48h" not in cohort:
        cohort["aki_48h"] = np.nan
    keep = [
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
    return cohort[[c for c in keep if c in cohort.columns]].copy()


def analysis_subset(
    cohort: pd.DataFrame,
    *,
    first_only: bool = True,
    year: int | None = None,
    outcome: str = "aki",
) -> pd.DataFrame:
    mask = cohort["has_baseline_cr"].fillna(False).astype(bool) & cohort["baseline_cr_under4"].fillna(False).astype(bool)
    mask &= cohort["tested_7d"].fillna(False).astype(bool) & cohort[outcome].notna()
    if first_only and "first_eligible" in cohort:
        mask &= cohort["first_eligible"].fillna(False).astype(bool)
    if year is not None:
        mask &= cohort["year"].eq(year)
    out = cohort.loc[mask].copy()
    out[outcome] = pd.to_numeric(out[outcome], errors="coerce").astype(int)
    return out


def cohort_flow_rows(dataset: str, cohort: pd.DataFrame, years: Sequence[int] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    year_values: list[int | None] = [None] if years is None else list(years)
    for year in year_values:
        frame = cohort if year is None else cohort.loc[cohort["year"].eq(year)]
        label = dataset if year is None else f"{dataset}_{year}"
        steps = [
            ("clinically_eligible_operations", pd.Series(True, index=frame.index)),
            ("first_clinically_eligible_surgery", frame.get("first_eligible", pd.Series(True, index=frame.index)).fillna(False).astype(bool)),
        ]
        first = steps[-1][1]
        steps.extend(
            [
                ("baseline_creatinine_within_7d", first & frame["has_baseline_cr"].fillna(False).astype(bool)),
                (
                    "baseline_creatinine_below_4",
                    first
                    & frame["has_baseline_cr"].fillna(False).astype(bool)
                    & frame["baseline_cr_under4"].fillna(False).astype(bool),
                ),
                (
                    "postoperative_creatinine_tested",
                    first
                    & frame["has_baseline_cr"].fillna(False).astype(bool)
                    & frame["baseline_cr_under4"].fillna(False).astype(bool)
                    & frame["tested_7d"].fillna(False).astype(bool),
                ),
            ]
        )
        for order, (step, mask) in enumerate(steps, start=1):
            n = int(mask.sum())
            events = int(frame.loc[mask, "aki"].fillna(0).sum()) if "aki" in frame else 0
            rows.append(
                {
                    "cohort": label,
                    "step_order": order,
                    "step": step,
                    "n": n,
                    "aki_events": events,
                    "aki_rate": events / n if n else np.nan,
                }
            )
    return rows


def summarize_characteristics(cohorts: dict[str, pd.DataFrame], outcome: str = "aki") -> pd.DataFrame:
    rows = []
    for label, frame in cohorts.items():
        y = pd.to_numeric(frame[outcome], errors="coerce")
        specs = [
            ("n", float(len(frame)), "count"),
            ("AKI events", float(y.sum()), "count"),
            ("AKI incidence", float(y.mean()), "proportion"),
            ("Age, years", float(frame["age"].mean()), "mean"),
            ("Age SD", float(frame["age"].std(ddof=1)), "sd"),
            ("Male sex", float(frame["sex_male"].mean()), "proportion"),
            ("ASA 4-5", float(frame["asa"].isin([4, 5]).mean()), "proportion"),
            ("Anesthesia duration, h", float(frame["duration_h"].median()), "median"),
            ("Anesthesia duration Q1", float(frame["duration_h"].quantile(0.25)), "q1"),
            ("Anesthesia duration Q3", float(frame["duration_h"].quantile(0.75)), "q3"),
            ("Baseline creatinine, mg/dL", float(frame["baseline_cr"].median()), "median"),
            ("Baseline creatinine Q1", float(frame["baseline_cr"].quantile(0.25)), "q1"),
            ("Baseline creatinine Q3", float(frame["baseline_cr"].quantile(0.75)), "q3"),
            ("Baseline creatinine >=1.2 mg/dL", float(frame["baseline_cr"].ge(1.2).mean()), "proportion"),
        ]
        for variable, value, statistic in specs:
            rows.append({"cohort": label, "variable": variable, "statistic": statistic, "value": value})
    return pd.DataFrame(rows)


def standardized_difference(tested: pd.Series, untested: pd.Series) -> float:
    a = pd.to_numeric(tested, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(untested, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def testing_selection_table(mover: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in [2021, 2022]:
        base = mover[
            mover["year"].eq(year)
            & mover["first_eligible"].fillna(False).astype(bool)
            & mover["has_baseline_cr"].fillna(False).astype(bool)
            & mover["baseline_cr_under4"].fillna(False).astype(bool)
        ].copy()
        tested = base[base["tested_7d"]]
        untested = base[~base["tested_7d"]]
        variables = ["age", "sex_male", "asa", "duration_h", "baseline_cr"]
        for variable in variables:
            rows.append(
                {
                    "year": year,
                    "variable": variable,
                    "tested_n": len(tested),
                    "untested_n": len(untested),
                    "tested_mean": pd.to_numeric(tested[variable], errors="coerce").mean(),
                    "untested_mean": pd.to_numeric(untested[variable], errors="coerce").mean(),
                    "standardized_mean_difference": standardized_difference(tested[variable], untested[variable]),
                }
            )
        rows.append(
            {
                "year": year,
                "variable": "postoperative_creatinine_testing_rate",
                "tested_n": len(tested),
                "untested_n": len(untested),
                "tested_mean": len(tested) / len(base) if len(base) else np.nan,
                "untested_mean": np.nan,
                "standardized_mean_difference": np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_model(model_name: str, seed: int) -> Pipeline:
    if model_name == "logistic_spline":
        continuous_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
            ]
        )
        binary_pipeline = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
        preprocess = ColumnTransformer(
            [("continuous", continuous_pipeline, CONTINUOUS), ("binary", binary_pipeline, BINARY)],
            remainder="drop",
        )
        classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=seed)
        return Pipeline([("preprocess", preprocess), ("model", classifier)])
    if model_name == "hist_gradient_boosting":
        preprocess = ColumnTransformer(
            [
                ("continuous", SimpleImputer(strategy="median"), CONTINUOUS),
                ("binary", SimpleImputer(strategy="most_frequent"), BINARY),
            ],
            remainder="drop",
        )
        classifier = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.0,
            random_state=seed,
        )
        return Pipeline([("preprocess", preprocess), ("model", classifier)])
    raise ValueError(f"Unknown model: {model_name}")


def clip_prob(p: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def fit_recalibration(y: np.ndarray, p: np.ndarray, method: str, penalty: float = 0.0) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    lp = logit(clip_prob(p))
    if method == "none":
        return 0.0, 1.0
    if method == "intercept":
        alpha = 0.0
        for _ in range(100):
            mu = expit(alpha + lp)
            gradient = np.sum(y - mu) - penalty * alpha
            information = np.sum(mu * (1 - mu)) + penalty + 1e-10
            step = gradient / information
            alpha += float(np.clip(step, -5.0, 5.0))
            if abs(step) < 1e-9:
                break
        return float(alpha), 1.0
    if method in {"logistic", "penalized_logistic"}:
        x = np.column_stack([np.ones(len(lp)), lp])
        theta = np.array([0.0, 1.0], dtype=float)
        ridge = float(penalty if method == "penalized_logistic" else 0.0)
        prior = np.array([0.0, 1.0])
        for _ in range(100):
            eta = x @ theta
            mu = expit(np.clip(eta, -30, 30))
            gradient = x.T @ (y - mu) - ridge * (theta - prior)
            weight = mu * (1 - mu)
            information = x.T @ (x * weight[:, None]) + ridge * np.eye(2) + 1e-9 * np.eye(2)
            try:
                step = np.linalg.solve(information, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(information) @ gradient
            step = np.clip(step, -5.0, 5.0)
            theta += step
            if np.max(np.abs(step)) < 1e-8:
                break
        theta[0] = np.clip(theta[0], -20, 20)
        theta[1] = np.clip(theta[1], -10, 10)
        return float(theta[0]), float(theta[1])
    raise ValueError(method)


def apply_recalibration(p: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return expit(alpha + beta * logit(clip_prob(p)))


def grouped_ici(y: np.ndarray, p: np.ndarray, groups: int = 10) -> float:
    frame = pd.DataFrame({"y": y, "p": p}).dropna()
    if frame.empty:
        return np.nan
    unique = frame["p"].nunique()
    q = min(groups, unique, len(frame))
    if q < 2:
        return float(abs(frame["y"].mean() - frame["p"].mean()))
    frame["bin"] = pd.qcut(frame["p"], q=q, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
    return float(np.average(np.abs(grouped["observed"] - grouped["predicted"]), weights=grouped["n"]))


def performance_metrics(
    y: np.ndarray | pd.Series,
    p: np.ndarray | pd.Series,
    *,
    include_discrimination: bool = True,
    include_ici: bool = True,
) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=int)
    p_arr = clip_prob(p)
    events = int(y_arr.sum())
    expected = float(p_arr.sum())
    alpha, slope = fit_recalibration(y_arr, p_arr, "logistic")
    intercept, _ = fit_recalibration(y_arr, p_arr, "intercept")
    return {
        "n": float(len(y_arr)),
        "events": float(events),
        "event_rate": float(np.mean(y_arr)),
        "predicted_mean": float(np.mean(p_arr)),
        "oe_ratio": events / expected if expected > 0 else np.nan,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_joint_intercept": alpha,
        "auroc": roc_auc_score(y_arr, p_arr) if include_discrimination and len(np.unique(y_arr)) == 2 else np.nan,
        "auprc": average_precision_score(y_arr, p_arr) if include_discrimination and len(np.unique(y_arr)) == 2 else np.nan,
        "brier": brier_score_loss(y_arr, p_arr),
        "grouped_ici": grouped_ici(y_arr, p_arr) if include_ici else np.nan,
    }


def bootstrap_fixed_predictions(
    y: np.ndarray,
    p: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    n = len(y)
    for rep in range(replicates):
        idx = rng.integers(0, n, n)
        metrics = performance_metrics(y[idx], p[idx], include_ici=False)
        rows.append({"replicate": rep, **metrics})
    return pd.DataFrame(rows)


def metric_summary_rows(
    label: str,
    point: dict[str, float],
    boot: pd.DataFrame | None,
    model: str,
) -> list[dict[str, Any]]:
    rows = []
    for metric, value in point.items():
        if metric in {"n", "events"}:
            lower = upper = value
        elif boot is not None and metric in boot:
            lower, upper = boot[metric].quantile([0.025, 0.975]).tolist()
        else:
            lower = upper = np.nan
        rows.append(
            {
                "model": model,
                "cohort": label,
                "metric": metric,
                "estimate": value,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    return rows


def cross_validated_predictions(model: Pipeline, frame: pd.DataFrame, outcome: str, seed: int) -> np.ndarray:
    y = frame[outcome].to_numpy(dtype=int)
    pred = np.full(len(frame), np.nan)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, valid_idx in folds.split(frame[FEATURES], y):
        fitted = clone(model).fit(frame.iloc[train_idx][FEATURES], y[train_idx])
        pred[valid_idx] = fitted.predict_proba(frame.iloc[valid_idx][FEATURES])[:, 1]
    return pred


def event_node_frames(update: pd.DataFrame, nodes: Sequence[Any], outcome: str = "aki") -> list[tuple[str, int, pd.DataFrame]]:
    ordered = update.sort_values(["an_start", "case_key"]).reset_index(drop=True)
    cumulative_events = ordered[outcome].astype(int).cumsum().to_numpy()
    total_events = int(cumulative_events[-1]) if len(cumulative_events) else 0
    result = []
    for node in nodes:
        if node == "all":
            end = len(ordered)
            label = "all"
            event_count = total_events
        else:
            requested = int(node)
            if requested > total_events:
                continue
            end = int(np.flatnonzero(cumulative_events >= requested)[0] + 1)
            label = str(requested)
            event_count = int(cumulative_events[end - 1])
        result.append((label, event_count, ordered.iloc[:end].copy()))
    return result


def calibration_bootstrap(
    update_y: np.ndarray,
    update_p: np.ndarray,
    test_y: np.ndarray,
    test_p: np.ndarray,
    method: str,
    replicates: int,
    rng: np.random.Generator,
    penalty: float = 0.0,
    precomputed_test_indices: list[np.ndarray] | None = None,
) -> pd.DataFrame:
    rows = []
    n_update = len(update_y)
    n_test = len(test_y)
    for rep in range(replicates):
        if method == "none":
            alpha, beta = 0.0, 1.0
        else:
            idx_u = rng.integers(0, n_update, n_update)
            alpha, beta = fit_recalibration(update_y[idx_u], update_p[idx_u], method, penalty=penalty)
        idx_t = precomputed_test_indices[rep] if precomputed_test_indices is not None else rng.integers(0, n_test, n_test)
        pred = apply_recalibration(test_p[idx_t], alpha, beta)
        y = test_y[idx_t]
        metrics = performance_metrics(y, pred, include_discrimination=False, include_ici=False)
        rows.append(
            {
                "replicate": rep,
                "update_alpha": alpha,
                "update_beta": beta,
                "oe_ratio": metrics["oe_ratio"],
                "calibration_intercept": metrics["calibration_intercept"],
                "calibration_slope": metrics["calibration_slope"],
                "brier": metrics["brier"],
                "grouped_ici": metrics["grouped_ici"],
            }
        )
    return pd.DataFrame(rows)


def readiness_probabilities(boot: pd.DataFrame, bounds: Bounds) -> tuple[float, float]:
    oe_ok = boot["oe_ratio"].between(bounds.oe[0], bounds.oe[1], inclusive="both")
    slope_ok = boot["calibration_slope"].between(bounds.slope[0], bounds.slope[1], inclusive="both")
    return float(oe_ok.mean()), float((oe_ok & slope_ok).mean())


def decision_consequences(y: np.ndarray, p: np.ndarray, thresholds: Sequence[float]) -> list[dict[str, float]]:
    rows = []
    n = len(y)
    prevalence = float(np.mean(y))
    for threshold in thresholds:
        alert = p >= threshold
        tp = int(np.sum(alert & (y == 1)))
        fp = int(np.sum(alert & (y == 0)))
        nb = tp / n - fp / n * threshold / (1 - threshold)
        treat_all_nb = prevalence - (1 - prevalence) * threshold / (1 - threshold)
        rows.append(
            {
                "threshold": threshold,
                "alerts_per_1000": 1000 * float(np.mean(alert)),
                "true_positives_per_1000": 1000 * tp / n,
                "false_positives_per_1000": 1000 * fp / n,
                "false_positives_per_true_positive": fp / tp if tp else np.nan,
                "sensitivity": tp / int(np.sum(y == 1)) if np.sum(y == 1) else np.nan,
                "positive_predictive_value": tp / int(np.sum(alert)) if np.sum(alert) else np.nan,
                "net_benefit": nb,
                "treat_all_net_benefit": treat_all_nb,
                "treat_none_net_benefit": 0.0,
                "net_benefit_vs_better_default": nb - max(0.0, treat_all_nb),
            }
        )
    return rows


def model_coefficient_rows(model_name: str, fitted: Pipeline) -> list[dict[str, Any]]:
    if model_name != "logistic_spline":
        return []
    preprocess = fitted.named_steps["preprocess"]
    classifier = fitted.named_steps["model"]
    names = preprocess.get_feature_names_out()
    values = classifier.coef_.ravel()
    rows = [
        {
            "model": model_name,
            "term": "intercept",
            "coefficient": float(classifier.intercept_[0]),
            "odds_ratio": float(np.exp(classifier.intercept_[0])),
        }
    ]
    for term, value in zip(names, values):
        rows.append(
            {
                "model": model_name,
                "term": str(term),
                "coefficient": float(value),
                "odds_ratio": float(np.exp(value)),
            }
        )
    return rows


def model_metadata(
    model_name: str,
    fitted: Pipeline,
    source: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "study_id": config["study_id"],
        "prediction_time": config["prediction_time"],
        "features_in_order": FEATURES,
        "outcome": config["outcome"],
        "source_dataset": "INSPIRE",
        "source_n": int(len(source)),
        "source_events": int(source["aki"].sum()),
        "source_event_rate": float(source["aki"].mean()),
        "source_feature_summary": {
            feature: {
                "mean": float(pd.to_numeric(source[feature], errors="coerce").mean()),
                "sd": float(pd.to_numeric(source[feature], errors="coerce").std(ddof=1)),
                "missing": int(source[feature].isna().sum()),
            }
            for feature in FEATURES
        },
        "estimator_parameters": {key: str(value) for key, value in fitted.get_params(deep=True).items()},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": __import__("sklearn").__version__,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    }


def analyze_models(
    source: pd.DataFrame,
    update: pd.DataFrame,
    test: pd.DataFrame,
    vitaldb: pd.DataFrame,
    config: dict[str, Any],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    model_names = ["logistic_spline", "hist_gradient_boosting"]
    thresholds = config["decision_thresholds"]
    event_nodes = config["event_nodes"]
    bounds_map = {
        "strict": Bounds(tuple(config["readiness"]["strict_oe_bounds"]), tuple(config["readiness"]["strict_slope_bounds"])),
        "primary": Bounds(tuple(config["readiness"]["primary_oe_bounds"]), tuple(config["readiness"]["primary_slope_bounds"])),
        "permissive": Bounds(tuple(config["readiness"]["permissive_oe_bounds"]), tuple(config["readiness"]["permissive_slope_bounds"])),
    }
    fixed_summary_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    readiness_rows: list[dict[str, Any]] = []
    workload_rows: list[dict[str, Any]] = []
    all_boot: list[pd.DataFrame] = []
    fitted_models: dict[str, Pipeline] = {}
    prediction_store: dict[str, dict[str, np.ndarray]] = {}
    node_parameters: dict[str, dict[str, tuple[float, float]]] = {}

    for model_index, model_name in enumerate(model_names):
        model_seed = seed + 1000 * model_index
        rng = np.random.default_rng(model_seed)
        base_model = make_model(model_name, model_seed)
        cv_pred = cross_validated_predictions(base_model, source, "aki", model_seed)
        cv_point = performance_metrics(source["aki"].to_numpy(dtype=int), cv_pred)
        cv_boot = bootstrap_fixed_predictions(source["aki"].to_numpy(dtype=int), cv_pred, replicates, rng)
        fixed_summary_rows.extend(metric_summary_rows("INSPIRE_5fold_CV", cv_point, cv_boot, model_name))

        fitted = clone(base_model).fit(source[FEATURES], source["aki"].to_numpy(dtype=int))
        fitted_models[model_name] = fitted
        joblib.dump(fitted, MODEL_DIR / f"source_{model_name}.joblib")
        metadata = model_metadata(model_name, fitted, source, config)
        write_json(metadata, MODEL_DIR / f"source_{model_name}_metadata.json")
        coefficient_rows.extend(model_coefficient_rows(model_name, fitted))

        source_apparent = fitted.predict_proba(source[FEATURES])[:, 1]
        update_pred = fitted.predict_proba(update[FEATURES])[:, 1]
        test_pred = fitted.predict_proba(test[FEATURES])[:, 1]
        vital_pred = fitted.predict_proba(vitaldb[FEATURES])[:, 1] if len(vitaldb) else np.array([])
        prediction_store[model_name] = {
            "source_apparent": source_apparent,
            "source_cv": cv_pred,
            "update": update_pred,
            "test": test_pred,
            "vitaldb": vital_pred,
        }

        fixed_cohorts = [
            ("INSPIRE_apparent", source["aki"].to_numpy(dtype=int), source_apparent),
            ("MOVER_2021_update_descriptive", update["aki"].to_numpy(dtype=int), update_pred),
            ("MOVER_2022_locked_test", test["aki"].to_numpy(dtype=int), test_pred),
        ]
        if len(vitaldb):
            fixed_cohorts.append(("VitalDB_supportive", vitaldb["aki"].to_numpy(dtype=int), vital_pred))
        for cohort_index, (label, y_values, p_values) in enumerate(fixed_cohorts):
            point = performance_metrics(y_values, p_values)
            boot = bootstrap_fixed_predictions(y_values, p_values, replicates, np.random.default_rng(model_seed + 50 + cohort_index))
            fixed_summary_rows.extend(metric_summary_rows(label, point, boot, model_name))

        nodes = event_node_frames(update, event_nodes, outcome="aki")
        test_y = test["aki"].to_numpy(dtype=int)
        update_y_all = update["aki"].to_numpy(dtype=int)
        test_indices = [rng.integers(0, len(test_y), len(test_y)) for _ in range(replicates)]
        none_boot = calibration_bootstrap(
            update_y_all,
            update_pred,
            test_y,
            test_pred,
            "none",
            replicates,
            rng,
            precomputed_test_indices=test_indices,
        )
        node_parameters[model_name] = {}
        for node_order, (node_label, node_events, node_frame) in enumerate(nodes, start=1):
            node_pred = fitted.predict_proba(node_frame[FEATURES])[:, 1]
            node_y = node_frame["aki"].to_numpy(dtype=int)
            composition_rows.append(
                {
                    "model": model_name,
                    "node_order": node_order,
                    "event_node": node_label,
                    "actual_events": node_events,
                    "update_n": len(node_frame),
                    "non_events": int(len(node_frame) - node_events),
                    "first_case_date": node_frame["an_start"].min(),
                    "last_case_date": node_frame["an_start"].max(),
                }
            )
            for method_index, method in enumerate(["none", "intercept", "logistic"]):
                alpha, beta = fit_recalibration(node_y, node_pred, method)
                updated_test_pred = apply_recalibration(test_pred, alpha, beta)
                point = performance_metrics(test_y, updated_test_pred)
                event_rows.append(
                    {
                        "model": model_name,
                        "node_order": node_order,
                        "event_node": node_label,
                        "actual_events": node_events,
                        "update_n": len(node_frame),
                        "method": method,
                        "update_alpha": alpha,
                        "update_beta": beta,
                        **point,
                    }
                )
                node_parameters[model_name][f"{node_label}|{method}"] = (alpha, beta)
                if method == "none":
                    boot = none_boot.copy()
                else:
                    boot = calibration_bootstrap(
                        node_y,
                        node_pred,
                        test_y,
                        test_pred,
                        method,
                        replicates,
                        np.random.default_rng(model_seed + node_order * 100 + method_index),
                        precomputed_test_indices=test_indices,
                    )
                boot.insert(0, "method", method)
                boot.insert(0, "actual_events", node_events)
                boot.insert(0, "event_node", node_label)
                boot.insert(0, "node_order", node_order)
                boot.insert(0, "model", model_name)
                all_boot.append(boot)
                for band, bounds in bounds_map.items():
                    global_prob, full_prob = readiness_probabilities(boot, bounds)
                    readiness_rows.append(
                        {
                            "model": model_name,
                            "node_order": node_order,
                            "event_node": node_label,
                            "actual_events": node_events,
                            "method": method,
                            "tolerance_band": band,
                            "oe_lower": bounds.oe[0],
                            "oe_upper": bounds.oe[1],
                            "slope_lower": bounds.slope[0],
                            "slope_upper": bounds.slope[1],
                            "probability_global_calibration_sufficient": global_prob,
                            "probability_full_calibration_sufficient": full_prob,
                            "probability_threshold": config["readiness"]["posterior_style_bootstrap_probability"],
                        }
                    )
                for row in decision_consequences(test_y, updated_test_pred, thresholds):
                    workload_rows.append(
                        {
                            "model": model_name,
                            "event_node": node_label,
                            "actual_events": node_events,
                            "method": method,
                            **row,
                        }
                    )

    fixed_summary = pd.DataFrame(fixed_summary_rows)
    event_performance = pd.DataFrame(event_rows)
    event_composition = pd.DataFrame(composition_rows)
    readiness = pd.DataFrame(readiness_rows)
    workload = pd.DataFrame(workload_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    boot_distribution = pd.concat(all_boot, ignore_index=True)
    boot_distribution.to_parquet(DATA_DIR / "event_node_bootstrap_distribution.parquet", index=False)

    threshold_rows = []
    threshold_probability = float(config["readiness"]["posterior_style_bootstrap_probability"])
    for (model_name, method, band), group in readiness.groupby(["model", "method", "tolerance_band"], sort=False):
        ordered = group.sort_values("node_order")
        for criterion, probability_col in [
            ("global", "probability_global_calibration_sufficient"),
            ("full", "probability_full_calibration_sufficient"),
        ]:
            values = ordered[probability_col].to_numpy(dtype=float)
            persistent = np.array([bool(np.all(values[i:] >= threshold_probability)) for i in range(len(values))])
            if persistent.any():
                chosen = ordered.iloc[int(np.flatnonzero(persistent)[0])]
                node = chosen["event_node"]
                events = int(chosen["actual_events"])
                status = "reached_and_persistent"
            else:
                node = "not_reached"
                events = np.nan
                status = "not_reached_within_available_events"
            threshold_rows.append(
                {
                    "model": model_name,
                    "method": method,
                    "tolerance_band": band,
                    "criterion": criterion,
                    "minimum_persistent_event_node": node,
                    "minimum_persistent_actual_events": events,
                    "status": status,
                    "maximum_available_events": int(ordered["actual_events"].max()),
                    "probability_threshold": threshold_probability,
                }
            )
    readiness_thresholds = pd.DataFrame(threshold_rows)

    write_table(fixed_summary, "06_unupdated_transport_performance.csv")
    write_table(event_composition, "07_event_node_composition.csv")
    write_table(event_performance, "08_event_node_performance.csv")
    write_table(readiness, "09_readiness_probabilities.csv")
    write_table(readiness_thresholds, "10_readiness_thresholds.csv")
    write_table(workload, "11_decision_workload.csv")
    write_table(coefficients, "18_model_coefficients.csv")
    return {
        "models": fitted_models,
        "predictions": prediction_store,
        "node_parameters": node_parameters,
        "fixed_summary": fixed_summary,
        "event_composition": event_composition,
        "event_performance": event_performance,
        "readiness": readiness,
        "readiness_thresholds": readiness_thresholds,
        "workload": workload,
        "bootstrap": boot_distribution,
    }


def bootstrap_summary_for_prediction(
    frame: pd.DataFrame,
    p: np.ndarray,
    model: str,
    method: str,
    subgroup: str,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    y = frame["aki"].to_numpy(dtype=int)
    point = performance_metrics(y, p)
    boot = bootstrap_fixed_predictions(y, p, replicates, np.random.default_rng(seed))
    rows = []
    for metric, value in point.items():
        if metric in {"n", "events"}:
            lower = upper = value
        else:
            lower, upper = boot[metric].quantile([0.025, 0.975]).tolist()
        rows.append(
            {
                "model": model,
                "method": method,
                "subgroup": subgroup,
                "metric": metric,
                "estimate": value,
                "ci_lower": lower,
                "ci_upper": upper,
                "precision_flag": "low_event_count" if int(point["events"]) < 30 else "adequate_for_descriptive_reporting",
            }
        )
    return rows


def analyze_subgroups_and_time(
    test: pd.DataFrame,
    analysis: dict[str, Any],
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_name = "logistic_spline"
    test_pred = analysis["predictions"][model_name]["test"]
    performance = analysis["event_performance"]
    candidates = performance[(performance["model"] == model_name) & (performance["event_node"] == "all")]
    if candidates.empty:
        max_order = performance.loc[performance["model"] == model_name, "node_order"].max()
        candidates = performance[(performance["model"] == model_name) & (performance["node_order"] == max_order)]
    params = {
        row["method"]: (float(row["update_alpha"]), float(row["update_beta"]))
        for _, row in candidates.iterrows()
    }
    masks = {
        "overall": pd.Series(True, index=test.index),
        "age_below_75": test["age"].lt(75),
        "age_75_or_older": test["age"].ge(75),
        "female": test["sex_male"].eq(0),
        "male": test["sex_male"].eq(1),
        "ASA_1_to_3": test["asa"].isin([1, 2, 3]),
        "ASA_4_to_5": test["asa"].isin([4, 5]),
        "baseline_creatinine_below_1_2": test["baseline_cr"].lt(1.2),
        "baseline_creatinine_1_2_or_higher": test["baseline_cr"].ge(1.2),
    }
    subgroup_rows = []
    for subgroup_index, (label, mask) in enumerate(masks.items()):
        positions = np.flatnonzero(mask.to_numpy())
        frame = test.loc[mask].copy()
        if len(frame) < 20 or frame["aki"].nunique() < 2:
            continue
        for method_index, method in enumerate(["none", "intercept", "logistic"]):
            alpha, beta = params.get(method, (0.0, 1.0))
            pred = apply_recalibration(test_pred[positions], alpha, beta)
            subgroup_rows.extend(
                bootstrap_summary_for_prediction(
                    frame,
                    pred,
                    model_name,
                    method,
                    label,
                    replicates,
                    seed + subgroup_index * 100 + method_index,
                )
            )
    subgroup_table = pd.DataFrame(subgroup_rows)
    write_table(subgroup_table, "12_subgroup_performance.csv")

    quarter_rows = []
    for quarter in [1, 2, 3, 4]:
        mask = test["quarter"].eq(quarter)
        positions = np.flatnonzero(mask.to_numpy())
        frame = test.loc[mask].copy()
        if len(frame) < 20 or frame["aki"].nunique() < 2:
            continue
        for method_index, method in enumerate(["none", "intercept", "logistic"]):
            alpha, beta = params.get(method, (0.0, 1.0))
            pred = apply_recalibration(test_pred[positions], alpha, beta)
            point = performance_metrics(frame["aki"].to_numpy(dtype=int), pred)
            boot = bootstrap_fixed_predictions(
                frame["aki"].to_numpy(dtype=int),
                pred,
                replicates,
                np.random.default_rng(seed + 1000 + quarter * 10 + method_index),
            )
            for metric in ["n", "events", "event_rate", "oe_ratio", "calibration_intercept", "calibration_slope", "brier", "auroc"]:
                if metric in {"n", "events"}:
                    lower = upper = point[metric]
                else:
                    lower, upper = boot[metric].quantile([0.025, 0.975]).tolist()
                quarter_rows.append(
                    {
                        "model": model_name,
                        "year": 2022,
                        "quarter": quarter,
                        "method": method,
                        "metric": metric,
                        "estimate": point[metric],
                        "ci_lower": lower,
                        "ci_upper": upper,
                    }
                )
    quarter_table = pd.DataFrame(quarter_rows)
    write_table(quarter_table, "13_temporal_quarters.csv")
    return subgroup_table, quarter_table


def analyze_penalized_recalibration(
    update: pd.DataFrame,
    test: pd.DataFrame,
    analysis: dict[str, Any],
    config: dict[str, Any],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    model_name = "logistic_spline"
    update_pred = analysis["predictions"][model_name]["update"]
    test_pred = analysis["predictions"][model_name]["test"]
    update_y = update["aki"].to_numpy(dtype=int)
    test_y = test["aki"].to_numpy(dtype=int)
    bounds = Bounds(tuple(config["readiness"]["primary_oe_bounds"]), tuple(config["readiness"]["primary_slope_bounds"]))
    rows = []
    test_rng = np.random.default_rng(seed + 5)
    test_indices = [test_rng.integers(0, len(test_y), len(test_y)) for _ in range(replicates)]
    for index, (method, penalty) in enumerate([("logistic", 0.0), ("penalized_logistic", 1.0), ("penalized_logistic", 10.0)]):
        alpha, beta = fit_recalibration(update_y, update_pred, method, penalty=penalty)
        pred = apply_recalibration(test_pred, alpha, beta)
        point = performance_metrics(test_y, pred)
        boot = calibration_bootstrap(
            update_y,
            update_pred,
            test_y,
            test_pred,
            method,
            replicates,
            np.random.default_rng(seed + 100 + index),
            penalty=penalty,
            precomputed_test_indices=test_indices,
        )
        global_prob, full_prob = readiness_probabilities(boot, bounds)
        rows.append(
            {
                "model": model_name,
                "method": method,
                "penalty": penalty,
                "update_n": len(update),
                "update_events": int(update_y.sum()),
                "update_alpha": alpha,
                "update_beta": beta,
                **point,
                "probability_global_calibration_sufficient": global_prob,
                "probability_full_calibration_sufficient": full_prob,
            }
        )
    table = pd.DataFrame(rows)
    write_table(table, "16_penalized_recalibration_sensitivity.csv")
    return table


def analyze_48h_outcome(
    source_all: pd.DataFrame,
    update_all: pd.DataFrame,
    test_all: pd.DataFrame,
    config: dict[str, Any],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    source = analysis_subset(source_all, outcome="aki_48h")
    update = analysis_subset(update_all, year=2021, outcome="aki_48h")
    test = analysis_subset(test_all, year=2022, outcome="aki_48h")
    model = make_model("logistic_spline", seed).fit(source[FEATURES], source["aki_48h"].to_numpy(dtype=int))
    joblib.dump(model, MODEL_DIR / "source_logistic_spline_aki48h.joblib")
    update_pred = model.predict_proba(update[FEATURES])[:, 1]
    test_pred = model.predict_proba(test[FEATURES])[:, 1]
    test_y = test["aki_48h"].to_numpy(dtype=int)
    bounds = Bounds(tuple(config["readiness"]["primary_oe_bounds"]), tuple(config["readiness"]["primary_slope_bounds"]))
    rows = []
    for node_order, (node_label, node_events, node_frame) in enumerate(event_node_frames(update, config["event_nodes"], "aki_48h"), start=1):
        node_pred = model.predict_proba(node_frame[FEATURES])[:, 1]
        node_y = node_frame["aki_48h"].to_numpy(dtype=int)
        test_rng = np.random.default_rng(seed + node_order)
        test_indices = [test_rng.integers(0, len(test_y), len(test_y)) for _ in range(replicates)]
        for method_index, method in enumerate(["none", "intercept", "logistic"]):
            alpha, beta = fit_recalibration(node_y, node_pred, method)
            pred = apply_recalibration(test_pred, alpha, beta)
            point = performance_metrics(test_y, pred)
            boot = calibration_bootstrap(
                node_y,
                node_pred,
                test_y,
                test_pred,
                method,
                replicates,
                np.random.default_rng(seed + node_order * 100 + method_index),
                precomputed_test_indices=test_indices,
            )
            global_prob, full_prob = readiness_probabilities(boot, bounds)
            rows.append(
                {
                    "outcome": "creatinine_AKI_within_48h",
                    "event_node": node_label,
                    "actual_events": node_events,
                    "update_n": len(node_frame),
                    "method": method,
                    "update_alpha": alpha,
                    "update_beta": beta,
                    **point,
                    "probability_global_calibration_sufficient": global_prob,
                    "probability_full_calibration_sufficient": full_prob,
                }
            )
    table = pd.DataFrame(rows)
    write_table(table, "15_outcome_48h_sensitivity.csv")
    return table


def prepare_cluster_groups(keys: pd.Series) -> list[np.ndarray]:
    key_array = keys.astype(str).to_numpy()
    unique = pd.unique(key_array)
    return [np.flatnonzero(key_array == key) for key in unique]


def cluster_bootstrap_indices(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    sampled = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[index] for index in sampled])


def analyze_all_surgeries(
    source_all: pd.DataFrame,
    mover_all: pd.DataFrame,
    config: dict[str, Any],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    source = analysis_subset(source_all, first_only=False)
    update = analysis_subset(mover_all, first_only=False, year=2021)
    test = analysis_subset(mover_all, first_only=False, year=2022)
    model = make_model("logistic_spline", seed).fit(source[FEATURES], source["aki"].to_numpy(dtype=int))
    update_pred = model.predict_proba(update[FEATURES])[:, 1]
    test_pred = model.predict_proba(test[FEATURES])[:, 1]
    update_y = update["aki"].to_numpy(dtype=int)
    test_y = test["aki"].to_numpy(dtype=int)
    bounds = Bounds(tuple(config["readiness"]["primary_oe_bounds"]), tuple(config["readiness"]["primary_slope_bounds"]))
    rows = []
    update_groups = prepare_cluster_groups(update["patient_key"])
    test_groups = prepare_cluster_groups(test["patient_key"])
    for method_index, method in enumerate(["none", "intercept", "logistic"]):
        alpha, beta = fit_recalibration(update_y, update_pred, method)
        pred = apply_recalibration(test_pred, alpha, beta)
        point = performance_metrics(test_y, pred)
        boot_rows = []
        rng = np.random.default_rng(seed + method_index)
        for rep in range(replicates):
            idx_u = cluster_bootstrap_indices(update_groups, rng)
            idx_t = cluster_bootstrap_indices(test_groups, rng)
            if method == "none":
                a_rep, b_rep = 0.0, 1.0
            else:
                a_rep, b_rep = fit_recalibration(update_y[idx_u], update_pred[idx_u], method)
            metric = performance_metrics(
                test_y[idx_t],
                apply_recalibration(test_pred[idx_t], a_rep, b_rep),
                include_discrimination=False,
                include_ici=False,
            )
            boot_rows.append(metric)
        boot = pd.DataFrame(boot_rows)
        global_prob, full_prob = readiness_probabilities(boot, bounds)
        rows.append(
            {
                "analysis": "all_clinically_eligible_surgeries_cluster_bootstrap",
                "method": method,
                "source_n": len(source),
                "source_events": int(source["aki"].sum()),
                "update_n": len(update),
                "update_events": int(update_y.sum()),
                "test_n": len(test),
                "test_events": int(test_y.sum()),
                "update_alpha": alpha,
                "update_beta": beta,
                **point,
                "oe_ci_lower": boot["oe_ratio"].quantile(0.025),
                "oe_ci_upper": boot["oe_ratio"].quantile(0.975),
                "slope_ci_lower": boot["calibration_slope"].quantile(0.025),
                "slope_ci_upper": boot["calibration_slope"].quantile(0.975),
                "probability_global_calibration_sufficient": global_prob,
                "probability_full_calibration_sufficient": full_prob,
            }
        )
    table = pd.DataFrame(rows)
    write_table(table, "17_all_surgeries_cluster_sensitivity.csv")
    return table


def calibration_curve_points(y: np.ndarray, p: np.ndarray, groups: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"observed": y, "predicted": p})
    frame["bin"] = pd.qcut(frame["predicted"], q=min(groups, frame["predicted"].nunique()), duplicates="drop")
    return (
        frame.groupby("bin", observed=True)
        .agg(predicted=("predicted", "mean"), observed=("observed", "mean"), n=("observed", "size"))
        .reset_index(drop=True)
    )


def make_figures(
    flow: pd.DataFrame,
    update: pd.DataFrame,
    test: pd.DataFrame,
    analysis: dict[str, Any],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    colors = {"none": "#4A5568", "intercept": "#007C83", "logistic": "#C44736"}

    selected = flow[flow["step"].eq("postoperative_creatinine_tested")].copy()
    selected = selected[selected["cohort"].isin(["INSPIRE", "MOVER_2021", "MOVER_2022"])]
    selected["display_label"] = selected["cohort"].map(
        {
            "INSPIRE": "INSPIRE\nSource model",
            "MOVER_2021": "MOVER 2021\nLocal update",
            "MOVER_2022": "MOVER 2022\nPost-exploration temporal evaluation",
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(selected["display_label"], selected["n"], color=["#4A5568", "#007C83", "#C44736"], width=0.62)
    for bar, (_, row) in zip(bars, selected.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={int(row['n']):,}\nAKI={int(row['aki_events']):,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Patients in primary analysis")
    ax.set_title("Study cohorts and prespecified temporal roles")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "Figure_1_cohort_roles.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "Figure_1_cohort_roles.pdf", bbox_inches="tight")
    plt.close(fig)

    ready = analysis["readiness"]
    ready = ready[
        (ready["model"] == "logistic_spline")
        & (ready["tolerance_band"] == "primary")
        & ready["method"].isin(["none", "intercept", "logistic"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True, sharey=True)
    for ax, metric, title in [
        (axes[0], "probability_global_calibration_sufficient", "Global calibration"),
        (axes[1], "probability_full_calibration_sufficient", "O/E and calibration slope"),
    ]:
        for method in ["none", "intercept", "logistic"]:
            part = ready[ready["method"] == method].sort_values("node_order")
            ax.plot(part["actual_events"], part[metric], marker="o", lw=2, color=colors[method], label=method)
        ax.axhline(0.90, color="#111827", lw=1.2, ls="--")
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("Cumulative local AKI events in MOVER 2021")
        ax.set_title(title)
        ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    axes[0].set_ylabel("Bootstrap probability within tolerance")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Local evidence accumulation in the temporally held-out MOVER 2022 cohort", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "Figure_2_readiness_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "Figure_2_readiness_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    model = "logistic_spline"
    base_pred = analysis["predictions"][model]["test"]
    event_perf = analysis["event_performance"]
    available = event_perf[(event_perf["model"] == model) & (event_perf["method"] == "logistic")]
    labels = ["none"]
    predictions = [base_pred]
    fifty = available[available["actual_events"] >= 50].sort_values("actual_events").head(1)
    all_row = available[available["event_node"] == "all"]
    if not fifty.empty:
        row = fifty.iloc[0]
        labels.append(f"logistic update at {int(row['actual_events'])} events")
        predictions.append(apply_recalibration(base_pred, float(row["update_alpha"]), float(row["update_beta"])))
    if not all_row.empty:
        row = all_row.iloc[0]
        labels.append(f"logistic update at all {int(row['actual_events'])} events")
        predictions.append(apply_recalibration(base_pred, float(row["update_alpha"]), float(row["update_beta"])))
    fig, ax = plt.subplots(figsize=(6.0, 5.3))
    palette = ["#4A5568", "#007C83", "#C44736"]
    y_test = test["aki"].to_numpy(dtype=int)
    for label, pred, color in zip(labels, predictions, palette):
        curve = calibration_curve_points(y_test, pred)
        ax.plot(curve["predicted"], curve["observed"], marker="o", lw=2, color=color, label=label)
    limit = max(0.25, max(float(np.quantile(p, 0.99)) for p in predictions))
    ax.plot([0, limit], [0, limit], color="#111827", ls="--", lw=1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed AKI proportion")
    ax.set_title("Calibration in the temporally held-out MOVER 2022 cohort")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#E5E7EB", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "Figure_3_calibration.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "Figure_3_calibration.pdf", bbox_inches="tight")
    plt.close(fig)

    workload = analysis["workload"]
    workload = workload[
        (workload["model"] == model)
        & workload["method"].isin(["none", "intercept", "logistic"])
        & np.isclose(workload["threshold"], 0.10)
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method in ["none", "intercept", "logistic"]:
        part = workload[workload["method"] == method].copy()
        order = analysis["event_composition"].query("model == @model")[["event_node", "node_order"]].drop_duplicates()
        part = part.merge(order, on="event_node", how="left").sort_values("node_order")
        ax.plot(part["actual_events"], part["alerts_per_1000"], marker="o", lw=2, color=colors[method], label=method)
    ax.set_xlabel("Cumulative local AKI events in MOVER 2021")
    ax.set_ylabel("Alerts per 1000 patients at 10% threshold")
    ax.set_title("Operational workload in the temporally held-out MOVER 2022 cohort")
    ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "Figure_4_operational_workload.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "Figure_4_operational_workload.pdf", bbox_inches="tight")
    plt.close(fig)


def build_metadata_tables(config: dict[str, Any], external_not_mounted: bool) -> None:
    authors = pd.DataFrame(
        [
            {"order": 1, "name": "Qingyu Teng", "affiliation": "1", "role_note": "equal contribution", "corresponding": False, "email": "", "orcid": ""},
            {"order": 2, "name": "Hui Zhang", "affiliation": "1", "role_note": "equal contribution", "corresponding": False, "email": "", "orcid": ""},
            {"order": 3, "name": "Yuping Yang", "affiliation": "1", "role_note": "", "corresponding": False, "email": "", "orcid": ""},
            {"order": 4, "name": "Chen Qian", "affiliation": "1", "role_note": "", "corresponding": False, "email": "", "orcid": ""},
            {"order": 5, "name": "Ziyan Gu", "affiliation": "1", "role_note": "", "corresponding": False, "email": "", "orcid": ""},
            {"order": 6, "name": "Qi Li", "affiliation": "1", "role_note": "", "corresponding": True, "email": "", "orcid": "0009-0003-3140-5887"},
            {"order": 7, "name": "Tao Xu", "affiliation": "1", "role_note": "", "corresponding": True, "email": "", "orcid": "0000-0001-5868-4079"},
        ]
    )
    write_table(authors, "24_author_metadata.csv")
    affiliation = pd.DataFrame(
        [
            {
                "affiliation_id": 1,
                "affiliation": "Shanghai Jiao Tong University, Shanghai 200240, China",
                "corresponding_department": "Department of Anesthesiology, Shanghai Sixth People's Hospital, Shanghai Jiao Tong University School of Medicine, Shanghai 200233, China",
            }
        ]
    )
    write_table(affiliation, "25_affiliation_metadata.csv")
    ethics = pd.DataFrame(
        [
            {
                "topic": "current_secondary_analysis",
                "statement": "This study used deidentified data released under the governance of the INSPIRE, MOVER, and VitalDB repositories; no participant contact, intervention, or re-identification was performed.",
                "submission_action": "Confirm the local institutional determination before submission; do not invent an exemption or approval number.",
            },
            {
                "topic": "VitalDB_original_collection",
                "statement": "Original VitalDB data collection was approved by the Seoul National University Hospital Institutional Review Board (H-1408-101-605) and registered as NCT02914444, as reported by the reference paper.",
                "submission_action": "Cite the original VitalDB data paper and repository terms.",
            },
            {
                "topic": "INSPIRE_and_MOVER_original_collection",
                "statement": "Original approvals and consent or waiver procedures must be reported exactly as stated in the corresponding dataset publications and data use agreements.",
                "submission_action": "Verify dataset-specific approval language against the primary data papers before manuscript submission.",
            },
            {
                "topic": "data_security",
                "statement": "Patient-level intermediate data remained on local storage. The workbook and repository package contain aggregate tables, code, and non-identifying model metadata only.",
                "submission_action": "Keep patient-level parquet files and fitted model binaries out of any public release.",
            },
        ]
    )
    write_table(ethics, "23_ethics_and_data_governance.csv")
    deviations = pd.DataFrame(
        [
            {"item": "external_volume", "status": "documented_limitation" if external_not_mounted else "available", "detail": "A configured external data volume was unavailable; analysis used verified local raw or analysis-ready copies."},
            {"item": "INSPIRE_calendar_split", "status": "not_feasible", "detail": "Released INSPIRE timestamps are relative within admissions and do not support a reliable cross-patient calendar split; all released source records were used for model development."},
            {"item": "VitalDB_temporal_validation", "status": "supportive_only", "detail": "VitalDB lacks a reliable calendar split and is treated as a related data-product stress test, not a third independent temporal validation."},
            {"item": "urine_output_AKI", "status": "not_harmonizable", "detail": "Urine-output KDIGO criteria were not harmonizable across the three public data products; the outcome is creatinine-based AKI."},
            {"item": "postoperative_testing", "status": "primary_complete_case_outcome", "detail": "Untested patients were not coded as no AKI; testing selection was quantified separately."},
            {"item": "procedure_exclusion", "status": "conservative_mapping", "detail": "INSPIRE used department, bypass, and ICD-10-PCS signals; MOVER used prespecified high-specificity procedure-name patterns because per-operation procedure codes were unavailable."},
            {"item": "legacy_predictions", "status": "not_used", "detail": "All source models and predictions were rebuilt and serialized; no prior-version prediction vector was reused."},
        ]
    )
    write_table(deviations, "20_analysis_deviations_and_limits.csv")


def runtime_table() -> pd.DataFrame:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": __import__("sklearn").__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
    }
    return pd.DataFrame([{"component": key, "version": value} for key, value in versions.items()])


def make_model_metadata_table(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_name in ["logistic_spline", "hist_gradient_boosting"]:
        model_path = MODEL_DIR / f"source_{model_name}.joblib"
        metadata_path = MODEL_DIR / f"source_{model_name}_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": model_name,
                "prediction_time": metadata["prediction_time"],
                "features": ", ".join(metadata["features_in_order"]),
                "source_n": metadata["source_n"],
                "source_events": metadata["source_events"],
                "source_event_rate": metadata["source_event_rate"],
                "model_object": model_path.name,
                "model_sha256": sha256_file(model_path),
                "metadata_file": metadata_path.name,
                "metadata_sha256": sha256_file(metadata_path),
            }
        )
    table = pd.DataFrame(rows)
    write_table(table, "19_model_metadata.csv")
    return table


def format_number(value: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def main_result_table(
    source: pd.DataFrame,
    update: pd.DataFrame,
    test: pd.DataFrame,
    analysis: dict[str, Any],
) -> pd.DataFrame:
    perf = analysis["event_performance"]
    primary = perf[perf["model"].eq("logistic_spline")]
    all_rows = primary[primary["event_node"].eq("all")]
    if all_rows.empty:
        all_rows = primary[primary["node_order"].eq(primary["node_order"].max())]
    rows = [
        {"result": "INSPIRE source sample", "value": len(source), "unit": "patients"},
        {"result": "INSPIRE source AKI events", "value": int(source["aki"].sum()), "unit": "events"},
        {"result": "MOVER 2021 update sample", "value": len(update), "unit": "patients"},
        {"result": "MOVER 2021 update AKI events", "value": int(update["aki"].sum()), "unit": "events"},
        {"result": "MOVER 2022 temporal evaluation sample", "value": len(test), "unit": "patients"},
        {"result": "MOVER 2022 temporal evaluation AKI events", "value": int(test["aki"].sum()), "unit": "events"},
    ]
    for _, row in all_rows.iterrows():
        for metric in ["oe_ratio", "calibration_slope", "calibration_intercept", "auroc", "auprc", "brier", "grouped_ici"]:
            rows.append(
                {
                    "result": f"Temporal evaluation {row['method']} at all update events: {metric}",
                    "value": row[metric],
                    "unit": "estimate",
                }
            )
    threshold = analysis["readiness_thresholds"]
    selected = threshold[
        (threshold["model"] == "logistic_spline")
        & (threshold["method"] == "logistic")
        & (threshold["tolerance_band"] == "primary")
        & (threshold["criterion"] == "full")
    ]
    if not selected.empty:
        row = selected.iloc[0]
        rows.append(
            {
                "result": "Minimum persistent event node for primary full-calibration readiness",
                "value": row["minimum_persistent_event_node"],
                "unit": row["status"],
            }
        )
    table = pd.DataFrame(rows)
    write_table(table, "26_main_results_summary.csv")
    return table


def generate_reports(
    source: pd.DataFrame,
    update: pd.DataFrame,
    test: pd.DataFrame,
    analysis: dict[str, Any],
    external_not_mounted: bool,
) -> None:
    perf = analysis["event_performance"]
    primary = perf[(perf["model"] == "logistic_spline")]
    all_rows = primary[primary["event_node"] == "all"]
    if all_rows.empty:
        all_rows = primary[primary["node_order"] == primary["node_order"].max()]
    no_update = all_rows[all_rows["method"] == "none"].iloc[0]
    logistic = all_rows[all_rows["method"] == "logistic"].iloc[0]
    intercept = all_rows[all_rows["method"] == "intercept"].iloc[0]
    threshold = analysis["readiness_thresholds"]
    selected = threshold[
        (threshold["model"] == "logistic_spline")
        & (threshold["method"] == "logistic")
        & (threshold["tolerance_band"] == "primary")
        & (threshold["criterion"] == "full")
    ].iloc[0]
    selected_global = threshold[
        (threshold["model"] == "logistic_spline")
        & (threshold["method"] == "logistic")
        & (threshold["tolerance_band"] == "primary")
        & (threshold["criterion"] == "global")
    ].iloc[0]
    if selected_global["status"] == "reached_and_persistent":
        global_sentence_cn = (
            f"仅考虑 O/E 的全局校准规则在 {selected_global['minimum_persistent_event_node']} 个本地 AKI 事件时首次达标并持续至全部事件。"
        )
        global_sentence_en = (
            f"The O/E-only global-calibration criterion was first met at {selected_global['minimum_persistent_event_node']} local AKI events and remained met thereafter."
        )
    else:
        global_sentence_cn = "仅考虑 O/E 的全局校准规则在可用事件范围内也未持续达标。"
        global_sentence_en = "The O/E-only global-calibration criterion was not persistently met within the available events."
    if selected["status"] == "reached_and_persistent":
        readiness_sentence_cn = (
            f"Logistic 再校准在 {selected['minimum_persistent_event_node']} 个本地 AKI 事件节点首次达到"
            "主要完整校准规则，且后续节点持续达标。"
        )
        readiness_sentence_en = (
            f"Logistic recalibration first met and subsequently maintained the prespecified full-calibration criterion at "
            f"{selected['minimum_persistent_event_node']} local AKI events."
        )
    else:
        readiness_sentence_cn = (
            f"在 MOVER 2021 可用的 {int(selected['maximum_available_events'])} 个 AKI 事件内，Logistic 再校准未达到"
            "并持续满足主要完整校准规则。"
        )
        readiness_sentence_en = (
            f"Logistic recalibration did not meet and maintain the prespecified full-calibration criterion within the "
            f"{int(selected['maximum_available_events'])} local AKI events available in MOVER 2021."
        )
    storage_sentence_cn = (
        "未挂载，已使用本机保存的原始或分析就绪副本"
        if external_not_mounted
        else "已挂载；其 MOVER 患者信息和检验文件与本次分析使用的本机解压文件在字节大小和 SHA-256 上完全一致"
    )

    chinese = f"""# 重新 1.0 研究结果总结

## 一、已执行的主设计

本次从本地原始/分析就绪数据重建队列，不使用旧版预测。INSPIRE 用于建立冻结源模型；MOVER 2021 仅用于按时间顺序累积本地事件并拟合校准更新；MOVER 2022 是未参与拟合、调参或方法选择的封存未来检验集。VitalDB 仅作支持性数据产品压力测试。

## 二、队列与事件

- INSPIRE 源队列：{len(source):,} 人，{int(source['aki'].sum()):,} 个 AKI 事件，发生率 {source['aki'].mean():.1%}。
- MOVER 2021 本地更新队列：{len(update):,} 人，{int(update['aki'].sum()):,} 个 AKI 事件，发生率 {update['aki'].mean():.1%}。
- MOVER 2022 封存检验队列：{len(test):,} 人，{int(test['aki'].sum()):,} 个 AKI 事件，发生率 {test['aki'].mean():.1%}。

## 三、主要结果

未更新的 Logistic 样条源模型在 MOVER 2022 中的 O/E 为 {no_update['oe_ratio']:.3f}，校准斜率为 {no_update['calibration_slope']:.3f}，AUROC 为 {no_update['auroc']:.3f}，Brier score 为 {no_update['brier']:.3f}。这表明运输后的主要风险不是单纯区分度问题，而是绝对风险尺度发生偏移。

使用 MOVER 2021 全部可用事件后，截距更新在 MOVER 2022 中的 O/E 为 {intercept['oe_ratio']:.3f}，校准斜率为 {intercept['calibration_slope']:.3f}；Logistic 再校准的 O/E 为 {logistic['oe_ratio']:.3f}，校准斜率为 {logistic['calibration_slope']:.3f}，Brier score 为 {logistic['brier']:.3f}。

{readiness_sentence_cn}

{global_sentence_cn}这一结果不能代替同时要求 O/E 和校准斜率达标的主要完整校准规则。

## 四、可以与不可以得出的结论

该结果只适用于本研究特定的 INSPIRE 源模型、MOVER 目标数据、肌酐 AKI 结局、更新方法和容差区间。不应将该事件数外推为所有 AKI 模型的通用要求。即使统计就绪规则达标，也只能支持进入前瞻性静默监测，不能直接支持对临床决策产生影响的实时部署。

## 五、关键限制

1. 术后未检测肌酐者不能判定无 AKI，主要分析存在结局验证选择。
2. 三库无法一致重建尿量 KDIGO 标准，结局仅为肌酐 AKI。
3. INSPIRE 释放时间为相对时间，无法进行可审计的跨患者日历切分。
4. MOVER 的非心脏/非产科映射部分依赖手术名称，仍有残余错分可能。
5. 当前结果为回顾性公开数据库研究，没有评估告警对医生行为或患者结局的真实影响。

## 六、数据和伦理说明

本研究仅使用 INSPIRE、MOVER 和 VitalDB 已去标识的数据产品，不进行参与者接触、干预或重新识别。原始数据收集的伦理审批与知情同意/豁免程序应按各数据库原始论文和数据使用协议如实报告。VitalDB 原始收集经首尔大学医院伦理委员会批准（H-1408-101-605），并注册为 NCT02914444。本次二次分析是否需要上海交通大学/上海市第六人民医院额外伦理认定，应由作者按本机构规则确认，不应在未确认时自行填写豁免号或批件号。

## 七、存储与审计

外接数据卷本次{storage_sentence_cn}。原始文件指纹、队列流程、模型对象指纹、中间 bootstrap 分布和质量检查均保存在本目录。
"""
    (REPORT_DIR / "研究结果总结.md").write_text(chinese, encoding="utf-8")

    english = f"""# Manuscript-ready results: local evidence requirements before transport of a perioperative AKI model

## Methods summary

We performed a retrospective, multi-database model transport and temporal updating study using deidentified INSPIRE, MOVER, and VitalDB data products. A prespecified logistic spline model using age, sex, anaesthesia duration, and baseline creatinine was developed in INSPIRE. MOVER cases from 2021 were ordered chronologically and used only for intercept updating or logistic recalibration at prespecified cumulative AKI-event nodes. MOVER cases from 2022 were used for post-exploration temporally held-out evaluation and were not used for model fitting, tuning, method selection, or threshold selection; this was not an independent confirmation cohort. The primary outcome was creatinine-defined KDIGO AKI within 7 days, with observation truncated at discharge. Calibration sufficiency required a bootstrap probability of at least 0.90 that O/E was 0.80-1.25 and the calibration slope was 0.80-1.20, with persistence at all subsequent event nodes.

## Results

The source cohort included {len(source):,} patients with {int(source['aki'].sum()):,} AKI events ({source['aki'].mean():.1%}). The chronological MOVER 2021 update cohort included {len(update):,} patients with {int(update['aki'].sum()):,} events ({update['aki'].mean():.1%}), and the post-exploration MOVER 2022 temporal evaluation cohort included {len(test):,} patients with {int(test['aki'].sum()):,} events ({test['aki'].mean():.1%}). Without local updating, the logistic spline model had an O/E ratio of {no_update['oe_ratio']:.3f}, calibration slope of {no_update['calibration_slope']:.3f}, AUROC of {no_update['auroc']:.3f}, and Brier score of {no_update['brier']:.3f} in the temporal evaluation cohort. After logistic recalibration using all available 2021 observations, the corresponding O/E ratio was {logistic['oe_ratio']:.3f}, calibration slope was {logistic['calibration_slope']:.3f}, and Brier score was {logistic['brier']:.3f}. {global_sentence_en} {readiness_sentence_en}

## Interpretation

The findings quantify the amount of local outcome evidence required for this specific transported model under prespecified calibration tolerances. They do not establish a universal event requirement for other models or settings. Meeting the statistical rule would support progression to prospective silent monitoring, not immediate decision-active deployment.

## Ethics and data governance

This secondary analysis used deidentified data released under the governance of the INSPIRE, MOVER, and VitalDB repositories and involved no participant contact, intervention, or re-identification. Original data-collection approvals and consent or waiver procedures should be reported exactly as described in the primary dataset publications and data-use agreements. The reference VitalDB study reports Seoul National University Hospital Institutional Review Board approval (H-1408-101-605) and registration as NCT02914444. The authors should confirm whether an additional local institutional determination is required before submission and should not insert an unverified exemption or approval number.
"""
    (REPORT_DIR / "manuscript_ready_results.md").write_text(english, encoding="utf-8")


def qa_table(
    source: pd.DataFrame,
    update: pd.DataFrame,
    test: pd.DataFrame,
    analysis: dict[str, Any],
    config: dict[str, Any],
    expected_replicates: int,
) -> pd.DataFrame:
    checks = [
        ("source_unique_patients", source["patient_key"].is_unique, f"n={len(source)}"),
        ("update_unique_patients", update["patient_key"].is_unique, f"n={len(update)}"),
        ("test_unique_patients", test["patient_key"].is_unique, f"n={len(test)}"),
        ("update_test_no_patient_overlap", len(set(update["patient_key"]) & set(test["patient_key"])) == 0, "first eligible surgery design"),
        ("source_events_feasible", int(source["aki"].sum()) >= 100, f"events={int(source['aki'].sum())}"),
        ("update_events_feasible", int(update["aki"].sum()) >= 100, f"events={int(update['aki'].sum())}"),
        ("test_events_feasible", int(test["aki"].sum()) >= 100, f"events={int(test['aki'].sum())}"),
        ("target_years_locked", set(update["year"].dropna().astype(int)) == {2021} and set(test["year"].dropna().astype(int)) == {2022}, "2021 update; 2022 test"),
        ("no_2023_in_primary", not update["year"].eq(2023).any() and not test["year"].eq(2023).any(), "2023 excluded"),
        ("source_models_saved", all((MODEL_DIR / f"source_{name}.joblib").exists() for name in ["logistic_spline", "hist_gradient_boosting"]), "serialized objects"),
        ("bootstrap_replicates", int(analysis["bootstrap"]["replicate"].nunique()) == int(expected_replicates), f"expected={expected_replicates}"),
        ("predictions_finite", all(np.isfinite(values).all() for store in analysis["predictions"].values() for values in store.values()), "all model predictions"),
        ("predictions_in_unit_interval", all(((values >= 0) & (values <= 1)).all() for store in analysis["predictions"].values() for values in store.values()), "all model predictions"),
        ("readiness_nodes_present", analysis["readiness"]["event_node"].nunique() >= 5, f"nodes={analysis['readiness']['event_node'].nunique()}"),
    ]
    table = pd.DataFrame([{"check": name, "passed": bool(passed), "detail": detail} for name, passed, detail in checks])
    write_table(table, "21_qa_checks.csv")
    return table


def table_index() -> pd.DataFrame:
    descriptions = {
        "01_source_file_audit.csv": "Raw/analysis-ready input locations, sizes, timestamps, and SHA-256 hashes",
        "02_mover_creatinine_code_counts.csv": "Strict blood creatinine LOINC counts after fresh raw-file extraction",
        "03_cohort_flow.csv": "Sequential cohort construction counts",
        "04_cohort_characteristics.csv": "Source, update, temporal evaluation, and supportive cohort characteristics",
        "05_source_internal_validation.csv": "Five-fold source internal validation",
        "06_unupdated_transport_performance.csv": "Unupdated model performance across data roles",
        "07_event_node_composition.csv": "Chronological local evidence composition by event node",
        "08_event_node_performance.csv": "Locked-test performance after each update method and event node",
        "09_readiness_probabilities.csv": "Bootstrap probabilities within strict, primary, and permissive tolerances",
        "10_readiness_thresholds.csv": "Minimum persistent local evidence nodes",
        "11_decision_workload.csv": "Alerts, true/false positives, and net benefit at fixed thresholds",
        "12_subgroup_performance.csv": "Locked-test subgroup performance with uncertainty",
        "13_temporal_quarters.csv": "MOVER 2022 quarterly stability",
        "14_vitaldb_supportive.csv": "VitalDB supportive stress-test performance",
        "15_outcome_48h_sensitivity.csv": "48-hour creatinine AKI sensitivity analysis",
        "16_penalized_recalibration_sensitivity.csv": "Shrinkage sensitivity for logistic recalibration",
        "17_all_surgeries_cluster_sensitivity.csv": "All eligible surgeries with patient-cluster bootstrap",
        "18_model_coefficients.csv": "Frozen logistic spline source-model coefficients",
        "19_model_metadata.csv": "Model object metadata and hashes",
        "20_analysis_deviations_and_limits.csv": "Prespecified feasibility limitations and documented departures",
        "21_qa_checks.csv": "Automated analytical QA checks",
        "22_runtime_versions.csv": "Software runtime versions",
        "23_ethics_and_data_governance.csv": "Ethics wording and data-governance actions",
        "24_author_metadata.csv": "Author order, correspondence, and equal-contribution metadata",
        "25_affiliation_metadata.csv": "Affiliation and corresponding-author department",
        "26_main_results_summary.csv": "High-level numerical findings",
        "27_testing_selection.csv": "Characteristics of postoperative creatinine-tested and untested patients",
        "28_external_source_equivalence.csv": "Byte-level comparison of mounted MOVER archive members and analysis inputs",
    }
    rows = []
    for path in sorted(TABLE_DIR.glob("*.csv")):
        frame = pd.read_csv(path)
        rows.append(
            {
                "file": path.name,
                "description": descriptions.get(path.name, "Analysis output table"),
                "rows": len(frame),
                "columns": len(frame.columns),
                "sha256": sha256_file(path),
            }
        )
    table = pd.DataFrame(rows)
    write_table(table, "00_analysis_index.csv")
    return table


def file_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "tmp" in path.parts or path.name.startswith("."):
            continue
        if path == MANIFEST_DIR / "file_manifest.csv":
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(MANIFEST_DIR / "file_manifest.csv", index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=None, help="Override locked bootstrap count for testing only")
    parser.add_argument("--reuse-mover-labs", action="store_true", help="Reuse the strict extraction in this version")
    parser.add_argument("--skip-hash", action="store_true", help="Skip input SHA-256 generation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    started = time.time()
    config = load_config()
    replicates = int(args.bootstrap or config["bootstrap_replicates"])
    seed = int(config["random_seed"])
    log = {"started_at": datetime.now(timezone.utc).isoformat(), "bootstrap_replicates": replicates}

    source_audit = source_file_audit(config, skip_hash=args.skip_hash)
    if not source_audit["exists"].all():
        missing = source_audit.loc[~source_audit["exists"], "path"].tolist()
        raise FileNotFoundError(f"Required local inputs missing: {missing}")
    storage = json.loads((MANIFEST_DIR / "storage_audit.json").read_text(encoding="utf-8"))
    external_not_mounted = bool(storage["external_data_volume_expected_but_not_mounted"])

    inspire_all, inspire_audit = build_inspire(config)
    mover_labs, mover_lab_audit = extract_mover_creatinine(config, reuse=args.reuse_mover_labs)
    mover_all, mover_audit = build_mover(config, mover_labs)
    vitaldb_all = load_vitaldb(config)
    write_json(
        {"INSPIRE": inspire_audit, "MOVER_labs": mover_lab_audit, "MOVER": mover_audit},
        MANIFEST_DIR / "cohort_rebuild_audit.json",
    )

    source = analysis_subset(inspire_all)
    update = analysis_subset(mover_all, year=2021)
    test = analysis_subset(mover_all, year=2022)
    vitaldb = analysis_subset(vitaldb_all)
    source.to_parquet(DATA_DIR / "analysis_source_INSPIRE.parquet", index=False)
    update.to_parquet(DATA_DIR / "analysis_update_MOVER_2021.parquet", index=False)
    test.to_parquet(DATA_DIR / "analysis_test_MOVER_2022.parquet", index=False)
    vitaldb.to_parquet(DATA_DIR / "analysis_supportive_VitalDB.parquet", index=False)

    flow_rows = []
    flow_rows.extend(cohort_flow_rows("INSPIRE", inspire_all))
    flow_rows.extend(cohort_flow_rows("MOVER", mover_all, years=[2021, 2022]))
    flow = pd.DataFrame(flow_rows)
    write_table(flow, "03_cohort_flow.csv")
    characteristics = summarize_characteristics(
        {"INSPIRE_source": source, "MOVER_2021_update": update, "MOVER_2022_locked_test": test, "VitalDB_supportive": vitaldb}
    )
    write_table(characteristics, "04_cohort_characteristics.csv")
    testing = testing_selection_table(mover_all)
    write_table(testing, "27_testing_selection.csv")

    if int(update["aki"].sum()) < 100 or int(test["aki"].sum()) < 100:
        raise RuntimeError(
            f"Feasibility stop: update events={int(update['aki'].sum())}, test events={int(test['aki'].sum())}; both must be >=100"
        )

    analysis = analyze_models(source, update, test, vitaldb, config, replicates, seed)
    source_cv = analysis["fixed_summary"][analysis["fixed_summary"]["cohort"] == "INSPIRE_5fold_CV"]
    write_table(source_cv, "05_source_internal_validation.csv")
    vital_support = analysis["fixed_summary"][analysis["fixed_summary"]["cohort"] == "VitalDB_supportive"]
    write_table(vital_support, "14_vitaldb_supportive.csv")
    analyze_subgroups_and_time(test, analysis, replicates, seed + 20_000)
    analyze_48h_outcome(inspire_all, mover_all, mover_all, config, replicates, seed + 30_000)
    analyze_penalized_recalibration(update, test, analysis, config, replicates, seed + 40_000)
    analyze_all_surgeries(inspire_all, mover_all, config, replicates, seed + 50_000)
    make_model_metadata_table(config)
    write_table(runtime_table(), "22_runtime_versions.csv")
    build_metadata_tables(config, external_not_mounted)
    main_result_table(source, update, test, analysis)
    qa = qa_table(source, update, test, analysis, config, replicates)
    make_figures(flow, update, test, analysis)
    generate_reports(source, update, test, analysis, external_not_mounted)
    table_index()

    log.update(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.time() - started,
            "source_n": len(source),
            "source_events": int(source["aki"].sum()),
            "update_n": len(update),
            "update_events": int(update["aki"].sum()),
            "test_n": len(test),
            "test_events": int(test["aki"].sum()),
            "qa_all_passed": bool(qa["passed"].all()),
        }
    )
    write_json(log, LOG_DIR / "run_summary.json")
    file_manifest()
    if not qa["passed"].all():
        failed = qa.loc[~qa["passed"], "check"].tolist()
        raise RuntimeError(f"QA failed: {failed}")
    print(json.dumps(log, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
