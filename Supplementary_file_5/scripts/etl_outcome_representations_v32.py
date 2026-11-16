"""Unchanged R2 calculation functions extracted from authoritative sources.
Path configuration and orchestration are supplied by portable.py, not this module.
Historical report/author metadata generators and legacy entry points are excluded.
"""
from __future__ import annotations


import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
import traceback
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
import etl_common_v32 as base


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


def interval_bounds_from_levels(
    values: pd.Series,
    levels: np.ndarray,
    valid_min: float,
    valid_max: float,
) -> tuple[pd.Series, pd.Series]:
    released = np.asarray(levels, dtype=float)
    lower_map = {
        float(level): float(valid_min if index == 0 else released[index - 1])
        for index, level in enumerate(released)
    }
    upper_map = {
        float(level): float(valid_max if index == len(released) - 1 else released[index + 1])
        for index, level in enumerate(released)
    }
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.map(lower_map), numeric.map(upper_map)

def apply_percentile_release(values: pd.Series, edges: np.ndarray, representatives: np.ndarray) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)
    finite = np.isfinite(numeric)
    bins = np.searchsorted(edges, numeric[finite], side="left")
    result[finite] = representatives[bins]
    return pd.Series(result, index=values.index)

def classify_exact_aki(frame: pd.DataFrame, baseline: str, max48: str, max7: str) -> pd.Series:
    from thresholds_r1 import ge_threshold
    observed = frame["tested_7d"].fillna(False).astype(bool) & frame[baseline].notna()
    event = (
        (frame[max48].notna() & ge_threshold(frame[max48] - frame[baseline], 0.3))
        | (frame[max7].notna() & ge_threshold(frame[max7], 1.5 * frame[baseline]))
    )
    return pd.Series(np.where(observed, event.astype(float), np.nan), index=frame.index)

def classify_interval_aki(
    frame: pd.DataFrame,
    baseline_lower: str,
    baseline_upper: str,
    max48_lower: str,
    max48_upper: str,
    max7_lower: str,
    max7_upper: str,
) -> tuple[pd.Series, pd.Series]:
    from thresholds_r1 import ge_threshold
    observed = frame["tested_7d"].fillna(False).astype(bool) & frame[baseline_lower].notna() & frame[baseline_upper].notna()
    definite = (
        (frame[max48_lower].notna() & ge_threshold(frame[max48_lower] - frame[baseline_upper], 0.3))
        | (frame[max7_lower].notna() & ge_threshold(frame[max7_lower], 1.5 * frame[baseline_upper]))
    )
    possible = (
        (frame[max48_upper].notna() & ge_threshold(frame[max48_upper] - frame[baseline_lower], 0.3))
        | (frame[max7_upper].notna() & ge_threshold(frame[max7_upper], 1.5 * frame[baseline_lower]))
    )
    return (
        pd.Series(np.where(observed, definite.astype(float), np.nan), index=frame.index),
        pd.Series(np.where(observed, possible.astype(float), np.nan), index=frame.index),
    )

def inspire_release_levels(config: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    values: list[np.ndarray] = []
    for chunk in base.iter_zip_gzip_csv(
        Path(config["raw_data"]["inspire_archive"]),
        "labs.csv.gz",
        usecols=["item_name", "value"],
    ):
        mask = chunk["item_name"].fillna("").astype(str).str.strip().str.casefold().eq("creatinine")
        numeric = pd.to_numeric(chunk.loc[mask, "value"], errors="coerce")
        numeric = numeric[numeric.between(0.05, 30.0, inclusive="both")]
        if len(numeric):
            values.append(numeric.to_numpy(dtype=float))
    all_values = np.concatenate(values)
    levels, counts = np.unique(all_values, return_counts=True)
    rows = []
    for index, (level, count) in enumerate(zip(levels, counts), start=1):
        rows.append(
            {
                "level_index": index,
                "released_value_mg_dl": level,
                "conservative_lower_mg_dl": 0.05 if index == 1 else levels[index - 2],
                "conservative_upper_mg_dl": 30.0 if index == len(levels) else levels[index],
                "laboratory_rows": int(count),
            }
        )
    return levels, pd.DataFrame(rows)
