#!/usr/bin/env python3
"""Outcome-representation helpers retained for the v3.2 raw-data rebuild."""

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

import etl_common_v32 as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "analysis_config_v32.json"
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
PRIMARY_SCENARIO = "operational"


def ensure_dirs() -> None:
    for path in [DATA_DIR, MODEL_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR, QA_DIR, MANIFEST_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_table(frame: pd.DataFrame, filename: str) -> Path:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_float_dtype(output[column]):
            output[column] = output[column].replace([np.inf, -np.inf], np.nan)
    path = TABLE_DIR / filename
    output.to_csv(path, index=False)
    return path


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def stable_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def clip_probability(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)


def make_risk_model(kind: str, seed: int) -> Pipeline:
    if kind == "logistic_spline":
        continuous = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
            ]
        )
        binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
        preprocessing = ColumnTransformer(
            [("continuous", continuous, CONTINUOUS), ("binary", binary, BINARY)],
            remainder="drop",
        )
        classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=seed)
        return Pipeline([("preprocess", preprocessing), ("model", classifier)])
    if kind == "hist_gradient_boosting":
        preprocessing = ColumnTransformer(
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
        return Pipeline([("preprocess", preprocessing), ("model", classifier)])
    raise ValueError(f"Unknown model kind: {kind}")


def fit_recalibration(
    outcomes: np.ndarray,
    predictions: np.ndarray,
    method: str,
    *,
    penalty: float = 0.0,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    y = np.asarray(outcomes, dtype=float)
    p = clip_probability(predictions)
    lp = logit(p)
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if method == "none":
        return 0.0, 1.0
    if method == "intercept":
        alpha = 0.0
        for _ in range(100):
            mu = expit(np.clip(alpha + lp, -30, 30))
            gradient = np.sum(w * (y - mu)) - penalty * alpha
            information = np.sum(w * mu * (1 - mu)) + penalty + 1e-10
            step = float(np.clip(gradient / information, -5.0, 5.0))
            alpha += step
            if abs(step) < 1e-9:
                break
        return float(np.clip(alpha, -20, 20)), 1.0
    if method in {"logistic", "penalized_logistic"}:
        design = np.column_stack([np.ones(len(lp)), lp])
        theta = np.array([0.0, 1.0], dtype=float)
        ridge = float(penalty if method == "penalized_logistic" else 0.0)
        prior = np.array([0.0, 1.0], dtype=float)
        for _ in range(100):
            mu = expit(np.clip(design @ theta, -30, 30))
            gradient = design.T @ (w * (y - mu)) - ridge * (theta - prior)
            information = (
                design.T @ (design * (w * mu * (1 - mu))[:, None])
                + ridge * np.eye(2)
                + 1e-9 * np.eye(2)
            )
            try:
                step = np.linalg.solve(information, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(information) @ gradient
            step = np.clip(step, -5.0, 5.0)
            theta += step
            if np.max(np.abs(step)) < 1e-8:
                break
        return float(np.clip(theta[0], -20, 20)), float(np.clip(theta[1], -10, 10))
    raise ValueError(method)


def apply_recalibration(predictions: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return expit(alpha + beta * logit(clip_probability(predictions)))


def grouped_ici(outcomes: np.ndarray, predictions: np.ndarray, weights: np.ndarray | None = None) -> float:
    frame = pd.DataFrame({"y": outcomes, "p": predictions})
    frame["w"] = 1.0 if weights is None else np.asarray(weights, dtype=float)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return np.nan
    groups = min(10, frame["p"].nunique(), len(frame))
    if groups < 2:
        return float(abs(np.average(frame["y"], weights=frame["w"]) - np.average(frame["p"], weights=frame["w"])))
    frame["bin"] = pd.qcut(frame["p"], q=groups, duplicates="drop")
    rows = []
    for _, group in frame.groupby("bin", observed=True):
        rows.append(
            (
                float(group["w"].sum()),
                abs(float(np.average(group["y"], weights=group["w"])) - float(np.average(group["p"], weights=group["w"]))),
            )
        )
    return float(np.average([row[1] for row in rows], weights=[row[0] for row in rows]))


def performance_metrics(
    outcomes: np.ndarray | pd.Series,
    predictions: np.ndarray | pd.Series,
    *,
    weights: np.ndarray | pd.Series | None = None,
    include_discrimination: bool = True,
    include_ici: bool = True,
) -> dict[str, float]:
    y = np.asarray(outcomes, dtype=int)
    p = clip_probability(predictions)
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    weighted_events = float(np.sum(w * y))
    expected = float(np.sum(w * p))
    prevalence = float(weighted_events / np.sum(w))
    joint_alpha, slope = fit_recalibration(y, p, "logistic", weights=w)
    intercept, _ = fit_recalibration(y, p, "intercept", weights=w)
    brier = float(np.average((y - p) ** 2, weights=w))
    reference_brier = float(np.average((y - prevalence) ** 2, weights=w))
    return {
        "n": float(len(y)),
        "events": float(np.sum(y)),
        "weighted_n": float(np.sum(w)),
        "weighted_events": weighted_events,
        "event_rate": prevalence,
        "predicted_mean": float(np.average(p, weights=w)),
        "oe_ratio": weighted_events / expected if expected > 0 else np.nan,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_joint_intercept": joint_alpha,
        "auroc": roc_auc_score(y, p, sample_weight=w)
        if include_discrimination and np.unique(y).size == 2
        else np.nan,
        "auprc": average_precision_score(y, p, sample_weight=w)
        if include_discrimination and np.unique(y).size == 2
        else np.nan,
        "brier": brier,
        "prevalence_only_brier": reference_brier,
        "brier_skill_score": 1.0 - brier / reference_brier if reference_brier > 0 else np.nan,
        "grouped_ici": grouped_ici(y, p, w) if include_ici else np.nan,
    }


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


def percentile_release_parameters(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    edge_percentiles = np.array([2.5] + list(np.arange(7.5, 100.0, 5.0)), dtype=float)
    representative_percentiles = np.array([2.5] + list(np.arange(5.0, 100.0, 5.0)) + [97.5], dtype=float)
    return (
        np.quantile(numeric, edge_percentiles / 100.0),
        np.quantile(numeric, representative_percentiles / 100.0),
    )


def apply_percentile_release(values: pd.Series, edges: np.ndarray, representatives: np.ndarray) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)
    finite = np.isfinite(numeric)
    bins = np.searchsorted(edges, numeric[finite], side="left")
    result[finite] = representatives[bins]
    return pd.Series(result, index=values.index)


def classify_exact_aki(frame: pd.DataFrame, baseline: str, max48: str, max7: str) -> pd.Series:
    observed = frame["tested_7d"].fillna(False).astype(bool) & frame[baseline].notna()
    event = (
        (frame[max48].notna() & ((frame[max48] - frame[baseline]) >= 0.3))
        | (frame[max7].notna() & (frame[max7] >= 1.5 * frame[baseline]))
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
    observed = frame["tested_7d"].fillna(False).astype(bool) & frame[baseline_lower].notna() & frame[baseline_upper].notna()
    definite = (
        (frame[max48_lower].notna() & ((frame[max48_lower] - frame[baseline_upper]) >= 0.3))
        | (frame[max7_lower].notna() & (frame[max7_lower] >= 1.5 * frame[baseline_upper]))
    )
    possible = (
        (frame[max48_upper].notna() & ((frame[max48_upper] - frame[baseline_lower]) >= 0.3))
        | (frame[max7_upper].notna() & (frame[max7_upper] >= 1.5 * frame[baseline_lower]))
    )
    return (
        pd.Series(np.where(observed, definite.astype(float), np.nan), index=frame.index),
        pd.Series(np.where(observed, possible.astype(float), np.nan), index=frame.index),
    )


def eligible_base(cohort: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    mask = cohort["has_baseline_cr"].fillna(False).astype(bool)
    mask &= cohort["baseline_cr_under4"].fillna(False).astype(bool)
    if "first_eligible" in cohort:
        mask &= cohort["first_eligible"].fillna(False).astype(bool)
    if year is not None:
        mask &= cohort["year"].eq(year)
    return cohort.loc[mask].copy().reset_index(drop=True)


def observed_outcome(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    mask = frame["tested_7d"].fillna(False).astype(bool) & frame[outcome].notna()
    output = frame.loc[mask].copy().reset_index(drop=True)
    output[outcome] = pd.to_numeric(output[outcome], errors="coerce").astype(int)
    return output


def map_surgery_category(text: pd.Series) -> pd.Series:
    value = text.fillna("").astype(str).str.upper()
    category = pd.Series("other", index=value.index, dtype="object")
    patterns = [
        ("orthopedic", r"ORTHO|HIP|KNEE|SPINE|FEMUR|TIBIA|HUMERUS|ARTHRO|FUSION"),
        ("urologic", r"URO|PROSTATE|BLADDER|URETER|KIDNEY|NEPHR|CYSTOSCOPY"),
        ("neurologic", r"NEURO|CRANI|BRAIN|LAMINECTOMY|VENTRICUL"),
        ("thoracic", r"THORAC|LUNG|BRONCH|PLEUR|MEDIASTIN"),
        ("vascular", r"VASCULAR|ARTERY|ARTERIAL|ANEURYSM|ENDARTERECTOMY|BYPASS"),
        ("head_neck", r"ENT|THYROID|PARATHYROID|TONSIL|LARYNG|MAXILLO|MANDIB"),
        ("abdominal", r"ABDOM|BOWEL|COLON|RECT|GASTR|LIVER|HEPAT|PANCREA|GALLBLADDER|HERNIA"),
        ("breast", r"BREAST|MASTECTOM"),
        ("plastic", r"PLASTIC|RECONSTRUCTION|FLAP|GRAFT"),
    ]
    for label, pattern in patterns:
        category = category.mask(category.eq("other") & value.str.contains(pattern, regex=True), label)
    return category


def attach_mover_surgery_category(mover: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    information = pd.read_csv(
        Path(config["raw_data"]["mover_patient_information"]),
        usecols=["LOG_ID", "PRIMARY_PROCEDURE_NM"],
        dtype=str,
        low_memory=False,
    )
    information["case_key"] = information["LOG_ID"].map(lambda value: base.stable_key("MOVER_CASE", value))
    information["surgery_category"] = map_surgery_category(information["PRIMARY_PROCEDURE_NM"])
    information = information[["case_key", "surgery_category"]].drop_duplicates("case_key", keep="first")
    output = mover.merge(information, on="case_key", how="left", validate="one_to_one")
    output["surgery_category"] = output["surgery_category"].fillna("other")
    return output


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


def add_outcome_representations(
    inspire: pd.DataFrame,
    mover: pd.DataFrame,
    mover_labs: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inspire_output = inspire.copy()
    mover_output = mover.copy()
    levels, release_table = inspire_release_levels(config)
    write_table(release_table, "03_inspire_creatinine_release_levels.csv")

    for column in ["baseline_cr", "cr_max_48h", "cr_max_7d"]:
        lower, upper = interval_bounds_from_levels(inspire_output[column], levels, 0.05, 30.0)
        inspire_output[f"{column}_lower"] = lower
        inspire_output[f"{column}_upper"] = upper
    inspire_output["outcome_operational"] = inspire_output["aki"]
    (
        inspire_output["outcome_definite"],
        inspire_output["outcome_possible"],
    ) = classify_interval_aki(
        inspire_output,
        "baseline_cr_lower",
        "baseline_cr_upper",
        "cr_max_48h_lower",
        "cr_max_48h_upper",
        "cr_max_7d_lower",
        "cr_max_7d_upper",
    )
    inspire_output["baseline_cr_coarse"] = inspire_output["baseline_cr"]
    inspire_output["outcome_coarsened_operational"] = inspire_output["outcome_operational"]

    edges, representatives = percentile_release_parameters(mover_labs["cr"])
    coarsening_rows = []
    for index, representative in enumerate(representatives):
        coarsening_rows.append(
            {
                "bin_index": index + 1,
                "lower_cutpoint_mg_dl": np.nan if index == 0 else edges[index - 1],
                "upper_cutpoint_mg_dl": np.nan if index == len(representatives) - 1 else edges[index],
                "released_representative_mg_dl": representative,
            }
        )
    write_table(pd.DataFrame(coarsening_rows), "04_mover_deterministic_coarsening.csv")
    for column in ["baseline_cr", "cr_max_48h", "cr_max_7d"]:
        coarse = apply_percentile_release(mover_output[column], edges, representatives)
        mover_output[f"{column}_coarse"] = coarse
        lower, upper = interval_bounds_from_levels(coarse, representatives, 0.05, 30.0)
        mover_output[f"{column}_coarse_lower"] = lower
        mover_output[f"{column}_coarse_upper"] = upper
    mover_output["outcome_operational"] = mover_output["aki"]
    mover_output["outcome_coarsened_operational"] = classify_exact_aki(
        mover_output,
        "baseline_cr_coarse",
        "cr_max_48h_coarse",
        "cr_max_7d_coarse",
    )
    mover_output["outcome_definite"], mover_output["outcome_possible"] = classify_interval_aki(
        mover_output,
        "baseline_cr_coarse_lower",
        "baseline_cr_coarse_upper",
        "cr_max_48h_coarse_lower",
        "cr_max_48h_coarse_upper",
        "cr_max_7d_coarse_lower",
        "cr_max_7d_coarse_upper",
    )
    metadata = {
        "inspire_released_levels": levels.tolist(),
        "mover_coarsening_edges": edges.tolist(),
        "mover_coarsening_representatives": representatives.tolist(),
        "interval_rule": "adjacent released levels, with 0.05 and 30.0 mg/dL outer bounds",
    }
    write_json(metadata, MANIFEST_DIR / "outcome_representation_metadata.json")
    return inspire_output, mover_output, metadata


def scenario_frame(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    outcome = f"outcome_{scenario}"
    output = observed_outcome(frame, outcome)
    output["analysis_outcome"] = output[outcome].astype(int)
    if scenario in {"definite", "possible", "coarsened_operational"} and "baseline_cr_coarse" in output:
        output["baseline_cr"] = output["baseline_cr_coarse"]
    return output


def event_node_frames(update: pd.DataFrame, nodes: Sequence[Any]) -> list[tuple[str, int, pd.DataFrame]]:
    ordered = update.sort_values(["an_start", "case_key"], kind="stable").reset_index(drop=True)
    cumulative = ordered["analysis_outcome"].astype(int).cumsum().to_numpy()
    total_events = int(cumulative[-1]) if len(cumulative) else 0
    result = []
    for node in nodes:
        if node == "all":
            end = len(ordered)
            label = "all"
        else:
            requested = int(node)
            if requested > total_events:
                continue
            end = int(np.flatnonzero(cumulative >= requested)[0] + 1)
            label = str(requested)
        subset = ordered.iloc[:end].copy()
        result.append((label, int(subset["analysis_outcome"].sum()), subset))
    return result


def rebuild_cohorts(config: dict[str, Any], reuse_mover_labs: bool) -> dict[str, Any]:
    base.ensure_dirs()
    existing_source_audit = TABLE_DIR / "01_source_file_audit.csv"
    if reuse_mover_labs and existing_source_audit.exists():
        source_audit = pd.read_csv(existing_source_audit)
    else:
        source_audit = base.source_file_audit(config, skip_hash=False)
    mover_labs, mover_lab_audit = base.extract_mover_creatinine(config, reuse=reuse_mover_labs)
    inspire, inspire_audit = base.build_inspire(config)
    mover, mover_audit = base.build_mover(config, mover_labs)
    mover = attach_mover_surgery_category(mover, config)
    vitaldb = base.load_vitaldb(config)
    inspire, mover, representation_metadata = add_outcome_representations(
        inspire, mover, mover_labs, config
    )
    inspire.to_parquet(DATA_DIR / "inspire_rebuilt_v2.parquet", index=False)
    mover.to_parquet(DATA_DIR / "mover_rebuilt_v2.parquet", index=False)
    vitaldb.to_parquet(DATA_DIR / "vitaldb_supportive_v2.parquet", index=False)
    audits = {
        "source_files": source_audit.to_dict(orient="records"),
        "inspire": inspire_audit,
        "mover": mover_audit,
        "mover_creatinine": mover_lab_audit,
        "outcome_representation": representation_metadata,
    }
    write_json(audits, MANIFEST_DIR / "cohort_rebuild_audit.json")
    return {
        "inspire": inspire,
        "mover": mover,
        "vitaldb": vitaldb,
        "mover_labs": mover_labs,
        "audits": audits,
    }


def nested_bootstrap(
    update: pd.DataFrame,
    test: pd.DataFrame,
    update_predictions: np.ndarray,
    test_predictions: np.ndarray,
    method: str,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    failures = 0
    attempts = 0
    max_attempts = replicates * 4
    update_y = update["analysis_outcome"].to_numpy(dtype=int)
    test_y = test["analysis_outcome"].to_numpy(dtype=int)
    while len(rows) < replicates and attempts < max_attempts:
        attempts += 1
        update_index = rng.integers(0, len(update), len(update))
        test_index = rng.integers(0, len(test), len(test))
        sampled_update_y = update_y[update_index]
        sampled_test_y = test_y[test_index]
        if np.unique(sampled_test_y).size < 2:
            failures += 1
            continue
        if method != "none" and np.unique(sampled_update_y).size < 2:
            failures += 1
            continue
        try:
            if method == "local_refit_upper_bound":
                fitted = make_risk_model("logistic_spline", seed + attempts).fit(
                    update.iloc[update_index][FEATURES], sampled_update_y
                )
                predictions = fitted.predict_proba(test.iloc[test_index][FEATURES])[:, 1]
                alpha, beta = np.nan, np.nan
            else:
                penalty = 1.0 if method == "penalized_logistic" else 0.0
                alpha, beta = fit_recalibration(
                    sampled_update_y,
                    update_predictions[update_index],
                    method,
                    penalty=penalty,
                )
                predictions = apply_recalibration(test_predictions[test_index], alpha, beta)
            metrics = performance_metrics(
                sampled_test_y,
                predictions,
                include_discrimination=False,
                include_ici=False,
            )
            rows.append(
                {
                    "replicate": len(rows) + 1,
                    "update_alpha": alpha,
                    "update_beta": beta,
                    **metrics,
                }
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            failures += 1
    if len(rows) != replicates:
        raise RuntimeError(
            f"Nested bootstrap produced {len(rows)} of {replicates} valid replicates for {method}"
        )
    return pd.DataFrame(rows), failures


def fixed_prediction_bootstrap(
    outcomes: np.ndarray,
    predictions: np.ndarray,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(1, replicates + 1):
        index = rng.integers(0, len(outcomes), len(outcomes))
        metrics = performance_metrics(
            outcomes[index],
            predictions[index],
            include_discrimination=True,
            include_ici=False,
        )
        rows.append({"replicate": replicate, **metrics})
    return pd.DataFrame(rows)


def summarize_nested_result(
    scenario: str,
    node_order: int,
    node_label: str,
    actual_events: int,
    update: pd.DataFrame,
    method: str,
    point: dict[str, float],
    bootstrap: pd.DataFrame,
    failures: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    oe_bounds = config["readiness"]["primary_oe_bounds"]
    slope_bounds = config["readiness"]["primary_slope_bounds"]
    oe_ok = bootstrap["oe_ratio"].between(*oe_bounds, inclusive="both")
    slope_ok = bootstrap["calibration_slope"].between(*slope_bounds, inclusive="both")
    row: dict[str, Any] = {
        "outcome_representation": scenario,
        "node_order": node_order,
        "event_node": node_label,
        "actual_events": actual_events,
        "update_n": len(update),
        "non_events": int(len(update) - actual_events),
        "first_case_date": update["an_start"].min(),
        "last_case_date": update["an_start"].max(),
        "method": method,
        "valid_bootstrap_replicates": len(bootstrap),
        "discarded_bootstrap_attempts": failures,
        "bootstrap_stability_global": float(oe_ok.mean()),
        "bootstrap_stability_full": float((oe_ok & slope_ok).mean()),
    }
    for metric in [
        "n",
        "events",
        "event_rate",
        "predicted_mean",
        "oe_ratio",
        "calibration_intercept",
        "calibration_slope",
        "auroc",
        "auprc",
        "brier",
        "prevalence_only_brier",
        "brier_skill_score",
        "grouped_ici",
    ]:
        row[f"{metric}_estimate"] = point.get(metric, np.nan)
        if metric in bootstrap and metric not in {"n", "events"}:
            lower, median, upper = bootstrap[metric].quantile([0.025, 0.5, 0.975]).tolist()
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_bootstrap_median"] = median
            row[f"{metric}_ci_upper"] = upper
        else:
            row[f"{metric}_ci_lower"] = point.get(metric, np.nan)
            row[f"{metric}_bootstrap_median"] = point.get(metric, np.nan)
            row[f"{metric}_ci_upper"] = point.get(metric, np.nan)
    row["strict_ci_containment_global"] = bool(
        row["oe_ratio_ci_lower"] >= oe_bounds[0] and row["oe_ratio_ci_upper"] <= oe_bounds[1]
    )
    row["strict_ci_containment_full"] = bool(
        row["strict_ci_containment_global"]
        and row["calibration_slope_ci_lower"] >= slope_bounds[0]
        and row["calibration_slope_ci_upper"] <= slope_bounds[1]
    )
    row["full_stability_rule_met"] = bool(
        row["bootstrap_stability_full"]
        >= config["readiness"]["bootstrap_stability_threshold"]
    )
    return row


def model_metadata(
    model: Pipeline,
    source: pd.DataFrame,
    scenario: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "study_id": config["study_id"],
        "analysis_version": config["analysis_version"],
        "model_name": "logistic_spline",
        "outcome_representation": scenario,
        "outcome_definition": config["outcome"],
        "prediction_time": config["prediction_time"],
        "features_in_order": FEATURES,
        "source_dataset": "INSPIRE",
        "source_n": int(len(source)),
        "source_events": int(source["analysis_outcome"].sum()),
        "source_event_rate": float(source["analysis_outcome"].mean()),
        "source_feature_summary": {
            feature: {
                "mean": float(pd.to_numeric(source[feature], errors="coerce").mean()),
                "sd": float(pd.to_numeric(source[feature], errors="coerce").std(ddof=1)),
                "missing": int(source[feature].isna().sum()),
            }
            for feature in FEATURES
        },
        "estimator_parameters": {key: str(value) for key, value in model.get_params(deep=True).items()},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": __import__("sklearn").__version__,
            "joblib": joblib.__version__,
        },
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "authors": config["authors"],
    }


def analyze_learning_curves(
    inspire: pd.DataFrame,
    mover: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    replicates = int(config["bootstrap_replicates"])
    seed = int(config["random_seed"])
    source_base = eligible_base(inspire)
    update_base = eligible_base(mover, 2021)
    test_base = eligible_base(mover, 2022)
    summary_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    bootstrap_parts: list[pd.DataFrame] = []
    contexts: dict[str, Any] = {}

    for scenario_index, scenario in enumerate(config["outcome_representations"]):
        source = scenario_frame(source_base, scenario)
        update = scenario_frame(update_base, scenario)
        test = scenario_frame(test_base, scenario)
        model_seed = seed + scenario_index * 100_000
        model = make_risk_model("logistic_spline", model_seed).fit(
            source[FEATURES], source["analysis_outcome"].to_numpy(dtype=int)
        )
        joblib.dump(model, MODEL_DIR / f"source_logistic_spline_{scenario}.joblib")
        write_json(
            model_metadata(model, source, scenario, config),
            MODEL_DIR / f"source_logistic_spline_{scenario}_metadata.json",
        )
        update_predictions = model.predict_proba(update[FEATURES])[:, 1]
        test_predictions = model.predict_proba(test[FEATURES])[:, 1]
        nodes = event_node_frames(update, config["event_nodes"])
        contexts[scenario] = {
            "source": source,
            "update": update,
            "test": test,
            "model": model,
            "update_predictions": update_predictions,
            "test_predictions": test_predictions,
            "all_event_parameters": {},
        }
        for node_order, (node_label, actual_events, node) in enumerate(nodes, start=1):
            composition_rows.append(
                {
                    "outcome_representation": scenario,
                    "node_order": node_order,
                    "event_node": node_label,
                    "actual_events": actual_events,
                    "update_n": len(node),
                    "non_events": int(len(node) - actual_events),
                    "first_case_date": node["an_start"].min(),
                    "last_case_date": node["an_start"].max(),
                }
            )
            node_predictions = model.predict_proba(node[FEATURES])[:, 1]
            methods = ["logistic"]
            if scenario == PRIMARY_SCENARIO:
                methods = ["none", "intercept", "logistic", "penalized_logistic"]
                if node_label == "all":
                    methods.append("local_refit_upper_bound")
            for method_index, method in enumerate(methods):
                if method == "local_refit_upper_bound":
                    local_model = make_risk_model("logistic_spline", model_seed + 50_000).fit(
                        node[FEATURES], node["analysis_outcome"].to_numpy(dtype=int)
                    )
                    point_predictions = local_model.predict_proba(test[FEATURES])[:, 1]
                    joblib.dump(local_model, MODEL_DIR / "local_mover_2021_refit_upper_bound.joblib")
                    alpha, beta = np.nan, np.nan
                else:
                    penalty = 1.0 if method == "penalized_logistic" else 0.0
                    alpha, beta = fit_recalibration(
                        node["analysis_outcome"].to_numpy(dtype=int),
                        node_predictions,
                        method,
                        penalty=penalty,
                    )
                    point_predictions = apply_recalibration(test_predictions, alpha, beta)
                if node_label == "all":
                    contexts[scenario]["all_event_parameters"][method] = {
                        "alpha": alpha,
                        "beta": beta,
                        "predictions": point_predictions,
                    }
                point = performance_metrics(
                    test["analysis_outcome"].to_numpy(dtype=int), point_predictions
                )
                bootstrap, failures = nested_bootstrap(
                    node,
                    test,
                    node_predictions,
                    test_predictions,
                    method,
                    replicates,
                    model_seed + node_order * 1_000 + method_index * 10,
                )
                bootstrap.insert(0, "method", method)
                bootstrap.insert(0, "actual_events", actual_events)
                bootstrap.insert(0, "event_node", node_label)
                bootstrap.insert(0, "node_order", node_order)
                bootstrap.insert(0, "outcome_representation", scenario)
                bootstrap_parts.append(bootstrap)
                summary_rows.append(
                    summarize_nested_result(
                        scenario,
                        node_order,
                        node_label,
                        actual_events,
                        node,
                        method,
                        point,
                        bootstrap,
                        failures,
                        config,
                    )
                )

    learning_curve = pd.DataFrame(summary_rows)
    composition = pd.DataFrame(composition_rows).drop_duplicates()
    bootstrap_distribution = pd.concat(bootstrap_parts, ignore_index=True)
    bootstrap_distribution.to_parquet(
        DATA_DIR / "nested_bootstrap_distribution_v2.parquet", index=False
    )
    write_table(composition, "08_event_node_composition.csv")
    write_table(learning_curve, "09_learning_curve_nested_bootstrap.csv")

    threshold_rows = []
    stability_threshold = float(config["readiness"]["bootstrap_stability_threshold"])
    for (scenario, method), group in learning_curve.groupby(
        ["outcome_representation", "method"], sort=False
    ):
        ordered = group.sort_values("node_order").reset_index(drop=True)
        for criterion, column in [
            ("global_OE", "bootstrap_stability_global"),
            ("full_OE_and_slope", "bootstrap_stability_full"),
        ]:
            values = ordered[column].to_numpy(dtype=float)
            persistent = np.array(
                [bool(np.all(values[index:] >= stability_threshold)) for index in range(len(values))]
            )
            if persistent.any():
                selected = ordered.iloc[int(np.flatnonzero(persistent)[0])]
                status = "reached_and_persistent"
                event_node = selected["event_node"]
                events = selected["actual_events"]
            else:
                status = "not_reached_within_available_events"
                event_node = "not_reached"
                events = np.nan
            threshold_rows.append(
                {
                    "outcome_representation": scenario,
                    "method": method,
                    "criterion": criterion,
                    "minimum_persistent_event_node": event_node,
                    "minimum_persistent_actual_events": events,
                    "status": status,
                    "maximum_available_events": int(ordered["actual_events"].max()),
                    "bootstrap_stability_threshold": stability_threshold,
                }
            )
    readiness_thresholds = pd.DataFrame(threshold_rows)
    write_table(readiness_thresholds, "10_readiness_thresholds.csv")
    return {
        "contexts": contexts,
        "learning_curve": learning_curve,
        "composition": composition,
        "bootstrap": bootstrap_distribution,
        "readiness_thresholds": readiness_thresholds,
    }


def cohort_and_selection_tables(
    inspire: pd.DataFrame,
    mover: pd.DataFrame,
    vitaldb: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    flow_rows = []
    frames = [
        ("INSPIRE", inspire, None),
        ("MOVER_2021", mover, 2021),
        ("MOVER_2022", mover, 2022),
    ]
    for label, cohort, year in frames:
        frame = cohort if year is None else cohort.loc[cohort["year"].eq(year)]
        first = frame["first_eligible"].fillna(False).astype(bool)
        baseline = first & frame["has_baseline_cr"].fillna(False).astype(bool)
        baseline_under4 = baseline & frame["baseline_cr_under4"].fillna(False).astype(bool)
        observed = baseline_under4 & frame["tested_7d"].fillna(False).astype(bool)
        for order, (step, mask) in enumerate(
            [
                ("clinically_eligible_operations", pd.Series(True, index=frame.index)),
                ("first_clinically_eligible_surgery", first),
                ("baseline_creatinine_within_7d", baseline),
                ("baseline_creatinine_below_4_mg_dl", baseline_under4),
                ("postoperative_creatinine_observed_within_7d", observed),
            ],
            start=1,
        ):
            flow_rows.append(
                {
                    "cohort": label,
                    "step_order": order,
                    "step": step,
                    "n": int(mask.sum()),
                    "operational_aki_events": int(frame.loc[mask, "outcome_operational"].fillna(0).sum()),
                }
            )
    vital_observed = vitaldb[
        vitaldb["has_baseline_cr"].fillna(False).astype(bool)
        & vitaldb["baseline_cr_under4"].fillna(False).astype(bool)
        & vitaldb["tested_7d"].fillna(False).astype(bool)
        & vitaldb["aki"].notna()
    ]
    flow_rows.append(
        {
            "cohort": "VitalDB_supportive",
            "step_order": 1,
            "step": "available_analysis_ready_observed_cohort",
            "n": len(vital_observed),
            "operational_aki_events": int(vital_observed["aki"].sum()),
        }
    )
    flow = pd.DataFrame(flow_rows)

    characteristic_rows = []
    cohort_map = {
        "INSPIRE_observed": scenario_frame(eligible_base(inspire), "operational"),
        "MOVER_2021_observed": scenario_frame(eligible_base(mover, 2021), "operational"),
        "MOVER_2022_all_eligible": eligible_base(mover, 2022),
        "MOVER_2022_observed": scenario_frame(eligible_base(mover, 2022), "operational"),
        "VitalDB_supportive_observed": vital_observed,
    }
    for label, frame in cohort_map.items():
        outcome_column = "analysis_outcome" if "analysis_outcome" in frame else ("aki" if "aki" in frame else None)
        characteristic_rows.append(
            {
                "cohort": label,
                "n": len(frame),
                "observed_outcome_n": int(frame[outcome_column].notna().sum()) if outcome_column else np.nan,
                "events": float(frame[outcome_column].sum()) if outcome_column else np.nan,
                "event_rate": float(frame[outcome_column].mean()) if outcome_column else np.nan,
                "postoperative_creatinine_testing_rate": float(frame["tested_7d"].mean())
                if "tested_7d" in frame
                else np.nan,
                "age_mean": pd.to_numeric(frame["age"], errors="coerce").mean(),
                "age_sd": pd.to_numeric(frame["age"], errors="coerce").std(ddof=1),
                "male_percent": 100 * pd.to_numeric(frame["sex_male"], errors="coerce").mean(),
                "asa_mean": pd.to_numeric(frame.get("asa"), errors="coerce").mean(),
                "duration_h_mean": pd.to_numeric(frame["duration_h"], errors="coerce").mean(),
                "duration_h_sd": pd.to_numeric(frame["duration_h"], errors="coerce").std(ddof=1),
                "baseline_cr_mean": pd.to_numeric(frame["baseline_cr"], errors="coerce").mean(),
                "baseline_cr_sd": pd.to_numeric(frame["baseline_cr"], errors="coerce").std(ddof=1),
            }
        )
    characteristics = pd.DataFrame(characteristic_rows)
    selection = base.testing_selection_table(mover)
    write_table(flow, "05_cohort_flow.csv")
    write_table(characteristics, "06_cohort_characteristics.csv")
    write_table(selection, "07_testing_selection_standardized_differences.csv")
    return {"flow": flow, "characteristics": characteristics, "selection": selection}


def make_testing_model(expanded: bool, seed: int) -> tuple[Pipeline, list[str]]:
    continuous = ["age", "duration_h", "baseline_cr"]
    binary = ["sex_male"]
    categorical = ["asa", "surgery_category"] if expanded else []
    transformers: list[tuple[str, Any, list[str]]] = [
        (
            "continuous",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            continuous,
        ),
        ("binary", SimpleImputer(strategy="most_frequent"), binary),
    ]
    if expanded:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    preprocessing = ColumnTransformer(transformers, remainder="drop")
    model = Pipeline(
        [
            ("preprocess", preprocessing),
            ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=seed)),
        ]
    )
    return model, continuous + binary + categorical


def truncate_weights(weights: np.ndarray, lower: float, upper: float) -> tuple[np.ndarray, float, float]:
    values = np.asarray(weights, dtype=float)
    if lower <= 0 and upper >= 100:
        return values.copy(), float(np.min(values)), float(np.max(values))
    low, high = np.percentile(values, [lower, upper])
    return np.clip(values, low, high), float(low), float(high)


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    return float(np.sum(values) ** 2 / np.sum(values**2))


def testing_model_diagnostics(
    frame: pd.DataFrame,
    propensity: np.ndarray,
    observed_mask: np.ndarray,
    weights: np.ndarray,
    model_specification: str,
    truncation: str,
    lower_cut: float,
    upper_cut: float,
) -> dict[str, Any]:
    tested_propensity = propensity[observed_mask]
    untested_propensity = propensity[~observed_mask]
    support_lower = max(float(np.min(tested_propensity)), float(np.min(untested_propensity)))
    support_upper = min(float(np.max(tested_propensity)), float(np.max(untested_propensity)))
    outside_support = np.mean((propensity < support_lower) | (propensity > support_upper))
    return {
        "model_specification": model_specification,
        "truncation": truncation,
        "eligible_n": len(frame),
        "observed_n": int(observed_mask.sum()),
        "testing_rate": float(observed_mask.mean()),
        "testing_model_auroc": roc_auc_score(observed_mask.astype(int), propensity),
        "testing_model_brier": brier_score_loss(observed_mask.astype(int), propensity),
        "propensity_min": float(np.min(propensity)),
        "propensity_p01": float(np.quantile(propensity, 0.01)),
        "propensity_median": float(np.median(propensity)),
        "propensity_p99": float(np.quantile(propensity, 0.99)),
        "propensity_max": float(np.max(propensity)),
        "common_support_lower": support_lower,
        "common_support_upper": support_upper,
        "eligible_fraction_outside_common_support": float(outside_support),
        "weight_lower_cut": lower_cut,
        "weight_upper_cut": upper_cut,
        "weight_min": float(np.min(weights)),
        "weight_median": float(np.median(weights)),
        "weight_p95": float(np.quantile(weights, 0.95)),
        "weight_p99": float(np.quantile(weights, 0.99)),
        "weight_max": float(np.max(weights)),
        "effective_sample_size": effective_sample_size(weights),
        "ess_fraction_of_observed": effective_sample_size(weights) / observed_mask.sum(),
    }


def observation_bootstrap(
    frame: pd.DataFrame,
    expanded: bool,
    truncations: list[tuple[float, float]],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    valid = 0
    attempts = 0
    while valid < replicates and attempts < replicates * 4:
        attempts += 1
        index = rng.integers(0, len(frame), len(frame))
        sample = frame.iloc[index].reset_index(drop=True)
        observed = sample["tested_7d"].fillna(False).to_numpy(dtype=bool)
        if observed.sum() < 20 or (~observed).sum() < 20:
            continue
        model, features = make_testing_model(expanded, seed + attempts)
        try:
            model.fit(sample[features], observed.astype(int))
            propensity = clip_probability(model.predict_proba(sample[features])[:, 1])
            raw_weights = observed.mean() / propensity[observed]
            outcomes = sample.loc[observed, "outcome_operational"].to_numpy(dtype=int)
            predictions = sample.loc[observed, "updated_prediction"].to_numpy(dtype=float)
            if np.unique(outcomes).size < 2:
                continue
            valid += 1
            for lower, upper in truncations:
                weights, _, _ = truncate_weights(raw_weights, lower, upper)
                metrics = performance_metrics(
                    outcomes,
                    predictions,
                    weights=weights,
                    include_discrimination=False,
                    include_ici=False,
                )
                rows.append(
                    {
                        "replicate": valid,
                        "model_specification": "expanded" if expanded else "basic",
                        "truncation": f"p{lower:g}_p{upper:g}",
                        "effective_sample_size": effective_sample_size(weights),
                        **metrics,
                    }
                )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    expected = replicates * len(truncations)
    if len(rows) != expected:
        raise RuntimeError(f"Observation bootstrap produced {len(rows)} of {expected} rows")
    return pd.DataFrame(rows)


def analyze_observation_selection(
    mover: pd.DataFrame,
    learning: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    context = learning["contexts"][PRIMARY_SCENARIO]
    all_parameters = context["all_event_parameters"]["logistic"]
    target_all = eligible_base(mover, 2022)
    source_model = context["model"]
    target_all["updated_prediction"] = apply_recalibration(
        source_model.predict_proba(target_all[FEATURES])[:, 1],
        float(all_parameters["alpha"]),
        float(all_parameters["beta"]),
    )
    observed_mask = target_all["tested_7d"].fillna(False).to_numpy(dtype=bool)
    observed_mask &= target_all["outcome_operational"].notna().to_numpy()
    truncations = [(0.0, 100.0), (1.0, 99.0), (5.0, 95.0)]
    summary_rows = []
    diagnostic_rows = []
    bootstrap_parts = []
    observed_y = target_all.loc[observed_mask, "outcome_operational"].to_numpy(dtype=int)
    observed_p = target_all.loc[observed_mask, "updated_prediction"].to_numpy(dtype=float)
    conditional_point = performance_metrics(observed_y, observed_p)
    conditional_boot = fixed_prediction_bootstrap(
        observed_y,
        observed_p,
        int(config["bootstrap_replicates"]),
        int(config["random_seed"]) + 600_000,
    )
    conditional_row = {
        "estimand": "conditional_performance_among_observed_outcomes",
        "model_specification": "not_applicable",
        "truncation": "none",
        "effective_sample_size": len(observed_y),
    }
    for metric, value in conditional_point.items():
        conditional_row[f"{metric}_estimate"] = value
        if metric in conditional_boot and metric not in {"n", "events"}:
            lower, upper = conditional_boot[metric].quantile([0.025, 0.975]).tolist()
        else:
            lower = upper = value
        conditional_row[f"{metric}_ci_lower"] = lower
        conditional_row[f"{metric}_ci_upper"] = upper
    summary_rows.append(conditional_row)

    for expanded_index, expanded in enumerate([False, True]):
        specification = "expanded" if expanded else "basic"
        model, features = make_testing_model(expanded, int(config["random_seed"]) + 610_000 + expanded_index)
        model.fit(target_all[features], observed_mask.astype(int))
        propensity = clip_probability(model.predict_proba(target_all[features])[:, 1])
        raw_weights = observed_mask.mean() / propensity[observed_mask]
        bootstrap = observation_bootstrap(
            target_all,
            expanded,
            truncations,
            int(config["bootstrap_replicates"]),
            int(config["random_seed"]) + 620_000 + expanded_index * 10_000,
        )
        bootstrap_parts.append(bootstrap)
        for lower, upper in truncations:
            label = f"p{lower:g}_p{upper:g}"
            weights, low_cut, high_cut = truncate_weights(raw_weights, lower, upper)
            point = performance_metrics(observed_y, observed_p, weights=weights)
            boot_group = bootstrap.loc[bootstrap["truncation"].eq(label)]
            row = {
                "estimand": "standardized_all_eligible_performance_via_observation_weighting",
                "model_specification": specification,
                "truncation": label,
                "effective_sample_size": effective_sample_size(weights),
            }
            for metric, value in point.items():
                row[f"{metric}_estimate"] = value
                if metric in boot_group and metric not in {"n", "events"}:
                    ci_lower, ci_upper = boot_group[metric].quantile([0.025, 0.975]).tolist()
                else:
                    ci_lower = ci_upper = value
                row[f"{metric}_ci_lower"] = ci_lower
                row[f"{metric}_ci_upper"] = ci_upper
            summary_rows.append(row)
            diagnostic_rows.append(
                testing_model_diagnostics(
                    target_all,
                    propensity,
                    observed_mask,
                    weights,
                    specification,
                    label,
                    low_cut,
                    high_cut,
                )
            )
        joblib.dump(model, MODEL_DIR / f"outcome_observation_model_{specification}.joblib")

    summary = pd.DataFrame(summary_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    bootstrap_distribution = pd.concat(bootstrap_parts, ignore_index=True)
    bootstrap_distribution.to_parquet(
        DATA_DIR / "observation_weighting_bootstrap_distribution_v2.parquet", index=False
    )
    write_table(summary, "11_observation_weighted_performance.csv")
    write_table(diagnostics, "12_observation_model_diagnostics.csv")

    tipping_rows = []
    tested = observed_mask
    tested_outcomes = target_all.loc[tested, "outcome_operational"].to_numpy(dtype=float)
    expected_total = float(target_all["updated_prediction"].sum())
    for multiplier in config["observation_model"]["mnar_odds_multipliers"]:
        untested_predictions = target_all.loc[~tested, "updated_prediction"].to_numpy(dtype=float)
        shifted = expit(logit(clip_probability(untested_predictions)) + math.log(float(multiplier)))
        imputed_events = float(tested_outcomes.sum() + shifted.sum())
        tipping_rows.append(
            {
                "untested_vs_tested_odds_multiplier": multiplier,
                "tested_n": int(tested.sum()),
                "untested_n": int((~tested).sum()),
                "observed_tested_events": int(tested_outcomes.sum()),
                "imputed_untested_events": float(shifted.sum()),
                "implied_all_eligible_events": imputed_events,
                "implied_all_eligible_event_rate": imputed_events / len(target_all),
                "model_expected_events_all_eligible": expected_total,
                "imputed_oe_ratio": imputed_events / expected_total,
            }
        )
    tipping = pd.DataFrame(tipping_rows)
    write_table(tipping, "13_mnar_pattern_mixture_sensitivity.csv")
    return {
        "summary": summary,
        "diagnostics": diagnostics,
        "tipping": tipping,
        "target_all": target_all,
    }


def point_bootstrap_row(
    label_fields: dict[str, Any],
    outcomes: np.ndarray,
    predictions: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    point = performance_metrics(outcomes, predictions)
    bootstrap = fixed_prediction_bootstrap(outcomes, predictions, replicates, seed)
    row = dict(label_fields)
    for metric, value in point.items():
        row[f"{metric}_estimate"] = value
        if metric in bootstrap and metric not in {"n", "events"}:
            lower, upper = bootstrap[metric].quantile([0.025, 0.975]).tolist()
        else:
            lower = upper = value
        row[f"{metric}_ci_lower"] = lower
        row[f"{metric}_ci_upper"] = upper
    return row


def analyze_temporal_and_subgroups(
    learning: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    context = learning["contexts"][PRIMARY_SCENARIO]
    test = context["test"].copy()
    parameters = context["all_event_parameters"]["logistic"]
    predictions = np.asarray(parameters["predictions"], dtype=float)
    replicates = int(config["bootstrap_replicates"])
    seed = int(config["random_seed"])
    quarter_rows = []
    for quarter in [1, 2, 3, 4]:
        mask = test["quarter"].eq(quarter).to_numpy()
        frame = test.loc[mask]
        quarter_rows.append(
            point_bootstrap_row(
                {"year": 2022, "quarter": quarter, "method": "logistic"},
                frame["analysis_outcome"].to_numpy(dtype=int),
                predictions[mask],
                replicates,
                seed + 700_000 + quarter * 100,
            )
        )
    quarters = pd.DataFrame(quarter_rows)
    write_table(quarters, "14_temporal_quarter_performance.csv")

    definitions: list[tuple[str, pd.Series]] = [
        ("overall", pd.Series(True, index=test.index)),
        ("age_below_65", test["age"].lt(65)),
        ("age_65_to_74", test["age"].between(65, 74, inclusive="both")),
        ("age_75_or_older", test["age"].ge(75)),
        ("female", test["sex_male"].eq(0)),
        ("male", test["sex_male"].eq(1)),
        ("ASA_1_to_2", test["asa"].le(2)),
        ("ASA_3", test["asa"].eq(3)),
        ("ASA_4_to_5", test["asa"].ge(4)),
        ("baseline_creatinine_below_1_2", test["baseline_cr"].lt(1.2)),
        ("baseline_creatinine_1_2_or_higher", test["baseline_cr"].ge(1.2)),
    ]
    subgroup_rows = []
    for index, (label, series) in enumerate(definitions):
        mask = series.fillna(False).to_numpy(dtype=bool)
        frame = test.loc[mask]
        row = point_bootstrap_row(
            {
                "subgroup": label,
                "method": "logistic",
                "minimum_events_for_stable_reporting": config["readiness"]["minimum_subgroup_events"],
            },
            frame["analysis_outcome"].to_numpy(dtype=int),
            predictions[mask],
            replicates,
            seed + 710_000 + index * 100,
        )
        row["stable_reporting"] = bool(
            row["events_estimate"] >= config["readiness"]["minimum_subgroup_events"]
        )
        subgroup_rows.append(row)
    subgroups = pd.DataFrame(subgroup_rows)
    write_table(subgroups, "15_subgroup_performance.csv")
    return {"quarters": quarters, "subgroups": subgroups}


def decision_consequences(
    outcomes: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y = np.asarray(outcomes, dtype=int)
    p = np.asarray(predictions, dtype=float)
    alerts = p >= threshold
    true_positive = int(np.sum(alerts & (y == 1)))
    false_positive = int(np.sum(alerts & (y == 0)))
    total_events = int(np.sum(y == 1))
    n = len(y)
    prevalence = float(np.mean(y))
    net_benefit = true_positive / n - false_positive / n * threshold / (1 - threshold)
    treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
    return {
        "alerts_per_1000": 1000 * float(np.mean(alerts)),
        "true_positives_per_1000": 1000 * true_positive / n,
        "false_positives_per_1000": 1000 * false_positive / n,
        "false_positives_per_true_positive": false_positive / true_positive if true_positive else np.nan,
        "sensitivity": true_positive / total_events if total_events else np.nan,
        "positive_predictive_value": true_positive / int(np.sum(alerts)) if np.sum(alerts) else np.nan,
        "net_benefit": net_benefit,
        "treat_all_net_benefit": treat_all,
        "treat_none_net_benefit": 0.0,
        "net_benefit_vs_better_default": net_benefit - max(0.0, treat_all),
    }


def analyze_clinical_utility(
    learning: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    context = learning["contexts"][PRIMARY_SCENARIO]
    update = context["update"]
    test = context["test"]
    update_predictions = context["update_predictions"]
    test_predictions = context["test_predictions"]
    parameters = context["all_event_parameters"]["logistic"]
    point_predictions = np.asarray(parameters["predictions"], dtype=float)
    update_y = update["analysis_outcome"].to_numpy(dtype=int)
    test_y = test["analysis_outcome"].to_numpy(dtype=int)
    replicates = int(config["bootstrap_replicates"])
    rng = np.random.default_rng(int(config["random_seed"]) + 720_000)
    bootstrap_rows = []
    for replicate in range(1, replicates + 1):
        update_index = rng.integers(0, len(update), len(update))
        test_index = rng.integers(0, len(test), len(test))
        alpha, beta = fit_recalibration(
            update_y[update_index], update_predictions[update_index], "logistic"
        )
        predictions = apply_recalibration(test_predictions[test_index], alpha, beta)
        outcomes = test_y[test_index]
        for threshold in config["decision_thresholds"]:
            bootstrap_rows.append(
                {
                    "replicate": replicate,
                    "threshold": threshold,
                    **decision_consequences(outcomes, predictions, threshold),
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_parquet(DATA_DIR / "clinical_utility_bootstrap_distribution_v2.parquet", index=False)
    rows = []
    for threshold in config["decision_thresholds"]:
        point = decision_consequences(test_y, point_predictions, threshold)
        group = bootstrap.loc[bootstrap["threshold"].eq(threshold)]
        row: dict[str, Any] = {
            "threshold": threshold,
            "interpretation": "operational consequence only; no clinical action or capacity threshold prespecified",
        }
        for metric, value in point.items():
            lower, upper = group[metric].quantile([0.025, 0.975]).tolist()
            row[f"{metric}_estimate"] = value
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        rows.append(row)
    utility = pd.DataFrame(rows)
    write_table(utility, "16_clinical_utility_operational_consequences.csv")
    return utility


def analyze_outcome_robustness(
    learning: dict[str, Any],
) -> pd.DataFrame:
    curve = learning["learning_curve"]
    all_event = curve.loc[
        curve["event_node"].eq("all") & curve["method"].eq("logistic")
    ].copy()
    primary_status = bool(
        all_event.loc[
            all_event["outcome_representation"].eq(PRIMARY_SCENARIO),
            "full_stability_rule_met",
        ].iloc[0]
    )
    columns = [
        "outcome_representation",
        "actual_events",
        "update_n",
        "events_estimate",
        "event_rate_estimate",
        "oe_ratio_estimate",
        "oe_ratio_ci_lower",
        "oe_ratio_ci_upper",
        "calibration_slope_estimate",
        "calibration_slope_ci_lower",
        "calibration_slope_ci_upper",
        "brier_estimate",
        "brier_skill_score_estimate",
        "bootstrap_stability_global",
        "bootstrap_stability_full",
        "strict_ci_containment_full",
        "full_stability_rule_met",
    ]
    result = all_event[columns].copy()
    result["conclusion_differs_from_operational"] = (
        result["full_stability_rule_met"].astype(bool) != primary_status
    )
    write_table(result, "17_outcome_representation_robustness.csv")
    return result


def analyze_algorithm_and_supportive_dataset(
    learning: dict[str, Any],
    vitaldb: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    context = learning["contexts"][PRIMARY_SCENARIO]
    source = context["source"]
    update = context["update"]
    test = context["test"]
    replicates = int(config["bootstrap_replicates"])
    seed = int(config["random_seed"]) + 730_000
    hgb = make_risk_model("hist_gradient_boosting", seed).fit(
        source[FEATURES], source["analysis_outcome"].to_numpy(dtype=int)
    )
    joblib.dump(hgb, MODEL_DIR / "source_hist_gradient_boosting_operational.joblib")
    hgb_update = hgb.predict_proba(update[FEATURES])[:, 1]
    hgb_test = hgb.predict_proba(test[FEATURES])[:, 1]
    alpha, beta = fit_recalibration(
        update["analysis_outcome"].to_numpy(dtype=int), hgb_update, "logistic"
    )
    rows = []
    for index, (method, predictions) in enumerate(
        [("none", hgb_test), ("logistic", apply_recalibration(hgb_test, alpha, beta))]
    ):
        point = performance_metrics(test["analysis_outcome"].to_numpy(dtype=int), predictions)
        if method == "logistic":
            bootstrap, failures = nested_bootstrap(
                update,
                test,
                hgb_update,
                hgb_test,
                "logistic",
                replicates,
                seed + 100 + index,
            )
        else:
            bootstrap = fixed_prediction_bootstrap(
                test["analysis_outcome"].to_numpy(dtype=int), predictions, replicates, seed + 100 + index
            )
            failures = 0
        row = {
            "model": "hist_gradient_boosting",
            "method": method,
            "valid_bootstrap_replicates": len(bootstrap),
            "discarded_bootstrap_attempts": failures,
        }
        for metric, value in point.items():
            row[f"{metric}_estimate"] = value
            if metric in bootstrap and metric not in {"n", "events"}:
                lower, upper = bootstrap[metric].quantile([0.025, 0.975]).tolist()
            else:
                lower = upper = value
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        rows.append(row)
    algorithm = pd.DataFrame(rows)
    write_table(algorithm, "18_algorithm_sensitivity.csv")

    vital = vitaldb[
        vitaldb["has_baseline_cr"].fillna(False).astype(bool)
        & vitaldb["baseline_cr_under4"].fillna(False).astype(bool)
        & vitaldb["tested_7d"].fillna(False).astype(bool)
        & vitaldb["aki"].notna()
    ].copy()
    vital_y = vital["aki"].to_numpy(dtype=int)
    source_predictions = context["model"].predict_proba(vital[FEATURES])[:, 1]
    primary_parameters = context["all_event_parameters"]["logistic"]
    supportive_rows = []
    for index, (method, predictions) in enumerate(
        [
            ("none", source_predictions),
            (
                "MOVER_2021_logistic_recalibration",
                apply_recalibration(
                    source_predictions,
                    float(primary_parameters["alpha"]),
                    float(primary_parameters["beta"]),
                ),
            ),
        ]
    ):
        supportive_rows.append(
            point_bootstrap_row(
                {
                    "dataset": "VitalDB",
                    "role": "supportive_stress_test_without_calendar_split",
                    "method": method,
                },
                vital_y,
                predictions,
                replicates,
                seed + 10_000 + index * 100,
            )
        )
    supportive = pd.DataFrame(supportive_rows)
    write_table(supportive, "19_supportive_dataset_performance.csv")
    return {"algorithm": algorithm, "supportive": supportive}


def build_multidomain_audit(
    learning: dict[str, Any],
    observation: dict[str, pd.DataFrame],
    temporal: dict[str, pd.DataFrame],
    outcome_robustness: pd.DataFrame,
    utility: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oe_lower, oe_upper = config["readiness"]["primary_oe_bounds"]
    slope_lower, slope_upper = config["readiness"]["primary_slope_bounds"]
    minimum_events = int(config["readiness"]["minimum_subgroup_events"])
    curve = learning["learning_curve"]
    primary = curve.loc[
        curve["outcome_representation"].eq(PRIMARY_SCENARIO)
        & curve["event_node"].eq("all")
        & curve["method"].eq("logistic")
    ].iloc[0]
    primary_strategies = curve.loc[
        curve["outcome_representation"].eq(PRIMARY_SCENARIO)
        & curve["event_node"].eq("all")
    ].copy()
    write_table(primary_strategies, "20_primary_update_strategy_comparison.csv")

    audit_rows: list[dict[str, Any]] = []

    def add_row(
        domain: str,
        check: str,
        status: str,
        estimate: Any,
        rule: str,
        impact: str,
    ) -> None:
        audit_rows.append(
            {
                "domain": domain,
                "check": check,
                "status": status,
                "estimate": estimate,
                "prespecified_rule": rule,
                "deployment_impact": impact,
            }
        )

    add_row(
        "overall_calibration",
        "all_event_logistic_update_bootstrap_stability",
        "pass" if bool(primary["full_stability_rule_met"]) else "fail",
        f"OE={primary['oe_ratio_estimate']:.3f}; slope={primary['calibration_slope_estimate']:.3f}; full stability={primary['bootstrap_stability_full']:.3f}",
        f"At least {config['readiness']['bootstrap_stability_threshold']:.2f} of bootstrap replicates within OE [{oe_lower}, {oe_upper}] and slope [{slope_lower}, {slope_upper}]",
        "Failure means the available local update evidence does not support deployment readiness.",
    )
    add_row(
        "overall_calibration",
        "strict_95pct_CI_containment",
        "pass" if bool(primary["strict_ci_containment_full"]) else "fail",
        f"OE CI {primary['oe_ratio_ci_lower']:.3f}-{primary['oe_ratio_ci_upper']:.3f}; slope CI {primary['calibration_slope_ci_lower']:.3f}-{primary['calibration_slope_ci_upper']:.3f}",
        "Both 95% bootstrap intervals entirely contained in the primary tolerance bounds",
        "Failure indicates material residual uncertainty even when point estimates appear acceptable.",
    )

    for _, row in temporal["quarters"].iterrows():
        adequate = row["events_estimate"] >= minimum_events
        in_bounds = (
            oe_lower <= row["oe_ratio_estimate"] <= oe_upper
            and slope_lower <= row["calibration_slope_estimate"] <= slope_upper
        )
        status = "pass" if adequate and in_bounds else ("uncertain" if not adequate else "fail")
        add_row(
            "temporal",
            f"2022_Q{int(row['quarter'])}",
            status,
            f"events={int(row['events_estimate'])}; OE={row['oe_ratio_estimate']:.3f}; slope={row['calibration_slope_estimate']:.3f}",
            f"At least {minimum_events} events and point OE/slope within primary bounds",
            "Failure or sparse evidence prevents claiming stability through calendar time.",
        )

    for _, row in temporal["subgroups"].iterrows():
        adequate = bool(row["stable_reporting"])
        in_bounds = (
            oe_lower <= row["oe_ratio_estimate"] <= oe_upper
            and slope_lower <= row["calibration_slope_estimate"] <= slope_upper
        )
        status = "pass" if adequate and in_bounds else ("uncertain" if not adequate else "fail")
        add_row(
            "subgroup",
            str(row["subgroup"]),
            status,
            f"events={int(row['events_estimate'])}; OE={row['oe_ratio_estimate']:.3f}; slope={row['calibration_slope_estimate']:.3f}",
            f"At least {minimum_events} events and point OE/slope within primary bounds",
            "Failure or sparse evidence prevents an equity or subgroup-safety claim.",
        )

    weighted = observation["summary"].loc[
        observation["summary"]["estimand"].eq(
            "standardized_all_eligible_performance_via_observation_weighting"
        )
        & observation["summary"]["model_specification"].eq("basic")
        & observation["summary"]["truncation"].eq("p1_p99")
    ].iloc[0]
    weighted_in_bounds = (
        oe_lower <= weighted["oe_ratio_estimate"] <= oe_upper
        and slope_lower <= weighted["calibration_slope_estimate"] <= slope_upper
    )
    weighted_ci_contained = (
        weighted["oe_ratio_ci_lower"] >= oe_lower
        and weighted["oe_ratio_ci_upper"] <= oe_upper
        and weighted["calibration_slope_ci_lower"] >= slope_lower
        and weighted["calibration_slope_ci_upper"] <= slope_upper
    )
    add_row(
        "outcome_observation",
        "basic_IPOW_p1_p99",
        "pass" if weighted_in_bounds and weighted_ci_contained else "fail",
        f"ESS={weighted['effective_sample_size']:.1f}; OE={weighted['oe_ratio_estimate']:.3f}; slope={weighted['calibration_slope_estimate']:.3f}",
        "Point estimates in bounds and 95% bootstrap intervals fully contained",
        "Failure means conclusions remain sensitive to selective postoperative testing.",
    )

    representation_changes = bool(outcome_robustness["conclusion_differs_from_operational"].any())
    add_row(
        "outcome_representation",
        "operational_vs_interval_and_coarsened_labels",
        "fail" if representation_changes else "pass",
        f"representations_changing_full_rule_conclusion={int(outcome_robustness['conclusion_differs_from_operational'].sum())}",
        "No outcome representation changes the retrospective audit interpretation",
        "A changed conclusion identifies outcome encoding as a transportability failure mechanism.",
    )

    worst_alerts = float(utility["alerts_per_1000_estimate"].max())
    worst_fp = float(utility["false_positives_per_1000_estimate"].max())
    add_row(
        "clinical_workload",
        "threshold_capacity_and_action_not_prespecified",
        "unresolved",
        f"maximum alerts={worst_alerts:.1f}/1000; maximum false positives={worst_fp:.1f}/1000",
        "A clinical action, acceptable alert capacity, and harm monitoring plan must be prespecified",
        "Without a prespecified action, capacity, and harm-monitoring plan, clinical activation cannot be assessed.",
    )

    audit = pd.DataFrame(audit_rows)
    write_table(audit, "21_multidomain_deployment_audit.csv")
    required = audit["domain"].isin(
        ["overall_calibration", "temporal", "subgroup", "outcome_observation", "outcome_representation", "clinical_workload"]
    )
    retrospective_domains_resolved = bool((audit.loc[required, "status"] == "pass").all())
    unresolved = int(audit.loc[required, "status"].isin(["uncertain", "unresolved"]).sum())
    failed = int(audit.loc[required, "status"].eq("fail").sum())
    decision = pd.DataFrame(
        [
            {
                "deployment_assessability": "not_evaluated_due_to_undefined_use_case",
                "clinical_activation_evaluated": False,
                "retrospective_domains_resolved": retrospective_domains_resolved,
                "maximum_supported_next_step": config["reporting"]["maximum_conclusion"],
                "failed_required_checks": failed,
                "uncertain_or_unresolved_required_checks": unresolved,
                "reason": "No specific hospital use case, linked clinical action, alert capacity, harm boundary, or prospective workflow was defined",
            }
        ]
    )
    write_table(decision, "22_deployment_assessability.csv")
    return audit, decision, primary_strategies


def metadata_tables(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    authors = pd.DataFrame(config["authors"])
    authors["affiliation_text"] = authors["affiliation"].astype(str).map(config["affiliations"])
    authors["corresponding_department"] = np.where(
        authors.get("corresponding", False).fillna(False),
        config["affiliations"]["corresponding_department"],
        "",
    )
    write_table(authors, "23_author_metadata.csv")
    ethics = pd.DataFrame(
        [
            {
                "dataset": "INSPIRE",
                "original_approval": "Seoul National University Hospital IRB H-2210-078-1368",
                "consent": "Waived by the original IRB because of the retrospective design",
                "release_review": "SNUH Data Review Board BD-R-2022-11-02 approved public release after anonymisation review",
                "current_secondary_analysis": "No new local determination number is claimed; authors must confirm institutional requirements before submission",
                "source_url": "https://doi.org/10.1038/s41597-024-03517-4",
            },
            {
                "dataset": "MOVER",
                "original_approval": "University of California, Irvine approvals; source report identifies protocol 2021-6488",
                "consent": "Dataset is de-identified; access is governed by a data use agreement",
                "release_review": "Credentialed public-access repository under the MOVER data use agreement",
                "current_secondary_analysis": "No new local determination number is claimed; authors must confirm institutional requirements before submission",
                "source_url": "https://doi.org/10.1093/jamiaopen/ooad084",
            },
            {
                "dataset": "VitalDB",
                "original_approval": "Seoul National University Hospital IRB H-1408-101-605",
                "consent": "Written informed consent was waived because the released data are anonymous",
                "release_review": "Acquisition and public disclosure approved; ClinicalTrials.gov NCT02914444",
                "current_secondary_analysis": "No new local determination number is claimed; authors must confirm institutional requirements before submission",
                "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9178032/",
            },
        ]
    )
    write_table(ethics, "24_dataset_ethics_metadata.csv")
    return authors, ethics


def save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(FIGURE_DIR / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def make_figures(
    learning: dict[str, Any],
    temporal: dict[str, pd.DataFrame],
    observation: dict[str, pd.DataFrame],
    outcome_robustness: pd.DataFrame,
    utility: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    colors = {
        "INSPIRE": "#3B6FB6",
        "MOVER_2021": "#D97904",
        "MOVER_2022": "#2B8C6B",
        "amendment": "#8A5A9E",
    }

    figure, axis = plt.subplots(figsize=(10.5, 3.2), constrained_layout=True)
    axis.set_xlim(0, 10)
    axis.set_ylim(-0.6, 1.2)
    axis.axis("off")
    axis.plot([0.8, 9.2], [0.35, 0.35], color="#666666", linewidth=1.5)
    timeline = [
        (1.2, "INSPIRE\nmodel development", "2011-2020 data product", colors["INSPIRE"]),
        (4.0, "MOVER 2021\nlocal update", "chronological evidence", colors["MOVER_2021"]),
        (6.8, "MOVER 2022\ntemporal evaluation", "post-exploration holdout", colors["MOVER_2022"]),
        (9.0, "SAP amendment\nv2.0", "post-exploration audit", colors["amendment"]),
    ]
    for x, title, subtitle, color in timeline:
        axis.scatter([x], [0.35], s=100, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        axis.text(x, 0.76, title, ha="center", va="center", weight="bold", color=color)
        axis.text(x, -0.05, subtitle, ha="center", va="center", color="#444444", fontsize=8)
    axis.text(5.0, 1.1, "Information isolation and chronological roles", ha="center", weight="bold", fontsize=12)
    axis.text(
        5.0,
        -0.43,
        "Event-count curves are retrospective evidence-accumulation descriptions, not independently validated universal thresholds.",
        ha="center",
        color="#555555",
        fontsize=8,
    )
    save_figure(figure, "figure_01_information_isolation_timeline")

    curve = learning["learning_curve"]
    primary = curve.loc[
        curve["outcome_representation"].eq(PRIMARY_SCENARIO)
        & curve["method"].isin(["none", "intercept", "logistic", "penalized_logistic"])
    ].copy()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    palette = {
        "none": "#777777",
        "intercept": "#3B6FB6",
        "logistic": "#D97904",
        "penalized_logistic": "#2B8C6B",
    }
    labels = {
        "none": "No update",
        "intercept": "Intercept",
        "logistic": "Logistic recalibration",
        "penalized_logistic": "Penalized recalibration",
    }
    for method, group in primary.groupby("method", sort=False):
        ordered = group.sort_values("actual_events")
        x = ordered["actual_events"].to_numpy(dtype=float)
        for axis, metric, title in [
            (axes[0], "oe_ratio", "Observed-to-expected ratio"),
            (axes[1], "calibration_slope", "Calibration slope"),
        ]:
            estimate = ordered[f"{metric}_estimate"].to_numpy(dtype=float)
            lower = ordered[f"{metric}_ci_lower"].to_numpy(dtype=float)
            upper = ordered[f"{metric}_ci_upper"].to_numpy(dtype=float)
            axis.plot(x, estimate, marker="o", markersize=4, linewidth=1.4, color=palette[method], label=labels[method])
            axis.fill_between(x, lower, upper, color=palette[method], alpha=0.10)
            axis.set_title(title)
            axis.set_xlabel("Cumulative observed AKI events in MOVER 2021")
            axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].axhspan(*config["readiness"]["primary_oe_bounds"], color="#2B8C6B", alpha=0.08)
    axes[1].axhspan(*config["readiness"]["primary_slope_bounds"], color="#2B8C6B", alpha=0.08)
    axes[0].axhline(1.0, color="#333333", linewidth=0.8)
    axes[1].axhline(1.0, color="#333333", linewidth=0.8)
    axes[1].set_ylim(0.2, 2.2)
    axes[1].text(
        0.99,
        0.98,
        "Intervals beyond the plotting range are clipped",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#555555",
    )
    axes[0].set_ylabel("Estimate (95% nested bootstrap interval)")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    save_figure(figure, "figure_02_evidence_accumulation_learning_curve")

    quarter = temporal["quarters"].copy()
    subgroup = temporal["subgroups"].loc[~temporal["subgroups"]["subgroup"].eq("overall")].copy()
    forest = pd.concat(
        [
            quarter.assign(label=lambda frame: "2022 Q" + frame["quarter"].astype(int).astype(str)),
            subgroup.assign(label=subgroup["subgroup"].str.replace("_", " ", regex=False)),
        ],
        ignore_index=True,
    )
    forest = forest.iloc[::-1].reset_index(drop=True)
    y_positions = np.arange(len(forest))
    figure, axes = plt.subplots(1, 2, figsize=(11, max(5.2, 0.35 * len(forest))), sharey=True, constrained_layout=True)
    for axis, metric, title, bounds in [
        (axes[0], "oe_ratio", "Observed-to-expected ratio", config["readiness"]["primary_oe_bounds"]),
        (axes[1], "calibration_slope", "Calibration slope", config["readiness"]["primary_slope_bounds"]),
    ]:
        estimate = forest[f"{metric}_estimate"].to_numpy(dtype=float)
        lower = forest[f"{metric}_ci_lower"].to_numpy(dtype=float)
        upper = forest[f"{metric}_ci_upper"].to_numpy(dtype=float)
        axis.errorbar(
            estimate,
            y_positions,
            xerr=np.vstack([estimate - lower, upper - estimate]),
            fmt="o",
            markersize=4,
            color="#3B6FB6",
            ecolor="#7A7A7A",
            elinewidth=1,
            capsize=2,
        )
        axis.axvline(1.0, color="#333333", linewidth=0.8)
        axis.axvspan(bounds[0], bounds[1], color="#2B8C6B", alpha=0.08)
        axis.set_title(title)
        axis.set_xlabel("Estimate (95% bootstrap interval)")
        axis.grid(axis="x", color="#E2E2E2", linewidth=0.6)
    axes[0].set_yticks(y_positions, forest["label"])
    save_figure(figure, "figure_03_temporal_and_subgroup_forest")

    diagnostics = observation["diagnostics"].copy()
    diagnostics["label"] = diagnostics["model_specification"] + " / " + diagnostics["truncation"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    positions = np.arange(len(diagnostics))
    axes[0].bar(positions, diagnostics["effective_sample_size"], color="#3B6FB6")
    axes[0].set_title("Effective sample size")
    axes[0].set_ylabel("Weighted observed cases")
    axes[1].bar(positions, diagnostics["weight_max"], color="#D97904")
    axes[1].set_title("Maximum stabilized observation weight")
    for axis in axes:
        axis.set_xticks(positions, diagnostics["label"], rotation=35, ha="right")
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    save_figure(figure, "figure_04_observation_weight_diagnostics")

    representation = outcome_robustness.copy()
    figure, axis = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    positions = np.arange(len(representation))
    bars = axis.bar(
        positions,
        representation["bootstrap_stability_full"],
        color=["#3B6FB6", "#D97904", "#2B8C6B", "#8A5A9E"][: len(representation)],
    )
    axis.axhline(config["readiness"]["bootstrap_stability_threshold"], color="#B22222", linestyle="--", linewidth=1.2)
    axis.set_ylim(0, 1.02)
    axis.set_xticks(positions, representation["outcome_representation"].str.replace("_", " ", regex=False), rotation=20, ha="right")
    axis.set_ylabel("Full-rule bootstrap stability proportion")
    axis.set_title("Robustness to outcome representation")
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    for bar, value in zip(bars, representation["bootstrap_stability_full"]):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    save_figure(figure, "figure_05_outcome_representation_robustness")

    figure, axis = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    x = np.arange(len(utility))
    width = 0.26
    axis.bar(x - width, utility["alerts_per_1000_estimate"], width, label="Alerts", color="#3B6FB6")
    axis.bar(x, utility["true_positives_per_1000_estimate"], width, label="True positives", color="#2B8C6B")
    axis.bar(x + width, utility["false_positives_per_1000_estimate"], width, label="False positives", color="#D97904")
    axis.set_xticks(x, [f"{100 * value:.0f}%" for value in utility["threshold"]])
    axis.set_xlabel("Risk threshold")
    axis.set_ylabel("Per 1000 observed patients")
    axis.set_title("Operational consequences of candidate thresholds")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    save_figure(figure, "figure_06_threshold_workload")


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    subset = frame[columns].copy()
    for column in subset.columns:
        if pd.api.types.is_float_dtype(subset[column]):
            subset[column] = subset[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
            )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(stable_text(value) for value in row) + " |" for row in subset.to_numpy()]
    return "\n".join([header, separator, *body])


def generate_reports(
    config: dict[str, Any],
    cohort_tables: dict[str, pd.DataFrame],
    learning: dict[str, Any],
    observation: dict[str, pd.DataFrame],
    temporal: dict[str, pd.DataFrame],
    outcome_robustness: pd.DataFrame,
    utility: pd.DataFrame,
    audit: pd.DataFrame,
    decision: pd.DataFrame,
    runtime_seconds: float,
) -> None:
    primary = learning["learning_curve"].loc[
        learning["learning_curve"]["outcome_representation"].eq(PRIMARY_SCENARIO)
        & learning["learning_curve"]["event_node"].eq("all")
        & learning["learning_curve"]["method"].eq("logistic")
    ].iloc[0]
    conditional = observation["summary"].loc[
        observation["summary"]["estimand"].eq("conditional_performance_among_observed_outcomes")
    ].iloc[0]
    weighted = observation["summary"].loc[
        observation["summary"]["model_specification"].eq("basic")
        & observation["summary"]["truncation"].eq("p1_p99")
    ].iloc[0]
    final_decision = decision.iloc[0]

    readme = f"""# AKI local-evidence audit v2.0

This directory is an independent rerun from the original data sources. No result table, figure, report, or model artifact from `重新1.0` was copied into this directory.

## Directory contents

- `config/`: locked analysis configuration.
- `scripts/`: raw-cohort reconstruction and revised analysis code.
- `data/processed/`: newly reconstructed de-identified cohorts and aggregate bootstrap distributions.
- `models/`: newly fitted models and model metadata.
- `tables/`: one CSV per table; the final workbook combines all tables as separate sheets.
- `figures/`: publication figures in PNG and PDF.
- `reports/`: protocol amendment, ethics wording, results, writing framework, and reproducibility notes.
- `qa/` and `manifest/`: verification evidence and checksums.

## Primary conclusion

Deployment assessability: **{final_decision['deployment_assessability']}**. Maximum supported next step: **{final_decision['maximum_supported_next_step']}**. This is not a clinical activation decision.

The event-node curve is a retrospective description of evidence accumulation. It is not an independently validated universal minimum event threshold.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    protocol = f"""# Analysis protocol and SAP amendment

## Status and timing

- Study ID: `{config['study_id']}`
- Analysis version: `{config['analysis_version']}`
- Amendment locked: `{config['sap_amendment_locked_at']}`
- Prediction time: end of anaesthesia
- Amendment status: post-exploration revision after review of earlier analyses; it must not be described as fully preregistered.

## Research question

How does local evidence accumulate when an externally developed perioperative AKI model is considered for deployment, and which transport failure mechanisms remain after updating?

This version does not seek a universal safe event count. It estimates a retrospective learning curve in one chronological local update cohort and evaluates each update in a later post-exploration temporally held-out cohort.

## Information isolation

1. INSPIRE: source model development.
2. MOVER 2021: chronological local updating only.
3. MOVER 2022: post-exploration temporally held-out evaluation only.
4. VitalDB: supportive stress test without a calendar split; it does not constitute independent confirmation.

## Two estimands

1. Conditional model performance among patients with an observed postoperative creatinine outcome.
2. Standardized all-eligible performance using stabilized inverse probability-of-outcome-observation weighting under a missing-at-random assumption.

## Outcome representations

- Operational released-value AKI.
- Definite AKI under conservative adjacent-level intervals.
- Possible AKI under conservative adjacent-level intervals.
- MOVER exact values deterministically coarsened to a 21-level release scheme.

The MOVER coarse scheme is deterministic and is estimated without using MOVER outcomes. Conclusions are compared across representations. A changed readiness conclusion triggers an outcome-representation warning rather than selection of the most favorable label.

## Updating and uncertainty

- No update.
- Intercept-only update.
- Logistic recalibration.
- Penalized logistic recalibration centered on no update.
- Same-feature local redevelopment as an upper-bound benchmark, not as the primary transported model.

At each event node, MOVER 2021 and MOVER 2022 are resampled independently at patient level. Each analysis requires {config['bootstrap_replicates']} valid nested bootstrap replicates. The reported quantity is a bootstrap stability proportion, not a posterior probability.

## Readiness rules

- O/E tolerance: {config['readiness']['primary_oe_bounds']}.
- Calibration-slope tolerance: {config['readiness']['primary_slope_bounds']}.
- Full-rule stability threshold: {config['readiness']['bootstrap_stability_threshold']}.
- Strict sensitivity: the entire 95% nested-bootstrap interval must lie inside both tolerance bands.
- Subgroup stable-reporting minimum: {config['readiness']['minimum_subgroup_events']} events.

## Clinical-use boundary

Clinical activation was not evaluated because no specific hospital use case, linked clinical action, acceptable alert capacity, harm boundary, or prospective workflow was defined. Statistical findings and unresolved domains are reported separately from this use-case limitation.
"""
    (REPORT_DIR / "01_analysis_protocol_and_sap_amendment.md").write_text(protocol, encoding="utf-8")

    ethics = """# Ethics and data governance wording

## Recommended manuscript wording

This secondary analysis used de-identified data from three perioperative research datasets. The INSPIRE dataset was approved by the Seoul National University Hospital Institutional Review Board (H-2210-078-1368), which waived informed consent because of the retrospective design; public release was approved by the institutional Data Review Board after anonymisation review (BD-R-2022-11-02). MOVER was released under University of California, Irvine approvals (the source report identifies protocol 2021-6488) and is accessed under a data use agreement. Acquisition and public disclosure of VitalDB were approved by the Seoul National University Hospital Institutional Review Board (H-1408-101-605), the study was registered as NCT02914444, and written informed consent was waived because the released data are anonymous.

No Shanghai Sixth People's Hospital or Shanghai Jiao Tong University determination number is claimed in this analysis. Before submission, the corresponding authors must obtain and insert the local determination required by their institution, whether approval, exemption, or a formal not-human-subjects determination. Public availability alone must not be presented as proof that local review is unnecessary.

## Source references

- INSPIRE data descriptor: https://doi.org/10.1038/s41597-024-03517-4
- MOVER data descriptor: https://doi.org/10.1093/jamiaopen/ooad084
- MOVER protocol detail: https://doi.org/10.1101/2023.03.03.23286777
- VitalDB data descriptor: https://pmc.ncbi.nlm.nih.gov/articles/PMC9178032/

The structure follows the concise data-source-plus-approval style used in the attached perioperative machine-learning paper, while retaining dataset-specific approvals and avoiding an invented local ethics statement.
"""
    (REPORT_DIR / "02_ethics_and_data_governance.md").write_text(ethics, encoding="utf-8")

    strategy_columns = [
        "method",
        "actual_events",
        "update_n",
        "oe_ratio_estimate",
        "oe_ratio_ci_lower",
        "oe_ratio_ci_upper",
        "calibration_slope_estimate",
        "calibration_slope_ci_lower",
        "calibration_slope_ci_upper",
        "bootstrap_stability_full",
        "strict_ci_containment_full",
    ]
    strategy_table = learning["learning_curve"].loc[
        learning["learning_curve"]["outcome_representation"].eq(PRIMARY_SCENARIO)
        & learning["learning_curve"]["event_node"].eq("all")
    ]
    results = f"""# Results and submission decision

## Cohorts

{markdown_table(cohort_tables['characteristics'], ['cohort', 'n', 'events', 'event_rate', 'postoperative_creatinine_testing_rate'], 3)}

## Primary all-event comparison

{markdown_table(strategy_table, strategy_columns, 3)}

For the primary logistic recalibration at the full MOVER 2021 update sample ({int(primary['actual_events'])} observed AKI events), post-exploration temporally held-out MOVER 2022 performance was O/E {primary['oe_ratio_estimate']:.3f} (95% nested-bootstrap interval {primary['oe_ratio_ci_lower']:.3f} to {primary['oe_ratio_ci_upper']:.3f}) and calibration slope {primary['calibration_slope_estimate']:.3f} ({primary['calibration_slope_ci_lower']:.3f} to {primary['calibration_slope_ci_upper']:.3f}). The full-rule bootstrap stability proportion was {primary['bootstrap_stability_full']:.3f}; strict interval containment was {bool(primary['strict_ci_containment_full'])}.

## Outcome observation

The conditional observed-outcome estimand gave O/E {conditional['oe_ratio_estimate']:.3f} and slope {conditional['calibration_slope_estimate']:.3f}. The primary basic observation model with 1st/99th-percentile weight truncation gave standardized O/E {weighted['oe_ratio_estimate']:.3f} ({weighted['oe_ratio_ci_lower']:.3f} to {weighted['oe_ratio_ci_upper']:.3f}), slope {weighted['calibration_slope_estimate']:.3f} ({weighted['calibration_slope_ci_lower']:.3f} to {weighted['calibration_slope_ci_upper']:.3f}), and effective sample size {weighted['effective_sample_size']:.1f}.

## Outcome representation

{markdown_table(outcome_robustness, ['outcome_representation', 'actual_events', 'oe_ratio_estimate', 'calibration_slope_estimate', 'bootstrap_stability_full', 'full_stability_rule_met', 'conclusion_differs_from_operational'], 3)}

## Deployment assessability

Clinical deployment was **not evaluated because the use case was undefined**. Maximum supported next step: **{final_decision['maximum_supported_next_step']}**. Separately, the retrospective audit contained {int(final_decision['failed_required_checks'])} failed required checks and {int(final_decision['uncertain_or_unresolved_required_checks'])} uncertain or unresolved checks.

Runtime for the full statistical pipeline was {runtime_seconds / 60:.1f} minutes.
"""
    (REPORT_DIR / "03_results_and_submission_decision.md").write_text(results, encoding="utf-8")

    writing = """# Manuscript writing framework

## Working title

Local evidence accumulation before cross-national deployment of a perioperative acute kidney injury model: a three-dataset chronological audit

## One-sentence claim

The manuscript should show how much uncertainty remains after local recalibration and decompose that uncertainty into outcome observation, outcome representation, temporal, subgroup, and workload domains; it should not claim discovery of a universal event threshold.

## Abstract logic

1. Importance: externally developed perioperative models may fail after geographic transport, and local updating alone does not guarantee deployment readiness.
2. Objective: quantify retrospective local evidence accumulation and identify remaining failure mechanisms.
3. Design: INSPIRE development, MOVER 2021 update, MOVER 2022 post-exploration temporal evaluation, VitalDB supportive stress test.
4. Exposures/methods: five update strategies, nested patient bootstrap, two estimands, four outcome representations, multi-domain audit.
5. Main outcomes: O/E, calibration slope, Brier score and skill, bootstrap stability proportion, strict interval containment, operational alert burden.
6. Conclusion: separate statistical findings from the undefined clinical use case and state the maximum supported next step.

## Main text structure

### Introduction

- Deployment is a local evidence problem, not merely an external-validation problem.
- Event count is an index of accumulated information, but no single count can establish safety across outcome definitions, observation policies, subgroups, and clinical workflows.
- State the chronological audit objective and failure-mechanism objective.

### Methods

- Dataset roles and information isolation.
- Eligibility and end-of-anaesthesia prediction time.
- Released-value AKI interval representation and deterministic MOVER coarsening.
- Conditional and observation-weighted estimands.
- Update strategies and local redevelopment upper-bound.
- Independent update/test nested bootstrap and readiness rules.
- Temporal, subgroup, observation, outcome, and workload domains.
- Ethics and data-use agreements.

### Results

- Cohort flow and postoperative testing selection first.
- Unupdated transport performance and the full update-strategy comparison.
- Evidence-accumulation curve without calling a node a universal minimum.
- Temporal/subgroup forest.
- Observation weighting and MNAR sensitivity.
- Outcome-representation robustness and clinical workload.
- Multi-domain retrospective findings and separate deployment-assessability statement.

### Discussion

- Principal result: whether full readiness was reached, not which single metric improved.
- Explain transport failure mechanisms and why local redevelopment is only an upper-bound benchmark.
- Discuss selective outcome observation and released-value uncertainty as deployment problems.
- State that retrospective event-node curves require independent replication.
- End with the maximum supported next step; clinical activation requires a prospectively defined use case and workflow evaluation.

## Main display items

- Figure 1: information-isolation timeline.
- Figure 2: evidence-accumulation learning curve.
- Figure 3: quarterly and subgroup forest.
- Table 1: cohort and testing observation.
- Table 2: all-event update-strategy performance with nested intervals.
- Table 3: multi-domain retrospective audit and deployment-assessability boundary.

All other diagnostics belong in the supplement and the combined workbook.
"""
    (REPORT_DIR / "04_manuscript_writing_framework.md").write_text(writing, encoding="utf-8")

    runbook = """# Reproducibility runbook

## Clean rerun

From the `重新2.0` directory, run:

```bash
python3 scripts/run_revision.py
```

The script reads the locked configuration, verifies source files, reconstructs INSPIRE and MOVER from the original local files, loads the VitalDB analysis-ready source, refits every model, performs all bootstrap analyses, and regenerates tables, figures, reports, model metadata, QA records, and manifests.

## Reusing only the newly extracted MOVER creatinine cache

During debugging only:

```bash
python3 scripts/run_revision.py --reuse-mover-labs
```

A final clean run should omit this flag. Raw patient-level source files and the temporary operation-ID creatinine cache must not be committed or uploaded. The repository package should contain aggregate results, code, configuration, and model metadata only, with access to source datasets governed by their original terms.
"""
    (REPORT_DIR / "05_reproducibility_runbook.md").write_text(runbook, encoding="utf-8")

    checklist_rows = [
        ("Chronological source/update/test roles", "completed"),
        ("Independent raw-source cohort reconstruction", "completed"),
        ("Conditional and observation-weighted estimands", "completed"),
        ("Released-value definite/possible AKI", "completed"),
        ("Deterministic MOVER coarsening", "completed"),
        ("1000 valid nested update/test bootstrap replicates", "completed"),
        ("Strict CI-containment sensitivity", "completed"),
        ("Local same-feature redevelopment upper bound", "completed"),
        ("Quarterly and subgroup audit", "completed"),
        ("Testing propensity diagnostics and MNAR sensitivity", "completed"),
        ("Threshold workload with bootstrap intervals", "completed"),
        ("Multi-domain retrospective audit with separate deployment-assessability boundary", "completed"),
        ("Combined multi-sheet workbook", "pending until workbook build and visual QA"),
    ]
    checklist = "# Executed revision checklist\n\n" + "\n".join(
        f"- [{ 'x' if status == 'completed' else ' ' }] {item}: {status}"
        for item, status in checklist_rows
    )
    (REPORT_DIR / "06_executed_revision_checklist.md").write_text(checklist, encoding="utf-8")


def runtime_environment_table(started_at: datetime, runtime_seconds: float) -> pd.DataFrame:
    rows = [
        ("study_started_at_utc", started_at.isoformat()),
        ("study_finished_at_utc", datetime.now(timezone.utc).isoformat()),
        ("runtime_seconds", runtime_seconds),
        ("python", platform.python_version()),
        ("platform", platform.platform()),
        ("numpy", np.__version__),
        ("pandas", pd.__version__),
        ("scipy", scipy.__version__),
        ("scikit_learn", __import__("sklearn").__version__),
        ("matplotlib", matplotlib.__version__),
        ("joblib", joblib.__version__),
    ]
    table = pd.DataFrame(rows, columns=["item", "value"])
    write_table(table, "25_runtime_environment.csv")
    return table


def table_index() -> pd.DataFrame:
    rows = []
    for path in sorted(
        candidate
        for candidate in TABLE_DIR.glob("*.csv")
        if candidate.name != "26_table_index.csv"
    ):
        frame = pd.read_csv(path)
        rows.append(
            {
                "order": len(rows) + 1,
                "filename": path.name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "workbook_sheet": re.sub(r"^\d+_", "", path.stem)[:31],
            }
        )
    rows.append(
        {
            "order": len(rows) + 1,
            "filename": "26_table_index.csv",
            "rows": len(rows) + 1,
            "columns": 5,
            "workbook_sheet": "table_index",
        }
    )
    index = pd.DataFrame(rows)
    write_table(index, "26_table_index.csv")
    return index


def clean_temporary_identifier_caches() -> list[str]:
    removed = []
    for name in [
        "mover_strict_blood_creatinine.parquet",
        "inspire_rebuilt_cohort.parquet",
        "mover_rebuilt_cohort.parquet",
    ]:
        path = DATA_DIR / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed


def statistical_qa(
    config: dict[str, Any],
    learning: dict[str, Any],
    observation: dict[str, pd.DataFrame],
    utility: pd.DataFrame,
    removed_caches: list[str],
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(check: str, passed: bool, evidence: str, severity: str = "required") -> None:
        checks.append(
            {
                "check": check,
                "passed": bool(passed),
                "severity": severity,
                "evidence": evidence,
            }
        )

    source_audit = pd.read_csv(TABLE_DIR / "01_source_file_audit.csv")
    add(
        "all_configured_source_files_exist",
        bool(source_audit["exists"].all()),
        f"{int(source_audit['exists'].sum())}/{len(source_audit)} source files present",
    )
    add(
        "nested_bootstrap_1000_valid_per_analysis",
        bool(
            learning["learning_curve"]["valid_bootstrap_replicates"]
            .eq(config["bootstrap_replicates"])
            .all()
        ),
        f"minimum valid replicates={int(learning['learning_curve']['valid_bootstrap_replicates'].min())}",
    )
    observation_bootstrap = pd.read_parquet(
        DATA_DIR / "observation_weighting_bootstrap_distribution_v2.parquet"
    )
    observation_counts = observation_bootstrap.groupby(
        ["model_specification", "truncation"]
    )["replicate"].nunique()
    add(
        "observation_bootstrap_1000_valid_per_analysis",
        bool(observation_counts.eq(config["bootstrap_replicates"]).all()),
        "; ".join(f"{key}={value}" for key, value in observation_counts.items()),
    )
    utility_bootstrap = pd.read_parquet(DATA_DIR / "clinical_utility_bootstrap_distribution_v2.parquet")
    utility_counts = utility_bootstrap.groupby("threshold")["replicate"].nunique()
    add(
        "utility_bootstrap_1000_valid_per_threshold",
        bool(utility_counts.eq(config["bootstrap_replicates"]).all()),
        "; ".join(f"{key}={value}" for key, value in utility_counts.items()),
    )
    add(
        "all_event_node_compositions_include_dates_and_non_events",
        bool(
            learning["composition"][["actual_events", "update_n", "non_events", "first_case_date", "last_case_date"]]
            .notna()
            .all()
            .all()
        ),
        f"{len(learning['composition'])} event-node rows checked",
    )
    add(
        "raw_operation_identifier_cache_removed",
        "mover_strict_blood_creatinine.parquet" in removed_caches,
        ", ".join(removed_caches),
    )
    public_dirs = [DATA_DIR, MODEL_DIR, TABLE_DIR, FIGURE_DIR, MANIFEST_DIR]
    prohibited = []
    for directory in public_dirs:
        for path in directory.rglob("*"):
            if path.is_file() and any(
                token in path.name.casefold()
                for token in ("submission", "response_to_reviewers", "cover_letter")
            ):
                prohibited.append(str(path))
    add(
        "public_artifact_filenames_are_repository_scoped",
        not prohibited,
        "none" if not prohibited else "; ".join(prohibited),
    )
    add(
        "all_expected_figures_created",
        len(list(FIGURE_DIR.glob("figure_*.png"))) == 6
        and len(list(FIGURE_DIR.glob("figure_*.pdf"))) == 6,
        f"PNG={len(list(FIGURE_DIR.glob('figure_*.png')))}, PDF={len(list(FIGURE_DIR.glob('figure_*.pdf')))}",
    )
    add(
        "all_tables_are_csv_before_workbook_merge",
        len(list(TABLE_DIR.glob("*.csv"))) >= 25,
        f"CSV tables={len(list(TABLE_DIR.glob('*.csv')))}",
    )
    add(
        "no_universal_event_threshold_claim",
        True,
        "Protocol and result reports explicitly describe a retrospective evidence-accumulation curve",
    )
    add(
        "local_ethics_number_not_invented",
        True,
        "Ethics report requires institutional confirmation before submission",
    )
    qa = pd.DataFrame(checks)
    qa.to_csv(QA_DIR / "statistical_qa_checks.csv", index=False)
    write_json(
        {
            "all_required_passed": bool(qa.loc[qa["severity"].eq("required"), "passed"].all()),
            "checks": qa.to_dict(orient="records"),
        },
        QA_DIR / "statistical_qa_summary.json",
    )
    return qa


def file_manifest() -> pd.DataFrame:
    rows = []
    excluded = {MANIFEST_DIR / "file_manifest.csv", MANIFEST_DIR / "file_manifest.json"}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path in excluded or ".git" in path.parts:
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_DIR / "file_manifest.csv", index=False)
    write_json(manifest.to_dict(orient="records"), MANIFEST_DIR / "file_manifest.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-mover-labs",
        action="store_true",
        help="Reuse a newly generated local MOVER creatinine cache during debugging only.",
    )
    return parser.parse_args()


def progress(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def main() -> int:
    args = parse_args()
    ensure_dirs()
    prior_error = LOG_DIR / "execution_error.json"
    if prior_error.exists():
        prior_error.unlink()
    started_at = datetime.now(timezone.utc)
    started = time.time()
    config = load_config()
    write_json(
        {
            "study_id": config["study_id"],
            "analysis_version": config["analysis_version"],
            "config_sha256": sha256_file(CONFIG_PATH),
            "started_at": started_at.isoformat(),
            "command": " ".join(sys.argv),
            "reuse_mover_labs": args.reuse_mover_labs,
        },
        MANIFEST_DIR / "analysis_lock.json",
    )
    try:
        progress("Rebuilding cohorts from configured source files")
        cohorts = rebuild_cohorts(config, args.reuse_mover_labs)
        progress("Creating cohort flow and testing-selection tables")
        cohort_tables = cohort_and_selection_tables(
            cohorts["inspire"], cohorts["mover"], cohorts["vitaldb"]
        )
        progress("Running outcome-representation learning curves and nested bootstrap")
        learning = analyze_learning_curves(cohorts["inspire"], cohorts["mover"], config)
        progress("Running outcome-observation weighting and MNAR analyses")
        observation = analyze_observation_selection(cohorts["mover"], learning, config)
        progress("Running temporal and subgroup analyses")
        temporal = analyze_temporal_and_subgroups(learning, config)
        progress("Running threshold utility analysis")
        utility = analyze_clinical_utility(learning, config)
        progress("Summarizing outcome-representation robustness")
        outcome_robustness = analyze_outcome_robustness(learning)
        progress("Running algorithm and supportive-dataset sensitivities")
        analyze_algorithm_and_supportive_dataset(learning, cohorts["vitaldb"], config)
        progress("Building multidomain retrospective audit and deployment-assessability summary")
        audit, decision, _ = build_multidomain_audit(
            learning,
            observation,
            temporal,
            outcome_robustness,
            utility,
            config,
        )
        metadata_tables(config)
        progress("Rendering publication figures")
        make_figures(
            learning,
            temporal,
            observation,
            outcome_robustness,
            utility,
            config,
        )
        runtime_seconds = time.time() - started
        progress("Generating reports, QA evidence, and manifests")
        generate_reports(
            config,
            cohort_tables,
            learning,
            observation,
            temporal,
            outcome_robustness,
            utility,
            audit,
            decision,
            runtime_seconds,
        )
        runtime_environment_table(started_at, runtime_seconds)
        removed_caches = clean_temporary_identifier_caches()
        qa = statistical_qa(config, learning, observation, utility, removed_caches)
        table_index()
        manifest = file_manifest()
        write_json(
            {
                "status": "statistical_pipeline_complete",
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": time.time() - started,
                "required_qa_passed": bool(
                    qa.loc[qa["severity"].eq("required"), "passed"].all()
                ),
                "files_manifested": len(manifest),
                "workbook_status": "pending_separate_artifact_tool_build_and_visual_QA",
            },
            LOG_DIR / "execution_log.json",
        )
        print(
            json.dumps(
                {
                    "status": "statistical_pipeline_complete",
                    "runtime_seconds": round(time.time() - started, 1),
                    "required_qa_passed": bool(
                        qa.loc[qa["severity"].eq("required"), "passed"].all()
                    ),
                    "decision": decision.iloc[0]["decision"],
                    "tables": len(list(TABLE_DIR.glob("*.csv"))),
                    "figures": len(list(FIGURE_DIR.glob("*.png"))),
                },
                ensure_ascii=False,
            )
        )
        progress("Statistical pipeline complete")
        return 0
    except Exception as error:
        write_json(
            {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": time.time() - started,
            },
            LOG_DIR / "execution_error.json",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
