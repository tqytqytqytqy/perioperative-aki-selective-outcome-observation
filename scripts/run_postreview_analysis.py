#!/usr/bin/env python3
"""Run the post-review perioperative AKI transportability analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "analysis_config.json"
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
TABLE_DIR = ROOT / "tables"
FIGURE_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"
QA_DIR = ROOT / "qa"

FEATURES = ["age", "sex_male", "duration_h", "baseline_cr"]
CONTINUOUS = ["age", "duration_h", "baseline_cr"]
BINARY = ["sex_male"]
AUX_CONTINUOUS = ["age", "duration_h", "baseline_cr"]
AUX_BINARY = ["sex_male"]
AUX_CATEGORICAL = ["asa", "surgery_category"]


def ensure_dirs() -> None:
    for directory in [MODEL_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR, QA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def write_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(frame: pd.DataFrame, filename: str) -> Path:
    path = TABLE_DIR / filename
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_float_dtype(output[column]):
            output[column] = output[column].replace([np.inf, -np.inf], np.nan)
    output.to_csv(path, index=False)
    return path


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def stable_key(namespace: str, value: Any) -> str:
    # Preserve the v2 cohort-key namespace so raw timing records join the analysis cohort.
    text = f"{namespace}|AKI_LOCAL_EVIDENCE_REDO_2_0|{value}".encode("utf-8")
    return hashlib.sha256(text).hexdigest()[:20]


def clip_probability(values: Iterable[float]) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)


def eligible_base(frame: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    mask = frame["has_baseline_cr"].fillna(False).astype(bool)
    mask &= frame["baseline_cr_under4"].fillna(False).astype(bool)
    if "first_eligible" in frame:
        mask &= frame["first_eligible"].fillna(False).astype(bool)
    if year is not None:
        mask &= frame["year"].eq(year)
    return frame.loc[mask].copy().reset_index(drop=True)


def observed_outcome(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    mask = frame["tested_7d"].fillna(False).astype(bool) & frame[outcome].notna()
    output = frame.loc[mask].copy().reset_index(drop=True)
    output["analysis_outcome"] = pd.to_numeric(output[outcome], errors="coerce").astype(int)
    return output


def map_surgery_category(values: pd.Series) -> pd.Series:
    text = values.fillna("").astype(str).str.upper()
    category = pd.Series("other", index=text.index, dtype="object")
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
        category = category.mask(category.eq("other") & text.str.contains(pattern, regex=True), label)
    return category


def attach_mover_auxiliary_data(mover: pd.DataFrame, patient_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    info = pd.read_csv(
        patient_path,
        usecols=["LOG_ID", "PRIMARY_PROCEDURE_NM", "HOSP_DISCH_TIME"],
        dtype=str,
        low_memory=False,
    )
    info["case_key"] = info["LOG_ID"].map(lambda value: stable_key("MOVER_CASE", value))
    info["surgery_category"] = map_surgery_category(info["PRIMARY_PROCEDURE_NM"])
    info["discharge"] = pd.to_datetime(info["HOSP_DISCH_TIME"], errors="coerce", format="mixed")
    info = info[["case_key", "surgery_category", "discharge"]].drop_duplicates("case_key", keep="first")
    output = mover.drop(columns=["surgery_category", "discharge"], errors="ignore").merge(
        info, on="case_key", how="left", validate="one_to_one"
    )
    output["surgery_category"] = output["surgery_category"].fillna("other")
    return output, info


def derive_testing_timing(target: pd.DataFrame, labs: pd.DataFrame, info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = target.copy()
    # Released MOVER anesthesia timestamps have minute resolution; rounding removes
    # subsecond reconstruction error from the stored floating-point duration.
    frame["an_end"] = (frame["an_start"] + pd.to_timedelta(frame["duration_h"], unit="h")).dt.round("min")
    frame = frame.drop(columns=["discharge"], errors="ignore").merge(
        info[["case_key", "discharge"]], on="case_key", how="left", validate="one_to_one"
    )
    frame["window_48h_end"] = frame["an_end"] + pd.Timedelta(hours=48)
    frame["window_7d_end"] = frame["an_end"] + pd.Timedelta(days=7)
    valid_discharge = frame["discharge"].where(frame["discharge"] > frame["an_end"])
    frame["window_48h_end"] = pd.concat([frame["window_48h_end"], valid_discharge], axis=1).min(axis=1)
    frame["window_7d_end"] = pd.concat([frame["window_7d_end"], valid_discharge], axis=1).min(axis=1)

    lab = labs.copy()
    lab["case_key"] = lab["case_raw"].map(lambda value: stable_key("MOVER_CASE", value))
    joined = lab.merge(
        frame[["case_key", "an_end", "window_48h_end", "window_7d_end", "baseline_cr"]],
        on="case_key",
        how="inner",
        validate="many_to_one",
    )
    joined = joined.loc[(joined["lab_time"] > joined["an_end"]) & (joined["lab_time"] <= joined["window_7d_end"])].copy()
    joined["hours_after_anesthesia"] = (joined["lab_time"] - joined["an_end"]).dt.total_seconds() / 3600
    joined = joined.sort_values(["case_key", "lab_time", "cr"], kind="stable")
    joined["test_sequence"] = joined.groupby("case_key", sort=False).cumcount() + 1
    joined["aki_component"] = (
        ((joined["hours_after_anesthesia"] <= 48) & ((joined["cr"] - joined["baseline_cr"]) >= 0.3))
        | (joined["cr"] >= 1.5 * joined["baseline_cr"])
    )

    rows: list[dict[str, Any]] = []
    for case_key, group in joined.groupby("case_key", sort=False):
        first = group.iloc[0]
        first_two = group.iloc[:2]
        rows.append(
            {
                "case_key": case_key,
                "time_to_first_postop_creatinine_h": float(first["hours_after_anesthesia"]),
                "first_postop_creatinine_mg_dl": float(first["cr"]),
                "first_measurement_aki": int(bool(first["aki_component"])),
                "first_two_measurements_aki": int(bool(first_two["aki_component"].any())),
                "derived_postop_count_48h": int((group["hours_after_anesthesia"] <= 48).sum()),
                "derived_postop_count_7d": int(len(group)),
            }
        )
    timing = pd.DataFrame(rows)
    frame = frame.merge(timing, on="case_key", how="left", validate="one_to_one")
    for column in ["derived_postop_count_48h", "derived_postop_count_7d"]:
        frame[column] = frame[column].fillna(0).astype(int)
    audit = pd.DataFrame(
        [
            {
                "check": "postoperative_48h_count_concordance",
                "expected_total": int(frame["postop_cr_48h_count"].sum()),
                "derived_total": int(frame["derived_postop_count_48h"].sum()),
                "discordant_cases": int((frame["postop_cr_48h_count"] != frame["derived_postop_count_48h"]).sum()),
            },
            {
                "check": "postoperative_7d_count_concordance",
                "expected_total": int(frame["postop_cr_7d_count"].sum()),
                "derived_total": int(frame["derived_postop_count_7d"].sum()),
                "discordant_cases": int((frame["postop_cr_7d_count"] != frame["derived_postop_count_7d"]).sum()),
            },
        ]
    )
    return frame, audit


def make_risk_model(seed: int) -> Pipeline:
    continuous = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
        ]
    )
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    preprocess = ColumnTransformer(
        [("continuous", continuous, CONTINUOUS), ("binary", binary, BINARY)],
        remainder="drop",
    )
    classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=seed)
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def make_nuisance_model(expanded: bool, seed: int) -> tuple[Pipeline, list[str]]:
    if expanded:
        continuous = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
            ]
        )
    else:
        continuous = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    transformers: list[tuple[str, Any, list[str]]] = [
        ("continuous", continuous, AUX_CONTINUOUS),
        ("binary", binary, AUX_BINARY),
    ]
    features = AUX_CONTINUOUS + AUX_BINARY
    if expanded:
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical, AUX_CATEGORICAL))
        features += AUX_CATEGORICAL
    preprocess = ColumnTransformer(transformers, remainder="drop")
    classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=seed)
    return Pipeline([("preprocess", preprocess), ("model", classifier)]), features


def fit_recalibration(
    outcomes: Iterable[float], predictions: Iterable[float], method: str = "logistic", weights: Iterable[float] | None = None
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
            gradient = np.sum(w * (y - mu))
            information = np.sum(w * mu * (1 - mu)) + 1e-10
            step = float(np.clip(gradient / information, -5.0, 5.0))
            alpha += step
            if abs(step) < 1e-9:
                break
        return float(alpha), 1.0
    design = np.column_stack([np.ones(len(lp)), lp])
    theta = np.array([0.0, 1.0], dtype=float)
    for _ in range(100):
        mu = expit(np.clip(design @ theta, -30, 30))
        gradient = design.T @ (w * (y - mu))
        information = design.T @ (design * (w * mu * (1 - mu))[:, None]) + 1e-9 * np.eye(2)
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ gradient
        step = np.clip(step, -5.0, 5.0)
        theta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(theta[0]), float(theta[1])


def apply_recalibration(predictions: Iterable[float], alpha: float, beta: float) -> np.ndarray:
    return expit(alpha + beta * logit(clip_probability(predictions)))


def grouped_ici(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    frame = pd.DataFrame({"y": y, "p": p, "w": w}).dropna()
    if frame.empty:
        return np.nan
    groups = min(10, frame["p"].nunique(), len(frame))
    if groups < 2:
        return float(abs(np.average(frame["y"], weights=frame["w"]) - np.average(frame["p"], weights=frame["w"])))
    frame["bin"] = pd.qcut(frame["p"], q=groups, duplicates="drop")
    errors = []
    masses = []
    for _, group in frame.groupby("bin", observed=True):
        errors.append(abs(np.average(group["y"], weights=group["w"]) - np.average(group["p"], weights=group["w"])))
        masses.append(group["w"].sum())
    return float(np.average(errors, weights=masses))


def fractional_discrimination(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    labels = np.concatenate([np.ones(len(y), dtype=int), np.zeros(len(y), dtype=int)])
    scores = np.concatenate([p, p])
    weights = np.concatenate([w * y, w * (1 - y)])
    keep = weights > 0
    if labels[keep].min() == labels[keep].max():
        return np.nan, np.nan
    return (
        float(roc_auc_score(labels[keep], scores[keep], sample_weight=weights[keep])),
        float(average_precision_score(labels[keep], scores[keep], sample_weight=weights[keep])),
    )


def performance_metrics(
    outcomes: Iterable[float], predictions: Iterable[float], weights: Iterable[float] | None = None
) -> dict[str, float]:
    y = np.asarray(outcomes, dtype=float)
    p = clip_probability(predictions)
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    alpha, slope = fit_recalibration(y, p, "logistic", w)
    intercept, _ = fit_recalibration(y, p, "intercept", w)
    weighted_n = float(w.sum())
    events = float(np.sum(w * y))
    expected = float(np.sum(w * p))
    prevalence = events / weighted_n
    brier = float(np.sum(w * (y * (1 - p) ** 2 + (1 - y) * p**2)) / weighted_n)
    reference = float(np.sum(w * (y * (1 - prevalence) ** 2 + (1 - y) * prevalence**2)) / weighted_n)
    auroc, auprc = fractional_discrimination(y, p, w)
    return {
        "n": float(len(y)),
        "weighted_n": weighted_n,
        "events": events,
        "event_rate": prevalence,
        "predicted_mean": float(np.sum(w * p) / weighted_n),
        "oe_ratio": events / expected if expected > 0 else np.nan,
        "calibration_intercept": intercept,
        "calibration_joint_intercept": alpha,
        "calibration_slope": slope,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "brier_skill_score": 1 - brier / reference if reference > 0 else np.nan,
        "grouped_ici": grouped_ici(y, p, w),
    }


def summarize_bootstrap(point: dict[str, float], bootstrap: pd.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {}
    for metric, value in point.items():
        output[f"{metric}_estimate"] = value
        if metric in bootstrap and bootstrap[metric].notna().any():
            lower, upper = bootstrap[metric].quantile([0.025, 0.975]).tolist()
            output[f"{metric}_ci_lower"] = float(lower)
            output[f"{metric}_ci_upper"] = float(upper)
        else:
            output[f"{metric}_ci_lower"] = value
            output[f"{metric}_ci_upper"] = value
    return output


def category_count(values: pd.Series) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce").fillna(0),
        bins=[-1, 0, 1, 2, 3, np.inf],
        labels=["0", "1", "2", "3", "4+"],
        ordered=True,
    )


def valid_splitter(frame: pd.DataFrame, labels: np.ndarray, folds: int, seed: int):
    groups = frame["case_key"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return splitter.split(frame, labels, groups)


def crossfit_observation_model(
    frame: pd.DataFrame, observed: np.ndarray, expanded: bool, folds: int, seed: int
) -> tuple[np.ndarray, Pipeline]:
    specification = "expanded" if expanded else "basic"
    prediction = np.full(len(frame), np.nan, dtype=float)
    model, features = make_nuisance_model(expanded, seed)
    labels = observed.astype(int)
    for fold, (train, valid) in enumerate(valid_splitter(frame, labels, folds, seed), start=1):
        fitted, _ = make_nuisance_model(expanded, seed + fold)
        fitted.fit(frame.iloc[train][features], labels[train])
        prediction[valid] = fitted.predict_proba(frame.iloc[valid][features])[:, 1]
    if np.isnan(prediction).any():
        raise RuntimeError(f"Cross-fitted {specification} observation probabilities are incomplete")
    model.fit(frame[features], labels)
    return clip_probability(prediction), model


def crossfit_outcome_model(
    frame: pd.DataFrame, observed: np.ndarray, outcomes: np.ndarray, folds: int, seed: int
) -> tuple[np.ndarray, Pipeline]:
    prediction = np.full(len(frame), np.nan, dtype=float)
    combined = np.where(observed, 1 + outcomes.astype(int), 0)
    model, features = make_nuisance_model(True, seed)
    for fold, (train, valid) in enumerate(valid_splitter(frame, combined, folds, seed), start=1):
        train_observed = train[observed[train]]
        fitted, _ = make_nuisance_model(True, seed + fold)
        fitted.fit(frame.iloc[train_observed][features], outcomes[train_observed].astype(int))
        prediction[valid] = fitted.predict_proba(frame.iloc[valid][features])[:, 1]
    if np.isnan(prediction).any():
        raise RuntimeError("Cross-fitted outcome probabilities are incomplete")
    model.fit(frame.loc[observed, features], outcomes[observed].astype(int))
    return clip_probability(prediction), model


def truncated_inverse_weights(propensity: np.ndarray, observed: np.ndarray, lower: float, upper: float) -> tuple[np.ndarray, float, float]:
    raw = 1.0 / clip_probability(propensity[observed])
    low, high = np.percentile(raw, [lower, upper])
    truncated = np.clip(raw, low, high)
    truncated *= len(observed) / truncated.sum()
    return truncated, float(low), float(high)


def effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.sum(weights**2))


def calibration_diagnostics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    alpha, slope = fit_recalibration(y, p)
    intercept, _ = fit_recalibration(y, p, "intercept")
    return {
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_intercept": intercept,
        "calibration_joint_intercept": alpha,
        "calibration_slope": slope,
    }


def smd_continuous(target: pd.Series, observed_values: pd.Series, weights: np.ndarray) -> tuple[float, float]:
    target_numeric = pd.to_numeric(target, errors="coerce")
    observed_numeric = pd.to_numeric(observed_values, errors="coerce")
    target_mean = float(target_numeric.mean())
    observed_mean = float(np.average(observed_numeric, weights=weights))
    scale = float(target_numeric.std(ddof=1))
    return target_mean, (observed_mean - target_mean) / scale if scale > 0 else np.nan


def workload_metrics(outcomes: np.ndarray, predictions: np.ndarray, threshold: float, weights: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(outcomes, dtype=float)
    p = np.asarray(predictions, dtype=float)
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    alert = p >= threshold
    denominator = float(w.sum())
    tp = float(np.sum(w * alert * y))
    fp = float(np.sum(w * alert * (1 - y)))
    events = float(np.sum(w * y))
    alerts = tp + fp
    prevalence = events / denominator
    net_benefit = tp / denominator - fp / denominator * threshold / (1 - threshold)
    treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
    return {
        "alerts": alerts,
        "alerts_per_1000": 1000 * alerts / denominator,
        "true_positives_per_1000": 1000 * tp / denominator,
        "false_positives_per_1000": 1000 * fp / denominator,
        "sensitivity": tp / events if events > 0 else np.nan,
        "positive_predictive_value": tp / alerts if alerts > 0 else np.nan,
        "false_positives_per_true_positive": fp / tp if tp > 0 else np.nan,
        "net_benefit": net_benefit,
        "treat_all_net_benefit": treat_all,
        "net_benefit_vs_better_default": net_benefit - max(0.0, treat_all),
    }


def export_model_specification(model: Pipeline, alpha: float, beta: float) -> None:
    preprocess = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    feature_names = preprocess.get_feature_names_out().tolist()
    coefficients = pd.DataFrame(
        {
            "transformed_feature_order": np.arange(1, len(feature_names) + 1),
            "transformed_feature": feature_names,
            "coefficient": classifier.coef_[0],
        }
    )
    coefficients.insert(0, "model_intercept", float(classifier.intercept_[0]))
    write_csv(coefficients, "03_source_model_coefficients.csv")

    continuous = preprocess.named_transformers_["continuous"]
    imputer = continuous.named_steps["impute"]
    scaler = continuous.named_steps["scale"]
    spline = continuous.named_steps["spline"]
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(CONTINUOUS):
        rows.append(
            {
                "feature": feature,
                "released_encoding": "continuous",
                "unit": {"age": "years", "duration_h": "hours", "baseline_cr": "mg/dL"}[feature],
                "imputation": "median",
                "imputation_value": float(imputer.statistics_[index]),
                "standardization_mean": float(scaler.mean_[index]),
                "standardization_scale": float(scaler.scale_[index]),
                "spline_degree": int(spline.degree),
                "spline_n_knots": int(spline.n_knots),
                "spline_extrapolation": spline.extrapolation,
                "spline_knot_vector_standardized": json.dumps(spline.bsplines_[index].t.tolist()),
            }
        )
    binary_imputer = preprocess.named_transformers_["binary"].named_steps["impute"]
    rows.append(
        {
            "feature": "sex_male",
            "released_encoding": "1=male; 0=female",
            "unit": "binary released field",
            "imputation": "most_frequent",
            "imputation_value": float(binary_imputer.statistics_[0]),
            "standardization_mean": np.nan,
            "standardization_scale": np.nan,
            "spline_degree": np.nan,
            "spline_n_knots": np.nan,
            "spline_extrapolation": "not applicable",
            "spline_knot_vector_standardized": "",
        }
    )
    write_csv(pd.DataFrame(rows), "04_source_model_preprocessing.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "update_cohort": "MOVER 2021 observed-outcome cohort",
                    "update_method": "logistic recalibration",
                    "update_alpha": alpha,
                    "update_beta": beta,
                    "prediction_equation": "expit(alpha + beta * logit(source_probability))",
                }
            ]
        ),
        "05_final_recalibration_parameters.csv",
    )
    examples = pd.DataFrame(
        [
            {"age": 40, "sex_male": 0, "duration_h": 2.0, "baseline_cr": 0.7},
            {"age": 65, "sex_male": 1, "duration_h": 4.0, "baseline_cr": 1.0},
            {"age": 80, "sex_male": 1, "duration_h": 8.0, "baseline_cr": 1.5},
        ]
    )
    examples["source_probability"] = model.predict_proba(examples[FEATURES])[:, 1]
    examples["mover_2021_recalibrated_probability"] = apply_recalibration(
        examples["source_probability"], alpha, beta
    )
    write_csv(examples, "06_scoring_examples.csv")
    joblib.dump(model, MODEL_DIR / "source_logistic_spline_operational_v3.joblib")
    write_json(
        {
            "model_type": "intentionally parsimonious audit model",
            "clinical_use": "not for clinical use",
            "features_in_order": FEATURES,
            "source_intercept": float(classifier.intercept_[0]),
            "source_coefficients": classifier.coef_[0].tolist(),
            "update_alpha": alpha,
            "update_beta": beta,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": __import__("sklearn").__version__,
                "joblib": joblib.__version__,
            },
        },
        MODEL_DIR / "model_specification_v3.json",
    )


def cohort_tables(
    source: pd.DataFrame, update: pd.DataFrame, target_all: pd.DataFrame, vitaldb: pd.DataFrame
) -> None:
    target_observed = observed_outcome(target_all, "outcome_operational")
    vital_observed = vitaldb.loc[
        vitaldb["has_baseline_cr"].fillna(False).astype(bool)
        & vitaldb["baseline_cr_under4"].fillna(False).astype(bool)
        & vitaldb["tested_7d"].fillna(False).astype(bool)
        & vitaldb["aki"].notna()
    ].copy()
    rows = [
        {
            "cohort": "INSPIRE source development",
            "role": "source model development",
            "eligible_n": len(source),
            "observed_outcome_n": len(source),
            "observed_events": int(source["analysis_outcome"].sum()),
            "observed_event_rate": float(source["analysis_outcome"].mean()),
            "outcome_observation_rate": 1.0,
        },
        {
            "cohort": "MOVER 2021",
            "role": "chronological local update",
            "eligible_n": len(update),
            "observed_outcome_n": len(update),
            "observed_events": int(update["analysis_outcome"].sum()),
            "observed_event_rate": float(update["analysis_outcome"].mean()),
            "outcome_observation_rate": 1.0,
        },
        {
            "cohort": "MOVER 2022 all eligible",
            "role": "post-exploration temporally held-out evaluation",
            "eligible_n": len(target_all),
            "observed_outcome_n": len(target_observed),
            "observed_events": int(target_observed["analysis_outcome"].sum()),
            "observed_event_rate": float(target_observed["analysis_outcome"].mean()),
            "outcome_observation_rate": len(target_observed) / len(target_all),
        },
        {
            "cohort": "VitalDB supportive",
            "role": "supportive transport stress test",
            "eligible_n": len(vital_observed),
            "observed_outcome_n": len(vital_observed),
            "observed_events": int(vital_observed["aki"].sum()),
            "observed_event_rate": float(vital_observed["aki"].mean()),
            "outcome_observation_rate": 1.0,
        },
    ]
    write_csv(pd.DataFrame(rows), "01_cohort_flow_and_roles.csv")

    characteristic_rows: list[dict[str, Any]] = []
    groups = {
        "MOVER 2022 all eligible": target_all,
        "MOVER 2022 outcome observed": target_all.loc[target_all["tested_7d"].fillna(False)],
        "MOVER 2022 outcome unobserved": target_all.loc[~target_all["tested_7d"].fillna(False)],
    }
    for label, frame in groups.items():
        characteristic_rows.append(
            {
                "group": label,
                "n": len(frame),
                "age_mean": frame["age"].mean(),
                "age_sd": frame["age"].std(ddof=1),
                "male_percent": 100 * frame["sex_male"].mean(),
                "asa_mean": frame["asa"].mean(),
                "duration_h_mean": frame["duration_h"].mean(),
                "duration_h_sd": frame["duration_h"].std(ddof=1),
                "baseline_cr_mean": frame["baseline_cr"].mean(),
                "baseline_cr_sd": frame["baseline_cr"].std(ddof=1),
                "predicted_risk_mean": frame["updated_prediction"].mean(),
            }
        )
    write_csv(pd.DataFrame(characteristic_rows), "02_target_cohort_characteristics.csv")


def event_node_frame(update: pd.DataFrame, node: Any) -> pd.DataFrame:
    ordered = update.sort_values(["an_start", "case_key"], kind="stable").reset_index(drop=True)
    if node == "all":
        return ordered
    cumulative = ordered["analysis_outcome"].cumsum().to_numpy()
    end = int(np.flatnonzero(cumulative >= int(node))[0] + 1)
    return ordered.iloc[:end].copy()


def nested_bootstrap_performance(
    update: pd.DataFrame,
    target: pd.DataFrame,
    update_source_prediction: np.ndarray,
    target_source_prediction: np.ndarray,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    update_y = update["analysis_outcome"].to_numpy(dtype=int)
    target_y = target["analysis_outcome"].to_numpy(dtype=int)
    for _ in range(replicates):
        update_index = rng.integers(0, len(update), len(update))
        target_index = rng.integers(0, len(target), len(target))
        alpha, beta = fit_recalibration(update_y[update_index], update_source_prediction[update_index])
        prediction = apply_recalibration(target_source_prediction[target_index], alpha, beta)
        rows.append(performance_metrics(target_y[target_index], prediction))
    return pd.DataFrame(rows)


def evidence_accumulation_analysis(
    update: pd.DataFrame,
    target_observed: pd.DataFrame,
    model: Pipeline,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bands = config["descriptive_audit_bands"]
    target_source = model.predict_proba(target_observed[FEATURES])[:, 1]
    for order, node in enumerate(config["event_nodes"], start=1):
        frame = event_node_frame(update, node)
        update_source = model.predict_proba(frame[FEATURES])[:, 1]
        alpha, beta = fit_recalibration(frame["analysis_outcome"], update_source)
        prediction = apply_recalibration(target_source, alpha, beta)
        point = performance_metrics(target_observed["analysis_outcome"], prediction)
        bootstrap = nested_bootstrap_performance(
            frame,
            target_observed,
            update_source,
            target_source,
            int(config["bootstrap_replicates"]),
            int(config["random_seed"]) + order * 1000,
        )
        row: dict[str, Any] = {
            "node_order": order,
            "event_node": node,
            "actual_events": int(frame["analysis_outcome"].sum()),
            "update_n": len(frame),
            "update_alpha": alpha,
            "update_beta": beta,
            "interpretation": "post-exploration descriptive evidence-accumulation analysis",
            "oe_band_fraction": float(bootstrap["oe_ratio"].between(*bands["oe_ratio"]).mean()),
            "joint_band_fraction": float(
                (
                    bootstrap["oe_ratio"].between(*bands["oe_ratio"])
                    & bootstrap["calibration_slope"].between(*bands["calibration_slope"])
                ).mean()
            ),
        }
        row.update(summarize_bootstrap(point, bootstrap))
        rows.append(row)
    result = pd.DataFrame(rows)
    write_csv(result, "08_evidence_accumulation_observed_outcomes.csv")
    return result


def observed_performance_analysis(
    update: pd.DataFrame, target_observed: pd.DataFrame, model: Pipeline, config: dict[str, Any]
) -> tuple[pd.DataFrame, float, float, np.ndarray]:
    update_source = model.predict_proba(update[FEATURES])[:, 1]
    target_source = model.predict_proba(target_observed[FEATURES])[:, 1]
    alpha, beta = fit_recalibration(update["analysis_outcome"], update_source)
    target_updated = apply_recalibration(target_source, alpha, beta)
    rows: list[dict[str, Any]] = []
    for label, prediction, method_offset in [
        ("source model without local updating", target_source, 0),
        ("MOVER 2021 logistic recalibration", target_updated, 1),
    ]:
        point = performance_metrics(target_observed["analysis_outcome"], prediction)
        if method_offset == 0:
            rng = np.random.default_rng(int(config["random_seed"]) + 20_000)
            bootstrap_rows = []
            y = target_observed["analysis_outcome"].to_numpy(dtype=int)
            for _ in range(int(config["bootstrap_replicates"])):
                index = rng.integers(0, len(y), len(y))
                bootstrap_rows.append(performance_metrics(y[index], prediction[index]))
            bootstrap = pd.DataFrame(bootstrap_rows)
        else:
            bootstrap = nested_bootstrap_performance(
                update,
                target_observed,
                update_source,
                target_source,
                int(config["bootstrap_replicates"]),
                int(config["random_seed"]) + 21_000,
            )
        row = {
            "estimand": "performance conditional on postoperative creatinine observation",
            "model_state": label,
            "evaluation_status": "post-exploration temporally held-out evaluation; not independent confirmation",
        }
        row.update(summarize_bootstrap(point, bootstrap))
        rows.append(row)
    result = pd.DataFrame(rows)
    write_csv(result, "07_observed_outcome_performance.csv")
    return result, alpha, beta, target_updated


def temporal_and_subgroup_analysis(
    update: pd.DataFrame,
    target_observed: pd.DataFrame,
    model: Pipeline,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    update_source = model.predict_proba(update[FEATURES])[:, 1]
    target_source = model.predict_proba(target_observed[FEATURES])[:, 1]
    update_y = update["analysis_outcome"].to_numpy(dtype=int)
    replicates = int(config["bootstrap_replicates"])
    rng_update = np.random.default_rng(int(config["random_seed"]) + 30_000)
    update_parameters = []
    for _ in range(replicates):
        index = rng_update.integers(0, len(update), len(update))
        update_parameters.append(fit_recalibration(update_y[index], update_source[index]))

    def analyze(label: str, frame: pd.DataFrame, seed: int) -> dict[str, Any]:
        indices = frame.index.to_numpy()
        y = frame["analysis_outcome"].to_numpy(dtype=int)
        source_prediction = target_source[indices]
        alpha, beta = fit_recalibration(update_y, update_source)
        point = performance_metrics(y, apply_recalibration(source_prediction, alpha, beta))
        rng = np.random.default_rng(seed)
        rows = []
        for replicate in range(replicates):
            target_index = rng.integers(0, len(frame), len(frame))
            a, b = update_parameters[replicate]
            rows.append(
                performance_metrics(
                    y[target_index], apply_recalibration(source_prediction[target_index], a, b)
                )
            )
        row: dict[str, Any] = {"stratum": label, "observed_n": len(frame), "observed_events": int(y.sum())}
        row.update(summarize_bootstrap(point, pd.DataFrame(rows)))
        return row

    quarter_rows = []
    for quarter in [1, 2, 3, 4]:
        frame = target_observed.loc[target_observed["quarter"].eq(quarter)].copy()
        quarter_rows.append(analyze(f"2022 Q{quarter}", frame, int(config["random_seed"]) + 31_000 + quarter))
    quarter_table = pd.DataFrame(quarter_rows)
    write_csv(quarter_table, "09_temporal_performance_observed_outcomes.csv")

    target_observed = target_observed.copy()
    subgroup_masks = {
        "age <65": target_observed["age"] < 65,
        "age 65-74": target_observed["age"].between(65, 74, inclusive="both"),
        "age >=75": target_observed["age"] >= 75,
        "female released field": target_observed["sex_male"] == 0,
        "male released field": target_observed["sex_male"] == 1,
        "ASA 1-2": target_observed["asa"].between(1, 2, inclusive="both"),
        "ASA 3": target_observed["asa"] == 3,
        "ASA 4-5": target_observed["asa"].between(4, 5, inclusive="both"),
        "baseline creatinine <1.2": target_observed["baseline_cr"] < 1.2,
        "baseline creatinine >=1.2": target_observed["baseline_cr"] >= 1.2,
    }
    subgroup_rows = []
    for index, (label, mask) in enumerate(subgroup_masks.items(), start=1):
        frame = target_observed.loc[mask].copy()
        subgroup_rows.append(analyze(label, frame, int(config["random_seed"]) + 32_000 + index))
    subgroup_table = pd.DataFrame(subgroup_rows)
    write_csv(subgroup_table, "10_subgroup_performance_observed_outcomes.csv")
    return quarter_table, subgroup_table


def observation_process_tables(target: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    for window, column in [("48 hours", "postop_cr_48h_count"), ("7 days", "postop_cr_7d_count")]:
        category = category_count(target[column])
        counts = category.value_counts(sort=False)
        for label, count in counts.items():
            mask = category == label
            events = target.loc[mask, "outcome_operational"].dropna()
            rows.append(
                {
                    "window": window,
                    "testing_frequency": str(label),
                    "n": int(count),
                    "percent_of_all_eligible": 100 * int(count) / len(target),
                    "outcome_observed_n": int(events.notna().sum()),
                    "observed_aki_events": int(events.sum()) if len(events) else 0,
                    "observed_aki_rate": float(events.mean()) if len(events) else np.nan,
                    "interpretation": "descriptive observation-process quantity; not a causal effect of testing",
                }
            )
    write_csv(pd.DataFrame(rows), "11_testing_frequency_distribution.csv")

    target = target.copy()
    target["risk_decile"] = pd.qcut(target["updated_prediction"], q=10, labels=False, duplicates="drop") + 1
    risk_rows = []
    for decile, group in target.groupby("risk_decile", observed=True):
        risk_rows.append(
            {
                "risk_decile": int(decile),
                "n": len(group),
                "mean_predicted_risk": group["updated_prediction"].mean(),
                "postoperative_creatinine_observation_rate": group["tested_7d"].mean(),
                "mean_7d_measurement_count": group["postop_cr_7d_count"].mean(),
            }
        )
    write_csv(pd.DataFrame(risk_rows), "12_testing_by_predicted_risk_decile.csv")

    timing = target["time_to_first_postop_creatinine_h"].dropna()
    timing_table = pd.DataFrame(
        [
            {
                "eligible_n": len(target),
                "tested_n": int(target["tested_7d"].sum()),
                "time_to_first_median_h": timing.median(),
                "time_to_first_q1_h": timing.quantile(0.25),
                "time_to_first_q3_h": timing.quantile(0.75),
                "time_to_first_min_h": timing.min(),
                "time_to_first_max_h": timing.max(),
                "tested_within_48h_n": int((target["postop_cr_48h_count"] > 0).sum()),
            }
        ]
    )
    write_csv(timing_table, "13_testing_timing_summary.csv")


def weighted_balance_table(target: pd.DataFrame, observed: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    continuous = ["age", "duration_h", "baseline_cr", "asa", "updated_prediction"]
    for variable in continuous:
        target_values = pd.to_numeric(target[variable], errors="coerce")
        observed_values = pd.to_numeric(target.loc[observed, variable], errors="coerce")
        target_mean = float(target_values.mean())
        observed_mean = float(observed_values.mean())
        weighted_mean = float(np.average(observed_values, weights=weights))
        scale = float(target_values.std(ddof=1))
        rows.append(
            {
                "variable": variable,
                "level": "continuous",
                "target_mean_or_proportion": target_mean,
                "observed_unweighted_mean_or_proportion": observed_mean,
                "observed_weighted_mean_or_proportion": weighted_mean,
                "unweighted_smd_vs_target": (observed_mean - target_mean) / scale if scale > 0 else np.nan,
                "weighted_smd_vs_target": (weighted_mean - target_mean) / scale if scale > 0 else np.nan,
            }
        )
    for variable in ["sex_male", "surgery_category"]:
        levels = sorted(target[variable].dropna().unique().tolist())
        for level in levels:
            target_binary = target[variable].eq(level).astype(float)
            observed_binary = target.loc[observed, variable].eq(level).astype(float)
            target_mean = float(target_binary.mean())
            observed_mean = float(observed_binary.mean())
            weighted_mean = float(np.average(observed_binary, weights=weights))
            scale = math.sqrt(max(target_mean * (1 - target_mean), 1e-12))
            rows.append(
                {
                    "variable": variable,
                    "level": str(level),
                    "target_mean_or_proportion": target_mean,
                    "observed_unweighted_mean_or_proportion": observed_mean,
                    "observed_weighted_mean_or_proportion": weighted_mean,
                    "unweighted_smd_vs_target": (observed_mean - target_mean) / scale,
                    "weighted_smd_vs_target": (weighted_mean - target_mean) / scale,
                }
            )
    return pd.DataFrame(rows)


def target_estimators(
    target: pd.DataFrame,
    propensity_basic: np.ndarray,
    propensity_expanded: np.ndarray,
    outcome_probability: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    observed = target["tested_7d"].fillna(False).to_numpy(dtype=bool) & target["outcome_operational"].notna().to_numpy()
    y_all = target["outcome_operational"].fillna(0).to_numpy(dtype=float)
    y_observed = y_all[observed]
    p_all = target["updated_prediction"].to_numpy(dtype=float)
    p_observed = p_all[observed]
    lower, upper = config["ipw_truncation_percentiles"]
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    conditional = performance_metrics(y_observed, p_observed)
    row = {
        "estimand": "conditional performance among patients with observed outcomes",
        "estimator": "complete case",
        "assumption": "conditions on postoperative creatinine observation",
    }
    row.update({f"{key}_estimate": value for key, value in conditional.items()})
    rows.append(row)

    for label, propensity in [("basic", propensity_basic), ("expanded", propensity_expanded)]:
        weights, low_cut, high_cut = truncated_inverse_weights(propensity, observed, lower, upper)
        metric = performance_metrics(y_observed, p_observed, weights)
        event_total = float(np.sum(weights * y_observed))
        metric["events"] = event_total
        metric["event_rate"] = event_total / len(target)
        metric["predicted_mean"] = float(p_all.mean())
        metric["oe_ratio"] = event_total / float(p_all.sum())
        row = {
            "estimand": "all-eligible performance under missing at random",
            "estimator": f"cross-fitted IPW {label}",
            "assumption": "outcome observation is independent of AKI conditional on measured model covariates",
        }
        row.update({f"{key}_estimate": value for key, value in metric.items()})
        rows.append(row)
        details[label] = {
            "weights": weights,
            "weight_low_cut": low_cut,
            "weight_high_cut": high_cut,
            "effective_sample_size": effective_sample_size(weights),
        }

    w_raw = 1 / clip_probability(propensity_expanded)
    pseudo_y = outcome_probability + observed * w_raw * (y_all - outcome_probability)
    event_total = float(pseudo_y.sum())
    predicted_total = float(p_all.sum())
    g_observed = y_all * (1 - p_all) ** 2 + (1 - y_all) * p_all**2
    g_model = outcome_probability * (1 - p_all) ** 2 + (1 - outcome_probability) * p_all**2
    brier_pseudo = g_model + observed * w_raw * (g_observed - g_model)
    expanded_metric = performance_metrics(y_observed, p_observed, details["expanded"]["weights"])
    aipw = {
        "n": float(len(target)),
        "weighted_n": float(len(target)),
        "events": event_total,
        "event_rate": event_total / len(target),
        "predicted_mean": float(p_all.mean()),
        "oe_ratio": event_total / predicted_total,
        "calibration_intercept": expanded_metric["calibration_intercept"],
        "calibration_joint_intercept": expanded_metric["calibration_joint_intercept"],
        "calibration_slope": expanded_metric["calibration_slope"],
        "auroc": expanded_metric["auroc"],
        "auprc": expanded_metric["auprc"],
        "brier": float(brier_pseudo.mean()),
        "brier_skill_score": np.nan,
        "grouped_ici": np.nan,
    }
    row = {
        "estimand": "all-eligible performance under missing at random",
        "estimator": "cross-fitted augmented IPW expanded",
        "assumption": "either the measured observation model or measured outcome model is correctly specified for linear functionals",
        "metric_specific_note": "O/E, event rate, Brier, and workload are augmented-IPW estimates; calibration slope and discrimination use expanded IPW",
    }
    row.update({f"{key}_estimate": value for key, value in aipw.items()})
    rows.append(row)
    details["aipw_pseudo_y"] = pseudo_y
    details["aipw_brier_pseudo"] = brier_pseudo
    details["aipw_event_total"] = event_total
    details["observed"] = observed
    details["y_all"] = y_all
    return pd.DataFrame(rows), details


def observation_analysis(
    target: pd.DataFrame, config: dict[str, Any], seed_offset: int = 0, save_models: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    observed = target["tested_7d"].fillna(False).to_numpy(dtype=bool) & target["outcome_operational"].notna().to_numpy()
    y_all = target["outcome_operational"].fillna(0).to_numpy(dtype=float)
    folds = int(config["crossfit_folds"])
    seed = int(config["random_seed"]) + seed_offset
    propensity_basic, basic_model = crossfit_observation_model(target, observed, False, folds, seed + 100)
    propensity_expanded, expanded_model = crossfit_observation_model(target, observed, True, folds, seed + 200)
    outcome_probability, outcome_model = crossfit_outcome_model(target, observed, y_all, folds, seed + 300)
    estimators, details = target_estimators(
        target, propensity_basic, propensity_expanded, outcome_probability, config
    )
    details.update(
        {
            "propensity_basic": propensity_basic,
            "propensity_expanded": propensity_expanded,
            "outcome_probability": outcome_probability,
        }
    )
    diagnostics_rows = []
    for label, propensity in [("basic", propensity_basic), ("expanded", propensity_expanded)]:
        weights = details[label]["weights"]
        tested_p = propensity[observed]
        untested_p = propensity[~observed]
        lower_support = max(float(tested_p.min()), float(untested_p.min()))
        upper_support = min(float(tested_p.max()), float(untested_p.max()))
        row = {
            "model": f"outcome observation {label}",
            "eligible_n": len(target),
            "observed_n": int(observed.sum()),
            "observation_rate": float(observed.mean()),
            "propensity_min": float(propensity.min()),
            "propensity_p01": float(np.quantile(propensity, 0.01)),
            "propensity_median": float(np.median(propensity)),
            "propensity_p99": float(np.quantile(propensity, 0.99)),
            "propensity_max": float(propensity.max()),
            "fraction_outside_common_support": float(
                np.mean((propensity < lower_support) | (propensity > upper_support))
            ),
            "effective_sample_size": effective_sample_size(weights),
            "weight_min": float(weights.min()),
            "weight_median": float(np.median(weights)),
            "weight_p99": float(np.quantile(weights, 0.99)),
            "weight_max": float(weights.max()),
        }
        row.update(calibration_diagnostics(observed.astype(int), propensity))
        diagnostics_rows.append(row)
    outcome_diag = {
        "model": "auxiliary outcome model among observed outcomes",
        "eligible_n": len(target),
        "observed_n": int(observed.sum()),
        "observation_rate": float(observed.mean()),
        "propensity_min": np.nan,
        "propensity_p01": np.nan,
        "propensity_median": np.nan,
        "propensity_p99": np.nan,
        "propensity_max": np.nan,
        "fraction_outside_common_support": np.nan,
        "effective_sample_size": np.nan,
        "weight_min": np.nan,
        "weight_median": np.nan,
        "weight_p99": np.nan,
        "weight_max": np.nan,
    }
    outcome_diag.update(calibration_diagnostics(y_all[observed].astype(int), outcome_probability[observed]))
    diagnostics_rows.append(outcome_diag)
    diagnostics = pd.DataFrame(diagnostics_rows)
    balance = weighted_balance_table(target, observed, details["expanded"]["weights"])
    if save_models:
        joblib.dump(basic_model, MODEL_DIR / "outcome_observation_model_basic_v3.joblib")
        joblib.dump(expanded_model, MODEL_DIR / "outcome_observation_model_expanded_v3.joblib")
        joblib.dump(outcome_model, MODEL_DIR / "auxiliary_outcome_model_v3.joblib")
    return estimators, diagnostics, balance, details


def mnar_analysis(target: pd.DataFrame, details: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = details["observed"]
    y_all = details["y_all"]
    outcome_probability = details["outcome_probability"]
    prediction = target["updated_prediction"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    workload_rows: list[dict[str, Any]] = []
    for multiplier in config["mnar_odds_multipliers"]:
        shifted = expit(logit(clip_probability(outcome_probability)) + math.log(float(multiplier)))
        fractional_y = np.where(observed, y_all, shifted)
        metric = performance_metrics(fractional_y, prediction)
        row = {
            "untested_vs_tested_outcome_odds_multiplier": multiplier,
            "tested_n": int(observed.sum()),
            "untested_n": int((~observed).sum()),
            "imputation_model": "cross-fitted auxiliary outcome model; candidate risk score not used as the imputation target",
            "interpretation": "MNAR pattern-mixture sensitivity scenario, not an identified full-cohort truth",
        }
        row.update({f"{key}_estimate": value for key, value in metric.items()})
        rows.append(row)
        for threshold in config["decision_thresholds"]:
            workload = workload_metrics(fractional_y, prediction, float(threshold))
            workload_rows.append(
                {
                    "estimator": "MNAR pattern-mixture",
                    "untested_vs_tested_outcome_odds_multiplier": multiplier,
                    "threshold": threshold,
                    "deployment_denominator_n": len(target),
                    **workload,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(workload_rows)


def nonparametric_missing_outcome_bounds(target: pd.DataFrame, details: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    observed = details["observed"]
    y_all = details["y_all"]
    prediction = target["updated_prediction"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for label, fill in [("all unobserved outcomes are non-events", 0.0), ("all unobserved outcomes are events", 1.0)]:
        y = np.where(observed, y_all, fill)
        metric = performance_metrics(y, prediction)
        row = {"bound_scenario": label, "untested_outcome_value": fill}
        row.update({f"{key}_estimate": value for key, value in metric.items()})
        for threshold in config["decision_thresholds"]:
            workload = workload_metrics(y, prediction, float(threshold))
            row[f"threshold_{threshold:g}_ppv"] = workload["positive_predictive_value"]
            row[f"threshold_{threshold:g}_net_benefit"] = workload["net_benefit"]
        rows.append(row)
    return pd.DataFrame(rows)


def complete_full_cohort_workload(
    target: pd.DataFrame,
    estimators: pd.DataFrame,
    details: dict[str, Any],
    mnar_workload: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    observed = details["observed"]
    y_observed = details["y_all"][observed]
    p_all = target["updated_prediction"].to_numpy(dtype=float)
    p_observed = p_all[observed]
    rows: list[dict[str, Any]] = []
    for threshold in config["decision_thresholds"]:
        alert = p_all >= float(threshold)
        exact_alerts = int(alert.sum())
        rows.append(
            {
                "estimator": "exact alert burden in all eligible patients",
                "assumption": "none for alert count; outcomes are not used",
                "threshold": threshold,
                "deployment_denominator_n": len(target),
                "alerts": exact_alerts,
                "alerts_per_1000": 1000 * exact_alerts / len(target),
                "true_positives_per_1000": np.nan,
                "false_positives_per_1000": np.nan,
                "sensitivity": np.nan,
                "positive_predictive_value": np.nan,
                "false_positives_per_true_positive": np.nan,
                "net_benefit": np.nan,
                "treat_all_net_benefit": np.nan,
                "net_benefit_vs_better_default": np.nan,
            }
        )
        conditional = workload_metrics(y_observed, p_observed, float(threshold))
        rows.append(
            {
                "estimator": "complete-case descriptive",
                "assumption": "conditions on postoperative creatinine observation; not a deployment-population estimate",
                "threshold": threshold,
                "deployment_denominator_n": int(observed.sum()),
                **conditional,
            }
        )
        for label in ["basic", "expanded"]:
            weights = details[label]["weights"]
            alert_observed = p_observed >= float(threshold)
            tp_total = float(np.sum(weights * alert_observed * y_observed))
            event_total = float(np.sum(weights * y_observed))
            fp_total = exact_alerts - tp_total
            ppv = tp_total / exact_alerts if exact_alerts else np.nan
            sensitivity = tp_total / event_total if event_total else np.nan
            net_benefit = tp_total / len(target) - fp_total / len(target) * threshold / (1 - threshold)
            prevalence = event_total / len(target)
            treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
            rows.append(
                {
                    "estimator": f"cross-fitted IPW {label}",
                    "assumption": "missing at random conditional on measured observation-model variables",
                    "threshold": threshold,
                    "deployment_denominator_n": len(target),
                    "alerts": exact_alerts,
                    "alerts_per_1000": 1000 * exact_alerts / len(target),
                    "true_positives_per_1000": 1000 * tp_total / len(target),
                    "false_positives_per_1000": 1000 * fp_total / len(target),
                    "sensitivity": sensitivity,
                    "positive_predictive_value": ppv,
                    "false_positives_per_true_positive": fp_total / tp_total if tp_total else np.nan,
                    "net_benefit": net_benefit,
                    "treat_all_net_benefit": treat_all,
                    "net_benefit_vs_better_default": net_benefit - max(0.0, treat_all),
                }
            )
        pseudo_y = details["aipw_pseudo_y"]
        aipw_workload = workload_metrics(pseudo_y, p_all, float(threshold))
        aipw_workload["alerts"] = exact_alerts
        aipw_workload["alerts_per_1000"] = 1000 * exact_alerts / len(target)
        rows.append(
            {
                "estimator": "cross-fitted augmented IPW expanded",
                "assumption": "MAR; either measured observation or outcome model correct for linear functionals",
                "threshold": threshold,
                "deployment_denominator_n": len(target),
                **aipw_workload,
            }
        )
    mnar = mnar_workload.copy()
    mnar["assumption"] = mnar.apply(
        lambda row: f"MNAR odds multiplier {row['untested_vs_tested_outcome_odds_multiplier']:.3g}", axis=1
    )
    rows_frame = pd.DataFrame(rows)
    columns = rows_frame.columns.tolist()
    for column in columns:
        if column not in mnar:
            mnar[column] = np.nan
    mnar = mnar[columns]
    return pd.concat([rows_frame, mnar], ignore_index=True)


def endpoint_and_frequency_sensitivity(
    update: pd.DataFrame,
    target: pd.DataFrame,
    model: Pipeline,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    update_source = model.predict_proba(update[FEATURES])[:, 1]
    alpha, beta = fit_recalibration(update["analysis_outcome"], update_source)
    target_source = model.predict_proba(target[FEATURES])[:, 1]
    target_prediction = apply_recalibration(target_source, alpha, beta)
    endpoints = [
        ("maximum creatinine through 7 days", "outcome_operational"),
        ("maximum creatinine through 48 hours", "aki_48h"),
        ("first postoperative creatinine only", "first_measurement_aki"),
        ("first two postoperative creatinine measurements", "first_two_measurements_aki"),
    ]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(config["random_seed"]) + 40_000)
    update_y = update["analysis_outcome"].to_numpy(dtype=int)
    replicate_parameters = []
    for _ in range(int(config["bootstrap_replicates"])):
        index = rng.integers(0, len(update), len(update))
        replicate_parameters.append(fit_recalibration(update_y[index], update_source[index]))
    for endpoint_index, (label, column) in enumerate(endpoints, start=1):
        if column == "aki_48h":
            mask = target["postop_cr_48h_count"].gt(0).to_numpy()
        else:
            mask = target[column].notna().to_numpy()
        y = target.loc[mask, column].astype(int).to_numpy()
        source_prediction = target_source[mask]
        point = performance_metrics(y, target_prediction[mask])
        rng_endpoint = np.random.default_rng(int(config["random_seed"]) + 41_000 + endpoint_index)
        boot_rows = []
        for replicate, (a, b) in enumerate(replicate_parameters):
            index = rng_endpoint.integers(0, len(y), len(y))
            boot_rows.append(performance_metrics(y[index], apply_recalibration(source_prediction[index], a, b)))
        row: dict[str, Any] = {
            "outcome_ascertainment_strategy": label,
            "observed_n": len(y),
            "observed_events": int(y.sum()),
            "interpretation": "deterministic detection-frequency sensitivity analysis",
        }
        row.update(summarize_bootstrap(point, pd.DataFrame(boot_rows)))
        rows.append(row)
    endpoint_table = pd.DataFrame(rows)

    frequency = category_count(target["postop_cr_7d_count"])
    frequency_rows: list[dict[str, Any]] = []
    for index, label in enumerate(["1", "2", "3", "4+"], start=1):
        mask = (frequency == label).to_numpy() & target["outcome_operational"].notna().to_numpy()
        y = target.loc[mask, "outcome_operational"].astype(int).to_numpy()
        prediction = target_prediction[mask]
        point = performance_metrics(y, prediction)
        rng_frequency = np.random.default_rng(int(config["random_seed"]) + 42_000 + index)
        boot_rows = []
        for _ in range(int(config["bootstrap_replicates"])):
            sample = rng_frequency.integers(0, len(y), len(y))
            boot_rows.append(performance_metrics(y[sample], prediction[sample]))
        row = {
            "seven_day_creatinine_measurement_count": label,
            "observed_n": len(y),
            "observed_events": int(y.sum()),
            "interpretation": "descriptive; testing intensity may respond to evolving postoperative illness",
        }
        row.update(summarize_bootstrap(point, pd.DataFrame(boot_rows)))
        frequency_rows.append(row)
    return endpoint_table, pd.DataFrame(frequency_rows)


def outcome_representation_analysis(
    inspire: pd.DataFrame, mover: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    representations = ["operational", "definite", "possible", "coarsened_operational"]
    for index, representation in enumerate(representations, start=1):
        outcome = f"outcome_{representation}"
        source = observed_outcome(eligible_base(inspire), outcome)
        update = observed_outcome(eligible_base(mover, 2021), outcome)
        target = observed_outcome(eligible_base(mover, 2022), outcome)
        if representation != "operational":
            if "baseline_cr_coarse" in source:
                source["baseline_cr"] = source["baseline_cr_coarse"]
            if "baseline_cr_coarse" in update:
                update["baseline_cr"] = update["baseline_cr_coarse"]
            if "baseline_cr_coarse" in target:
                target["baseline_cr"] = target["baseline_cr_coarse"]
        model = make_risk_model(int(config["random_seed"]) + 50_000 + index).fit(
            source[FEATURES], source["analysis_outcome"]
        )
        update_source = model.predict_proba(update[FEATURES])[:, 1]
        target_source = model.predict_proba(target[FEATURES])[:, 1]
        alpha, beta = fit_recalibration(update["analysis_outcome"], update_source)
        prediction = apply_recalibration(target_source, alpha, beta)
        point = performance_metrics(target["analysis_outcome"], prediction)
        bootstrap = nested_bootstrap_performance(
            update,
            target,
            update_source,
            target_source,
            int(config["bootstrap_replicates"]),
            int(config["random_seed"]) + 51_000 + index,
        )
        row = {
            "outcome_representation": representation,
            "source_n": len(source),
            "source_events": int(source["analysis_outcome"].sum()),
            "update_n": len(update),
            "update_events": int(update["analysis_outcome"].sum()),
            "evaluation_n": len(target),
            "evaluation_events": int(target["analysis_outcome"].sum()),
            "update_alpha": alpha,
            "update_beta": beta,
        }
        row.update(summarize_bootstrap(point, bootstrap))
        rows.append(row)
    return pd.DataFrame(rows)


def vitaldb_supportive_analysis(
    update: pd.DataFrame, vitaldb: pd.DataFrame, model: Pipeline, config: dict[str, Any]
) -> pd.DataFrame:
    target = vitaldb.loc[
        vitaldb["has_baseline_cr"].fillna(False).astype(bool)
        & vitaldb["baseline_cr_under4"].fillna(False).astype(bool)
        & vitaldb["tested_7d"].fillna(False).astype(bool)
        & vitaldb["aki"].notna()
    ].copy().reset_index(drop=True)
    target["analysis_outcome"] = target["aki"].astype(int)
    update_source = model.predict_proba(update[FEATURES])[:, 1]
    target_source = model.predict_proba(target[FEATURES])[:, 1]
    alpha, beta = fit_recalibration(update["analysis_outcome"], update_source)
    target_updated = apply_recalibration(target_source, alpha, beta)
    rows = []
    for index, (label, prediction) in enumerate(
        [("source model", target_source), ("MOVER 2021 recalibration applied", target_updated)], start=1
    ):
        point = performance_metrics(target["analysis_outcome"], prediction)
        rng = np.random.default_rng(int(config["random_seed"]) + 60_000 + index)
        boot_rows = []
        for _ in range(int(config["bootstrap_replicates"])):
            update_index = rng.integers(0, len(update), len(update))
            target_index = rng.integers(0, len(target), len(target))
            if index == 1:
                p = target_source[target_index]
            else:
                a, b = fit_recalibration(
                    update["analysis_outcome"].to_numpy()[update_index], update_source[update_index]
                )
                p = apply_recalibration(target_source[target_index], a, b)
            boot_rows.append(performance_metrics(target["analysis_outcome"].to_numpy()[target_index], p))
        row = {"model_state": label, "supportive_dataset": "VitalDB", "evaluation_n": len(target)}
        row.update(summarize_bootstrap(point, pd.DataFrame(boot_rows)))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_full_target_analysis(
    update: pd.DataFrame,
    target: pd.DataFrame,
    model: Pipeline,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replicates = int(config["bootstrap_replicates"])
    rng = np.random.default_rng(int(config["random_seed"]) + 70_000)
    update_source = model.predict_proba(update[FEATURES])[:, 1]
    target_source = model.predict_proba(target[FEATURES])[:, 1]
    update_y = update["analysis_outcome"].to_numpy(dtype=int)
    mar_rows: list[dict[str, Any]] = []
    mnar_rows: list[dict[str, Any]] = []
    workload_rows: list[dict[str, Any]] = []
    valid = 0
    attempts = 0
    while valid < replicates and attempts < replicates * 2:
        attempts += 1
        update_index = rng.integers(0, len(update), len(update))
        target_index = rng.integers(0, len(target), len(target))
        update_sample = update.iloc[update_index].reset_index(drop=True)
        target_sample = target.iloc[target_index].reset_index(drop=True)
        alpha, beta = fit_recalibration(update_y[update_index], update_source[update_index])
        target_sample["updated_prediction"] = apply_recalibration(target_source[target_index], alpha, beta)
        try:
            estimators, _, _, details = observation_analysis(
                target_sample, config, seed_offset=80_000 + attempts * 10, save_models=False
            )
            mnar, mnar_workload = mnar_analysis(target_sample, details, config)
            workload = complete_full_cohort_workload(target_sample, estimators, details, mnar_workload, config)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        valid += 1
        for row in estimators.to_dict(orient="records"):
            row["replicate"] = valid
            mar_rows.append(row)
        for row in mnar.to_dict(orient="records"):
            row["replicate"] = valid
            mnar_rows.append(row)
        for row in workload.to_dict(orient="records"):
            row["replicate"] = valid
            workload_rows.append(row)
        if valid % 50 == 0 or valid == replicates:
            print(f"Completed {valid}/{replicates} full-target bootstrap replicates", flush=True)
    if valid != replicates:
        raise RuntimeError(f"Full-target bootstrap produced {valid} of {replicates} replicates")
    mar = pd.DataFrame(mar_rows)
    mnar = pd.DataFrame(mnar_rows)
    workload = pd.DataFrame(workload_rows)
    mar.to_parquet(DATA_DIR / "bootstrap_mar_estimators_v3.parquet", index=False)
    mnar.to_parquet(DATA_DIR / "bootstrap_mnar_estimators_v3.parquet", index=False)
    workload.to_parquet(DATA_DIR / "bootstrap_workload_estimators_v3.parquet", index=False)
    return mar, mnar, workload


def add_estimator_intervals(
    point: pd.DataFrame, bootstrap: pd.DataFrame, keys: list[str], estimate_suffix: bool
) -> pd.DataFrame:
    output = point.copy()
    numeric_columns = [
        column
        for column in output.columns
        if pd.api.types.is_numeric_dtype(output[column]) and column not in keys
    ]
    for column in numeric_columns:
        if column not in bootstrap:
            continue
        lower_values = []
        upper_values = []
        for _, row in output.iterrows():
            mask = pd.Series(True, index=bootstrap.index)
            for key in keys:
                if pd.isna(row[key]):
                    mask &= bootstrap[key].isna()
                else:
                    mask &= bootstrap[key].eq(row[key])
            values = pd.to_numeric(bootstrap.loc[mask, column], errors="coerce").dropna()
            if len(values):
                lower, upper = values.quantile([0.025, 0.975]).tolist()
            else:
                lower = upper = np.nan
            lower_values.append(lower)
            upper_values.append(upper)
        base = column[:-9] if estimate_suffix and column.endswith("_estimate") else column
        output[f"{base}_ci_lower"] = lower_values
        output[f"{base}_ci_upper"] = upper_values
    return output


def table_mapping_and_boundaries() -> None:
    mapping = pd.DataFrame(
        [
            {"common_variable": "age", "unit_or_encoding": "years", "INSPIRE": "age", "MOVER": "BIRTH_DATE released as age", "VitalDB": "age", "timing": "available before prediction"},
            {"common_variable": "sex_male", "unit_or_encoding": "1=male; 0=female", "INSPIRE": "sex", "MOVER": "SEX", "VitalDB": "sex", "timing": "available before prediction"},
            {"common_variable": "duration_h", "unit_or_encoding": "hours", "INSPIRE": "anesthesia end minus start", "MOVER": "AN_STOP_DATETIME minus AN_START_DATETIME", "VitalDB": "anesthesia duration", "timing": "known at end of anesthesia"},
            {"common_variable": "baseline_cr", "unit_or_encoding": "mg/dL", "INSPIRE": "latest valid creatinine in previous 7 days", "MOVER": "latest blood LOINC creatinine in previous 7 days", "VitalDB": "released baseline creatinine", "timing": "before anesthesia"},
            {"common_variable": "postoperative AKI", "unit_or_encoding": "KDIGO creatinine component", "INSPIRE": "maximum released creatinine through 48 h and 7 d", "MOVER": "strict blood creatinine through 48 h and 7 d", "VitalDB": "released AKI field", "timing": "through discharge or postoperative day 7"},
            {"common_variable": "surgery_category", "unit_or_encoding": "rule-based category; auxiliary only", "INSPIRE": "not used", "MOVER": "PRIMARY_PROCEDURE_NM regex mapping", "VitalDB": "not used", "timing": "known before prediction"},
        ]
    )
    write_csv(mapping, "23_cross_database_mapping_rules.csv")
    boundaries = pd.DataFrame(
        [
            {"question": "Was MOVER 2022 an independent confirmation cohort?", "answer": "No", "manuscript_handling": "Described as a post-exploration temporally held-out evaluation cohort."},
            {"question": "Was a clinical deployment evaluated?", "answer": "No", "manuscript_handling": "The model is an intentionally parsimonious methodological audit model."},
            {"question": "Was a hospital workflow or alert-linked intervention specified?", "answer": "No", "manuscript_handling": "Deployment assessability is reported as not assessed, not as a performance failure."},
            {"question": "Are all 2,587 eligible target patients included in alert-burden denominators?", "answer": "Yes", "manuscript_handling": "Alert counts are exact for all eligible patients; outcome-dependent quantities are assumption-dependent."},
            {"question": "Are missing outcomes point identified?", "answer": "No", "manuscript_handling": "Observed-outcome, MAR, MNAR, and nonparametric-bound results are reported separately."},
            {"question": "Are descriptive audit bands clinical safety margins?", "answer": "No", "manuscript_handling": "They are retained only as post-exploration descriptive bands."},
        ]
    )
    write_csv(boundaries, "24_claim_and_deployment_assessability.csv")


def make_figures(
    evidence: pd.DataFrame,
    risk_testing: pd.DataFrame,
    mnar: pd.DataFrame,
    workload: pd.DataFrame,
    temporal: pd.DataFrame,
    subgroups: pd.DataFrame,
    testing_frequency: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titleweight": "bold"})
    colors = {"source": "#35618D", "update": "#2F7D6D", "accent": "#A64B3C", "neutral": "#6B7280"}

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    nodes = [
        (0.08, "INSPIRE", "source development"),
        (0.36, "MOVER 2021", "chronological local update"),
        (0.66, "MOVER 2022", "post-exploration temporally\nheld-out evaluation"),
        (0.91, "VitalDB", "supportive stress test"),
    ]
    ax.plot([0.08, 0.91], [0.5, 0.5], color="#374151", lw=2)
    for x, title, subtitle in nodes:
        ax.scatter([x], [0.5], s=140, color=colors["update"] if "MOVER" in title else colors["source"], zorder=3)
        ax.text(x, 0.68, title, ha="center", va="bottom", fontsize=11, weight="bold")
        ax.text(x, 0.35, subtitle, ha="center", va="top", fontsize=9)
    ax.text(
        0.5,
        0.05,
        "The analysis amendment followed data access and exploration; the 2022 evaluation is not independent confirmation.",
        ha="center",
        color=colors["accent"],
        fontsize=9,
    )
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURE_DIR / f"figure_01_data_roles_postexploration.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    x = evidence["actual_events"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, metric, title, ideal, band in [
        (axes[0], "oe_ratio", "Observed-to-expected ratio", 1.0, config["descriptive_audit_bands"]["oe_ratio"]),
        (axes[1], "calibration_slope", "Calibration slope", 1.0, config["descriptive_audit_bands"]["calibration_slope"]),
    ]:
        y = evidence[f"{metric}_estimate"].to_numpy(dtype=float)
        low = evidence[f"{metric}_ci_lower"].to_numpy(dtype=float)
        high = evidence[f"{metric}_ci_upper"].to_numpy(dtype=float)
        axis.fill_between(x, low, high, color="#BCD6D0", alpha=0.65)
        axis.plot(x, y, marker="o", color=colors["update"], lw=2)
        axis.axhline(ideal, color="#111827", lw=1)
        axis.axhspan(band[0], band[1], color="#E5E7EB", alpha=0.6, zorder=0)
        axis.set_title(title)
        axis.set_xlabel("Observed AKI events used for local update")
        axis.grid(axis="y", color="#E5E7EB", lw=0.7)
    fig.text(0.5, -0.01, "Bands are post-exploration descriptive ranges, not clinical safety margins.", ha="center", fontsize=8)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURE_DIR / f"figure_02_evidence_accumulation.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(risk_testing["risk_decile"], 100 * risk_testing["postoperative_creatinine_observation_rate"], marker="o", color=colors["source"])
    axes[0].set_xlabel("Predicted-risk decile")
    axes[0].set_ylabel("Postoperative creatinine observed (%)")
    axes[0].set_title("Selective outcome observation")
    axes[0].grid(axis="y", color="#E5E7EB")
    axes[1].plot(mnar["untested_vs_tested_outcome_odds_multiplier"], mnar["oe_ratio_estimate"], marker="o", label="O/E ratio", color=colors["update"])
    axes[1].plot(mnar["untested_vs_tested_outcome_odds_multiplier"], mnar["calibration_slope_estimate"], marker="s", label="Calibration slope", color=colors["accent"])
    axes[1].axhline(1, color="#111827", lw=1)
    axes[1].set_xscale("log")
    axes[1].xaxis.set_minor_locator(mticker.NullLocator())
    axes[1].xaxis.set_minor_formatter(mticker.NullFormatter())
    axes[1].set_xticks(
        config["mnar_odds_multipliers"],
        labels=[f"{x:.2g}" for x in config["mnar_odds_multipliers"]],
    )
    axes[1].set_xlabel("Assumed AKI odds multiplier in untested patients")
    axes[1].set_title("MNAR sensitivity")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", color="#E5E7EB")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURE_DIR / f"figure_03_outcome_observation_sensitivity.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    exact = workload.loc[workload["estimator"].eq("exact alert burden in all eligible patients")].sort_values("threshold")
    aipw = workload.loc[workload["estimator"].eq("cross-fitted augmented IPW expanded")].sort_values("threshold")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(exact))
    ax.bar(positions, exact["alerts_per_1000"], width=0.7, color=colors["source"], label="All alerts")
    ax.bar(positions, aipw["true_positives_per_1000"], width=0.7, color=colors["update"], label="Estimated true-positive alerts (MAR)")
    ax.set_xticks(positions, [f"{100 * value:.0f}%" for value in exact["threshold"]])
    ax.set_xlabel("Candidate risk threshold")
    ax.set_ylabel("Alerts per 1,000 all eligible patients")
    ax.set_title("Full-cohort alert burden")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E5E7EB", zorder=0)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURE_DIR / f"figure_04_full_cohort_workload.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    combined = pd.concat(
        [
            temporal.assign(display_group="Calendar quarter"),
            subgroups.assign(display_group="Patient subgroup"),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, max(5, 0.33 * len(combined))))
    y_pos = np.arange(len(combined))[::-1]
    for axis, metric, title in [(axes[0], "oe_ratio", "O/E ratio"), (axes[1], "calibration_slope", "Calibration slope")]:
        estimate = combined[f"{metric}_estimate"].to_numpy()
        lower = combined[f"{metric}_ci_lower"].to_numpy()
        upper = combined[f"{metric}_ci_upper"].to_numpy()
        axis.errorbar(estimate, y_pos, xerr=[estimate - lower, upper - estimate], fmt="o", color=colors["source"], ecolor="#94A3B8", capsize=2)
        axis.axvline(1, color="#111827", lw=1)
        axis.set_title(title)
        axis.grid(axis="x", color="#E5E7EB")
    axes[0].set_yticks(y_pos, combined["stratum"])
    axes[1].set_yticks(y_pos, [""] * len(combined))
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURE_DIR / f"figure_05_temporal_subgroup_performance.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    seven_day = testing_frequency.loc[testing_frequency["window"].eq("7 days")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(seven_day["testing_frequency"].astype(str), seven_day["percent_of_all_eligible"], color=["#6B7280", "#8FB8D8", "#6EA5CF", "#4F91BF", "#35618D"])
    ax.set_xlabel("Number of postoperative creatinine measurements through day 7")
    ax.set_ylabel("All eligible patients (%)")
    ax.set_title("Outcome ascertainment intensity")
    ax.grid(axis="y", color="#E5E7EB")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURE_DIR / f"figure_06_testing_frequency.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_reports(
    config: dict[str, Any],
    observed_performance: pd.DataFrame,
    mar: pd.DataFrame,
    mnar: pd.DataFrame,
    workload: pd.DataFrame,
    audit: pd.DataFrame,
    started: datetime,
    runtime_seconds: float,
) -> None:
    observed_updated = observed_performance.iloc[1]
    aipw = mar.loc[mar["estimator"].eq("cross-fitted augmented IPW expanded")].iloc[0]
    exact = workload.loc[workload["estimator"].eq("exact alert burden in all eligible patients")]
    summary = f"""# Results summary

## Study position

This is a post-exploration temporal evaluation and multidatabase methodological case study. MOVER 2022 is not an independent confirmation cohort, and clinical deployment was not evaluated.

## Cohorts

- INSPIRE source development: 24,874 patients and 1,671 observed AKI events.
- MOVER 2021 local update: 2,212 patients and 245 observed AKI events.
- MOVER 2022 target population: 2,587 eligible patients; 2,033 (78.6%) had an observed postoperative creatinine outcome; 224 observed AKI events.

## Main estimates

- Among patients with observed outcomes, logistic recalibration gave O/E {observed_updated['oe_ratio_estimate']:.3f} ({observed_updated['oe_ratio_ci_lower']:.3f} to {observed_updated['oe_ratio_ci_upper']:.3f}) and calibration slope {observed_updated['calibration_slope_estimate']:.3f} ({observed_updated['calibration_slope_ci_lower']:.3f} to {observed_updated['calibration_slope_ci_upper']:.3f}).
- The augmented MAR analysis of all eligible patients gave O/E {aipw['oe_ratio_estimate']:.3f} ({aipw['oe_ratio_ci_lower']:.3f} to {aipw['oe_ratio_ci_upper']:.3f}).
- Across MNAR odds multipliers 0.5 to 3.0, O/E ranged from {mnar['oe_ratio_estimate'].min():.3f} to {mnar['oe_ratio_estimate'].max():.3f}; this range is assumption dependent.

## Full-cohort alert burden

"""
    for row in exact.itertuples(index=False):
        summary += f"- At {100 * row.threshold:.0f}%, {int(row.alerts)} of 2,587 patients would generate an alert ({row.alerts_per_1000:.1f} per 1,000).\n"
    summary += """

## Interpretation

Local recalibration improved average calibration, but all-eligible performance remained dependent on unverifiable outcome-observation assumptions. Because no hospital workflow, alert-linked action, capacity limit, or harm boundary was specified, the study does not estimate clinical deployment suitability.
"""
    (REPORT_DIR / "01_results_summary.md").write_text(summary, encoding="utf-8")

    runbook = """# Reproducibility runbook

1. Use Python 3.9.6 with the package versions in `config/requirements-lock.txt`.
2. Place `inspire_rebuilt_v2.parquet`, `mover_rebuilt_v2.parquet`, `mover_strict_blood_creatinine.parquet`, and `vitaldb_supportive_v2.parquet` in `data/processed/`.
3. Run `python scripts/run_postreview_analysis.py` from the package root.
4. Confirm the two creatinine count-concordance checks report zero discordant cases.
5. Confirm the target denominator is 2,587 for every exact alert-burden row.
6. Run `python scripts/run_statistical_qa.py` after the analysis tables and serialized models have been generated.
7. Run `python scripts/run_document_qa.py` after the manuscript and supplement files have been generated.

Raw public datasets are not redistributed. Dataset-specific acquisition and licensing remain governed by INSPIRE, MOVER, and VitalDB.
"""
    (REPORT_DIR / "02_reproducibility_runbook.md").write_text(runbook, encoding="utf-8")
    claim_report = """# Claim-boundary report

- Independent confirmation: not claimed.
- Clinical deployment study: not claimed.
- Clinical safety threshold: not claimed.
- Universal local-event requirement: not claimed.
- MOVER 2022 role: post-exploration temporally held-out evaluation.
- Model role: intentionally parsimonious methodological audit model.
- Workload denominator: all 2,587 eligible patients for exact alerts.
- Outcome-dependent workload: reported under complete-case, MAR, and MNAR assumptions.
"""
    (REPORT_DIR / "03_claim_boundary_report.md").write_text(claim_report, encoding="utf-8")
    write_json(
        {
            "study_id": config["study_id"],
            "analysis_version": config["analysis_version"],
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": runtime_seconds,
            "bootstrap_replicates": config["bootstrap_replicates"],
            "random_seed": config["random_seed"],
            "testing_timing_audit": audit.to_dict(orient="records"),
        },
        REPORT_DIR / "04_analysis_run_record.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=None,
        help="Override the configured bootstrap count for a diagnostic run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    config = load_config()
    if args.bootstrap_replicates is not None:
        if args.bootstrap_replicates < 5:
            raise ValueError("At least five bootstrap replicates are required")
        config["bootstrap_replicates"] = int(args.bootstrap_replicates)
    started = datetime.now(timezone.utc)
    clock = time.time()
    print("Loading analysis-ready cohorts", flush=True)
    inspire = pd.read_parquet(DATA_DIR / "inspire_rebuilt_v2.parquet")
    mover = pd.read_parquet(DATA_DIR / "mover_rebuilt_v2.parquet")
    vitaldb = pd.read_parquet(DATA_DIR / "vitaldb_supportive_v2.parquet")
    labs = pd.read_parquet(DATA_DIR / "mover_strict_blood_creatinine.parquet")
    mover, info = attach_mover_auxiliary_data(mover, Path(config["raw_source_paths"]["mover_patient_information"]))

    source = observed_outcome(eligible_base(inspire), "outcome_operational")
    update = observed_outcome(eligible_base(mover, 2021), "outcome_operational")
    target = eligible_base(mover, 2022)
    target, timing_audit = derive_testing_timing(target, labs, info)
    write_csv(timing_audit, "14_testing_timing_derivation_audit.csv")
    if int(timing_audit["discordant_cases"].sum()) != 0:
        raise RuntimeError("Derived postoperative creatinine counts do not match the analysis cohort")

    print("Fitting source model and MOVER 2021 recalibration", flush=True)
    model = make_risk_model(int(config["random_seed"])).fit(source[FEATURES], source["analysis_outcome"])
    update_source = model.predict_proba(update[FEATURES])[:, 1]
    alpha, beta = fit_recalibration(update["analysis_outcome"], update_source)
    target["source_prediction"] = model.predict_proba(target[FEATURES])[:, 1]
    target["updated_prediction"] = apply_recalibration(target["source_prediction"], alpha, beta)
    export_model_specification(model, alpha, beta)
    cohort_tables(source, update, target, vitaldb)

    print("Running core temporal evaluation", flush=True)
    target_observed = observed_outcome(target, "outcome_operational")
    observed_performance, alpha, beta, _ = observed_performance_analysis(update, target_observed, model, config)
    evidence = evidence_accumulation_analysis(update, target_observed, model, config)
    temporal, subgroups = temporal_and_subgroup_analysis(update, target_observed, model, config)

    print("Running observation-process, MAR, and MNAR analyses", flush=True)
    observation_process_tables(target)
    estimators, diagnostics, balance, details = observation_analysis(target, config)
    mnar, mnar_workload = mnar_analysis(target, details, config)
    bounds = nonparametric_missing_outcome_bounds(target, details, config)
    workload = complete_full_cohort_workload(target, estimators, details, mnar_workload, config)

    print("Bootstrapping full-target estimators", flush=True)
    bootstrap_mar, bootstrap_mnar, bootstrap_workload = bootstrap_full_target_analysis(update, target, model, config)
    estimators = add_estimator_intervals(estimators, bootstrap_mar, ["estimator"], True)
    mnar = add_estimator_intervals(
        mnar, bootstrap_mnar, ["untested_vs_tested_outcome_odds_multiplier"], True
    )
    workload = add_estimator_intervals(
        workload,
        bootstrap_workload,
        ["estimator", "threshold", "assumption"],
        False,
    )
    write_csv(estimators, "15_mar_full_cohort_performance.csv")
    write_csv(diagnostics, "16_observation_and_outcome_model_diagnostics.csv")
    write_csv(balance, "17_observation_weighted_balance.csv")
    write_csv(mnar, "18_mnar_full_metric_sensitivity.csv")
    write_csv(bounds, "19_nonparametric_missing_outcome_bounds.csv")
    write_csv(workload, "20_full_cohort_workload_and_utility.csv")

    print("Running testing-frequency and supportive sensitivities", flush=True)
    endpoint, frequency = endpoint_and_frequency_sensitivity(update, target, model, config)
    write_csv(endpoint, "21_detection_frequency_endpoint_sensitivity.csv")
    write_csv(frequency, "22_performance_by_testing_frequency.csv")
    representation = outcome_representation_analysis(inspire, mover, config)
    write_csv(representation, "25_outcome_representation_sensitivity.csv")
    supportive = vitaldb_supportive_analysis(update, vitaldb, model, config)
    write_csv(supportive, "26_vitaldb_supportive_stress_test.csv")
    table_mapping_and_boundaries()

    risk_testing = pd.read_csv(TABLE_DIR / "12_testing_by_predicted_risk_decile.csv")
    testing_frequency = pd.read_csv(TABLE_DIR / "11_testing_frequency_distribution.csv")
    make_figures(evidence, risk_testing, mnar, workload, temporal, subgroups, testing_frequency, config)

    runtime_seconds = time.time() - clock
    generate_reports(config, observed_performance, estimators, mnar, workload, timing_audit, started, runtime_seconds)
    table_files = sorted(TABLE_DIR.glob("*.csv"))
    table_index = pd.DataFrame(
        [
            {
                "order": index,
                "filename": path.name,
                "rows": len(pd.read_csv(path)),
                "sha256": sha256_file(path),
            }
            for index, path in enumerate(table_files, start=1)
        ]
    )
    write_csv(table_index, "99_table_index.csv")
    print(f"Analysis complete in {runtime_seconds:.1f} seconds", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
