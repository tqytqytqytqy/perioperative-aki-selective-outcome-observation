#!/usr/bin/env python3
"""Run the v3.1 selective-outcome development, updating, and evaluation chain."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_postreview_analysis as base  # noqa: E402


ROOT = SCRIPT_DIR.parent
CONFIG_PATH = ROOT / "config" / "analysis_config.json"
DATA_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "tables"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
QA_DIR = ROOT / "qa"
TMP_DIR = ROOT / "tmp" / "v31_diagnostic"

FEATURES = ["age", "sex_male", "duration_h", "baseline_cr"]
CONTINUOUS = ["age", "duration_h", "baseline_cr"]
BINARY = ["sex_male"]
PERFORMANCE_METRICS = [
    "events",
    "event_rate",
    "predicted_mean",
    "oe_ratio",
    "calibration_intercept",
    "calibration_joint_intercept",
    "calibration_slope",
    "auroc",
    "auprc",
    "brier",
    "brier_skill_score",
    "grouped_ici",
]

_WORKER_COHORTS: dict[str, pd.DataFrame] | None = None
_WORKER_CONFIG: dict[str, Any] | None = None


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def write_csv(frame: pd.DataFrame, filename: str, directory: Path = TABLE_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_float_dtype(output[column]):
            output[column] = output[column].replace([np.inf, -np.inf], np.nan)
    output.to_csv(path, index=False)
    return path


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_risk_model(seed: int) -> Pipeline:
    continuous = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False, extrapolation="constant")),
        ]
    )
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    preprocess = ColumnTransformer(
        [("continuous", continuous, CONTINUOUS), ("binary", binary, BINARY)],
        remainder="drop",
    )
    classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=seed)
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def nuisance_specification(phase: str, expanded: bool, seed: int) -> tuple[Pipeline, list[str]]:
    continuous = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "spline" if expanded else "identity",
                SplineTransformer(n_knots=4, degree=3, include_bias=False, extrapolation="constant")
                if expanded
                else "passthrough",
            ),
        ]
    )
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    transformers: list[tuple[str, Any, list[str]]] = [
        ("continuous", continuous, CONTINUOUS),
        ("binary", binary, BINARY),
    ]
    features = CONTINUOUS + BINARY
    if expanded:
        categorical_features = ["asa"] + (["surgery_category"] if phase.startswith("MOVER") else [])
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical, categorical_features))
        features += categorical_features
    preprocess = ColumnTransformer(transformers, remainder="drop")
    classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=seed)
    return Pipeline([("preprocess", preprocess), ("model", classifier)]), features


def crossfit_binary_model(
    frame: pd.DataFrame,
    labels: np.ndarray,
    phase: str,
    expanded: bool,
    folds: int,
    seed: int,
    train_mask: np.ndarray | None = None,
    stratification_labels: np.ndarray | None = None,
    fit_final: bool = True,
) -> tuple[np.ndarray, Pipeline | None, list[str]]:
    labels = np.asarray(labels, dtype=int)
    strata = labels if stratification_labels is None else np.asarray(stratification_labels, dtype=int)
    groups = frame["case_key"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    predictions = np.full(len(frame), np.nan, dtype=float)
    _, features = nuisance_specification(phase, expanded, seed)
    for fold, (train, valid) in enumerate(splitter.split(frame, strata, groups), start=1):
        if train_mask is not None:
            train = train[np.asarray(train_mask, dtype=bool)[train]]
        if len(train) == 0 or len(np.unique(labels[train])) < 2:
            raise RuntimeError(f"{phase} nuisance fold {fold} lacks both outcome classes")
        model, _ = nuisance_specification(phase, expanded, seed + fold)
        model.fit(frame.iloc[train][features], labels[train])
        predictions[valid] = model.predict_proba(frame.iloc[valid][features])[:, 1]
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"Incomplete cross-fitted predictions for {phase}")
    final_model: Pipeline | None = None
    if fit_final:
        final_model, _ = nuisance_specification(phase, expanded, seed + 100)
        final_train = np.arange(len(frame)) if train_mask is None else np.flatnonzero(train_mask)
        final_model.fit(frame.iloc[final_train][features], labels[final_train])
    return base.clip_probability(predictions), final_model, features


def observed_indicator(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["tested_7d"].fillna(False).to_numpy(dtype=bool)
        & frame["outcome_operational"].notna().to_numpy(dtype=bool)
    )


def truncate_ipw(propensity: np.ndarray, observed: np.ndarray, lower: float, upper: float) -> tuple[np.ndarray, dict[str, float]]:
    raw = 1.0 / base.clip_probability(propensity[observed])
    low_cut, high_cut = np.percentile(raw, [lower, upper])
    weights = np.clip(raw, low_cut, high_cut)
    weights *= len(observed) / weights.sum()
    diagnostics = {
        "raw_weight_min": float(raw.min()),
        "raw_weight_max": float(raw.max()),
        "truncation_low": float(low_cut),
        "truncation_high": float(high_cut),
        "weight_min": float(weights.min()),
        "weight_median": float(np.median(weights)),
        "weight_p99": float(np.quantile(weights, 0.99)),
        "weight_max": float(weights.max()),
        "effective_sample_size": float(weights.sum() ** 2 / np.sum(weights**2)),
    }
    return weights, diagnostics


def weighted_balance(frame: pd.DataFrame, observed: np.ndarray, weights: np.ndarray, phase: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in FEATURES:
        target = pd.to_numeric(frame[variable], errors="coerce")
        selected = pd.to_numeric(frame.loc[observed, variable], errors="coerce")
        valid = selected.notna().to_numpy()
        scale = float(target.std(ddof=1))
        target_mean = float(target.mean())
        observed_mean = float(selected.mean())
        weighted_mean = float(np.average(selected[valid], weights=weights[valid]))
        rows.append(
            {
                "phase": phase,
                "variable": variable,
                "level": "continuous_or_binary",
                "target_mean_or_proportion": target_mean,
                "observed_unweighted_mean_or_proportion": observed_mean,
                "observed_weighted_mean_or_proportion": weighted_mean,
                "unweighted_smd_vs_target": (observed_mean - target_mean) / scale if scale > 0 else 0.0,
                "weighted_smd_vs_target": (weighted_mean - target_mean) / scale if scale > 0 else 0.0,
            }
        )
    categorical_variables = ["asa"] + (["surgery_category"] if phase.startswith("MOVER") else [])
    for variable in categorical_variables:
        target = frame[variable].astype("object").where(frame[variable].notna(), "<missing>").astype(str)
        selected = target[observed]
        for level in sorted(target.unique()):
            target_binary = target.eq(level).astype(float)
            selected_binary = selected.eq(level).astype(float)
            target_mean = float(target_binary.mean())
            observed_mean = float(selected_binary.mean())
            weighted_mean = float(np.average(selected_binary, weights=weights))
            scale = math.sqrt(max(target_mean * (1 - target_mean), 1e-12))
            rows.append(
                {
                    "phase": phase,
                    "variable": variable,
                    "level": level,
                    "target_mean_or_proportion": target_mean,
                    "observed_unweighted_mean_or_proportion": observed_mean,
                    "observed_weighted_mean_or_proportion": weighted_mean,
                    "unweighted_smd_vs_target": (observed_mean - target_mean) / scale,
                    "weighted_smd_vs_target": (weighted_mean - target_mean) / scale,
                }
            )
    return pd.DataFrame(rows)


def observation_diagnostic_row(
    phase: str,
    specification: str,
    frame: pd.DataFrame,
    observed: np.ndarray,
    propensity: np.ndarray,
    weight_info: dict[str, float] | None,
) -> dict[str, Any]:
    tested = propensity[observed]
    untested = propensity[~observed]
    lower_support = max(float(tested.min()), float(untested.min()))
    upper_support = min(float(tested.max()), float(untested.max()))
    row: dict[str, Any] = {
        "phase": phase,
        "model_specification": specification,
        "eligible_n": len(frame),
        "observed_n": int(observed.sum()),
        "unobserved_n": int((~observed).sum()),
        "observation_rate": float(observed.mean()),
        "propensity_min": float(propensity.min()),
        "propensity_p01": float(np.quantile(propensity, 0.01)),
        "propensity_median": float(np.median(propensity)),
        "propensity_p99": float(np.quantile(propensity, 0.99)),
        "propensity_max": float(propensity.max()),
        "fraction_outside_common_support": float(
            np.mean((propensity < lower_support) | (propensity > upper_support))
        ),
        "auroc": float(roc_auc_score(observed.astype(int), propensity)),
        "brier": float(brier_score_loss(observed.astype(int), propensity)),
    }
    alpha, slope = base.fit_recalibration(observed.astype(int), propensity)
    intercept, _ = base.fit_recalibration(observed.astype(int), propensity, "intercept")
    row.update(
        {
            "calibration_intercept": intercept,
            "calibration_joint_intercept": alpha,
            "calibration_slope": slope,
        }
    )
    if weight_info:
        row.update(weight_info)
    return row


def observation_bundle(
    frame: pd.DataFrame,
    phase: str,
    config: dict[str, Any],
    seed: int,
    *,
    include_basic: bool,
    fit_outcome: bool,
    fit_final: bool,
) -> dict[str, Any]:
    observed = observed_indicator(frame)
    y = frame["outcome_operational"].fillna(0).to_numpy(dtype=float)
    diagnostics: list[dict[str, Any]] = []
    basic_propensity = None
    basic_model = None
    if include_basic:
        basic_propensity, basic_model, _ = crossfit_binary_model(
            frame, observed.astype(int), phase, False, int(config["crossfit_folds"]), seed + 10, fit_final=fit_final
        )
        diagnostics.append(
            observation_diagnostic_row(phase, "basic", frame, observed, basic_propensity, None)
        )
    expanded_propensity, expanded_model, _ = crossfit_binary_model(
        frame, observed.astype(int), phase, True, int(config["crossfit_folds"]), seed + 20, fit_final=fit_final
    )
    lower, upper = config["observation_models"]["ipw_truncation_percentiles"]
    weights, weight_info = truncate_ipw(expanded_propensity, observed, float(lower), float(upper))
    diagnostics.append(
        observation_diagnostic_row(phase, "expanded", frame, observed, expanded_propensity, weight_info)
    )
    outcome_probability = None
    outcome_model = None
    if fit_outcome:
        strata = np.where(observed, 1 + y.astype(int), 0)
        outcome_probability, outcome_model, _ = crossfit_binary_model(
            frame,
            y.astype(int),
            phase,
            True,
            int(config["crossfit_folds"]),
            seed + 30,
            train_mask=observed,
            stratification_labels=strata,
            fit_final=fit_final,
        )
    return {
        "observed": observed,
        "y": y,
        "basic_propensity": basic_propensity,
        "expanded_propensity": expanded_propensity,
        "weights": weights,
        "outcome_probability": outcome_probability,
        "basic_model": basic_model,
        "expanded_model": expanded_model,
        "outcome_model": outcome_model,
        "diagnostics": pd.DataFrame(diagnostics),
        "balance": weighted_balance(frame, observed, weights, phase),
    }


def prepare_cohorts() -> dict[str, pd.DataFrame]:
    inspire = pd.read_parquet(DATA_DIR / "inspire_rebuilt_v2.parquet")
    mover = pd.read_parquet(DATA_DIR / "mover_rebuilt_v2.parquet")
    vital = pd.read_parquet(DATA_DIR / "vitaldb_supportive_v2.parquet")
    source = base.eligible_base(inspire).reset_index(drop=True)
    update = base.eligible_base(mover, 2021).reset_index(drop=True)
    target = base.eligible_base(mover, 2022).reset_index(drop=True)
    for frame, label in [(source, "INSPIRE"), (update, "MOVER 2021"), (target, "MOVER 2022")]:
        frame["analysis_phase"] = label
        if "surgery_category" not in frame:
            frame["surgery_category"] = "not_available"
        frame["surgery_category"] = frame["surgery_category"].fillna("other").astype(str)
    vital_observed = vital.loc[
        vital["has_baseline_cr"].fillna(False).astype(bool)
        & vital["baseline_cr_under4"].fillna(False).astype(bool)
        & vital["tested_7d"].fillna(False).astype(bool)
        & vital["aki"].notna()
    ].copy().reset_index(drop=True)
    vital_observed["outcome_operational"] = vital_observed["aki"]
    return {"source": source, "update": update, "target": target, "vital_observed": vital_observed}


def cohort_flow_table(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, role in [
        ("source", "MAR source development"),
        ("update", "MAR local recalibration"),
        ("target", "post-exploration temporal evaluation"),
    ]:
        frame = cohorts[key]
        observed = observed_indicator(frame)
        rows.append(
            {
                "cohort": frame["analysis_phase"].iloc[0],
                "role": role,
                "eligible_n": len(frame),
                "observed_outcome_n": int(observed.sum()),
                "unobserved_outcome_n": int((~observed).sum()),
                "outcome_observation_rate": float(observed.mean()),
                "observed_events": int(frame.loc[observed, "outcome_operational"].sum()),
                "retained_patient_overlap_with_other_MOVER_year": (
                    0 if key in {"update", "target"} else np.nan
                ),
                "denominator_note": "all baseline-eligible first retained operations",
            }
        )
    vital = cohorts["vital_observed"]
    rows.append(
        {
            "cohort": "VitalDB available analysis-ready observed cohort",
            "role": "supportive observed-cohort stress test",
            "eligible_n": np.nan,
            "observed_outcome_n": len(vital),
            "unobserved_outcome_n": np.nan,
            "outcome_observation_rate": np.nan,
            "observed_events": int(vital["outcome_operational"].sum()),
            "retained_patient_overlap_with_other_MOVER_year": np.nan,
            "denominator_note": "complete source-eligible denominator cannot be reconstructed from the retained analysis-ready cache",
        }
    )
    return pd.DataFrame(rows)


def group_characteristics(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key in ["source", "update", "target"]:
        frame = cohorts[key]
        observed = observed_indicator(frame)
        for group, mask in [
            ("all eligible", np.ones(len(frame), dtype=bool)),
            ("outcome observed", observed),
            ("outcome unobserved", ~observed),
        ]:
            part = frame.loc[mask]
            rows.append(
                {
                    "phase": frame["analysis_phase"].iloc[0],
                    "group": group,
                    "n": len(part),
                    "age_mean": part["age"].mean(),
                    "age_sd": part["age"].std(ddof=1),
                    "male_percent": 100 * part["sex_male"].mean(),
                    "asa_mean": part["asa"].mean(),
                    "duration_h_mean": part["duration_h"].mean(),
                    "duration_h_sd": part["duration_h"].std(ddof=1),
                    "baseline_cr_mean": part["baseline_cr"].mean(),
                    "baseline_cr_sd": part["baseline_cr"].std(ddof=1),
                    "observed_events": (
                        float(part["outcome_operational"].sum())
                        if group == "outcome observed"
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def missingness_table(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    methods = {
        "age": "median imputation",
        "sex_male": "most-frequent imputation",
        "duration_h": "median imputation",
        "baseline_cr": "eligibility requires observed baseline creatinine",
        "asa": "most-frequent category in expanded nuisance models; not used by audit risk model",
        "surgery_category": "most-frequent category in MOVER nuisance models; unavailable in INSPIRE",
        "outcome_operational": "not imputed as observed truth; addressed with IPW/AIPW and MNAR scenarios",
    }
    for key in ["source", "update", "target"]:
        frame = cohorts[key]
        for variable, method in methods.items():
            if variable not in frame:
                missing_n = len(frame)
            else:
                missing_n = int(frame[variable].isna().sum())
            rows.append(
                {
                    "phase": frame["analysis_phase"].iloc[0],
                    "variable": variable,
                    "eligible_n": len(frame),
                    "missing_n": missing_n,
                    "missing_percent": 100 * missing_n / len(frame),
                    "handling": method,
                }
            )
    return pd.DataFrame(rows)


def fit_source_model(frame: pd.DataFrame, bundle: dict[str, Any], weighted: bool, seed: int) -> Pipeline:
    observed = bundle["observed"]
    model = make_risk_model(seed)
    kwargs = {"model__sample_weight": bundle["weights"]} if weighted else {}
    model.fit(
        frame.loc[observed, FEATURES],
        frame.loc[observed, "outcome_operational"].astype(int),
        **kwargs,
    )
    return model


def ipw_full_metrics(frame: pd.DataFrame, prediction: np.ndarray, bundle: dict[str, Any]) -> dict[str, float]:
    observed = bundle["observed"]
    y = bundle["y"][observed]
    weights = bundle["weights"]
    metric = base.performance_metrics(y, prediction[observed], weights)
    event_total = float(np.sum(weights * y))
    metric["n"] = float(len(frame))
    metric["weighted_n"] = float(len(frame))
    metric["events"] = event_total
    metric["event_rate"] = event_total / len(frame)
    metric["predicted_mean"] = float(np.mean(prediction))
    metric["oe_ratio"] = event_total / float(np.sum(prediction))
    return metric


def aipw_pseudo_outcome(bundle: dict[str, Any]) -> np.ndarray:
    observed = bundle["observed"]
    y = bundle["y"]
    propensity = base.clip_probability(bundle["expanded_propensity"])
    outcome_probability = base.clip_probability(bundle["outcome_probability"])
    return outcome_probability + observed * (y - outcome_probability) / propensity


def aipw_hybrid_metrics(frame: pd.DataFrame, prediction: np.ndarray, bundle: dict[str, Any]) -> dict[str, float]:
    metric = ipw_full_metrics(frame, prediction, bundle)
    pseudo_y = aipw_pseudo_outcome(bundle)
    event_total = float(np.sum(pseudo_y))
    event_rate = event_total / len(frame)
    observed = bundle["observed"]
    y = bundle["y"]
    propensity = base.clip_probability(bundle["expanded_propensity"])
    outcome_probability = base.clip_probability(bundle["outcome_probability"])
    observed_loss = y * (1 - prediction) ** 2 + (1 - y) * prediction**2
    modeled_loss = outcome_probability * (1 - prediction) ** 2 + (1 - outcome_probability) * prediction**2
    pseudo_loss = modeled_loss + observed * (observed_loss - modeled_loss) / propensity
    brier = float(np.mean(pseudo_loss))
    reference = event_rate * (1 - event_rate)
    metric.update(
        {
            "n": float(len(frame)),
            "weighted_n": float(len(frame)),
            "events": event_total,
            "event_rate": event_rate,
            "predicted_mean": float(np.mean(prediction)),
            "oe_ratio": event_total / float(np.sum(prediction)),
            "brier": brier,
            "brier_skill_score": 1 - brier / reference if reference > 0 else np.nan,
        }
    )
    return metric


def target_workload(prediction: np.ndarray, bundle: dict[str, Any], thresholds: Iterable[float]) -> list[dict[str, float]]:
    pseudo_y = aipw_pseudo_outcome(bundle)
    rows = []
    for threshold in thresholds:
        metric = base.workload_metrics(pseudo_y, prediction, float(threshold))
        metric["alerts"] = float(np.sum(prediction >= float(threshold)))
        metric["alerts_per_1000"] = 1000 * metric["alerts"] / len(prediction)
        rows.append({"threshold": float(threshold), **metric})
    return rows


def update_parameters(
    source_models: dict[str, Pipeline],
    update: pd.DataFrame,
    bundle: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    observed = bundle["observed"]
    y_observed = bundle["y"][observed]
    rows: list[dict[str, Any]] = []
    strategies: list[dict[str, Any]] = []

    for source_label, model in source_models.items():
        prediction = model.predict_proba(update[FEATURES])[:, 1]
        alpha_cc, beta_cc = base.fit_recalibration(y_observed, prediction[observed])
        rows.append(
            {
                "source_model_estimand": source_label,
                "update_estimand": "conditional on observed postoperative creatinine",
                "update_estimator": "complete-case logistic recalibration",
                "eligible_n": len(update),
                "observed_n": int(observed.sum()),
                "unobserved_n": int((~observed).sum()),
                "alpha": alpha_cc,
                "beta": beta_cc,
                "weight_rule": "none",
                "mnar_odds_multiplier": np.nan,
            }
        )
        strategies.append(
            {
                "strategy": f"{source_label} + observed-outcome update",
                "source_label": source_label,
                "source_model": model,
                "update_estimator": "complete case",
                "alpha": alpha_cc,
                "beta": beta_cc,
                "primary": False,
            }
        )
        alpha_ipw, beta_ipw = base.fit_recalibration(
            y_observed, prediction[observed], weights=bundle["weights"]
        )
        rows.append(
            {
                "source_model_estimand": source_label,
                "update_estimand": "all eligible MOVER 2021 under MAR",
                "update_estimator": "expanded cross-fitted IPW logistic recalibration",
                "eligible_n": len(update),
                "observed_n": int(observed.sum()),
                "unobserved_n": int((~observed).sum()),
                "alpha": alpha_ipw,
                "beta": beta_ipw,
                "weight_rule": "inverse observation probability truncated at 1st/99th percentiles",
                "mnar_odds_multiplier": np.nan,
            }
        )
        is_primary = source_label == "expanded-IPW all-eligible source model"
        strategies.append(
            {
                "strategy": f"{source_label} + IPW all-eligible update",
                "source_label": source_label,
                "source_model": model,
                "update_estimator": "IPW",
                "alpha": alpha_ipw,
                "beta": beta_ipw,
                "primary": is_primary,
            }
        )

    primary_model = source_models["expanded-IPW all-eligible source model"]
    primary_prediction = primary_model.predict_proba(update[FEATURES])[:, 1]
    pseudo_y = aipw_pseudo_outcome(bundle)
    alpha_aipw, beta_aipw = base.fit_recalibration(pseudo_y, primary_prediction)
    rows.append(
        {
            "source_model_estimand": "expanded-IPW all-eligible source model",
            "update_estimand": "all eligible MOVER 2021 under MAR",
            "update_estimator": "augmented estimating-equation recalibration sensitivity",
            "eligible_n": len(update),
            "observed_n": int(observed.sum()),
            "unobserved_n": int((~observed).sum()),
            "alpha": alpha_aipw,
            "beta": beta_aipw,
            "weight_rule": "untruncated cross-fitted 1/g correction in pseudo-outcome",
            "mnar_odds_multiplier": np.nan,
        }
    )
    strategies.append(
        {
            "strategy": "expanded-IPW all-eligible source model + augmented update sensitivity",
            "source_label": "expanded-IPW all-eligible source model",
            "source_model": primary_model,
            "update_estimator": "AIPW sensitivity",
            "alpha": alpha_aipw,
            "beta": beta_aipw,
            "primary": False,
        }
    )
    for multiplier in config["mnar_odds_multipliers"]:
        shifted = expit(logit(base.clip_probability(bundle["outcome_probability"])) + math.log(float(multiplier)))
        fractional_y = np.where(observed, bundle["y"], shifted)
        alpha, beta = base.fit_recalibration(fractional_y, primary_prediction)
        rows.append(
            {
                "source_model_estimand": "expanded-IPW all-eligible source model",
                "update_estimand": "all eligible MOVER 2021 under selective MNAR scenario",
                "update_estimator": "pattern-mixture fractional-outcome recalibration",
                "eligible_n": len(update),
                "observed_n": int(observed.sum()),
                "unobserved_n": int((~observed).sum()),
                "alpha": alpha,
                "beta": beta,
                "weight_rule": "outcomes observed as recorded; unobserved outcome odds shifted",
                "mnar_odds_multiplier": multiplier,
            }
        )
        strategies.append(
            {
                "strategy": f"expanded-IPW source + update MNAR scenario {multiplier:.3g}",
                "source_label": "expanded-IPW all-eligible source model",
                "source_model": primary_model,
                "update_estimator": f"MNAR {multiplier:.3g}",
                "alpha": alpha,
                "beta": beta,
                "primary": False,
            }
        )
    return pd.DataFrame(rows), strategies


def evaluate_strategies(
    strategies: list[dict[str, Any]],
    target: pd.DataFrame,
    bundle: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    performance_rows: list[dict[str, Any]] = []
    workload_rows: list[dict[str, Any]] = []
    primary_result: dict[str, Any] | None = None
    observed = bundle["observed"]
    for item in strategies:
        source_prediction = item["source_model"].predict_proba(target[FEATURES])[:, 1]
        prediction = base.apply_recalibration(source_prediction, item["alpha"], item["beta"])
        estimators = [
            (
                "conditional complete case",
                base.performance_metrics(bundle["y"][observed], prediction[observed]),
                "all metrics condition on routine postoperative creatinine observation",
            ),
            (
                "expanded IPW all-eligible MAR",
                ipw_full_metrics(target, prediction, bundle),
                "all metrics use expanded IPW; expected predictions use all eligible patients",
            ),
            (
                "canonical hybrid AIPW/IPW all-eligible MAR",
                aipw_hybrid_metrics(target, prediction, bundle),
                "event rate, O/E, Brier and workload are AIPW; calibration and discrimination are expanded IPW",
            ),
        ]
        alerts = {
            f"alerts_at_{int(round(threshold * 100))}_percent": int(np.sum(prediction >= threshold))
            for threshold in config["decision_thresholds"]
        }
        for estimator, metric, note in estimators:
            performance_rows.append(
                {
                    "strategy": item["strategy"],
                    "source_model_estimand": item["source_label"],
                    "update_estimator": item["update_estimator"],
                    "target_estimand": estimator,
                    "primary_strategy": item["primary"],
                    "metric_specific_note": note,
                    **metric,
                    **alerts,
                }
            )
        workload = target_workload(prediction, bundle, config["decision_thresholds"])
        for row in workload:
            workload_rows.append(
                {
                    "strategy": item["strategy"],
                    "primary_strategy": item["primary"],
                    "target_estimand": "AIPW linear workload under MAR; exact alert count",
                    **row,
                }
            )
        if item["primary"]:
            primary_result = {
                "strategy": item,
                "prediction": prediction,
                "metrics": aipw_hybrid_metrics(target, prediction, bundle),
                "workload": workload,
            }
    if primary_result is None:
        raise RuntimeError("Primary strategy was not generated")
    return pd.DataFrame(performance_rows), pd.DataFrame(workload_rows), primary_result


def target_mnar_and_bounds(
    target: pd.DataFrame,
    bundle: dict[str, Any],
    primary: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Propagate target-outcome MNAR scenarios without treating them as identified estimates."""
    observed = bundle["observed"]
    y = bundle["y"]
    outcome_probability = bundle["outcome_probability"]
    prediction = primary["prediction"]
    mnar_rows: list[dict[str, Any]] = []
    for multiplier in config["mnar_odds_multipliers"]:
        shifted = expit(
            logit(base.clip_probability(outcome_probability)) + math.log(float(multiplier))
        )
        fractional_y = np.where(observed, y, shifted)
        metrics = base.performance_metrics(fractional_y, prediction)
        row: dict[str, Any] = {
            "target_estimand": "MOVER 2022 all eligible under a selective target-outcome MNAR scenario",
            "target_mnar_odds_multiplier": multiplier,
            "eligible_n": len(target),
            "observed_n": int(observed.sum()),
            "unobserved_n": int((~observed).sum()),
            "interpretation": "pattern-mixture sensitivity scenario, not an identified full-population truth",
            **metrics,
        }
        for threshold in config["decision_thresholds"]:
            workload = base.workload_metrics(fractional_y, prediction, float(threshold))
            label = int(round(float(threshold) * 100))
            for key, value in workload.items():
                row[f"threshold_{label}_{key}"] = value
            row[f"threshold_{label}_alerts"] = int(np.sum(prediction >= float(threshold)))
        mnar_rows.append(row)

    bound_rows: list[dict[str, Any]] = []
    for label, fill in [
        ("all unobserved outcomes are non-events", 0.0),
        ("all unobserved outcomes are events", 1.0),
    ]:
        completed_y = np.where(observed, y, fill)
        metrics = base.performance_metrics(completed_y, prediction)
        row = {
            "bound_scenario": label,
            "unobserved_outcome_value": fill,
            "eligible_n": len(target),
            "observed_n": int(observed.sum()),
            "unobserved_n": int((~observed).sum()),
            **metrics,
        }
        for threshold in config["decision_thresholds"]:
            workload = base.workload_metrics(completed_y, prediction, float(threshold))
            threshold_label = int(round(float(threshold) * 100))
            row[f"threshold_{threshold_label}_positive_predictive_value"] = workload[
                "positive_predictive_value"
            ]
            row[f"threshold_{threshold_label}_net_benefit"] = workload["net_benefit"]
        bound_rows.append(row)
    return pd.DataFrame(mnar_rows), pd.DataFrame(bound_rows)


def vitaldb_supportive_stress_test(
    vital: pd.DataFrame,
    primary: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate the v3.1 source and MOVER update in the retained observed VitalDB cohort."""
    strategy = primary["strategy"]
    source_prediction = strategy["source_model"].predict_proba(vital[FEATURES])[:, 1]
    updated_prediction = base.apply_recalibration(
        source_prediction, strategy["alpha"], strategy["beta"]
    )
    outcome = pd.to_numeric(vital["outcome_operational"], errors="coerce").to_numpy(dtype=int)
    rows = []
    for model_label, prediction in [
        ("expanded-IPW INSPIRE source model", source_prediction),
        ("source model plus expanded-IPW MOVER 2021 recalibration", updated_prediction),
    ]:
        rows.append(
            {
                "cohort_label": "VitalDB available analysis-ready observed cohort",
                "model": model_label,
                "n": len(vital),
                "events": int(outcome.sum()),
                "interpretation": "supportive observed-cohort stress test; not an independent validation and not a complete eligible-population analysis",
                **base.performance_metrics(outcome, prediction),
            }
        )
    return pd.DataFrame(rows)


def model_specification(
    model: Pipeline,
    alpha: float,
    beta: float,
    cohorts: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, Any]:
    preprocess = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    continuous = preprocess.named_transformers_["continuous"]
    binary = preprocess.named_transformers_["binary"]
    feature_names = preprocess.get_feature_names_out().tolist()
    continuous_metadata = []
    for index, feature in enumerate(CONTINUOUS):
        continuous_metadata.append(
            {
                "feature": feature,
                "imputation_value": float(continuous.named_steps["impute"].statistics_[index]),
                "standardization_mean": float(continuous.named_steps["scale"].mean_[index]),
                "standardization_scale": float(continuous.named_steps["scale"].scale_[index]),
                "spline_knot_vector_standardized": continuous.named_steps["spline"].bsplines_[index].t.tolist(),
            }
        )
    return {
        "study_id": config["study_id"],
        "analysis_version": config["analysis_version"],
        "clinical_use": "not for clinical use",
        "source_estimand": config["estimands"]["source_development"],
        "update_estimand": config["estimands"]["local_update"],
        "target_estimand": config["estimands"]["target_evaluation"],
        "features_in_order": FEATURES,
        "transformed_feature_names": feature_names,
        "source_intercept": float(classifier.intercept_[0]),
        "source_coefficients": [float(value) for value in classifier.coef_[0]],
        "continuous_preprocessing": continuous_metadata,
        "binary_preprocessing": {
            "feature": "sex_male",
            "encoding": "1=male; 0=female as released",
            "imputation": "most_frequent",
            "imputation_value": float(binary.named_steps["impute"].statistics_[0]),
        },
        "estimator_parameters": {
            "C": classifier.C,
            "penalty": classifier.penalty,
            "solver": classifier.solver,
            "max_iter": classifier.max_iter,
            "spline_n_knots": continuous.named_steps["spline"].n_knots,
            "spline_degree": continuous.named_steps["spline"].degree,
            "spline_include_bias": continuous.named_steps["spline"].include_bias,
            "spline_extrapolation": continuous.named_steps["spline"].extrapolation,
        },
        "recalibration": {
            "method": "expanded cross-fitted IPW logistic recalibration",
            "alpha": alpha,
            "beta": beta,
            "equation": "expit(alpha + beta * logit(source_probability))",
        },
        "cohort_denominators": {
            "INSPIRE": {"eligible": len(cohorts["source"]), "observed": int(observed_indicator(cohorts["source"]).sum())},
            "MOVER_2021": {"eligible": len(cohorts["update"]), "observed": int(observed_indicator(cohorts["update"]).sum())},
            "MOVER_2022": {"eligible": len(cohorts["target"]), "observed": int(observed_indicator(cohorts["target"]).sum())},
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "joblib": joblib.__version__,
        },
    }


def bootstrap_once(
    source: pd.DataFrame,
    update: pd.DataFrame,
    target: pd.DataFrame,
    config: dict[str, Any],
    replicate: int,
    attempt: int,
) -> dict[str, float]:
    seed = int(config["random_seed"]) + 100_000 + replicate * 100 + attempt
    rng = np.random.default_rng(seed)
    source_sample = source.iloc[rng.integers(0, len(source), len(source))].reset_index(drop=True)
    update_sample = update.iloc[rng.integers(0, len(update), len(update))].reset_index(drop=True)
    target_sample = target.iloc[rng.integers(0, len(target), len(target))].reset_index(drop=True)

    source_bundle = observation_bundle(
        source_sample, "INSPIRE", config, seed + 1, include_basic=False, fit_outcome=False, fit_final=False
    )
    source_model = fit_source_model(source_sample, source_bundle, True, seed + 2)

    update_bundle = observation_bundle(
        update_sample, "MOVER 2021", config, seed + 3, include_basic=False, fit_outcome=False, fit_final=False
    )
    update_prediction = source_model.predict_proba(update_sample[FEATURES])[:, 1]
    update_observed = update_bundle["observed"]
    alpha, beta = base.fit_recalibration(
        update_bundle["y"][update_observed],
        update_prediction[update_observed],
        weights=update_bundle["weights"],
    )

    target_bundle = observation_bundle(
        target_sample, "MOVER 2022", config, seed + 4, include_basic=False, fit_outcome=True, fit_final=False
    )
    target_prediction = base.apply_recalibration(
        source_model.predict_proba(target_sample[FEATURES])[:, 1], alpha, beta
    )
    metrics = aipw_hybrid_metrics(target_sample, target_prediction, target_bundle)
    workload = target_workload(target_prediction, target_bundle, config["decision_thresholds"])
    row: dict[str, float] = {
        "replicate": float(replicate),
        "attempt": float(attempt),
        "update_alpha": float(alpha),
        "update_beta": float(beta),
        **{metric: float(metrics[metric]) for metric in PERFORMANCE_METRICS},
    }
    for item in workload:
        label = int(round(item["threshold"] * 100))
        for metric in [
            "alerts",
            "alerts_per_1000",
            "true_positives_per_1000",
            "false_positives_per_1000",
            "sensitivity",
            "positive_predictive_value",
            "false_positives_per_true_positive",
            "net_benefit",
            "treat_all_net_benefit",
            "net_benefit_vs_better_default",
        ]:
            row[f"threshold_{label}_{metric}"] = float(item[metric])
    numeric = np.asarray([value for key, value in row.items() if key not in {"replicate", "attempt"}], dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Non-finite canonical bootstrap metric")
    return row


def bootstrap_worker_init(config: dict[str, Any]) -> None:
    global _WORKER_COHORTS, _WORKER_CONFIG
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _WORKER_CONFIG = config
    _WORKER_COHORTS = prepare_cohorts()


def bootstrap_worker_run(replicate: int) -> dict[str, float]:
    if _WORKER_COHORTS is None or _WORKER_CONFIG is None:
        raise RuntimeError("Bootstrap worker was not initialized")
    last_error = ""
    for attempt in range(1, 6):
        try:
            return bootstrap_once(
                _WORKER_COHORTS["source"],
                _WORKER_COHORTS["update"],
                _WORKER_COHORTS["target"],
                _WORKER_CONFIG,
                replicate,
                attempt,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            last_error = str(exc)
    raise RuntimeError(f"Replicate {replicate} failed after five attempts: {last_error}")


def canonical_bootstrap(
    cohorts: dict[str, pd.DataFrame],
    config: dict[str, Any],
    replicates: int,
    output_dir: Path,
    jobs: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "canonical_bootstrap_distribution_v31.partial.parquet"
    rows: list[dict[str, float]] = []
    if partial_path.exists() and output_dir == DATA_DIR:
        prior = pd.read_parquet(partial_path)
        rows = prior.to_dict(orient="records")
    completed = {int(row["replicate"]) for row in rows}
    started = time.time()
    pending = [replicate for replicate in range(1, replicates + 1) if replicate not in completed]
    if jobs <= 1:
        bootstrap_worker_init(config)
        for replicate in pending:
            rows.append(bootstrap_worker_run(replicate))
            if len(rows) % 10 == 0 or len(rows) == replicates:
                frame = pd.DataFrame(rows).sort_values("replicate")
                frame.to_parquet(partial_path, index=False)
                elapsed = time.time() - started
                print(f"Canonical bootstrap {len(frame)}/{replicates}; elapsed {elapsed:.1f}s", flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=context,
            initializer=bootstrap_worker_init,
            initargs=(config,),
        ) as executor:
            futures = {executor.submit(bootstrap_worker_run, replicate): replicate for replicate in pending}
            for future in as_completed(futures):
                rows.append(future.result())
                if len(rows) % 10 == 0 or len(rows) == replicates:
                    frame = pd.DataFrame(rows).sort_values("replicate")
                    frame.to_parquet(partial_path, index=False)
                    elapsed = time.time() - started
                    print(f"Canonical bootstrap {len(frame)}/{replicates}; elapsed {elapsed:.1f}s", flush=True)
    result = pd.DataFrame(rows).sort_values("replicate").reset_index(drop=True)
    if result["replicate"].nunique() != replicates:
        raise RuntimeError("Canonical bootstrap did not produce the requested valid replicates")
    final_path = output_dir / "canonical_bootstrap_distribution_v31.parquet"
    result.to_parquet(final_path, index=False)
    if partial_path.exists():
        partial_path.unlink()
    return result


def canonical_long_table(
    primary: dict[str, Any],
    bootstrap: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_sources = {
        "events": "AIPW",
        "event_rate": "AIPW",
        "predicted_mean": "exact all-eligible predictions",
        "oe_ratio": "AIPW observed events / exact predicted events",
        "calibration_intercept": "expanded IPW",
        "calibration_joint_intercept": "expanded IPW",
        "calibration_slope": "expanded IPW",
        "auroc": "expanded IPW",
        "auprc": "expanded IPW",
        "brier": "AIPW",
        "brier_skill_score": "AIPW",
        "grouped_ici": "expanded IPW",
    }
    for metric in PERFORMANCE_METRICS:
        lower, upper = bootstrap[metric].quantile([0.025, 0.975]).tolist()
        rows.append(
            {
                "endpoint": config["primary_endpoint"],
                "estimand": config["estimands"]["target_evaluation"],
                "model": config["canonical_analysis"]["model"],
                "metric": metric,
                "estimate": primary["metrics"][metric],
                "ci_lower": lower,
                "ci_upper": upper,
                "estimator_component": metric_sources[metric],
                "bootstrap_replicates": len(bootstrap),
                "bootstrap_unit": config["canonical_analysis"]["bootstrap_unit"],
                "canonical_source": "data/processed/canonical_bootstrap_distribution_v31.parquet",
            }
        )
    rows.extend(
        [
            {
                "endpoint": config["primary_endpoint"],
                "estimand": config["estimands"]["local_update"],
                "model": config["canonical_analysis"]["model"],
                "metric": metric,
                "estimate": primary["strategy"][point_key],
                "ci_lower": bootstrap[key].quantile(0.025),
                "ci_upper": bootstrap[key].quantile(0.975),
                "estimator_component": "expanded IPW update estimating equation",
                "bootstrap_replicates": len(bootstrap),
                "bootstrap_unit": config["canonical_analysis"]["bootstrap_unit"],
                "canonical_source": "data/processed/canonical_bootstrap_distribution_v31.parquet",
            }
            for metric, key, point_key in [
                ("update_alpha", "update_alpha", "alpha"),
                ("update_beta", "update_beta", "beta"),
            ]
        ]
    )
    for item in primary["workload"]:
        label = int(round(item["threshold"] * 100))
        for metric in [
            "alerts",
            "alerts_per_1000",
            "true_positives_per_1000",
            "false_positives_per_1000",
            "sensitivity",
            "positive_predictive_value",
            "false_positives_per_true_positive",
            "net_benefit",
            "treat_all_net_benefit",
            "net_benefit_vs_better_default",
        ]:
            column = f"threshold_{label}_{metric}"
            lower, upper = bootstrap[column].quantile([0.025, 0.975]).tolist()
            rows.append(
                {
                    "endpoint": config["primary_endpoint"],
                    "estimand": f"threshold workload in all eligible MOVER 2022 patients at {label}%",
                    "model": config["canonical_analysis"]["model"],
                    "metric": metric,
                    "estimate": item[metric],
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "estimator_component": (
                        "exact all-eligible predictions" if metric in {"alerts", "alerts_per_1000"} else "AIPW"
                    ),
                    "bootstrap_replicates": len(bootstrap),
                    "bootstrap_unit": config["canonical_analysis"]["bootstrap_unit"],
                    "canonical_source": "data/processed/canonical_bootstrap_distribution_v31.parquet",
                }
            )
    return pd.DataFrame(rows)


def run_point_analysis(cohorts: dict[str, pd.DataFrame], config: dict[str, Any], write_outputs: bool) -> dict[str, Any]:
    seed = int(config["random_seed"])
    source_bundle = observation_bundle(
        cohorts["source"], "INSPIRE", config, seed + 1000, include_basic=True, fit_outcome=True, fit_final=True
    )
    update_bundle = observation_bundle(
        cohorts["update"], "MOVER 2021", config, seed + 2000, include_basic=True, fit_outcome=True, fit_final=True
    )
    target_bundle = observation_bundle(
        cohorts["target"], "MOVER 2022", config, seed + 3000, include_basic=True, fit_outcome=True, fit_final=True
    )
    source_models = {
        "observed-outcome source model": fit_source_model(cohorts["source"], source_bundle, False, seed + 4000),
        "expanded-IPW all-eligible source model": fit_source_model(cohorts["source"], source_bundle, True, seed + 4001),
    }
    recalibration, strategies = update_parameters(source_models, cohorts["update"], update_bundle, config)
    performance, workload, primary = evaluate_strategies(strategies, cohorts["target"], target_bundle, config)
    target_mnar, target_bounds = target_mnar_and_bounds(
        cohorts["target"], target_bundle, primary, config
    )
    vitaldb = vitaldb_supportive_stress_test(cohorts["vital_observed"], primary)

    if write_outputs:
        flow = cohort_flow_table(cohorts)
        characteristics = group_characteristics(cohorts)
        diagnostics = pd.concat(
            [source_bundle["diagnostics"], update_bundle["diagnostics"], target_bundle["diagnostics"]],
            ignore_index=True,
        )
        balance = pd.concat(
            [source_bundle["balance"], update_bundle["balance"], target_bundle["balance"]],
            ignore_index=True,
        )
        write_csv(flow, "01_cohort_flow_and_roles_v31.csv")
        write_csv(characteristics, "27_source_update_observation_flow_v31.csv")
        write_csv(diagnostics, "28_stage_observation_diagnostics_v31.csv")
        write_csv(recalibration, "29_mover2021_recalibration_estimands_v31.csv")
        write_csv(performance, "30_update_selection_target_performance_v31.csv")
        write_csv(balance, "32_stage_observation_balance_v31.csv")
        write_csv(missingness_table(cohorts), "33_predictor_missingness_v31.csv")
        write_csv(workload, "34_update_strategy_workload_v31.csv")
        write_csv(target_mnar, "40_target_mnar_sensitivity_v31.csv")
        write_csv(target_bounds, "41_target_nonparametric_bounds_v31.csv")
        write_csv(vitaldb, "42_vitaldb_supportive_v31.csv")
        for label, model in source_models.items():
            filename = (
                "source_logistic_spline_ipw_v31.joblib"
                if label.startswith("expanded-IPW")
                else "source_logistic_spline_observed_v31.joblib"
            )
            joblib.dump(model, MODEL_DIR / filename)
        for phase, bundle in [("source", source_bundle), ("update", update_bundle), ("target", target_bundle)]:
            joblib.dump(bundle["expanded_model"], MODEL_DIR / f"{phase}_observation_model_expanded_v31.joblib")
            joblib.dump(bundle["outcome_model"], MODEL_DIR / f"{phase}_auxiliary_outcome_model_v31.joblib")
        spec = model_specification(
            source_models["expanded-IPW all-eligible source model"],
            primary["strategy"]["alpha"],
            primary["strategy"]["beta"],
            cohorts,
            config,
        )
        write_json(spec, MODEL_DIR / "model_specification_v31.json")
    return {
        "source_bundle": source_bundle,
        "update_bundle": update_bundle,
        "target_bundle": target_bundle,
        "source_models": source_models,
        "recalibration": recalibration,
        "performance": performance,
        "workload": workload,
        "primary": primary,
        "target_mnar": target_mnar,
        "target_bounds": target_bounds,
        "vitaldb": vitaldb,
    }


def write_run_reports(config: dict[str, Any], canonical: pd.DataFrame, started: datetime, elapsed: float) -> None:
    oe = canonical.loc[canonical["metric"].eq("oe_ratio")].iloc[0]
    slope = canonical.loc[canonical["metric"].eq("calibration_slope")].iloc[0]
    report = f"""# v3.1 canonical analysis result

Status: **NOT READY FOR SUBMISSION**. Scientific P0 analyses have been rerun, but author and institutional confirmations remain outstanding.

The primary estimand follows selective outcome observation through source development, MOVER 2021 updating, and MOVER 2022 evaluation. The canonical chain uses expanded cross-fitted inverse-probability weighting for source-model development and local recalibration. In MOVER 2022, event rate, O/E, Brier score, and workload use augmented inverse-probability estimation; calibration slope and discrimination use expanded IPW.

The canonical all-eligible O/E estimate is {oe['estimate']:.3f} (95% bootstrap interval {oe['ci_lower']:.3f} to {oe['ci_upper']:.3f}); the calibration slope is {slope['estimate']:.3f} ({slope['ci_lower']:.3f} to {slope['ci_upper']:.3f}). These are assumption-dependent MAR estimates, not identified full-population truths.

The {int(config['bootstrap_replicates'])} valid patient-level bootstrap replicates independently resampled INSPIRE, MOVER 2021, and MOVER 2022 and refitted observation models, the source model, local recalibration, and the target auxiliary outcome model.
"""
    (REPORT_DIR / "10_v31_canonical_results.md").write_text(report, encoding="utf-8")
    write_json(
        {
            "analysis_version": config["analysis_version"],
            "status": config["analysis_status"],
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": elapsed,
            "bootstrap_replicates": int(config["bootstrap_replicates"]),
            "random_seed": int(config["random_seed"]),
            "canonical_table_sha256": sha256_file(TABLE_DIR / "31_canonical_primary_bootstrap_v31.csv"),
            "canonical_distribution_sha256": sha256_file(DATA_DIR / "canonical_bootstrap_distribution_v31.parquet"),
        },
        REPORT_DIR / "11_v31_analysis_run_record.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--jobs", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    if args.bootstrap_replicates is not None:
        if args.bootstrap_replicates < 5:
            raise ValueError("At least five bootstrap replicates are required")
        config["bootstrap_replicates"] = int(args.bootstrap_replicates)
    for directory in [TABLE_DIR, MODEL_DIR, REPORT_DIR, QA_DIR, DATA_DIR, TMP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    clock = time.time()
    print("Loading complete eligible cohorts", flush=True)
    cohorts = prepare_cohorts()
    overlap = len(set(cohorts["update"]["patient_key"]) & set(cohorts["target"]["patient_key"]))
    if overlap != 0:
        raise RuntimeError(f"Expected zero retained MOVER patient overlap, found {overlap}")
    print("Fitting point-estimate source, update, and target observation chains", flush=True)
    point = run_point_analysis(cohorts, config, write_outputs=not args.diagnostic)
    output_dir = TMP_DIR if args.diagnostic else DATA_DIR
    print(f"Running {config['bootstrap_replicates']} canonical bootstrap replicates", flush=True)
    jobs = int(args.jobs or config["canonical_analysis"].get("parallel_workers", 1))
    bootstrap = canonical_bootstrap(
        cohorts, config, int(config["bootstrap_replicates"]), output_dir, jobs
    )
    canonical = canonical_long_table(point["primary"], bootstrap, config)
    if args.diagnostic:
        write_csv(canonical, "31_canonical_primary_bootstrap_v31_diagnostic.csv", TMP_DIR)
        print(canonical.loc[canonical["metric"].isin(["oe_ratio", "calibration_slope", "brier"])] .to_json(orient="records", indent=2))
        return 0
    write_csv(canonical, "31_canonical_primary_bootstrap_v31.csv")
    assertions = [
        f"analysis_version={config['analysis_version']}",
        f"requested_replicates={config['bootstrap_replicates']}",
        f"valid_replicates={bootstrap['replicate'].nunique()}",
        "resampled=INSPIRE,MOVER_2021,MOVER_2022",
        "refitted=observation_models,source_model,local_recalibration,target_auxiliary_outcome_model",
        f"seed={config['random_seed']}",
        "canonical_distribution=data/processed/canonical_bootstrap_distribution_v31.parquet",
        "status=PASS" if bootstrap["replicate"].nunique() == int(config["bootstrap_replicates"]) else "status=FAIL",
    ]
    (QA_DIR / "canonical_bootstrap_assertions_v31.txt").write_text("\n".join(assertions) + "\n", encoding="utf-8")
    elapsed = time.time() - clock
    write_run_reports(config, canonical, started, elapsed)
    print(f"v3.1 analysis complete in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
