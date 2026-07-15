#!/usr/bin/env python3
"""Run the v3.2 selective-outcome source-update-target analysis."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
import platform
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_postreview_analysis as base  # noqa: E402
import run_v31_analysis as v31  # noqa: E402


ROOT = SCRIPT_DIR.parent
CONFIG_PATH = Path(
    os.environ.get("V32_CONFIG_PATH", ROOT / "config" / "analysis_config_v32.json")
).expanduser().resolve()
DATA_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "tables"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"
QA_DIR = ROOT / "qa"
TMP_DIR = ROOT / "tmp" / "v32_diagnostic"

FEATURES = v31.FEATURES
CONTINUOUS = v31.CONTINUOUS
BINARY = v31.BINARY
GROUPED_CALIBRATION = "grouped_absolute_calibration_error_10q"
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
    GROUPED_CALIBRATION,
]
WORKLOAD_METRICS = [
    "alerts",
    "alerts_per_1000",
    "true_positives_per_1000",
    "false_positives_per_1000",
    "sensitivity",
    "positive_predictive_value",
    "false_positives_per_true_positive",
    "absolute_net_benefit",
    "treat_all_net_benefit",
    "incremental_net_benefit_vs_better_default",
]
WEIGHT_RULES: list[tuple[str, float | None, float | None]] = [
    ("none", None, None),
    ("0.5/99.5", 0.5, 99.5),
    ("1/99", 1.0, 99.0),
    ("2.5/97.5", 2.5, 97.5),
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


def performance_metrics(
    outcomes: Iterable[float], predictions: Iterable[float], weights: Iterable[float] | None = None
) -> dict[str, float]:
    metrics = base.performance_metrics(outcomes, predictions, weights)
    metrics[GROUPED_CALIBRATION] = metrics.pop("grouped_ici")
    return metrics


def workload_metrics(
    outcomes: Iterable[float], predictions: Iterable[float], threshold: float, weights: Iterable[float] | None = None
) -> dict[str, float]:
    raw = base.workload_metrics(
        np.asarray(outcomes, dtype=float),
        np.asarray(predictions, dtype=float),
        float(threshold),
        None if weights is None else np.asarray(weights, dtype=float),
    )
    raw["absolute_net_benefit"] = raw.pop("net_benefit")
    raw["incremental_net_benefit_vs_better_default"] = raw.pop("net_benefit_vs_better_default")
    return raw


def observed_indicator(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["tested_7d"].fillna(False).to_numpy(dtype=bool)
        & frame["outcome_operational"].notna().to_numpy(dtype=bool)
    )


def prepare_cohorts() -> dict[str, pd.DataFrame]:
    inspire = pd.read_parquet(DATA_DIR / "inspire_rebuilt_v32.parquet")
    mover = pd.read_parquet(DATA_DIR / "mover_rebuilt_v32.parquet")
    vital = pd.read_parquet(DATA_DIR / "vitaldb_supportive_v32.parquet")
    source = base.eligible_base(inspire).reset_index(drop=True)
    update = base.eligible_base(mover, 2021).reset_index(drop=True)
    target = base.eligible_base(mover, 2022).reset_index(drop=True)
    for frame, label in [
        (source, "INSPIRE"),
        (update, "MOVER 2021"),
        (target, "MOVER 2022"),
    ]:
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
    cohorts = {
        "source": source,
        "update": update,
        "target": target,
        "vital_observed": vital_observed,
    }
    overlap = len(set(cohorts["update"]["patient_key"]) & set(cohorts["target"]["patient_key"]))
    if overlap != 0:
        raise RuntimeError(f"Expected zero retained MOVER patient overlap, found {overlap}")
    return cohorts


def cohort_flow_table_v32(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    flow = v31.cohort_flow_table(cohorts)
    vital = flow["cohort"].str.startswith("VitalDB")
    flow.loc[vital, "denominator_note"] = (
        "official API tables reproduce all 6,388 released cases; no source-compatible "
        "all-eligible denominator was used for this supportive observed-cohort analysis"
    )
    return flow


def crossfit_binary_model(
    frame: pd.DataFrame,
    labels: np.ndarray,
    phase: str,
    expanded: bool,
    folds: int,
    seed: int,
    *,
    train_mask: np.ndarray | None = None,
    stratification_labels: np.ndarray | None = None,
    fit_final: bool = True,
    model_role: str,
) -> tuple[np.ndarray, Pipeline | None, list[str], pd.DataFrame]:
    labels = np.asarray(labels, dtype=int)
    strata = labels if stratification_labels is None else np.asarray(stratification_labels, dtype=int)
    groups = frame["case_key"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    predictions = np.full(len(frame), np.nan, dtype=float)
    _, features = v31.nuisance_specification(phase, expanded, seed)
    checks: list[dict[str, Any]] = []
    for fold, (train, valid) in enumerate(splitter.split(frame, strata, groups), start=1):
        if train_mask is not None:
            train = train[np.asarray(train_mask, dtype=bool)[train]]
        events = int(labels[train].sum())
        non_events = int(len(train) - events)
        if len(train) == 0 or events == 0 or non_events == 0:
            raise RuntimeError(f"{phase} {model_role} fold {fold} lacks both outcome classes")
        model, _ = v31.nuisance_specification(phase, expanded, seed + fold)
        model.fit(frame.iloc[train][features], labels[train])
        predictions[valid] = model.predict_proba(frame.iloc[valid][features])[:, 1]
        checks.append(
            {
                "phase": phase,
                "model_role": model_role,
                "fold": fold,
                "training_n": len(train),
                "training_events": events,
                "training_non_events": non_events,
                "validation_n": len(valid),
                "validation_observed_n": int(
                    np.asarray(train_mask, dtype=bool)[valid].sum() if train_mask is not None else len(valid)
                ),
                "prediction_min": float(predictions[valid].min()),
                "prediction_max": float(predictions[valid].max()),
            }
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"Incomplete cross-fitted predictions for {phase} {model_role}")
    final_model: Pipeline | None = None
    if fit_final:
        final_model, _ = v31.nuisance_specification(phase, expanded, seed + 100)
        final_train = np.arange(len(frame)) if train_mask is None else np.flatnonzero(train_mask)
        final_model.fit(frame.iloc[final_train][features], labels[final_train])
    return base.clip_probability(predictions), final_model, features, pd.DataFrame(checks)


def make_weights(
    propensity: np.ndarray,
    observed: np.ndarray,
    rule_label: str,
    lower: float | None,
    upper: float | None,
) -> tuple[np.ndarray, dict[str, float | str]]:
    raw = 1.0 / base.clip_probability(propensity[observed])
    if lower is None or upper is None:
        low_cut = float(raw.min())
        high_cut = float(raw.max())
        selected = raw.copy()
    else:
        low_cut, high_cut = np.percentile(raw, [float(lower), float(upper)]).tolist()
        selected = np.clip(raw, low_cut, high_cut)
    selected *= len(observed) / selected.sum()
    info: dict[str, float | str] = {
        "weight_rule": rule_label,
        "raw_weight_min": float(raw.min()),
        "raw_weight_p99": float(np.quantile(raw, 0.99)),
        "raw_weight_max": float(raw.max()),
        "truncation_low": float(low_cut),
        "truncation_high": float(high_cut),
        "weight_min": float(selected.min()),
        "weight_p99": float(np.quantile(selected, 0.99)),
        "weight_max": float(selected.max()),
        "effective_sample_size": float(selected.sum() ** 2 / np.sum(selected**2)),
    }
    return selected, info


def observation_diagnostic_row(
    phase: str,
    specification: str,
    frame: pd.DataFrame,
    observed: np.ndarray,
    propensity: np.ndarray,
    weight_info: dict[str, Any] | None,
) -> dict[str, Any]:
    tested = propensity[observed]
    untested = propensity[~observed]
    lower_support = max(float(tested.min()), float(untested.min()))
    upper_support = min(float(tested.max()), float(untested.max()))
    alpha, slope = base.fit_recalibration(observed.astype(int), propensity)
    intercept, _ = base.fit_recalibration(observed.astype(int), propensity, "intercept")
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
        "calibration_intercept": intercept,
        "calibration_joint_intercept": alpha,
        "calibration_slope": slope,
    }
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
    weight_rule: tuple[str, float | None, float | None] = ("1/99", 1.0, 99.0),
) -> dict[str, Any]:
    observed = observed_indicator(frame)
    y = frame["outcome_operational"].fillna(0).to_numpy(dtype=float)
    diagnostics: list[dict[str, Any]] = []
    fold_checks: list[pd.DataFrame] = []
    basic_propensity = None
    basic_model = None
    if include_basic:
        basic_propensity, basic_model, _, checks = crossfit_binary_model(
            frame,
            observed.astype(int),
            phase,
            False,
            int(config["crossfit_folds"]),
            seed + 10,
            fit_final=fit_final,
            model_role="basic observation model",
        )
        diagnostics.append(
            observation_diagnostic_row(phase, "basic", frame, observed, basic_propensity, None)
        )
        fold_checks.append(checks)
    expanded_propensity, expanded_model, _, checks = crossfit_binary_model(
        frame,
        observed.astype(int),
        phase,
        True,
        int(config["crossfit_folds"]),
        seed + 20,
        fit_final=fit_final,
        model_role="expanded observation model",
    )
    fold_checks.append(checks)
    weights, weight_info = make_weights(expanded_propensity, observed, *weight_rule)
    diagnostics.append(
        observation_diagnostic_row(phase, "expanded", frame, observed, expanded_propensity, weight_info)
    )
    outcome_probability = None
    outcome_model = None
    outcome_checks = pd.DataFrame()
    if fit_outcome:
        strata = np.where(observed, 1 + y.astype(int), 0)
        outcome_probability, outcome_model, _, outcome_checks = crossfit_binary_model(
            frame,
            y.astype(int),
            phase,
            True,
            int(config["crossfit_folds"]),
            seed + 30,
            train_mask=observed,
            stratification_labels=strata,
            fit_final=fit_final,
            model_role="auxiliary outcome model",
        )
        fold_checks.append(outcome_checks)
    return {
        "observed": observed,
        "y": y,
        "basic_propensity": basic_propensity,
        "expanded_propensity": expanded_propensity,
        "weights": weights,
        "weight_info": weight_info,
        "outcome_probability": outcome_probability,
        "basic_model": basic_model,
        "expanded_model": expanded_model,
        "outcome_model": outcome_model,
        "diagnostics": pd.DataFrame(diagnostics),
        "balance": v31.weighted_balance(frame, observed, weights, phase),
        "fold_checks": pd.concat(fold_checks, ignore_index=True) if fold_checks else pd.DataFrame(),
        "outcome_fold_checks": outcome_checks,
    }


def bundle_with_weight_rule(
    frame: pd.DataFrame,
    phase: str,
    bundle: dict[str, Any],
    rule: tuple[str, float | None, float | None],
) -> dict[str, Any]:
    output = dict(bundle)
    weights, info = make_weights(bundle["expanded_propensity"], bundle["observed"], *rule)
    output["weights"] = weights
    output["weight_info"] = info
    output["balance"] = v31.weighted_balance(frame, bundle["observed"], weights, phase)
    return output


def fit_source_model(
    frame: pd.DataFrame, bundle: dict[str, Any], weighted: bool, seed: int
) -> Pipeline:
    observed = bundle["observed"]
    model = v31.make_risk_model(seed)
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    preprocessor.fit(frame[FEATURES])
    transformed = preprocessor.transform(frame.loc[observed, FEATURES])
    kwargs = {"sample_weight": bundle["weights"]} if weighted else {}
    classifier.fit(
        transformed,
        frame.loc[observed, "outcome_operational"].astype(int),
        **kwargs,
    )
    return model


def fit_fractional_source_model(
    frame: pd.DataFrame, fractional_outcome: np.ndarray, seed: int
) -> Pipeline:
    outcome = np.asarray(fractional_outcome, dtype=float)
    if not np.isfinite(outcome).all() or np.any((outcome < 0) | (outcome > 1)):
        raise RuntimeError("Fractional source outcomes must be finite and between zero and one")
    model = v31.make_risk_model(seed)
    preprocessor = model.named_steps["preprocess"]
    classifier: LogisticRegression = model.named_steps["model"]
    preprocessor.fit(frame[FEATURES])
    transformed = np.asarray(preprocessor.transform(frame[FEATURES]), dtype=float)
    duplicated_x = np.concatenate([transformed, transformed], axis=0)
    duplicated_y = np.concatenate(
        [np.ones(len(frame), dtype=int), np.zeros(len(frame), dtype=int)]
    )
    fractional_weights = np.concatenate([outcome, 1.0 - outcome])
    classifier.fit(duplicated_x, duplicated_y, sample_weight=fractional_weights)
    return model


def fractional_logistic_equivalence_test(source: pd.DataFrame, seed: int) -> dict[str, Any]:
    observed = observed_indicator(source)
    binary = source["outcome_operational"].fillna(0).to_numpy(dtype=int)
    binary_model = v31.make_risk_model(seed)
    binary_model.named_steps["preprocess"].fit(source[FEATURES])
    transformed = binary_model.named_steps["preprocess"].transform(source[FEATURES])
    binary_model.named_steps["model"].fit(transformed, binary)
    fractional_model = fit_fractional_source_model(source, binary.astype(float), seed)
    p_binary = binary_model.predict_proba(source[FEATURES])[:, 1]
    p_fractional = fractional_model.predict_proba(source[FEATURES])[:, 1]
    coefficient_difference = float(
        np.max(
            np.abs(
                binary_model.named_steps["model"].coef_
                - fractional_model.named_steps["model"].coef_
            )
        )
    )
    prediction_difference = float(np.max(np.abs(p_binary - p_fractional)))
    tolerance = 1e-7
    return {
        "test": "fractional logistic equals ordinary logistic for binary outcomes",
        "n": len(source),
        "observed_outcome_n_in_binary_fixture": int(observed.sum()),
        "max_absolute_coefficient_difference": coefficient_difference,
        "max_absolute_prediction_difference": prediction_difference,
        "tolerance": tolerance,
        "status": "PASS"
        if coefficient_difference <= tolerance and prediction_difference <= tolerance
        else "FAIL",
    }


def ipw_full_metrics(
    frame: pd.DataFrame, prediction: np.ndarray, bundle: dict[str, Any]
) -> dict[str, float]:
    observed = bundle["observed"]
    y = bundle["y"][observed]
    weights = bundle["weights"]
    metrics = performance_metrics(y, prediction[observed], weights)
    event_total = float(np.sum(weights * y))
    metrics["n"] = float(len(frame))
    metrics["weighted_n"] = float(len(frame))
    metrics["events"] = event_total
    metrics["event_rate"] = event_total / len(frame)
    metrics["predicted_mean"] = float(np.mean(prediction))
    metrics["oe_ratio"] = event_total / float(np.sum(prediction))
    return metrics


def aipw_pseudo_outcome(bundle: dict[str, Any]) -> np.ndarray:
    observed = bundle["observed"]
    y = bundle["y"]
    propensity = base.clip_probability(bundle["expanded_propensity"])
    outcome_probability = base.clip_probability(bundle["outcome_probability"])
    return outcome_probability + observed * (y - outcome_probability) / propensity


def aipw_hybrid_metrics(
    frame: pd.DataFrame, prediction: np.ndarray, bundle: dict[str, Any]
) -> dict[str, float]:
    metrics = ipw_full_metrics(frame, prediction, bundle)
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
    metrics.update(
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
    return metrics


def target_workload(
    prediction: np.ndarray,
    bundle: dict[str, Any],
    thresholds: Iterable[float],
) -> list[dict[str, float]]:
    pseudo_y = aipw_pseudo_outcome(bundle)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        metrics = workload_metrics(pseudo_y, prediction, float(threshold))
        metrics["alerts"] = float(np.sum(prediction >= float(threshold)))
        metrics["alerts_per_1000"] = 1000 * metrics["alerts"] / len(prediction)
        rows.append({"threshold": float(threshold), **metrics})
    return rows


def completed_workload(
    fractional_outcome: np.ndarray,
    prediction: np.ndarray,
    thresholds: Iterable[float],
) -> list[dict[str, float]]:
    return [
        {"threshold": float(threshold), **workload_metrics(fractional_outcome, prediction, float(threshold))}
        for threshold in thresholds
    ]


def recalibrate_ipw(
    source_model: Pipeline, update: pd.DataFrame, bundle: dict[str, Any]
) -> tuple[float, float]:
    prediction = source_model.predict_proba(update[FEATURES])[:, 1]
    observed = bundle["observed"]
    return base.fit_recalibration(
        bundle["y"][observed], prediction[observed], weights=bundle["weights"]
    )


def shifted_fractional_outcome(bundle: dict[str, Any], multiplier: float) -> np.ndarray:
    shifted = expit(
        logit(base.clip_probability(bundle["outcome_probability"])) + math.log(float(multiplier))
    )
    return np.where(bundle["observed"], bundle["y"], shifted)


def add_workload_columns(row: dict[str, Any], workload: list[dict[str, float]]) -> None:
    for item in workload:
        label = int(round(item["threshold"] * 100))
        for metric in WORKLOAD_METRICS:
            row[f"threshold_{label}_{metric}"] = float(item[metric])


def canonical_result(
    source: pd.DataFrame,
    update: pd.DataFrame,
    target: pd.DataFrame,
    source_bundle: dict[str, Any],
    update_bundle: dict[str, Any],
    target_bundle: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    source_model = fit_source_model(source, source_bundle, True, seed)
    alpha, beta = recalibrate_ipw(source_model, update, update_bundle)
    source_prediction = source_model.predict_proba(target[FEATURES])[:, 1]
    prediction = base.apply_recalibration(source_prediction, alpha, beta)
    metrics = aipw_hybrid_metrics(target, prediction, target_bundle)
    workload = target_workload(prediction, target_bundle, config["decision_thresholds"])
    return {
        "source_model": source_model,
        "alpha": float(alpha),
        "beta": float(beta),
        "source_prediction_target": source_prediction,
        "prediction": prediction,
        "metrics": metrics,
        "workload": workload,
    }


def scenario_row(
    *,
    stage: str,
    estimator_family: str,
    reference_type: str,
    is_canonical: bool,
    multiplier: float | None,
    source_expected_events: float,
    update_expected_events: float,
    target_expected_events: float,
    alpha: float,
    beta: float,
    metrics: dict[str, float],
    workload: list[dict[str, float]],
    source_n: int,
    update_n: int,
    target_n: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage_varied": stage,
        "estimator_family": estimator_family,
        "reference_type": reference_type,
        "is_canonical": bool(is_canonical),
        "odds_multiplier": multiplier,
        "log_odds_shift": math.log(float(multiplier)) if multiplier is not None else np.nan,
        "interpretation": (
            "Canonical measured-variable MAR reference using truncated IPW/AIPW"
            if is_canonical
            else "Outcome-regression completion sensitivity parameter; not asserted clinically plausible"
        ),
        "source_eligible_n": source_n,
        "update_eligible_n": update_n,
        "target_eligible_n": target_n,
        "source_expected_events": float(source_expected_events),
        "update_expected_events": float(update_expected_events),
        "target_expected_events": float(target_expected_events),
        "update_alpha": float(alpha),
        "update_beta": float(beta),
        **{f"target_{metric}": float(metrics[metric]) for metric in PERFORMANCE_METRICS},
    }
    add_workload_columns(row, workload)
    return row


def canonical_scenario_row(
    stage: str,
    source: pd.DataFrame,
    update: pd.DataFrame,
    target: pd.DataFrame,
    source_bundle: dict[str, Any],
    update_bundle: dict[str, Any],
    target_bundle: dict[str, Any],
    canonical: dict[str, Any],
) -> dict[str, Any]:
    source_events = float(
        np.sum(source_bundle["weights"] * source_bundle["y"][source_bundle["observed"]])
    )
    update_events = float(
        np.sum(update_bundle["weights"] * update_bundle["y"][update_bundle["observed"]])
    )
    return scenario_row(
        stage=stage,
        estimator_family="canonical truncated IPW/AIPW MAR",
        reference_type="Canonical MAR reference",
        is_canonical=True,
        multiplier=None,
        source_expected_events=source_events,
        update_expected_events=update_events,
        target_expected_events=canonical["metrics"]["events"],
        alpha=canonical["alpha"],
        beta=canonical["beta"],
        metrics=canonical["metrics"],
        workload=canonical["workload"],
        source_n=len(source),
        update_n=len(update),
        target_n=len(target),
    )


def source_mnar_propagation(
    cohorts: dict[str, pd.DataFrame],
    bundles: dict[str, dict[str, Any]],
    canonical: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    source, update, target = cohorts["source"], cohorts["update"], cohorts["target"]
    source_bundle, update_bundle, target_bundle = (
        bundles["source"],
        bundles["update"],
        bundles["target"],
    )
    rows = [
        canonical_scenario_row(
            "source", source, update, target, source_bundle, update_bundle, target_bundle, canonical
        )
    ]
    update_expected = float(
        np.sum(update_bundle["weights"] * update_bundle["y"][update_bundle["observed"]])
    )
    for index, multiplier in enumerate(config["mnar_odds_multipliers"]):
        fractional_source = shifted_fractional_outcome(source_bundle, float(multiplier))
        source_model = fit_fractional_source_model(source, fractional_source, seed + 100 + index)
        alpha, beta = recalibrate_ipw(source_model, update, update_bundle)
        prediction = base.apply_recalibration(
            source_model.predict_proba(target[FEATURES])[:, 1], alpha, beta
        )
        metrics = aipw_hybrid_metrics(target, prediction, target_bundle)
        workload = target_workload(prediction, target_bundle, config["decision_thresholds"])
        rows.append(
            scenario_row(
                stage="source",
                estimator_family="source outcome-regression completion propagated through canonical update and target MAR",
                reference_type="OR-completion reference, k=1"
                if math.isclose(float(multiplier), 1.0)
                else "OR-completion sensitivity",
                is_canonical=False,
                multiplier=float(multiplier),
                source_expected_events=float(fractional_source.sum()),
                update_expected_events=update_expected,
                target_expected_events=metrics["events"],
                alpha=alpha,
                beta=beta,
                metrics=metrics,
                workload=workload,
                source_n=len(source),
                update_n=len(update),
                target_n=len(target),
            )
        )
    return pd.DataFrame(rows)


def update_mnar_propagation(
    cohorts: dict[str, pd.DataFrame],
    bundles: dict[str, dict[str, Any]],
    canonical: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    source, update, target = cohorts["source"], cohorts["update"], cohorts["target"]
    source_bundle, update_bundle, target_bundle = (
        bundles["source"],
        bundles["update"],
        bundles["target"],
    )
    rows = [
        canonical_scenario_row(
            "update", source, update, target, source_bundle, update_bundle, target_bundle, canonical
        )
    ]
    source_expected = float(
        np.sum(source_bundle["weights"] * source_bundle["y"][source_bundle["observed"]])
    )
    source_model = canonical["source_model"]
    update_prediction = source_model.predict_proba(update[FEATURES])[:, 1]
    for multiplier in config["mnar_odds_multipliers"]:
        fractional_update = shifted_fractional_outcome(update_bundle, float(multiplier))
        alpha, beta = base.fit_recalibration(fractional_update, update_prediction)
        prediction = base.apply_recalibration(
            source_model.predict_proba(target[FEATURES])[:, 1], alpha, beta
        )
        metrics = aipw_hybrid_metrics(target, prediction, target_bundle)
        workload = target_workload(prediction, target_bundle, config["decision_thresholds"])
        rows.append(
            scenario_row(
                stage="update",
                estimator_family="update outcome-regression completion propagated to canonical target MAR",
                reference_type="OR-completion reference, k=1"
                if math.isclose(float(multiplier), 1.0)
                else "OR-completion sensitivity",
                is_canonical=False,
                multiplier=float(multiplier),
                source_expected_events=source_expected,
                update_expected_events=float(fractional_update.sum()),
                target_expected_events=metrics["events"],
                alpha=alpha,
                beta=beta,
                metrics=metrics,
                workload=workload,
                source_n=len(source),
                update_n=len(update),
                target_n=len(target),
            )
        )
    return pd.DataFrame(rows)


def target_mnar_sensitivity(
    cohorts: dict[str, pd.DataFrame],
    bundles: dict[str, dict[str, Any]],
    canonical: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    source, update, target = cohorts["source"], cohorts["update"], cohorts["target"]
    source_bundle, update_bundle, target_bundle = (
        bundles["source"],
        bundles["update"],
        bundles["target"],
    )
    rows = [
        canonical_scenario_row(
            "target", source, update, target, source_bundle, update_bundle, target_bundle, canonical
        )
    ]
    source_expected = float(
        np.sum(source_bundle["weights"] * source_bundle["y"][source_bundle["observed"]])
    )
    update_expected = float(
        np.sum(update_bundle["weights"] * update_bundle["y"][update_bundle["observed"]])
    )
    for multiplier in config["mnar_odds_multipliers"]:
        fractional_target = shifted_fractional_outcome(target_bundle, float(multiplier))
        metrics = performance_metrics(fractional_target, canonical["prediction"])
        workload = completed_workload(
            fractional_target, canonical["prediction"], config["decision_thresholds"]
        )
        rows.append(
            scenario_row(
                stage="target",
                estimator_family="target outcome-regression completion",
                reference_type="OR-completion reference, k=1"
                if math.isclose(float(multiplier), 1.0)
                else "OR-completion sensitivity",
                is_canonical=False,
                multiplier=float(multiplier),
                source_expected_events=source_expected,
                update_expected_events=update_expected,
                target_expected_events=float(fractional_target.sum()),
                alpha=canonical["alpha"],
                beta=canonical["beta"],
                metrics=metrics,
                workload=workload,
                source_n=len(source),
                update_n=len(update),
                target_n=len(target),
            )
        )
    return pd.DataFrame(rows)


def mnar_anchor_table(
    source_mnar: pd.DataFrame, update_mnar: pd.DataFrame, target_mnar: pd.DataFrame
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for table in [source_mnar, update_mnar, target_mnar]:
        rows.append(table.loc[table["is_canonical"]].copy())
        rows.append(table.loc[table["odds_multiplier"].eq(1.0)].copy())
    output = pd.concat(rows, ignore_index=True)
    output["anchor_note"] = (
        "An odds multiplier of 1 denotes the outcome-regression completion reference and is not "
        "numerically constrained to equal the canonical IPW/AIPW MAR estimate."
    )
    keep = [
        "stage_varied",
        "estimator_family",
        "reference_type",
        "is_canonical",
        "odds_multiplier",
        "log_odds_shift",
        "source_expected_events",
        "update_expected_events",
        "update_alpha",
        "update_beta",
        "target_expected_events",
        "target_oe_ratio",
        "target_calibration_slope",
        "target_auroc",
        "target_brier",
        "threshold_10_alerts",
        "anchor_note",
    ]
    return output[keep]


def auxiliary_outcome_diagnostics(
    cohorts: dict[str, pd.DataFrame], bundles: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    fold_rows: list[pd.DataFrame] = []
    for key, phase in [("source", "INSPIRE"), ("update", "MOVER 2021"), ("target", "MOVER 2022")]:
        frame = cohorts[key]
        bundle = bundles[key]
        observed = bundle["observed"]
        prediction = bundle["outcome_probability"]
        if prediction is None or not np.isfinite(prediction).all():
            raise RuntimeError(f"Missing auxiliary outcome predictions for {phase}")
        for population, weights in [
            ("observed-unweighted", None),
            ("observed-IPW-weighted", bundle["weights"]),
        ]:
            metrics = performance_metrics(
                bundle["y"][observed], prediction[observed], weights
            )
            rows.append(
                {
                    "phase": phase,
                    "evaluation_population": population,
                    "n": int(observed.sum()),
                    "eligible_n": len(frame),
                    "observed_events_unweighted": int(bundle["y"][observed].sum()),
                    "predicted_probability_min": float(prediction.min()),
                    "predicted_probability_p01": float(np.quantile(prediction, 0.01)),
                    "predicted_probability_median": float(np.median(prediction)),
                    "predicted_probability_p99": float(np.quantile(prediction, 0.99)),
                    "predicted_probability_max": float(prediction.max()),
                    "observed_probability_min": float(prediction[observed].min()),
                    "observed_probability_median": float(np.median(prediction[observed])),
                    "observed_probability_max": float(prediction[observed].max()),
                    "unobserved_probability_min": float(prediction[~observed].min()),
                    "unobserved_probability_median": float(np.median(prediction[~observed])),
                    "unobserved_probability_max": float(prediction[~observed].max()),
                    "diagnostic_role": "nuisance/auxiliary outcome-model diagnostic; not independent prediction-model validation",
                    **metrics,
                }
            )
        fold_rows.append(bundle["outcome_fold_checks"])
    diagnostics = pd.DataFrame(rows)
    checks = pd.concat(fold_rows, ignore_index=True)
    return diagnostics, checks


def weight_truncation_sensitivity(
    cohorts: dict[str, pd.DataFrame],
    bundles: dict[str, dict[str, Any]],
    config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    source, update, target = cohorts["source"], cohorts["update"], cohorts["target"]
    rows: list[dict[str, Any]] = []
    for index, rule in enumerate(WEIGHT_RULES):
        source_bundle = bundle_with_weight_rule(source, "INSPIRE", bundles["source"], rule)
        update_bundle = bundle_with_weight_rule(update, "MOVER 2021", bundles["update"], rule)
        target_bundle = bundle_with_weight_rule(target, "MOVER 2022", bundles["target"], rule)
        result = canonical_result(
            source,
            update,
            target,
            source_bundle,
            update_bundle,
            target_bundle,
            config,
            seed + index,
        )
        row: dict[str, Any] = {
            "weight_rule": rule[0],
            "is_canonical": rule[0] == "1/99",
            "source_preprocessor_training_n": len(source),
            "update_alpha": result["alpha"],
            "update_beta": result["beta"],
            **{f"target_{metric}": result["metrics"][metric] for metric in PERFORMANCE_METRICS},
        }
        for phase, frame, bundle in [
            ("source", source, source_bundle),
            ("update", update, update_bundle),
            ("target", target, target_bundle),
        ]:
            for key, value in bundle["weight_info"].items():
                if key != "weight_rule":
                    row[f"{phase}_{key}"] = value
            row[f"{phase}_max_absolute_weighted_smd"] = float(
                bundle["balance"]["weighted_smd_vs_target"].abs().max()
            )
        add_workload_columns(row, result["workload"])
        rows.append(row)
    return pd.DataFrame(rows)


def preprocessing_audit(model: Pipeline, source: pd.DataFrame) -> pd.DataFrame:
    preprocess = model.named_steps["preprocess"]
    continuous = preprocess.named_transformers_["continuous"]
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(CONTINUOUS):
        rows.append(
            {
                "feature": feature,
                "feature_type": "continuous",
                "preprocessor_training_population": "all eligible INSPIRE",
                "preprocessor_training_n": len(source),
                "imputation_value": float(continuous.named_steps["impute"].statistics_[index]),
                "standardization_mean": float(continuous.named_steps["scale"].mean_[index]),
                "standardization_scale": float(continuous.named_steps["scale"].scale_[index]),
                "spline_knot_vector_standardized": json.dumps(
                    continuous.named_steps["spline"].bsplines_[index].t.tolist()
                ),
            }
        )
    binary = preprocess.named_transformers_["binary"]
    rows.append(
        {
            "feature": "sex_male",
            "feature_type": "binary",
            "preprocessor_training_population": "all eligible INSPIRE",
            "preprocessor_training_n": len(source),
            "imputation_value": float(binary.named_steps["impute"].statistics_[0]),
            "standardization_mean": np.nan,
            "standardization_scale": np.nan,
            "spline_knot_vector_standardized": "not applicable",
        }
    )
    return pd.DataFrame(rows)


def target_nonparametric_bounds(
    target: pd.DataFrame,
    target_bundle: dict[str, Any],
    prediction: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    observed = target_bundle["observed"]
    for label, fill in [
        ("all unobserved outcomes are non-events", 0.0),
        ("all unobserved outcomes are events", 1.0),
    ]:
        completed = np.where(observed, target_bundle["y"], fill)
        metrics = performance_metrics(completed, prediction)
        row: dict[str, Any] = {
            "bound_scenario": label,
            "unobserved_outcome_value": fill,
            "eligible_n": len(target),
            "observed_n": int(observed.sum()),
            "unobserved_n": int((~observed).sum()),
            **metrics,
        }
        add_workload_columns(
            row, completed_workload(completed, prediction, config["decision_thresholds"])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def selection_chain_comparators(
    cohorts: dict[str, pd.DataFrame],
    bundles: dict[str, dict[str, Any]],
    config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source, update, target = cohorts["source"], cohorts["update"], cohorts["target"]
    source_models = {
        "observed-outcome source model": fit_source_model(source, bundles["source"], False, seed),
        "1/99 truncated-IPW source model": fit_source_model(source, bundles["source"], True, seed + 1),
    }
    performance_rows: list[dict[str, Any]] = []
    workload_rows: list[dict[str, Any]] = []
    for source_label, model in source_models.items():
        update_prediction = model.predict_proba(update[FEATURES])[:, 1]
        observed = bundles["update"]["observed"]
        updates = [
            (
                "unupdated source-only comparator",
                0.0,
                1.0,
                source_label == "1/99 truncated-IPW source model",
            ),
            (
                "complete-case recalibration",
                *base.fit_recalibration(
                    bundles["update"]["y"][observed], update_prediction[observed]
                ),
                False,
            ),
            (
                "1/99 truncated-IPW recalibration",
                *base.fit_recalibration(
                    bundles["update"]["y"][observed],
                    update_prediction[observed],
                    weights=bundles["update"]["weights"],
                ),
                source_label == "1/99 truncated-IPW source model",
            ),
        ]
        for update_label, alpha, beta, canonical_or_source_comparator in updates:
            target_prediction = base.apply_recalibration(
                model.predict_proba(target[FEATURES])[:, 1], alpha, beta
            )
            metrics = aipw_hybrid_metrics(target, target_prediction, bundles["target"])
            strategy = f"{source_label} + {update_label}"
            performance_rows.append(
                {
                    "strategy": strategy,
                    "source_model_estimand": source_label,
                    "update_estimator": update_label,
                    "primary_or_required_comparator": canonical_or_source_comparator,
                    "update_alpha": alpha,
                    "update_beta": beta,
                    **metrics,
                }
            )
            for item in target_workload(
                target_prediction, bundles["target"], config["decision_thresholds"]
            ):
                workload_rows.append(
                    {
                        "strategy": strategy,
                        "primary_or_required_comparator": canonical_or_source_comparator,
                        **item,
                    }
                )
    return pd.DataFrame(performance_rows), pd.DataFrame(workload_rows)


def vitaldb_supportive(cohorts: dict[str, pd.DataFrame], canonical: dict[str, Any]) -> pd.DataFrame:
    vital = cohorts["vital_observed"]
    source_prediction = canonical["source_model"].predict_proba(vital[FEATURES])[:, 1]
    updated_prediction = base.apply_recalibration(
        source_prediction, canonical["alpha"], canonical["beta"]
    )
    outcome = vital["outcome_operational"].to_numpy(dtype=int)
    rows = []
    for model_label, prediction in [
        ("1/99 truncated-IPW INSPIRE source model", source_prediction),
        ("source model plus 1/99 truncated-IPW MOVER 2021 recalibration", updated_prediction),
    ]:
        rows.append(
            {
                "cohort_label": "VitalDB available analysis-ready observed cohort",
                "model": model_label,
                "n": len(vital),
                "events": int(outcome.sum()),
                "interpretation": "supportive observed-cohort stress test; not independent validation and not a complete eligible-population analysis",
                **performance_metrics(outcome, prediction),
            }
        )
    return pd.DataFrame(rows)


def canonical_bootstrap_row(
    replicate: int,
    attempt: int,
    canonical: dict[str, Any],
) -> dict[str, float]:
    row: dict[str, float] = {
        "replicate": float(replicate),
        "attempt": float(attempt),
        "update_alpha": float(canonical["alpha"]),
        "update_beta": float(canonical["beta"]),
        **{metric: float(canonical["metrics"][metric]) for metric in PERFORMANCE_METRICS},
    }
    add_workload_columns(row, canonical["workload"])
    numeric = np.asarray(
        [value for key, value in row.items() if key not in {"replicate", "attempt"}],
        dtype=float,
    )
    if not np.isfinite(numeric).all():
        raise RuntimeError("Non-finite canonical bootstrap metric")
    return row


def mnar_bootstrap_row(
    replicate: int,
    attempt: int,
    point: dict[str, Any],
) -> dict[str, Any]:
    keys = [
        "stage_varied",
        "estimator_family",
        "reference_type",
        "is_canonical",
        "odds_multiplier",
        "log_odds_shift",
        "source_eligible_n",
        "update_eligible_n",
        "target_eligible_n",
        "source_expected_events",
        "update_expected_events",
        "target_expected_events",
        "update_alpha",
        "update_beta",
        "target_oe_ratio",
        "target_calibration_slope",
        "target_auroc",
        "target_brier",
    ]
    row = {key: point[key] for key in keys}
    row.update({"replicate": replicate, "attempt": attempt})
    for threshold in [5, 10, 20]:
        for metric in [
            "alerts",
            "alerts_per_1000",
            "true_positives_per_1000",
            "false_positives_per_1000",
            "sensitivity",
            "positive_predictive_value",
            "absolute_net_benefit",
            "incremental_net_benefit_vs_better_default",
        ]:
            row[f"threshold_{threshold}_{metric}"] = point[f"threshold_{threshold}_{metric}"]
    numeric = np.asarray(
        [
            value
            for key, value in row.items()
            if key
            not in {
                "stage_varied",
                "estimator_family",
                "reference_type",
                "is_canonical",
                "replicate",
                "attempt",
            }
        ],
        dtype=float,
    )
    if not np.isfinite(numeric).all():
        raise RuntimeError("Non-finite MNAR bootstrap metric")
    return row


def bootstrap_once(
    source: pd.DataFrame,
    update: pd.DataFrame,
    target: pd.DataFrame,
    config: dict[str, Any],
    replicate: int,
    attempt: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    seed = int(config["random_seed"]) + 1_000_000 + replicate * 100 + attempt
    rng = np.random.default_rng(seed)
    source_sample = source.iloc[rng.integers(0, len(source), len(source))].reset_index(drop=True)
    update_sample = update.iloc[rng.integers(0, len(update), len(update))].reset_index(drop=True)
    target_sample = target.iloc[rng.integers(0, len(target), len(target))].reset_index(drop=True)

    source_bundle = observation_bundle(
        source_sample,
        "INSPIRE",
        config,
        seed + 1,
        include_basic=False,
        fit_outcome=True,
        fit_final=False,
    )
    update_bundle = observation_bundle(
        update_sample,
        "MOVER 2021",
        config,
        seed + 2,
        include_basic=False,
        fit_outcome=True,
        fit_final=False,
    )
    target_bundle = observation_bundle(
        target_sample,
        "MOVER 2022",
        config,
        seed + 3,
        include_basic=False,
        fit_outcome=True,
        fit_final=False,
    )
    cohorts = {"source": source_sample, "update": update_sample, "target": target_sample}
    bundles = {"source": source_bundle, "update": update_bundle, "target": target_bundle}
    canonical = canonical_result(
        source_sample,
        update_sample,
        target_sample,
        source_bundle,
        update_bundle,
        target_bundle,
        config,
        seed + 4,
    )
    canonical_row = canonical_bootstrap_row(replicate, attempt, canonical)
    mnar_points: list[dict[str, Any]] = []

    source_expected = float(
        np.sum(source_bundle["weights"] * source_bundle["y"][source_bundle["observed"]])
    )
    update_expected = float(
        np.sum(update_bundle["weights"] * update_bundle["y"][update_bundle["observed"]])
    )
    for index, multiplier in enumerate(config["mnar_odds_multipliers"]):
        fractional_source = shifted_fractional_outcome(source_bundle, float(multiplier))
        source_model = fit_fractional_source_model(
            source_sample, fractional_source, seed + 100 + index
        )
        alpha, beta = recalibrate_ipw(source_model, update_sample, update_bundle)
        prediction = base.apply_recalibration(
            source_model.predict_proba(target_sample[FEATURES])[:, 1], alpha, beta
        )
        metrics = aipw_hybrid_metrics(target_sample, prediction, target_bundle)
        workload = target_workload(prediction, target_bundle, config["decision_thresholds"])
        mnar_points.append(
            scenario_row(
                stage="source",
                estimator_family="source outcome-regression completion propagated through canonical update and target MAR",
                reference_type="OR-completion reference, k=1"
                if math.isclose(float(multiplier), 1.0)
                else "OR-completion sensitivity",
                is_canonical=False,
                multiplier=float(multiplier),
                source_expected_events=float(fractional_source.sum()),
                update_expected_events=update_expected,
                target_expected_events=metrics["events"],
                alpha=alpha,
                beta=beta,
                metrics=metrics,
                workload=workload,
                source_n=len(source_sample),
                update_n=len(update_sample),
                target_n=len(target_sample),
            )
        )

    update_prediction = canonical["source_model"].predict_proba(update_sample[FEATURES])[:, 1]
    for multiplier in config["mnar_odds_multipliers"]:
        fractional_update = shifted_fractional_outcome(update_bundle, float(multiplier))
        alpha, beta = base.fit_recalibration(fractional_update, update_prediction)
        prediction = base.apply_recalibration(
            canonical["source_model"].predict_proba(target_sample[FEATURES])[:, 1], alpha, beta
        )
        metrics = aipw_hybrid_metrics(target_sample, prediction, target_bundle)
        workload = target_workload(prediction, target_bundle, config["decision_thresholds"])
        mnar_points.append(
            scenario_row(
                stage="update",
                estimator_family="update outcome-regression completion propagated to canonical target MAR",
                reference_type="OR-completion reference, k=1"
                if math.isclose(float(multiplier), 1.0)
                else "OR-completion sensitivity",
                is_canonical=False,
                multiplier=float(multiplier),
                source_expected_events=source_expected,
                update_expected_events=float(fractional_update.sum()),
                target_expected_events=metrics["events"],
                alpha=alpha,
                beta=beta,
                metrics=metrics,
                workload=workload,
                source_n=len(source_sample),
                update_n=len(update_sample),
                target_n=len(target_sample),
            )
        )

    for multiplier in config["mnar_odds_multipliers"]:
        fractional_target = shifted_fractional_outcome(target_bundle, float(multiplier))
        metrics = performance_metrics(fractional_target, canonical["prediction"])
        workload = completed_workload(
            fractional_target, canonical["prediction"], config["decision_thresholds"]
        )
        mnar_points.append(
            scenario_row(
                stage="target",
                estimator_family="target outcome-regression completion",
                reference_type="OR-completion reference, k=1"
                if math.isclose(float(multiplier), 1.0)
                else "OR-completion sensitivity",
                is_canonical=False,
                multiplier=float(multiplier),
                source_expected_events=source_expected,
                update_expected_events=update_expected,
                target_expected_events=float(fractional_target.sum()),
                alpha=canonical["alpha"],
                beta=canonical["beta"],
                metrics=metrics,
                workload=workload,
                source_n=len(source_sample),
                update_n=len(update_sample),
                target_n=len(target_sample),
            )
        )
    if len(mnar_points) != 18:
        raise RuntimeError(f"Expected 18 MNAR scenarios, found {len(mnar_points)}")
    return canonical_row, [mnar_bootstrap_row(replicate, attempt, row) for row in mnar_points]


def bootstrap_worker_init(config: dict[str, Any]) -> None:
    global _WORKER_COHORTS, _WORKER_CONFIG
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _WORKER_CONFIG = config
    _WORKER_COHORTS = prepare_cohorts()


def bootstrap_worker_run(replicate: int) -> dict[str, Any]:
    if _WORKER_COHORTS is None or _WORKER_CONFIG is None:
        raise RuntimeError("Bootstrap worker was not initialized")
    errors: list[dict[str, Any]] = []
    maximum_attempts = int(
        _WORKER_CONFIG["canonical_analysis"].get("maximum_attempts_per_replicate", 5)
    )
    for attempt in range(1, maximum_attempts + 1):
        try:
            canonical, mnar = bootstrap_once(
                _WORKER_COHORTS["source"],
                _WORKER_COHORTS["update"],
                _WORKER_COHORTS["target"],
                _WORKER_CONFIG,
                replicate,
                attempt,
            )
            return {"canonical": canonical, "mnar": mnar, "errors": errors}
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            errors.append(
                {
                    "replicate": replicate,
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    raise RuntimeError(
        f"Replicate {replicate} failed after {maximum_attempts} attempts: "
        f"{errors[-1]['error_message']}"
    )


def run_bootstrap(
    config: dict[str, Any], replicates: int, output_dir: Path, jobs: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_partial = output_dir / "canonical_bootstrap_distribution_v32.partial.parquet"
    mnar_partial = output_dir / "mnar_chain_bootstrap_v32.partial.parquet"
    failure_partial = output_dir / "mnar_bootstrap_failures_v32.partial.csv"
    canonical_rows: list[dict[str, Any]] = []
    mnar_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    if canonical_partial.exists() and mnar_partial.exists():
        canonical_rows = pd.read_parquet(canonical_partial).to_dict(orient="records")
        mnar_rows = pd.read_parquet(mnar_partial).to_dict(orient="records")
        if failure_partial.exists():
            failure_rows = pd.read_csv(failure_partial).to_dict(orient="records")
    completed = {int(row["replicate"]) for row in canonical_rows}
    pending = [replicate for replicate in range(1, replicates + 1) if replicate not in completed]
    started = time.time()

    def checkpoint() -> None:
        pd.DataFrame(canonical_rows).sort_values("replicate").to_parquet(
            canonical_partial, index=False
        )
        pd.DataFrame(mnar_rows).sort_values(
            ["replicate", "stage_varied", "odds_multiplier"]
        ).to_parquet(mnar_partial, index=False)
        pd.DataFrame(failure_rows).to_csv(failure_partial, index=False)

    if jobs <= 1:
        bootstrap_worker_init(config)
        for replicate in pending:
            result = bootstrap_worker_run(replicate)
            canonical_rows.append(result["canonical"])
            mnar_rows.extend(result["mnar"])
            failure_rows.extend(result["errors"])
            if len(canonical_rows) % 10 == 0 or len(canonical_rows) == replicates:
                checkpoint()
                print(
                    f"Combined bootstrap {len(canonical_rows)}/{replicates}; "
                    f"elapsed {time.time() - started:.1f}s",
                    flush=True,
                )
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
                result = future.result()
                canonical_rows.append(result["canonical"])
                mnar_rows.extend(result["mnar"])
                failure_rows.extend(result["errors"])
                if len(canonical_rows) % 10 == 0 or len(canonical_rows) == replicates:
                    checkpoint()
                    print(
                        f"Combined bootstrap {len(canonical_rows)}/{replicates}; "
                        f"elapsed {time.time() - started:.1f}s",
                        flush=True,
                    )
    canonical = pd.DataFrame(canonical_rows).sort_values("replicate").reset_index(drop=True)
    mnar = pd.DataFrame(mnar_rows).sort_values(
        ["replicate", "stage_varied", "odds_multiplier"]
    ).reset_index(drop=True)
    failure_columns = ["replicate", "attempt", "error_type", "error_message"]
    failures = pd.DataFrame(failure_rows, columns=failure_columns)
    if canonical["replicate"].nunique() != replicates:
        raise RuntimeError("Canonical bootstrap did not produce the requested valid replicates")
    expected_mnar_rows = replicates * 18
    if len(mnar) != expected_mnar_rows:
        raise RuntimeError(
            f"MNAR bootstrap expected {expected_mnar_rows} rows, found {len(mnar)}"
        )
    canonical.to_parquet(output_dir / "canonical_bootstrap_distribution_v32.parquet", index=False)
    mnar.to_parquet(output_dir / "mnar_chain_bootstrap_v32.parquet", index=False)
    failures.to_csv(output_dir / "mnar_bootstrap_failures_v32.csv", index=False)
    for path in [canonical_partial, mnar_partial, failure_partial]:
        if path.exists():
            path.unlink()
    return canonical, mnar, failures


def canonical_long_table(
    point: dict[str, Any], bootstrap: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = {
        "events": "AIPW",
        "event_rate": "AIPW",
        "predicted_mean": "exact all-eligible predictions",
        "oe_ratio": "plug-in ratio of AIPW observed-event total to exact predicted-event total",
        "calibration_intercept": "1/99 truncated expanded IPW",
        "calibration_joint_intercept": "1/99 truncated expanded IPW",
        "calibration_slope": "1/99 truncated expanded IPW",
        "auroc": "1/99 truncated expanded IPW",
        "auprc": "1/99 truncated expanded IPW",
        "brier": "AIPW",
        "brier_skill_score": "AIPW",
        GROUPED_CALIBRATION: "1/99 truncated expanded IPW",
    }
    for metric in PERFORMANCE_METRICS:
        lower, upper = bootstrap[metric].quantile([0.025, 0.975]).tolist()
        rows.append(
            {
                "endpoint": config["primary_endpoint"],
                "estimand": config["estimands"]["target_evaluation"],
                "model": config["canonical_analysis"]["model"],
                "metric": metric,
                "estimate": point["metrics"][metric],
                "ci_lower": lower,
                "ci_upper": upper,
                "estimator_component": sources[metric],
                "bootstrap_replicates": len(bootstrap),
                "bootstrap_unit": config["canonical_analysis"]["bootstrap_unit"],
                "canonical_source": "data/processed/canonical_bootstrap_distribution_v32.parquet",
            }
        )
    for metric, column, point_key in [
        ("update_alpha", "update_alpha", "alpha"),
        ("update_beta", "update_beta", "beta"),
    ]:
        rows.append(
            {
                "endpoint": config["primary_endpoint"],
                "estimand": config["estimands"]["local_update"],
                "model": config["canonical_analysis"]["model"],
                "metric": metric,
                "estimate": point[point_key],
                "ci_lower": bootstrap[column].quantile(0.025),
                "ci_upper": bootstrap[column].quantile(0.975),
                "estimator_component": "1/99 truncated expanded-IPW update score equation",
                "bootstrap_replicates": len(bootstrap),
                "bootstrap_unit": config["canonical_analysis"]["bootstrap_unit"],
                "canonical_source": "data/processed/canonical_bootstrap_distribution_v32.parquet",
            }
        )
    for item in point["workload"]:
        label = int(round(item["threshold"] * 100))
        for metric in WORKLOAD_METRICS:
            column = f"threshold_{label}_{metric}"
            lower, upper = bootstrap[column].quantile([0.025, 0.975]).tolist()
            rows.append(
                {
                    "endpoint": config["primary_endpoint"],
                    "estimand": f"threshold workload among all eligible MOVER 2022 patients at {label}%",
                    "model": config["canonical_analysis"]["model"],
                    "metric": metric,
                    "estimate": item[metric],
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "estimator_component": (
                        "exact all-eligible predictions"
                        if metric in {"alerts", "alerts_per_1000"}
                        else "plug-in ratio or total derived from AIPW components"
                    ),
                    "bootstrap_replicates": len(bootstrap),
                    "bootstrap_unit": config["canonical_analysis"]["bootstrap_unit"],
                    "canonical_source": "data/processed/canonical_bootstrap_distribution_v32.parquet",
                }
            )
    return pd.DataFrame(rows)


def mnar_interval_table(
    point_tables: dict[str, pd.DataFrame],
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    metric_map = {
        "update_alpha": "update_alpha",
        "update_beta": "update_beta",
        "target_oe_ratio": "target_oe_ratio",
        "target_calibration_slope": "target_calibration_slope",
        "target_auroc": "target_auroc",
        "target_brier": "target_brier",
        "threshold_5_alerts": "threshold_5_alerts",
        "threshold_5_alerts_per_1000": "threshold_5_alerts_per_1000",
        "threshold_10_alerts": "threshold_10_alerts",
        "threshold_10_alerts_per_1000": "threshold_10_alerts_per_1000",
        "threshold_20_alerts": "threshold_20_alerts",
        "threshold_20_alerts_per_1000": "threshold_20_alerts_per_1000",
    }
    rows: list[dict[str, Any]] = []
    for stage in ["source", "update", "target"]:
        points = point_tables[stage].loc[~point_tables[stage]["is_canonical"]]
        for point in points.to_dict(orient="records"):
            multiplier = float(point["odds_multiplier"])
            subset = bootstrap.loc[
                bootstrap["stage_varied"].eq(stage)
                & np.isclose(bootstrap["odds_multiplier"], multiplier)
            ]
            if subset["replicate"].nunique() != bootstrap["replicate"].nunique():
                raise RuntimeError(f"Incomplete MNAR bootstrap for {stage}, multiplier={multiplier}")
            for metric, column in metric_map.items():
                lower, upper = subset[column].quantile([0.025, 0.975]).tolist()
                rows.append(
                    {
                        "stage_varied": stage,
                        "estimator_family": point["estimator_family"],
                        "reference_type": point["reference_type"],
                        "is_canonical": False,
                        "odds_multiplier": multiplier,
                        "log_odds_shift": math.log(multiplier),
                        "metric": metric,
                        "estimate": point[metric],
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "bootstrap_replicates": subset["replicate"].nunique(),
                        "bootstrap_unit": "retained patient/operation",
                        "canonical_source": "data/processed/mnar_chain_bootstrap_v32.parquet",
                        "interpretation": "parameterized OR-completion sensitivity with sampling interval; multiplier is not asserted clinically plausible",
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
    spec = v31.model_specification(model, alpha, beta, cohorts, config)
    spec["analysis_version"] = config["analysis_version"]
    spec["preprocessor_training_population"] = "all eligible INSPIRE patients"
    spec["preprocessor_training_n"] = len(cohorts["source"])
    spec["classifier_training_population"] = "outcome-observed INSPIRE patients"
    spec["classifier_training_n"] = int(observed_indicator(cohorts["source"]).sum())
    spec["classifier_weighting"] = "normalized inverse observation probability truncated at the 1st and 99th percentiles"
    spec["fractional_logistic_implementation"] = (
        "Each fractional row is represented by event and non-event copies weighted y and 1-y; "
        "the same L2 logistic classifier is then fitted."
    )
    spec["not_for_clinical_use"] = True
    return spec


def main_tables(
    cohorts: dict[str, pd.DataFrame],
    comparators: pd.DataFrame,
    canonical: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    flow = cohort_flow_table_v32(cohorts).rename(
        columns={"denominator_note": "denominator_definition"}
    )
    table2 = comparators.loc[
        comparators["primary_or_required_comparator"],
        [
            "strategy",
            "update_alpha",
            "update_beta",
            "events",
            "oe_ratio",
            "calibration_slope",
            "auroc",
            "brier",
        ],
    ].copy()
    metrics = ["oe_ratio", "calibration_slope", "auroc", "auprc", "brier"]
    table3 = canonical.loc[canonical["metric"].isin(metrics)].copy()
    workload_metrics_keep = [
        "alerts",
        "alerts_per_1000",
        "true_positives_per_1000",
        "false_positives_per_1000",
        "positive_predictive_value",
        "absolute_net_benefit",
        "incremental_net_benefit_vs_better_default",
    ]
    rows: list[dict[str, Any]] = []
    for threshold in [5, 10, 20]:
        row: dict[str, Any] = {"threshold_percent": threshold}
        for metric in workload_metrics_keep:
            found = canonical.loc[
                canonical["estimand"].str.endswith(f"at {threshold}%")
                & canonical["metric"].eq(metric)
            ].iloc[0]
            row[f"{metric}_estimate"] = found["estimate"]
            row[f"{metric}_ci_lower"] = found["ci_lower"]
            row[f"{metric}_ci_upper"] = found["ci_upper"]
        rows.append(row)
    return {"table1": flow, "table2": table2, "table3": table3, "table4": pd.DataFrame(rows)}


def metric_reporting_map() -> pd.DataFrame:
    rows = [
        ("Update alpha and beta", "Table 2", "Table S8", "31_canonical_primary_bootstrap_v32.csv"),
        ("Canonical target performance", "Table 3", "Table S10", "31_canonical_primary_bootstrap_v32.csv"),
        ("Operational workload", "Table 4", "Table S10", "31_canonical_primary_bootstrap_v32.csv"),
        ("Source-stage MNAR", "Results", "Table S11 / Figure S3", "43_source_mnar_propagation_v32.csv"),
        ("Update-stage MNAR", "Results", "Table S12", "38_update_mnar_propagation_v32.csv"),
        ("Target-stage MNAR", "Results", "Table S13", "40_target_mnar_sensitivity_v32.csv"),
        ("MNAR sampling intervals", "Results", "Table S14", "47_mnar_chain_bootstrap_intervals_v32.csv"),
        ("Auxiliary outcome-model diagnostics", "Methods/Results", "Table S4", "44_auxiliary_outcome_model_diagnostics_v32.csv"),
        ("Weight truncation sensitivity", "Results", "Table S15", "46_weight_truncation_sensitivity_v32.csv"),
        (
            "10-quantile grouped absolute calibration error",
            "Methods / Table 3",
            "Tables S4 and S10",
            "31_canonical_primary_bootstrap_v32.csv",
        ),
    ]
    return pd.DataFrame(rows, columns=["content", "main_location", "supplement_location", "unique_numeric_source"])


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def set_odds_multiplier_axis(ax: plt.Axes, include_intermediate: bool = True) -> None:
    """Show only the prespecified MNAR multipliers and suppress log minor labels."""
    ticks = [0.5, 0.67, 1, 1.5, 2, 3] if include_intermediate else [0.5, 1, 2, 3]
    labels = ["0.5", "0.67", "1", "1.5", "2", "3"] if include_intermediate else ["0.5", "1", "2", "3"]
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.46, 3.25)


def build_sensitivity_figures(
    source_mnar: pd.DataFrame,
    update_mnar: pd.DataFrame,
    target_mnar: pd.DataFrame,
    intervals: pd.DataFrame,
    cohorts: dict[str, pd.DataFrame],
    bundles: dict[str, dict[str, Any]],
) -> None:
    source = source_mnar.loc[~source_mnar["is_canonical"]].sort_values("odds_multiplier")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    for ax, metric, label in [
        (axes[0, 0], "target_oe_ratio", "Target O/E ratio"),
        (axes[0, 1], "target_calibration_slope", "Target calibration slope"),
        (axes[1, 0], "target_auroc", "Target AUROC"),
        (axes[1, 1], "threshold_10_alerts", "Alerts at 10%"),
    ]:
        ax.plot(source["odds_multiplier"], source[metric], marker="o", color="#087E8B")
        subset = intervals.loc[
            intervals["stage_varied"].eq("source") & intervals["metric"].eq(metric)
        ].sort_values("odds_multiplier")
        if len(subset) == len(source):
            ax.fill_between(
                subset["odds_multiplier"], subset["ci_lower"], subset["ci_upper"], color="#087E8B", alpha=0.16
            )
        ax.axvline(1.0, color="#555555", linestyle="--", linewidth=1)
        if metric in {"target_oe_ratio", "target_calibration_slope"}:
            ax.axhline(1.0, color="#999999", linestyle=":", linewidth=1)
        set_odds_multiplier_axis(ax)
        ax.set_xlabel("Source unobserved-outcome odds multiplier")
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    save_figure(fig, "Figure_S3_source_mnar_propagation_v32")

    fig, axes = plt.subplots(3, 4, figsize=(12, 8.2), sharex=True, constrained_layout=True)
    for row_index, (stage, table) in enumerate(
        [("source", source_mnar), ("update", update_mnar), ("target", target_mnar)]
    ):
        subset = table.loc[~table["is_canonical"]].sort_values("odds_multiplier")
        canonical = table.loc[table["is_canonical"]].iloc[0]
        for col_index, (metric, label) in enumerate(
            [
                ("target_oe_ratio", "O/E"),
                ("target_calibration_slope", "Slope"),
                ("target_auroc", "AUROC"),
                ("threshold_10_alerts", "10% alerts"),
            ]
        ):
            ax = axes[row_index, col_index]
            ax.plot(subset["odds_multiplier"], subset[metric], marker="o", color="#C8553D")
            ax.axhline(canonical[metric], color="#087E8B", linestyle="--", linewidth=1.2)
            ax.axvline(1.0, color="#555555", linestyle=":", linewidth=1)
            set_odds_multiplier_axis(ax, include_intermediate=False)
            ax.grid(axis="y", color="#E2E2E2", linewidth=0.5)
            if row_index == 0:
                ax.set_title(label)
            if col_index == 0:
                ax.set_ylabel(f"{stage.capitalize()} varied")
            if row_index == 2:
                ax.set_xlabel("Odds multiplier")
    save_figure(fig, "Figure_S4_mnar_chain_overview_v32")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for ax, (key, phase) in zip(
        axes, [("source", "INSPIRE"), ("update", "MOVER 2021"), ("target", "MOVER 2022")]
    ):
        observed = bundles[key]["observed"]
        prediction = bundles[key]["outcome_probability"]
        ax.hist(prediction[observed], bins=30, density=True, alpha=0.55, color="#087E8B", label="Observed")
        ax.hist(prediction[~observed], bins=30, density=True, alpha=0.5, color="#F2B134", label="Unobserved")
        ax.set_title(phase)
        ax.set_xlabel("Cross-fitted modeled AKI probability")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "Figure_S5_auxiliary_outcome_probability_v32")


def run_point_analysis(
    cohorts: dict[str, pd.DataFrame], config: dict[str, Any], write_outputs: bool
) -> dict[str, Any]:
    seed = int(config["random_seed"])
    bundles = {
        "source": observation_bundle(
            cohorts["source"], "INSPIRE", config, seed + 1000, include_basic=True, fit_outcome=True, fit_final=True
        ),
        "update": observation_bundle(
            cohorts["update"], "MOVER 2021", config, seed + 2000, include_basic=True, fit_outcome=True, fit_final=True
        ),
        "target": observation_bundle(
            cohorts["target"], "MOVER 2022", config, seed + 3000, include_basic=True, fit_outcome=True, fit_final=True
        ),
    }
    canonical = canonical_result(
        cohorts["source"],
        cohorts["update"],
        cohorts["target"],
        bundles["source"],
        bundles["update"],
        bundles["target"],
        config,
        seed + 4000,
    )
    source_mnar = source_mnar_propagation(cohorts, bundles, canonical, config, seed + 5000)
    update_mnar = update_mnar_propagation(cohorts, bundles, canonical, config)
    target_mnar = target_mnar_sensitivity(cohorts, bundles, canonical, config)
    anchors = mnar_anchor_table(source_mnar, update_mnar, target_mnar)
    auxiliary, fold_checks = auxiliary_outcome_diagnostics(cohorts, bundles)
    truncation = weight_truncation_sensitivity(cohorts, bundles, config, seed + 6000)
    comparators, comparator_workload = selection_chain_comparators(
        cohorts, bundles, config, seed + 7000
    )
    bounds = target_nonparametric_bounds(
        cohorts["target"], bundles["target"], canonical["prediction"], config
    )
    vitaldb = vitaldb_supportive(cohorts, canonical)
    fractional_test = fractional_logistic_equivalence_test(cohorts["source"], seed + 8000)
    if fractional_test["status"] != "PASS":
        raise RuntimeError(f"Fractional logistic equivalence test failed: {fractional_test}")

    if write_outputs:
        flow = cohort_flow_table_v32(cohorts)
        characteristics = v31.group_characteristics(cohorts)
        diagnostics = pd.concat(
            [bundles[key]["diagnostics"] for key in ["source", "update", "target"]],
            ignore_index=True,
        )
        balance = pd.concat(
            [bundles[key]["balance"] for key in ["source", "update", "target"]],
            ignore_index=True,
        )
        update_table = pd.concat(
            [
                anchors.loc[anchors["stage_varied"].eq("update")],
                update_mnar.loc[~update_mnar["is_canonical"]],
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["stage_varied", "reference_type", "odds_multiplier"])
        write_csv(flow, "01_cohort_flow_and_roles_v32.csv")
        write_csv(characteristics, "27_source_update_observation_flow_v32.csv")
        write_csv(diagnostics, "28_stage_observation_diagnostics_v32.csv")
        write_csv(update_table, "29_mover2021_recalibration_estimands_v32.csv")
        write_csv(comparators, "30_update_selection_target_performance_v32.csv")
        write_csv(balance, "32_stage_observation_balance_v32.csv")
        write_csv(v31.missingness_table(cohorts), "33_predictor_missingness_v32.csv")
        write_csv(comparator_workload, "34_update_strategy_workload_v32.csv")
        write_csv(comparators, "37_selection_chain_primary_comparators_v32.csv")
        write_csv(update_mnar, "38_update_mnar_propagation_v32.csv")
        write_csv(metric_reporting_map(), "39_metric_reporting_map_v32.csv")
        write_csv(target_mnar, "40_target_mnar_sensitivity_v32.csv")
        write_csv(bounds, "41_target_nonparametric_bounds_v32.csv")
        write_csv(anchors, "42_mnar_anchor_comparison_v32.csv")
        write_csv(source_mnar, "43_source_mnar_propagation_v32.csv")
        write_csv(auxiliary, "44_auxiliary_outcome_model_diagnostics_v32.csv")
        write_csv(preprocessing_audit(canonical["source_model"], cohorts["source"]), "45_source_preprocessing_audit_v32.csv")
        write_csv(truncation, "46_weight_truncation_sensitivity_v32.csv")
        write_csv(vitaldb, "49_vitaldb_supportive_v32.csv")
        write_csv(fold_checks, "auxiliary_outcome_fold_checks_v32.csv", QA_DIR)
        write_json(fractional_test, QA_DIR / "fractional_logistic_equivalence_v32.json")
        joblib.dump(canonical["source_model"], MODEL_DIR / "source_logistic_spline_ipw_v32.joblib")
        for key in ["source", "update", "target"]:
            joblib.dump(bundles[key]["expanded_model"], MODEL_DIR / f"{key}_observation_model_expanded_v32.joblib")
            joblib.dump(bundles[key]["outcome_model"], MODEL_DIR / f"{key}_auxiliary_outcome_model_v32.joblib")
        spec = model_specification(
            canonical["source_model"], canonical["alpha"], canonical["beta"], cohorts, config
        )
        write_json(spec, MODEL_DIR / "model_specification_v32.json")
    return {
        "bundles": bundles,
        "canonical": canonical,
        "source_mnar": source_mnar,
        "update_mnar": update_mnar,
        "target_mnar": target_mnar,
        "anchors": anchors,
        "auxiliary": auxiliary,
        "fold_checks": fold_checks,
        "truncation": truncation,
        "comparators": comparators,
        "comparator_workload": comparator_workload,
        "bounds": bounds,
        "vitaldb": vitaldb,
        "fractional_test": fractional_test,
    }


def write_analysis_reports(
    config: dict[str, Any],
    canonical: pd.DataFrame,
    intervals: pd.DataFrame,
    failures: pd.DataFrame,
    started: datetime,
    elapsed: float,
) -> None:
    def metric(name: str) -> pd.Series:
        return canonical.loc[canonical["metric"].eq(name)].iloc[0]

    oe = metric("oe_ratio")
    slope = metric("calibration_slope")
    source_range = intervals.loc[
        intervals["stage_varied"].eq("source") & intervals["metric"].eq("target_oe_ratio")
    ]
    report = f"""# v3.2 canonical and stage-specific MNAR results

Status: **SCIENTIFIC ANALYSIS COMPLETE; NOT READY FOR SUBMISSION**. Administrative, authorship, ethics, license, and independent-review gates remain open.

The source preprocessor was fitted in all 33,396 eligible INSPIRE patients, while the canonical classifier was fitted in the 24,874 outcome-observed patients using normalized inverse-observation weights truncated at the 1st and 99th percentiles. The same sequence was repeated within every bootstrap sample.

The canonical MOVER 2022 O/E estimate was {oe['estimate']:.3f} (95% percentile interval {oe['ci_lower']:.3f} to {oe['ci_upper']:.3f}), and the calibration slope was {slope['estimate']:.3f} ({slope['ci_lower']:.3f} to {slope['ci_upper']:.3f}). These are measured-variable MAR estimates, not identified full-population truths.

Source-, update-, and target-stage outcome-regression completion scenarios used fixed odds multipliers 0.5, 0.67, 1, 1.5, 2, and 3. Multiplier 1 is the outcome-regression completion reference and is not numerically constrained to equal the canonical IPW/AIPW estimate. All {int(config['bootstrap_replicates'])} valid patient-level replicates refitted observation models, auxiliary outcome models, the full-eligible source preprocessor, the source classifier, local recalibration, and target metrics. The source-stage target O/E point estimates over the prespecified multipliers ranged from {source_range['estimate'].min():.3f} to {source_range['estimate'].max():.3f}.

Clinical utility was not evaluated. The results do not establish readiness for clinical deployment.
"""
    (REPORT_DIR / "22_v32_canonical_and_mnar_results.md").write_text(report, encoding="utf-8")
    write_json(
        {
            "analysis_version": config["analysis_version"],
            "status": config["analysis_status"],
            "deployment_status": "not_evaluated_for_clinical_use",
            "peer_review_status": "not_peer_reviewed",
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": elapsed,
            "bootstrap_replicates": int(config["bootstrap_replicates"]),
            "mnar_scenarios_per_replicate": 18,
            "mnar_rows": int(config["bootstrap_replicates"]) * 18,
            "recorded_failed_attempts": len(failures),
            "random_seed": int(config["random_seed"]),
            "parallel_workers": int(config["canonical_analysis"]["parallel_workers"]),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "canonical_table_sha256": sha256_file(TABLE_DIR / "31_canonical_primary_bootstrap_v32.csv"),
            "canonical_distribution_sha256": sha256_file(DATA_DIR / "canonical_bootstrap_distribution_v32.parquet"),
            "mnar_distribution_sha256": sha256_file(DATA_DIR / "mnar_chain_bootstrap_v32.parquet"),
            "mnar_interval_table_sha256": sha256_file(TABLE_DIR / "47_mnar_chain_bootstrap_intervals_v32.csv"),
        },
        REPORT_DIR / "23_v32_analysis_run_record.json",
    )


def write_bootstrap_assertions(
    config: dict[str, Any],
    canonical: pd.DataFrame,
    mnar: pd.DataFrame,
    intervals: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    expected = int(config["bootstrap_replicates"])
    checks: list[tuple[str, bool, str]] = []
    checks.append(("canonical_valid_replicates", canonical["replicate"].nunique() == expected, str(canonical["replicate"].nunique())))
    for stage in ["source", "update", "target"]:
        for multiplier in config["mnar_odds_multipliers"]:
            count = mnar.loc[
                mnar["stage_varied"].eq(stage) & np.isclose(mnar["odds_multiplier"], float(multiplier)),
                "replicate",
            ].nunique()
            checks.append((f"{stage}_k_{float(multiplier):g}_valid_replicates", count == expected, str(count)))
    point_tables = {
        "source": pd.read_csv(TABLE_DIR / "43_source_mnar_propagation_v32.csv"),
        "update": pd.read_csv(TABLE_DIR / "38_update_mnar_propagation_v32.csv"),
        "target": pd.read_csv(TABLE_DIR / "40_target_mnar_sensitivity_v32.csv"),
    }
    point_mismatches: list[str] = []
    max_difference = 0.0
    for row in intervals.to_dict(orient="records"):
        source = point_tables[str(row["stage_varied"])]
        source = source.loc[
            ~source["is_canonical"].astype(bool)
            & np.isclose(source["odds_multiplier"], float(row["odds_multiplier"]))
        ]
        if len(source) != 1 or row["metric"] not in source.columns:
            point_mismatches.append(
                f"{row['stage_varied']}|{row['odds_multiplier']}|{row['metric']}:missing"
            )
            continue
        expected_point = float(source.iloc[0][row["metric"]])
        difference = abs(float(row["estimate"]) - expected_point)
        max_difference = max(max_difference, difference)
        if difference > 1e-12:
            point_mismatches.append(
                f"{row['stage_varied']}|{row['odds_multiplier']}|{row['metric']}:{difference:.3e}"
            )
    checks.append(
        (
            "interval_point_estimates_match_point_tables",
            not point_mismatches,
            f"mismatches={len(point_mismatches)}; max_difference={max_difference:.3e}",
        )
    )
    checks.append(("interval_point_estimates_finite", np.isfinite(intervals["estimate"]).all(), f"rows={len(intervals)}"))
    checks.append(("all_bootstrap_values_finite", np.isfinite(mnar.select_dtypes(include=[np.number])).all().all(), f"rows={len(mnar)}"))
    lines = [
        f"analysis_version={config['analysis_version']}",
        f"requested_replicates={expected}",
        f"recorded_failed_attempts={len(failures)}",
        "resampled=INSPIRE,MOVER_2021,MOVER_2022",
        "refitted=observation_models,auxiliary_outcome_models,source_preprocessor,source_classifier,local_recalibration,target_metrics",
        f"seed={config['random_seed']}",
    ]
    lines.extend(
        f"{name}={'PASS' if passed else 'FAIL'}; detail={detail}" for name, passed, detail in checks
    )
    lines.append(f"status={'PASS' if all(item[1] for item in checks) else 'FAIL'}")
    (QA_DIR / "mnar_bootstrap_assertions_v32.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not all(item[1] for item in checks):
        raise RuntimeError("MNAR bootstrap assertions failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--point-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    if args.bootstrap_replicates is not None:
        if args.bootstrap_replicates < 2:
            raise ValueError("At least two bootstrap replicates are required")
        config["bootstrap_replicates"] = int(args.bootstrap_replicates)
    for directory in [TABLE_DIR, MODEL_DIR, FIGURE_DIR, REPORT_DIR, QA_DIR, DATA_DIR, TMP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    clock = time.time()
    print("Loading complete eligible cohorts", flush=True)
    cohorts = prepare_cohorts()
    print("Fitting v3.2 point-estimate chain and prespecified sensitivities", flush=True)
    point = run_point_analysis(cohorts, config, write_outputs=not args.diagnostic)
    if args.point_only:
        print("v3.2 point analysis complete", flush=True)
        return 0
    output_dir = TMP_DIR if args.diagnostic else DATA_DIR
    jobs = int(args.jobs or config["canonical_analysis"].get("parallel_workers", 1))
    print(
        f"Running {config['bootstrap_replicates']} combined canonical/MNAR bootstrap replicates with {jobs} workers",
        flush=True,
    )
    canonical_distribution, mnar_distribution, failures = run_bootstrap(
        config, int(config["bootstrap_replicates"]), output_dir, jobs
    )
    canonical = canonical_long_table(point["canonical"], canonical_distribution, config)
    interval_table = mnar_interval_table(
        {
            "source": point["source_mnar"],
            "update": point["update_mnar"],
            "target": point["target_mnar"],
        },
        mnar_distribution,
    )
    if args.diagnostic:
        write_csv(canonical, "31_canonical_primary_bootstrap_v32_diagnostic.csv", TMP_DIR)
        write_csv(interval_table, "47_mnar_chain_bootstrap_intervals_v32_diagnostic.csv", TMP_DIR)
        print(
            canonical.loc[
                canonical["metric"].isin(["oe_ratio", "calibration_slope", "brier"])
            ].to_json(orient="records", indent=2),
            flush=True,
        )
        return 0

    write_csv(canonical, "31_canonical_primary_bootstrap_v32.csv")
    write_csv(interval_table, "47_mnar_chain_bootstrap_intervals_v32.csv")
    write_csv(failures, "mnar_bootstrap_failures_v32.csv", QA_DIR)
    tables = main_tables(cohorts, point["comparators"], canonical)
    write_csv(tables["table1"], "35_main_table1_cohorts_v32.csv")
    write_csv(tables["table2"], "36_main_table2_propagation_v32.csv")
    write_csv(tables["table3"], "37_main_table3_canonical_performance_v32.csv")
    write_csv(tables["table4"], "38_main_table4_workload_v32.csv")
    build_sensitivity_figures(
        point["source_mnar"],
        point["update_mnar"],
        point["target_mnar"],
        interval_table,
        cohorts,
        point["bundles"],
    )
    write_bootstrap_assertions(config, canonical_distribution, mnar_distribution, interval_table, failures)
    elapsed = time.time() - clock
    write_analysis_reports(config, canonical, interval_table, failures, started, elapsed)
    print(f"v3.2 analysis complete in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
