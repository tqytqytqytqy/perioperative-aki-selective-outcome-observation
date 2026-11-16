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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import joblib
import matplotlib
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
