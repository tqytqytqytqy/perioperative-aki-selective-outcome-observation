"""Unchanged R2 calculation functions extracted from authoritative sources.
Path configuration and orchestration are supplied by portable.py, not this module.
Historical report/author metadata generators and legacy entry points are excluded.
"""
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
import run_postreview_analysis as base


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


FEATURES = ["age", "sex_male", "duration_h", "baseline_cr"]

CONTINUOUS = ["age", "duration_h", "baseline_cr"]

BINARY = ["sex_male"]

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

def observed_indicator(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["tested_7d"].fillna(False).to_numpy(dtype=bool)
        & frame["outcome_operational"].notna().to_numpy(dtype=bool)
    )

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
