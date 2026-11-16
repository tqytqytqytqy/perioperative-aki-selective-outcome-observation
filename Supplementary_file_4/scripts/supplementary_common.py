"""Unchanged R2 calculation functions extracted from authoritative sources.
Path configuration and orchestration are supplied by portable.py, not this module.
Historical report/author metadata generators and legacy entry points are excluded.
"""
from __future__ import annotations


from pathlib import Path
from decimal import Decimal
import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd


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


def threshold_label(baseline, max48, max7, observed):
    if not observed:
        return np.nan
    b = Decimal(str(baseline))
    tol = Decimal("1e-12")
    a = None if pd.isna(max48) else Decimal(str(max48))
    c = None if pd.isna(max7) else Decimal(str(max7))
    assert c is not None, "An observed 7-day outcome requires a 7-day peak"
    return int((a is not None and a-b >= Decimal(".3")-tol) or c >= b*Decimal("1.5")-tol)

def reproduction_difference(calculated, stored, metrics, tolerance=1e-10):
    assert not calculated.strategy.duplicated().any() and not stored.strategy.duplicated().any()
    assert set(calculated.strategy) == set(stored.strategy)
    x = calculated.set_index("strategy").sort_index()[metrics].to_numpy(dtype=float)
    y = stored.set_index("strategy").sort_index()[metrics].to_numpy(dtype=float)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    difference = float(np.max(np.abs(x-y)))
    assert difference < tolerance, f"Current point estimate mismatch: {difference:.6g}"
    return difference

def safe_groups(prediction, observed, y):
    for q in [10, 5]:
        codes, edges = pd.qcut(prediction, q, labels=False, retbins=True, duplicates="drop")
        assert np.isfinite(codes).all()
        groups = sorted(np.unique(codes))
        safe = bool(groups) and all(min(y[codes[observed] == g].sum(),
            len(y[codes[observed] == g])-y[codes[observed] == g].sum()) >= 5 for g in groups)
        if safe:
            return np.asarray(codes, dtype=int), edges
    raise AssertionError("No original-rule grouping meets >=5 observed events AND non-events per bin")

def conditional_bootstrap(codes, weights, outcomes, seed, replicates):
    n = len(codes)
    groups = sorted(np.unique(codes))
    denom = np.array([weights[codes == g].sum() for g in groups])
    assert (denom > 0).all()
    point = np.array([(weights[codes == g]*outcomes[codes == g]).sum() / denom[j] for j, g in enumerate(groups)])
    rng = np.random.default_rng(seed)
    boot = np.zeros((replicates, len(groups)))
    for i in range(replicates):
        ix = rng.integers(0, n, n)
        for j, g in enumerate(groups):
            sub = ix[codes[ix] == g]
            denominator = weights[sub].sum()
            assert denominator > 0, "Empty weighted bootstrap group: no redraw or silent deletion allowed"
            boot[i, j] = (weights[sub]*outcomes[sub]).sum() / denominator
    assert np.isfinite(boot).all()
    return point, np.quantile(boot, [.025, .975], axis=0), boot
