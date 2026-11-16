"""Unchanged R2 calculation functions extracted from authoritative sources.
Path configuration and orchestration are supplied by portable.py, not this module.
Historical report/author metadata generators and legacy entry points are excluded.
"""
from __future__ import annotations


import numpy as np
from pathlib import Path


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


def ge_threshold(left, right):
    # The tolerance only absorbs binary floating-point roundoff at an exact boundary.
    return (left >= right) | np.isclose(left, right, rtol=0.0, atol=1e-12)
