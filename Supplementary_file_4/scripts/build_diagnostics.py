"""Unchanged R2 calculation functions extracted from authoritative sources.
Path configuration and orchestration are supplied by portable.py, not this module.
Historical report/author metadata generators and legacy entry points are excluded.
"""
from __future__ import annotations


import os
import sys
import hashlib
import importlib.util
import json
import math
import unittest
from decimal import Decimal
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import timing_helpers as timing
import audit_mover as audit


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


def save(rows, name):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    forbidden = {"MRN", "LOG_ID", "PAT_ID", "case_key", "patient_key", "patient_source", "_group"}
    assert not forbidden.intersection(frame.columns), name
    frame.to_csv(EXACT / name, index=False)
    return frame

def first_rows(f):
    ordered = f.sort_values(["patient_source", "an_start", "LOG_ID"])
    return ordered.loc[~ordered.patient_source.duplicated()].copy()

def category_smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return 0.0
    delta = (a - b)[1:]
    cov = ((np.diag(a) - np.outer(a, a) + np.diag(b) - np.outer(b, b)) / 2)[1:, 1:]
    inv = np.linalg.pinv(cov, hermitian=True, rcond=1e-12)
    if np.linalg.norm(cov @ inv @ delta - delta) > 1e-9:
        return np.nan
    return float(np.sqrt(max(0, delta @ inv @ delta)))

def selection_tables(original, consistent):
    denominators, comparisons, categories = [], [], []
    for scenario, f in [("historical_LOG_ID_original_first", original),
                        ("main_original_first_then_identityQC", consistent)]:
        for year in ["all", 2021, 2022]:
            g = f if year == "all" else f.loc[f.year.eq(year)]
            av = g.loc[g.has_baseline_cr]
            mi = g.loc[~g.has_baseline_cr]
            denominators.append(dict(scenario=scenario, year=year, index_before_baseline_n=len(g),
                baseline_available_n=len(av), baseline_missing_n=len(mi),
                baseline_missing_proportion=len(mi) / len(g),
                baseline_ge4_n=int(av.baseline_cr.ge(4).sum()),
                baseline_under4_n=int(g.baseline_cr_under4.sum()),
                available_tested_n=int(av.tested_7d.sum()), missing_tested_n=int(mi.tested_7d.sum()),
                available_sex_known_n=int(av.sex_observed.notna().sum()),
                missing_sex_known_n=int(mi.sex_observed.notna().sum()),
                available_asa_missing_n=int(av.asa.isna().sum()),
                missing_asa_missing_n=int(mi.asa.isna().sum())))
            specs = [("age", False), ("duration_h", False), ("sex_observed", True), ("tested_7d", True)]
            for col, binary in specs:
                x = pd.to_numeric(av[col], errors="coerce").dropna().astype(float)
                y = pd.to_numeric(mi[col], errors="coerce").dropna().astype(float)
                var = ((x.mean() * (1 - x.mean()) + y.mean() * (1 - y.mean())) / 2 if binary
                       else (x.var(ddof=1) + y.var(ddof=1)) / 2)
                smd = ((x.mean() - y.mean()) / np.sqrt(var) if var > 0
                       else (0 if x.mean() == y.mean() else np.nan))
                comparisons.append(dict(scenario=scenario, year=year, variable=col,
                    available_group_n=len(av), missing_group_n=len(mi),
                    available_nonmissing_n=len(x), missing_nonmissing_n=len(y),
                    available_missing_n=len(av)-len(x), missing_missing_n=len(mi)-len(y),
                    available_mean=x.mean(), missing_mean=y.mean(),
                    available_sd=x.std(ddof=1), missing_sd=y.std(ddof=1),
                    available_median=x.median(), missing_median=y.median(), signed_smd=smd))
            for col in ["asa", "surgery_category"]:
                levels = sorted(g[col].dropna().unique())
                x, y = av[col].dropna(), mi[col].dropna()
                pa = [float(x.eq(v).mean()) for v in levels]
                pb = [float(y.eq(v).mean()) for v in levels]
                categories.append(dict(scenario=scenario, year=year, variable=col, level="GLOBAL_NOMINAL",
                    available_denominator=len(x), missing_denominator=len(y),
                    available_missing_n=len(av)-len(x), missing_missing_n=len(mi)-len(y),
                    overall_categorical_smd=category_smd(pa, pb)))
                for lev, a, b in zip(levels, pa, pb):
                    den = math.sqrt((a * (1-a) + b * (1-b))/2)
                    categories.append(dict(scenario=scenario, year=year, variable=col, level=str(lev),
                        available_denominator=len(x), missing_denominator=len(y),
                        available_n=int(x.eq(lev).sum()), missing_n=int(y.eq(lev).sum()),
                        available_proportion=a, missing_proportion=b,
                        signed_smd=(a-b)/den if den else (0 if a == b else np.nan)))
    save(denominators, "A1_baseline_denominators_exact.csv")
    save(comparisons, "A2_selection_characteristics_exact.csv")
    save(categories, "A2_ASA_surgery_nominal_exact.csv")
    return denominators, comparisons, categories

def coverage_tables(records):
    overall, risk, stop, combined_stop, daily, sens = [], [], [], [], [], []
    for cohort, g in records.groupby("cohort", sort=False):
        n = len(g)
        obs = g.loc[g.tested]
        masks = {
            "Any postoperative test": (g.tested, n),
            "No postoperative test": (~g.tested, n),
            "Test in (0,48] h": (g.early, n),
            "Test in (48,168] h before recorded stop": (g.late, n),
            "Tests in both early and late windows": (g.both, n),
            "Only one distinct testing time": (g.occasions.eq(1), n),
            "Algorithmic observation window <48 h": (g.cap_h.lt(48), n),
            "Algorithmic observation window <168 h": (g.cap_h.lt(168), n),
            "Discharge stop time missing or unusable": (g.discharge_unusable, n),
            "Early test among algorithmic window >=48 h": (g.loc[g.cap_h.ge(48), "early"], int(g.cap_h.ge(48).sum())),
            "Late test among algorithmic window >48 h": (g.loc[g.cap_h.gt(48), "late"], int(g.cap_h.gt(48).sum())),
        }
        for metric, (arr, denominator) in masks.items():
            k = int(arr.sum())
            low, high = timing.wilson(k, denominator)
            overall.append(dict(cohort=cohort, metric=metric, n=k, denominator=denominator,
                percent=100*k/denominator if denominator else np.nan, ci_low=low, ci_high=high,
                display=timing.prop(k, denominator)))
        for metric, col, sub in [
            ("First test hours among observed", "first_h", obs),
            ("Last test hours among observed", "last_h", obs),
            ("Testing occasions among observed", "occasions", obs),
            ("Elapsed 24 h bins with test among observed", "tested_bins", obs),
            ("Gap from last test to recorded stop among observed", "gap_to_end_h", obs),
            ("Observation opportunity hours", "cap_h", g)]:
            q = sub[col].quantile([.25, .5, .75])
            overall.append(dict(cohort=cohort, metric=metric, denominator=len(sub),
                q25=q.loc[.25], median=q.loc[.5], q75=q.loc[.75], display=timing.qstr(sub[col])))
        def group_row(sub, label):
            observed = sub.loc[sub.tested]
            late = sub.loc[sub.cap_h.gt(48)]
            return dict(cohort=cohort, group=label, eligible_n=len(sub), observed_n=int(sub.tested.sum()),
                observed_pct=100*sub.tested.mean(), first_test_h_median_iqr=timing.qstr(observed.first_h),
                occasions_median_iqr=timing.qstr(observed.occasions),
                opportunity_h_median_iqr=timing.qstr(sub.cap_h),
                early_test_n=int(sub.early.sum()), early_test_pct=100*sub.early.mean(),
                late_test_n=int(sub.late.sum()), late_test_pct=100*sub.late.mean(),
                late_opportunity_n=len(late), late_among_opportunity_n=int(late.late.sum()),
                late_among_opportunity_pct=100*late.late.mean() if len(late) else np.nan)
        for label, sub in g.groupby("risk_quintile", sort=True):
            row = group_row(sub, label)
            row.update(source_risk_min=sub.source_risk.min(), source_risk_max=sub.source_risk.max(),
                       source_risk_median=sub.source_risk.median())
            risk.append(row)
        for label, sub in g.groupby("stop_group", sort=True):
            stop.append(group_row(sub, label))
        for label, sub in g.groupby(np.where(g.cap_h.le(48), "<=48 h", ">48 h"), sort=True):
            combined_stop.append(group_row(sub, label))
        for d in range(1, 8):
            left, right = 24*(d-1), 24*d
            any_opp, full = g.cap_h.gt(left), g.cap_h.ge(right)
            k, kfull = int(g[f"day{d}"].sum()), int(g.loc[full, f"day{d}"].sum())
            low, high = timing.wilson(k, int(any_opp.sum()))
            flo, fhi = timing.wilson(kfull, int(full.sum()))
            daily.append(dict(cohort=cohort, elapsed_day=d, window_hours=f"({left},{right}]",
                tested_n=k, all_eligible_n=n, any_opportunity_n=int(any_opp.sum()),
                full_opportunity_n=int(full.sum()), tested_full_opportunity_n=kfull,
                all_eligible_pct=100*k/n, any_opportunity_pct=100*k/any_opp.sum(),
                full_opportunity_pct=100*kfull/full.sum() if full.sum() else np.nan,
                any_opportunity_ci_low=low, any_opportunity_ci_high=high,
                full_opportunity_ci_low=flo, full_opportunity_ci_high=fhi))
        choices = [("early", g.cap_h.ge(48), "early"), ("late", g.cap_h.gt(48), "late")]
        for d in range(1, 8):
            choices += [(f"day{d}_any", g.cap_h.gt(24*(d-1)), f"day{d}"),
                        (f"day{d}_full", g.cap_h.ge(24*d), f"day{d}")]
        for metric, mask, col in choices:
            a, b = g.loc[mask, col], g.loc[mask & ~g.discharge_unusable, col]
            sens.append(dict(cohort=cohort, metric=metric, algorithmic_n=len(a),
                confirmed_discharge_n=len(b), algorithmic_pct=100*a.mean(),
                confirmed_discharge_pct=100*b.mean(), difference_pp=100*(b.mean()-a.mean())))
    for rows, name in [(overall, "S28_overall_timing_exact.csv"), (risk, "S29_source_risk_strata_exact.csv"),
                       (stop, "S30_followup_strata_exact.csv"), (daily, "S31_elapsed_day_coverage_exact.csv"),
                       (combined_stop, "S30_followup_combined_2groups_exact.csv"),
                       (sens, "confirmed_stop_sensitivity_exact.csv")]:
        save(rows, name)
    return pd.DataFrame(overall), pd.DataFrame(risk), pd.DataFrame(stop), pd.DataFrame(daily), pd.DataFrame(sens)
