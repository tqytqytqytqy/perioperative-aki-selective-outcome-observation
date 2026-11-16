"""Unchanged R2 calculation functions extracted from authoritative sources.
Path configuration and orchestration are supplied by portable.py, not this module.
Historical report/author metadata generators and legacy entry points are excluded.
"""
from __future__ import annotations


from pathlib import Path
import argparse, csv, gzip, hashlib, json, re, time, zipfile
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit, logit


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


def summarize_times(times, cap):
    x=np.asarray(times,dtype=float)
    x=x[(x>0)&(x<=cap)]
    u=np.unique(x)
    early=u[u<=48]
    late=u[u>48]
    return {'raw_rows':len(x),'raw_rows48':int((x<=48).sum()),'occasions':len(u),
        'tested':len(u)>0,'early':len(early)>0,'late':len(late)>0,'both':len(early)>0 and len(late)>0,
        'first_h':float(u[0]) if len(u) else np.nan,'last_h':float(u[-1]) if len(u) else np.nan,
        'gap_to_end_h':float(cap-u[-1]) if len(u) else np.nan,
        'tested_bins':len(np.unique(np.ceil(u/24).astype(int))) if len(u) else 0,
        **{f'day{d}':bool(((u>24*(d-1))&(u<=24*d)).any()) for d in range(1,8)}}

def unit_tests():
    r=summarize_times([0,1,1,24,48,48.001,168,169],168)
    assert r['raw_rows']==6 and r['occasions']==5 and r['raw_rows48']==4
    assert r['tested_bins']==4 and r['both'] and r['day1'] and r['day2'] and r['day3'] and r['day7']
    assert not summarize_times([0,30],24)['tested']
    assert not summarize_times([48,49],48)['late']
    assert summarize_times([48,49],49)['late']
    assert np.isnan(summarize_times([],168)['first_h'])

def wilson(k,n):
    if not n:return np.nan,np.nan
    z=1.959963984540054; p=k/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return 100*(mid-half),100*(mid+half)

def qstr(v):
    x=pd.Series(v).dropna()
    if x.empty:return 'NA'
    a,b,c=np.quantile(x,[.25,.5,.75]); return f'{b:.1f} ({a:.1f}-{c:.1f})'

def prop(k,n):
    return f'{int(k)}/{int(n)} ({100*k/n:.1f}%)' if n else 'NA'
