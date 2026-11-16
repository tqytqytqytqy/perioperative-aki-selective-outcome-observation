"""Unchanged R2 calculation functions extracted from authoritative sources.
Path configuration and orchestration are supplied by portable.py, not this module.
Historical report/author metadata generators and legacy entry points are excluded.
"""
from __future__ import annotations


import os
import sys,json,hashlib,re,unittest
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from audit_mover import key, NUM


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


def present(x):
    return pd.notna(x) and str(x)!=""

def exact(a,b):
    return present(a) and present(b) and str(a)==str(b)

def unique_triples(c):
    triples=c[["LOG_ID","PAT_ID","MRN"]].drop_duplicates()
    counts=triples.groupby("LOG_ID",dropna=False).size()
    valid=triples.LOG_ID.map(counts).eq(1)&triples[["LOG_ID","PAT_ID","MRN"]].notna().all(axis=1)
    valid &= triples[["LOG_ID","PAT_ID","MRN"]].ne("").all(axis=1)
    return triples.loc[valid].copy(),set(counts[counts.gt(1)].index),triples

def explains_difference(info,lab,triple,info_roles,lab_roles,event_log_id=None):
    if triple is None or not all(present(triple[x]) for x in ["LOG_ID","PAT_ID","MRN"]):
        return False
    if not present(info) or not present(lab) or str(info)==str(lab):
        return False
    if not exact(event_log_id,triple["LOG_ID"]):
        return False
    return any(exact(info,triple[r]) for r in info_roles) and any(exact(lab,triple[r]) for r in lab_roles)

def classify(frame,unique,ambiguous):
    x=frame.merge(unique.rename(columns={"MRN":"cw_MRN","PAT_ID":"cw_PAT_ID"}),
      on="LOG_ID",how="left",validate="many_to_one",indicator=True)
    a=x.input_MRN.notna()&x.input_MRN.ne("")
    ep=a&x.cw_PAT_ID.notna()&x.input_MRN.eq(x.cw_PAT_ID)
    em=a&x.cw_MRN.notna()&x.input_MRN.eq(x.cw_MRN)
    x["mapping_role"]=np.select([
      ~a,x.LOG_ID.isna()|x.LOG_ID.eq(""),x.LOG_ID.isin(ambiguous),
      x._merge.eq("left_only"),ep&em,ep,em],
      ["missing_MRN","missing_LOG_ID","ambiguous_LOG_ID","unmapped_LOG_ID","both","PAT_ID_only","MRN_only"],
      default="neither_in_unique_LOG_ID_triple")
    return x.drop(columns="_merge")
