"""Unchanged R2 calculation functions extracted from authoritative sources.
Path configuration and orchestration are supplied by portable.py, not this module.
Historical report/author metadata generators and legacy entry points are excluded.
"""
from __future__ import annotations


import os
import sys, json, re, math, hashlib, time, unittest
from pathlib import Path
from collections import Counter
from decimal import Decimal
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


DAY = 86400 * 10**9

HOUR = 3600 * 10**9

NUM = re.compile(r"[-+]?[0-9]*\.?[0-9]+")

PLAIN = re.compile(r"\s*[-+]?(?:[0-9]*\.)?[0-9]+\s*")

CARD = r"CABG|CORONARY\s+ARTERY\s+BYPASS|CATHETERIZATION,?\s*HEART|CARDIAC|CARDIOPULMONARY\s+BYPASS|HEART\s+TRANSPLANT|VENTRICULAR\s+ASSIST|VALVE\s+(REPAIR|REPLACEMENT)|AORTIC\s+VALVE|MITRAL\s+VALVE|TRICUSPID\s+VALVE|TRANSESOPHAGEAL\s+ECHOCARDIOGRAM|PACEMAKER"

OB = r"CESAREAN|C-SECTION|DELIVERY|OBSTETRIC|LABOR|PLACENTA|ECTOPIC\s+PREGNANCY|ABORTION|DILATION\s+AND\s+EVACUATION|DILATION\s+AND\s+CURETTAGE"

CATS = [
 ("orthopedic",r"ORTHO|HIP|KNEE|SPINE|FEMUR|TIBIA|HUMERUS|ARTHRO|FUSION"),
 ("urologic",r"URO|PROSTATE|BLADDER|URETER|KIDNEY|NEPHR|CYSTOSCOPY"),
 ("neurologic",r"NEURO|CRANI|BRAIN|LAMINECTOMY|VENTRICUL"),
 ("thoracic",r"THORAC|LUNG|BRONCH|PLEUR|MEDIASTIN"),
 ("vascular",r"VASCULAR|ARTERY|ARTERIAL|ANEURYSM|ENDARTERECTOMY|BYPASS"),
 ("head_neck",r"ENT|THYROID|PARATHYROID|TONSIL|LARYNG|MAXILLO|MANDIB"),
 ("abdominal",r"ABDOM|BOWEL|COLON|RECT|GASTR|LIVER|HEPAT|PANCREA|GALLBLADDER|HERNIA"),
 ("breast",r"BREAST|MASTECTOM"),("plastic",r"PLASTIC|RECONSTRUCTION|FLAP|GRAFT")]

def key(kind, value):
    return hashlib.sha256(f"{kind}|AKI_LOCAL_EVIDENCE_REDO_2_0|{value}".encode()).hexdigest()[:20]

def ge(x, threshold):
    return x >= threshold or abs(x-threshold) <= Decimal("1e-12")

def positive(b, a, c):
    if b is None or (a is None and c is None):
        return None
    return int((a is not None and ge(a-b,Decimal(".3"))) or
               (c is not None and ge(c,Decimal("1.5")*b)))

def rise(t0, v0, t1, v1):
    dt = t1-t0
    return dt > 0 and ((dt <= 48*HOUR and ge(v1-v0,Decimal(".3"))) or
                      (dt <= 7*DAY and ge(v1,Decimal("1.5")*v0)))

def serial(rows, anchor, start=None):
    # Earlier comparator may be outside the 7-day later-test interval.
    start = anchor if start is None else start
    r = [x for x in rows if start-14*DAY <= x[0] < anchor]
    late = [x for x in r if x[0] >= start-7*DAY]
    times = sorted({x[0] for x in late})
    any_ev, latest_flags, usable = False, [], False
    for t1,v1,*_ in late:
        pri = [(t0,v0) for t0,v0,*_ in r if 0 < t1-t0 <= 7*DAY]
        usable |= bool(pri)
        hit = any(rise(t0,v0,t1,v1) for t0,v0 in pri)
        any_ev |= hit
        if times and t1 == times[-1]:
            latest_flags.append(hit)
    return dict(any_rise=any_ev, latest_any=any(latest_flags),
                latest_all=bool(latest_flags) and all(latest_flags),
                insufficient=not usable, fewer_two_times=len({x[0] for x in r}) < 2,
                distinct_times_7d=len(times))

def derive(rows, start, end, stop):
    pre = [x for x in rows if start-7*DAY <= x[0] < start]
    post = [x for x in rows if end < x[0] <= min(end+7*DAY,stop)]
    p48 = [x for x in post if x[0] <= min(end+48*HOUR,stop)]
    bt = max((x[0] for x in pre),default=None)
    tied = [x for x in pre if x[0] == bt]
    chosen = min(tied,key=lambda x:x[2]) if tied else None
    b = chosen[1] if chosen else None
    m48 = max((x[1] for x in p48),default=None)
    m7 = max((x[1] for x in post),default=None)
    y = positive(b,m48,m7)
    first = None
    if y == 1:
        first = min(x[0] for x in post if
                    (x[0]-end <= 48*HOUR and ge(x[1]-b,Decimal(".3"))) or
                    ge(x[1],Decimal("1.5")*b))
    return dict(baseline_cr=float(b) if b is not None else np.nan,
                cr_max_48h=float(m48) if m48 is not None else np.nan,
                cr_max_7d=float(m7) if m7 is not None else np.nan,
                postop_cr_48h_count=len(p48),postop_cr_7d_count=len(post),
                tested_7d=bool(post),has_baseline_cr=b is not None,
                baseline_cr_under4=b is not None and b<4,aki=np.nan if y is None else y,
                aki_exact_literal=np.nan if b is None or not post else
                int((m48 is not None and m48-b>=Decimal(".3")) or (m7 is not None and m7>=Decimal("1.5")*b)),
                aki_48h=np.nan if b is None or not post else
                int(m48 is not None and (ge(m48-b,Decimal(".3")) or ge(m48,Decimal("1.5")*b))),
                baseline_tie_rows=len(tied),
                baseline_tie_conflicting=len({x[1] for x in tied})>1,
                baseline_chosen_qualified=bool(chosen and chosen[3]),
                recognition_h=(first-end)/HOUR if first is not None else np.nan,
                lab_equal_start=sum(x[0]==start for x in rows),
                lab_equal_end=sum(x[0]==end for x in rows),
                lab_equal_lookback=sum(x[0]==start-7*DAY for x in rows),
                lab_equal_48h=sum(x[0]==end+48*HOUR for x in rows),
                lab_equal_7d=sum(x[0]==end+7*DAY for x in rows),
                qualified_in_window=sum(x[3] for x in pre+post))

class Boundaries(unittest.TestCase):
    def test_decimal(self):
        self.assertEqual(positive(Decimal(".9"),Decimal("1.2"),Decimal("1.2")),1)
        self.assertEqual(positive(Decimal(".6"),None,Decimal(".9")),1)
        self.assertEqual(positive(Decimal(".9"),Decimal("1.19999999"),Decimal("1.2")),0)
        self.assertIsNone(positive(Decimal(".9"),None,None))
    def test_pair(self):
        self.assertFalse(rise(0,Decimal(".5"),0,Decimal("1")))
        self.assertTrue(rise(0,Decimal("1"),48*HOUR,Decimal("1.3")))
        self.assertFalse(rise(0,Decimal("1"),48*HOUR+1,Decimal("1.3")))
        self.assertTrue(rise(0,Decimal("1"),7*DAY,Decimal("1.5")))
        self.assertFalse(rise(0,Decimal("1"),7*DAY+1,Decimal("1.5")))
    def test_aligned_anchors(self):
        s=20*DAY;e=s+6*HOUR
        r=[(s-7*DAY-HOUR,Decimal(".5"),1,False),
           (s-7*DAY,Decimal("1"),2,False),
           (e,Decimal("3"),3,False)]
        self.assertTrue(serial(r,e,s)["any_rise"])
        self.assertEqual(serial(r,e,s)["distinct_times_7d"],1)
        self.assertEqual(serial(r,s,s)["any_rise"],True)
        self.assertTrue(ge(Decimal("8.199999999999998")-Decimal("7.9"),Decimal(".3")))
        self.assertFalse(ge(Decimal(".29999999"),Decimal(".3")))
    def test_windows_and_ties(self):
        s,e=10*DAY,11*DAY
        rows=[(s-7*DAY,Decimal(".5"),1,False),(s-1,Decimal(".9"),2,False),
              (s-1,Decimal("1"),3,False),(s,Decimal("5"),4,False),
              (e,Decimal("5"),5,False),(e+48*HOUR,Decimal("1.2"),6,False),
              (e+7*DAY,Decimal("1.5"),7,False),(e+7*DAY+1,Decimal("9"),8,False)]
        d=derive(rows,s,e,e+7*DAY)
        self.assertEqual(d["baseline_cr"],.9)
        self.assertEqual(d["postop_cr_7d_count"],2)
        self.assertTrue(d["baseline_tie_conflicting"])
        self.assertEqual(d["aki"],1)
        self.assertEqual(derive(rows,s,e,e+HOUR)["postop_cr_7d_count"],0)
    def test_persistence(self):
        a=20*DAY
        r=[(a-3*DAY,Decimal(".5"),1,False),(a-2*DAY,Decimal("1"),2,False),
           (a-DAY,Decimal(".5"),3,False)]
        self.assertTrue(serial(r,a)["any_rise"])
        self.assertFalse(serial(r,a)["latest_any"])
        self.assertTrue(serial(r[:1],a)["insufficient"])
