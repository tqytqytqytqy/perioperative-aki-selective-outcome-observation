"""Aggregate input validation. Suppressed cells stay missing for plotting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

SUPPRESSED = {'<5', '< 5', 'suppressed', 'Suppressed', 'withheld', 'Withheld',
              'not disclosed', 'Not disclosed', '*', '**', '\u2014', '-', 'NA', 'N/A',
              'Not estimable', 'not estimable'}
FORBIDDEN = {'case_key', 'patient_key', 'MRN', 'LOG_ID', 'PAT_ID', 'patient_source',
             'raw_lab_csv_row', 'replicate', 'replicate_zero_based'}


def numeric(series):
    return pd.to_numeric(series.mask(series.astype(str).isin(SUPPRESSED)), errors='coerce')


def read_aggregate(path, required=()):
    f = pd.read_csv(path)
    if f.empty:
        raise ValueError('Empty aggregate: ' + Path(path).name)
    if FORBIDDEN.intersection(f.columns):
        raise ValueError('Individual / replicate identifiers are not aggregate inputs')
    missing = set(required) - set(f.columns)
    if missing:
        raise ValueError('Missing columns in ' + Path(path).name + ': ' + ', '.join(sorted(missing)))
    for col in f:
        if pd.api.types.is_numeric_dtype(f[col]) and np.isinf(f[col].to_numpy(dtype=float)).any():
            raise ValueError('Infinite aggregate value: ' + col)
    return f


def validate_contract(f, spec):
    missing = set(spec.get('required', [])) - set(f.columns)
    if missing:
        raise ValueError('Required contract columns missing: ' + ', '.join(sorted(missing)))
    for col in spec.get('numeric', []):
        if col not in f:
            continue
        v = f[col]
        bad = v.notna() & ~v.astype(str).isin(SUPPRESSED) & numeric(v).isna()
        if bad.any():
            raise ValueError('Unrecognized numeric/suppression value in ' + col)
    for col in spec.get('probabilities', []):
        if col in f:
            v = numeric(f[col]).dropna()
            if not v.between(0, 1).all():
                raise ValueError('Probability outside [0,1]: ' + col)
    for col in spec.get('counts', []):
        if col in f:
            v = numeric(f[col]).dropna()
            if not ((v >= 0) & (v == np.floor(v))).all():
                raise ValueError('Invalid count: ' + col)


def save_figure(fig, out, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(out / (stem + '.' + ext), dpi=300, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
