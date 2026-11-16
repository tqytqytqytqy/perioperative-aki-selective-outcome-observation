"""Authorized local timing/coverage summaries using current diagnostic functions."""
from pathlib import Path
import argparse
import sys
from portable import HERE, output_directory, dump, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True,
                   help='Cohort from rebuild_mover.py, including per-operation timing fields')
    p.add_argument('--source-model', type=Path, required=True,
                   help='Trusted frozen source model used by the current timing diagnostic')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = output_directory(args.output, (args.cohort.parent, args.source_model.parent))
    sys.path.insert(0, str(HERE / 'scripts'))
    import joblib
    import numpy as np
    import pandas as pd
    import build_diagnostics as d
    before = [sha(args.cohort), sha(args.source_model)]
    f = pd.read_parquet(args.cohort)
    mask = f.first_eligible & f.has_baseline_cr & f.baseline_cr_under4 & f.year.isin([2021, 2022])
    f = f.loc[mask].copy()
    model = joblib.load(args.source_model)
    features = ['age', 'sex_male', 'duration_h', 'baseline_cr']
    f['source_risk'] = model.predict_proba(f[features])[:, 1]
    f['cohort'] = f.year.map(lambda y: f'MOVER {int(y)}')
    for _, g in f.groupby('cohort'):
        f.loc[g.index, 'risk_quintile'] = pd.qcut(g.source_risk, 5,
            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']).astype(str)
    f['stop_group'] = np.select([f.cap_h.le(48), f.cap_h.lt(168)],
        ['<=48 h', '>48 to <168 h'], default='>=168 h')
    if not f.tested.eq(f.tested_7d).all():
        raise ValueError('Timing tested indicator differs from operational observation')
    d.EXACT = out
    d.coverage_tables(f)
    unchanged = before == [sha(args.cohort), sha(args.source_model)]
    if not unchanged:
        raise RuntimeError('Timing input changed')
    dump(out / 'timing_qa.json', {'input_hashes_unchanged': unchanged,
        'source_model_fits': 0, 'input_sha256': before,
        'scope': 'MOVER only; exact local aggregates require publication filtering',
        'risk_strata_model': 'Explicit supplied frozen diagnostic source model, not automatically substituted with R2 refit'})


if __name__ == '__main__':
    main()
