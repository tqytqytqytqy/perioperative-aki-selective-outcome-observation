"""Level 1: validate all supplied CSV aggregates and redraw supported summaries."""
from pathlib import Path
import argparse
import json
from portable import HERE, output_directory, dump, sha


def plot_table(f, name, out):
    import numpy as np
    import matplotlib.pyplot as plt
    import textwrap
    from aggregate_support import numeric, save_figure
    plots = []
    if 'odds_multiplier' in f:
        g = f
        if 'is_canonical' in g:
            g = g.loc[~g.is_canonical.astype(str).str.lower().eq('true')]
        metrics = [c for c in ('target_oe_ratio', 'target_calibration_slope', 'target_auroc',
                              'target_brier', 'threshold_10_alerts') if c in g]
        x = numeric(g.odds_multiplier)
    elif 'elapsed_day' in f:
        g, x = f, numeric(f.elapsed_day)
        metrics = [c for c in ('all_eligible_pct', 'any_opportunity_pct', 'full_opportunity_pct') if c in g]
    else:
        g, x = f, None
        metrics = [c for c in ('oe_ratio', 'calibration_slope', 'auroc', 'brier',
            'alerts_per_1000', 'sensitivity', 'positive_predictive_value', 'percent',
            'observed_pct', 'late_test_pct', 'late_among_opportunity_pct') if c in g]
    for metric in metrics:
        size = (10, 4.2) if x is not None else (13, max(4.2, min(36, len(g) * .8)))
        fig, ax = plt.subplots(figsize=size, layout='constrained')
        groups = g.groupby('cohort', sort=False) if 'cohort' in g else [('Aggregate', g)]
        any_values = False
        for label, sub in groups:
            y = numeric(sub[metric])
            if 'odds_multiplier' in sub:
                xx = numeric(sub.odds_multiplier)
                order = np.argsort(xx.to_numpy())
                ax.plot(xx.iloc[order], y.iloc[order], 'o-', label=label)
                ax.set_xscale('log')
            elif 'elapsed_day' in sub:
                ax.plot(numeric(sub.elapsed_day), y, 'o-', label=label)
            else:
                labels = [c for c in ('source_model_estimand', 'update_estimator', 'strategy',
                                     'threshold', 'metric', 'group') if c in sub]
                text = sub[labels].astype(str).agg(' | '.join, axis=1) if labels else sub.index.astype(str)
                ax.plot(y, np.arange(len(sub)), 'o', label=label)
                labels = [textwrap.fill(str(value), width=62) for value in text]
                ax.set_yticks(np.arange(len(sub)), labels, fontsize=7)
            any_values |= y.notna().any()
        if any_values:
            if x is None:
                ax.set_xlabel(metric.replace('_', ' '))
                ax.invert_yaxis()
            else:
                ax.set_ylabel(metric.replace('_', ' '))
            ax.set_title(name.replace('_', ' '))
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(alpha=.15)
            if 'cohort' in g:
                ax.legend(frameon=False)
            stem = name + '__' + metric
            save_figure(fig, out, stem)
            plots.append(stem)
        else:
            plt.close(fig)
    return plots


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aggregates', type=Path, default=HERE / 'aggregates')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = output_directory(args.output, (args.aggregates,))
    from aggregate_support import read_aggregate, validate_contract
    contract = json.loads((HERE / 'contracts.json').read_text())
    paths = sorted(args.aggregates.rglob('*.csv'))
    if not paths:
        raise ValueError('No aggregate CSV files supplied; see contracts.json')
    rows = []
    for path in paths:
        rel = str(path.relative_to(args.aggregates))
        f = read_aggregate(path)
        spec = contract['aggregate_files'].get(rel)
        if spec is not None:
            validate_contract(f, spec)
        plots = plot_table(f, path.stem, out)
        rows.append({'file': rel, 'sha256': sha(path), 'rows': len(f),
                     'validation': 'schema' if spec is not None else 'structural_only',
                     'plots': plots, 'suppressed_cells': 'not inferred or replaced by zero'})
    missing = [name for name in contract['aggregate_files']
               if not (args.aggregates / name).is_file()]
    dump(out / 'aggregate_checks.json', {'level': 1, 'files': rows,
        'not_supplied_contract_files': missing,
        'scope': 'All supplied CSVs checked; unknown filenames structural-only. No model fit or patient access.',
        'exact_counts_or_suppressed_differences_reconstructed': False})
    print(json.dumps({'csv_files_checked': len(rows), 'plots': sum(len(r['plots']) for r in rows),
                      'not_supplied_contract_files': len(missing)}))


if __name__ == '__main__':
    main()
