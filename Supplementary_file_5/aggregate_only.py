"""Level 1: validate S6 aggregate tables and redraw S6 directly from CSV."""
from pathlib import Path
import argparse
import json
from portable import HERE, output_directory, dump, sha


def plot_calibration(points, distribution, out):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from aggregate_support import numeric, save_figure
    strategies = ['Unupdated', 'Complete-case recalibration', 'IPW recalibration']
    if set(points.strategy) != set(strategies) or set(distribution.strategy) != set(strategies):
        raise ValueError('S6 requires exactly the three current strategies')
    fig, axs = plt.subplots(2, 3, figsize=(10.2, 5.1),
        gridspec_kw={'height_ratios': [5, .7]}, layout='constrained')
    colors = ['#60666E', '#087F8C', '#B14C2E']
    maximum = points[['weighted_mean_prediction', 'conditional_ci_upper']].apply(numeric).max().max()
    limit = max(.4, np.ceil(maximum * 10) / 10)
    for j, strategy in enumerate(strategies):
        d = points.loc[points.strategy.eq(strategy)].sort_values('risk_group')
        x, y = numeric(d.weighted_mean_prediction), numeric(d.ipw_event_frequency)
        lo, hi = numeric(d.conditional_ci_lower), numeric(d.conditional_ci_upper)
        ax = axs[0, j]
        ax.plot([0, 1], [0, 1], ls='--', color='0.6', lw=1)
        ax.errorbar(x, y, yerr=np.vstack([y-lo, hi-y]), fmt='o-', ms=4,
                    lw=1.3, capsize=2, color=colors[j])
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel='IPW-weighted mean predicted risk',
               title=f'{"ABC"[j]}  {strategy}')
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_aspect('equal')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.13)
        if j == 0:
            ax.set_ylabel('IPW-estimated event frequency')
        q = distribution.loc[distribution.strategy.eq(strategy)].sort_values('quantile')
        if not np.array_equal(numeric(q['quantile']).to_numpy(), [0, .05, .25, .5, .75, .95, 1]):
            raise ValueError('Seven exact distribution quantiles required per strategy')
        qs = numeric(q.prediction).to_numpy()
        if not np.isfinite(qs).all() or (np.diff(qs) < 0).any():
            raise ValueError('Invalid quantile ordering')
        dax = axs[1, j]
        dax.plot([qs[0], qs[-1]], [0, 0], color='0.75', lw=1)
        dax.plot([qs[1], qs[-2]], [0, 0], color=colors[j], lw=2)
        dax.plot([qs[2], qs[4]], [0, 0], color=colors[j], lw=7)
        dax.plot(qs[3], 0, '|', color='white', ms=9, mew=1.5)
        dax.set(xlim=(0, 1), ylim=(-1, 1), yticks=[], xlabel='Eligible predicted risk (0-100%)')
        dax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        dax.spines[['top', 'right', 'left']].set_visible(False)
    fig.suptitle('Figure S6. Grouped calibration in MOVER 2022', fontsize=12, fontweight='bold')
    save_figure(fig, out, 'Figure_S6_calibration_curves')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aggregates', type=Path, default=HERE / 'aggregates')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = output_directory(args.output, (args.aggregates,))
    from aggregate_support import read_aggregate, validate_contract, numeric
    import numpy as np
    contract = json.loads((HERE / 'contracts.json').read_text())
    tables, checks = {}, []
    for name, spec in contract['aggregate_files'].items():
        path = args.aggregates / name
        f = read_aggregate(path, spec['required'])
        validate_contract(f, spec)
        tables[name] = f
        checks.append({'file': name, 'sha256': sha(path), 'rows': len(f)})
    for path in sorted(args.aggregates.glob('*.csv')):
        if path.name not in tables:
            read_aggregate(path)
            checks.append({'file': path.name, 'sha256': sha(path), 'validation': 'structural_only'})
    points = tables['S6_calibration_points.csv']
    if points.duplicated(['strategy', 'risk_group']).any():
        raise ValueError('Duplicated S6 strategy/risk group')
    core = ['weighted_mean_prediction', 'ipw_event_frequency', 'conditional_ci_lower', 'conditional_ci_upper']
    if not np.isfinite(points[core].apply(numeric).to_numpy()).all():
        raise ValueError('S6 plotted coordinates must be unsuppressed numeric aggregates')
    lo, y, hi = (numeric(points[c]) for c in ('conditional_ci_lower', 'ipw_event_frequency', 'conditional_ci_upper'))
    if not ((lo <= y) & (y <= hi)).all():
        raise ValueError('Point outside supplied interval; cannot use asymmetric error bars')
    plot_calibration(points, tables['prediction_distribution.csv'], out)
    dump(out / 'aggregate_checks.json', {'level': 1, 'files': checks,
        'plot_source': 'aggregate CSV only; no original PDF, patient data, model or bootstrap needed',
        'interval_scope': 'Conditional pointwise intervals supplied in aggregate inputs; not refitted',
        'suppressed_cells': 'not inferred or replaced by zero'})
    print(json.dumps({'csv_files_checked': len(checks), 'S6_redrawn_from_aggregates': True}))


if __name__ == '__main__':
    main()
