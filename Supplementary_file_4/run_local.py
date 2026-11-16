"""Level 2: authorized prepared R2 point analysis and optional full-chain bootstrap."""
from pathlib import Path
import argparse
import json
from portable import bind, check_cohorts, dump, input_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--full-chain-bootstrap', action='store_true')
    p.add_argument('--replicates', type=int)
    p.add_argument('--allow-long-run', action='store_true')
    args = p.parse_args()
    a, cfg, out = bind(args.analysis_root, args.output)
    import pandas as pd
    before = input_snapshot(args.analysis_root)
    cohorts = a.prepare_cohorts()
    check_cohorts(cohorts, a)
    point = a.run_point_analysis(cohorts, cfg, write_outputs=False)
    tables = {
        '30_update_selection_target_performance_v32.csv': 'comparators',
        '34_update_strategy_workload_v32.csv': 'comparator_workload',
        '38_update_mnar_propagation_v32.csv': 'update_mnar',
        '40_target_mnar_sensitivity_v32.csv': 'target_mnar',
        '41_target_nonparametric_bounds_v32.csv': 'bounds',
        '42_mnar_anchor_comparison_v32.csv': 'anchors',
        '43_source_mnar_propagation_v32.csv': 'source_mnar',
        '44_auxiliary_outcome_model_diagnostics_v32.csv': 'auxiliary',
        '46_weight_truncation_sensitivity_v32.csv': 'truncation',
        '49_vitaldb_supportive_v32.csv': 'vitaldb',
    }
    for filename, key in tables.items():
        point[key].to_csv(out / filename, index=False)
    canonical = point['canonical']
    dump(out / 'canonical_point.json', {'alpha': canonical['alpha'], 'beta': canonical['beta'],
                                       'metrics': canonical['metrics'], 'workload': canonical['workload']})
    boot_n = 0
    if args.full_chain_bootstrap:
        n = args.replicates if args.replicates is not None else int(cfg['bootstrap_replicates'])
        if n <= 0 or (n > 2 and not args.allow_long_run):
            raise ValueError('Positive replicates required; >2 requires --allow-long-run')
        a._WORKER_COHORTS, a._WORKER_CONFIG = cohorts, cfg
        rows, mnar = [], []
        for replicate in range(1, n + 1):
            result = a.bootstrap_worker_run(replicate)
            rows.append(result['canonical'])
            mnar.extend(result['mnar'])
        distribution, sensitivity = pd.DataFrame(rows), pd.DataFrame(mnar)
        a.canonical_long_table(canonical, distribution, cfg).to_csv(
            out / '31_canonical_primary_bootstrap_v32.csv', index=False)
        a.mnar_interval_table({k: point[k + '_mnar'] for k in ('source', 'update', 'target')}, sensitivity).to_csv(
            out / '47_mnar_chain_bootstrap_intervals_v32.csv', index=False)
        boot_n = n
    unchanged = before == input_snapshot(args.analysis_root)
    if not unchanged:
        raise RuntimeError('Read-only input hashes changed')
    dump(out / 'local_run_qa.json', {'analysis_version': cfg['analysis_version'],
        'level': 2, 'prepared_R2_point_completed': True, 'bootstrap_replicates': boot_n,
        'input_hashes_unchanged': unchanged, 'input_sha256': before,
        'publication_status': 'NOT assessed; local aggregates require disclosure review',
        'no_individual_or_replicate_outputs_written': True})
    print(json.dumps({'point_completed': True, 'bootstrap_replicates': boot_n,
                      'input_hashes_unchanged': unchanged}))


if __name__ == '__main__':
    main()
