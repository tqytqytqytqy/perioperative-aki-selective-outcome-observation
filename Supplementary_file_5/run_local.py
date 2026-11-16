"""Level 2: current R2 grouped calibration from authorized prepared inputs."""
from pathlib import Path
import argparse
import json
from portable import bind, check_cohorts, input_snapshot, dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    a, cfg, out = bind(args.analysis_root, args.output)
    from calibration_calculation import calculate
    before = input_snapshot(args.analysis_root)
    spec = json.loads((args.analysis_root / 'models/model_specification_v32.json').read_text())
    cohorts = a.prepare_cohorts()
    check_cohorts(cohorts, a, spec)
    points, metrics, distribution, qa = calculate(a, cfg, cohorts, args.analysis_root)
    points.to_csv(out / 'S6_calibration_points.csv', index=False)
    metrics.to_csv(out / 'reproduction_metrics.csv', index=False)
    distribution.to_csv(out / 'prediction_distribution.csv', index=False)
    unchanged = before == input_snapshot(args.analysis_root)
    if not unchanged:
        raise RuntimeError('Read-only inputs changed')
    qa.update(analysis_version=cfg['analysis_version'], level=2,
              input_hashes_unchanged=unchanged, input_sha256=before,
              publication_status='NOT assessed; local aggregates require disclosure review')
    dump(out / 'local_calibration_qa.json', qa)
    print(json.dumps({'calibration_completed': True,
        'maximum_absolute_reproduction_difference': qa['maximum_absolute_reproduction_difference'],
        'input_hashes_unchanged': unchanged}))


if __name__ == '__main__':
    main()
