"""Authorized raw INSPIRE ZIP -> prepared cohort using unchanged source extraction."""
from pathlib import Path
import argparse
import sys
from portable import HERE, output_directory, sha, dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = output_directory(args.output, (args.archive.parent,))
    sys.path.insert(0, str(HERE / 'scripts'))
    import etl_common_v32 as e
    import etl_outcome_representations_v32 as rep
    before = sha(args.archive)
    cfg = {'raw_data': {'inspire_archive': str(args.archive)}}
    e.DATA_DIR = out
    f, audit = e.build_inspire(cfg)
    levels, _ = rep.inspire_release_levels(cfg)
    for col in ['baseline_cr', 'cr_max_48h', 'cr_max_7d']:
        f[col + '_lower'], f[col + '_upper'] = rep.interval_bounds_from_levels(f[col], levels, .05, 30.)
    f['outcome_operational'] = f.aki
    f['outcome_definite'], f['outcome_possible'] = rep.classify_interval_aki(f,
        'baseline_cr_lower', 'baseline_cr_upper', 'cr_max_48h_lower', 'cr_max_48h_upper',
        'cr_max_7d_lower', 'cr_max_7d_upper')
    f['baseline_cr_coarse'] = f.baseline_cr
    f['outcome_coarsened_operational'] = f.outcome_operational
    if before != sha(args.archive):
        raise RuntimeError('INSPIRE archive changed')
    f.to_parquet(out / 'inspire_rebuilt_v32.parquet', index=False)
    (out / 'inspire_rebuilt_cohort.parquet').unlink()
    dump(out / 'inspire_rebuild_qa.json', {'input_hashes_unchanged': True,
        'archive_sha256': before, 'audit': audit,
        'extraction': 'verbatim current etl_common build_inspire and its dependency closure',
        'representation': 'current INSPIRE released-level interval rules',
        'real_raw_to_authoritative_R2_equivalence': 'Not tested in this task'})
    print('INSPIRE cohort rebuilt locally; no individual rows printed.')


if __name__ == '__main__':
    main()
