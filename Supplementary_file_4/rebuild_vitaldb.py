"""Official local VitalDB cases/labs CSV -> the actual upstream supportive cohort."""
from pathlib import Path
import argparse
import sys
from portable import HERE, output_directory, sha, dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--labs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = output_directory(args.output, (args.cases.parent, args.labs.parent))
    sys.path.insert(0, str(HERE / 'scripts'))
    import pandas as pd
    from prepare_vitaldb_v32 import build_vitaldb
    before = [sha(args.cases), sha(args.labs)]
    f = build_vitaldb(pd.read_csv(args.cases, encoding='utf-8-sig', low_memory=False),
                     pd.read_csv(args.labs, encoding='utf-8-sig', low_memory=False))
    if before != [sha(args.cases), sha(args.labs)]:
        raise RuntimeError('VitalDB raw inputs changed')
    f.to_parquet(out / 'vitaldb_supportive_v32.parquet', index=False)
    dump(out / 'vitaldb_rebuild_qa.json', {'input_hashes_unchanged': True,
        'input_sha256': before, 'raw_cases_labs_route': True,
        'legacy_analysis_cache_used': False,
        'rule': 'Actual upstream prepare_vitaldb_v32 calculation statements unchanged',
        'boundary': 'Supportive released cohort; not source-compatible all-eligible denominator',
        'real_raw_to_authoritative_R2_equivalence': 'Not tested in this task'})
    print('VitalDB supportive cohort rebuilt from local raw tables; no individual rows printed.')


if __name__ == '__main__':
    main()
