"""Authorized raw MOVER CSV -> identity-consistent cohort, without legacy caches."""
from pathlib import Path
import argparse
from decimal import Decimal
import sys
from portable import HERE, output_directory, sha, dump

sys.path.insert(0, str(HERE / 'scripts'))


def reconstruct(info, crosswalk, labs):
    import numpy as np
    import pandas as pd
    import audit_mover as audit
    from raw_linkage_audit import unique_triples, classify
    from timing_helpers import summarize_times
    fields = ['MRN', 'BIRTH_DATE', 'ASA_RATING_C', 'SEX', 'AN_START_DATETIME',
              'AN_STOP_DATETIME', 'HOSP_DISCH_TIME', 'PRIMARY_PROCEDURE_NM', 'PRIMARY_ANES_TYPE_NM']
    dup = info[info.LOG_ID.duplicated(False)]
    conflicts = {k for k, g in dup.groupby('LOG_ID', dropna=False)
                 if any(g[c].nunique(dropna=False) > 1 for c in fields)}
    info = info[~info.LOG_ID.isin(conflicts)].drop_duplicates('LOG_ID', keep='first').copy()
    info['patient_source'] = info.MRN.fillna(info.LOG_ID)
    info['age'] = pd.to_numeric(info.BIRTH_DATE, errors='coerce')
    info['asa'] = pd.to_numeric(info.ASA_RATING_C, errors='coerce')
    info['sex_male'] = info.SEX.fillna('').str.casefold().eq('male').astype(float)
    info['sex_observed'] = info.SEX.str.casefold().map({'male': 1., 'female': 0.})
    for dest, src in [('an_start', 'AN_START_DATETIME'), ('an_end', 'AN_STOP_DATETIME'),
                      ('discharge', 'HOSP_DISCH_TIME')]:
        info[dest] = pd.to_datetime(info[src], errors='coerce')
    info['duration_h'] = (info.an_end - info.an_start).dt.total_seconds() / 3600
    proc = info.PRIMARY_PROCEDURE_NM.fillna('')
    keep = info.age.ge(18) & info.PRIMARY_ANES_TYPE_NM.fillna('').str.strip().str.casefold().eq('general')
    keep &= info.an_start.notna() & info.an_end.notna() & info.duration_h.between(.5, 24)
    keep &= ~proc.str.contains(audit.CARD, case=False, regex=True)
    keep &= ~proc.str.contains(audit.OB, case=False, regex=True)
    clinical = info.loc[keep].sort_values(['patient_source', 'an_start', 'LOG_ID']).copy()
    clinical['first_eligible'] = ~clinical.patient_source.duplicated()
    clinical['surgery_category'] = 'other'
    for label, pattern in audit.CATS:
        mask = clinical.surgery_category.eq('other') & clinical.PRIMARY_PROCEDURE_NM.fillna('').str.upper().str.contains(pattern)
        clinical.loc[mask, 'surgery_category'] = label
    unique, ambiguous, _ = unique_triples(crosswalk)
    pairs = crosswalk[['MRN', 'PAT_ID']].dropna().drop_duplicates()
    m2p = pairs.groupby('MRN').PAT_ID.nunique()
    p2m = pairs.groupby('PAT_ID').MRN.nunique()
    aliases = pairs[pairs.MRN.map(m2p).eq(1) & pairs.PAT_ID.map(p2m).eq(1)]
    amap = aliases.drop_duplicates('MRN').set_index('MRN').PAT_ID
    c = classify(clinical.rename(columns={'MRN': 'input_MRN'}), unique, ambiguous)
    alias = c.input_MRN.map(amap)
    gate = c.mapping_role.eq('MRN_only') & alias.notna() & c.cw_PAT_ID.eq(alias)
    c = c.loc[gate].rename(columns={'input_MRN': 'MRN'})
    joined = labs.merge(c[['LOG_ID', 'MRN']], on='LOG_ID', how='inner',
                        suffixes=('_lab', '_info'), validate='many_to_one')
    good = (joined.MRN_lab.notna() & joined.MRN_info.notna() & joined.MRN_lab.ne('')
            & joined.MRN_info.ne('') & joined.MRN_lab.eq(joined.MRN_info))
    rowmap = {key: [(int(t.value), Decimal(v), int(i), bool(q))
                   for t, v, i, q in zip(g.lab_time, g.literal, g.row, g.qualified)]
              for key, g in joined.loc[good].groupby('LOG_ID', sort=False)}
    records = []
    for r in c.itertuples():
        start, end = int(r.an_start.value), int(r.an_end.value)
        dis = int(r.discharge.value) if pd.notna(r.discharge) else None
        valid = dis is not None and dis > end
        stop = min(end + 7 * audit.DAY, dis) if valid else end + 7 * audit.DAY
        rr = rowmap.get(r.LOG_ID, [])
        d = audit.derive(rr, start, end, stop)
        d.update(case_key=audit.key('MOVER_CASE', r.LOG_ID),
            patient_key=audit.key('MOVER_PATIENT', r.patient_source),
            age=r.age, sex_male=r.sex_male, sex_observed=r.sex_observed,
            asa=r.asa, duration_h=r.duration_h, surgery_category=r.surgery_category,
            first_eligible=r.first_eligible, year=r.an_start.year,
            an_start=r.an_start, an_end=r.an_end,
            window_48h_end=pd.Timestamp(min(end + 48 * audit.HOUR, stop)),
            window_7d_end=pd.Timestamp(stop), discharge_valid=valid,
            cap_h=(stop-end) / audit.HOUR, discharge_unusable=not valid)
        for label, anchor in [('pre_start', start), ('pre_end', end)]:
            d.update({label + '_' + k: v for k, v in audit.serial(rr, anchor, start).items()})
        d.update(summarize_times([(t-end) / audit.HOUR for t, *_ in rr], d['cap_h']))
        records.append(d)
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError('No identity-confirmed clinical operations')
    result['outcome_operational'] = result.aki
    return result


def read_strict_labs(path, chunksize=500000):
    import numpy as np
    import pandas as pd
    import audit_mover as audit
    parts, offset = [], 0
    use = ['LOG_ID', 'MRN', 'Lab Code', 'Lab Name', 'Observation Value',
           'Measurement Units', 'Collection Datetime']
    for ch in pd.read_csv(path, usecols=use, dtype=str, chunksize=chunksize, low_memory=False):
        name = ch['Lab Name'].fillna('').str.strip().str.casefold().eq('creatinine')
        unit = ch['Measurement Units'].fillna('').str.replace(' ', '', regex=False).str.upper().eq('MG/DL')
        code = ch['Lab Code'].fillna('').str.strip().isin(['2160-0', '38483-4'])
        mask = name & unit & code
        sub = ch.loc[mask].copy()
        sub['row'] = np.flatnonzero(mask.to_numpy()) + offset + 2
        sub['literal'] = sub['Observation Value'].str.extract(r'([-+]?[0-9]*\.?[0-9]+)', expand=False)
        sub['cr'] = pd.to_numeric(sub.literal, errors='coerce')
        sub['lab_time'] = pd.to_datetime(sub['Collection Datetime'], errors='coerce')
        sub['qualified'] = ~sub['Observation Value'].fillna('').map(lambda x: bool(audit.PLAIN.fullmatch(x)))
        parts.append(sub.loc[sub.cr.between(.05, 30) & sub.lab_time.notna(),
                             ['LOG_ID', 'MRN', 'literal', 'lab_time', 'row', 'qualified']])
        offset += len(ch)
    return pd.concat(parts, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('patient-information', 'patient-labs', 'crosswalk', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--coarsening-map', type=Path)
    args = p.parse_args()
    inputs = [args.patient_information, args.patient_labs, args.crosswalk]
    if args.coarsening_map:
        inputs.append(args.coarsening_map)
    out = output_directory(args.output, [v.parent for v in inputs])
    import pandas as pd
    before = {f'input_{i}': sha(v) for i, v in enumerate(inputs)}
    f = reconstruct(pd.read_csv(args.patient_information, dtype=str),
                    pd.read_csv(args.crosswalk, dtype=str), read_strict_labs(args.patient_labs))
    import etl_common_v32 as e
    f = e.finalize_outcomes(f)
    f['outcome_operational'] = f.aki
    if args.coarsening_map:
        import etl_outcome_representations_v32 as rep
        mapping = pd.read_csv(args.coarsening_map)
        edges = mapping.upper_cutpoint_mg_dl.dropna().to_numpy()
        levels = mapping.released_representative_mg_dl.to_numpy()
        for col in ['baseline_cr', 'cr_max_48h', 'cr_max_7d']:
            f[col + '_coarse'] = rep.apply_percentile_release(f[col], edges, levels)
            f[col + '_coarse_lower'], f[col + '_coarse_upper'] = rep.interval_bounds_from_levels(
                f[col + '_coarse'], levels, .05, 30.)
        f['outcome_coarsened_operational'] = rep.classify_exact_aki(
            f, 'baseline_cr_coarse', 'cr_max_48h_coarse', 'cr_max_7d_coarse')
        f['outcome_definite'], f['outcome_possible'] = rep.classify_interval_aki(
            f, *[c + '_coarse_' + s for c, s in [('baseline_cr', 'lower'), ('baseline_cr', 'upper'),
                ('cr_max_48h', 'lower'), ('cr_max_48h', 'upper'), ('cr_max_7d', 'lower'), ('cr_max_7d', 'upper')]])
    if before != {f'input_{i}': sha(v) for i, v in enumerate(inputs)}:
        raise RuntimeError('Raw input hashes changed')
    f.to_parquet(out / 'mover_rebuilt_v32.parquet', index=False)
    dump(out / 'rebuild_qa.json', {'raw_input_hashes_unchanged': True,
        'input_sha256': before, 'identity_rule': 'original first before QC; exact nonmissing lab identity',
        'coarsened_outcomes_included': bool(args.coarsening_map),
        'real_raw_to_authoritative_R2_equivalence': 'NOT established by synthetic tests'})
    print('MOVER cohort reconstructed locally; no individual records printed.')


if __name__ == '__main__':
    main()
