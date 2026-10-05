"""Portable preserved analysis body; invoke through replay.py."""
import argparse


def main():
    from pathlib import Path
    import sys, json, zipfile, hashlib, re
    import pandas as pd
    import numpy as np
    A=Path(__file__).resolve().parents[1]
    OUT=A
    sys.path.insert(0,str(A/'scripts'))
    import etl_common_v32 as e
    import etl_outcome_representations_v32 as rep
    cfg=json.loads((A/'config/analysis_config_v32.json').read_text())
    print('Imports complete; auditing raw operations', flush=True)
    flow=[]; flags=[]; audit={}; labels=[]
    def add(ds,stage,n): flow.append(dict(dataset=ds,stage=stage,n=int(n)))
    def tolerant(frame, suffix=''):
        b=frame['baseline_cr'+suffix].to_numpy(); a=frame['cr_max_48h'+suffix].to_numpy(); c=frame['cr_max_7d'+suffix].to_numpy()
        obs=frame.tested_7d.fillna(False).to_numpy() & np.isfinite(b)
        d=a-b; r=1.5*b
        event=(d>=.3)|np.isclose(d,.3,rtol=0,atol=1e-12)|(c>=r)|np.isclose(c,r,rtol=0,atol=1e-12)
        return np.where(obs,event.astype(float),np.nan)
    ops=e.read_zip_gzip_csv(Path(cfg['raw_data']['inspire_archive']),'operations.csv.gz')
    for c in ['age','anstart_time','anend_time','cpbon_time']: ops[c]=pd.to_numeric(ops[c],errors='coerce')
    ops['patient_raw']=ops.subject_id.astype(str); ops['case_raw']=ops.op_id.astype(str)
    ops['case_key']=ops.case_raw.map(lambda x:e.stable_key('INSPIRE_CASE',x))
    dept=ops.department.fillna('').str.upper(); pcs=ops.icd10_pcs.fillna('').str.upper()
    adultga=ops.age.ge(18)&ops.antype.fillna('').str.casefold().eq('general')
    timeok=ops.anstart_time.notna()&ops.anend_time.notna()&((ops.anend_time-ops.anstart_time)/60).between(.5,24)
    cts=dept.eq('CTS'); cpb=ops.cpbon_time.notna(); cardiacpcs=pcs.str.contains(r'(?:^|[,;\s])02[A-Z0-9]',regex=True)
    og=dept.eq('OG'); obs7=pcs.str.contains(r'(?:^|[,;\s])1[A-Z0-9]{6}',regex=True)
    obs5=pcs.str.contains(r'(?:^|[,;\s])1[A-Z0-9]{4}(?=$|[,;\s])',regex=True)
    card=cts|cpb|cardiacpcs; oldob=og|obs7; newob=og|obs5|obs7
    valid=adultga&timeok
    add('INSPIRE','raw_operations',len(ops)); add('INSPIRE','adult_general_anesthesia',adultga.sum()); add('INSPIRE','valid_time_duration',valid.sum())
    for name,m in [('CTS_department',cts),('CPB_record',cpb),('cardiac_PCS',cardiacpcs),('cardiac_union',card),('OG_department',og),('obstetric_PCS_7char',obs7),('obstetric_PCS_5char',obs5),('obstetric_union_corrected',newob),('cardiac_obstetric_overlap',card&newob),('CTS_without_CPB_or_cardiac_PCS',cts&~cpb&~cardiacpcs),('OG_without_obstetric_PCS',og&~obs5&~obs7)]:
        flags.append(dict(dataset='INSPIRE',flag=name,n_before_surgery_exclusions=int((valid&m).sum())))
    oldvalid=valid&~card&~oldob; newvalid=valid&~card&~newob
    sel=ops.loc[newvalid].sort_values(['patient_raw','anstart_time','case_raw']).drop_duplicates('patient_raw')
    oldsel=ops.loc[oldvalid].sort_values(['patient_raw','anstart_time','case_raw']).drop_duplicates('patient_raw')
    audit['inspire_newly_excluded_before_baseline']=int((oldvalid&~newvalid).sum())
    def original_input(stem):
        backup=A/f'data/processed/{stem}_original_v325.parquet'
        return backup if backup.exists() else A/f'data/processed/{stem}_rebuilt_v32.parquet'
    f=pd.read_parquet(original_input('inspire'))
    baseline=f.has_baseline_cr.fillna(False)&f.baseline_cr_under4.fillna(False)
    oldkeys=set(f.loc[baseline&f.first_eligible,'case_key']); newkeys=set(f.loc[baseline&f.case_key.isin(sel.case_key),'case_key'])
    audit['inspire_final_removed']=len(oldkeys-newkeys); audit['inspire_final_added']=len(newkeys-oldkeys)
    add('INSPIRE','surgery_scope_eligible_corrected',newvalid.sum()); add('INSPIRE','first_clinically_eligible_corrected',len(sel))
    first=f.case_key.isin(sel.case_key)
    add('INSPIRE','first_with_baseline', (first&f.has_baseline_cr).sum()); add('INSPIRE','first_baseline_under4',(first&baseline).sum())
    f['first_eligible']=first
    f=f.loc[f.case_key.isin(ops.loc[newvalid,'case_key'])].copy()
    f.to_parquet(A/'data/processed/inspire_cohort_scope_corrected.parquet',index=False)
    for ds,stem in [('INSPIRE','inspire'),('MOVER','mover')]:
        p=original_input(stem)
        z=pd.read_parquet(p)
        groups=[('INSPIRE',z)] if ds=='INSPIRE' else [(f'MOVER {yr}',z.loc[z.year.eq(yr)]) for yr in [2021,2022]]
        for name,g in groups:
            g=rep.eligible_base(g)
            fixed=tolerant(g)
            changed=np.isfinite(fixed)&(fixed!=g.outcome_operational.to_numpy())
            labels.append(dict(cohort=name,n=len(g),observed=int(g.tested_7d.sum()),original_events=int(g.outcome_operational.sum()),corrected_events=int(np.nansum(fixed)),labels_changed=int(changed.sum())))
            if ds=='MOVER':
                fixedc=tolerant(g,'_coarse'); cc=np.isfinite(fixedc)&(fixedc!=g.outcome_coarsened_operational.to_numpy())
                labels[-1].update(original_coarse_events=int(g.outcome_coarsened_operational.sum()),corrected_coarse_events=int(np.nansum(fixedc)),coarse_labels_changed=int(cc.sum()))
        # Only write audit candidates, not overwrite the locked inputs until reviewed.
        z['aki']=tolerant(z); z['outcome_operational']=z.aki
        if ds=='MOVER': z['outcome_coarsened_operational']=tolerant(z,'_coarse')
        z.to_parquet(A/f'data/processed/{stem}_threshold_corrected_candidate.parquet',index=False)
    audit['threshold_tolerance_mg_dl']=1e-12
    info=pd.read_csv(cfg['raw_data']['mover_patient_information'],dtype=str,low_memory=False)
    add('MOVER all years','raw_operation_rows',len(info))
    fields=['MRN','BIRTH_DATE','ASA_RATING_C','SEX','AN_START_DATETIME','AN_STOP_DATETIME','HOSP_DISCH_TIME','PRIMARY_PROCEDURE_NM','PRIMARY_ANES_TYPE_NM']
    dups=info.loc[info.LOG_ID.duplicated(keep=False)]
    conflict=[k for k,g in dups.groupby('LOG_ID',sort=False,dropna=False) if any(g[c].nunique(dropna=False)>1 for c in fields)]
    info=info.loc[~info.LOG_ID.isin(conflict)].drop_duplicates('LOG_ID').copy()
    add('MOVER all years','after_duplicate_resolution',len(info))
    info['patient_raw']=info.MRN.fillna(info.LOG_ID).astype(str)
    info['case_raw']=info.LOG_ID.astype(str)
    info['case_key']=info.case_raw.map(lambda x:e.stable_key('MOVER_CASE',x))
    start=pd.to_datetime(info.AN_START_DATETIME,errors='coerce'); end=pd.to_datetime(info.AN_STOP_DATETIME,errors='coerce')
    info['an_start']=start
    ag=pd.to_numeric(info.BIRTH_DATE,errors='coerce').ge(18)&info.PRIMARY_ANES_TYPE_NM.fillna('').str.strip().str.casefold().eq('general')
    tim=start.notna()&end.notna()&((end-start).dt.total_seconds()/3600).between(.5,24)
    proc=info.PRIMARY_PROCEDURE_NM.fillna(''); ca=proc.str.contains(e.CARDIAC_RE); ob=proc.str.contains(e.OBSTETRIC_RE)
    add('MOVER all years','adult_general_anesthesia',ag.sum()); add('MOVER all years','valid_time_duration',(ag&tim).sum())
    for name,m in [('cardiac_procedure',ca),('obstetric_procedure',ob),('cardiac_obstetric_overlap',ca&ob)]: flags.append(dict(dataset='MOVER all years',flag=name,n_before_surgery_exclusions=int((ag&tim&m).sum())))
    v=ag&tim&~ca&~ob
    add('MOVER all years','surgery_scope_eligible',v.sum())
    firstinfo=info.loc[v].sort_values(['patient_raw','an_start','case_raw']).drop_duplicates('patient_raw')
    add('MOVER all years','first_clinically_eligible',len(firstinfo))
    m=pd.read_parquet(original_input('mover'))
    mf=m.case_key.isin(firstinfo.case_key)
    assert (mf==m.first_eligible).all()
    add('MOVER all years','first_with_baseline',(mf&m.has_baseline_cr).sum())
    add('MOVER all years','first_baseline_under4',(mf&m.has_baseline_cr&m.baseline_cr_under4).sum())
    for yr in [2021,2022]:
        g=rep.eligible_base(m,yr)
        add(f'MOVER {yr}','final_eligible',len(g)); add(f'MOVER {yr}','outcome_observed',g.tested_7d.sum()); add(f'MOVER {yr}','outcome_unobserved',(~g.tested_7d).sum())
    audit['mover_conflicting_duplicate_ids']=len(conflict)
    audit['mover_update_target_patient_overlap']=len(set(rep.eligible_base(m,2021).patient_key)&set(rep.eligible_base(m,2022).patient_key))
    pd.DataFrame(flow).to_csv(A/'tables/R1_cohort_flow_audit.csv',index=False)
    pd.DataFrame(flags).to_csv(A/'tables/R1_surgical_flags.csv',index=False)
    pd.DataFrame(labels).to_csv(A/'tables/R1_threshold_boundary_audit.csv',index=False)
    (A/'qa/R1_P0_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2)); print(pd.DataFrame(labels).to_string(index=False)); print(pd.DataFrame(flow).to_string(index=False))
    arc=A/'private/v3.2.5_public.zip'
    if arc.exists() and zipfile.is_zipfile(arc):
        diffs=[]
        with zipfile.ZipFile(arc) as zf:
            for name in zf.namelist():
                if '/scripts/' in name and not name.endswith('/'):
                    local=A/'scripts'/Path(name).name
                    if local.exists(): diffs.append({'file':local.name,'equal':hashlib.sha256(local.read_bytes()).hexdigest()==hashlib.sha256(zf.read(name)).hexdigest()})
        pd.DataFrame(diffs).to_csv(A/'qa/R1_v325_code_comparison.csv',index=False)
        print('Public code differences:',[d for d in diffs if not d['equal']])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    main()
