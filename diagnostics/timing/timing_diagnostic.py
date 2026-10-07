"""Read-only raw extraction and post hoc coverage summaries for frozen AKI cohorts."""
from pathlib import Path
import argparse, csv, gzip, hashlib, json, re, time, zipfile
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit, logit

ANALYSIS = None
OUT = Path('timing_diagnostic_output')
RESULT = OUT / '04_新增分析复现材料'
PRIVATE = OUT / '90_内部核验_请勿上传/检测时序_PRIVATE'
FEATURES = ['age','sex_male','duration_h','baseline_cr']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def case_key(ds, value):
    return hashlib.sha256(f'{ds}|AKI_LOCAL_EVIDENCE_REDO_2_0|{value}'.encode()).hexdigest()[:20]

def retained(f):
    return f.loc[f.has_baseline_cr.fillna(False) & f.baseline_cr_under4.fillna(False) & f.first_eligible.fillna(False)].copy()

def summarize_times(times, cap):
    x=np.asarray(times,dtype=float)
    x=x[(x>0)&(x<=cap)]
    u=np.unique(x)
    early=u[u<=48]
    late=u[u>48]
    return {'raw_rows':len(x),'raw_rows48':int((x<=48).sum()),'occasions':len(u),
        'tested':len(u)>0,'early':len(early)>0,'late':len(late)>0,'both':len(early)>0 and len(late)>0,
        'first_h':float(u[0]) if len(u) else np.nan,'last_h':float(u[-1]) if len(u) else np.nan,
        'gap_to_end_h':float(cap-u[-1]) if len(u) else np.nan,
        'tested_bins':len(np.unique(np.ceil(u/24).astype(int))) if len(u) else 0,
        **{f'day{d}':bool(((u>24*(d-1))&(u<=24*d)).any()) for d in range(1,8)}}

def unit_tests():
    r=summarize_times([0,1,1,24,48,48.001,168,169],168)
    assert r['raw_rows']==6 and r['occasions']==5 and r['raw_rows48']==4
    assert r['tested_bins']==4 and r['both'] and r['day1'] and r['day2'] and r['day3'] and r['day7']
    assert not summarize_times([0,30],24)['tested']
    assert not summarize_times([48,49],48)['late']
    assert summarize_times([48,49],49)['late']
    assert np.isnan(summarize_times([],168)['first_h'])

def wilson(k,n):
    if not n:return np.nan,np.nan
    z=1.959963984540054; p=k/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return 100*(mid-half),100*(mid+half)

def qstr(v):
    x=pd.Series(v).dropna()
    if x.empty:return 'NA'
    a,b,c=np.quantile(x,[.25,.5,.75]); return f'{b:.1f} ({a:.1f}-{c:.1f})'

def prop(k,n):
    return f'{int(k)}/{int(n)} ({100*k/n:.1f}%)' if n else 'NA'

def write_summaries(f):
    overall=[]; by_risk=[]; by_stop=[]; days=[]; facts={}
    for cohort,g in f.groupby('cohort',sort=False):
        obs=g.loc[g.tested]; n=len(g)
        cases={
          'Any postoperative test':(g.tested,n),
          'No postoperative test':(~g.tested,n),
          'Test in (0,48] h':(g.early,n),
          'Test in (48,168] h before recorded stop':(g.late,n),
          'Tests in both early and late windows':(g.both,n),
          'Only one distinct testing time':(g.occasions.eq(1),n),
          'Algorithmic observation window <48 h':(g.cap_h.lt(48),n),
          'Algorithmic observation window <168 h':(g.cap_h.lt(168),n),
          'Discharge stop time missing or unusable':(g.discharge_unusable,n),
          'Early test among algorithmic window >=48 h':(g.loc[g.cap_h.ge(48),'early'],int(g.cap_h.ge(48).sum())),
          'Late test among algorithmic window >48 h':(g.loc[g.cap_h.gt(48),'late'],int(g.cap_h.gt(48).sum())),
        }
        for name,(arr,den) in cases.items():
            k=int(arr.sum());lo,hi=wilson(k,den)
            small=0<k<5
            overall.append({'cohort':cohort,'metric':name,'n':np.nan if small else k,'denominator':den,'percent':np.nan if small else (100*k/den if den else np.nan),'ci_low':np.nan if small else lo,'ci_high':np.nan if small else hi,'display':f'<5/{den} (suppressed)' if small else prop(k,den)})
        for name,col,sub in [('First test hours among observed','first_h',obs),('Last test hours among observed','last_h',obs),('Testing occasions among observed','occasions',obs),('Elapsed 24 h bins with test among observed','tested_bins',obs),('Gap from last test to recorded stop among observed','gap_to_end_h',obs),('Observation opportunity hours','cap_h',g)]:
            overall.append({'cohort':cohort,'metric':name,'n':np.nan,'denominator':len(sub),'percent':np.nan,'ci_low':np.nan,'ci_high':np.nan,'display':qstr(sub[col])})
        def group_row(sub,label):
            o=sub.loc[sub.tested]; avail=sub.loc[sub.cap_h.gt(48)]
            return {'cohort':cohort,'group':label,'eligible_n':len(sub),'observed_n':int(sub.tested.sum()),'observed_pct':100*sub.tested.mean(),'first_test_h_median_iqr':qstr(o.first_h),'occasions_median_iqr':qstr(o.occasions),'opportunity_h_median_iqr':qstr(sub.cap_h),'early_test_n':int(sub.early.sum()),'early_test_pct':100*sub.early.mean(),'late_test_n':int(sub.late.sum()),'late_test_pct':100*sub.late.mean(),'late_opportunity_n':len(avail),'late_among_opportunity_n':int(avail.late.sum()),'late_among_opportunity_pct':100*avail.late.mean() if len(avail) else np.nan}
        for label,sub in g.groupby('risk_quintile',observed=True,sort=True):
            row=group_row(sub,str(label))
            # Early versus any-test differences can disclose small late-only cells.
            row.pop('early_test_n');row.pop('early_test_pct')
            row.update(source_risk_min=float(sub.source_risk.min()),source_risk_max=float(sub.source_risk.max()),source_risk_median=float(sub.source_risk.median()))
            by_risk.append(row)
        public_stop=np.where(g.cap_h.le(48),'<=48 h','>48 h')
        for label,sub in g.groupby(public_stop,sort=True):by_stop.append(group_row(sub,str(label)))
        for d in range(1,8):
            left,right=24*(d-1),24*d; at=g.cap_h.gt(left); full=g.cap_h.ge(right)
            k=int(g[f'day{d}'].sum()); kfull=int(g.loc[full,f'day{d}'].sum())
            lo,hi=wilson(k,int(at.sum()))
            days.append({'cohort':cohort,'elapsed_day':d,'window_hours':f'({left},{right}]','tested_n':k,'all_eligible_n':n,'any_opportunity_n':int(at.sum()),'full_opportunity_n':int(full.sum()),'tested_full_opportunity_n':kfull,'all_eligible_pct':100*k/n,'any_opportunity_pct':100*k/at.sum() if at.sum() else np.nan,'full_opportunity_pct':100*kfull/full.sum() if full.sum() else np.nan,'any_opportunity_ci_low':lo,'any_opportunity_ci_high':hi})
        facts[cohort]={'n':n,'observed':int(g.tested.sum()),'aki_events':int(g.aki.sum()),'early_n':int(g.early.sum()),'late_n':int(g.late.sum()),'both_n':int(g.both.sum()),'one_occasion_n':int(g.occasions.eq(1).sum()),'opportunity_lt48_n':int(g.cap_h.lt(48).sum()),'opportunity_lt168_n':int(g.cap_h.lt(168).sum()),'opportunity_gt48_n':int(g.cap_h.gt(48).sum()),'first_test_h_median_iqr':qstr(obs.first_h),'occasions_median_iqr':qstr(obs.occasions),'gap_h_median_iqr':qstr(obs.gap_to_end_h),'discharge_unusable_n':int(g.discharge_unusable.sum())}
    for g in facts.values():
        if 0<g['discharge_unusable_n']<5:g['discharge_unusable_n']='<5'
    for name,rows in [('S28_overall_timing',overall),('S29_source_risk_strata',by_risk),('S30_followup_strata',by_stop),('S31_elapsed_day_coverage',days)]:
        pd.DataFrame(rows).to_csv(RESULT/(name+'.csv'),index=False)
    (RESULT/'aggregate_facts.json').write_text(json.dumps(facts,indent=2))
    return facts

def confirm_stop_sensitivity(f):
    # Exact excluded denominators stay private to avoid reconstructing small cells.
    rows=[]
    for cohort,g in f.groupby('cohort',sort=False):
        known=~g.discharge_unusable
        choices=[('early',g.cap_h.ge(48),'early'),('late',g.cap_h.gt(48),'late')]
        for d in range(1,8):
            choices += [(f'day{d}_any',g.cap_h.gt(24*(d-1)),f'day{d}'),(f'day{d}_full',g.cap_h.ge(24*d),f'day{d}')]
        for label,mask,col in choices:
            a=g.loc[mask,col]; b=g.loc[mask&known,col]
            delta=100*(b.mean()-a.mean())
            rows.append({'cohort':cohort,'metric':label,'algorithmic_n':len(a),'confirmed_discharge_n':len(b),'algorithmic_pct':100*a.mean(),'confirmed_discharge_pct':100*b.mean(),'difference_pp':delta})
    pd.DataFrame(rows).to_csv(PRIVATE/'confirmed_stop_sensitivity_PRIVATE.csv',index=False)
    passed=bool(max(abs(r['difference_pp']) for r in rows)<1)
    result={'comparison':'Original algorithmic opportunity versus excluding unusable discharge stops','comparisons':len(rows),'all_absolute_changes_less_than_1_percentage_point':passed,'exact_counts':'Withheld from public output to prevent complementary disclosure of small cells','interpretation':'Metadata-restricted sensitivity, not confirmation of complete testing or postdischarge outcomes','plan_deviation':'Unusable discharge combines missing, parsing failure and nonpositive duration; causes and effective death truncation are not separately enumerated'}
    (RESULT/'confirmed_stop_sensitivity_summary.json').write_text(json.dumps(result,indent=2))
    return result

def main():
    unit_tests()
    RESULT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True);PRIVATE.chmod(0o700)
    cfg=json.loads((ANALYSIS/'config/analysis_config_v32.json').read_text())
    raw={k:Path(v).resolve() for k,v in cfg['raw_data'].items() if k in ['inspire_archive','mover_patient_information','mover_patient_labs']}
    checks=[]; manifests=[]; out=[]
    def ck(name,ok,details=''):
        checks.append({'check':name,'passed':bool(ok),'details':details})
        if not ok:raise AssertionError(name+': '+str(details))
    def fingerprint(name,p):
        s=p.stat();manifests.append({'source':name,'path':str(p),'size':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':sha(p)})
    for name,p in raw.items():fingerprint(name,p)
    print('Raw inputs fingerprinted. Extracting retained cohort measurement times.',flush=True)
    source_model_path=ANALYSIS/'models/source_logistic_spline_ipw_v32.joblib'
    fingerprint('frozen_source_model',source_model_path)
    model=joblib.load(source_model_path)
    spec=json.loads((ANALYSIS/'models/model_specification_v32.json').read_text())
    for ds in ['INSPIRE','MOVER']:
        path=ANALYSIS/'data/processed'/('inspire_rebuilt_v32.parquet' if ds=='INSPIRE' else 'mover_rebuilt_v32.parquet')
        fingerprint('frozen_'+ds,path)
        current=retained(pd.read_parquet(path))
        if ds=='MOVER':current=current.loc[current.year.isin([2021,2022])].copy()
        ck(ds+'_unique_case',not current.case_key.duplicated().any())
        current['source_risk']=model.predict_proba(current[FEATURES])[:,1]
        if ds=='MOVER':
            t=current.loc[current.year.eq(2022)]
            pp=expit(spec['recalibration']['alpha']+spec['recalibration']['beta']*logit(t.source_risk.clip(1e-6,1-1e-6)))
            ck('Frozen target 10pct alert count',int((pp>=.1).sum())==980)
        meta={};labs={}; raw_scan=0
        if ds=='INSPIRE':
            with zipfile.ZipFile(raw['inspire_archive']) as z:
                member=next(n for n in z.namelist() if Path(n).name=='operations.csv.gz')
                with gzip.GzipFile(fileobj=z.open(member)) as stream:info=pd.read_csv(stream,dtype=str)
                info['case_key']=[case_key('INSPIRE_CASE',v) for v in info.op_id]
                sel=info.loc[info.case_key.isin(current.case_key)]
                ck(ds+'_unique_raw_operation',not sel.case_key.duplicated().any())
                for r in sel.itertuples():
                    start=float(r.anstart_time);end=float(r.anend_time)
                    dis=pd.to_numeric(r.discharge_time,errors='coerce'); death=pd.to_numeric(r.inhosp_death_time,errors='coerce')
                    valid=[x for x in [dis,death] if pd.notna(x) and x>end]
                    cap=min([end+10080]+valid)
                    meta[r.case_key]={'labkey':r.subject_id,'end':end,'cap_h':(cap-end)/60,'scale':1/60,'discharge_unusable':not(pd.notna(dis) and dis>end),'death_before_7d':pd.notna(death) and end<death<end+10080}
                wanted={x['labkey'] for x in meta.values()}
                member=next(n for n in z.namelist() if Path(n).name=='labs.csv.gz')
                with gzip.GzipFile(fileobj=z.open(member)) as stream:
                    for ch in pd.read_csv(stream,dtype=str,usecols=['subject_id','chart_time','item_name','value'],chunksize=400000):
                        sub=ch.loc[ch.item_name.fillna('').str.strip().str.lower().eq('creatinine')&ch.subject_id.isin(wanted)].copy()
                        sub['v']=pd.to_numeric(sub.value,errors='coerce');sub['t']=pd.to_numeric(sub.chart_time,errors='coerce')
                        sub=sub.loc[sub.v.between(.05,30)&sub.t.notna()]
                        for r in sub.itertuples():labs.setdefault(r.subject_id,[]).append((r.t,r.v))
                        raw_scan+=len(ch)
            print('INSPIRE laboratory scan complete.',flush=True)
        else:
            info=pd.read_csv(raw['mover_patient_information'],dtype=str)
            info['case_key']=[case_key('MOVER_CASE',v) for v in info.LOG_ID]
            sel=info.loc[info.case_key.isin(current.case_key)].copy()
            fields=['AN_STOP_DATETIME','HOSP_DISCH_TIME']
            ck(ds+'_duplicate_metadata_consistency',all(all(g[c].nunique(dropna=False)==1 for c in fields) for _,g in sel.groupby('case_key')))
            sel=sel.drop_duplicates('case_key')
            for r in sel.itertuples():
                end=pd.Timestamp(r.AN_STOP_DATETIME).value/1e9;dis=pd.to_datetime(r.HOSP_DISCH_TIME,errors='coerce')
                dt=dis.value/1e9 if pd.notna(dis) else np.nan;valid=[dt] if pd.notna(dt) and dt>end else []
                cap=min([end+604800]+valid)
                meta[r.case_key]={'labkey':r.LOG_ID,'end':end,'cap_h':(cap-end)/3600,'scale':1/3600,'discharge_unusable':not(pd.notna(dt) and dt>end),'death_before_7d':False}
            wanted={x['labkey'] for x in meta.values()}
            for ch in pd.read_csv(raw['mover_patient_labs'],dtype=str,usecols=['LOG_ID','Lab Code','Lab Name','Observation Value','Measurement Units','Collection Datetime'],chunksize=400000):
                mask=ch.LOG_ID.isin(wanted)&ch['Lab Name'].fillna('').str.strip().str.lower().eq('creatinine')&ch['Lab Code'].fillna('').str.strip().isin(['2160-0','38483-4'])&ch['Measurement Units'].fillna('').str.replace(' ','').str.upper().eq('MG/DL')
                sub=ch.loc[mask].copy();sub['v']=pd.to_numeric(sub['Observation Value'].str.extract(r'([-+]?[0-9]*\.?[0-9]+)',expand=False),errors='coerce');sub['t']=pd.to_datetime(sub['Collection Datetime'],errors='coerce')
                sub=sub.loc[sub.v.between(.05,30)&sub.t.notna()]
                for _,r in sub.iterrows():labs.setdefault(r.LOG_ID,[]).append((r.t.value/1e9,r.v))
                raw_scan+=len(ch)
                if raw_scan%4000000==0:print(f'MOVER laboratory rows scanned: {raw_scan}',flush=True)
        ck(ds+'_all_retained_linked',set(current.case_key)==set(meta))
        mismatch={k:0 for k in ['observed','count48','count7','max48','max7','aki']}
        for r in current.itertuples():
            m=meta[r.case_key]; rows=labs.get(m['labkey'],[])
            converted=[((t-m['end'])*m['scale'],v) for t,v in rows]
            within=[(t,v) for t,v in converted if 0<t<=m['cap_h']]
            d=summarize_times([t for t,v in within],m['cap_h'])
            max7=max([v for t,v in within],default=np.nan);max48=max([v for t,v in within if t<=48],default=np.nan)
            y=int((pd.notna(max48) and max48-r.baseline_cr>=.3-1e-12) or (pd.notna(max7) and max7>=1.5*r.baseline_cr-1e-12)) if within else np.nan
            mismatch['observed']+=d['tested']!=r.tested_7d;mismatch['count48']+=d['raw_rows48']!=r.postop_cr_48h_count;mismatch['count7']+=d['raw_rows']!=r.postop_cr_7d_count
            for k,a,b in [('max48',max48,r.cr_max_48h),('max7',max7,r.cr_max_7d),('aki',y,r.aki)]:mismatch[k]+=not np.isclose(a,b,atol=1e-10,rtol=0,equal_nan=True)
            out.append({'case_key':r.case_key,'cohort':ds if ds=='INSPIRE' else f'MOVER {int(r.year)}','aki':r.aki,'source_risk':r.source_risk,**{k:v for k,v in m.items() if k in ['cap_h','discharge_unusable','death_before_7d']},**d})
        for name,n in mismatch.items():ck(ds+'_raw_reconciliation_'+name,n==0,f'{n} mismatches across {len(current)} patients')
        print(json.dumps({'cohort':ds,'retained':len(current),'raw_rows_scanned':raw_scan,'reconciliation_mismatches':mismatch}),flush=True)
    f=pd.DataFrame(out)
    for name,n,obs,events in [('INSPIRE',33394,24872,1680),('MOVER 2021',2802,2212,274),('MOVER 2022',2587,2033,259)]:
        g=f.loc[f.cohort.eq(name)];ck(name+'_frozen_counts',len(g)==n and g.tested.sum()==obs and g.aki.sum()==events)
        f.loc[g.index,'risk_quintile']=pd.qcut(g.source_risk,5,labels=['Q1','Q2','Q3','Q4','Q5']).astype(str)
    f['stop_group']=np.select([f.cap_h.le(48),f.cap_h.lt(168)],['<=48 h','>48 to <168 h'],default='>=168 h')
    ck('At most seven measured bins',f.tested_bins.between(0,7).all())
    ck('Opportunity nonnegative capped 168h',f.cap_h.gt(0).all() and f.cap_h.le(168+1e-9).all())
    ck('Late testing implies late opportunity',(~f.late|f.cap_h.gt(48)).all())
    ck('Both windows identity',(f.both==(f.early&f.late)).all())
    for d in range(1,8):ck(f'Day {d} test implies opportunity',(~f[f'day{d}']|f.cap_h.gt(24*(d-1))).all())
    f.to_parquet(PRIVATE/'timing_records_PRIVATE.parquet',index=False)
    facts=write_summaries(f)
    sensitivity=confirm_stop_sensitivity(f)
    ck('Metadata-restricted coverage sensitivity below 1pp',sensitivity['all_absolute_changes_less_than_1_percentage_point'])
    for m in manifests:
        s=Path(m['path']).stat();ck('Input untouched '+m['source'],s.st_size==m['size'] and s.st_mtime_ns==m['mtime_ns'])
    pd.DataFrame(manifests).to_csv(PRIVATE/'source_manifest_PRIVATE.csv',index=False)
    pd.DataFrame(checks).to_csv(RESULT/'timing_qa_checks.csv',index=False)
    (RESULT/'timing_run_summary.json').write_text(json.dumps({'date':'2026-10-07','status':'PASS','checks_passed':len(checks),'unit_tests':'PASS','cohorts':facts,'analysis_type':'post hoc descriptive raw-measurement diagnostic; original labels and predictions frozen','statistical_intervals':'Wilson 95% intervals for proportions; conditional on retained cohorts, not full-chain model intervals','privacy':'record-level derivation restricted locally; aggregate outputs only','publication':'Analysis supplied in the v3.4.0 reproducibility release; obtain citation metadata from the repository release'},indent=2))
    print(json.dumps({'status':'PASS','checks':len(checks),'facts':facts}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--test-only',action='store_true');ap.add_argument('--summaries-only',action='store_true');ap.add_argument('--analysis-root',type=Path);ap.add_argument('--output-root',type=Path);args=ap.parse_args()
    if args.analysis_root: ANALYSIS=args.analysis_root.resolve()
    if args.output_root:
        OUT=args.output_root.resolve(); RESULT=OUT/'04_新增分析复现材料'; PRIVATE=OUT/'90_内部核验_请勿上传/检测时序_PRIVATE'
    if args.test_only:unit_tests();print('Boundary unit tests PASS')
    elif args.summaries_only:
        unit_tests();f=pd.read_parquet(PRIVATE/'timing_records_PRIVATE.parquet')
        facts=write_summaries(f);result=confirm_stop_sensitivity(f)
        summary=json.loads((RESULT/'timing_run_summary.json').read_text());summary['cohorts']=facts;summary['metadata_sensitivity']=result
        summary['public_output_postprocessing']='Small-cell counts and linked fields suppressed in the analytical writer; raw-reconciliation checks retained from full extraction'
        (RESULT/'timing_run_summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(result))
    else:
        if ANALYSIS is None:ap.error('--analysis-root is required for raw reconstruction')
        main()
