"""Portable preserved analysis body; invoke through replay.py."""
import argparse


def main():
    """Independent raw-file audit; never imports the production ETL or label code."""
    from pathlib import Path
    from decimal import Decimal, InvalidOperation
    import csv, gzip, hashlib, json, re, zipfile, os, time
    import numpy as np
    import pandas as pd

    A=Path(__file__).resolve().parents[1]
    AUD=A/'audit'
    PRIVATE=A/'private'
    for p in [AUD,PRIVATE]: p.mkdir(parents=True,exist_ok=True)
    os.chmod(PRIVATE,0o700)
    CFG=json.loads((A/'config/analysis_config_v32.json').read_text())
    RAW={k:Path(v).resolve() for k,v in CFG['raw_data'].items() if 'verification_only' not in k}
    checks=[]; evidence=[]; summaries=[]; details={}; fingerprints=[]

    def log(s): print(s,flush=True)
    def check(name,n,bad): checks.append({'check':name,'n':int(n),'discrepancies':int(bad),'passed':int(bad)==0})
    def key(ds,v): return hashlib.sha256(f'{ds}|AKI_LOCAL_EVIDENCE_REDO_2_0|{v}'.encode()).hexdigest()[:20]
    def dec(v):
        try:
            d=Decimal(str(v))
            return d if d.is_finite() else None
        except InvalidOperation: return None
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
        return h.hexdigest()
    def eligible(f):
        m=f.has_baseline_cr.fillna(False)&f.baseline_cr_under4.fillna(False)
        if 'first_eligible' in f: m &= f.first_eligible.fillna(False)
        return f.loc[m].copy()
    def same(a,b):
        if a is None or pd.isna(a): return pd.isna(b)
        return pd.notna(b) and abs(float(a)-float(b))<1e-10
    def label(b,a,c):
        if b is None or c is None: return None
        return int((a is not None and a-b>=Decimal('0.3')) or c>=Decimal('1.5')*b)
    def zipread(name,**kw):
        z=zipfile.ZipFile(RAW['inspire_archive'])
        match=[n for n in z.namelist() if Path(n).name==name]
        assert len(match)==1
        return z,gzip.GzipFile(fileobj=z.open(match[0]))

    assert same(None,np.nan) and same(np.nan,np.nan) and same(Decimal('.3'),.3)
    assert not same(0,np.nan) and not same(np.nan,0) and not same(.3,.31)
    assert label(Decimal('.9'),Decimal('1.2'),Decimal('1.2'))==1
    assert label(Decimal('.6'),None,Decimal('.9'))==1
    assert label(Decimal('1'),Decimal('1.29999999'),Decimal('1.49999999'))==0
    assert label(Decimal('1'),None,None) is None
    check('independent_audit_missing_and_decimal_boundary_unit_tests',10,0)

    for k,p in RAW.items():
        s=p.stat(); fingerprints.append({'source':k,'path':str(p),'size':s.st_size,'mtime_ns_before':s.st_mtime_ns,'sha256':sha(p)})
    log('Raw source files fingerprinted; starting independent extraction.')

    def audit_cases(ds,current,original,rawcases,labgroups,unit,baseline_override=False):
        cur=current.set_index('case_key'); orig=original.set_index('case_key')
        rebuild=[]; changed_n=0; missing_keys=0; tie_n=0
        for seq,(ck,row) in enumerate(cur.iterrows(),1):
            if ck not in rawcases: missing_keys+=1; continue
            meta=rawcases[ck]; start,end,w48,w7=meta['start'],meta['end'],meta['w48'],meta['w7']
            labs=labgroups.get(meta['labkey'],[])
            pre=[x for x in labs if x['t']<start and (baseline_override or x['t']>=start-7*86400/unit)]
            pre.sort(key=lambda x:(-x['t'],x['line']))
            baseline=pre[0] if pre else None; tied=[]
            if baseline:
                tied=[x['d'] for x in pre if x['t']==baseline['t']]
                if len(set(tied))>1: tie_n+=1
            if baseline_override and meta.get('preop') is not None:
                baseline={'d':meta['preop'],'t':None,'line':meta['case_line'],'qualifier':False,'origin':'cases.preop_cr'}
            p48=[x for x in labs if end<x['t']<=w48]
            p7=[x for x in labs if end<x['t']<=w7]
            a=max(p48,key=lambda x:x['d']) if p48 else None
            c=max(p7,key=lambda x:x['d']) if p7 else None
            bd=baseline['d'] if baseline else None; ad=a['d'] if a else None; cd=c['d'] if c else None
            y=label(bd,ad,cd)
            oldy=orig.loc[ck,'aki'] if ck in orig.index else np.nan
            changed=pd.notna(oldy) and y is not None and int(oldy)!=y
            changed_n+=int(changed)
            datum={'case_key':ck,'baseline_cr':float(bd) if bd is not None else np.nan,'cr_max_48h':float(ad) if ad is not None else np.nan,'cr_max_7d':float(cd) if cd is not None else np.nan,'tested_7d':bool(p7),'postop_cr_7d_count':len(p7),'aki':y,'original_aki':oldy,'threshold_changed':changed,'year':row.get('year',None)}
            rebuild.append(datum)
            sources=[baseline,a,c]
            ev={'audit_id':f'{ds.replace(" ","")}-{seq:05d}','dataset':ds,'case_key':ck,'threshold_changed':changed,'old_label':oldy,'revised_label':row.aki,'raw_decimal_label':y,'baseline_decimal':str(bd) if bd is not None else '', 'max48_decimal':str(ad) if ad is not None else '', 'max7_decimal':str(cd) if cd is not None else '', 'absolute_change_decimal':str(ad-bd) if ad is not None and bd is not None else '', 'relative_threshold_decimal':str(Decimal('1.5')*bd) if bd is not None else '', 'baseline_source':baseline.get('origin','labs') if baseline else '', 'raw_baseline_row':baseline['line'] if baseline else None,'raw_max48_row':a['line'] if a else None,'raw_max7_row':c['line'] if c else None,'baseline_hours_before_start':(start-baseline['t'])*unit/3600 if baseline and baseline['t'] is not None else None,'max48_hours_after_end':(a['t']-end)*unit/3600 if a else None,'max7_hours_after_end':(c['t']-end)*unit/3600 if c else None,'followup_hours':(w7-end)*unit/3600,'contributing_qualified_value':any(x and x.get('qualifier',False) for x in sources),'label_matches':same(y,row.aki)}
            evidence.append(ev)
            ev['latest_baseline_conflicting_values']=len(set(tied))>1
            ev['label_sensitive_to_baseline_tie']=any(label(v,ad,cd)!=y for v in tied) if not baseline_override else False
            ev['baseline_tie_min']=str(min(tied)) if len(set(tied))>1 else ''
            ev['baseline_tie_max']=str(max(tied)) if len(set(tied))>1 else ''
        rr=pd.DataFrame(rebuild).set_index('case_key')
        check(ds+'_raw_operation_link',len(cur),missing_keys)
        for col in ['baseline_cr','cr_max_48h','cr_max_7d','postop_cr_7d_count','tested_7d','aki']:
            if col not in cur: continue
            bad=sum(not same(v,cur.loc[k,col]) for k,v in rr[col].items())
            check(ds+'_'+col,len(rr),bad)
        altered=[x for x in evidence if x['dataset']==ds and x['threshold_changed']]
        check(ds+'_changed_labels_decimal_confirmation',len(altered),sum(not x['label_matches'] for x in altered))
        details[ds]={'baseline_same_time_conflicting_values':tie_n,'labels_sensitive_to_baseline_tie':sum(x['label_sensitive_to_baseline_tie'] for x in evidence if x['dataset']==ds),'changed_cases_with_baseline_tie':sum(x['latest_baseline_conflicting_values'] for x in altered),'raw_linked_cases':len(rr),'threshold_changed':changed_n,'changed_contributing_qualified_values':sum(x['contributing_qualified_value'] for x in altered)}
        for yr in ([2021,2022] if ds=='MOVER' else [None]):
            g=rr if yr is None else rr.loc[rr.year.eq(yr)]
            summaries.append({'cohort':ds if yr is None else f'MOVER {yr}','retained_eligible_or_supportive_cases':len(g),'observed_outcomes':int(g.tested_7d.sum()),'raw_decimal_events':int(g.aki.sum()),'threshold_changed':int(g.threshold_changed.sum()),'verified_changed_labels':int(g.threshold_changed.sum())})
        return rr

    # INSPIRE retains released values rather than unavailable pre-release measurements.
    cur=eligible(pd.read_parquet(A/'data/processed/inspire_rebuilt_v32.parquet'))
    orig=eligible(pd.read_parquet(A/'data/processed/inspire_original_v325.parquet'))
    z,stream=zipread('operations.csv.gz'); ops=pd.read_csv(stream,dtype=str);stream.close();z.close()
    ops['case_key']=[key('INSPIRE_CASE',v) for v in ops.op_id]
    rawcases={}; wanted_subjects=set()
    for i,r in ops.loc[ops.case_key.isin(set(cur.case_key)|set(orig.case_key)-set(cur.case_key))].iterrows():
        st=float(r.anstart_time); en=float(r.anend_time)
        caps=[float(v) for v in [r.discharge_time,r.inhosp_death_time] if pd.notna(v) and float(v)>en]
        rawcases[r.case_key]={'start':st,'end':en,'w48':min([en+2880]+caps),'w7':min([en+10080]+caps),'labkey':r.subject_id}
        wanted_subjects.add(r.subject_id)
    removed=ops.loc[ops.case_key.isin(set(orig.case_key)-set(cur.case_key))]
    removed_flags=removed.icd10_pcs.fillna('').str.contains(r'(?:^|[,;\s])1[A-Z0-9]{4}(?=$|[,;\s])')
    check('INSPIRE_removed_eligible_cases_have_five_character_obstetric_code',len(removed),int((~removed_flags).sum()))
    details['scope']={'removed_eligible':len(removed),'removed_previously_observed':int(orig.loc[orig.case_key.isin(removed.case_key),'tested_7d'].sum()),'removed_previous_events':int(orig.loc[orig.case_key.isin(removed.case_key),'aki'].sum()),'new_eligible_cases':len(set(cur.case_key)-set(orig.case_key))}
    groups={}; scanned=0; retained=0
    z,stream=zipread('labs.csv.gz')
    for ch in pd.read_csv(stream,dtype=str,usecols=['subject_id','chart_time','item_name','value'],chunksize=400000):
        mask=ch.item_name.fillna('').str.strip().str.lower().eq('creatinine')&ch.subject_id.isin(wanted_subjects)
        for idx,r in ch.loc[mask].iterrows():
            d=dec(r.value); t=dec(r.chart_time)
            if d is None or t is None or not Decimal('.05')<=d<=Decimal('30'): continue
            groups.setdefault(r.subject_id,[]).append({'t':float(t),'d':d,'line':int(idx)+2,'qualifier':False});retained+=1
        scanned+=len(ch)
    stream.close();z.close()
    details['INSPIRE_scan']={'all_lab_rows_scanned':scanned,'retained_creatinine_rows_for_audit':retained,'measurement_basis':'released INSPIRE 1.4.2 values; not unavailable pre-release laboratory values'}
    ir=audit_cases('INSPIRE',cur,orig,rawcases,groups,60)
    log('INSPIRE raw laboratory audit completed: '+json.dumps(summaries[-1]))

    # MOVER independently reads all raw lab rows, including the full quantile reference.
    cur=eligible(pd.read_parquet(A/'data/processed/mover_rebuilt_v32.parquet'))
    cur=cur.loc[cur.year.isin([2021,2022])]
    orig=eligible(pd.read_parquet(A/'data/processed/mover_original_v325.parquet'))
    info=pd.read_csv(RAW['mover_patient_information'],dtype=str)
    info['case_key']=[key('MOVER_CASE',v) for v in info.LOG_ID]
    sel=info.loc[info.case_key.isin(cur.case_key)]
    fields=['MRN','BIRTH_DATE','ASA_RATING_C','SEX','AN_START_DATETIME','AN_STOP_DATETIME','HOSP_DISCH_TIME','PRIMARY_PROCEDURE_NM','PRIMARY_ANES_TYPE_NM']
    conflicts=sum(any(g[x].nunique(dropna=False)>1 for x in fields) for _,g in sel.groupby('LOG_ID'))
    check('MOVER_retained_operation_duplicate_conflicts',len(cur),conflicts)
    rawcases={};rawids=set()
    for _,r in sel.drop_duplicates('LOG_ID').iterrows():
        st=pd.Timestamp(r.AN_START_DATETIME).value/1e9; en=pd.Timestamp(r.AN_STOP_DATETIME).value/1e9
        dis=pd.to_datetime(r.HOSP_DISCH_TIME,errors='coerce'); caps=[dis.value/1e9] if pd.notna(dis) and dis.value/1e9>en else []
        rawcases[r.case_key]={'start':st,'end':en,'w48':min([en+172800]+caps),'w7':min([en+604800]+caps),'labkey':r.LOG_ID};rawids.add(r.LOG_ID)
    groups={};globalvalues=[];scanned=0;qualified=0
    for ch in pd.read_csv(RAW['mover_patient_labs'],dtype=str,usecols=['LOG_ID','Lab Code','Lab Name','Observation Value','Measurement Units','Collection Datetime'],chunksize=300000):
        mask=ch['Lab Name'].fillna('').str.strip().str.lower().eq('creatinine')&ch['Lab Code'].fillna('').str.strip().isin(['2160-0','38483-4'])&ch['Measurement Units'].fillna('').str.replace(' ','').str.upper().eq('MG/DL')
        sub=ch.loc[mask].copy(); times=pd.to_datetime(sub['Collection Datetime'],errors='coerce')
        for (idx,r),t in zip(sub.iterrows(),times):
            literal=str(r['Observation Value']);m=re.search(r'[-+]?[0-9]*\.?[0-9]+',literal)
            d=dec(m.group()) if m else None
            if d is None or not Decimal('.05')<=d<=Decimal('30') or pd.isna(t): continue
            q=not bool(re.fullmatch(r'\s*[-+]?(?:[0-9]*\.)?[0-9]+\s*',literal));qualified+=int(q)
            globalvalues.append(float(d))
            if r.LOG_ID in rawids: groups.setdefault(r.LOG_ID,[]).append({'t':t.value/1e9,'d':d,'line':int(idx)+2,'qualifier':q})
        scanned+=len(ch)
        if scanned%3000000==0: log('MOVER raw lab rows scanned: '+str(scanned))
    details['MOVER_scan']={'all_lab_rows_scanned':scanned,'strict_blood_creatinine_rows':len(globalvalues),'qualified_or_nonplain_numeric_values':qualified}
    mr=audit_cases('MOVER',cur,orig,rawcases,groups,1)
    edges=np.quantile(globalvalues,np.array([2.5]+list(np.arange(7.5,100,5)))/100)
    reps=np.quantile(globalvalues,np.array([2.5]+list(np.arange(5,100,5))+[97.5])/100)
    qc=[]
    cc=cur.set_index('case_key')
    for ck,r in mr.iterrows():
        vals=[]
        for col in ['baseline_cr','cr_max_48h','cr_max_7d']:
            value=r[col]
            co=None if pd.isna(value) else Decimal(str(round(float(reps[sum(value>e for e in edges)]),10)))
            vals.append(co);qc.append(not same(co,cc.loc[ck,col+'_coarse']))
        y=label(*vals);mr.loc[ck,'coarse_label_raw']=np.nan if y is None else y
    check('MOVER_independent_coarsening_values',len(qc),sum(qc))
    check('MOVER_independent_coarsened_labels',len(mr),sum(not same(None if pd.isna(y) else y,cc.loc[k,'outcome_coarsened_operational']) for k,y in mr.coarse_label_raw.items()))
    target=mr.loc[mr.year.eq(2022)&mr.tested_7d];changed=target.aki.ne(target.coarse_label_raw)
    details['coarsening']={'observed_denominator':len(target),'changed_labels':int(changed.sum()),'non_AKI_to_AKI':int(((target.aki==0)&(target.coarse_label_raw==1)).sum()),'AKI_to_non_AKI':int(((target.aki==1)&(target.coarse_label_raw==0)).sum()),'quantiles_recomputed_from_all_valid_raw_MOVER_creatinine':True,'not_an_estimate_of_actual_INSPIRE_misclassification':True}
    oldcoarse=orig.set_index('case_key')['outcome_coarsened_operational']
    details['coarsening_boundary_corrections']={str(yr):int((g.coarse_label_raw.notna()&g.coarse_label_raw.ne(oldcoarse.reindex(g.index))).sum()) for yr,g in mr.groupby('year')}
    check('MOVER2022_paired_coarsening_change_count',len(target),int(changed.sum()!=163))
    coarse_changed_keys=set(target.loc[changed].index)
    for ev in evidence:
        ev['paired_coarsening_changed']=ev['dataset']=='MOVER' and ev['case_key'] in coarse_changed_keys
        if ev['dataset']=='MOVER':ev['raw_coarsened_label']=mr.loc[ev['case_key'],'coarse_label_raw']
    log('MOVER raw audit completed: '+json.dumps(summaries[-2:]))

    # VitalDB reproduces the explicitly supportive release-specific definition.
    allcur=pd.read_parquet(A/'data/processed/vitaldb_supportive_v32.parquet')
    cur=eligible(allcur);cur=cur.loc[cur.tested_7d&cur.aki.notna()]
    orig=pd.read_parquet(A/'data/processed/vitaldb_supportive_v2.parquet')
    cases=pd.read_csv(RAW['vitaldb_cases'],dtype=str);cases['case_key']=[key('VITALDB_CASE',v) for v in cases.caseid]
    rawcases={};rawids=set()
    for idx,r in cases.loc[cases.case_key.isin(cur.case_key)].iterrows():
        st=float(r.opstart);en=float(r.opend)
        rawcases[r.case_key]={'start':st,'end':en,'w48':en+172800,'w7':en+604800,'labkey':r.caseid,'preop':dec(r.preop_cr),'case_line':idx+2};rawids.add(r.caseid)
    groups={};labs=pd.read_csv(RAW['vitaldb_labs'],dtype=str)
    for idx,r in labs.loc[labs.name.fillna('').str.strip().str.lower().eq('cr')&labs.caseid.isin(rawids)].iterrows():
        d=dec(r['result']);t=dec(r['dt'])
        if d is None or t is None:continue
        groups.setdefault(r.caseid,[]).append({'t':float(t),'d':d,'line':int(idx)+2,'qualifier':False})
    vr=audit_cases('VitalDB',cur,orig,rawcases,groups,1,baseline_override=True)
    details['VitalDB_definition_boundary']={'baseline':'cases.preop_cr preferred, otherwise latest preoperative lab without seven-day restriction','time_origin':'surgery end rather than anesthesia end','discharge_censoring':False,'numeric_range_filter':False,'interpretation':'supportive available observed cohort only, not harmonized eligible population or independent hospital confirmation'}
    log('VitalDB raw audit completed: '+json.dumps(summaries[-1]))

    ev=pd.DataFrame(evidence)
    ev.to_csv(PRIVATE/'record_level_evidence_PRIVATE.csv',index=False)
    ev.loc[ev.threshold_changed|ev.paired_coarsening_changed].to_csv(PRIVATE/'affected_record_evidence_PRIVATE.csv',index=False)
    for fp in fingerprints:
        p=Path(fp['path']);s=p.stat();fp['mtime_ns_after']=s.st_mtime_ns;fp['size_after']=s.st_size
        fp['unchanged_during_audit']=fp['mtime_ns_before']==s.st_mtime_ns and fp['size']==s.st_size
        check('source_unchanged_'+fp['source'],1,int(not fp['unchanged_during_audit']))
    pd.DataFrame(fingerprints).to_csv(PRIVATE/'raw_source_manifest_PRIVATE.csv',index=False)
    pd.DataFrame(checks).to_csv(AUD/'raw_audit_checks.csv',index=False)
    pd.DataFrame(summaries).to_csv(AUD/'raw_audit_cohort_summary.csv',index=False)
    report={'audit_date':'2026-10-05','implementation':'independent raw CSV/ZIP extraction and Decimal threshold comparison; no production ETL or classifier imports','status':'PASS' if all(x['passed'] for x in checks) else 'DISCREPANCIES_FOUND','checks':checks,'summaries':summaries,'details':details,'privacy':'record-level evidence is local restricted internal material; not for public release','boundaries':['Computational source-data verification, not physician chart adjudication.','INSPIRE verification is against publicly released representations, not unavailable pre-release lab values.','No inference about AKI status in untested patients.','No new clinical validation or publication authorization.']}
    (AUD/'raw_audit_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    log(json.dumps({k:report[k] for k in ['status','summaries','details']},ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    main()
