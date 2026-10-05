"""Portable preserved analysis body; invoke through replay.py."""
import argparse


def main():
    """Post-audit baseline tie sensitivity within the original retained starting set."""
    import os,sys,json
    from pathlib import Path
    from decimal import Decimal
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
    import pandas as pd
    import numpy as np
    A=Path(__file__).resolve().parents[1]
    O=A
    sys.path.insert(0,str(A/'scripts'))
    sys.dont_write_bytecode=True
    import run_v32_analysis as r
    e=pd.read_csv(O/'private/record_level_evidence_PRIVATE.csv',low_memory=False)
    e=e.loc[e.latest_baseline_conflicting_values].set_index('case_key')
    cfg=r.load_config();seed=int(cfg['random_seed']);rows=[]
    for scenario in ['original_row_order','minimum_tied_baseline','maximum_tied_baseline']:
        cohorts=r.prepare_cohorts();changed=0;labels_changed=0;removed={}
        for phase in ['source','update','target']:
            f=cohorts[phase]
            if scenario!='original_row_order':
                col='baseline_tie_min' if scenario.startswith('minimum') else 'baseline_tie_max'
                for idx in f.index[f.case_key.isin(e.index)]:
                    rec=e.loc[f.at[idx,'case_key']];b=Decimal(str(rec[col]));assert Decimal('.05')<=b<=Decimal('30')
                    changed+=int(float(b)!=f.at[idx,'baseline_cr']);f.at[idx,'baseline_cr']=float(b)
                    if f.at[idx,'tested_7d']:
                        a=Decimal(str(rec.max48_decimal)) if pd.notna(rec.max48_decimal) else None
                        c=Decimal(str(rec.max7_decimal)) if pd.notna(rec.max7_decimal) else None
                        y=int((a is not None and a-b>=Decimal('.3')) or (c is not None and c>=b*Decimal('1.5')))
                        labels_changed+=int(f.at[idx,'aki']!=y)
                        f.at[idx,'aki']=y;f.at[idx,'outcome_operational']=y
            removed[phase]=int((f.baseline_cr>=4).sum())
            cohorts[phase]=f.loc[f.baseline_cr<4].copy().reset_index(drop=True)
        bundles={phase:r.observation_bundle(cohorts[phase],label,cfg,seed+offset,include_basic=False,fit_outcome=True,fit_final=True) for phase,label,offset in [('source','INSPIRE',1000),('update','MOVER 2021',2000),('target','MOVER 2022',3000)]}
        result=r.canonical_result(cohorts['source'],cohorts['update'],cohorts['target'],bundles['source'],bundles['update'],bundles['target'],cfg,seed+4000)
        m=result['metrics'];rows.append({'scenario':scenario,'baseline_values_changed':changed,'labels_changed_before_eligibility_recheck':labels_changed,**{phase+'_removed_at_baseline_recheck':removed[phase] for phase in removed},**{phase+'_n':len(cohorts[phase]) for phase in removed},'source_events':int(cohorts['source'].aki.sum()),'update_events':int(cohorts['update'].aki.sum()),'target_observed_events':int(cohorts['target'].aki.sum()),'alpha':result['alpha'],'beta':result['beta'],**{x:m[x] for x in ['oe_ratio','calibration_slope','auroc','brier']},'alerts_10pct':int((result['prediction']>=.1).sum())})
        print(json.dumps(rows[-1]),flush=True)
    pd.DataFrame(rows).to_csv(O/'audit/Table_S27_baseline_tie_sensitivity.csv',index=False)
    stored=pd.read_csv(A/'tables/31_canonical_primary_bootstrap_v32.csv').set_index('metric')
    match=all(np.isclose(rows[0][x],stored.loc[x,'estimate'],rtol=0,atol=1e-8) for x in ['oe_ratio','calibration_slope','auroc','brier'])
    assert match
    (O/'audit/baseline_tie_sensitivity.json').write_text(json.dumps({'original_point_estimates_reproduced':match,'audit_discovered_post_hoc_analysis':True,'original_retained_starting_set':True,'baseline_below4_eligibility_reapplied_after_substitution':True,'no_additions_from_previously_excluded_cases_assessed':True,'all_observation_and_auxiliary_models_source_model_and_update_refitted':True,'bootstrap_rerun':False,'clinical_adjudication_performed':False,'primary_rule_unchanged':'first record in released file among latest-time ties','results':rows},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    main()
