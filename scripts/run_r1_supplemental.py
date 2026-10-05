"""Portable preserved analysis body; invoke through replay.py."""
import argparse


def main():
    from pathlib import Path
    import json, joblib, hashlib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    import run_v32_analysis as r
    import etl_outcome_representations_v32 as rep

    ROOT=Path(__file__).resolve().parents[1]; T=ROOT/'tables'; Q=ROOT/'qa'; D=ROOT/'data/processed'
    cfg=r.load_config(); cohorts=r.prepare_cohorts()
    model=joblib.load(ROOT/'models/source_logistic_spline_ipw_v32.joblib')
    spec=json.loads((ROOT/'models/model_specification_v32.json').read_text())
    alpha,beta=spec['recalibration']['alpha'],spec['recalibration']['beta']
    rows=[]
    for key in ['update','target']:
        f=cohorts[key]; f=f.loc[r.observed_indicator(f)].copy()
        y=f.outcome_operational.to_numpy().astype(int); z=f.outcome_coarsened_operational.to_numpy().astype(int)
        row={'cohort':f.analysis_phase.iloc[0],'observed_n':len(f),'original_AKI':int(y.sum()),'coarsened_AKI':int(z.sum())}
        for a in [0,1]:
            for b in [0,1]: row[f'original_{a}_coarsened_{b}']=int(((y==a)&(z==b)).sum())
        row.update(discordant_n=int((y!=z).sum()),discordant_percent=float((y!=z).mean()*100),net_event_change=int(z.sum()-y.sum()),nonAKI_to_AKI_percent_of_original_nonAKI=float(((y==0)&(z==1)).sum()/(y==0).sum()*100),AKI_to_nonAKI_percent_of_original_AKI=float(((y==1)&(z==0)).sum()/(y==1).sum()*100))
        rows.append(row)
    pd.DataFrame(rows).to_csv(T/'R1_S21_paired_reclassification.csv',index=False)
    target=cohorts['target']; obs=r.observed_indicator(target)
    pred=r.base.apply_recalibration(model.predict_proba(target[r.FEATURES])[:,1],alpha,beta)[obs]
    f=target.loc[obs]; y=f.outcome_operational.to_numpy(); z=f.outcome_coarsened_operational.to_numpy()
    def metrics(label,p): return {'event_rate':label.mean(),'oe_ratio':label.sum()/p.sum(),'auroc':roc_auc_score(label,p)}
    p0=metrics(y,pred); p1=metrics(z,pred)
    rng=np.random.default_rng(20260930); b_rows=[]
    for b in range(1000):
        ix=rng.integers(0,len(y),len(y)); m0=metrics(y[ix],pred[ix]); m1=metrics(z[ix],pred[ix])
        row={'replicate':b}
        for m in p0: row[f'original_{m}']=m0[m]; row[f'coarsened_{m}']=m1[m]; row[f'difference_{m}']=m1[m]-m0[m]
        b_rows.append(row)
    bs=pd.DataFrame(b_rows); bs.to_parquet(D/'R1_paired_fixed_prediction_bootstrap.parquet',index=False)
    results=[]
    for m in p0:
        for mode,p in [('original',p0[m]),('coarsened',p1[m]),('difference',p1[m]-p0[m])]:
            ci=bs[f'{mode}_{m}'].quantile([.025,.975])
            results.append({'metric':m,'contrast':mode,'estimate':p,'ci_lower':ci.iloc[0],'ci_upper':ci.iloc[1],'observed_n':len(y),'replicates':1000,'prediction_policy':'fixed revised primary model and original predictor values'})
    pd.DataFrame(results).to_csv(T/'R1_S22_fixed_prediction_metrics.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(results).to_string(index=False))
    levels=pd.read_csv(T/'51_mover_deterministic_coarsening_v32.csv')
    edges=levels.upper_cutpoint_mg_dl.dropna().to_numpy()
    representatives=levels.released_representative_mg_dl.to_numpy()
    g=pd.read_parquet(D/'mover_rebuilt_v32.parquet')
    mapchecks=[]
    for col in ['baseline_cr','cr_max_48h','cr_max_7d']:
        mapped=rep.apply_percentile_release(g[col],edges,representatives)
        mapchecks.append({'check':'reproduce_stored_'+col+'_coarse','passed':bool(np.allclose(mapped,g[col+'_coarse'],atol=1e-12,equal_nan=True))})
    assert all(c['passed'] for c in mapchecks)
    mapping={'boundary_rule':'searchsorted side=left: values exactly at a cutpoint enter the lower interval','repeated_cutpoint_pairs':int((np.diff(edges)==0).sum()),'empty_bins_due_to_tied_cutpoints':int((np.diff(edges)==0).sum()),'coarsening_population':'full retained strict blood-creatinine release used in the original deterministic mapping; not re-estimated by cohort','frozen_prediction_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),'fixed_prediction_n':len(pred),'tests':mapchecks}
    (Q/'R1_coarsening_checks.json').write_text(json.dumps(mapping,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    main()
