"""Portable preserved analysis body; invoke through replay.py."""
import argparse


def main():
    from pathlib import Path
    import os,sys,json
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[name]='1'
    import numpy as np
    import pandas as pd
    A=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(A/'scripts'))
    import run_v32_analysis as r
    import joblib
    from scipy.interpolate import BSpline
    from scipy.special import expit,logit
    spec=json.loads((A/'models/model_specification_v32.json').read_text())
    target=r.prepare_cohorts()['target']
    design=[]
    for entry in spec['continuous_preprocessing']:
        raw=target[entry['feature']].to_numpy(dtype=float)
        filled=np.where(np.isfinite(raw),raw,entry['imputation_value'])
        z=(filled-entry['standardization_mean'])/entry['standardization_scale']
        knots=np.asarray(entry['spline_knot_vector_standardized'])
        degree=spec['estimator_parameters']['spline_degree']
        z=np.clip(z,knots[degree],knots[-degree-1])
        design.append(BSpline(knots,np.eye(len(knots)-degree-1),degree)(z)[:,:-1])
    binary=target[spec['binary_preprocessing']['feature']].to_numpy(dtype=float)
    binary=np.where(np.isfinite(binary),binary,spec['binary_preprocessing']['imputation_value'])
    design.append(binary[:,None])
    manual=expit(spec['source_intercept']+np.column_stack(design)@np.asarray(spec['source_coefficients']))
    pipeline=joblib.load(A/'models/source_logistic_spline_ipw_v32.joblib')
    stored=pipeline.predict_proba(target[r.FEATURES])[:,1]
    err=float(np.max(np.abs(manual-stored)))
    model_check={'scope':'Reconstruct source predictions from JSON coefficients, imputation, scaling and SciPy B-spline bases; compare with saved pipeline','n':len(target),'max_absolute_probability_difference':err,'passed':bool(np.allclose(manual,stored,atol=1e-12,rtol=0))}
    (A/'qa/R1_JSON_model_reconstruction.json').write_text(json.dumps(model_check,indent=2))
    print(json.dumps(model_check),flush=True)
    assert model_check['passed']
    b=pd.read_parquet(A/'data/processed/canonical_bootstrap_distribution_v32.parquet')
    m=pd.read_parquet(A/'data/processed/mnar_chain_bootstrap_v32.parquet')
    r.bootstrap_worker_init(r.load_config())
    ids=sorted(b.replicate.unique())
    checks=[]
    for replicate in [int(ids[0]),int(ids[len(ids)//2]),int(ids[-1])]:
        fresh=r.bootstrap_worker_run(replicate)
        old=b.loc[b.replicate.eq(replicate)].iloc[0]
        differences=[key for key,value in fresh['canonical'].items() if not np.isclose(value,old[key],atol=1e-10,rtol=1e-10)]
        mf=pd.DataFrame(fresh['mnar']).sort_values(['stage_varied','odds_multiplier']).reset_index(drop=True)
        mo=m.loc[m.replicate.eq(replicate)].sort_values(['stage_varied','odds_multiplier']).reset_index(drop=True)
        cols=mf.select_dtypes('number').columns
        equal=np.allclose(mf[cols],mo[cols],atol=1e-6,rtol=0,equal_nan=True)
        diffs={c:float(np.nanmax(np.abs(mf[c]-mo[c]))) for c in cols if not np.allclose(mf[c],mo[c],atol=1e-10,rtol=1e-10,equal_nan=True)}
        check={'replicate':replicate,'canonical_equal':not differences,'canonical_mismatches':differences,'all_18_MNAR_within_1e_6':bool(equal),'mnar_max_differences_above_1e_10':diffs,'fresh_failed_attempts':len(fresh['errors'])}
        checks.append(check)
        print(json.dumps(check),flush=True)
    (A/'qa/R1_independent_replay_spotcheck.json').write_text(json.dumps({'scope':'Three deterministic full-chain replicates rerun from corrected patient derivatives, not read from checkpoint; same algorithm and environment, not independent statistical adjudication','canonical_tolerance':1e-10,'mnar_absolute_tolerance':1e-6,'numeric_note':'A strict 1e-10 comparison detected MNAR optimizer-result differences up to 1.243e-7 between the original parallel run and this serial replay. These are retained below, not asserted bit-identical. All are below 1e-6, much smaller than displayed clinical metric precision.','checks':checks},indent=2))
    assert all(c['canonical_equal'] and c['all_18_MNAR_within_1e_6'] and c['fresh_failed_attempts']==0 for c in checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    main()
