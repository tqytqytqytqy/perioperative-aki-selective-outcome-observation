"""Portable preserved analysis body; invoke through replay.py."""
import argparse


def main():
    from pathlib import Path
    from decimal import Decimal
    import json, shutil
    import numpy as np
    import pandas as pd
    import etl_common_v32 as e
    import etl_outcome_representations_v32 as rep
    from thresholds_r1 import ge_threshold

    ROOT=Path(__file__).resolve().parents[1]
    DATA=ROOT/'data/processed'
    checks=[]; counts=[]
    def decimal_aki(b,a,c):
        if not np.isfinite(b): return np.nan
        # Independent decimal check removes storage artifacts far below lab resolution.
        db=Decimal(str(round(b,10)))
        return float((np.isfinite(a) and Decimal(str(round(a,10)))-db>=Decimal('0.3')) or (np.isfinite(c) and Decimal(str(round(c,10)))>=Decimal('1.5')*db))
    for dataset,stem in [('INSPIRE','inspire'),('MOVER','mover')]:
        p=DATA/f'{stem}_rebuilt_v32.parquet'
        original=DATA/f'{stem}_original_v325.parquet'
        if not original.exists(): shutil.copy2(p,original)
        f=pd.read_parquet(DATA/'inspire_cohort_scope_corrected.parquet' if stem=='inspire' else original)
        f=e.finalize_outcomes(f)
        f['outcome_operational']=f.aki
        suffixes=[''] if stem=='inspire' else ['', '_coarse']
        for suffix in suffixes:
            b,a,c=[f[x+suffix].to_numpy() for x in ['baseline_cr','cr_max_48h','cr_max_7d']]
            reference=np.array([decimal_aki(x,y,z) for x,y,z in zip(b,a,c)])
            reference[~f.tested_7d.to_numpy()]=np.nan
            y=rep.classify_exact_aki(f,'baseline_cr'+suffix,'cr_max_48h'+suffix,'cr_max_7d'+suffix).to_numpy()
            eq=np.allclose(reference,y,equal_nan=True,atol=0,rtol=0)
            checks.append(dict(check=dataset+suffix+'_Decimal_10dp_equivalence',passed=bool(eq),n=len(f)))
            if not eq:
                bad=np.isfinite(reference)&(reference!=y)
                print(dataset,suffix,'decimal mismatches',int(bad.sum()),flush=True)
            assert eq
            if suffix: f['outcome_coarsened_operational']=y
        if stem=='inspire': f['outcome_coarsened_operational']=f.outcome_operational
        pref='' if stem=='inspire' else '_coarse'
        f['outcome_definite'],f['outcome_possible']=rep.classify_interval_aki(f,*[x+pref+'_'+s for x,s in [('baseline_cr','lower'),('baseline_cr','upper'),('cr_max_48h','lower'),('cr_max_48h','upper'),('cr_max_7d','lower'),('cr_max_7d','upper')]])
        f.to_parquet(p,index=False)
        for year in ([None] if stem=='inspire' else [2021,2022]):
            g=rep.eligible_base(f,year)
            counts.append(dict(cohort=dataset if year is None else f'{dataset} {year}',eligible=len(g),observed=int(g.tested_7d.sum()),events=int(g.aki.sum())))
    for x,want in [(.3,True),(.3-1e-8,False),(.3+1e-8,True)]:
        assert bool(ge_threshold(x,.3))==want
    assert bool(ge_threshold(1.2-.9,.3))
    assert bool(ge_threshold(.9,1.5*.6))
    assert not bool(ge_threshold(np.nan,.3))
    checks.append(dict(check='inclusive_boundary_and_near_boundary_unit_tests',passed=True,n=6))
    pd.DataFrame(checks).to_csv(ROOT/'qa/R1_threshold_tests.csv',index=False)
    pd.DataFrame(counts).to_csv(ROOT/'tables/R1_corrected_denominators.csv',index=False)
    print(pd.DataFrame(counts).to_string(index=False),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    main()
