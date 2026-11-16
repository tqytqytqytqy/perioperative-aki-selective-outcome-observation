"""Unchanged calculation statements from the actual upstream VitalDB raw ETL."""
import numpy as np
import pandas as pd
import etl_common_v32 as base
from thresholds_r1 import ge_threshold

def build_vitaldb(cases, labs):
    creatinine = labs.loc[labs['name'].fillna('').astype(str).str.strip().str.casefold().eq('cr')].copy()
    creatinine['result'] = pd.to_numeric(creatinine['result'], errors='coerce')
    creatinine = creatinine.loc[creatinine['result'].notna()].merge(cases[['caseid', 'opstart', 'opend', 'preop_cr']], on='caseid', how='left')
    pre = creatinine.loc[creatinine['dt'] < creatinine['opstart']]
    pre = pre.sort_values(['caseid', 'dt']).groupby('caseid', sort=False).tail(1)
    post48 = creatinine.loc[(creatinine['dt'] > creatinine['opend']) & (creatinine['dt'] <= creatinine['opend'] + 2 * 86400)]
    post7 = creatinine.loc[(creatinine['dt'] > creatinine['opend']) & (creatinine['dt'] <= creatinine['opend'] + 7 * 86400)]
    derived = cases.set_index('caseid').copy()
    derived['baseline_cr_lab'] = pre.set_index('caseid')['result']
    derived['baseline_cr'] = pd.to_numeric(derived['preop_cr'], errors='coerce').combine_first(derived['baseline_cr_lab'])
    derived['cr_max_48h'] = post48.groupby('caseid')['result'].max()
    derived['cr_max_7d'] = post7.groupby('caseid')['result'].max()
    derived['postop_cr_7d_count'] = post7.groupby('caseid')['result'].count().reindex(derived.index).fillna(0)
    derived['tested_7d'] = derived['postop_cr_7d_count'].gt(0)
    derived['has_baseline_cr'] = derived['baseline_cr'].notna() & derived['baseline_cr'].gt(0)
    derived['baseline_cr_under4'] = derived['baseline_cr'].lt(4.0)
    derived['aki'] = (ge_threshold(derived['cr_max_48h'] - derived['baseline_cr'], 0.3) | ge_threshold(derived['cr_max_7d'], 1.5 * derived['baseline_cr'])).astype(float)
    derived.loc[~(derived['has_baseline_cr'] & derived['tested_7d']), 'aki'] = np.nan
    derived['dataset'] = 'VitalDB'
    derived['patient_key'] = [base.stable_key('VITALDB_CASE', caseid) for caseid in derived.index]
    derived['case_key'] = derived['patient_key']
    derived['age'] = pd.to_numeric(derived['age'], errors='coerce')
    derived['sex_male'] = derived['sex'].fillna('').astype(str).str.upper().str.startswith('M').astype(float)
    derived['asa'] = pd.to_numeric(derived['asa'], errors='coerce')
    derived['duration_h'] = (derived['opend'] - derived['opstart']) / 3600.0
    derived['adult_general_duration'] = derived['age'].ge(18) & derived['ane_type'].fillna('').astype(str).str.contains('General', case=False) & derived['opend'].gt(derived['opstart']) & derived['duration_h'].ge(0.5)
    derived['aki_48h'] = np.nan
    cohort = derived[['dataset', 'patient_key', 'case_key', 'age', 'sex_male', 'asa', 'duration_h', 'baseline_cr', 'tested_7d', 'aki', 'aki_48h', 'adult_general_duration', 'has_baseline_cr', 'baseline_cr_under4', 'postop_cr_7d_count']].reset_index(drop=True)
    return cohort
