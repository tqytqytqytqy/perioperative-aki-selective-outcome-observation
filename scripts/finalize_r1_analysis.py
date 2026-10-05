"""Portable preserved analysis body; invoke through replay.py."""
import argparse


def main():
    from pathlib import Path
    import json, shutil, hashlib, platform
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import build_v32_figures as figures
    ROOT=Path(__file__).resolve().parents[1]; T=ROOT/'tables'; D=ROOT/'data/processed'; Q=ROOT/'qa'
    boot=pd.read_parquet(D/'canonical_bootstrap_distribution_v32.parquet')
    assert len(boot)==1000 and 'comparison_complete_case_update_alpha' in boot
    assert np.isfinite(boot.select_dtypes('number')).all().all()
    mnar=pd.read_parquet(D/'mnar_chain_bootstrap_v32.parquet')
    assert len(mnar)==18000 and mnar.groupby(['stage_varied','odds_multiplier']).size().eq(1000).all()
    comparators=pd.read_csv(T/'37_selection_chain_primary_comparators_v32.csv')
    three=comparators.loc[comparators.source_model_estimand.eq('1/99 truncated-IPW source model')].copy()
    three.to_csv(T/'36_main_table2_propagation_v32.csv',index=False)
    rows=[]; contrasts=[]
    for (_,p),label in zip(three.iterrows(),['unupdated','complete_case','IPW']):
        for metric in ['update_alpha','update_beta','oe_ratio','calibration_slope','auroc','brier']:
            col=metric if label=='IPW' else f'comparison_{label}_{metric}'
            fixed=label=='unupdated' and metric in ['update_alpha','update_beta']
            ci=[np.nan,np.nan] if fixed else boot[col].quantile([.025,.975]).tolist()
            rows.append(dict(strategy=label,metric=metric,estimate=float(p[metric]),ci_lower=ci[0],ci_upper=ci[1],fixed=fixed,replicates=0 if fixed else 1000))
    for metric in ['oe_ratio','calibration_slope','auroc','brier']:
        diff=boot[metric]-boot[f'comparison_complete_case_{metric}']
        contrasts.append(dict(contrast='IPW minus complete-case recalibration',metric=metric,estimate=float(three.iloc[2][metric]-three.iloc[1][metric]),ci_lower=diff.quantile(.025),ci_upper=diff.quantile(.975),replicates=1000))
    ci=pd.DataFrame(rows); ci.to_csv(T/'R1_S25_strategy_bootstrap_intervals.csv',index=False)
    pd.DataFrame(contrasts).to_csv(T/'R1_S25_paired_strategy_differences.csv',index=False)
    for name in ['31_canonical_primary_bootstrap_v32.csv','37_main_table3_canonical_performance_v32.csv']:
        f=pd.read_csv(T/name)
        f['estimator_component']=f.estimator_component.str.replace('AIPW observed-event total','AIPW-estimated event total',regex=False)
        f.to_csv(T/name,index=False)
    vmap=pd.read_csv(T/'48_raw_to_analysis_variable_map_v32.csv')
    vmap.loc[vmap.analysis_variable.eq('obstetric exclusion'),'derivation']='OG department or section-1 PCS tokens of five or seven characters'
    vmap.loc[vmap.analysis_variable.eq('cardiac exclusion'),'analysis_use_or_note']='Independent CTS exclusion can also exclude noncardiac thoracic operations; absence of a supporting code is not clinical adjudication; scope not equivalent to MOVER'
    vmap.loc[vmap.analysis_variable.eq('obstetric exclusion'),'analysis_use_or_note']='Independent OG exclusion can also exclude nonobstetric gynecologic operations; absence of a supporting code is not clinical adjudication; scope not equivalent to MOVER'
    vmap.loc[vmap.analysis_variable.eq('aki'),'derivation']='Inclusive >=0.3 mg/dL by 48 h or >=1.5 times baseline by day 7; absolute numerical comparison tolerance 1e-12 mg/dL'
    vmap.loc[vmap.analysis_variable.eq('aki')&vmap.dataset.eq('VitalDB'),'analysis_use_or_note']='Supportive only; thresholds corrected; release-specific baseline and follow-up definitions differ from primary cohorts'
    vmap.to_csv(T/'48_raw_to_analysis_variable_map_v32.csv',index=False)
    figures.figure1(); figures.figure_s1(); figures.figure_s2(); figures.figure_s3(); figures.figure_s4()
    fig,axes=plt.subplots(2,2,figsize=(8.5,6.4),constrained_layout=True)
    labels=['No update','Complete-case\nrecalibration','IPW\nrecalibration']; colors=['#60646C','#A76426','#087E8B']
    for ax,metric,ylabel,ref in zip(axes.flat,['update_alpha','update_beta','oe_ratio','calibration_slope'],['Recalibration intercept alpha','Recalibration slope beta','Target O/E','Target calibration slope'],[0,1,1,1]):
        for j,label in enumerate(['unupdated','complete_case','IPW']):
            row=ci.loc[ci.strategy.eq(label)&ci.metric.eq(metric)].iloc[0]
            if row.fixed:
                ax.plot(j,row.estimate,marker='s',color=colors[j],ms=6)
                ax.annotate('Fixed',(j,row.estimate),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
            else:
                ax.errorbar(j,row.estimate,yerr=[[row.estimate-row.ci_lower],[row.ci_upper-row.estimate]],fmt='o',color=colors[j],capsize=4,lw=1.5)
        ax.axhline(ref,ls='--',color='#A23B3B',lw=.8)
        ax.set_xticks(range(3),labels,fontsize=8); ax.set_ylabel(ylabel,fontsize=10); ax.set_xlim(-.4,2.4)
        figures.style_axis(ax)
    figures.save(fig,'Figure_2_selection_propagation_v32')
    version='R1_20260930_corrected_analysis'
    cfg=json.loads((ROOT/'config/analysis_config_v32.json').read_text()); cfg['analysis_version']=version; cfg['release_status']='local author-review revision; not publicly archived'
    cfg['corrections']=['INSPIRE five-character obstetric PCS exclusion','inclusive AKI numerical threshold boundaries in all datasets']
    cfg['bootstrap_replicates']=1000
    (ROOT/'config/analysis_config_v32.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    model=json.loads((ROOT/'models/model_specification_v32.json').read_text()); model['analysis_version']=version
    (ROOT/'models/model_specification_v32.json').write_text(json.dumps(model,indent=2))
    bal=pd.read_csv(T/'32_stage_observation_balance_v32.csv')
    bal.groupby('phase')[['unweighted_smd_vs_target','weighted_smd_vs_target']].agg(lambda x:abs(x).max()).to_csv(T/'R1_balance_maxima.csv')
    run=json.loads((ROOT/'reports/23_v32_analysis_run_record.json').read_text()); run.update(analysis_version=version,peer_review_status='revised after external peer review; not accepted',release_status='local only')
    for key,relative in [('canonical_table_sha256','tables/31_canonical_primary_bootstrap_v32.csv'),('canonical_distribution_sha256','data/processed/canonical_bootstrap_distribution_v32.parquet'),('mnar_distribution_sha256','data/processed/mnar_chain_bootstrap_v32.parquet'),('mnar_interval_table_sha256','tables/47_mnar_chain_bootstrap_intervals_v32.csv')]:
        run[key]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    (ROOT/'reports/23_v32_analysis_run_record.json').write_text(json.dumps(run,indent=2))
    oe=ci.loc[ci.strategy.eq('IPW')&ci.metric.eq('oe_ratio')].iloc[0]
    slope=ci.loc[ci.strategy.eq('IPW')&ci.metric.eq('calibration_slope')].iloc[0]
    (ROOT/'reports/22_v32_canonical_and_mnar_results.md').write_text(f'''# Corrected R1 analysis results

    Local author-review revision dated 30 September 2026. Not a public release or submission-ready package.

    The source preprocessor was fitted in 33,394 eligible INSPIRE operations, and the classifier in 24,872 outcome-observed operations. The five-character obstetric exclusion and inclusive AKI threshold comparisons were corrected before full refitting.

    Primary MOVER 2022 O/E: {oe.estimate:.3f} (95% percentile interval {oe.ci_lower:.3f} to {oe.ci_upper:.3f}); calibration slope: {slope.estimate:.3f} ({slope.ci_lower:.3f} to {slope.ci_upper:.3f}). These are assumption-dependent measured-variable MAR estimates.

    All 1,000 full-chain bootstrap attempts succeeded; 18 stage-specific MNAR scenarios were refitted per replicate. Fixed-prediction coarsening comparisons use a separate 1,000-replicate paired bootstrap. Complete-case and IPW updating are compared within the same primary replicates.

    No clinical utility, prospective validation, harmonized-scope validation, or independent confirmation is established. MOVER 2022 remains post-exploration temporally held-out evaluation. The historical v3.2.5 DOI does not reproduce the corrected results.
    ''',encoding='utf-8')
    print(ci.to_string(index=False)); print(pd.DataFrame(contrasts).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    main()
