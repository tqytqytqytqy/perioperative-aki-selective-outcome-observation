"""Current S6 calculation body; path adapter and replicate export separated."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit, logit
import supplementary_common as c
SEED = 20261007
REPLICATES = 1000

def calculate(a, cfg, cohorts, root):
    counts = {k: {"eligible": len(f), "observed": int(a.observed_indicator(f).sum()),
        "observed_events": int(f.loc[a.observed_indicator(f), "outcome_operational"].sum())}
        for k, f in cohorts.items() if k != "vital_observed"}
    t = cohorts['target']
    b = a.observation_bundle(t, 'MOVER 2022', cfg, cfg['random_seed'] + 3000, include_basic=False, fit_outcome=True, fit_final=False)
    obs = b['observed']
    y = b['y'][obs]
    w = b['weights']
    assert len(t) == counts['target']['eligible']
    assert int(obs.sum()) == counts['target']['observed']
    assert int(y.sum()) == counts['target']['observed_events']
    assert not t.patient_key.duplicated().any()
    model = joblib.load(root / 'models/source_logistic_spline_ipw_v32.joblib')
    p0 = model.predict_proba(t[a.FEATURES])[:, 1]
    comparators = pd.read_csv(root / 'tables/30_update_selection_target_performance_v32.csv')
    spec = json.loads((root / 'models/model_specification_v32.json').read_text())
    params = spec['recalibration']
    match = comparators.loc[comparators['source_model_estimand'].eq('1/99 truncated-IPW source model') & comparators['update_estimator'].eq('complete-case recalibration')]
    assert len(match) == 1, 'Complete-case comparator must be unambiguous'
    cc = match.iloc[0]
    preds = {'Unupdated': p0, 'Complete-case recalibration': expit(float(cc['update_alpha']) + float(cc['update_beta']) * logit(np.clip(p0, 1e-06, 1 - 1e-06))), 'IPW recalibration': expit(params['alpha'] + params['beta'] * logit(np.clip(p0, 1e-06, 1 - 1e-06)))}
    metric_rows = []
    for strategy, p in preds.items():
        m = a.aipw_hybrid_metrics(t, p, b)
        metric_rows.append({'strategy': strategy, **m})
    metrics = pd.DataFrame(metric_rows)
    primary_rows = comparators.loc[comparators.source_model_estimand.eq('1/99 truncated-IPW source model')].copy()
    strategy_names = {'unupdated source-only comparator': 'Unupdated', 'complete-case recalibration': 'Complete-case recalibration', '1/99 truncated-IPW recalibration': 'IPW recalibration'}
    primary_rows['strategy'] = primary_rows.update_estimator.map(strategy_names)
    check_metrics = ['events', 'event_rate', 'predicted_mean', 'oe_ratio', 'calibration_intercept', 'calibration_joint_intercept', 'calibration_slope', 'auroc', 'auprc', 'brier', 'brier_skill_score', 'grouped_absolute_calibration_error_10q']
    max_reproduction_difference = c.reproduction_difference(metrics, primary_rows, check_metrics)
    primary = primary_rows.loc[primary_rows.strategy.eq('IPW recalibration')].iloc[0]
    assert np.isclose(primary.update_alpha, params['alpha'], atol=1e-10, rtol=0)
    assert np.isclose(primary.update_beta, params['beta'], atol=1e-10, rtol=0)
    workload = pd.read_csv(root / 'tables/34_update_strategy_workload_v32.csv')
    workload = workload.loc[workload.strategy.eq('1/99 truncated-IPW source model + 1/99 truncated-IPW recalibration') & np.isclose(workload.threshold, 0.1)]
    assert len(workload) == 1
    assert int((preds['IPW recalibration'] >= 0.1).sum()) == int(workload.iloc[0].alerts)
    requested_bins = 10
    codes, edges = c.safe_groups(p0, obs, y)
    groups = sorted(np.unique(codes))
    ng = len(groups)
    n = len(t)
    wa = np.zeros(n)
    wa[obs] = w
    ya = np.zeros(n)
    ya[obs] = y
    point_y, ci, boot = c.conditional_bootstrap(codes, wa, ya, SEED, REPLICATES)
    rows = []
    for strategy, p in preds.items():
        for j, g in enumerate(groups):
            allg = codes == g
            obsg = allg & obs
            rows.append({'strategy': strategy, 'risk_group': int(g + 1), 'eligible_n': int(allg.sum()), 'observed_n': int(obsg.sum()), 'observed_events': int(ya[obsg].sum()), 'effective_observed_n': float(wa[obsg].sum() ** 2 / np.sum(wa[obsg] ** 2)), 'source_lower_boundary': float(edges[j]), 'source_upper_boundary': float(edges[j + 1]), 'weighted_mean_prediction': float(np.average(p[obsg], weights=wa[obsg])), 'all_eligible_mean_prediction': float(p[allg].mean()), 'ipw_event_frequency': float(point_y[j]), 'conditional_ci_lower': float(ci[0, j]), 'conditional_ci_upper': float(ci[1, j])})
    points = pd.DataFrame(rows)
    assert points.observed_events.min() >= 5
    assert (points.observed_n - points.observed_events).min() >= 5
    for _, d in points.groupby('strategy'):
        assert d.eligible_n.sum() == n and d.observed_n.sum() == int(obs.sum())
        assert d.observed_events.sum() == int(y.sum())
    distribution = []
    for strategy, p in preds.items():
        qs = np.quantile(p, [0, .05, .25, .5, .75, .95, 1])
        distribution.extend({"strategy": strategy, "quantile": q, "prediction": float(v)}
            for q, v in zip([0, .05, .25, .5, .75, .95, 1], qs))
    return points, metrics, pd.DataFrame(distribution), {
        "maximum_absolute_reproduction_difference": max_reproduction_difference,
        "pointwise_bootstrap_replicates": REPLICATES, "seed": SEED,
        "interval_scope": "Conditional target-patient percentile intervals, fixed predictions, reconstructed cross-fitted weights and fixed groups; not full-chain or simultaneous.",
        "group_rule": "Try 10, fall back to 5; duplicates dropped; >=5 observed events AND non-events in every bin; shared membership.",
        "grouped_estimator": "1/99 truncated IPW; overall metrics are hybrid AIPW/IPW.",
        "no_patient_or_replicate_exports": True}
