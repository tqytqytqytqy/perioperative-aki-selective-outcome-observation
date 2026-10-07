# Post hoc timing and follow-up coverage diagnostic

Date: 2026-10-07. Status: specified before computing new descriptive outputs.

Use the retained corrected v3.3.0 INSPIRE, MOVER 2021 and MOVER 2022 cohorts without reselection. Preserve original labels, predictions, calibration, thresholds and all primary inferences. VitalDB is excluded from this diagnostic because its supportive cohort and surgery-end timing are not harmonized with the principal cohorts.

Time zero is anesthesia completion. Valid creatinine rows use the original laboratory-name, code, unit and value rules. Follow-up ends at the original minimum of 168 h and available discharge/death (INSPIRE) or discharge (MOVER). Audit missing, nonpositive and competing stop times explicitly. Do not invent postdischarge follow-up.

Reconcile eligibility, observed status, 48 h/7 d raw-row counts, maxima and AKI labels against the frozen analysis set before reporting results. Count distinct timestamps for timing/frequency summaries, retaining raw-row counts only for reconciliation. At shared timestamps, coverage is one measurement occasion irrespective of duplicate rows.

Report first test time among observed patients; unique testing occasions and distinct elapsed 24 h bins; early (0,48] h and late (48,168] h testing; both-window coverage; last-test-to-observation-end gap. Report early test rates in all eligible patients and separately those with >=48 h opportunity; report late testing in all eligible patients and separately those with >48 h opportunity. These are descriptive coverage indicators, not an adequate-surveillance or gold-standard definition.

Report 24 h bin coverage with explicit all-eligible and at-risk denominators: any positive follow-up within a bin is opportunity; a separately reported full-bin opportunity denominator avoids equating a partial day with a full day. No bin can prove absence of AKI. Summarize actual stop-time groups <=48 h, >48-<168 h, >=168 h; this is post hoc descriptive stratification, not a baseline predictor or causal analysis.

Risk-stratified diagnostics use within-cohort quintiles of the existing frozen source-model predictions without local recalibration, labeled as descriptive model-defined strata. Source predictions are in-sample and are not external validation. No refit, hypothesis tests, subgroup causal claims or model performance comparisons are introduced. Main summaries use counts, percentages and medians/IQR; Wilson intervals describe binomial proportions conditional on the retained cohort and observed measurement process.

Outputs: Tables S28 (overall), S29 (risk), S30 (available follow-up strata), S31 (elapsed-day coverage); aggregate CSVs and combined workbook; code/definitions; private record-level derivation only in restricted internal folder. New analyses are not covered by the existing DOI until a separately authorized release.
