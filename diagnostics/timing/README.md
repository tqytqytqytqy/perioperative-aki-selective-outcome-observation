# Postoperative creatinine timing and follow-up coverage

## Scope
Post hoc raw-record diagnostic of postoperative creatinine testing times and follow-up windows, added 7 October 2026. This supplement extends, but does not replace, the corrected primary v3.3.0 analysis at https://doi.org/10.5281/zenodo.23157606. This diagnostic is included in v3.4.0; the preceding v3.3.0 archive is retained as the corrected primary-analysis provenance. Original prediction models, labels, main performance estimates, and threshold choices were frozen. No patient-level records or serialized model objects are distributed here.

## Reproduction
1. Obtain INSPIRE and MOVER from their original providers under their access and use conditions. Rebuild the corrected v3.3.0 analysis using the archived repository instructions. Do not use superseded v3.2 outputs. The rebuild creates the processed cohorts and local source-model object used below; the latter is not a public clinical model service.
2. Use a Python environment matching `requirements-timing.txt`. Set your local raw-data locations in the rebuilt analysis directory's `config/analysis_config_v32.json`. The v32 filenames are retained by v3.3.0 for compatibility and do not mean that the old labels should be used.
3. Run from this supplement directory:

```sh
python timing_diagnostic.py --test-only
python timing_diagnostic.py --analysis-root /your/rebuilt/analysis --output-root /your/private/timing_output
```

The script reads the INSPIRE archive, MOVER patient information and laboratory CSVs; the retained `data/processed` cohorts; `models/source_logistic_spline_ipw_v32.joblib`; and `models/model_specification_v32.json`. The source configuration supplies raw paths. It checks the source fingerprints and reconciles observation status, raw row counts, maxima and AKI labels before summaries. Source input files are never modified. The full raw extraction was run once for this supplement; the later metadata and disclosure correction re-summarized the exact same private timestamp cache, without refitting or re-extracting the laboratories. For that optional operation use `--summaries-only` with the same output root.

## Definitions and denominators
Time zero is anesthesia completion. INSPIRE uses released chart_time; MOVER uses Collection Datetime. These are not assumed to be equivalent specimen-collection semantics. Valid-laboratory rules match the frozen analysis. Distinct timestamps define occasions; duplicate rows count once for coverage but remain in the reconciliation checks. Windows are (0,48] and (48,168] hours, truncated by the original valid discharge/death rule. Day 1 is (0,24] hours, not a calendar day.

All-eligible rates describe the complete retained population. Opportunity rates describe the ORIGINAL ALGORITHMIC observation window: early requires >=48 hours; late requires >48 hours. Any-bin opportunity requires a window extending past its start; full-bin opportunity requires a window reaching its end, and its numerator is restricted accordingly. Early and late conditional rates use different opportunity criteria and are not an adequacy comparison. An unavailable discharge endpoint may default to the original 168-hour cap; that is not confirmed follow-up. Postdischarge outcomes remain unavailable.

The metadata-restricted sensitivity excludes unusable discharge stops. All 48 early/late/daily coverage-rate contrasts changed by less than 1 percentage point. Exact excluded denominators remain private to avoid complementary disclosure of small counts. This sensitivity does not demonstrate continuous surveillance. Missing, unparseable and nonpositive discharge times were combined, and the effective death/discharge truncation reason was not separately enumerated. This is an explicit deviation from the original descriptive audit plan, not a completed clinical endpoint adjudication.

Risk quintiles use frozen unupdated source predictions within each cohort. The included score minima, maxima and medians show that the groups are not common clinical risk categories. Source-cohort scores are descriptive and in-sample. Realized window duration is a post-prediction grouping variable, never a baseline predictor.

Wilson intervals describe proportions conditional on retained cohorts and recorded measurements, not uncertainty from the primary model chain. Medians use empirical 25th and 75th percentiles. No new hypothesis tests, causal contrasts or model refits were performed. One test, testing in both windows, or a short last-test-to-end interval cannot establish a true negative or sufficient surveillance. The last-test gap is not the longest gap between tests.

## Contents
- S28 overall timing; S29 risk strata; S30 algorithmic window strata; S31 elapsed-day coverage.
- `aggregate_facts.json`, `timing_run_summary.json`, and `confirmed_stop_sensitivity_summary.json`: public summaries.
- `timing_qa_checks.csv`: the 39 checks from full raw extraction. Later sensitivity checks are recorded separately and do not change that original run log.
- `input_fingerprints.csv`: source identifiers and checksums without local paths or record identifiers.
- `analysis_plan_before_computation.md`: original plan, read together with the deviations above.
- `timing_diagnostic.py` and requirements: portable raw-reconstruction and summary code.

## Confidentiality and use
Positive counts below five and linked percentages/intervals are suppressed. Early-window counts within risk quintiles are omitted to prevent reconstructing rare late-only patterns. The two longer follow-up groups are combined into >48 hours in every cohort, because the originally planned >=168-hour group could disclose a rare untested count by subtraction. This public grouping change does not change the underlying windows, labels or primary estimates. The script necessarily creates a restricted PRIVATE record-level cache for local reproducibility. NEVER upload or redistribute that directory. This archive contains only code, aggregate outputs, source fingerprints and documentation. Data-provider restrictions continue to apply. Code is provided under the primary repository's code licence; other materials follow its content licence. No clinical decision-support deployment or improved patient outcome is claimed.
