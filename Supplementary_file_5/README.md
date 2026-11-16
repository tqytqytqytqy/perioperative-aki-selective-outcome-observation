# Supplementary File 5: Current R2 Grouped Calibration

**Author-review candidate. Validation performed 2026-11-15.** This package uses the current
`R2_IDQC_20261113` S6 calculation, not a historical public-release reproduction
claim. The packaged numerical core is self-contained; File 4 is not required.
Data, fitted joblib objects, individual predictions and replicate distributions
are deliberately absent. Load joblib only from a trusted authorized source.

## Environment and Tests

Use the authorized Python 3.12 interpreter (`PYTHON`) with the frozen dependencies
in `requirements.txt`. An existing matching site-packages directory can be supplied
read-only via `PYTHONPATH`. Set `PYTHONDONTWRITEBYTECODE=1`, `MPLBACKEND=Agg` and
`MPLCONFIGDIR` to a writable local directory outside the supplementary package.
Actual tested versions and scope are recorded in `TEST_REPORT.md`.

```sh
"$PYTHON" -B -m unittest discover -s tests -v
```

## Level 1: Aggregate-Only Checks and Direct Redraw

The authors supply publication-filtered CSVs, with consistent suppression:

```text
aggregates/S6_calibration_points.csv
aggregates/prediction_distribution.csv
aggregates/reproduction_metrics.csv
```

`contracts.json` fixes filenames, field names, strategy labels and output names.
Calibration points use shared source-risk group membership. Distribution inputs
contain exactly seven quantiles per strategy: 0, .05, .25, .5, .75, .95 and 1.
Count fields in the points table are optional and may be suppressed. No count
is inferred from a proportion, confidence interval or comparison with older data.
Plotted coordinates must be permitted numeric aggregates; suppressed coordinates
cause an explicit error rather than an invented graph.

```sh
"$PYTHON" -B aggregate_only.py --aggregates aggregates --output ../file5_aggregate_plot
```

This command checks all supplied aggregate CSVs, validates the three contracted
tables and writes `Figure_S6_calibration_curves.pdf`, its PNG preview and
`aggregate_checks.json`. The figure is drawn directly from numeric CSVs. No
original PDF is copied, converted or rasterized. No AUTHOR REVIEW label is added.
This verifies schema and plotting, not agreement with inaccessible patient data,
nor a disclosure audit. Extra CSVs are explicitly structural-only checks.

## Level 2: Authorized Prepared R2 Recalculation

Point `R2_ANALYSIS` to the authorized R2 analysis directory. Required local input
filenames are fixed in `contracts.json`: three prepared cohort files, the trusted
source model, model specification, current table-30 comparators and table-34
workload. Nothing from those files is bundled here. Field-name/type schemas are in
`PREPARED_SCHEMA.json`, without data values or private paths.

```sh
"$PYTHON" -B run_local.py --analysis-root "$R2_ANALYSIS" --output ../file5_local_calibration
```

The entry point reconstructs target observation/outcome nuisance models using the
current seed offsets and five-fold cross-fitting, reads the frozen R2 source model,
applies the archived complete-case and IPW recalibration parameters, and checks
all 36 current overall metrics against table 30 at tolerance `1e-10`. Eligibility
and observed denominators are checked against the supplied model specification,
not hardcoded historical values. Table-34 alert agreement is checked dynamically.

Grouped calibration retains the current rules: quantiles of unupdated source risk;
try 10 groups, fall back to 5; drop duplicate edges; require at least five observed
events **and** non-events in every group; share membership across strategies.
Grouped observed frequency uses 1/99-truncated IPW, while reproduced overall
metrics use the current hybrid AIPW/IPW implementation. These are distinct estimators.

The conditional bootstrap seed is **20261007** with **1000** target-patient
resamples. Predictions, once-reconstructed cross-fitted weights and group boundaries
are fixed. These are pointwise conditional percentile intervals, **not** full-chain
or simultaneous intervals. Original empty-weight-group failure behavior is retained:
no silent redraw or deletion. No replicate-level file is written.

Outputs are the same three contracted aggregate CSVs plus `local_calibration_qa.json`.
They are exact local calculations and require author disclosure review before
being copied into `aggregates/`. Use a new/empty output directory disjoint from the
analysis inputs and outside this code directory. Hash checks confirm read-only inputs.

`provenance.json` records unchanged numerical helper AST hashes.
`calibration_adaptation.json` documents the path/orchestration and export changes
to the current S6 source. Clinical rules, numeric calculation statements, seeds and
grouping rules are retained; private metadata, historical path bindings and the
replicate CSV export are removed.

## Verification Boundary

The current authorized prepared R2 calculation was executed and checked against
archived point metrics. Tests also cover grouping fallback, deterministic resampling,
empty-group failure and direct aggregate CSV-to-PDF/PNG rendering with synthetic
data. See `TEST_REPORT.md` for the fresh recorded result.

Raw extraction is not performed by this package. File 4 supplies INSPIRE, MOVER
and VitalDB raw-data entry points tested with synthetic fixtures; full real raw-to-R2
equivalence remains unverified. Full-chain
sampling uncertainty is outside S6's conditional intervals; File 4 has a separate
explicit full-chain entry point, not run with real 1000-replicate data on 2026-11-15.
No patient-specific advice, verified incident-AKI claim, clinical deployment
assessment, upload or publication-readiness claim is made.
