# Supplementary File 4: Portable R2 Analysis and Cohort Rules

**Author-review candidate. Validation performed 2026-11-15.**
Authoritative analysis: `R2_IDQC_20261113`. Historical script names describe
the implementation lineage, not a claim that an earlier public release reproduces R2.
No patient data, models, predictions, private metadata or bootstrap distributions
are included. The two supplementary directories are independently runnable.

## Environment

Use Python 3.12 and `requirements.txt`. Set `PYTHON` to the authorized interpreter.
For an existing matching frozen environment, its site-packages may be supplied via
`PYTHONPATH`; do not run its executable or modify it. Before any command:

```sh
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/../local_plot_cache"
```

The adapter sets single-thread calculation variables before importing numerical
libraries. Requirements pin the authoritative numerical versions; Pillow is only
used for raster inspection. See `TEST_REPORT.md` for the tested runtime.

## Level 1: Aggregate-Only Checks and Plotting

The authors supply **publication-filtered**, consistently suppressed CSVs:

```text
aggregates/
  analysis_tables/   current main-chain safe aggregate CSVs
  timing/           S28-S31 publication-filtered aggregate CSVs
```

Exact filenames, required columns and outputs are in `contracts.json`. All
supplied CSVs are checked for individual/replicate identifiers and infinite
numeric values. Known filenames also receive schema/type/range checks. Unknown
filenames are explicitly marked `structural_only`, not falsely certified.
Supported performance, workload, MNAR and timing metrics are redrawn directly
from CSV. There is no PDF conversion and no patient access/model fitting.

```sh
"$PYTHON" -B aggregate_only.py --aggregates aggregates --output ../file4_aggregate_plots
```

This command accepts a partial selection and reports every absent contracted
file. `aggregate_checks.json` lists each input fingerprint, check scope and plot.
`<5`, declared suppression strings and missing cells are not replaced with zero
or back-calculated. Suppressed plot coordinates are omitted. These checks do not
constitute a disclosure audit or verify suppressed denominators.

## Level 2: Authorized Prepared R2 Recalculation

`R2_ANALYSIS` must identify an authorized local R2 analysis directory containing
the three prepared parquet files listed in `contracts.json`. `PREPARED_SCHEMA.json`
records **field names/types only**, never rows or identifiers. Keep the full schema.
The packaged seed is 20260714 and the cross-fit rule is five folds. The source,
update and target samples remain INSPIRE, MOVER 2021 and MOVER 2022, respectively.

```sh
"$PYTHON" -B run_local.py --analysis-root "$R2_ANALYSIS" --output ../file4_local_point
```

This is the actual current source-update-target point/sensitivity calculation,
not an aggregate-only surrogate. Numerical functions are extracted verbatim;
`provenance.json` records source SHA-256 and per-function AST hashes. Historical
report generators, author metadata and stale numerical-result assertions are
excluded. Current cohort rules and consistency checks remain. Models, predictions
and all replicate-level data stay in memory; only aggregate results are written.
Local aggregates must be reviewed and suppressed before being placed in `aggregates/`.

Optional full-chain bootstrap uses the unchanged original resampling, retry,
cross-fitting and seed schedule. It does not reuse historical distributions:

```sh
"$PYTHON" -B run_local.py --analysis-root "$R2_ANALYSIS" --output ../file4_local_full \
  --full-chain-bootstrap --replicates 1000 --allow-long-run
```

This is deliberately expensive and was **not run on real data in the 2026-11-15 validation**.
One full-chain replicate was exercised on synthetic data. Per-replicate results
remain in memory, without resumable checkpoint exports. `--replicates 1` is a
smoke test, not a valid uncertainty analysis. No omission/redraw of failed
replicates is added; the original bounded retry logic is retained.

Every input/output directory must be disjoint. Outputs require a new or empty
directory outside this code directory. Input hashes are checked after calculations.
Unknown postoperative outcomes remain unknown, not negative. The endpoint is a
recorded creatinine-threshold outcome, not clinically verified incident AKI.

## Raw MOVER Reconstruction

`rebuild_mover.py` is a real CSV-to-cohort entry point with an end-to-end synthetic
CSV test. It does not depend on private historical caches. Obtain the original
patient-information CSV, patient-labs CSV and crosswalk under their source access
terms. Supply their paths using the following flags:

```sh
"$PYTHON" -B rebuild_mover.py --patient-information "$MOVER_INFORMATION" \
  --patient-labs "$MOVER_LABS" --crosswalk "$MOVER_CROSSWALK" \
  --output ../mover_raw_rebuild
```

Optional `--coarsening-map "$MOVER_COARSENING_MAP"` applies the frozen table-51
mapping and derives coarsened/definite/possible outcomes. It does not refit a
mapping on the new cohort. Without this map the output supports operational
modeling but is not a complete replacement for every R2 sensitivity column.

Rules: resolve information duplicates only when all frozen clinical fields agree;
select the original first clinically eligible operation across all years **before**
identity QC and baseline checks; do not substitute later operations. Require a
unique complete event crosswalk, exact information-MRN agreement, a bidirectionally
unique global alias and matching event/global alias. Labs require nonmissing,
nonempty, exact information-MRN agreement. No namespace swap/fuzzy repair is used.
Strict labs, valid ranges, decimal tolerance, last-preoperative baseline,
original-row-order ties and discharge-truncated windows follow the extracted
audit functions. No real identifier literal is embedded in the code/tests.

Timing/coverage from this locally derived cohort can be calculated using the
explicit trusted **frozen diagnostic source model**:

```sh
"$PYTHON" -B run_timing.py --cohort ../mover_raw_rebuild/mover_rebuilt_v32.parquet \
  --source-model "$TIMING_FROZEN_SOURCE_MODEL" --output ../mover_local_timing
```

S29 in the current timing source used an unchanged earlier frozen model for
inference, not a new fit. Do not silently substitute the R2 refitted model when
matching that diagnostic. Outputs are exact local aggregates, not public copies.

## Tested Boundary and Remaining Gaps

- Prepared R2 point analysis and S6 were tested with authorized read-only inputs.
- Synthetic tests exercise identity ambiguity, no replacement, threshold/windows,
  raw CSV reconstruction, unknown outcomes, suppression, and one full-chain replicate.
- Real raw CSV -> authoritative R2 whole-cohort equivalence was **not tested**;
  the full raw laboratory file was not rescanned on 2026-11-15. Hash/row/field reconciliation
  against the authoritative R2 prepared cohort is still required after a full raw run.
- INSPIRE raw ZIP extraction is supplied via the unchanged `build_inspire` dependency
  closure and current released-level outcome representations. A synthetic ZIP test
  exercises the actual archive-to-prepared command:

```sh
"$PYTHON" -B rebuild_inspire.py --archive "$INSPIRE_ARCHIVE" --output ../inspire_raw_rebuild
```

- Real raw INSPIRE -> authoritative R2 equivalence was not tested on 2026-11-15.
  The input archive must contain exactly one `operations.csv.gz` and one `labs.csv.gz`.
  Operation fields are `subject_id, op_id, age, asa, sex, anstart_time, anend_time,
  discharge_time, inhosp_death_time, cpbon_time, department, icd10_pcs, antype`;
  lab fields are `subject_id, chart_time, item_name, value`.
- VitalDB now uses the **actual upstream** `prepare_vitaldb_v32.py` computation,
  inherited by the R2 prepared input, not `etl_common_v32.load_vitaldb`'s legacy cache
  path. Supply local official cases/labs CSV tables:

```sh
"$PYTHON" -B rebuild_vitaldb.py --cases "$VITALDB_CASES" --labs "$VITALDB_LABS" \
  --output ../vitaldb_raw_rebuild
```

- Required case fields: `caseid, opstart, opend, preop_cr, age, sex, asa, ane_type`;
  lab fields: `caseid, dt, name, result`. The original baseline fallback, supportive
  cohort semantics, observation window and numeric-value handling are retained.
  In particular this route does not apply MOVER's strict-lab range/filter rules to
  VitalDB. Synthetic raw CSV tests exercise this command; real full-table equivalence
  against R2 was not tested on 2026-11-15. See `upstream_etl_adaptation.json` for source hashes.
- The R1 independent Decimal boundary-check function is retained, without running
  the old correction script's top-level writes or historical denominator assertions.
- Thus the package is **not a verified all-raw three-database end-to-end reproduction**.
  Prepared R2 is the validated complete-chain entry point. Historical INSPIRE timing
  aggregate reuse gates are also not an end-to-end raw timing route in this package.
- Real MOVER raw equivalence was not attempted: configured/resolved information and
  labs paths were unavailable, the expected crosswalk was missing, and the external
  data volume was not mounted. The private validation report records availability
  by input role only, without identifiers. No matching by approximate numbers/dates
  or new inclusion criteria has been introduced.
- Fine timing source records cannot be recreated from an aggregate CSV or from the
  reduced prepared modeling columns alone. The raw route above supplies MOVER timing;
  INSPIRE timing is aggregate-only in this package.
- No journal submission, clinical deployment or
  publication-readiness assessment has been performed.

```sh
"$PYTHON" -B -m unittest discover -s tests -v
```
