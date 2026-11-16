# Validation Report

Author-review candidate. Validation performed **2026-11-15**.

Runtime: Python 3.12.14 (bundled interpreter), read-only frozen numerical
dependencies: numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, scikit-learn 1.6.1,
joblib 1.5.3, pyarrow 21.0.0, matplotlib 3.9.4 and Pillow 11.3.0.
Bytecode was disabled; plot caches and generated outputs were kept in local
validation directories, outside the supplementary code directories.

## Executed

- Three File 5 synthetic tests passed: ten-to-five grouping fallback; deterministic
  conditional resampling with explicit empty-group failure; aggregate-only CSV
  validation and direct PDF/PNG plotting, including a nonblank-raster check.
- Twelve File 4 tests and three package structural/provenance tests also passed.
- 104 unchanged helper function/class AST hashes per package were verified;
  adapted S6 calculation statements have a separate statement-level source record.
- Actual authorized prepared R2 calibration was executed. All 36 overall metrics
  matched the archived current table-30 metrics within 1e-10; maximum absolute
  difference was **2.0122792321330962e-16**. Current recalibration parameters and
  table-34 alert agreement were checked dynamically, without hardcoded real counts.
- The original 1000 conditional target-patient resamples used seed 20261007.
  No individual predictions or replicate-level distribution was exported.
- The three publication-filtered S6 aggregate CSVs were checked and used to draw
  S6 PDF/PNG directly, without an original PDF. The rendered S6 was inspected.
- Input hashes before and after the calculation were identical; 28 protected
  authoritative script/config/prepared/model file hashes were also unchanged.

## Boundaries

The metric comparison validates prepared-input point reproduction. It is not
real raw-to-prepared equivalence, full-chain uncertainty replication, simultaneous
calibration inference, a disclosure audit or clinical validation. File 4's three
raw-data routes were tested with synthetic inputs only. Real MOVER raw files and
crosswalk were unavailable and its external source volume was not mounted.
No real-data 1000-replicate full-chain bootstrap, submission, or clinical-deployment assessment was performed. Distribution tests are recorded below.


## Packaged Distribution Verification

On 2026-11-15 the two archives were extracted into a new local validation
directory. The File 4 and File 5 suites passed 12 and 3 tests, respectively.
The aggregate commands checked 39 File 4 CSVs and 3 File 5 CSVs and directly
redrew the supported summaries and Figure S6. All 208 extracted function AST
hashes matched their provenance records, and protected analytical inputs were
unchanged. These are software and numerical consistency checks, not a provider
disclosure approval or a test of clinical benefit. See
`DISTRIBUTION_VALIDATION.json` for the recorded scope. The unverified raw-data
and real full-chain-rerun boundaries above are unchanged.
