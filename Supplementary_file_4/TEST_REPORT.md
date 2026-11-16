# Validation Report

Author-review candidate. Validation performed **2026-11-15**.
No publication-readiness or clinical-deployment assessment is implied.

## Runtime

Python 3.12.14 (bundled interpreter). Final validation used the read-only frozen
analysis site-packages, with bytecode disabled and plot caches in writable local
validation output directories. No source script, prepared data or fitted-model
file was modified. Numerical packages: numpy 2.0.2; pandas 2.3.3; scipy 1.13.1;
scikit-learn 1.6.1; joblib 1.5.3; pyarrow 21.0.0; matplotlib 3.9.4;
Pillow 11.3.0. Initial bounded checks also used the same pinned package versions
in an isolated dependency directory; final checks used the frozen environment.

## Executed Checks

| Check | Result |
| --- | --- |
| File 4 synthetic tests | 12 passed; includes actual CLI invocations |
| File 5 synthetic tests | 3 passed |
| Package structural/provenance tests | 3 passed |
| Source-function integrity | 104 unchanged function/class AST hashes per package, 208 total |
| Python parsing | 43 Python files parsed without writing bytecode |
| Protected authoritative inputs | 28 script/config/prepared/model files retained identical SHA-256 |
| Upstream raw ETL provenance | INSPIRE, VitalDB and R1-correction source hashes checked unchanged |
| Prepared R2 point and sensitivity chain | Completed using actual authorized read-only inputs |
| Table 30 numeric comparison | 16 numeric columns; maximum absolute difference 2.0122792321330962e-16 |
| Table 34 numeric comparison | 10 numeric columns; maximum absolute difference 2.1316282072803006e-14 |
| S6 overall metric comparison | 36 metrics; maximum absolute difference 2.0122792321330962e-16 |
| File 4 supplied public aggregates | All 39 CSVs checked; 10 contracted schemas and 29 structural-only checks; 65 metric plots generated |
| File 5 supplied public aggregates | 3 CSVs checked; S6 PDF/PNG redrawn directly from CSV |

The 12 File 4 tests cover identity ambiguity/missingness, original-first selection
without replacement, original decimal/window boundary tests, distinct testing
occasions, CSV-to-MOVER cohort, ZIP-to-INSPIRE cohort, CSV-to-VitalDB cohort,
R1 inclusive thresholds, suppression without zero-filling, timing CLI,
aggregate plotting and one full-chain bootstrap replicate on a synthetic cohort.
Synthetic identifiers/counts do not represent real patients or retained cohort sizes.

The aggregate checks are not a disclosure audit. Files without a contracted schema
are explicitly structural-only, and suppressed values are not reconstructed.
The 65 plots are numeric validation views, not replacements for the manuscript's
selected figure set or claims of exact layout equivalence.

## Not Executed or Not Established

- **No real MOVER whole-cohort raw CSV reconstruction was run.** Both configured and
  resolved information/labs files were unavailable, the expected crosswalk was
  missing and the external source volume was not mounted. The chunked CSV reader
  accumulates retained strict-lab rows in memory; it is not a constant-memory ETL.
- Real INSPIRE ZIP and VitalDB cases/labs tables were not fully rescanned. Their raw
  routes are parameterized and synthetic-fixture tested, not verified real-data
  raw-to-authoritative-R2 equivalence.
- No real-data 1000-replicate full-chain bootstrap was run. The S6 calculation did
  use its original 1000 **conditional target-patient** resamples; this is not a
  full-chain bootstrap and does not establish full-chain uncertainty reproduction.
- MOVER's optional frozen coarsening-map branch has not been established against
  the authoritative full real cohort. No approximate identifier/date matching or
  replacement inclusion rule was introduced.
- No upload, submission, or clinical-deployment assessment was performed. Distribution tests are recorded below.

Frozen dependency warnings about deprecated plotting-parser APIs and capturing
groups in the unchanged cardiac regex were observed; the tests passed. These
warnings do not certify future dependency versions. The final aggregate redraw
used wrapped horizontal labels after resolving the initial layout warnings.


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
