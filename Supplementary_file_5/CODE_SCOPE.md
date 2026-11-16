# Code Scope and Provenance

Author-review candidate. Validation performed 2026-11-15.

This directory is independently runnable. Numerical helpers duplicated from File 4
do not require importing a sibling directory. The entry points are
`aggregate_only.py` (public aggregate validation/redraw) and `run_local.py`
(authorized current R2 grouped calibration). No raw ETL is invoked by these entries.

`provenance.json` records 104 unchanged helper function/class AST hashes, original
source spans and source SHA-256 values. `adapted_statement_provenance.json` records
the numerical statements extracted from the actual current S6 main and the upstream
VitalDB raw ETL main. Source path bindings, private snapshot/export orchestration
and the S6 replicate-level CSV export are excluded, without changing the retained
numerical statements. New wrappers are not claimed to have unchanged source ASTs.

`calibration_adaptation.json` records the current S6 source and adaptation scope.
`upstream_etl_adaptation.json` identifies the original INSPIRE/VitalDB/R1-correction
source files. `config_adaptation.json` lists unused metadata removals; actual retained
statistical settings are unchanged. Source-level provenance is distinct from
empirical real raw-to-prepared equivalence, which was not established on 2026-11-15.

The grouped estimator, overall hybrid estimator, group fallback rule and original
conditional resampling/empty-group behavior remain separate and unchanged.
See `TEST_REPORT.md` for empirical checks and their limits.
