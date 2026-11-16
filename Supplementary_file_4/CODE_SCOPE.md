# Code Scope and Provenance

Author-review candidate. Validation performed 2026-11-15.

`provenance.json` identifies the authoritative source file SHA-256, original
line span and location-independent AST hash of each extracted function/class.
There are 104 unchanged function/class records in each supplementary directory.
Script suffixes retain implementation lineage; the analysis is R2_IDQC_20261113.
No historical public-release result is asserted to reproduce R2.

| Source family | Retained scope |
| --- | --- |
| run_v32_analysis | Current point/sensitivity chain, original bootstrap calculation/retry helpers and interval summaries |
| run_v31_analysis, run_postreview_analysis | Transitive numerical model, preprocessing, recalibration and metric dependencies |
| etl_common_v32 | Current INSPIRE extraction and outcome-finalization dependency closure; no cache-only VitalDB route |
| etl_outcome_representations_v32, thresholds_r1 | Release-level intervals, frozen coarsening, exact/interval labels and inclusive thresholds |
| raw_linkage_audit, derived_cohort_comparison, audit_mover | Exact identity rules, original-first/no-substitution helpers and decimal/window derivations |
| build_diagnostics, timing_helpers | Latest coverage/selection summaries and their time-summary dependencies |
| supplementary_common | Current S6 grouping, conditional bootstrap and metric comparison helpers |
| prepare_r1_corrected | Independent Decimal threshold-check function only; no top-level historical file writes |
| prepare_vitaldb_v32 | Actual upstream raw cases/labs calculation statements, extracted from main into a callable function |

The VitalDB and S6 source mains mix numerical calculations with private path
bindings and export/report orchestration. Their retained numerical statements are
recorded separately in `adapted_statement_provenance.json`, including source and
output AST hashes. `upstream_etl_adaptation.json` records the upstream INSPIRE,
VitalDB and R1-correction file hashes. `calibration_adaptation.json` in File 5
records the S6 IO/export changes. New CLI/path adapters are not represented as
unchanged original functions.

Manual orchestration: `portable.py`, `run_local.py`, `rebuild_mover.py`,
`rebuild_inspire.py`, `rebuild_vitaldb.py`, `run_timing.py`, `aggregate_only.py`
and `aggregate_support.py`. These parameterize input/output locations, forbid
overlapping/nonempty output destinations, suppress patient/replicate exports
where applicable and expose the existing computation. Tests exercise the interfaces;
real raw-to-authoritative-prepared equivalence remains explicitly unverified.

Removed material: source directory bindings, author metadata, legacy entry points,
historical reporting/result assertions and private snapshots. `config_adaptation.json`
lists unused metadata removals. No retained statistical setting, seed, threshold,
identity rule or numeric calculation has been intentionally changed. The removal
of unused readiness/event-node metadata does not implement a clinical readiness rule.

The independently runnable File 5 includes duplicated numerical helper modules
to avoid requiring a sibling code directory. Presence of a helper does not imply
that every raw/diagnostic route is invoked by its S6 entry point.
