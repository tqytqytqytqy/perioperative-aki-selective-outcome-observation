# Source map

The machine-readable `SOURCE_MAP.json` identifies every copied or mechanically
ported scientific source using relative source-set handles, source basenames,
SHA-256 hashes and the staged artifact path. It contains no private absolute
paths or journal-specific submission filenames.

| Handle | Scope |
| --- | --- |
| corrected_analysis_20260930 | Retained corrected computational scripts, aggregate tables, model specification and verified environment |
| raw_audit_20261005 | Independent raw audit and retained-cohort tie-sensitivity scripts |
| current_revision_2_1_20261005 | Current figures, aggregate audit outputs and Table-S27-inclusive workbook |
| staging_authored_v330 | Version-3.3.0 replay wrapper, synthetic tests, public checks, disclosure controls and documentation |

Core analysis and raw ETL dependencies retain source bytes where possible.
The historical v3.2.5 source ZIP was used only for code-hash comparison.
No historical output table, model JSON, QA approval or license claim from that
snapshot was promoted to a current result.

Mechanical code changes are enumerated per file. They remove local path
constants, guard execution-only bodies, confine outputs to a private workspace,
remove manuscript-figure copying, remove unused administrative metadata, or
suppress record-valued failure logging. Statistical equations, seeds,
resampling settings and the original tie-scenario calculations are unchanged.
The supplemental coarsening check now reads the same retained creatinine
columns from the corrected derivative, removing an unnecessary historical-label
input for that check.

Public disclosure transformations are documented separately. Source hashes
remain those of the original source files; staged hashes reflect the
disclosure-controlled CSV/workbook derivatives. This ledger is a scientific
provenance map. Use the repository-level release manifest for archive-wide
integrity verification.

## Dependency chain

- `run_v32_analysis.py` imports shared model, metric, preprocessing, cohort,
  balance and metadata functions from `run_v31_analysis.py` and
  `run_postreview_analysis.py`.
- Corrected dataset preparation imports `etl_common_v32.py`,
  `etl_outcome_representations_v32.py` and `thresholds_r1.py`.
- Supplemental correction code imports the same corrected analysis modules.
- The finalizer uses `build_v32_figures.py` as a helper, while supplying the
  corrected three-strategy Figure 2 itself.
- The raw audit uses an independent CSV/ZIP and Decimal implementation.
- `tie_sensitivity.py` consumes private raw-audit evidence and calls the
  preserved corrected point-estimate functions.

The dependency-only presence of legacy-named modules is intentional.
No `v31` result files are shipped and no supported entrypoint invokes the
historical main functions. See `DEPENDENCY_AUDIT.json` for the actual import
inventory and the explicit supported entrypoint list.
