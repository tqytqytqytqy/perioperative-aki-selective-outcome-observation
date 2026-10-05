# Public disclosure controls

The public scientific package excludes patient-level raw and derived data,
predictions, linkage identifiers, record-level audit evidence, serialized fitted
models, private configurations, manuscript documents and review correspondence.
Code can generate such artifacts only inside a separately initialized private
replay workspace. They are not part of the redistribution allowlist.

Positive aggregate subgroup counts below five are displayed as `<5`.
Corresponding percentages are upper bounds calculated from five divided by
the displayed denominator. A bound is rounded upwards, never to an exact
small numerator. Zero, model coefficients, spline knots, category codes,
thresholds and synthetic test values are not patient counts.

The current public workbook is a **disclosure-controlled derivative**, not a
byte-identical copy of the local current workbook. Its primary results and
Table S27 are unchanged. All original source files remain unchanged.
The companion machine-readable control plan lists only public replacement
values and cell coordinates, not the suppressed exact values.

Controls cover:

- The small missingness cell and its percentage, both in CSV and Table S5.
- A small positive surgical-flag overlap, plus complementary surgical-flag
  totals that could reconstruct it, in CSV, Table S23 and S23_numeric.
- High-precision categorical balance proportions that imply fewer than five
  patients. For affected cohort-variable groups, related category proportions,
  weighted summaries and standardized differences are suppressed together.
  Directly affected unweighted proportions display upper bounds. This avoids
  reconstructing small counts from complementary categories or diagnostics.
- The same controls in both presentation and numeric workbook sheets.

Whole-population correction facts, including 16 ties (12/3/1 by analytic
cohort), the one-event change and the one unobserved target exclusion, are
retained without case attributes. They are explicitly distinguished from
small clinical subgroup disclosures.

`public_disclosure.py` applies the reviewed CSV controls and generates a
workbook edit plan from a disposable, unsuppressed staging copy.
`public_workbook.mjs` applies that plan using the artifact-tool spreadsheet
engine to a separate workbook. Never point these utilities at original
scientific source outputs. Do not treat a private replay folder as a public
package or publish it wholesale.

Suppressions protect these released aggregate presentations; they are not a
formal anonymization guarantee or a substitute for provider terms and privacy
review. Current data-access and acquisition-time terms are documented by the
root package owner. No raw data are redistributed here irrespective of provider
license differences.
