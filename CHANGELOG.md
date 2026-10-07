# Version v3.4.0 2026-10-07

## New diagnostic and presentation

- Added post hoc raw-record testing timing and follow-up coverage in the retained corrected INSPIRE/MOVER cohorts, with Tables S28-S31 and a consolidated workbook.
- Distinguished any testing from early/late and elapsed-day coverage, with explicit algorithmic observation-opportunity denominators and timestamp semantics.
- Added metadata-restricted coverage sensitivity and documented departures from the pre-computation descriptive plan. Missing/unusable discharge stops were combined; effective competing-stop causes were not separately adjudicated.
- Added current four-figure presentation and portable aggregate-figure code, using retained full-chain and paired intervals where applicable.
- Controlled small counts and linked fields; risk-quintile early counts are withheld and longer follow-up groups are combined to prevent complementary disclosure.
- Preserved all corrected primary model coefficients, labels, performance estimates and statistical intervals from v3.3.0. No model refit or new bootstrap was performed for publication.

The diagnostic clarifies recorded measurement coverage. It does not identify true outcomes in untested patients, count missed AKI episodes, or establish clinical surveillance adequacy.

# Version v3.3.0 2026-10-05

This scientific correction release supersedes v3.2.5 for current results. Prior version identifiers and archived files are retained as provenance; they are not evidence that the corrected results were available in July 2026.

## Corrections

- Corrected cohort handling and floating-point boundary comparisons for creatinine-defined AKI. The primary cohorts comprise INSPIRE 33,394 operations, MOVER 2021 2,802 operations, and MOVER 2022 2,587 operations. Observed AKI event counts are 1,680, 274, and 259, respectively; VitalDB supportive observed events are 271.
- Recomputed the affected scientific results, model parameters, full-chain uncertainty summaries, and figures in the corrected analysis. Target IPW-recalibrated O/E is approximately 1.012 and AUROC 0.674; these differ from the earlier v3.2.5 archive.
- Added paired comparisons that distinguish the benefit of local recalibration from the incremental contribution of weighting. Equal AUROC does not establish that IPW is superior to complete-case recalibration.
- Added an aggregate independent raw-record verification summary. Case-level evidence and source patient records are not redistributed. This is computational verification against released records, not clinical adjudication.
- Documented the deterministic first-valid-source-row rule when different baseline creatinine values share the latest eligible timestamp. Alternative minimum/maximum choices concern 16 retained operations (12 INSPIRE, 3 MOVER 2021, 1 MOVER 2022).
- Added Table S27: the maximum-value scenario changes one source AKI label and separately excludes one outcome-unobserved target operation after reapplying baseline eligibility. Target n is 2,586 in that scenario; observed target n=2,033 and events=259 remain unchanged. The sensitivity scenarios refit point estimates but do not add bootstrap intervals or establish statistical equivalence.

## Interpretation and provenance

The study remains a retrospective, post-exploration methodological analysis with selective outcome observation. It neither identifies unobserved AKI without assumptions nor validates clinical deployment. The revised narrative and release remain journal-neutral. The October release packages corrected analyses already completed locally; release-day packaging checks are not described as a new end-to-end statistical rerun.
