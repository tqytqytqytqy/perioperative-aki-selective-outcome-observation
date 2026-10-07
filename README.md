# Outcome ascertainment and laboratory representation in perioperative AKI model transport

Version v3.4.0 combines the corrected multidatabase analysis with a new postoperative creatinine timing and follow-up coverage diagnostic. The package contains research code, aggregate tables, figures, model parameters and one consolidated workbook.

## Citation and provenance

- Repository: https://github.com/tqytqytqytqy/perioperative-aki-selective-outcome-observation
- Release: https://github.com/tqytqytqytqy/perioperative-aki-selective-outcome-observation/releases/tag/v3.4.0
- All-versions concept DOI: https://doi.org/10.5281/zenodo.21366088
- The version-specific DOI for v3.4.0 is assigned by Zenodo after the GitHub release and will be added to this main-branch citation record.
- Corrected primary-analysis predecessor: https://doi.org/10.5281/zenodo.23157606 (v3.3.0).

The new diagnostic is first included in v3.4.0. Previous immutable versions retain their historical contents. The older v3.2 releases do not reproduce the corrected primary findings.

## What this release adds

The new raw-record diagnostic separates whether a creatinine result is available from when measurements occurred and how they cover the recorded observation window. It retains INSPIRE 33,394 operations, MOVER 2021 2,802, and MOVER 2022 2,587. VitalDB remains supportive and is not included in the new time-origin comparison.

In MOVER 2022, any postoperative testing was recorded in 2,033/2,587 (78.6%), while both early and late windows were covered in 1,290/2,587 (49.9%). Among the 2,033 patients with records, 405 (19.9%) had only one distinct testing time. The algorithmic follow-up window ended before 168 hours in 1,898/2,587 (73.4%). Early discharge and differences in timestamp semantics constrain interpretation; these statistics do not estimate the number of missed AKI cases.

Tables S28-S31 describe overall timing, within-cohort source-score quintiles, follow-up-window groups and elapsed-day coverage. Full-population, partial-window and complete-window opportunity denominators are reported separately. Small counts and complementary reconstruction paths are controlled. The 48 metadata-restricted coverage contrasts all changed by less than one percentage point.

The corrected primary model, coefficients, AKI labels and existing inferential results are unchanged. The current presentation figures also show stage-specific sensitivity of predictions and workload, and label changes under fixed-patient, fixed-prediction numerical coarsening.

## Reproduction

Read `docs/REPLAY.md` for the corrected primary replay. Acquire the source data separately through the original providers. Read `diagnostics/timing/README.md` for the new diagnostic and its assumptions, source fingerprints, definitions, plan deviations and portable command.

The full raw extraction for the diagnostic was completed before this release, with 39 recorded reconciliation checks. Later metadata sensitivity and disclosure adjustments re-summarized the same private cache. Release checks verify retained outputs, code and package contents; they do not constitute another raw-data extraction or a new full-chain bootstrap.

Raw and patient-level derivatives, individual predictions, serialized models and local runtime settings remain outside the public archive. The code creates private local derivatives when users rebuild the analysis; those derivatives must not be uploaded.

## Interpretation

The four-variable model is an auditable methodological probe. MOVER 2022 remains a post-exploration temporal evaluation. Missing postoperative creatinine does not establish absence of AKI, and sparse observed measurements may leave episodes undetected. The diagnostic describes recorded testing and follow-up; it does not establish adequate surveillance, clinical deployability or patient benefit.

## Authors and rights

Qingyu Teng, Tingting Niu, Qian Chen, Min Tao, Ziyan Gu, Qi Li, Yingya Zhao, and Hui Zhang. Qingyu Teng, Tingting Niu, and Qian Chen contributed equally. Corresponding authors are Qi Li, Yingya Zhao, and Hui Zhang.

Code is MIT-licensed. Original documentation, aggregate results, figures, model metadata and workbook content are CC BY 4.0 except where otherwise noted. These licenses do not replace source-dataset conditions. See `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `DATA_ACCESS_AND_REDISTRIBUTION.md`.
