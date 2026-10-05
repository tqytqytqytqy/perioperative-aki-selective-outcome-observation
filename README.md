# Selective postoperative creatinine testing and AKI model transport

Version v3.3.0 is the corrected reproducibility release for the multidatabase study. It contains research code, aggregate results, figures, model parameters, and a consolidated workbook. It is not a clinical model release or an accepted journal publication.

## Version and citation

- Repository: https://github.com/tqytqytqytqy/perioperative-aki-selective-outcome-observation
- Release: https://github.com/tqytqytqytqy/perioperative-aki-selective-outcome-observation/releases/tag/v3.3.0
- All-versions concept DOI: https://doi.org/10.5281/zenodo.21366088
- Version-specific DOI: https://doi.org/10.5281/zenodo.23157606 (published v3.3.0 archive).

The v3.3.0 tag and Zenodo archive are immutable snapshots of the scientific release. This main-branch citation update adds the DOI returned after automatic archiving; it does not change the released analysis or results.

This version supersedes v3.2.5 for the corrected findings. The earlier archive, https://doi.org/10.5281/zenodo.21666483, remains immutable provenance and does not reproduce the revised numerical results. See `CHANGELOG.md` for the scientific corrections.

## What changed

The corrected analysis addresses cohort construction and numerical comparisons at AKI threshold boundaries, recomputes the affected results, and distinguishes local recalibration from the incremental effect of inverse-probability weighting. Primary eligible cohort sizes are 33,394 in INSPIRE, 2,802 in MOVER 2021, and 2,587 in MOVER 2022.

An independent computational check against released laboratory records is summarized without redistributing patient records. Table S27 examines alternative choices when different baseline creatinine values share the latest eligible timestamp. The primary first-valid-source-row rule is reproducible, not clinically adjudicated. The minimum/maximum scenarios are post hoc point-estimate analyses of the retained cohorts; they are not performance bounds or statistical equivalence tests.

## Reproduction

Begin with the instructions under `docs/` and the portable configuration template under `config/`. Acquire the specified source datasets separately under their providers' current terms. Use the documented stages of `scripts/replay.py` to rebuild private cohorts and regenerate aggregate results.

Raw inputs, patient-level derived data, individual predictions, serialized estimators, and private runtime settings must remain outside the public archive. Historical comparisons against earlier local patient-level files require those separately retained inputs; the public aggregate summaries cannot reconstruct those files.

The computational results were generated in the corrected analysis and its subsequent sensitivity analysis. Release packaging checks inspect provenance, dependencies, syntax, tests, aggregate values, and excluded content. They do not constitute a fresh full-chain bootstrap run on release day. Small subgroup cells are suppressed in public-facing outputs as documented in the release checks; unsuppressed patient-level material is never redistributed.

## Interpretation

The four-variable model is an auditable methodological probe. MOVER 2022 is a post-exploration temporal evaluation, not an untouched independent confirmation cohort. Missing postoperative creatinine is not absence of AKI. Calibration close to one does not establish clinical deployability, patient benefit, or superiority of weighting over complete-case recalibration.

## Authors and rights

Qingyu Teng, Tingting Niu, Qian Chen, Min Tao, Ziyan Gu, Qi Li, Yingya Zhao, and Hui Zhang. Qingyu Teng, Tingting Niu, and Qian Chen contributed equally. Corresponding authors are Qi Li, Yingya Zhao, and Hui Zhang.

Code under `scripts/` is MIT-licensed. Original documentation, aggregate results, figures, model metadata, and workbook content are CC BY 4.0 except where otherwise noted. These licenses do not replace source-dataset access conditions. See `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `DATA_ACCESS_AND_REDISTRIBUTION.md`.
