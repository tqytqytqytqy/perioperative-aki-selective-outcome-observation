# Selective outcome observation study v3.2

Public reproducibility package containing aggregate results, analysis code, and model metadata. It is not a clinical model release and is not approved for patient care, triage, treatment selection, or deployment.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21366751.svg)](https://doi.org/10.5281/zenodo.21366751)

## Status

- Scientific analysis: complete.
- Submission: NO-GO pending author, ethics, disclosure, and independent-review gates.
- Repository: https://github.com/tqytqytqytqy/perioperative-aki-selective-outcome-observation
- Current archived v3.2.1 release: https://doi.org/10.5281/zenodo.21366751
- Release-family concept DOI: https://doi.org/10.5281/zenodo.21366088
- Previous immutable v3.2.0 archive: https://doi.org/10.5281/zenodo.21366089
- Study design: post-exploration methodological audit, not an independent confirmation study.

## Reuse status

No open-source or content-reuse licence is granted in this version. The public repository permits inspection and reproducibility review; all other rights are reserved unless a file states otherwise.

## Reproduce the analysis

1. Create Python 3.9.6 environment and install `config/requirements-analysis-v32.txt`.
2. Obtain INSPIRE 1.4.2, MOVER, and VitalDB from their official providers under current access terms.
3. Run `scripts/configure_runtime_v32.py` with local paths. The generated `runtime_config_v32.local.json` is private and ignored.
4. From a clean clone with empty `data/processed`, run `scripts/run_analysis_v32.sh`.

The full 1,000-replicate chain refits observation models, auxiliary outcome models, the source preprocessor and classifier, local recalibration, target metrics, and 18 stage-specific MNAR cells. Aggregate CSVs and the workbook are included for inspection. Patient-level raw and derived data, serialized model objects, and private runtime configuration are excluded.

## Claim boundary

The four-variable model is an auditable methodological probe. Clinical utility, fairness, prospective workflow effects, and patient benefit were not evaluated. MOVER 2022 is a post-exploration temporally held-out evaluation.
