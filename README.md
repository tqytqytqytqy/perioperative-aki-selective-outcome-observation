# Selective outcome observation study v3.2.2

Public reproducibility package for aggregate results, analysis code, and model metadata. It is not a clinical model release and is not approved for patient care, triage, treatment selection, or deployment.

This non-peer-reviewed reproducibility package contains no manuscript file and is not itself a journal submission or a clinical model release.

## Status

- Scientific analysis: complete.
- Peer review: not peer reviewed.
- Clinical deployment: not evaluated; this study does not support clinical use.
- Repository: https://github.com/tqytqytqytqy/perioperative-aki-selective-outcome-observation
- Version-specific DOI: 10.5281/zenodo.21378367
- All-versions concept DOI: 10.5281/zenodo.21366088
- Study design: post-exploration methodological audit, not an independent confirmation study.
- Supersession: v3.2.2 is the current version for use and citation; v3.2.0 and v3.2.1 are retained as immutable provenance.

## Reproduce the analysis

1. Create Python 3.9.6 environment and install `config/requirements-analysis-v32.txt`.
2. Obtain INSPIRE 1.4.2, MOVER, and VitalDB from their official providers under current access terms.
3. Run `scripts/configure_runtime_v32.py` with local paths. The generated `runtime_config_v32.local.json` is private and ignored.
4. From a clean clone with empty `data/processed`, run `scripts/run_analysis_v32.sh`.

The full 1,000-replicate chain refits observation models, auxiliary outcome models, the source preprocessor and classifier, local recalibration, target metrics, and 18 stage-specific MNAR cells. Aggregate CSVs and the workbook are included for inspection. Patient-level raw and derived data, serialized model objects, and private runtime configuration are excluded.

`qa/raw_rebuild_equivalence_v32.csv` is frozen internal release evidence comparing the current raw rebuild with prior locked patient-level analysis inputs. It cannot be regenerated from this public package because those prior locked inputs are excluded. A fresh public replay instead regenerates the current cohort denominators and analysis outputs, which can be compared with the included aggregate release outputs.

Positive aggregate cells below 5 are suppressed in public displays as `<5`; the corresponding percentage is shown as the upper bound implied by 5 and the displayed denominator. Scientific source tables retain their exact values outside the public package.

## Claim boundary

The four-variable model is an auditable methodological probe. Clinical utility, fairness, prospective workflow effects, and patient benefit were not evaluated. MOVER 2022 is a post-exploration temporally held-out evaluation.

## License

Source code under `scripts/` is licensed under the MIT License. Original documentation, aggregate tables, figures, model metadata, and workbook content are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0), except where otherwise noted. See `LICENSE`, `LICENSE-CODE`, `LICENSE-CONTENT`, and `THIRD_PARTY_NOTICES.md`. These licenses do not grant rights to any source dataset, which remains governed by its originating provider.
