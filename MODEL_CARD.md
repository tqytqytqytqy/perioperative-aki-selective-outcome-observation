# Model card v3.2

## Purpose

Auditable methodological probe for selective outcome observation across model development, local recalibration, and post-exploration temporal evaluation.

## Inputs and timing

Age, released binary sex, anesthesia duration, and baseline creatinine; prediction at the end of anesthesia.

## Model

- Source preprocessor population: all 33,396 eligible INSPIRE operations.
- Source classifier population: 24,874 outcome-observed INSPIRE operations.
- Classifier weighting: normalized inverse observation probability truncated at the 1st and 99th percentiles.
- Source intercept: -1.6689550104.
- Recalibration: `expit(0.0884660935 + 0.8963305835 * logit(source_probability))`.
- Full ordered coefficients and preprocessing: `models/model_specification_v32.json` and Supplement Tables S6-S7.

## Claim boundaries

Not for clinical use, triage, treatment selection, patient communication, or deployment. MOVER 2022 is a post-exploration temporally held-out evaluation, not independent confirmation. Clinical utility, fairness, prospective workflow performance, and patient benefit were not evaluated.

## Outcome-observation boundary

Canonical estimates rely on measured-variable MAR with truncated IPW/AIPW. Source-, update-, and target-stage MNAR sensitivities show assumption dependence; multiplier 1 is an outcome-regression reference and is not the canonical estimator.

## Release status

Public reproducibility package: https://github.com/tqytqytqytqy/perioperative-aki-selective-outcome-observation. Release-family concept DOI: 10.5281/zenodo.21366088. Previous immutable v3.2.0 DOI: 10.5281/zenodo.21366089.
