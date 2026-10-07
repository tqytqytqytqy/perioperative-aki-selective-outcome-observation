# Perioperative AKI methodological model

## Purpose

This four-variable model examines selective outcome observation across model development, local recalibration, and temporal evaluation. It is not intended for clinical care, triage, treatment selection, or prospective deployment.

## Inputs and fitted specification

Inputs are age, sex as recorded in the source datasets, anesthesia duration in hours, and baseline creatinine. The complete fitted coefficients, spline knots, preprocessing parameters, source and target estimands, and recalibration equation are in `models/model_specification_v32.json`. The retained filename is a code-interface identifier; its content and analysis-version field belong to the corrected analysis, not the earlier v3.2.5 fitted model.

Baseline creatinine uses the latest valid pre-anesthesia measurement in the specified window. Different values at the same latest timestamp are resolved deterministically by the first valid record in released-file order. The release does not claim that this technical convention identifies the clinically correct specimen or amended report.

## Population and missing outcomes

INSPIRE supplies model development, MOVER 2021 local recalibration, and MOVER 2022 post-exploration temporal evaluation. VitalDB is supportive observed-cohort evidence, not a full eligible-population validation. Primary eligible denominators are 33,394, 2,802, and 2,587; corresponding observed-outcome denominators are 24,872, 2,212, and 2,033.

Selective postoperative creatinine testing leaves some AKI outcomes unknown. The canonical full-population analysis depends on measured-variable MAR assumptions and positivity, with truncated inverse-probability weights and hybrid AIPW/IPW evaluation. MNAR scenarios explore dependence on departures from these assumptions; they do not identify the missing outcomes.

## Corrected performance and limitations

The target canonical IPW-recalibrated point estimates are O/E approximately 1.012, calibration slope 0.884, AUROC 0.674, and Brier score 0.0937. Consult the aggregate tables for uncertainty intervals and paired comparisons. Recalibration does not improve ranking when it is a monotone transformation, and these values do not demonstrate that weighting is superior to complete-case recalibration.

Same-timestamp minimum/maximum scenarios are retained-cohort point-estimate checks. Under the maximum rule, one source event changes and one outcome-unobserved target operation is excluded by the reapplied baseline threshold. No additional bootstrap intervals or statistical-equivalence conclusion are supplied for these scenarios.

The work does not establish clinical utility, subgroup fairness, prospective workflow effects, or patient benefit. The released parameters are research metadata; no patient-level predictions or serialized fitted model objects are distributed.

## Testing-time diagnostic in v3.4.0

The post hoc diagnostic retains the original operational AKI labels and prediction values. It measures recorded testing times and coverage within the original discharge/death-limited windows. INSPIRE chart time and MOVER specimen collection datetime are not assumed to be interchangeable. The absolute-creatinine threshold uses the selected preoperative baseline; the analysis does not evaluate every possible rolling 48-hour measurement pair. Timing outputs are descriptive indicators, not a clinically adjudicated endpoint or a new model performance estimator.
