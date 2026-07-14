# v3.2 canonical and stage-specific MNAR results

Status: **SCIENTIFIC ANALYSIS COMPLETE; NOT READY FOR SUBMISSION**. Administrative, authorship, ethics, license, and independent-review gates remain open.

The source preprocessor was fitted in all 33,396 eligible INSPIRE patients, while the canonical classifier was fitted in the 24,874 outcome-observed patients using normalized inverse-observation weights truncated at the 1st and 99th percentiles. The same sequence was repeated within every bootstrap sample.

The canonical MOVER 2022 O/E estimate was 0.983 (95% percentile interval 0.829 to 1.161), and the calibration slope was 0.952 (0.727 to 1.236). These are measured-variable MAR estimates, not identified full-population truths.

Source-, update-, and target-stage outcome-regression completion scenarios used fixed odds multipliers 0.5, 0.67, 1, 1.5, 2, and 3. Multiplier 1 is the outcome-regression completion reference and is not numerically constrained to equal the canonical IPW/AIPW estimate. All 1000 valid patient-level replicates refitted observation models, auxiliary outcome models, the full-eligible source preprocessor, the source classifier, local recalibration, and target metrics. The source-stage target O/E point estimates over the prespecified multipliers ranged from 0.974 to 0.983.

Clinical utility was not evaluated. The results do not establish readiness for clinical deployment.
