# Corrected results and same-timestamp baseline selection

## Primary extraction rule

For INSPIRE and MOVER, baseline creatinine is the latest valid value in the
seven days before anesthesia. When different valid values share that latest
timestamp, the extraction retains the **first valid row in original released-file
order**. Technically, the preserved ETL uses groupwise timestamp `idxmax()`;
on a tie this selects the first matching row in the retained source ordering.
Do not sort tied values by magnitude, remove repeated timestamps, or change
released-file order before applying this rule.

This is a deterministic technical convention, not a preference for the
clinically most accurate result. Available source fields did not establish
which tied value was correct. The audit found 16 conflicts: 12 in INSPIRE,
3 in MOVER 2021 and 1 in MOVER 2022. These are aggregate correction facts,
not a case listing. None was among the 75 threshold-corrected principal-cohort
cases. No case identifiers, timestamps or patient attributes accompany them.

Inclusive AKI thresholds use an absolute numerical comparison tolerance of
1e-12 for floating-point roundoff, not a clinically meaningful tolerance.
Missing postoperative creatinine remains an unobserved outcome, not no AKI.
INSPIRE's section-1 procedure exclusion accepts five- and seven-character
tokens; its independent department exclusions are not harmonized with MOVER.

## Table S27 scope

The minimum and maximum tied-value scenarios began **only with the original
retained cohort**. The baseline-below-4-mg/dL rule was reapplied after substitution.
Previously excluded patients were not screened for possible entry. Thus this
is not a complete reconstruction of all alternative eligible populations.

The maximum-value scenario changed source observed events from 1,680 to 1,679.
It also changed target eligible n from 2,587 to 2,586 because one outcome-unobserved
target case failed the baseline eligibility recheck. Target observed events
remained 259. These are whole-population correction facts, not subgroup
characteristics.

The existing scenarios refitted observation models, auxiliary outcome models,
source preprocessing/classification, local recalibration and target point
estimates. No new bootstrap intervals, statistical equivalence assessment, or
paired IPW-versus-complete-case strategy-effect analysis was performed for
these tie scenarios. The original source-row rule remains the primary analysis.

## Reference results

| Item | Corrected result |
| --- | --- |
| Main eligible INSPIRE / MOVER 2021 / MOVER 2022 | 33,394 / 2,802 / 2,587 |
| Observed target outcomes / observed target events | 2,033 / 259 |
| No-update target O/E | 1.487 |
| Complete-case recalibration target O/E | 0.967 |
| IPW recalibration target O/E / AUROC | 1.012 / 0.674 |

These rounded anchors are checked against unrounded CSVs, the coefficient
JSON and the workbook. The O/E and AUROC estimates are assumption-dependent
hybrid AIPW/IPW performance estimates for the eligible target population;
they are not simple observed-event ratios or established population truths.
AUROC is unchanged across the three main recalibration strategies. The results
do not show comprehensive IPW superiority or clinical deployability.

MOVER 2022 is post-exploration temporally held-out evaluation, not unseen
independent confirmation. VitalDB remains supportive, with release-specific
baseline/follow-up rules, not a harmonized eligible-population validation.
