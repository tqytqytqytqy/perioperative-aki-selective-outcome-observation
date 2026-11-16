# Perioperative AKI ascertainment and representation

## Current release

**v4.0.0** contains the author-approved identity-consistent reanalysis
`R2_IDQC_20261113`, distributed on 16 November 2026. It supersedes v3.4.0 for
the current analysis. Earlier tags and their Zenodo records remain historical
records and must not be substituted for this version.

This release examines how selective creatinine testing, follow-up coverage,
and laboratory representation affect model development, local recalibration,
and temporal evaluation. It is research software, not a clinically validated
prediction service or an accepted journal publication.

## Contents

- `Supplementary_file_4/`: portable R2 analysis, raw-data entry points, testing-time
  diagnostic, tests, contracts, provenance, and publication-filtered aggregates.
- `Supplementary_file_5/`: independent grouped-calibration calculation, tests,
  contracts, provenance, and three publication-filtered aggregate tables.
- `workbook/Supplementary_file_2_Reported_aggregate_results.xlsx`: the complete
  approved 44-sheet results workbook, unchanged from the submission source.
- `workbook/csv/` and `workbook/INDEX.json`: lossless cell-value exports for
  convenient inspection. The original workbook controls formatting and notes.
- `RELEASE_MANIFEST_SHA256.csv`: exact file fingerprints, excluding itself.

Download the tagged release from https://github.com/tqytqytqytqy/perioperative-aki-selective-outcome-observation/releases/tag/v4.0.0.
Extract the archive without renaming the two supplementary directories. Start
with the README in the directory of interest. Both packages are independently
runnable; repeated helper files are intentional, not additional observations.

## Aggregate checks without patient data

Use Python 3.12 and the pinned `requirements.txt` inside each package. With a
matching environment and the working directory set to `Supplementary_file_4`:

```sh
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/../local_plot_cache"
python -B aggregate_only.py --aggregates aggregates --output ../file4_aggregate_plots
```

From `Supplementary_file_5`, run:

```sh
python -B aggregate_only.py --aggregates aggregates --output ../file5_aggregate_plot
```

The first command checks the supplied tables and draws supported metrics. The
second redraws Figure S6 directly from numeric calibration summaries. Suppressed
and unknown values are not replaced with zero or reconstructed.

## Validation and access boundary

The packaged validation reports dated 15 November 2026 record prepared-R2 point
analysis and grouped-calibration agreement, synthetic raw-route tests, and
aggregate plotting. Their original "author-review candidate" labels describe
their preparation status. The unchanged workbook likewise retains its preparation
snapshot notes about approval and archive status. Author approval for this public
distribution was confirmed on 16 November; this release README records the current
distribution status. No new numerical analysis is asserted by packaging.

Full real raw-to-R2 equivalence across all three databases and a new real-data
1000-replicate full-chain bootstrap were not established by that distribution
validation. S6 uses conditional target-patient intervals, not full-chain or
simultaneous intervals. Testing-time source-risk stratification retains its
specified earlier frozen diagnostic model rather than silently substituting an
R2 refit. Detailed commands, frozen rules and limitations are in each README.

Obtain source data separately from INSPIRE, MOVER and VitalDB under their access
and redistribution terms. This release contains no raw data, patient identifiers,
row-level clinical records, individual predictions, serialized models, bootstrap
replicates, manuscript files or confidential peer-review correspondence. Small
cells and linked summaries remain suppressed in the supplied aggregate tables.
Source-provider restrictions are not replaced by this repository's licenses.

## Licensing and citation

Original code is MIT-licensed. Original documentation and permitted aggregate
content are CC BY 4.0. See the license files, third-party notices and the provider
attribution in each supplementary directory. Cite this exact tagged version and
its corresponding Zenodo version DOI, rather than an earlier version's DOI.

Authors: Qingyu Teng, Tingting Niu, Qian Chen, Min Tao, Ziyan Gu, Qi Li,
Yingya Zhao and Hui Zhang. The first three authors contributed equally.
Corresponding authors: Qi Li, Yingya Zhao and Hui Zhang.
