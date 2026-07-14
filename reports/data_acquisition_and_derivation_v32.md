# v3.2 data acquisition and derivation

Verified on 2026-07-14. This is an internal reproducibility record, not permission to redistribute any dataset.

## Data sources

**INSPIRE.** The analysis used PhysioNet release 1.4.2 (DOI: 10.13026/1eay-yc85). The official release is restricted to credentialed users who complete the required training and sign the project data-use agreement: https://physionet.org/content/inspire/1.4.2/.

**MOVER.** The analysis used the first MOVER release, specifically the EPIC component covering 2017-2022 (dataset DOI: 10.24432/C5VS5G). Access requires a signed MOVER DUA: https://archive.ics.uci.edu/dataset/877/mover-medical-informatics-operating-room-vitals-and-events-repository.

**VitalDB.** The supportive analysis used official API snapshots of the cases and labs tables retained on 2026-06-17. The dataset citation is VitalDB v1.0.0 (DOI: 10.13026/czw8-9p62). The retained labs SHA-256 equals the PhysioNet v1.0.0 `lab_data.csv` hash. The API cases snapshot contains numeric ages for eight patients that the current PhysioNet CSV top-codes as `>89`; this difference is recorded rather than concealed. Provider terms and the VitalDB data-use agreement apply: https://physionet.org/content/vitaldb/1.0.0/ and https://vitaldb.net/docs/?documentId=OpenDataset%2FOverview.md.

## Cohort derivation

INSPIRE and MOVER included adults aged at least 18 years who underwent general anesthesia lasting 30 minutes to 24 hours. Cardiac and obstetric procedures were excluded, and only the first clinically eligible operation per patient was retained. Baseline creatinine was the latest valid value in the seven days before anesthesia and had to be below 4 mg/dL. MOVER 2021 was used for local recalibration; MOVER 2022 was used for post-exploration temporal evaluation. No retained patient appeared in both periods.

Outcome observation required at least one postoperative creatinine. Creatinine-defined AKI was an increase of at least 0.3 mg/dL within 48 hours or at least 1.5 times baseline through discharge or postoperative day 7, whichever came first, for INSPIRE and MOVER. Urine output was not used because it was not consistently available.

VitalDB remained a supportive observed-cohort stress test. The official API reconstruction reproduced all 6,388 released cases, 4,095 supportive outcome-observed cases, and 269 AKI events. Two released postoperative creatinine values above 30 mg/dL were retained because they were present in the official API snapshot and were required to reproduce the locked supportive cache; excluding them would reduce the event count to 268. This detail does not affect the INSPIRE-MOVER primary chain.

## Audit trail

The raw ETL scripts are `prepare_inspire_v32.py`, `prepare_mover_v32.py`, and `prepare_vitaldb_v32.py`; `validate_analysis_schema_v32.py` enforces denominators and required fields. `compare_rebuilt_cohorts_v32.py` confirmed zero differing cells between the raw-rebuilt v3.2 files and the locked analysis inputs. The MOVER strict-creatinine Parquet is an ETL intermediate and audit artifact; the main analysis reads only `mover_rebuilt_v32.parquet`.

Required input hashes:

- `INSPIRE` / `inspire-a-publicly-available-research-dataset-for-perioperative-medicine-1.4.2.zip`: `abfe6fd97ec902caab9fe7d75a32090a31d8e0691bcddd1326b7b8360e1d0a4d`
- `MOVER` / `patient_information.csv`: `203af139aac2fd5a45a7ec749df7273a7adad450fef6c0648ec700a5d397285a`
- `MOVER` / `patient_labs.csv`: `760e87e1058f08dad6bcfa34227ce359d7f498b2a91578330c1f6b0060d98f42`
- `VitalDB` / `cases.csv`: `d74684b4794b5095c32ca607ab5d6b1f8d04f888a607f4fd48470c8ffe885a0b`
- `VitalDB` / `labs.csv`: `c6e84fb397afe8182a7e6cc3aac6b34502d6ac0fadf1abf0ef643cc8bd50ea8b`

## Release boundary

Raw archives, source tables, patient-level Parquet files, bootstrap patient-level derivatives, and serialized joblib/pickle objects are excluded from any public repository. A public release may contain scripts, configuration templates without local paths, aggregate CSV tables, figures, a JSON model specification, and documentation only after author and license approval. No GitHub URL or Zenodo DOI has been created or asserted in v3.2.
