#!/usr/bin/env python3
"""Build the v3.2 input manifest, schema, variable map, and derivation report."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(
    os.environ.get("V32_CONFIG_PATH", ROOT / "config" / "analysis_config_v32.json")
).expanduser().resolve()
ACCESS_DATE = "2026-07-14"
PUBLIC_RELEASE_VERSION = os.environ.get("PUBLIC_RELEASE_VERSION", "3.2.2")
ZENODO_VERSION_DOI = os.environ.get("ZENODO_VERSION_DOI", "see CITATION.cff")
ZENODO_CONCEPT_DOI = "10.5281/zenodo.21366088"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(config: dict[str, object]) -> pd.DataFrame:
    raw = config["raw_data"]
    definitions = [
        {
            "dataset": "INSPIRE",
            "input_role": "required raw release archive",
            "path_key": "inspire_archive",
            "official_name": "INSPIRE, a publicly available research dataset for perioperative medicine",
            "version": "1.4.2",
            "persistent_identifier": "doi:10.13026/1eay-yc85",
            "official_location": "https://physionet.org/content/inspire/1.4.2/",
            "access_requirement": "PhysioNet credential, required training, and signed project DUA",
            "redistribution": "prohibited; do not publish raw or patient-level derived data",
            "retrieval_note": "local release archive; official metadata verified 2026-07-14",
        },
        {
            "dataset": "MOVER",
            "input_role": "required raw EPIC patient-information table",
            "path_key": "mover_patient_information",
            "official_name": "MOVER: Medical Informatics Operating Room Vitals and Events Repository",
            "version": "first release; EPIC component, 2017-2022",
            "persistent_identifier": "doi:10.24432/C5VS5G",
            "official_location": "https://archive.ics.uci.edu/dataset/877/mover-medical-informatics-operating-room-vitals-and-events-repository",
            "access_requirement": "signed MOVER DUA",
            "redistribution": "prohibited unless separately authorized by the provider",
            "retrieval_note": "original EPIC export table retained locally",
        },
        {
            "dataset": "MOVER",
            "input_role": "required raw EPIC laboratory table",
            "path_key": "mover_patient_labs",
            "official_name": "MOVER: Medical Informatics Operating Room Vitals and Events Repository",
            "version": "first release; EPIC component, 2017-2022",
            "persistent_identifier": "doi:10.24432/C5VS5G",
            "official_location": "https://archive.ics.uci.edu/dataset/877/mover-medical-informatics-operating-room-vitals-and-events-repository",
            "access_requirement": "signed MOVER DUA",
            "redistribution": "prohibited unless separately authorized by the provider",
            "retrieval_note": "original EPIC export table retained locally",
        },
        {
            "dataset": "VitalDB",
            "input_role": "required raw official API cases table for supportive analysis",
            "path_key": "vitaldb_cases",
            "official_name": "VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients",
            "version": "1.0.0 dataset; official API snapshot retained 2026-06-17",
            "persistent_identifier": "doi:10.13026/czw8-9p62",
            "official_location": "https://api.vitaldb.net/cases",
            "access_requirement": "provider terms and data-use agreement apply",
            "redistribution": "exclude from repository; provider DUA and CC BY-NC-SA 4.0 terms apply",
            "retrieval_note": "official API snapshot retains numeric ages for 8 patients now top-coded as >89 in the PhysioNet CSV",
        },
        {
            "dataset": "VitalDB",
            "input_role": "required raw official API laboratory table for supportive analysis",
            "path_key": "vitaldb_labs",
            "official_name": "VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients",
            "version": "1.0.0 dataset; official API snapshot retained 2026-06-17",
            "persistent_identifier": "doi:10.13026/czw8-9p62",
            "official_location": "https://api.vitaldb.net/labs",
            "access_requirement": "provider terms and data-use agreement apply",
            "redistribution": "exclude from repository; provider DUA and CC BY-NC-SA 4.0 terms apply",
            "retrieval_note": "SHA-256 matches the PhysioNet v1.0.0 lab_data.csv file",
        },
    ]
    rows = []
    for definition in definitions:
        path = Path(raw[definition.pop("path_key")])
        stat = path.stat()
        rows.append(
            {
                **definition,
                "access_or_verification_date": ACCESS_DATE,
                "expected_filename": path.name,
                "local_path_internal_only": str(path),
                "file_size_bytes": stat.st_size,
                "file_modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
                "sha256": sha256(path),
                "required_for_v32_analysis": True,
                "public_release_allowed": False,
            }
        )
        print(f"Hashed {path.name}", flush=True)
    return pd.DataFrame(rows)


def build_variable_map() -> pd.DataFrame:
    rows = [
        ("INSPIRE", "patient_key", "operations.subject_id", "SHA-256 namespace key", "deidentified grouping key; never release raw identifier"),
        ("INSPIRE", "case_key", "operations.op_id", "SHA-256 namespace key", "internal deidentified operation grouping key; raw values are not released"),
        ("INSPIRE", "age", "operations.age", "numeric", "age >=18"),
        ("INSPIRE", "sex_male", "operations.sex", "starts with M", "binary predictor"),
        ("INSPIRE", "asa", "operations.asa", "numeric", "observation-model covariate"),
        ("INSPIRE", "duration_h", "operations.anend_time - anstart_time", "minutes / 60", "general anesthesia; 0.5-24 h"),
        ("INSPIRE", "cardiac exclusion", "department, cpbon_time, icd10_pcs", "CTS or CPB or ICD-10-PCS 02*", "excluded"),
        ("INSPIRE", "obstetric exclusion", "department, icd10_pcs", "OG or obstetric ICD-10-PCS", "excluded"),
        ("INSPIRE", "baseline_cr", "labs: creatinine, chart_time, value", "latest valid value in 7 d before anesthesia", "must be <4 mg/dL"),
        ("INSPIRE", "tested_7d", "labs: postoperative creatinine", ">=1 valid value after anesthesia through discharge/death/day 7", "outcome-observation indicator"),
        ("INSPIRE", "aki", "baseline_cr, cr_max_48h, cr_max_7d", ">=0.3 mg/dL by 48 h or >=1.5x by day 7", "creatinine KDIGO; urine output unavailable"),
        ("MOVER", "patient_key", "patient_information.MRN", "SHA-256 namespace key", "patient grouping and 2021/2022 overlap check"),
        ("MOVER", "case_key", "patient_information.LOG_ID", "SHA-256 namespace key", "internal deidentified operation grouping key; raw values are not released"),
        ("MOVER", "age", "patient_information.BIRTH_DATE", "released numeric age", "field name is provider-specific"),
        ("MOVER", "sex_male", "patient_information.SEX", "equals male, case-insensitive", "binary predictor"),
        ("MOVER", "asa", "patient_information.ASA_RATING_C", "numeric", "observation-model covariate"),
        ("MOVER", "duration_h", "AN_STOP_DATETIME - AN_START_DATETIME", "hours", "general anesthesia; 0.5-24 h"),
        ("MOVER", "surgery_category", "PRIMARY_PROCEDURE_NM", "deterministic keyword map", "expanded observation model only"),
        ("MOVER", "cardiac/obstetric exclusion", "PRIMARY_PROCEDURE_NM", "prespecified regular expressions", "excluded"),
        ("MOVER", "baseline_cr", "patient_labs", "latest strict blood creatinine in 7 d before anesthesia", "Lab Name=Creatinine; mg/dL; allowed blood LOINC; <4"),
        ("MOVER", "tested_7d", "patient_labs", ">=1 strict blood creatinine after anesthesia through discharge/day 7", "outcome-observation indicator"),
        ("MOVER", "aki", "baseline_cr, cr_max_48h, cr_max_7d", ">=0.3 mg/dL by 48 h or >=1.5x by day 7", "creatinine KDIGO"),
        ("MOVER", "analysis phase", "AN_START_DATETIME", "calendar year 2021 or 2022", "2021 update; 2022 post-exploration temporal evaluation"),
        ("VitalDB", "case_key/patient_key", "cases.caseid", "SHA-256 namespace key", "one supportive case per released case"),
        ("VitalDB", "age/sex/asa", "cases.age, sex, asa", "numeric / M indicator / numeric", "supportive observed-cohort predictors"),
        ("VitalDB", "duration_h", "cases.opend - opstart", "seconds / 3600", "supportive observed-cohort predictor"),
        ("VitalDB", "baseline_cr", "cases.preop_cr then latest pre-op labs cr", "preop_cr preferred", "must be present and <4 for supportive analysis"),
        ("VitalDB", "tested_7d", "labs.name=cr", ">=1 numeric value after opend through day 7", "outcome-observation indicator"),
        ("VitalDB", "aki", "baseline_cr and postoperative cr", ">=0.3 mg/dL by 48 h or >=1.5x by day 7", "supportive-only; 2 values >30 mg/dL retained to reproduce locked cache"),
        ("ALL", "first_eligible", "patient_key + anesthesia start", "earliest clinically eligible operation", "applied to INSPIRE and MOVER; VitalDB supportive only"),
        ("ALL", "prediction_time", "derived design variable", "end of anesthesia", "baseline creatinine remains pre-anesthesia"),
    ]
    return pd.DataFrame(
        rows,
        columns=["dataset", "analysis_variable", "raw_field_or_table", "derivation", "analysis_use_or_note"],
    )


def build_schema() -> dict[str, object]:
    return {
        "schema_version": "v32",
        "prediction_time": "end of anesthesia",
        "endpoint": {
            "name": "creatinine-defined postoperative AKI",
            "observation_window": "through discharge or postoperative day 7 for INSPIRE/MOVER",
            "absolute_rule": "increase >=0.3 mg/dL within 48 hours",
            "relative_rule": "postoperative creatinine >=1.5 times baseline within 7 days",
            "urine_output_used": False,
        },
        "required_analysis_columns": {
            "dataset": "string",
            "patient_key": "deidentified string",
            "case_key": "unique deidentified string",
            "age": "numeric",
            "sex_male": "0/1 numeric",
            "asa": "numeric, nullable",
            "duration_h": "numeric >0",
            "baseline_cr": "numeric, nullable",
            "tested_7d": "boolean",
            "outcome_operational": "0/1, nullable when unobserved",
        },
        "eligibility": {
            "age_minimum": 18,
            "general_anesthesia": True,
            "duration_hours_inclusive": [0.5, 24.0],
            "exclude_cardiac": True,
            "exclude_obstetric": True,
            "first_eligible_operation_per_patient": True,
            "baseline_creatinine_lookback_days": 7,
            "baseline_creatinine_below_mg_dl": 4.0,
        },
        "locked_counts": {
            "INSPIRE": {"eligible": 33396, "observed": 24874, "unobserved": 8522, "events": 1671},
            "MOVER_2021": {"eligible": 2802, "observed": 2212, "unobserved": 590, "events": 245},
            "MOVER_2022": {"eligible": 2587, "observed": 2033, "unobserved": 554, "events": 224},
            "VitalDB_supportive_observed": {"released_cases": 6388, "observed": 4095, "events": 269},
        },
        "cross_stage_constraint": {"MOVER_2021_2022_patient_overlap": 0},
        "analysis_ready_outputs": [
            "data/processed/inspire_rebuilt_v32.parquet",
            "data/processed/mover_rebuilt_v32.parquet",
            "data/processed/vitaldb_supportive_v32.parquet",
        ],
        "redistribution": "No raw or patient-level analysis-ready data may be included in the public release.",
    }


def write_report(manifest: pd.DataFrame) -> None:
    hashes = "\n".join(
        f"- `{row.dataset}` / `{row.expected_filename}`: `{row.sha256}`"
        for row in manifest.itertuples(index=False)
    )
    report = f"""# v3.2 data acquisition and derivation

Verified on {ACCESS_DATE}. This is an internal reproducibility record, not permission to redistribute any dataset.

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

{hashes}

## Release boundary

Raw archives, source tables, patient-level Parquet files, bootstrap patient-level derivatives, and serialized joblib/pickle objects are excluded from the public repository. The verified release contains scripts, configuration templates without local paths, aggregate CSV tables, figures, a JSON model specification, and documentation. Repository: https://github.com/tqytqytqytqy/perioperative-aki-selective-outcome-observation. Version-specific DOI: {ZENODO_VERSION_DOI}. All-versions concept DOI: {ZENODO_CONCEPT_DOI}.

Positive aggregate cells below 5 are suppressed only in public displays. The underlying analysis source tables remain unchanged so that scientific calculations and audit checks retain their exact values.
"""
    (ROOT / "reports" / "data_acquisition_and_derivation_v32.md").write_text(
        report, encoding="utf-8"
    )


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = build_manifest(config)
    manifest.to_csv(ROOT / "config" / "input_data_manifest_v32.csv", index=False)
    build_variable_map().to_csv(
        ROOT / "tables" / "48_raw_to_analysis_variable_map_v32.csv", index=False
    )
    (ROOT / "config" / "analysis_schema_v32.json").write_text(
        json.dumps(build_schema(), indent=2) + "\n", encoding="utf-8"
    )
    write_report(manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "raw_inputs": len(manifest),
                "all_inputs_hashed": bool(manifest["sha256"].str.len().eq(64).all()),
                "variable_map_rows": len(build_variable_map()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
