#!/usr/bin/env python3
"""Apply reviewed disclosure controls to a STAGING copy, never to source data."""
import argparse
import csv
import json
import math
from pathlib import Path


def read_rows(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_rows(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def upper_bound(denominator, percent=False):
    value = 5 / float(denominator) * (100 if percent else 1)
    digits = 2 if percent else 6
    value = math.ceil(value * 10 ** digits) / 10 ** digits
    return f"<{value:.{digits}f}"


def apply(package):
    plan = []
    changed_files = []
    file = package / "tables/33_predictor_missingness_v32.csv"
    fields, rows = read_rows(file)
    for index, row in enumerate(rows, 2):
        if not str(row["missing_n"]).startswith("<") and 0 < float(row["missing_n"]) < 5:
            row["missing_n"] = "<5"
            row["missing_percent"] = upper_bound(row["eligible_n"], percent=True)
            for cell, value in [(f"D{index}", "<5"), (f"E{index}", row["missing_percent"])]:
                plan.append({"sheet": "Table_S5", "cell": cell, "value": value})
    write_rows(file, fields, rows)
    changed_files.append(file.relative_to(package).as_posix())
    file = package / "tables/R1_surgical_flags.csv"
    fields, rows = read_rows(file)
    for index, row in enumerate(rows, 2):
        value = row["n_before_surgery_exclusions"]
        if not value.startswith("<") and 0 < float(value) < 5:
            row["n_before_surgery_exclusions"] = "<5"
            for sheet in ["Table_S23", "S23_numeric"]:
                plan.append({"sheet": sheet, "cell": f"C{index}", "value": "<5"})
        elif row["dataset"] == "INSPIRE" and row["flag"] in {
            "obstetric_union_corrected", "OG_without_obstetric_PCS"
        }:
            row["n_before_surgery_exclusions"] = "suppressed"
            for sheet in ["Table_S23", "S23_numeric"]:
                plan.append({"sheet": sheet, "cell": f"C{index}", "value": "suppressed"})
    write_rows(file, fields, rows)
    changed_files.append(file.relative_to(package).as_posix())
    _, flow = read_rows(package / "tables/01_cohort_flow_and_roles_v32.csv")
    denominators = {r["cohort"]: (float(r["eligible_n"]), float(r["observed_outcome_n"]))
                    for r in flow if r["cohort"] in {"INSPIRE", "MOVER 2021", "MOVER 2022"}}
    file = package / "tables/32_stage_observation_balance_v32.csv"
    fields, rows = read_rows(file)
    small = {}
    affected = set()
    for index, row in enumerate(rows, 2):
        if row["level"] == "continuous_or_binary":
            continue
        for j, column in enumerate(fields[3:5]):
            try:
                count = float(row[column]) * denominators[row["phase"]][j]
            except ValueError:
                continue
            if 0 < count < 5 - 1e-8 and abs(count - round(count)) < 1e-6:
                small[index, column] = upper_bound(denominators[row["phase"]][j])
                affected.add((row["phase"], row["variable"]))
    for index, row in enumerate(rows, 2):
        if (row["phase"], row["variable"]) in affected:
            # Complementary categories and SMDs could disclose the suppressed count.
            for j, column in enumerate(fields[3:], 3):
                value = small.get((index, column), "suppressed")
                row[column] = value
                for sheet in ["Table_S24", "S24_numeric"]:
                    plan.append({"sheet": sheet, "cell": f"{chr(65+j)}{index}", "value": value})
    write_rows(file, fields, rows)
    changed_files.append(file.relative_to(package).as_posix())
    return {"policy": "Positive aggregate subgroup counts below 5 are <5; linked rates use upper bounds.",
            "secondary_control": "Categorical balance diagnostics are suppressed for affected phase-variable groups to prevent reconstruction.",
            "exceptions": "Whole-population correction and tie-sensitivity facts without case attributes.",
            "changed_csv_files": changed_files, "workbook_edits": plan}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="A disposable staging copy.")
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.plan.exists():
        raise SystemExit("Disclosure plan already exists; start from an unmodified staging copy.")
    args.plan.write_text(json.dumps(apply(args.package), indent=2) + "\n")
