# Scientific replay, version 3.3.0

This versioned research-software package supports scientific reproducibility.
It is not a journal publication or a clinical model release, and does not
claim a new end-to-end statistical rerun. It contains the corrected
2026-09-30 computation, the 2026-10-05 raw audit and tie-sensitivity code,
current aggregate figures, and a disclosure-controlled derivative of the
current 2.1 workbook including Table S27.

## Supported entrypoints

Use only `python scripts/replay.py`. Help and `verify` do not read raw data,
fit models, or perform network operations. Internal filenames retaining
`v32` or `R1` are provenance identifiers, not the scientific release version.
`run_v31_analysis.py` and `run_postreview_analysis.py` are imported function
dependencies only. Their historical main programs and outputs are not current
entrypoints. Do not execute them directly.

The exact retained statistical environment was Python 3.12.14 with
`config/requirements-R1-verified.txt`. The file also records ancillary
document-checking packages; no document source or patient data are shipped.
Use a separately managed environment with these versions. No installation
or data acquisition is performed by the replay wrapper.

```sh
python scripts/replay.py --help
python scripts/replay.py verify
python -B -m unittest discover -s tests -v
```

These checks compare frozen public aggregates and run synthetic boundary tests.
They are not an independent statistical validation or a raw-file rerun.

## Private runtime configuration

Create a private copy of `config/analysis_config.example.json` outside the
public package. Replace its five raw-input placeholders with files obtained
through the original providers and the applicable access process. Relative
paths resolve against the private configuration file, not the shell directory.
The wrapper resolves these paths only in the private workspace.

| Role | Required acquisition-format input |
| --- | --- |
| INSPIRE | Original 1.4.2 archive with a unique operations.csv.gz and labs.csv.gz member |
| MOVER operations | Original patient_information.csv from the retained EMR release; retain original fields and row order |
| MOVER labs | Original patient_labs.csv from the same release; retain row order, names, codes, units and timestamps |
| VitalDB cases | cases.csv snapshot used for the original acquisition |
| VitalDB labs | labs.csv snapshot used with those cases, retained 2026-06-17 in the original analysis |

No MOVER release version absent from the acquisition record is invented here.
Changing provider snapshots can change extraction results. This package does
not reinterpret ethics determinations or assert current data-access terms.
Current versus acquisition-time access documentation is maintained separately.
Regardless of provider redistribution terms, no raw or patient-level data
are included in this package.

```sh
python scripts/replay.py init --config ../private-runtime/runtime.json --workdir ../private-replay
python scripts/replay.py etl --workdir ../private-replay --execute
```

Initialization requires an empty or new workspace separate from this package.
It copies the code, creates a local runtime configuration, and copies only
the static variable-definition table. It does not seed the workspace with
frozen results, patient records, predictions, models, or checkpoints.
ETL rebuilds the corrected INSPIRE, MOVER and supportive VitalDB derivatives.
The INSPIRE scope correction and inclusive numerical thresholds are already
in the corrected ETL; fresh raw replay does not require the historical
retained-derivative correction stage.

## Full statistical replay, not run during packaging

```sh
python scripts/replay.py analysis --workdir ../private-replay --execute --bootstrap-replicates 1000 --jobs 6
python scripts/replay.py supplemental --workdir ../private-replay --execute
python scripts/replay.py finalize --workdir ../private-replay --execute
```

The primary seed is 20260714. The existing full-chain procedure includes
1,000 replicates and 18 stage-specific MNAR scenarios per replicate. The
separate fixed-prediction coarsening bootstrap uses seed 20260930 and 1,000
replicates. These are the retained analysis procedures, not new computations
claimed by preparation of this version. Full replay is computationally substantial.

Existing partial checkpoints require explicit `--resume`; use a new workspace
for a fresh replay. `--point-only` runs the existing point-analysis program
without its bootstrap but is not a complete replay. The original finalizer
expects exactly 1,000 canonical and 18,000 MNAR rows, so smaller diagnostic
runs must not be finalized as a scientific reproduction.

Compare regenerated aggregates with the public reference using the
disclosure policy in `DISCLOSURE_CONTROL.md`. Suppressed text is not numeric
input and must never be converted to zero. The public workbook is a current
reference artifact, not a computational input to the statistical analysis.
Generated figures retain computational filenames; Figure_1 and Figure_2
in the public package are the corresponding current presentation aliases.
The corrected Figure_2 is built by the finalizer, not the old two-strategy
`build_v32_figures.py` main program.

## Historical correction and raw audit

For users already holding authorized historical derivatives, the optional
`retained-correction` entrypoint reproduces the original scope/threshold
correction path. Place `inspire_original_v325.parquet` and
`mover_original_v325.parquet` in the private workspace's `data/processed`.
It reads the configured raw operation files, applies the preserved correction
code, and rebuilds VitalDB. It does not supply or acquire those restricted
historical derivatives.

```sh
python scripts/replay.py retained-correction --workdir ../private-replay --execute
python scripts/replay.py raw-audit --workdir ../private-replay --execute
python scripts/replay.py ties --workdir ../private-replay --execute
```

`raw-audit` additionally requires the authorized historical
`vitaldb_supportive_v2.parquet`. The exact audit compares original and
corrected labels and therefore cannot recreate its historical delta counts
without those historical inputs. This is an explicit dependency, not a claim
that frozen audit results have been reverified from the public files alone.
The fresh corrected ETL and main analysis do not require these inputs.

The independent raw audit produces record-level evidence under the private
workspace's `private/` folder. The tie-sensitivity entrypoint consumes that
evidence and the regenerated canonical aggregate table. Never distribute
the private workspace as a public artifact: it contains records, deterministic
linkage keys, runtime paths, fitted model objects and potentially private logs.
The public staging allowlist excludes all of these.

Optional `spotcheck` reruns three full-chain replicates and compares JSON-based
model reconstruction with a locally regenerated model object. It is supplied
as retained code, but was not executed during this packaging task. It must
not be described as a new independent equivalence or clinical validation.

## Replay boundaries

- No complete raw extraction, model refit, bootstrap, or tie-scenario refit was run during packaging.
- Syntax/import checks, safe help, synthetic tests, source hashes, aggregate comparisons and workbook checks are narrower evidence.
- Historical v3.2.5 aggregate results are not current corrected outputs. No v3.1 output table/model/figure is shipped.
- All replay output remains local. There is no uploader, release creator, browser editing or publishing step.
- Consult the repository-level citation, license, data-access documentation and release manifest for archive-wide metadata.
