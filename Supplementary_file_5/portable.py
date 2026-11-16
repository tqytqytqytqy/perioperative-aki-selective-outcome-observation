"""Path-only adapter for the extracted, unchanged R2 calculations."""
from pathlib import Path
import hashlib
import importlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def output_directory(path, inputs=()):
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(HERE):
        raise ValueError('Keep local outputs outside the supplementary code directory')
    for source in inputs:
        source = Path(source).resolve()
        if path == source or path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError('Output directory must not overlap an input directory')
    if path.exists() and any(path.iterdir()):
        raise ValueError('Use a new or empty output directory; existing results are not overwritten')
    path.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(path / '.mplconfig')
    os.environ['MPLBACKEND'] = 'Agg'
    sys.dont_write_bytecode = True
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    return path


def bind(analysis_root, output):
    root = Path(analysis_root).expanduser().resolve()
    out = output_directory(output, (root,))
    sys.path.insert(0, str(HERE / 'scripts'))
    a = importlib.import_module('run_v32_analysis')
    for name in ('run_postreview_analysis', 'run_v31_analysis', 'run_v32_analysis',
                 'etl_common_v32', 'etl_outcome_representations_v32'):
        m = importlib.import_module(name)
        m.DATA_DIR = root / 'data/processed'
        m.CONFIG_PATH = HERE / 'config/analysis_config_v32.json'
        for attr, sub in (('TABLE_DIR', 'tables'), ('MODEL_DIR', 'models'),
                          ('FIGURE_DIR', 'figures'), ('REPORT_DIR', 'reports'),
                          ('QA_DIR', 'qa'), ('TMP_DIR', 'tmp')):
            setattr(m, attr, out / sub)
    cfg = json.loads((HERE / 'config/analysis_config_v32.json').read_text())
    local_cfg = root / 'config/analysis_config_v32.json'
    if local_cfg.is_file():
        local = json.loads(local_cfg.read_text())
        for key in ('analysis_version', 'random_seed', 'crossfit_folds', 'identity_qc'):
            if local.get(key) != cfg.get(key):
                raise ValueError('Prepared input configuration does not match packaged R2 rules: ' + key)
    return a, cfg, out


def input_snapshot(root):
    root = Path(root)
    paths = [root / 'data/processed' / name for name in (
        'inspire_rebuilt_v32.parquet', 'mover_rebuilt_v32.parquet', 'vitaldb_supportive_v32.parquet')]
    paths += list((root / 'models').glob('*'))
    paths += list((root / 'config').glob('*'))
    paths += [root / 'tables' / name for name in (
        '30_update_selection_target_performance_v32.csv', '34_update_strategy_workload_v32.csv')]
    return {str(p.relative_to(root)): sha(p) for p in paths if p.is_file()}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check_cohorts(cohorts, a, spec=None):
    for phase in ('source', 'update', 'target'):
        f = cohorts[phase]
        obs = a.observed_indicator(f)
        if (f.patient_key.isna().any() or f.patient_key.duplicated().any()
                or f.case_key.duplicated().any()):
            raise ValueError('Missing or duplicate prepared cohort keys: ' + phase)
        if not ((f.baseline_cr < 4).all() and f.first_eligible.all()
                and f.loc[~obs, 'outcome_operational'].isna().all()):
            raise ValueError('Eligibility / unknown-outcome rule failed: ' + phase)
        if spec is not None:
            label = {'source': 'INSPIRE', 'update': 'MOVER_2021', 'target': 'MOVER_2022'}[phase]
            expected = spec['cohort_denominators'][label]
            if len(f) != expected['eligible'] or int(obs.sum()) != expected['observed']:
                raise ValueError('Prepared cohort differs from supplied model specification: ' + phase)
    if set(cohorts['update'].patient_key) & set(cohorts['target'].patient_key):
        raise ValueError('Update/target patient overlap')
