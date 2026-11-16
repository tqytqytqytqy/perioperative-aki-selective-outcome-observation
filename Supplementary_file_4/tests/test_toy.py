from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'scripts'))
import numpy as np
import pandas as pd


def workspace():
    parent = Path(os.environ.get('TMPDIR', HERE.parent / '_validation/tmp'))
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


def toy_prepared(root):
    rng = np.random.default_rng(20260714)
    data = root / 'data/processed'
    data.mkdir(parents=True)
    def frame(label, year, n=240):
        age = rng.uniform(25, 85, n)
        y = rng.binomial(1, .3, n).astype(float)
        tested = rng.uniform(size=n) > .2
        y[~tested] = np.nan
        return pd.DataFrame({'case_key': [f'toy-{label}-case-{i}' for i in range(n)],
            'patient_key': [f'toy-{label}-person-{i}' for i in range(n)],
            'age': age, 'sex_male': rng.integers(0, 2, n).astype(float),
            'duration_h': rng.uniform(.6, 8, n), 'baseline_cr': rng.uniform(.4, 1.5, n),
            'asa': rng.integers(1, 4, n).astype(float),
            'surgery_category': rng.choice(['orthopedic', 'abdominal', 'other'], n),
            'year': year, 'first_eligible': True, 'has_baseline_cr': True,
            'baseline_cr_under4': True, 'tested_7d': tested,
            'outcome_operational': y, 'aki': y,
            'an_start': pd.date_range(f'{year}-01-01', periods=n, freq='D')})
    source, update, target = frame('source', 2020), frame('update', 2021), frame('target', 2022)
    source.to_parquet(data / 'inspire_rebuilt_v32.parquet', index=False)
    pd.concat([update, target], ignore_index=True).to_parquet(data / 'mover_rebuilt_v32.parquet', index=False)
    source.to_parquet(data / 'vitaldb_supportive_v32.parquet', index=False)


class ToyTests(unittest.TestCase):
    def test_raw_vitaldb_csv_cli(self):
        with workspace() as directory:
            root = Path(directory)
            raw = root / 'raw'
            raw.mkdir()
            cases = pd.DataFrame({'caseid': ['toy-1', 'toy-2', 'toy-3'],
                'opstart': [0., 0., 0.], 'opend': [3600., 3600., 3600.],
                'preop_cr': [1., np.nan, .9], 'age': [70, 70, 70],
                'sex': ['M', 'F', 'F'], 'asa': [2, 2, 2], 'ane_type': ['General'] * 3})
            labs = pd.DataFrame({'caseid': ['toy-1', 'toy-2', 'toy-2'],
                'dt': [7200., -3600., 7200.], 'name': ['cr'] * 3, 'result': [1.3, .8, 31.]})
            cases.to_csv(raw / 'cases.csv', index=False)
            labs.to_csv(raw / 'labs.csv', index=False)
            proc = subprocess.run([sys.executable, '-B', str(HERE / 'rebuild_vitaldb.py'),
                '--cases', str(raw / 'cases.csv'), '--labs', str(raw / 'labs.csv'),
                '--output', str(root / 'prepared')], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            f = pd.read_parquet(root / 'prepared/vitaldb_supportive_v32.parquet')
            self.assertEqual(len(f), 3)
            self.assertEqual(int(f.aki.eq(1).sum()), 2)
            self.assertEqual(f.baseline_cr.iloc[1], .8)
            self.assertTrue(pd.isna(f.aki.iloc[2]))
            self.assertFalse({'caseid', 'MRN'}.intersection(f.columns))

    def test_r1_decimal_thresholds(self):
        from r1_threshold_validation import decimal_aki
        from thresholds_r1 import ge_threshold
        self.assertEqual(decimal_aki(.9, 1.2, 1.2), 1)
        self.assertTrue(ge_threshold(.9, 1.5 * .6))
        self.assertFalse(ge_threshold(.3 - 1e-8, .3))
        self.assertFalse(ge_threshold(np.nan, .3))

    def test_raw_inspire_zip_cli(self):
        import gzip
        import zipfile
        with workspace() as directory:
            root = Path(directory)
            raw = root / 'raw'
            raw.mkdir()
            ops = pd.DataFrame({
                'subject_id': ['toy-s', 'toy-s', 'toy-t', 'toy-ob'],
                'op_id': ['toy-o1', 'toy-o2', 'toy-o3', 'toy-o4'],
                'age': [70] * 4, 'asa': [2] * 4, 'sex': ['M', 'M', 'F', 'F'],
                'anstart_time': [10000, 20000, 30000, 40000],
                'anend_time': [10120, 20120, 30120, 40120],
                'discharge_time': [25000, 35000, 45000, 55000],
                'inhosp_death_time': [None] * 4, 'cpbon_time': [None] * 4,
                'department': ['ORTHO'] * 4, 'icd10_pcs': ['', '', '', '10D00'],
                'antype': ['General'] * 4})
            labs = pd.DataFrame({'subject_id': ['toy-s', 'toy-s', 'toy-t'],
                'chart_time': [9900, 10200, 29900], 'item_name': ['creatinine'] * 3,
                'value': [1.0, 1.3, .9]})
            archive = raw / 'toy.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for name, f in [('operations', ops), ('labs', labs)]:
                    z.writestr('toy/' + name + '.csv.gz', gzip.compress(f.to_csv(index=False).encode()))
            proc = subprocess.run([sys.executable, '-B', str(HERE / 'rebuild_inspire.py'),
                '--archive', str(archive), '--output', str(root / 'prepared')], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            f = pd.read_parquet(root / 'prepared/inspire_rebuilt_v32.parquet')
            self.assertEqual(len(f), 3)
            self.assertEqual(int(f.first_eligible.sum()), 2)
            self.assertEqual(int(f.aki.eq(1).sum()), 1)
            self.assertTrue(f.outcome_operational.isna().any())
            self.assertTrue({'outcome_definite', 'outcome_possible'}.issubset(f.columns))

    def test_original_boundary_suite(self):
        import audit_mover
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(audit_mover.Boundaries)
        result = unittest.TestResult()
        suite.run(result)
        self.assertTrue(result.wasSuccessful(), str(result.errors + result.failures))

    def test_identity_ambiguity_and_missing(self):
        from raw_linkage_audit import unique_triples, exact, explains_difference
        c = pd.DataFrame([['toy-e', 'toy-p', 'toy-m'], ['toy-e', 'toy-q', 'toy-n'],
                          [None, 'toy-z', 'toy-k']], columns=['LOG_ID', 'PAT_ID', 'MRN'])
        u, ambiguous, _ = unique_triples(c)
        self.assertTrue(u.empty)
        self.assertIn('toy-e', ambiguous)
        self.assertFalse(exact(None, None))
        self.assertFalse(exact('', ''))
        triple = dict(LOG_ID='toy-e', PAT_ID='toy-p', MRN='toy-m')
        self.assertFalse(explains_difference('toy-m', 'toy-p', triple, {'MRN'}, {'MRN'}, 'toy-e'))
        self.assertFalse(explains_difference('toy-m', 'toy-p', triple, {'MRN'}, {'PAT_ID'}, 'toy-other'))

    def test_no_later_substitution(self):
        from derived_cohort_comparison import first_subset, retained
        f = pd.DataFrame({'person': ['toy-a', 'toy-a'], 'LOG_ID': ['toy-1', 'toy-2'],
            'an_start': pd.to_datetime(['2021-01-01', '2022-01-01']),
            'has_baseline_cr': [False, True], 'baseline_cr_under4': [False, True]})
        self.assertTrue(retained(first_subset(f, 'person')).empty)

    def test_timing_distinct_occasions(self):
        from timing_helpers import unit_tests, summarize_times
        unit_tests()
        self.assertEqual(summarize_times([1, 1, 24], 168)['occasions'], 2)

    def test_raw_csv_to_cohort_cli(self):
        from rebuild_mover import read_strict_labs, reconstruct
        columns = ['LOG_ID', 'MRN', 'BIRTH_DATE', 'ASA_RATING_C', 'SEX', 'AN_START_DATETIME',
                   'AN_STOP_DATETIME', 'HOSP_DISCH_TIME', 'PRIMARY_PROCEDURE_NM', 'PRIMARY_ANES_TYPE_NM']
        info = pd.DataFrame([
            ['toy-a1', 'toy-a', '70', '2', 'Male', '2021-01-01 08:00', '2021-01-01 10:00', '', 'HIP', 'General'],
            ['toy-a2', 'toy-a', '70', '2', 'Male', '2022-01-01 08:00', '2022-01-01 10:00', '', 'HIP', 'General'],
            ['toy-b', 'toy-b', '70', '2', 'Male', '2022-01-02 08:00', '2022-01-02 10:00', '', 'HIP', 'General'],
            ['toy-c', 'toy-c', '70', '2', 'Female', '2022-01-03 08:00', '2022-01-03 10:00', '', 'HIP', 'General']], columns=columns)
        cw = pd.DataFrame([['toy-a2', 'toy-pa', 'toy-a'], ['toy-b', 'toy-pb', 'toy-b'],
                           ['toy-c', 'toy-pc', 'toy-c']], columns=['LOG_ID', 'PAT_ID', 'MRN'])
        labs = pd.DataFrame([
            ['toy-b', 'toy-b', '2160-0', 'creatinine', '1.0', 'MG/DL', '2022-01-02 07:00'],
            ['toy-b', 'toy-b', '2160-0', 'creatinine', '1.3', 'MG/DL', '2022-01-02 11:00'],
            ['toy-b', 'toy-wrong', '2160-0', 'creatinine', '9', 'MG/DL', '2022-01-02 12:00'],
            ['toy-c', 'toy-c', '2160-0', 'creatinine', '0.9', 'MG/DL', '2022-01-03 07:00']],
            columns=['LOG_ID', 'MRN', 'Lab Code', 'Lab Name', 'Observation Value', 'Measurement Units', 'Collection Datetime'])
        with workspace() as directory:
            root = Path(directory)
            raw = root / 'raw'
            raw.mkdir()
            for name, f in [('patient_information', info), ('crosswalk', cw), ('patient_labs', labs)]:
                f.to_csv(raw / (name + '.csv'), index=False)
            strict = read_strict_labs(raw / 'patient_labs.csv', chunksize=2)
            result = reconstruct(pd.read_csv(raw / 'patient_information.csv', dtype=str), cw, strict)
            self.assertEqual(int(result.first_eligible.sum()), 2)
            self.assertFalse(result.loc[result.age.eq(70) & result.year.eq(2022)].empty)
            self.assertEqual(int(result.aki.eq(1).sum()), 1)
            self.assertEqual(int(result.postop_cr_7d_count.sum()), 1)
            self.assertEqual(int(result.aki.isna().sum()), 2)
            proc = subprocess.run([sys.executable, '-B', str(HERE / 'rebuild_mover.py'),
                '--patient-information', str(raw / 'patient_information.csv'),
                '--patient-labs', str(raw / 'patient_labs.csv'), '--crosswalk', str(raw / 'crosswalk.csv'),
                '--output', str(root / 'result')], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            rebuilt = pd.read_parquet(root / 'result/mover_rebuilt_v32.parquet')
            self.assertFalse({'LOG_ID', 'MRN', 'PAT_ID'}.intersection(rebuilt.columns))

    def test_suppression_is_not_zero(self):
        from aggregate_support import numeric, read_aggregate
        values = numeric(pd.Series(['<5', 'suppressed', '0', '12']))
        self.assertTrue(values.iloc[:2].isna().all())
        self.assertEqual(values.iloc[2], 0)
        with workspace() as directory:
            p = Path(directory) / 'bad.csv'
            pd.DataFrame({'case_key': ['toy-only']}).to_csv(p, index=False)
            with self.assertRaises(ValueError):
                read_aggregate(p)

    def test_aggregate_cli_with_suppressed_s28(self):
        with workspace() as directory:
            root = Path(directory)
            ag = root / 'aggregates'
            (ag / 'timing').mkdir(parents=True)
            (ag / 'analysis_tables').mkdir()
            pd.DataFrame({'cohort': ['Toy', 'Toy'],
                'metric': ['Any postoperative test', 'Discharge stop time missing or unusable'],
                'n': [40, None], 'denominator': [50, 50], 'percent': [80, None],
                'ci_low': [65, None], 'ci_high': [90, None],
                'display': ['40/50 (80%)', '<5']}).to_csv(ag / 'timing/S28_overall_timing.csv', index=False)
            pd.DataFrame({'source_model_estimand': ['Toy'], 'update_estimator': ['Toy'],
                'oe_ratio': [1.0], 'calibration_slope': [1.0], 'auroc': [.7], 'brier': [.12]}).to_csv(
                    ag / 'analysis_tables/30_update_selection_target_performance_v32.csv', index=False)
            proc = subprocess.run([sys.executable, '-B', str(HERE / 'aggregate_only.py'),
                '--aggregates', str(ag), '--output', str(root / 'plots')], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            qa = json.loads((root / 'plots/aggregate_checks.json').read_text())
            self.assertEqual(len(qa['files']), 2)
            self.assertTrue(list((root / 'plots').glob('*.png')))

    def test_timing_cli(self):
        from timing_helpers import summarize_times
        from sklearn.linear_model import LogisticRegression
        import joblib
        with workspace() as directory:
            root = Path(directory)
            inputs = root / 'inputs'
            toy_prepared(inputs)
            f = pd.read_parquet(inputs / 'data/processed/mover_rebuilt_v32.parquet')
            for i in f.index:
                summary = summarize_times([12, 60] if f.at[i, 'tested_7d'] else [], 168)
                for key, value in summary.items():
                    f.loc[i, key] = value
            for col in ['tested', 'early', 'late', 'both'] + [f'day{i}' for i in range(1, 8)]:
                f[col] = f[col].astype(bool)
            f['cap_h'], f['discharge_unusable'] = 168., False
            cohort = inputs / 'cohort.parquet'
            f.to_parquet(cohort, index=False)
            features = ['age', 'sex_male', 'duration_h', 'baseline_cr']
            model = LogisticRegression().fit(f[features], np.tile([0, 1], len(f) // 2))
            model_path = inputs / 'toy_model.joblib'
            joblib.dump(model, model_path)
            proc = subprocess.run([sys.executable, '-B', str(HERE / 'run_timing.py'),
                '--cohort', str(cohort), '--source-model', str(model_path),
                '--output', str(root / 'timing')], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((root / 'timing/S31_elapsed_day_coverage_exact.csv').is_file())

    def test_point_and_one_bootstrap_cli(self):
        with workspace() as directory:
            root = Path(directory)
            toy_prepared(root / 'prepared')
            proc = subprocess.run([sys.executable, '-B', str(HERE / 'run_local.py'),
                '--analysis-root', str(root / 'prepared'), '--output', str(root / 'results'),
                '--full-chain-bootstrap', '--replicates', '1'], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            qa = json.loads((root / 'results/local_run_qa.json').read_text())
            self.assertEqual(qa['bootstrap_replicates'], 1)
            self.assertTrue(qa['input_hashes_unchanged'])
            self.assertFalse(list((root / 'results').glob('*.parquet')))


if __name__ == '__main__':
    unittest.main()
