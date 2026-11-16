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


class ToyTests(unittest.TestCase):
    def test_safe_groups_fallback(self):
        from supplementary_common import safe_groups
        p = np.linspace(.01, .99, 80)
        y = np.tile([0., 1.], 40)
        codes, edges = safe_groups(p, np.ones(80, bool), y)
        self.assertEqual(len(np.unique(codes)), 5)
        self.assertEqual(len(edges), 6)

    def test_conditional_seed_and_no_silent_redraw(self):
        from supplementary_common import conditional_bootstrap
        codes = np.repeat(np.arange(5), 40)
        y = np.tile([0., 1.], 100)
        first = conditional_bootstrap(codes, np.ones(200), y, 20261007, 20)
        second = conditional_bootstrap(codes, np.ones(200), y, 20261007, 20)
        np.testing.assert_array_equal(first[2], second[2])
        np.testing.assert_allclose(first[0], .5)
        with self.assertRaises(AssertionError):
            conditional_bootstrap(np.arange(5), np.ones(5), np.zeros(5), 20261007, 20)

    def test_aggregate_only_cli(self):
        parent = Path(os.environ.get('TMPDIR', HERE.parent / '_validation/tmp'))
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            root = Path(directory)
            ag = root / 'aggregates'
            ag.mkdir()
            strategies = ['Unupdated', 'Complete-case recalibration', 'IPW recalibration']
            points = pd.DataFrame([{'strategy': s, 'risk_group': i + 1,
                'weighted_mean_prediction': .05 + .05 * i,
                'ipw_event_frequency': .06 + .05 * i,
                'conditional_ci_lower': .03 + .05 * i,
                'conditional_ci_upper': .09 + .05 * i, 'observed_events': '<5'}
                for s in strategies for i in range(5)])
            points.to_csv(ag / 'S6_calibration_points.csv', index=False)
            pd.DataFrame([{'strategy': s, 'quantile': q, 'prediction': q * .5}
                for s in strategies for q in [0, .05, .25, .5, .75, .95, 1]]).to_csv(
                    ag / 'prediction_distribution.csv', index=False)
            contract = json.loads((HERE / 'contracts.json').read_text())
            cols = contract['aggregate_files']['reproduction_metrics.csv']['required']
            pd.DataFrame([{'strategy': s, **{c: .2 for c in cols if c != 'strategy'}}
                          for s in strategies]).to_csv(ag / 'reproduction_metrics.csv', index=False)
            proc = subprocess.run([sys.executable, '-B', str(HERE / 'aggregate_only.py'),
                '--aggregates', str(ag), '--output', str(root / 'plots')], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((root / 'plots/Figure_S6_calibration_curves.pdf').is_file())
            from PIL import Image
            im = np.asarray(Image.open(root / 'plots/Figure_S6_calibration_curves.png').convert('RGB'))
            self.assertGreater(float(im.std()), 10)


if __name__ == '__main__':
    unittest.main()
