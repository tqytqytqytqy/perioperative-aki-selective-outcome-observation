"""Small synthetic checks only; no fitting, resampling or clinical records."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import etl_common_v32 as etl
from thresholds_r1 import ge_threshold
from etl_outcome_representations_v32 import apply_percentile_release


class ScientificBoundaryTests(unittest.TestCase):
    def test_inclusive_numerical_boundaries(self):
        self.assertTrue(ge_threshold(1.2 - 0.9, 0.3))
        self.assertTrue(ge_threshold(0.9, 1.5 * 0.6))
        self.assertFalse(ge_threshold(0.3 - 1e-8, 0.3))
        self.assertFalse(ge_threshold(np.nan, 0.3))

    def test_unknown_outcome_not_non_event(self):
        data = pd.DataFrame({"baseline_cr": [1.0], "cr_max_48h": [np.nan],
                             "cr_max_7d": [np.nan], "postop_cr_7d_count": [0]})
        self.assertTrue(np.isnan(etl.finalize_outcomes(data).aki.iloc[0]))

    def test_first_source_row_wins_same_time_tie(self):
        case = pd.DataFrame({"case_raw": ["SYNTHETIC"], "patient_raw": ["SYNTHETIC"],
                             "an_start_num": [10080], "an_end_num": [10200],
                             "window_48h_end": [13080], "window_7d_end": [20280]})
        labs = pd.DataFrame({"patient_raw": ["SYNTHETIC"] * 3,
                             "lab_time": [10000, 10000, 10300], "cr": [1.0, 1.2, 1.3]})
        result = etl.attach_creatinine_outcomes_relative(case, labs)
        self.assertEqual(result.baseline_cr.iloc[0], 1.0)
        self.assertEqual(result.aki.iloc[0], 1)
        reversed_tie = etl.attach_creatinine_outcomes_relative(case, labs.iloc[[1, 0, 2]])
        self.assertEqual(reversed_tie.baseline_cr.iloc[0], 1.2)
        self.assertEqual(reversed_tie.aki.iloc[0], 0)

    def test_datetime_boundary_and_discharge_cap(self):
        start = pd.Timestamp("2000-01-10")
        end = start + pd.Timedelta(hours=2)
        case = pd.DataFrame({"case_raw": ["SYNTHETIC"], "an_start": [start],
                             "an_end": [end], "window_48h_end": [end + pd.Timedelta(hours=1)],
                             "window_7d_end": [end + pd.Timedelta(hours=1)]})
        labs = pd.DataFrame({"case_raw": ["SYNTHETIC"] * 4,
                             "lab_time": [start-pd.Timedelta(hours=1)] * 2 +
                                         [end+pd.Timedelta(minutes=30), end+pd.Timedelta(hours=2)],
                             "cr": [0.9, 1.0, 1.2, 3.0]})
        result = etl.attach_creatinine_outcomes_datetime(case, labs)
        self.assertEqual(result.baseline_cr.iloc[0], 0.9)
        self.assertEqual(result.cr_max_7d.iloc[0], 1.2)
        self.assertEqual(result.aki.iloc[0], 1)

    def test_exact_cutpoint_enters_lower_interval(self):
        value = apply_percentile_release(pd.Series([0.5, 0.50001, np.nan]),
                                         np.array([0.5, 0.5]), np.array([0.3, 0.4, 0.6]))
        self.assertEqual(value.iloc[0], 0.3)
        self.assertEqual(value.iloc[1], 0.6)
        self.assertTrue(np.isnan(value.iloc[2]))


if __name__ == "__main__":
    unittest.main()
