import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from prism.common import ROOT, load_config
from prism.metrics import METRIC_KEYS

spec = importlib.util.spec_from_file_location('prism_inference_report', Path(ROOT) / 'scripts/05_infer.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseReportIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = str(Path(self.temp.name) / 'release.xlsx')
        self.items = [{'id': str(i), 'sample_id': 'sample', 'release_id': 'R' + str(i)} for i in range(3)]
        self.cfg = load_config()

    def test_incomplete_group_keeps_missing_invalid_rows_and_has_no_winner(self):
        rows = [{'id': '0', 'pred': {key: .9 for key in METRIC_KEYS}}, {'id': '1', 'pred': None}]
        module.release_report(self.cfg, rows, self.items, self.out)
        frame = pd.read_excel(self.out, sheet_name='releases')
        self.assertEqual(frame.prediction_status.tolist(), ['valid', 'invalid', 'missing'])
        self.assertTrue(frame['rank'].isna().all())
        self.assertFalse(frame.is_best_release.any())

    def test_complete_equal_best_scores_receive_equal_rank(self):
        rows = [{'id': str(i), 'pred': {key: .9 if i < 2 else .8 for key in METRIC_KEYS}} for i in range(3)]
        module.release_report(self.cfg, rows, self.items, self.out)
        frame = pd.read_excel(self.out, sheet_name='releases')
        self.assertEqual(frame['rank'].tolist(), [1, 1, 3])
        self.assertEqual(frame.is_best_release.tolist(), [True, True, False])

    def test_entirely_missing_group_is_reported(self):
        module.release_report(self.cfg, [], self.items, self.out)
        coverage = pd.read_excel(self.out, sheet_name='coverage')
        self.assertEqual(coverage.valid_releases.tolist(), [0])
        self.assertEqual(coverage.expected_releases.tolist(), [3])


if __name__ == '__main__':
    unittest.main()
