import copy
import json
from pathlib import Path
import tempfile
import unittest

from prism.common import ROOT, load_config
from prism.metrics import METRIC_KEYS
from prism.runtime import prepare_prediction_run
from prism.validation import validate_output, validate_scores


class ScoreIntegrity(unittest.TestCase):
    def test_invalid_values_do_not_become_high_scores(self):
        good = {k: 1.0 for k in METRIC_KEYS}
        for value in (True, "1.0", None, float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value):
                self.assertIsNone(validate_scores(dict(good, ocr_accuracy=value))[0])

    def test_duplicate_aliases_and_json_keys_are_rejected(self):
        good = {k: 1.0 for k in METRIC_KEYS}
        self.assertIsNone(validate_scores(dict(good, **{"OCR Accuracy": 0.1}))[0])
        raw = json.dumps(good).replace('"ocr_accuracy": 1.0', '"ocr_accuracy": 1.0, "ocr_accuracy": 0.1')
        self.assertIsNone(validate_output(raw)[0])


class PredictionResume(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "predictions.jsonl"
        self.cfg = load_config()
        self.items = [{"id": "a", "sample_id": "a", "release_id": "R"}]
        self.row = {**self.items[0], "pred": {k: 1.0 for k in METRIC_KEYS}}

    def prepare(self, cfg=None, items=None):
        return prepare_prediction_run(None, cfg or self.cfg, items or self.items, "P4", 768, str(self.path))

    def test_incompatible_model_and_duplicate_ids_refused(self):
        self.prepare()
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["base"] += "-changed"
        with self.assertRaisesRegex(ValueError, "resume refused"):
            self.prepare(cfg)
        with self.assertRaisesRegex(ValueError, "duplicate pair"):
            self.prepare(items=self.items * 2)

    def test_legacy_predictions_are_not_silently_adopted(self):
        self.path.write_text(json.dumps(self.row) + "\n")
        with self.assertRaisesRegex(ValueError, "no provenance"):
            self.prepare()

    def test_crash_tail_is_recoverable_and_newline_is_preserved(self):
        self.prepare()
        self.path.write_text(json.dumps(self.row) + '\n{"id":')
        self.assertEqual(self.prepare(), [self.row])
        self.assertEqual(self.path.read_text(), json.dumps(self.row) + "\n")
        self.path.write_text(json.dumps(self.row))
        self.prepare()
        self.assertTrue(self.path.read_bytes().endswith(b"\n"))

    def test_duplicate_unknown_or_corrupt_complete_rows_fail(self):
        self.prepare()
        for content in (json.dumps(self.row) + "\n" + json.dumps(self.row) + "\n",
                        json.dumps(dict(self.row, id="unknown")) + "\n", "not JSON\n"):
            self.path.write_text(content)
            with self.assertRaises(ValueError):
                self.prepare()

    def test_actual_changed_image_bytes_invalidate_resume(self):
        image = Path(self.temp.name) / "image.bin"
        image.write_bytes(b"old-content")
        self.items[0].update(original_image_path=str(image), release_image_path=str(image))
        self.prepare()
        image.write_bytes(b"new-content")
        with self.assertRaisesRegex(ValueError, "resume refused"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
