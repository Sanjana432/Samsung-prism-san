"""Prompt selection must use expected pairs and never label invalid answers as accuracy."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prism.common import ROOT, load_config
from prism.metrics import METRIC_KEYS

SPEC = importlib.util.spec_from_file_location("prompt_study_stage", Path(ROOT) / "scripts/03_prompt_study_and_baseline.py")
STAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE)


def scores(value):
    return dict.fromkeys(METRIC_KEYS, value)


class PromptStudyIntegrityTests(unittest.TestCase):
    def test_missing_predictions_remain_in_expected_denominator(self):
        result = STAGE.score_rows([{"id": "a", "pred": scores(.8)}], {"a": scores(.9), "b": scores(.9)})
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["n_missing"], 1)
        self.assertEqual(result["valid_rate"], .5)
        self.assertAlmostEqual(result["MAE_pooled"], .1)
        self.assertAlmostEqual(result["selection_score"], .35)

    def test_duplicate_unknown_and_invalid_ground_truth_are_rejected(self):
        for rows in ([{"id": "unknown", "pred": scores(.8)}], [{"id": "a", "pred": scores(.8)}] * 2):
            with self.assertRaisesRegex(ValueError, "unknown or duplicate"):
                STAGE.score_rows(rows, {"a": scores(.9)})
        for expected in ({}, {"a": {**scores(.9), "font_similarity": float("nan")}}):
            with self.assertRaisesRegex(ValueError, "ground-truth"):
                STAGE.score_rows([], expected)

    def test_malformed_scores_are_invalid_and_undefined_mae_is_json_null(self):
        for value in (True, "0.9", float("nan"), 1.1):
            result = STAGE.score_rows([{"id": "a", "pred": {**scores(.9), "font_similarity": value}}], {"a": scores(.9)})
            self.assertEqual(result["n_invalid_outputs"], 1)
            self.assertEqual(result["valid_rate"], 0)
            self.assertIsNone(result["MAE_pooled"])
            self.assertFalse(result["eligible_for_selection"])
            self.assertIsNone(json.loads(json.dumps(result, allow_nan=False))["MAE_font_similarity"])

    def test_selection_preserves_rule_and_excludes_entirely_invalid_prompts(self):
        gt = {"a": scores(.9), "b": scores(.9)}
        table = [
            {"prompt": "invalid", **STAGE.score_rows([], gt)},
            {"prompt": "partial", **STAGE.score_rows([{"id": "a", "pred": scores(.9)}], gt)},
            {"prompt": "complete", **STAGE.score_rows([{"id": key, "pred": scores(.7)} for key in gt], gt)},
        ]
        self.assertEqual(STAGE.select_prompt(table), "complete")
        self.assertIsNone(STAGE.select_prompt(table[:1]))

    def test_all_invalid_prompt_study_saves_strict_diagnostics_and_stops_before_test(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = load_config()
            cfg["paths"]["reports_dir"] = directory
            cfg["paths"]["predictions_dir"] = str(Path(directory) / "predictions")
            val = [{"id": "val", "scores": scores(.9)}]
            test = [{"id": "test", "scores": scores(.9)}]
            with (patch.object(STAGE, "load_config", return_value=cfg),
                  patch("prism.dataset.assert_dataset_ready"),
                  patch.object(STAGE, "read_jsonl", side_effect=[val, test]),
                  patch.object(STAGE, "load_model"), patch.object(STAGE, "load_processor"),
                  patch.object(STAGE, "PROMPTS", {"P1": "one", "P4": "four"}),
                  patch.object(STAGE, "predict", return_value=[{"id": "val", "pred": None}]) as predictor,
                  patch("sys.argv", ["03_prompt_study_and_baseline.py"])):
                with self.assertRaisesRegex(RuntimeError, "No prompt produced valid"):
                    STAGE.main()
                self.assertEqual(predictor.call_count, 2)
            def invalid_constant(value):
                raise AssertionError(f"nonstandard JSON literal: {value}")
            report = json.loads((Path(directory) / "prompt_study.json").read_text(encoding="utf-8"),
                                parse_constant=invalid_constant)
            self.assertIsNone(report["selected_prompt"])
            self.assertEqual(report["status"], "failed_no_valid_prompt")
            self.assertTrue(all(row["MAE_pooled"] is None for row in report["table"]))


if __name__ == "__main__":
    unittest.main()
