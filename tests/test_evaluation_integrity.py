"""Failure-mode tests for honest evaluation; no model, GPU or pretrained weights required."""

import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prism.common import ROOT, load_config
from prism.dataset import canonical_hash, dataset_fingerprint, split_fingerprint
from prism.evaluation import evaluate, explanation_check, ladder_sensitivity, score_errors
from prism.metrics import METRIC_KEYS
from prism.report import write_report

SPEC = importlib.util.spec_from_file_location("evaluation_stage", Path(ROOT) / "scripts/06_evaluate.py")
STAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE)


def scores(value):
    return dict.fromkeys(METRIC_KEYS, value)


def reasoning():
    return "<analysis>\n" + "\n".join(f"- {key}: top-left change of 4 px -> 0.80" for key in METRIC_KEYS) + "\n</analysis>\n"


def row(release="r1", gt=0.8, pred=0.8, sample="s1", **extra):
    return {"id": f"{sample}_{release}", "sample_id": sample, "release_id": release,
            "template": "invoice", "gt": scores(gt), "pred": scores(pred) if pred is not None else None, **extra}


def as_ground_truth(record):
    return {key: value for key, value in {**record, "scores": record["gt"]}.items()
            if key not in ("gt", "pred", "raw", "missing_prediction")}


class LoaderIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.cfg = load_config()
        self.cfg["paths"]["pairs_dir"] = str(self.directory / "pairs")
        self.pred_dir = self.directory / "predictions"
        self.pred_dir.mkdir()
        self.gt = [as_ground_truth(row(f"r{i}")) for i in range(3)]
        for split in ("train", "val", "test"):
            self.write(self.directory / "pairs" / f"{split}.jsonl", self.gt)

    def write(self, path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def predictions(self, records):
        self.write(self.pred_dir / "model_test.jsonl", records)

    def load(self):
        return STAGE.load_rows(self.cfg, "model", "test", str(self.pred_dir))

    def manifest(self, expected):
        identity = {"schema_version": 1, "model": {"base": "actual/model"},
                    "pairs_fingerprint": split_fingerprint(expected)}
        (self.pred_dir / "model_test.jsonl.manifest.json").write_text(
            json.dumps({"identity": identity, "fingerprint": canonical_hash(identity)}), encoding="utf-8")

    def test_missing_outputs_remain_in_coverage_denominator(self):
        self.predictions([{"id": self.gt[0]["id"], "pred": scores(0.8)}])
        loaded = self.load()
        result = evaluate(loaded)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(result["coverage"]["n_missing"], 2)
        self.assertEqual(result["coverage"]["n_valid"], 1)
        self.assertAlmostEqual(result["coverage"]["valid_rate"], 1 / 3)
        self.assertEqual(result["quality_gate"]["decision_accuracy"], 1)
        self.assertAlmostEqual(result["quality_gate"]["decision_accuracy_end_to_end"], 1 / 3)

    def test_missing_prediction_file_is_zero_coverage(self):
        result = evaluate(self.load())
        self.assertEqual(result["coverage"]["n_missing"], 3)
        self.assertEqual(result["coverage"]["valid_rate"], 0)

    def test_duplicate_and_unknown_prediction_ids_are_rejected(self):
        for records in ([{"id": "unknown", "pred": scores(.8)}],
                        [{"id": self.gt[0]["id"], "pred": scores(.8)}] * 2):
            self.predictions(records)
            with self.assertRaises(ValueError):
                self.load()

    def test_duplicate_ground_truth_ids_are_rejected(self):
        self.write(self.directory / "pairs/test.jsonl", [self.gt[0], self.gt[0]])
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            self.load()

    def test_effective_retry_text_is_used_for_explanation(self):
        self.predictions([{"id": self.gt[0]["id"], "pred": scores(.8), "retried": True,
                           "raw": "wrong initial answer", "retry_raw": "final retry"}])
        self.assertEqual(self.load()[0]["raw"], "final retry")
        self.predictions([{"id": self.gt[0]["id"], "pred": scores(.8), "retried": True,
                           "raw": "wrong", "retry_raw": "retry", "effective_raw": "explicit effective"}])
        self.assertEqual(self.load()[0]["raw"], "explicit effective")

    def test_sidecar_identity_and_partial_subset_are_checked(self):
        self.predictions([{"id": self.gt[0]["id"], "pred": scores(.8)}])
        self.manifest(self.gt[:1])
        loaded = self.load()
        self.assertIn("partial", loaded[0]["prediction_scope"])
        self.assertEqual(evaluate(loaded)["coverage"]["n_total"], 3)
        altered = copy.deepcopy(self.gt)
        altered[0]["scores"] = scores(.7)
        self.manifest(altered)
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.load()

    def test_provenance_tracks_split_dataset_and_actual_model_identity(self):
        self.predictions([{"id": record["id"], "pred": scores(.8)} for record in self.gt])
        self.manifest(self.gt)
        out = self.directory / "reports"
        def minimal_report(results, rows, directory, **kwargs):
            Path(directory).mkdir(parents=True, exist_ok=True)
            (Path(directory) / "evaluation_results.json").write_text("{}", encoding="utf-8")
            return str(Path(directory) / "evaluation_report.md")
        with patch.object(STAGE, "write_report", side_effect=minimal_report):
            results = STAGE.run_evaluation(self.cfg, ["model"], "test", str(self.pred_dir), str(out))
        provenance = json.loads((out / "evaluation_provenance.json").read_text())
        self.assertEqual(provenance["split"], "test")
        self.assertEqual(provenance["dataset_fingerprint"], dataset_fingerprint(self.cfg))
        self.assertEqual(provenance["split_fingerprint"], split_fingerprint(self.gt))
        self.assertEqual(results["model"]["provenance"]["base_model"], "actual/model")


class AccuracyIntegrityTests(unittest.TestCase):
    def test_nonfinite_bool_string_and_extra_metric_scores_are_invalid(self):
        for value in (True, False, "0.8", float("nan"), float("inf"), -.1, 1.1):
            bad = row()
            bad["pred"][METRIC_KEYS[0]] = value
            result = evaluate([bad])
            self.assertEqual(result["coverage"]["valid_rate"], 0, repr(value))
            self.assertTrue(result["invalid_outputs"][0]["errors"])
        self.assertTrue(score_errors({**scores(.8), "unexpected": .8}))

    def test_duplicate_semantic_pair_with_different_ids_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate sample/release"):
            evaluate([row(), dict(row(), id="different")])

    def test_ground_truth_ties_allow_different_correct_member_and_order(self):
        result = evaluate([row("a", .9, .89), row("b", .9, .95), row("c", .5, .5)])
        self.assertEqual(result["ranking"]["best_release_accuracy_RSA"], 1)
        self.assertEqual(result["ranking"]["exact_ranking_accuracy"], 1)

    def test_constant_predicted_ties_do_not_receive_full_selection_credit(self):
        result = evaluate([row("a", .9, 1), row("b", .9, 1), row("c", .5, 1)])
        self.assertAlmostEqual(result["ranking"]["best_release_accuracy_RSA"], 2 / 3)
        self.assertEqual(result["ranking"]["exact_ranking_accuracy"], 0)

    def test_incomplete_groups_cannot_inflate_ranking_accuracy(self):
        result = evaluate([row("a", .9, .9), row("b", .5, None)])
        self.assertEqual(result["ranking"]["n_samples_incomplete"], 1)
        self.assertTrue(math.isnan(result["ranking"]["best_release_accuracy_RSA"]))
        self.assertEqual(result["ranking"]["best_release_accuracy_end_to_end"], 0)

    def test_missing_control_never_supplies_an_assumed_perfect_score(self):
        release = row("font_size_1", .8, .7, kind="ladder", artifact_type="font_size", level=1,
                      primary_metric="font_similarity")
        result = ladder_sensitivity([release])
        self.assertTrue(math.isnan(result["rows"][0]["detection_rate"]))
        self.assertEqual(result["rows"][0]["detection_rate_end_to_end"], 0)
        self.assertIsNone(result["jnd"]["font_size"])

    def test_missing_analysis_counts_against_explanation_accuracy(self):
        good = row("font_size_1", kind="ladder", artifact_type="font_size", primary_metric="font_similarity",
                   target=reasoning(), raw=reasoning())
        missing = dict(good, id="s1_font_size_2", release_id="font_size_2", raw=json.dumps(scores(.8)))
        result = explanation_check([good, missing])
        self.assertEqual(result["n_eligible"], 2)
        self.assertEqual(result["explanation_coverage"], .5)
        self.assertEqual(result["location_accuracy"], .5)
        self.assertEqual(result["quantity_accuracy"], .5)
        self.assertEqual(result["location_accuracy_given_explanation"], 1)

    def test_all_invalid_reports_are_writable_and_strict_json(self):
        records = [row("a", pred=None, missing_prediction=True), row("b", pred=None)]
        result = evaluate(records)
        with tempfile.TemporaryDirectory() as directory:
            write_report({"bad_a": result, "bad_b": result}, {"bad_a": records, "bad_b": records}, directory)
            def invalid_constant(value):
                raise AssertionError(f"nonstandard JSON literal: {value}")
            exported = json.loads((Path(directory) / "evaluation_results.json").read_text(), parse_constant=invalid_constant)
            self.assertIsNone(exported["bad_a"]["all_metrics_pooled"]["MAE"])
            self.assertEqual(exported["bad_a"]["coverage"]["valid_rate"], 0)
            self.assertTrue((Path(directory) / "evaluation_metrics.xlsx").is_file())
            self.assertIn("No paired MAE comparison", (Path(directory) / "evaluation_report.md").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
