"""Regression tests for immutable generated data, content identities and leakage guards."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from prism.common import ROOT, load_config
from prism.dataset import (assert_dataset_ready, atomic_json, atomic_jsonl, canonical_hash, dataset_fingerprint,
                           file_sha256, render_sample_once, split_fingerprint, validate_pairs, validate_samples,
                           validate_scores, generation_fingerprint)
from prism.metrics import METRIC_KEYS


def stage(name):
    spec = importlib.util.spec_from_file_location("test_stage_" + name, Path(ROOT) / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DataIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.cfg = load_config()
        for key in ("images_dir", "metadata_dir", "pairs_dir", "reports_dir"):
            self.cfg["paths"][key] = str(self.path / key)
        self.cfg["ladder"]["knobs"] = ["font_size"]
        self.cfg["_generation_fingerprint"] = "test-generation-v1"

    def pair(self, identity="p1", sample="s1"):
        return {"id": identity, "sample_id": sample, "template": "invoice",
                "original_image_path": "a.png", "release_image_path": "b.png",
                "scores": {k: 1.0 for k in METRIC_KEYS}}

    def populate_pairs(self):
        Image.new("RGB", (8, 8), "white").save(self.path / "a.png")
        Image.new("RGB", (8, 8), "grey").save(self.path / "b.png")
        self.cfg["_root"] = str(self.path)
        rows = {sp: [self.pair(sp, sp)] for sp in ("train", "val", "test")}
        for sp in rows:
            atomic_jsonl(Path(self.cfg["paths"]["pairs_dir"]) / f"{sp}.jsonl", rows[sp])
        return rows

    def test_finite_ground_truth_rejects_nan_boolean_and_missing_metric(self):
        good = self.pair()["scores"]
        for value in (float("nan"), float("inf"), True, -0.01, 1.01, "0.9"):
            with self.assertRaises(ValueError):
                validate_scores(dict(good, font_similarity=value))
        with self.assertRaises(ValueError):
            validate_scores({k: v for k, v in good.items() if k != "ocr_accuracy"})
        validate_scores(good)

    def test_generation_identity_ignores_model_count_and_workers_but_tracks_seed(self):
        first = generation_fingerprint(self.cfg)
        another = copy.deepcopy(self.cfg)
        another["model"]["base"] = "a-different-model"
        another["ladder"]["n_originals"] += 50
        another["dataset"]["workers"] = 7
        self.assertEqual(first, generation_fingerprint(another))
        another["project_seed"] += 1
        self.assertNotEqual(first, generation_fingerprint(another))

    def test_content_fingerprint_detects_same_filename_edit_and_is_portable(self):
        rows = self.populate_pairs()
        first = split_fingerprint(rows["train"], root=self.path)
        other = self.path / "copied"
        other.mkdir()
        for name in ("a.png", "b.png"):
            shutil.copyfile(self.path / name, other / name)
        self.assertEqual(first, split_fingerprint(rows["train"], root=other))
        Image.new("RGB", (8, 8), "black").save(self.path / "b.png")
        self.assertNotEqual(first, split_fingerprint(rows["train"], root=self.path))
        self.assertEqual(split_fingerprint([{"id": "mock"}]), split_fingerprint([{"id": "mock"}]))
        with self.assertRaises(FileNotFoundError):
            split_fingerprint([dict(self.pair(), original_image_path="missing.png")], root=self.path)

    def test_split_validator_catches_leakage_duplicates_holdouts_and_empty_splits(self):
        pairs = {sp: [self.pair(sp, sp)] for sp in ("train", "val", "test")}
        validate_pairs(pairs)
        bad = copy.deepcopy(pairs)
        bad["val"][0]["sample_id"] = "train"
        with self.assertRaisesRegex(ValueError, "LEAKAGE"):
            validate_pairs(bad)
        bad = copy.deepcopy(pairs)
        bad["val"][0]["id"] = "train"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_pairs(bad)
        with self.assertRaisesRegex(ValueError, "holdout"):
            validate_pairs(pairs, holdout={"invoice"})
        pairs["val"] = []
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_pairs(pairs)
        validate_pairs(pairs, allow_empty=True)

    def test_dataset_completion_marker_and_manifest_reject_partial_or_modified_data(self):
        self.populate_pairs()
        base = Path(self.cfg["paths"]["pairs_dir"])
        atomic_json(base / "split_manifest.json", {"dataset_fingerprint": dataset_fingerprint(self.cfg)})
        assert_dataset_ready(self.cfg)
        atomic_json(base / ".publishing.json", {})
        with self.assertRaisesRegex(ValueError, "interrupted"):
            assert_dataset_ready(self.cfg)
        (base / ".publishing.json").unlink()
        Image.new("RGB", (8, 8), "red").save(self.path / "a.png")
        with self.assertRaisesRegex(ValueError, "matches"):
            assert_dataset_ready(self.cfg)

    def test_sample_generation_resumes_without_rewriting_images(self):
        first = render_sample_once(self.cfg, 0)
        original = Path(first["original_image_path"])
        before = (file_sha256(original), original.stat().st_mtime_ns)
        second = render_sample_once(self.cfg, 0)
        self.assertEqual(first, second)
        self.assertEqual(before, (file_sha256(original), original.stat().st_mtime_ns))
        self.assertEqual(validate_samples([second])["n_pairs"], 5)
        changed = copy.deepcopy(self.cfg)
        changed["_generation_fingerprint"] = "different"
        with self.assertRaisesRegex(ValueError, "differ"):
            render_sample_once(changed, 0)
        duplicate = copy.deepcopy(first)
        duplicate["releases"].append(copy.deepcopy(first["releases"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_samples([duplicate])

    def test_pixel_locality_is_checked_against_actual_images(self):
        sample = render_sample_once(self.cfg, 0)
        filename = sample["releases"][1]["release_image_path"]
        with Image.open(filename) as image:
            candidate = image.convert("RGB")
        candidate.putpixel((0, 0), (255, 0, 255))
        candidate.save(filename)
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_samples([sample])

    def test_incomplete_ladder_is_rejected(self):
        sample = render_sample_once(self.cfg, 0)
        sample["releases"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_samples([sample])

    def test_engine_rejects_test_unknown_stale_and_mismatched_provenance(self):
        self.populate_pairs()
        engine = stage("08_data_engine.py")
        report = self.path / "evaluation_results.json"
        atomic_json(report, {"model": {"ladder": {}}})
        provenance = {"schema_version": 1, "split": "val", "models": ["model"],
                      "split_fingerprint": split_fingerprint([self.pair("val", "val")], root=self.path),
                      "dataset_fingerprint": dataset_fingerprint(self.cfg), "report_sha256": file_sha256(report)}
        sidecar = self.path / "evaluation_provenance.json"
        for split in ("test", "unknown", None):
            atomic_json(sidecar, dict(provenance, split=split))
            with self.assertRaisesRegex(ValueError, "validation"):
                engine.validation_source(str(report), "model", self.cfg)
        atomic_json(sidecar, provenance)
        engine.validation_source(str(report), "model", self.cfg)
        atomic_json(report, {"tampered": True})
        with self.assertRaisesRegex(ValueError, "do not match"):
            engine.validation_source(str(report), "model", self.cfg)

    def test_assignments_require_valid_fractions_and_unique_ids(self):
        import random
        splits = stage("02_build_splits.py")
        with self.assertRaisesRegex(ValueError, "sum"):
            splits.assign_splits([], {"train": .7, "val": .2, "test": .2}, random.Random(1))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            splits.assign_splits([{"sample_id": "same"}] * 2, self.cfg["splits"], random.Random(1))

    def test_split_publication_is_validated_frozen_and_recoverable(self):
        splits = stage("02_build_splits.py")
        self.cfg["dataset"]["templates"] = ["invoice"]
        records = [render_sample_once(self.cfg, i) for i in range(3)]
        metadata = Path(self.cfg["paths"]["metadata_dir"])
        pairs_dir = Path(self.cfg["paths"]["pairs_dir"])
        atomic_jsonl(metadata / "ladder.jsonl", records)
        with patch.object(splits, "load_config", return_value=self.cfg), patch.object(sys, "argv", ["02_build_splits.py"]):
            splits.main()
        original = {sp: (pairs_dir / f"{sp}.jsonl").read_bytes() for sp in ("train", "val", "test")}
        assert_dataset_ready(self.cfg)
        # A partial interrupted write is repairable from unchanged input metadata; the sentinel blocks consumers.
        atomic_json(pairs_dir / ".publishing.json", {})
        atomic_jsonl(pairs_dir / "train.jsonl", [])
        with patch.object(splits, "load_config", return_value=self.cfg), patch.object(sys, "argv", ["02_build_splits.py"]):
            splits.main()
        self.assertFalse((pairs_dir / ".publishing.json").exists())
        self.assertEqual(original, {sp: (pairs_dir / f"{sp}.jsonl").read_bytes() for sp in original})
        # Duplicate source IDs fail before modifying any published pairs.
        atomic_jsonl(metadata / "ladder.jsonl", records + [records[0]])
        with patch.object(splits, "load_config", return_value=self.cfg), patch.object(sys, "argv", ["02_build_splits.py"]):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                splits.main()
        self.assertEqual(original, {sp: (pairs_dir / f"{sp}.jsonl").read_bytes() for sp in original})


if __name__ == "__main__":
    unittest.main()
