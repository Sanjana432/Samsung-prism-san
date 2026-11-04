"""Offline tests for run identity, checkpoint safety, context budgets and validation state."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from prism.metrics import METRIC_KEYS
from prism.training import (ValidationGeneration, adapter_complete, assert_finite_training,
                            assert_lora_trainables, atomic_json, context_preflight, ensure_training_run,
                            latest_checkpoint, save_adapter_preserving_previous, training_complete,
                            training_identity, validate_training_pairs, validation_mode)


class Processor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1}, special_tokens_map={"pad_token": "a"},
                                         chat_template="native")
    def to_dict(self):
        return {"size": 280}
    def save_pretrained(self, directory):
        atomic_json(Path(directory) / "processor_config.json", {"native": True})


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_B = torch.nn.Parameter(torch.tensor([1.0]))
        self.config = SimpleNamespace(use_cache=False, text_config=SimpleNamespace(use_cache=False))
    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        atomic_json(Path(directory) / "adapter_config.json", {"peft_type": "LORA"})
        (Path(directory) / "adapter_model.safetensors").write_bytes(b"fake test weights")


class TrainingIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "image.png"
        self.image.write_bytes(b"image pixels")
        self.cfg = {"project_seed": 2026, "model": {"base": "a/model", "max_seq_len": 16},
                    "qlora": {"r": 16}, "train": {"epochs": 3},
                    "inference": {"max_new_tokens": 4, "batch_size": 2}}
        self.item = {"id": "train-r1", "sample_id": "train", "release_id": "r1", "target": json.dumps({k: 1.0 for k in METRIC_KEYS}),
                     "scores": {key: 1.0 for key in METRIC_KEYS},
                     "original_image_path": str(self.image), "release_image_path": str(self.image)}
        self.val = dict(self.item, id="val-r1", sample_id="val")
        self.proc = Processor()

    def test_training_pairs_reject_leakage_duplicates_and_wrong_targets(self):
        validate_training_pairs([self.item], [self.val])
        with self.assertRaisesRegex(ValueError, "cross"):
            validate_training_pairs([self.item], [dict(self.val, sample_id="train")])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_training_pairs([self.item, self.item], [self.val])
        with self.assertRaisesRegex(ValueError, "invalid target"):
            validate_training_pairs([dict(self.item, target="bad")], [self.val])
        with self.assertRaisesRegex(ValueError, "both contain"):
            validate_training_pairs([], [self.val])

    def test_identity_changes_for_model_processor_train_qlora_split_or_pixels(self):
        original = training_identity(self.cfg, self.proc, [self.item], [self.val])["fingerprint"]
        for section, key, value in (("model", "base", "other"), ("train", "epochs", 2), ("qlora", "r", 8)):
            cfg = copy.deepcopy(self.cfg)
            cfg[section][key] = value
            self.assertNotEqual(training_identity(cfg, self.proc, [self.item], [self.val])["fingerprint"], original)
        self.proc.tokenizer.chat_template = "other template"
        self.assertNotEqual(training_identity(self.cfg, self.proc, [self.item], [self.val])["fingerprint"], original)
        self.proc.tokenizer.chat_template = "native"
        self.assertNotEqual(training_identity(self.cfg, self.proc, [dict(self.item, target="other")], [self.val])["fingerprint"], original)
        self.image.write_bytes(b"changed pixels")
        self.assertNotEqual(training_identity(self.cfg, self.proc, [self.item], [self.val])["fingerprint"], original)

    def test_manifest_refuses_stale_or_unidentified_outputs_and_preserves_them(self):
        out = self.root / "outputs"
        identity = training_identity(self.cfg, self.proc, [self.item], [self.val])
        self.assertEqual(ensure_training_run(out, identity), identity["fingerprint"])
        changed = training_identity(dict(self.cfg, project_seed=42), self.proc, [self.item], [self.val])
        with self.assertRaisesRegex(RuntimeError, "preserved"):
            ensure_training_run(out, changed)
        self.assertEqual(json.loads((out / "training_manifest.json").read_text())["fingerprint"], identity["fingerprint"])
        old = self.root / "old"
        old.mkdir()
        (old / "weights").write_bytes(b"user work")
        with self.assertRaisesRegex(RuntimeError, "preserved"):
            ensure_training_run(old, identity)
        self.assertEqual((old / "weights").read_bytes(), b"user work")

    def test_local_base_weight_and_custom_code_edits_change_training_identity(self):
        base = self.root / "local_model"
        base.mkdir()
        weights = base / "model.safetensors"
        weights.write_bytes(b"original weights")
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["base"] = str(base)
        original = training_identity(cfg, self.proc, [self.item], [self.val])["fingerprint"]
        weights.write_bytes(b"modified weights")
        changed = training_identity(cfg, self.proc, [self.item], [self.val])["fingerprint"]
        self.assertNotEqual(original, changed)
        (base / "modeling_custom.py").write_text("implementation = 2", encoding="utf-8")
        self.assertNotEqual(changed, training_identity(cfg, self.proc, [self.item], [self.val])["fingerprint"])

    def test_corrupt_training_manifest_payload_is_rejected_without_replacement(self):
        out = self.root / "outputs"
        identity = training_identity(self.cfg, self.proc, [self.item], [self.val])
        ensure_training_run(out, identity)
        corrupt = copy.deepcopy(identity)
        corrupt["payload"]["seed"] = 99
        atomic_json(out / "training_manifest.json", corrupt)
        with self.assertRaisesRegex(RuntimeError, "corrupt.*preserved"):
            ensure_training_run(out, identity)
        self.assertEqual(json.loads((out / "training_manifest.json").read_text())["payload"]["seed"], 99)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            ensure_training_run(self.root / "invalid", corrupt)
        self.assertFalse((self.root / "invalid").exists())

    def checkpoint(self, step, complete=True):
        path = self.root / "checkpoints" / f"checkpoint-{step}"
        Model().save_pretrained(path)
        atomic_json(path / "trainer_state.json", {"global_step": step})
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (path / name).write_bytes(b"state")
        if not complete:
            (path / "optimizer.pt").unlink()
        return path

    def test_latest_checkpoint_is_numeric_and_skips_incomplete_newer_state(self):
        self.checkpoint(2)
        expected = self.checkpoint(10)
        incomplete = self.checkpoint(20, complete=False)
        self.assertEqual(latest_checkpoint(self.root), str(expected))
        self.assertTrue(incomplete.exists())

    def test_empty_checkpoint_does_not_trigger_resume_and_only_partial_is_rejected(self):
        empty = self.root / "checkpoints" / "checkpoint-1"
        empty.mkdir(parents=True)
        self.assertIsNone(latest_checkpoint(self.root))
        self.checkpoint(2, complete=False)
        with self.assertRaisesRegex(RuntimeError, "preserved"):
            latest_checkpoint(self.root)

    def test_completion_requires_matching_identity_and_both_complete_adapters(self):
        atomic_json(self.root / "training_log.json", {"completed": True, "run_fingerprint": "a"})
        self.assertFalse(training_complete(self.root, "a"))
        Model().save_pretrained(self.root / "best_adapter")
        Model().save_pretrained(self.root / "final_adapter")
        atomic_json(self.root / "best_adapter" / "val_score.json", {"selection_score": .1, "run_fingerprint": "a"})
        self.assertTrue(training_complete(self.root, "a"))
        self.assertFalse(training_complete(self.root, "b"))
        (self.root / "best_adapter" / "val_score.json").unlink()
        self.assertFalse(training_complete(self.root, "a"))

    def collator(self, items):
        item = items[0]
        n = item.get("length", 8)
        labels = torch.full((1, n), -100)
        labels[:, 4:] = 1
        return {"input_ids": torch.ones((1, n), dtype=torch.long), "labels": labels,
                "attention_mask": torch.ones((1, n), dtype=torch.long)}

    def test_context_preflight_checks_every_pair_and_records_late_failure(self):
        out = self.root / "preflight.json"
        seen = []
        def encode(items):
            seen.append(items[0]["id"])
            return self.collator(items)
        with self.assertRaisesRegex(RuntimeError, "before model loading"):
            context_preflight([self.item], [self.val, dict(self.val, id="bad", length=20)], encode, self.cfg, out, "a")
        self.assertEqual(seen, ["train-r1", "val-r1", "bad"])
        report = json.loads(out.read_text())
        self.assertEqual(report["n_failed"], 1)
        self.assertEqual(report["first_failures"][0]["id"], "bad")

    def test_successful_context_preflight_is_cached_only_for_same_fingerprint(self):
        out = self.root / "preflight.json"
        report = context_preflight([self.item], [self.val], self.collator, self.cfg, out, "a")
        self.assertTrue(report["ok"])
        with patch.object(self, "collator", side_effect=AssertionError("cache miss")):
            self.assertEqual(context_preflight([self.item], [self.val], self.collator, self.cfg, out, "a"), report)

    def test_context_preflight_reserves_generation_and_requires_supervision(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["inference"]["max_new_tokens"] = 16
        with self.assertRaisesRegex(RuntimeError, "generation reserve"):
            context_preflight([self.item], [self.val], self.collator, cfg)
        with self.assertRaisesRegex(ValueError, "both contain"):
            context_preflight([], [self.val], self.collator, cfg)

    def test_context_preflight_checks_answer_output_budget(self):
        with self.assertRaisesRegex(RuntimeError, "output would be truncated"):
            context_preflight([dict(self.item, length=10)], [self.val], self.collator, self.cfg)

    def test_mode_and_both_cache_flags_restore_after_exception(self):
        model = Model()
        model.train()
        with self.assertRaises(ValueError):
            with validation_mode(model):
                self.assertFalse(model.training)
                self.assertTrue(model.config.text_config.use_cache)
                raise ValueError("generation broke")
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)
        self.assertFalse(model.config.text_config.use_cache)

    def test_adapter_save_archives_previous_and_incomplete_save_leaves_it_intact(self):
        path = self.root / "best_adapter"
        save_adapter_preserving_previous(Model(), self.proc, path, {"selection_score": .2})
        save_adapter_preserving_previous(Model(), self.proc, path, {"selection_score": .1})
        previous = list((self.root / "adapter_history").iterdir())
        self.assertEqual(len(previous), 1)
        self.assertEqual(json.loads((previous[0] / "val_score.json").read_text())["selection_score"], .2)
        with patch.object(Model, "save_pretrained", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "intact"):
                save_adapter_preserving_previous(Model(), self.proc, path)
        self.assertTrue(adapter_complete(path))

    def test_callback_restores_training_state_and_logs_oom_without_selecting(self):
        model = Model()
        callback = ValidationGeneration(self.proc, self.cfg, [self.val], str(self.root))
        with patch("prism.vlm.predict", side_effect=torch.cuda.OutOfMemoryError("out of memory")):
            callback.on_evaluate(None, SimpleNamespace(global_step=1, epoch=1.0), None, model=model)
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)
        self.assertEqual(callback.best, float("inf"))
        self.assertEqual(json.loads((self.root / "val_generation_log.jsonl").read_text())["validation_failed"], "cuda_oom")

    def test_callback_restores_state_on_unexpected_prediction_failure(self):
        model = Model()
        callback = ValidationGeneration(self.proc, self.cfg, [self.val], str(self.root))
        with patch("prism.vlm.predict", side_effect=ValueError("bad prediction IDs")):
            with self.assertRaisesRegex(ValueError, "bad prediction"):
                callback.on_evaluate(None, SimpleNamespace(global_step=1, epoch=1.0), None, model=model)
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)

    def test_callback_keeps_original_selection_rule_and_ignores_incomplete_best(self):
        atomic_json(self.root / "best_adapter" / "val_score.json", {"selection_score": -10})
        callback = ValidationGeneration(self.proc, self.cfg, [self.val], str(self.root), "run")
        self.assertEqual(callback.best, float("inf"))
        rows = [{"id": self.val["id"], "pred": {key: .8 for key in METRIC_KEYS}}]
        with patch("prism.vlm.predict", return_value=rows):
            callback.on_evaluate(None, SimpleNamespace(global_step=1, epoch=1.0), None, model=Model())
        self.assertAlmostEqual(callback.best, .2)
        record = json.loads((self.root / "best_adapter" / "val_score.json").read_text())
        self.assertEqual(record["run_fingerprint"], "run")
        self.assertTrue(adapter_complete(self.root / "best_adapter"))

    def test_entirely_invalid_validation_cannot_create_a_best_adapter(self):
        callback = ValidationGeneration(self.proc, self.cfg, [self.val], str(self.root), "run")
        with patch("prism.vlm.predict", return_value=[{"id": self.val["id"], "pred": None}]):
            callback.on_evaluate(None, SimpleNamespace(global_step=1, epoch=1.0), None, model=Model())
        self.assertEqual(callback.best, float("inf"))
        self.assertFalse((self.root / "best_adapter").exists())
        record = json.loads((self.root / "val_generation_log.jsonl").read_text())
        self.assertFalse(record["is_best"])
        self.assertIsNone(record["val_gen_MAE"])
        self.assertEqual(record["validation_failed"], "no_valid_predictions")

    def test_trainables_loss_and_gradients_fail_closed(self):
        model = Model()
        self.assertEqual(assert_lora_trainables(model), 1)
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            assert_finite_training(model)
        model.lora_B.grad = torch.ones(1)
        assert_finite_training(model, torch.tensor(1.0))
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            assert_finite_training(model, torch.tensor(float("nan")))
        model.lora_B.grad[:] = float("inf")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            assert_finite_training(model)
        model.lora_B.grad[:] = 0
        with self.assertRaisesRegex(RuntimeError, "zero"):
            assert_finite_training(model)
        model.other_weight = torch.nn.Parameter(torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, "unfrozen"):
            assert_lora_trainables(model)

    def test_gradient_health_includes_every_lora_parameter(self):
        model = Model()
        model.lora_A = torch.nn.Parameter(torch.ones(3))
        model.lora_B.grad = torch.zeros(1)
        model.lora_A.grad = torch.ones(3)
        assert_finite_training(model)
        model.lora_A.grad[2] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            assert_finite_training(model)
        model.lora_A.grad[:] = 0
        with self.assertRaisesRegex(RuntimeError, "zero"):
            assert_finite_training(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
