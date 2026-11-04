"""Training integrity, bounded context preflight and safe validation/checkpoint handling.

These guards preserve the training objective and validation selection criterion. They reject
incompatible resumptions and keep earlier adapters intact when publishing a new best adapter.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from numbers import Real
import random
import tempfile
import uuid

import numpy as np
import torch

from .common import abspath
from .metrics import METRIC_KEYS
from .prompts import PROMPTS, SYSTEM, TRAIN_PROMPT


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False, default=str).encode("utf-8")).hexdigest()


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with open(staged, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staged, path)


def seed_training(seed):
    """Seed model/data RNGs; avoid nondeterministic benchmark-driven kernel selection.

    warn_only permits a model's unsupported deterministic CUDA operation to run with a warning;
    this is reproducible seeding, not a promise of bitwise equality across GPU/library versions.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def validate_training_pairs(train, val):
    from .validation import validate_output
    if not train or not val:
        raise ValueError("Training and validation splits must both contain image pairs")
    for split, items in (("train", train), ("val", val)):
        ids = [item["id"] for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate pair IDs in {split} split")
        for item in items:
            scores = item.get("scores")
            if not isinstance(scores, dict) or set(scores) != set(METRIC_KEYS) or not all(
                    isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
                    and 0 <= value <= 1 for value in scores.values()):
                raise ValueError(f"{split} pair {item['id']} has invalid ground-truth scores")
            target_scores, _ = validate_output(item.get("target", ""))
            if target_scores is None or any(abs(target_scores[key] - item["scores"][key]) > 1e-4 for key in METRIC_KEYS):
                raise ValueError(f"{split} pair {item['id']} has a missing/invalid target or scores inconsistent with its target")
    overlap = {item["sample_id"] for item in train} & {item["sample_id"] for item in val}
    if overlap:
        raise ValueError(f"Original sample IDs cross train/validation splits: {sorted(overlap)[:10]}")


def disable_training_cache(model):
    for config in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if config is not None:
            config.use_cache = False


def training_identity(cfg, processor, train, val):
    """Hash actual inputs, targets, processor vocabulary/settings and the training protocol."""
    image_hashes = {}
    def records(items):
        result = []
        for item in items:
            row = dict(item)
            for key in ("original_image_path", "release_image_path"):
                path = abspath(item[key])
                if path not in image_hashes:
                    image_hashes[path] = _file_hash(path)
                row[key + "_sha256"] = image_hashes[path]
            result.append(row)
        return _digest(result)
    tokenizer = processor.tokenizer
    proc_config = processor.to_dict() if hasattr(processor, "to_dict") else {}
    # Image processors are not serialized by every Transformers ProcessorMixin version.
    for name in ("image_processor", "feature_extractor", "video_processor"):
        component = getattr(processor, name, None)
        if component is not None and hasattr(component, "to_dict"):
            proc_config[name] = component.to_dict()
    versions = {}
    for name in ("torch", "transformers", "peft", "trl", "bitsandbytes"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parents[1]
    source_hashes = {name: _file_hash(root / name) for name in (
        "prism/training.py", "prism/vlm.py", "prism/prompts.py", "prism/validation.py", "scripts/04_train_qlora.py")
        if (root / name).is_file()}
    payload = {"schema": 2, "kind": "prism-training", "model": cfg["model"], "qlora": cfg["qlora"],
               "train": cfg["train"], "seed": cfg["project_seed"], "inference": cfg["inference"],
               "processor_class": type(processor).__name__, "processor": proc_config,
               "chat_template": getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None),
               "tokenizer_class": type(tokenizer).__name__, "vocabulary": _digest(tokenizer.get_vocab()),
               "special_tokens": tokenizer.special_tokens_map,
               "system": SYSTEM, "prompts": PROMPTS, "training_prompt": TRAIN_PROMPT,
               "train_split": records(train), "val_split": records(val), "versions": versions,
               "source_sha256": source_hashes}
    return {"fingerprint": _digest(payload), "payload": payload}


def ensure_training_run(out_dir, identity):
    """Fail closed on stale/unidentified training artifacts, without overwriting user work."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "training_manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != identity["fingerprint"]:
            raise RuntimeError("Training inputs/configuration changed. Existing checkpoints are preserved; "
                               "choose a fresh paths.outputs_dir instead of resuming this run.")
    else:
        # Empty directories can remain after a disconnect before the first checkpoint write.
        existing = [p for p in root.rglob("*") if p.is_file()]
        if existing:
            raise RuntimeError("Training artifacts have no compatible training_manifest.json. "
                               "Files are preserved; choose a fresh paths.outputs_dir.")
        atomic_json(manifest, identity)
    return identity["fingerprint"]


def adapter_complete(path):
    path = Path(path)
    try:
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        return (isinstance(config, dict) and config.get("peft_type") == "LORA"
                and any((path / name).is_file() and (path / name).stat().st_size > 0
                        for name in ("adapter_model.safetensors", "adapter_model.bin")))
    except (OSError, ValueError):
        return False


def latest_checkpoint(out_dir):
    """Return the latest complete *numeric* checkpoint; ignore empty/incomplete newer writes."""
    root = Path(out_dir) / "checkpoints"
    candidates, incomplete = [], []
    for path in root.glob("checkpoint-*"):
        if not path.is_dir() or not path.name.removeprefix("checkpoint-").isdigit():
            continue
        if not any(path.iterdir()):
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
            state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
            valid = (isinstance(state["global_step"], int) and not isinstance(state["global_step"], bool)
                     and state["global_step"] == step and adapter_complete(path)
                     and all((path / name).is_file() and (path / name).stat().st_size > 0
                             for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth")))
        except (OSError, ValueError, KeyError):
            valid = False
        if valid:
            candidates.append((step, str(path)))
        else:
            incomplete.append(str(path))
    if incomplete:
        print(f"[train] preserved {len(incomplete)} incomplete checkpoint(s); using only complete optimizer/RNG states")
    if candidates:
        return max(candidates)[1]
    if incomplete:
        raise RuntimeError("No complete resumable checkpoint remains. Incomplete files are preserved; "
                           "choose a fresh outputs directory to restart training.")
    return None


def training_complete(out_dir, fingerprint):
    path = Path(out_dir) / "training_log.json"
    if not path.is_file():
        return False
    try:
        log = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (log.get("completed") is True and log.get("run_fingerprint") == fingerprint
            and adapter_complete(Path(out_dir) / "best_adapter")
            and adapter_complete(Path(out_dir) / "final_adapter"))


def context_preflight(train, val, collator, cfg, out_path=None, fingerprint=None):
    """Process every pair one at a time on CPU, including full targets and generation reserve."""
    if not train or not val:
        raise ValueError("Training and validation splits must both contain image pairs")
    if out_path and Path(out_path).exists() and fingerprint:
        report = json.loads(Path(out_path).read_text(encoding="utf-8"))
        if (report.get("schema") == 2 and report.get("fingerprint") == fingerprint and report.get("ok") is True
                and report.get("n_pairs") == len(train) + len(val)):
            print("[context] verified cached preflight for identical inputs and processor")
            return report
    maximum = cfg["model"]["max_seq_len"]
    failures, n_failed, largest, largest_generation, largest_answer = [], 0, 0, 0, 0
    for split, items in (("train", train), ("val", val)):
        for index, item in enumerate(items):
            try:
                encoded = collator([item])
                ids, labels, attention = encoded["input_ids"], encoded["labels"], encoded["attention_mask"]
                length = int(attention[0].sum())
                largest = max(largest, length)
                supervised = (labels[0] != -100).nonzero(as_tuple=False).flatten()
                if not len(supervised):
                    raise ValueError("answer has no supervised tokens")
                largest_answer = max(largest_answer, len(supervised))
                if len(supervised) > cfg["inference"]["max_new_tokens"]:
                    raise ValueError(f"answer requires {len(supervised)} output tokens but max_new_tokens="
                                     f"{cfg['inference']['max_new_tokens']}; output would be truncated")
                if bool((labels[attention == 0] != -100).any()):
                    raise ValueError("padding is incorrectly supervised")
                if bool((labels[labels != -100] < 0).any()):
                    raise ValueError("visual/negative tokens are incorrectly supervised")
                reserve = int(supervised[0]) + cfg["inference"]["max_new_tokens"]
                largest_generation = max(largest_generation, reserve)
                if length > maximum or reserve > maximum:
                    raise ValueError(f"training={length}, generation reserve={reserve}, maximum={maximum}; "
                                     "increase max_seq_len or record a lower native image budget; never truncate")
                del encoded, ids, labels, attention
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
                n_failed += 1
                if len(failures) < 20:
                    failures.append({"split": split, "id": item.get("id", index), "error": str(error)})
            if (index + 1) % 100 == 0 or index + 1 == len(items):
                print(f"[context] {split} {index + 1}/{len(items)}", flush=True)
    report = {"schema": 2, "fingerprint": fingerprint, "ok": n_failed == 0,
              "n_pairs": len(train) + len(val), "n_failed": n_failed,
              "max_training_tokens": largest, "max_generation_tokens": largest_generation,
              "max_answer_tokens": largest_answer,
              "max_seq_len": maximum, "first_failures": failures}
    if out_path:
        atomic_json(out_path, report)
    if n_failed:
        raise RuntimeError(f"Context preflight failed for {n_failed} pair(s) before model loading: {failures}")
    return report


def assert_lora_trainables(model):
    trainables = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainables or any("lora_" not in name for name, _ in trainables):
        raise RuntimeError("Expected only nonempty LoRA trainables; unexpected/unfrozen base parameters detected")
    return sum(param.numel() for _, param in trainables)


def assert_finite_training(model, loss=None):
    if loss is not None and not bool(torch.isfinite(loss.detach()).all()):
        raise RuntimeError("Non-finite training loss; stop before using this adapter")
    gradients = [param.grad for name, param in model.named_parameters()
                 if param.requires_grad and "lora_" in name and param.grad is not None]
    if not gradients or not all(bool(torch.isfinite(grad).all()) for grad in gradients):
        raise RuntimeError("Missing or non-finite LoRA gradients; optimizer update rejected")
    if not any(bool(torch.count_nonzero(grad)) for grad in gradients):
        raise RuntimeError("All LoRA gradients are zero; check answer masks and model preparation")


class TrainingHealth:
    """Mixin for TrainerCallback: inspect unscaled gradients before each optimizer update."""
    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        assert_finite_training(model)


@contextmanager
def validation_mode(model):
    training = model.training
    configs = []
    for config in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if config is not None and hasattr(config, "use_cache"):
            configs.append((config, config.use_cache))
            config.use_cache = True
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for config, value in configs:
            config.use_cache = value
        model.train(training)


def save_adapter_preserving_previous(model, processor, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".adapter-stage-", dir=path.parent))
    model.save_pretrained(str(staged))
    processor.save_pretrained(str(staged))
    if not adapter_complete(staged):
        raise RuntimeError(f"Adapter save is incomplete; previous adapter is intact, staged files at {staged}")
    if metadata is not None:
        atomic_json(staged / "val_score.json", metadata)
    archived = None
    if path.exists():
        history = path.parent / "adapter_history"
        history.mkdir(exist_ok=True)
        archived = history / f"{path.name}-{uuid.uuid4().hex}"
        os.replace(path, archived)
    try:
        os.replace(staged, path)
    except OSError:
        if archived is not None and not path.exists():
            os.replace(archived, path)
        raise


class ValidationGeneration:
    """Mixin for TrainerCallback; selection stays MAE + 0.5 * invalid-output rate, on val only."""
    def __init__(self, proc, cfg, val_items, out_dir, run_fingerprint=None):
        if not val_items:
            raise ValueError("Validation generation requires nonempty validation pairs")
        self.proc, self.cfg, self.items, self.out_dir = proc, cfg, val_items, out_dir
        self.run_fingerprint = run_fingerprint
        self.best_path = os.path.join(out_dir, "best_adapter")
        self.log_path = os.path.join(out_dir, "val_generation_log.jsonl")
        self.best = float("inf")
        meta = Path(self.best_path) / "val_score.json"
        if meta.exists() and adapter_complete(self.best_path):
            previous = json.loads(meta.read_text(encoding="utf-8"))
            if run_fingerprint and previous.get("run_fingerprint") != run_fingerprint:
                raise RuntimeError("Best adapter belongs to an incompatible training run; files preserved")
            score = previous["selection_score"]
            if not isinstance(score, Real) or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1.5:
                raise ValueError("Stored best adapter selection score is not finite")
            self.best = score

    def _log(self, record):
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        from .vlm import predict
        try:
            with validation_mode(model):
                rows = predict(model, self.proc, self.cfg, self.items, TRAIN_PROMPT, out_path=None,
                               batch_size=self.cfg["inference"]["batch_size"], log_every=10**9)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._log({"step": state.global_step, "epoch": state.epoch, "validation_failed": "cuda_oom",
                       "is_best": False, "run_fingerprint": self.run_fingerprint})
            print("[val-gen] CUDA memory exhausted even after inference fallback; kept previous best adapter")
            return
        gt = {item["id"]: item["scores"] for item in self.items}
        if len(rows) != len(self.items) or {row["id"] for row in rows} != set(gt):
            raise ValueError("Validation predictor did not return exactly the fixed validation subset")
        valid = [row for row in rows if isinstance(row.get("pred"), dict) and all(
            isinstance(row["pred"].get(key), Real) and not isinstance(row["pred"][key], bool)
            and math.isfinite(row["pred"][key]) and 0 <= row["pred"][key] <= 1 for key in METRIC_KEYS)]
        mae = float(np.mean([abs(row["pred"][key] - gt[row["id"]][key]) for row in valid for key in METRIC_KEYS])) if valid else 1.0
        rate = len(valid) / len(rows)
        score = mae + 0.5 * (1 - rate)
        if not math.isfinite(score):
            raise RuntimeError("Non-finite validation selection score; best adapter is preserved")
        record = {"step": state.global_step, "epoch": state.epoch, "val_gen_MAE": mae, "val_valid_rate": rate,
                  "selection_score": score, "is_best": score < self.best, "run_fingerprint": self.run_fingerprint}
        if score < self.best:
            save_adapter_preserving_previous(model, self.proc, self.best_path, record)
            self.best = score
        self._log(record)
        print(f"[val-gen] step {state.global_step}: MAE={mae:.4f} valid={rate:.3f} score={score:.4f}", flush=True)
