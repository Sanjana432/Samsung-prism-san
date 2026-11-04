"""Stage 4 - QLoRA supervised fine-tuning (GPU).

SmolVLM2-2.2B-Instruct in 4-bit NF4 + LoRA (r16 / alpha32 / dropout .05 / all-linear), TRL SFTTrainer,
cosine LR, warmup, gradient accumulation. Loss is computed ONLY on the assistant answer
(<analysis> CoT + JSON). Image tokens are never truncated: a sequence longer than max_seq_len aborts.

Checkpoint selection follows the validation set's *metric behaviour*, not training loss: at every
evaluation the callback generates on a fixed validation subset, runs the validation gate and computes
MAE; the adapter with the lowest (MAE + 0.5 * invalid rate) is saved to outputs/best_adapter.

Resumable: re-running continues from the last checkpoint in outputs/checkpoints.
"""

import argparse
import dataclasses
import glob
import json
import math
import os
import random
import sys

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainerCallback, set_seed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prism.common import P, abspath, load_config, read_jsonl  # noqa: E402
from prism.metrics import METRIC_KEYS  # noqa: E402
from prism.prompts import PROMPTS, TRAIN_PROMPT  # noqa: E402
from prism.vlm import VLMClass, assistant_marker, bnb_config, compute_dtype, encode_pairs, load_processor, predict  # noqa: E402

from prism.training import (ValidationGeneration, TrainingHealth, adapter_complete, assert_lora_trainables,
                            atomic_json, context_preflight, disable_training_cache, ensure_training_run,
                            latest_checkpoint, save_adapter_preserving_previous, seed_training,
                            training_complete, training_identity, validate_training_pairs)  # noqa: E402


class Collator:
    def __init__(self, proc, cfg):
        self.proc, self.cfg = proc, cfg
        self.marker = assistant_marker(proc)
        self.max_len = cfg["model"]["max_seq_len"]

    def _start(self, ids):
        m = self.marker
        for s in range(len(ids) - len(m), -1, -1):
            if ids[s:s + len(m)] == m:
                return s + len(m)
        raise ValueError("assistant marker not found")

    def __call__(self, batch):
        enc = encode_pairs(self.proc, batch, PROMPTS[TRAIN_PROMPT], [b["target"] for b in batch], padding_side="right")
        if enc["input_ids"].shape[1] > self.max_len:
            raise RuntimeError(f"sequence length {enc['input_ids'].shape[1]} > max_seq_len {self.max_len}; "
                               "raise model.max_seq_len or lower model.image_longest_edge (never truncate image tokens)")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, ids in enumerate(enc["input_ids"].tolist()):
            labels[i, : self._start(ids)] = -100
        labels[enc["input_ids"] == self.proc.tokenizer.convert_tokens_to_ids(self.proc.image_token)] = -100
        enc["labels"] = labels
        return enc


class ValGenerationCallback(ValidationGeneration, TrainerCallback):
    """Shared val-only selection with state restoration and safe adapter publication."""


class TrainingHealthCallback(TrainingHealth, TrainerCallback):
    """Reject non-finite or missing LoRA gradients before an optimizer update."""


def build_args(cls, **kw):
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in kw.items() if k in names})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max_train", type=int, default=0, help="debug: limit training pairs")
    a = ap.parse_args()
    cfg = load_config(a.config)
    from prism.dataset import assert_dataset_ready
    assert_dataset_ready(cfg)
    set_seed(cfg["project_seed"])
    seed_training(cfg["project_seed"])
    if a.max_train < 0:
        raise ValueError("--max_train must be nonnegative")
    tc, q = cfg["train"], cfg["qlora"]
    out_dir = P(cfg, "outputs_dir")
    os.makedirs(out_dir, exist_ok=True)

    train = read_jsonl(P(cfg, "pairs_dir", "train.jsonl"))
    val = read_jsonl(P(cfg, "pairs_dir", "val.jsonl"))
    if a.max_train:
        train = train[: a.max_train]
    rng = random.Random(cfg["project_seed"])
    val_sorted = sorted(val, key=lambda r: r["id"])
    rng.shuffle(val_sorted)
    val_loss = val_sorted[: tc["val_loss_subset"]]
    val_gen = val_sorted[: tc["val_generation_subset"]]

    validate_training_pairs(train, val)
    proc = load_processor(cfg, padding_side="right")
    collator = Collator(proc, cfg)
    identity = training_identity(cfg, proc, train, val)
    fingerprint = ensure_training_run(out_dir, identity)
    context_preflight(train, val, collator, cfg, os.path.join(out_dir, "context_preflight.json"), fingerprint)
    if training_complete(out_dir, fingerprint):
        print("[train] matching run already completed; preserved validated adapters")
        return
    resume = latest_checkpoint(out_dir)
    if not torch.cuda.is_available():
        raise SystemExit("Context preflight passed. A CUDA GPU is required before downloading/training model weights.")
    model = VLMClass.from_pretrained(cfg["model"]["base"], revision=cfg["model"].get("revision"), quantization_config=bnb_config(cfg), torch_dtype=compute_dtype(),
                                     device_map="auto", attn_implementation="sdpa")
    disable_training_cache(model)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=tc["gradient_checkpointing"],
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=q["r"], lora_alpha=q["alpha"], lora_dropout=q["dropout"],
                                             target_modules=q["target_modules"], bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    assert_lora_trainables(model)

    steps_per_epoch = math.ceil(len(train) / (tc["per_device_batch_size"] * tc["grad_accum"]))
    eval_steps = max(1, steps_per_epoch // tc["evals_per_epoch"])
    bf16 = compute_dtype() == torch.bfloat16
    common = dict(
        output_dir=os.path.join(out_dir, "checkpoints"), num_train_epochs=tc["epochs"],
        per_device_train_batch_size=tc["per_device_batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=tc["grad_accum"], learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler"], warmup_ratio=tc["warmup_ratio"], weight_decay=tc["weight_decay"],
        logging_steps=tc["logging_steps"], eval_strategy="steps", evaluation_strategy="steps", eval_steps=eval_steps,
        save_strategy="steps", save_steps=eval_steps, save_total_limit=tc["save_total_limit"],
        bf16=bf16, fp16=not bf16, gradient_checkpointing=tc["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False}, remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True}, report_to="none", seed=cfg["project_seed"],
        data_seed=cfg["project_seed"], logging_nan_inf_filter=False,
        dataloader_num_workers=2, optim="paged_adamw_8bit", max_grad_norm=1.0,
        max_length=None, max_seq_length=None, label_names=["labels"],
    )
    cb = ValGenerationCallback(proc, cfg, val_gen, out_dir, run_fingerprint=fingerprint)
    ds_train, ds_val = Dataset.from_list(train), Dataset.from_list(val_loss)
    try:
        from trl import SFTConfig, SFTTrainer
        args = build_args(SFTConfig, **common)
        try:
            trainer = SFTTrainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                                 data_collator=collator, processing_class=proc.tokenizer, callbacks=[cb, TrainingHealthCallback()])
        except TypeError:
            trainer = SFTTrainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                                 data_collator=collator, tokenizer=proc.tokenizer, callbacks=[cb, TrainingHealthCallback()])
        print("[train] using TRL SFTTrainer")
    except Exception as e:  # TRL API drift -> identical loop on transformers.Trainer
        print(f"[train] SFTTrainer unavailable ({type(e).__name__}: {e}); falling back to transformers.Trainer")
        from transformers import Trainer, TrainingArguments
        args = build_args(TrainingArguments, **common)
        trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                          data_collator=collator, callbacks=[cb, TrainingHealthCallback()])

    print(f"[train] {len(train)} pairs, {steps_per_epoch} optimizer steps/epoch, eval every {eval_steps} steps, "
          f"bf16={bf16}, resume={resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    trainer.evaluate()  # final validation pass (also triggers the selection callback)
    final = os.path.join(out_dir, "final_adapter")
    save_adapter_preserving_previous(model, proc, final)
    if not adapter_complete(os.path.join(out_dir, "best_adapter")):
        raise RuntimeError("Training finished but no validated best adapter was saved. Final weights and "
                           "checkpoints are preserved; fix validation memory/output failures before evaluation.")
    atomic_json(os.path.join(out_dir, "training_log.json"), {
        "log_history": trainer.state.log_history,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "torch": torch.__version__, "n_train_pairs": len(train),
        "run_fingerprint": fingerprint, "completed": True,
    })
    print(f"[train] done. best adapter -> {os.path.join(out_dir, 'best_adapter')}")


if __name__ == "__main__":
    main()
