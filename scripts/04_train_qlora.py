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


class ValGenerationCallback(TrainerCallback):
    """Checkpoint selection by validation-set metric behaviour (generation + gate + MAE)."""

    def __init__(self, proc, cfg, val_items, out_dir):
        self.proc, self.cfg, self.items, self.out_dir = proc, cfg, val_items, out_dir
        self.best_path = os.path.join(out_dir, "best_adapter")
        self.log_path = os.path.join(out_dir, "val_generation_log.jsonl")
        self.best = float("inf")
        meta = os.path.join(self.best_path, "val_score.json")
        if os.path.exists(meta):
            self.best = json.load(open(meta))["selection_score"]

    def on_evaluate(self, args, state, control, model=None, **kw):
        was_training = model.training
        model.eval()
        rows = predict(model, self.proc, self.cfg, self.items, TRAIN_PROMPT, out_path=None,
                       batch_size=self.cfg["inference"]["batch_size"], log_every=10**9)
        gt = {it["id"]: it["scores"] for it in self.items}
        valid = [r for r in rows if r["pred"]]
        mae = float(np.mean([abs(r["pred"][k] - gt[r["id"]][k]) for r in valid for k in METRIC_KEYS])) if valid else 1.0
        vr = len(valid) / max(1, len(rows))
        score = mae + 0.5 * (1 - vr)
        rec = {"step": state.global_step, "epoch": state.epoch, "val_gen_MAE": mae, "val_valid_rate": vr,
               "selection_score": score, "is_best": score < self.best}
        print(f"[val-gen] step {state.global_step}: MAE={mae:.4f} valid={vr:.2f} score={score:.4f}"
              + ("  <- new best" if score < self.best else ""), flush=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if score < self.best:
            self.best = score
            model.save_pretrained(self.best_path)
            self.proc.save_pretrained(self.best_path)
            json.dump(rec, open(os.path.join(self.best_path, "val_score.json"), "w"), indent=2)
        if was_training:
            model.train()


def build_args(cls, **kw):
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in kw.items() if k in names})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max_train", type=int, default=0, help="debug: limit training pairs")
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["project_seed"])
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

    proc = load_processor(cfg, padding_side="right")
    model = VLMClass.from_pretrained(cfg["model"]["base"], quantization_config=bnb_config(cfg), torch_dtype=compute_dtype(),
                                     device_map="auto", attn_implementation="sdpa")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=tc["gradient_checkpointing"],
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=q["r"], lora_alpha=q["alpha"], lora_dropout=q["dropout"],
                                             target_modules=q["target_modules"], bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

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
        dataloader_num_workers=2, optim="paged_adamw_8bit", max_grad_norm=1.0,
        max_length=None, max_seq_length=None, label_names=["labels"],
    )
    collator = Collator(proc, cfg)
    cb = ValGenerationCallback(proc, cfg, val_gen, out_dir)
    ds_train, ds_val = Dataset.from_list(train), Dataset.from_list(val_loss)
    try:
        from trl import SFTConfig, SFTTrainer
        args = build_args(SFTConfig, **common)
        try:
            trainer = SFTTrainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                                 data_collator=collator, processing_class=proc.tokenizer, callbacks=[cb])
        except TypeError:
            trainer = SFTTrainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                                 data_collator=collator, tokenizer=proc.tokenizer, callbacks=[cb])
        print("[train] using TRL SFTTrainer")
    except Exception as e:  # TRL API drift -> identical loop on transformers.Trainer
        print(f"[train] SFTTrainer unavailable ({type(e).__name__}: {e}); falling back to transformers.Trainer")
        from transformers import Trainer, TrainingArguments
        args = build_args(TrainingArguments, **common)
        trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                          data_collator=collator, callbacks=[cb])

    resume = bool(glob.glob(os.path.join(out_dir, "checkpoints", "checkpoint-*")))
    print(f"[train] {len(train)} pairs, {steps_per_epoch} optimizer steps/epoch, eval every {eval_steps} steps, "
          f"bf16={bf16}, resume={resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.evaluate()  # final validation pass (also triggers the selection callback)
    final = os.path.join(out_dir, "final_adapter")
    model.save_pretrained(final)
    proc.save_pretrained(final)
    if not os.path.exists(os.path.join(out_dir, "best_adapter", "adapter_config.json")):
        model.save_pretrained(os.path.join(out_dir, "best_adapter"))
        proc.save_pretrained(os.path.join(out_dir, "best_adapter"))
    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump({"log_history": trainer.state.log_history, "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
                   "torch": torch.__version__, "n_train_pairs": len(train)}, f, indent=1, default=str)
    print(f"[train] done. best adapter -> {os.path.join(out_dir, 'best_adapter')}")


if __name__ == "__main__":
    main()
