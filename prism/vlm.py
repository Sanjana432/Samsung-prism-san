"""Model loading + batched CoT inference with the validation gate (one retry on malformed output).
GPU required. Used by the prompt study/baseline (03), the training callback (04) and inference (05)."""

from __future__ import annotations

import json
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, SmolVLMForConditionalGeneration as VLMClass

from .common import abspath
from .prompts import PROMPTS, RETRY, build_messages
from .validation import validate_output


def compute_dtype():
    """bf16 only on GPUs with native bf16 (Ampere, compute capability >= 8: A100, L4, A10, RTX 30xx+).
    torch.cuda.is_bf16_supported() also returns True on a T4 (sm_75) where bf16 is only emulated, which made
    Kaggle T4 training ~688 s per optimizer step; T4 / P100 / V100 therefore use native fp16."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_processor(cfg, padding_side="left"):
    m = cfg["model"]
    proc = AutoProcessor.from_pretrained(m["base"], revision=m.get("revision"))
    # Set both copies: SmolVLM caches the image size at construction. Passing `size` to
    # AutoProcessor also puts it in tokenizer init_kwargs in 4.51.3, causing duplicate kwargs.
    proc.image_processor.size = {"longest_edge": m["image_longest_edge"]}
    proc.image_size = proc.image_processor.size
    proc.tokenizer.padding_side = padding_side
    return proc


def assistant_marker(proc):
    """Derive the answer header from this checkpoint's template, rather than a Qwen token literal."""
    conversation = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    before = proc.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    after = proc.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    if not after.startswith(before):
        raise ValueError("chat template does not append an assistant generation header")
    marker = proc.tokenizer.encode(after[len(before):], add_special_tokens=False)
    if not marker:
        raise ValueError("chat template has no assistant generation header")
    return marker


def encode_pairs(proc, items, prompt, targets=None, padding_side="left"):
    """Keep each Original/Release together: SmolVLM expects a nested image list per conversation.

    Prompt wording, image order and training targets are the same as in the Qwen experiment.
    Only the checkpoint's chat template and native image processor change.
    """
    if targets is not None and len(targets) != len(items):
        raise ValueError("one training target is required per image pair")
    messages, images = [], []
    for i, item in enumerate(items):
        original = abspath(item["original_image_path"])
        release = abspath(item["release_image_path"])
        messages.append(build_messages(original, release, prompt, None if targets is None else targets[i]))
        pair = []
        for path in (original, release):
            with Image.open(path) as image:
                pair.append(image.convert("RGB"))
        images.append(pair)
    texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=targets is None) for m in messages]
    old = proc.tokenizer.padding_side
    try:
        proc.tokenizer.padding_side = padding_side
        return proc(text=texts, images=images, padding=True, truncation=False, return_tensors="pt")
    finally:
        proc.tokenizer.padding_side = old


def bnb_config(cfg):
    q = cfg["qlora"]
    return BitsAndBytesConfig(load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["quant_type"],
                              bnb_4bit_use_double_quant=q["double_quant"], bnb_4bit_compute_dtype=compute_dtype())


def load_model(cfg, adapter: str | None = None):
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for model stages (03/04/05). See README - 'No GPU?'.")
    model = VLMClass.from_pretrained(cfg["model"]["base"], revision=cfg["model"].get("revision"), quantization_config=bnb_config(cfg),
                                     torch_dtype=compute_dtype(), device_map="auto", attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


@torch.no_grad()
def generate(model, proc, cfg, items: list, prompt: str, max_new_tokens: int) -> list:
    inputs = encode_pairs(proc, items, prompt).to(model.device, dtype=compute_dtype())
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                         top_k=None, use_cache=True, pad_token_id=proc.tokenizer.pad_token_id)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def predict(model, proc, cfg, items: list, prompt_id: str, out_path: str | None = None,
            batch_size: int | None = None, max_new_tokens: int | None = None, log_every: int = 20) -> list:
    """Validated inference with content-aware resume, one retry and crash-safe appends."""
    from .runtime import prepare_prediction_run
    inf = cfg["inference"]
    bs = batch_size or inf["batch_size"]
    if bs < 1:
        raise ValueError("batch_size must be positive")
    mnt = max_new_tokens or (inf["max_new_tokens"] if prompt_id == "P4" else 256)
    saved = prepare_prediction_run(model, cfg, items, prompt_id, mnt, out_path, processor=proc)
    done = {row["id"]: row for row in saved}
    todo = [item for item in items if item["id"] not in done]
    fh = open(out_path, "a", encoding="utf-8") if out_path else None
    t0 = time.time()
    try:
        for i in range(0, len(todo), bs):
            batch = todo[i:i + bs]
            try:
                raws = generate(model, proc, cfg, batch, PROMPTS[prompt_id], mnt)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raws = [generate(model, proc, cfg, [it], PROMPTS[prompt_id], mnt)[0] for it in batch]
            if len(raws) != len(batch):
                raise RuntimeError("model returned a different number of answers than input pairs")
            for it, raw in zip(batch, raws):
                scores, errors = validate_output(raw)
                row = {"id": it["id"], "sample_id": it["sample_id"], "release_id": it["release_id"],
                       "prompt_id": prompt_id, "pred": scores, "raw": raw, "effective_raw": raw,
                       "retried": False, "errors": errors}
                if scores is None:
                    retry = generate(model, proc, cfg, [it], RETRY, inf["retry_max_new_tokens"])[0]
                    scores2, errors2 = validate_output(retry)
                    row.update(pred=scores2, retried=True, retry_raw=retry, effective_raw=retry,
                               errors=errors + ["retry:"] + errors2)
                done[it["id"]] = row
                if fh:
                    fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            n = i + len(batch)
            if n % log_every < bs or n == len(todo):
                elapsed = time.time() - t0
                print(f"  [{prompt_id}] {n}/{len(todo)}  {elapsed / n:.1f}s/pair  "
                      f"eta {elapsed / n * (len(todo) - n) / 60:.0f} min", flush=True)
    finally:
        if fh:
            fh.close()
    return [done[item["id"]] for item in items]
