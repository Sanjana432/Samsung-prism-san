"""Model loading + batched CoT inference with the validation gate (one retry on malformed output).
GPU required. Used by the prompt study/baseline (03), the training callback (04) and inference (05)."""

from __future__ import annotations

import json
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.feature_extraction_utils import BatchFeature

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
    proc = AutoProcessor.from_pretrained(m["base"], revision=m["revision"], code_revision=m["revision"],
                                         trust_remote_code=True, num_crops=m["num_crops"])
    proc.tokenizer.padding_side = padding_side
    return proc


def assistant_marker(proc):
    """Derive the answer header from this checkpoint's template, rather than a Qwen token literal."""
    conversation = [{"role": "user", "content": ""}]
    before = proc.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    after = proc.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    if not after.startswith(before):
        raise ValueError("chat template does not append an assistant generation header")
    marker = proc.tokenizer.encode(after[len(before):], add_special_tokens=False)
    if not marker:
        raise ValueError("chat template has no assistant generation header")
    return marker


def encode_pairs(proc, items, prompt, targets=None, padding_side="left"):
    """Encode each pair with Phi's processor, then pad text and concatenate vision features.

    The official processor accepts one prompt and uses negative image IDs. Keep those IDs and
    attention-mask ones intact; replacing them with padding would hide the comparison images.
    """
    if targets is not None and len(targets) != len(items):
        raise ValueError("one training target is required per image pair")
    encoded = []
    for i, item in enumerate(items):
        original = abspath(item["original_image_path"])
        release = abspath(item["release_image_path"])
        messages = build_messages(original, release, prompt, None if targets is None else targets[i])
        native_messages, image_number = [], 0
        for message in messages:
            content = []
            for block in message["content"]:
                if block["type"] == "image":
                    image_number += 1
                    content.append(f"<|image_{image_number}|>")
                else:
                    content.append(block["text"])
            native_messages.append({"role": message["role"], "content": "".join(content)})
        pair = []
        for path in (original, release):
            with Image.open(path) as image:
                pair.append(image.convert("RGB"))
        text = proc.tokenizer.apply_chat_template(native_messages, tokenize=False, add_generation_prompt=targets is None)
        encoded.append(proc(text, images=pair, padding=False, truncation=False, return_tensors="pt"))
    old = proc.tokenizer.padding_side
    try:
        proc.tokenizer.padding_side = padding_side
        text_inputs = proc.tokenizer.pad(
            [{"input_ids": e["input_ids"][0].tolist(), "attention_mask": e["attention_mask"][0].tolist()} for e in encoded],
            padding=True, return_tensors="pt")
        return BatchFeature(data={**text_inputs,
                                  "pixel_values": torch.cat([e["pixel_values"] for e in encoded], dim=0),
                                  "image_sizes": torch.cat([torch.as_tensor(e["image_sizes"]) for e in encoded], dim=0)})
    finally:
        proc.tokenizer.padding_side = old


def bnb_config(cfg):
    q = cfg["qlora"]
    return BitsAndBytesConfig(load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["quant_type"],
                              bnb_4bit_use_double_quant=q["double_quant"], bnb_4bit_compute_dtype=compute_dtype())


def load_base_model(cfg):
    """One loading path for inference, training and the GPU smoke test."""
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for model stages (03/04/05). See README - 'No GPU?'.")
    m = cfg["model"]
    return AutoModelForCausalLM.from_pretrained(m["base"], revision=m["revision"], code_revision=m["revision"],
                                               trust_remote_code=True, quantization_config=bnb_config(cfg),
                                               torch_dtype=compute_dtype(), device_map="auto", attn_implementation="eager")


def load_model(cfg, adapter: str | None = None):
    model = load_base_model(cfg)
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
    """Run inference + validation gate + one retry. Appends to out_path and resumes if it exists."""
    inf = cfg["inference"]
    bs = batch_size or inf["batch_size"]
    mnt = max_new_tokens or (inf["max_new_tokens"] if prompt_id == "P4" else 256)
    done = {}
    if out_path and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["id"]] = r
    todo = [it for it in items if it["id"] not in done]
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fh = open(out_path, "a") if out_path else None
    rows, t0 = list(done.values()), time.time()
    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        try:
            raws = generate(model, proc, cfg, batch, PROMPTS[prompt_id], mnt)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raws = [generate(model, proc, cfg, [it], PROMPTS[prompt_id], mnt)[0] for it in batch]
        for it, raw in zip(batch, raws):
            scores, errs = validate_output(raw)
            row = {"id": it["id"], "sample_id": it["sample_id"], "release_id": it["release_id"], "prompt_id": prompt_id,
                   "pred": scores, "raw": raw, "retried": False, "errors": errs}
            if scores is None:  # one retry with the strict JSON-only prompt
                raw2 = generate(model, proc, cfg, [it], RETRY, inf["retry_max_new_tokens"])[0]
                scores2, errs2 = validate_output(raw2)
                row.update({"pred": scores2, "retried": True, "retry_raw": raw2, "errors": errs + ["retry:"] + errs2})
            rows.append(row)
            if fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
        n = i + len(batch)
        if n % log_every < bs or n == len(todo):
            el = time.time() - t0
            print(f"  [{prompt_id}] {n}/{len(todo)}  {el / n:.1f}s/pair  eta {el / n * (len(todo) - n) / 60:.0f} min", flush=True)
    if fh:
        fh.close()
    return rows
