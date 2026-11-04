"""Shared helpers: config loading, project paths, JSONL io."""

from __future__ import annotations

import json
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# SmolVLM2 processor/tokenizer files. Weights may be a single safetensors file or indexed shards.
MODEL_FILES = ["config.json", "preprocessor_config.json", "processor_config.json",
               "tokenizer.json", "tokenizer_config.json"]


def load_config(path: str | None = None) -> dict:
    path = path or os.path.join(ROOT, "configs", "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = ROOT
    local = cfg["model"].get("local_dir")
    if local:
        local = os.path.expanduser(local)
        local = local if os.path.isabs(local) else os.path.join(ROOT, local)
        if os.path.isdir(local):  # a downloaded copy is used instead of the Hub (no internet needed)
            check_model_dir(local)
            cfg["model"]["base"] = local
    return cfg


def check_model_dir(d: str) -> None:
    """Fail early, naming the missing files, if a manually downloaded model folder is incomplete."""
    missing = [f for f in MODEL_FILES if not os.path.isfile(os.path.join(d, f))]
    config = os.path.join(d, "config.json")
    if os.path.isfile(config):
        with open(config) as f:
            if json.load(f).get("model_type") != "smolvlm":
                raise SystemExit(f"model folder {d} is not a SmolVLM checkpoint; use SmolVLM2-2.2B-Instruct")
    template = os.path.join(d, "chat_template.json")
    tokenizer_config = os.path.join(d, "tokenizer_config.json")
    if not os.path.isfile(template) and os.path.isfile(tokenizer_config):
        with open(tokenizer_config) as f:
            if not json.load(f).get("chat_template"):
                missing.append("chat_template.json (or chat_template in tokenizer_config.json)")
    index = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        missing += [s for s in shards if not os.path.isfile(os.path.join(d, s))]
    elif not os.path.isfile(os.path.join(d, "model.safetensors")):
        missing.append("model.safetensors (or model.safetensors.index.json and its shards)")
    if missing:
        raise SystemExit(f"model folder {d} is incomplete - missing: {', '.join(missing)}")


def P(cfg, key, *parts) -> str:
    """Absolute path for a configured directory (+ optional sub-parts)."""
    return os.path.join(ROOT, cfg["paths"][key], *parts)


def abspath(rel: str) -> str:
    rel = rel.replace("\\", "/")  # metadata written on Windows may hold backslashes
    return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(ROOT, rel))


def read_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
