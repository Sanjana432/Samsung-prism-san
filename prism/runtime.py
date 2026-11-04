"""Crash-safe predictions with content-aware provenance and strict resume compatibility."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

from .prompts import PROMPTS, RETRY, SYSTEM
from .validation import validate_scores


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _weights_identity(path):
    if not path:
        return None
    folder = Path(path)
    if not folder.is_dir():
        raise ValueError(f"model/adapter directory does not exist: {path}")
    files = sorted(p for p in folder.iterdir() if p.suffix in (".json", ".safetensors", ".bin"))
    return {p.name: file_hash(p) for p in files}


def prediction_identity(model, cfg, items, prompt_id, max_new_tokens, processor=None):
    from .dataset import split_fingerprint
    versions = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    base = cfg["model"]["base"]
    processor_info = {"chat_template": getattr(processor, "chat_template", None)}
    for key in ("image_processor", "video_processor"):
        component = getattr(processor, key, None)
        if component is not None and hasattr(component, "to_dict"):
            processor_info[key] = component.to_dict()
    return {"schema_version": 1, "model": cfg["model"], "qlora": cfg["qlora"], "processor": processor_info,
            "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
            "local_base": _weights_identity(base) if os.path.isdir(base) else None,
            "adapter": _weights_identity(cfg.get("_adapter_path")), "versions": versions,
            "pairs_fingerprint": split_fingerprint(items, root=cfg.get("_root")),
            "prompt_id": prompt_id, "system_prompt": SYSTEM, "prompt": PROMPTS[prompt_id], "retry_prompt": RETRY,
            "max_new_tokens": max_new_tokens, "retry_max_new_tokens": cfg["inference"]["retry_max_new_tokens"],
            "backend": file_hash(Path(__file__).with_name("vlm.py")),
            "validator": file_hash(Path(__file__).with_name("validation.py"))}


def prepare_prediction_run(model, cfg, items, prompt_id, max_new_tokens, out_path, processor=None):
    expected = {item["id"]: item for item in items}
    if len(expected) != len(items):
        raise ValueError("duplicate pair IDs in inference inputs")
    if not out_path:
        return []
    path = Path(out_path)
    sidecar = Path(str(path) + ".manifest.json")
    identity = prediction_identity(model, cfg, items, prompt_id, max_new_tokens, processor)
    fingerprint = digest(identity)
    if sidecar.exists():
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Prediction resume refused: model, adapter, data, prompts or settings changed. "
                             "Use a new output name or archive the previous run.")
    elif path.exists() and path.stat().st_size:
        raise ValueError("Prediction resume refused: existing output has no provenance manifest. "
                         "Archive legacy predictions or use a new name.")
    else:
        atomic_json(sidecar, {"fingerprint": fingerprint, "identity": identity})
    rows, seen = [], set()
    if not path.exists():
        return rows
    data = path.read_bytes()
    consumed = 0
    for index, line in enumerate(data.splitlines(keepends=True)):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if consumed + len(line) == len(data) and not line.endswith(b"\n"):
                # Only an interrupted final append is recoverable; never hide corruption in the middle.
                with open(path, "r+b") as stream:
                    stream.truncate(consumed)
                break
            raise ValueError(f"corrupt prediction JSONL at line {index + 1}: {path}")
        pair_id = row.get("id")
        if pair_id not in expected or pair_id in seen:
            raise ValueError(f"unknown/duplicate prediction ID: {pair_id}")
        if row.get("pred") is not None and validate_scores(row["pred"])[0] is None:
            raise ValueError(f"invalid saved scores for {pair_id}")
        if any(row.get(k) != expected[pair_id].get(k) for k in ("sample_id", "release_id")):
            raise ValueError(f"prediction identity mismatch for {pair_id}")
        seen.add(pair_id)
        rows.append(row)
        consumed += len(line)
    # A complete final JSON object without a newline is valid, but the next append needs a separator.
    if path.stat().st_size:
        with open(path, "rb+") as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
    return rows
