"""Dataset integrity, portable content fingerprints and atomic generated-data writes.

Renderer scores are synthetic measurement labels, not human judgments or model accuracy.
This module deliberately has no model/GPU dependency.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import copy
import uuid
from importlib.metadata import version

from PIL import Image, ImageChops

from .common import ROOT
from .metrics import METRIC_KEYS


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_path(path, root=None):
    path = Path(str(path).replace("\\", "/"))
    return path if path.is_absolute() else Path(root or ROOT) / path


def _portable(value):
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items() if k != "created_at"}
    if isinstance(value, list):
        return [_portable(v) for v in value]
    return value


def split_fingerprint(rows, root=None):
    """Hash full records and actual image bytes; tolerate path-free unit-test mock rows.

    A supplied missing image is always an error. This fingerprint is not a schema validator.
    Relative image paths make copied shared datasets portable between model workspaces.
    """
    records = sorted((_portable(r) for r in rows), key=lambda r: (r.get("id", ""), canonical_hash(r)))
    paths = sorted({str(r[k]).replace("\\", "/") for r in rows
                    for k in ("original_image_path", "release_image_path") if k in r})
    images = {p: file_sha256(image_path(p, root)) for p in paths}
    return canonical_hash({"records": records, "images": images})


def dataset_fingerprint(cfg):
    root = cfg.get("_root", ROOT)
    base = Path(root) / cfg["paths"]["pairs_dir"]
    parts = {}
    for split in ("train", "val", "test"):
        with open(base / f"{split}.jsonl", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        parts[split] = split_fingerprint(rows, root=root)
    return canonical_hash(parts)


def assert_dataset_ready(cfg):
    """Fail before model loading if publication was interrupted or published data changed."""
    base = Path(cfg.get("_root", ROOT)) / cfg["paths"]["pairs_dir"]
    if (base / ".publishing.json").exists():
        raise ValueError("dataset publication was interrupted; rerun scripts/02_build_splits.py before model stages")
    manifest_path = base / "split_manifest.json"
    if not manifest_path.exists():
        raise ValueError("dataset manifest missing; run scripts/02_build_splits.py before model stages")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("dataset_fingerprint"):
        raise ValueError("dataset has no integrity fingerprint; validate existing team data with scripts/02_build_splits.py")
    current = dataset_fingerprint(cfg)
    if manifest["dataset_fingerprint"] != current:
        raise ValueError("published dataset/image content no longer matches its manifest; use an intact shared copy or rerun stage 02")
    return current


def atomic_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_jsonl(path, rows):
    _atomic_text(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=".prism-", delete=False) as handle:
            name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def validate_scores(scores, context="scores"):
    if not isinstance(scores, dict) or set(scores) != set(METRIC_KEYS):
        raise ValueError(f"{context}: exactly the 12 canonical metrics are required")
    for key, value in scores.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{context}.{key}: expected a finite number in [0, 1]")


def validate_samples(samples, root=None, verify_images=True):
    """Check IDs, complete ladders, scores and rendered image locality before publishing data."""
    seen, paths, originals = set(), set(), {}
    n_pairs = 0
    for sample in samples:
        sid = sample.get("sample_id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError(f"duplicate or invalid sample_id: {sid!r}")
        seen.add(sid)
        releases = sample.get("releases", [])
        knobs = sample.get("knobs", [])
        expected = {("control", 0)} | {(k, level) for k in knobs for level in range(1, 5)}
        settings, ids = set(), set()
        original_name = sample.get("original_image_path")
        if not original_name or original_name in paths:
            raise ValueError(f"{sid}: original_image_path is required")
        paths.add(original_name)
        original = None
        if verify_images:
            p = image_path(original_name, root)
            with Image.open(p) as image:
                original = image.convert("RGB")
            digest = hashlib.sha256(str(original.size).encode() + original.tobytes()).hexdigest()
            if digest in originals:
                raise ValueError(f"identical Original pixels under different sample IDs: {originals[digest]} and {sid}")
            originals[digest] = sid
            if list(original.size) != sample.get("image_size"):
                raise ValueError(f"{sid}: Original dimensions do not match metadata")
        for release in releases:
            rid = release.get("release_id")
            if not rid or rid in ids:
                raise ValueError(f"{sid}: duplicate or missing release_id {rid!r}")
            ids.add(rid)
            setting = (release.get("artifact_type"), release.get("level"))
            if setting in settings:
                raise ValueError(f"{sid}: duplicate artifact/level {setting}")
            settings.add(setting)
            validate_scores(release.get("scores"), f"{sid}/{rid}")
            name = release.get("release_image_path")
            if not name or name in paths or name == original_name:
                raise ValueError(f"{sid}/{rid}: missing, reused or Original-overwriting image path")
            paths.add(name)
            audit = release.get("render_audit", {})
            if audit.get("changed_pixels_outside_region") != 0:
                raise ValueError(f"{sid}/{rid}: renderer reports changes outside the allowed region")
            if verify_images:
                with Image.open(image_path(name, root)) as image:
                    candidate = image.convert("RGB")
                if candidate.size != original.size:
                    raise ValueError(f"{sid}/{rid}: image dimensions differ")
                difference = ImageChops.difference(original, candidate)
                bbox = difference.getbbox()
                if setting == ("control", 0):
                    if bbox is not None:
                        raise ValueError(f"{sid}/{rid}: unchanged control differs from Original")
                else:
                    if bbox is None:
                        raise ValueError(f"{sid}/{rid}: edited release has no visible pixel change")
                    region = release.get("control_parameters", {}).get("local", {}).get("region_bbox")
                    if not region or len(region) != 4:
                        raise ValueError(f"{sid}/{rid}: region_bbox missing for locality audit")
                    x0, y0, x1, y1 = region
                    allowed = (max(0, x0 // 8 * 8 - 8), max(0, y0 // 8 * 8 - 8),
                               min(original.width, x1 // 8 * 8 + 16), min(original.height, y1 // 8 * 8 + 16))
                    if not (allowed[0] <= bbox[0] < bbox[2] <= allowed[2] and allowed[1] <= bbox[1] < bbox[3] <= allowed[3]):
                        raise ValueError(f"{sid}/{rid}: actual changed pixels extend outside the edited region")
            n_pairs += 1
        if not knobs or settings != expected:
            raise ValueError(f"{sid}: incomplete ladder; expected control + 4 levels per active knob")
    if not samples:
        raise ValueError("dataset has no Original samples")
    return {"n_samples": len(samples), "n_pairs": n_pairs, "n_images": len(samples) + n_pairs,
            "label_source": "synthetic renderer measurements; not human annotations"}


def validate_pairs(pairs, holdout=(), allow_empty=False):
    ids, membership = set(), {}
    for split in ("train", "val", "test"):
        rows = pairs.get(split, [])
        if not rows and not allow_empty:
            raise ValueError(f"{split} split is empty; use more Originals per template, or --allow-empty-splits for a CPU pilot")
        for row in rows:
            rid, sid = row.get("id"), row.get("sample_id")
            if not rid or rid in ids:
                raise ValueError(f"duplicate or missing pair id: {rid!r}")
            ids.add(rid)
            if membership.setdefault(sid, split) != split:
                raise ValueError(f"LEAKAGE: {sid} appears in multiple splits")
            if split != "test" and row.get("template") in holdout:
                raise ValueError(f"holdout template leaked into {split}")
            if not row.get("original_image_path") or not row.get("release_image_path"):
                raise ValueError(f"{rid}: both image paths are required")
            validate_scores(row.get("scores"), str(rid))
    return membership


def generation_fingerprint(cfg):
    """Only renderer inputs/code/fonts: independent of model, machine path and requested sample count."""
    root = Path(cfg.get("_root", ROOT))
    settings = {"seed": cfg["project_seed"], "dataset": cfg["dataset"],
                "ladder": {k: v for k, v in cfg["ladder"].items() if k != "n_originals"},
                "scoring": cfg["scoring"]}
    settings["dataset"] = {k: v for k, v in cfg["dataset"].items() if k != "workers"}
    sources = {f"prism/{name}.py": hashlib.sha256((root / "prism" / f"{name}.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in
               ("ladder", "render", "templates", "document", "objects", "metrics", "reasoning", "pixel_evidence")}
    fonts = {p.name: file_sha256(p) for p in sorted((root / "assets/fonts").glob("*.ttf"))}
    libraries = {name: version(name) for name in ("Pillow", "numpy", "scikit-image")}
    return canonical_hash({"settings": settings, "renderer": sources, "fonts": fonts, "libraries": libraries})


def render_sample_once(cfg, index, sample_id=None):
    """Stage a whole sample with its recovery sidecar, then publish without overwriting images."""
    from .ladder import make_ladder_sample
    sid = sample_id or f"ladder_{index:04d}"
    if not sid or Path(sid).name != sid or "/" in sid or "\\" in sid:
        raise ValueError("sample_id must be a single safe directory name")
    fingerprint = cfg.get("_generation_fingerprint") or generation_fingerprint(cfg)
    root, image_dir = Path(cfg.get("_root", ROOT)), cfg["paths"]["images_dir"]
    destination = root / image_dir / sid
    if destination.exists():
        sidecar = destination / ".prism_sample.json"
        if not sidecar.exists():
            raise ValueError(f"{destination} exists without recovery metadata; move it aside or use a new dataset directory")
        with open(sidecar, encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved["generation_fingerprint"] != fingerprint:
            raise ValueError(f"{sid}: generation settings/code/fonts differ from the existing sample")
        validate_samples([saved["sample"]], root=root)
        return saved["sample"]
    staged_cfg = copy.deepcopy(cfg)
    staging = f"{image_dir}/.pending/{uuid.uuid4().hex}"
    staged_cfg["paths"]["images_dir"] = staging
    record = make_ladder_sample(staged_cfg, index)
    validate_samples([record], root=root)
    old_sid = record["sample_id"]
    old_prefix, new_prefix = f"{staging}/{old_sid}", f"{image_dir}/{sid}"
    def rebase(value):
        if isinstance(value, dict):
            return {k: rebase(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rebase(v) for v in value]
        return new_prefix + value[len(old_prefix):] if isinstance(value, str) and value.startswith(old_prefix) else value
    record = rebase(record)
    record["sample_id"] = sid
    staged = root / staging / old_sid
    atomic_json(staged / ".prism_sample.json", {"generation_fingerprint": fingerprint, "sample": record})
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, destination)
    return record
