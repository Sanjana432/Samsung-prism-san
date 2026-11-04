"""Stage 2 - Leakage-free split (grouped by sample_id, stratified by template) + pairwise JSONL.

Every pairwise record = Original + ONE release, the P4 prompt id, and the CoT+JSON training target.
The dataset is the "24 generation" ladder (data/metadata/ladder.jsonl): the control and all 24 releases of
the same Original always land in the same split (asserted).

Outputs: data/pairs/{train,val,test}.jsonl, data/pairs/split_manifest.json
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prism.common import P, load_config, read_jsonl  # noqa: E402
from prism.dataset import atomic_json, atomic_jsonl, canonical_hash, split_fingerprint, validate_pairs, validate_samples  # noqa: E402
from prism.metrics import METRIC_KEYS  # noqa: E402
from prism.prompts import TRAIN_PROMPT  # noqa: E402


def assign_splits(samples, fr, rng, holdout=()):
    """Grouped by sample_id, stratified by template. Templates in `holdout` go entirely to test, so the
    test split contains document types the model has never seen."""
    if any(not 0 < fr[k] < 1 for k in ("train", "val", "test")) or abs(sum(fr[k] for k in ("train", "val", "test")) - 1) > 1e-8:
        raise ValueError("train/val/test fractions must be positive and sum to 1")
    if len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("duplicate sample IDs must be resolved before assigning splits")
    by_t = defaultdict(list)
    for s in samples:
        by_t[s["template"]].append(s["sample_id"])
    assign = {}
    for t, ids in sorted(by_t.items()):
        if t in holdout:
            assign.update({sid: "test" for sid in ids})
            continue
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_va = max(1, round(n * fr["val"])) if n >= 3 else 0
        n_te = max(1, round(n * fr["test"])) if n >= 2 else 0
        n_tr = n - n_va - n_te
        for j, sid in enumerate(ids):
            assign[sid] = "train" if j < n_tr else ("val" if j < n_tr + n_va else "test")
    return assign


def to_pairs(samples, assign):
    pairs = {"train": [], "val": [], "test": []}
    for s in samples:
        sp = assign[s["sample_id"]]
        for rel in s["releases"]:
            target = ("<analysis>\n" + rel["reasoning"] + "\n</analysis>\n"
                      + json.dumps({k: rel["scores"][k] for k in METRIC_KEYS}))
            pairs[sp].append({
                "id": f"{s['sample_id']}_{rel['release_id']}",
                "sample_id": s["sample_id"],
                "release_id": rel["release_id"],
                "kind": s.get("kind", "ladder"),
                "template": s["template"],
                "original_image_path": s["original_image_path"],
                "release_image_path": rel["release_image_path"],
                "prompt_id": TRAIN_PROMPT,
                "target": target,
                "scores": rel["scores"],
                "artifact_type": rel["artifact_type"],
                "level": rel["level"],
                "artifact_value": rel["artifact_value"],
                "primary_metric": rel["primary_metric"],
            })
    return pairs


def _manifest(pairs):
    return {sp: {"n_samples": len({r["sample_id"] for r in rows}), "n_pairs": len(rows),
                 "templates": {t: len({r["sample_id"] for r in rows if r["template"] == t})
                               for t in sorted({r["template"] for r in rows})},
                 "sample_ids": sorted({r["sample_id"] for r in rows})} for sp, rows in pairs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--allow-empty-splits", action="store_true", help="CPU pilot only: permit a split with no Originals")
    a = ap.parse_args()
    cfg = load_config(a.config)
    fr = cfg["splits"]
    samples = read_jsonl(P(cfg, "metadata_dir", "ladder.jsonl"))
    holdout = set(fr.get("holdout_templates") or [])
    manifest_path = P(cfg, "pairs_dir", "split_manifest.json")
    previous = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            previous = json.load(f)
    policy = {"project_seed": cfg["project_seed"], "splits": fr}
    if previous.get("split_policy") not in (None, policy):
        raise ValueError("split policy changed; preserve the old run and use a new dataset directory")
    assign = assign_splits(samples, fr, random.Random(cfg["project_seed"]), holdout)
    # Existing Original assignments are frozen, including when new train-only engine rounds arrive.
    prior_assign = {}
    for sp in ("train", "val", "test"):
        for sid in previous.get(sp, {}).get("sample_ids", []):
            if sid in prior_assign:
                raise ValueError(f"existing manifest already leaks sample {sid}")
            prior_assign[sid] = sp
    assign.update({sid: prior_assign[sid] for sid in assign if sid in prior_assign})
    validate_samples(samples, root=cfg["_root"])
    pairs = to_pairs(samples, assign)
    base_split_fingerprints = {sp: split_fingerprint(rows, root=cfg["_root"]) for sp, rows in pairs.items()}
    base_fingerprint = canonical_hash(base_split_fingerprints)
    if previous.get("base_dataset_fingerprint") not in (None, base_fingerprint):
        raise ValueError("base dataset/images/labels changed; preserve the old splits and use a new dataset directory")
    random.Random(cfg["project_seed"] + 2).shuffle(pairs["train"])

    # data-engine rounds (08_data_engine.py): new documents aimed at the model's blind spots -> training only
    engine_path = P(cfg, "metadata_dir", "ladder_engine.jsonl")
    n_engine = 0
    if os.path.exists(engine_path):
        engine = read_jsonl(engine_path)
        for sample in engine:
            source = sample.get("engine_source", {})
            if source.get("split") != "val" or source.get("schema_version") != 1 or not source.get("report_sha256"):
                raise ValueError("engine data lacks validation-only provenance; test/unknown-source adaptation cannot enter training")
            if source.get("split_fingerprint") != base_split_fingerprints["val"]:
                raise ValueError("engine data was derived from a different validation split; use an intact matching dataset")
        validate_samples(samples + engine, root=cfg["_root"])
        epairs = to_pairs(engine, {s["sample_id"]: "train" for s in engine})["train"]
        pairs["train"] += epairs
        n_engine = len(epairs)
        random.Random(cfg["project_seed"] + 3).shuffle(pairs["train"])

    # Validate everything BEFORE any output file is changed (assertions must not disappear under python -O).
    validate_pairs(pairs, holdout=holdout, allow_empty=a.allow_empty_splits)
    if set(prior_assign) - {r["sample_id"] for rows in pairs.values() for r in rows}:
        raise ValueError("dataset would drop previously published samples; use a new dataset directory")
    manifest = _manifest(pairs)
    manifest["holdout_templates"] = sorted(holdout)
    manifest["data_engine_train_pairs"] = n_engine
    manifest["split_policy"] = policy
    manifest["base_dataset_fingerprint"] = base_fingerprint
    manifest["split_fingerprints"] = {sp: split_fingerprint(rows, root=cfg["_root"]) for sp, rows in pairs.items()}
    manifest["dataset_fingerprint"] = canonical_hash(manifest["split_fingerprints"])
    manifest["label_source"] = "synthetic renderer measurements; not human annotations"
    manifest["leakage_check"] = "passed: every sample_id appears in exactly one split"
    # Publish each file atomically, then the content-fingerprinted manifest as completion marker.
    pending = P(cfg, "pairs_dir", ".publishing.json")
    atomic_json(pending, {"dataset_fingerprint": manifest["dataset_fingerprint"]})
    for sp, rows in pairs.items():
        atomic_jsonl(P(cfg, "pairs_dir", f"{sp}.jsonl"), rows)
    atomic_json(manifest_path, manifest)
    os.unlink(pending)
    print({sp: (manifest[sp]["n_samples"], manifest[sp]["n_pairs"]) for sp in pairs}, manifest["leakage_check"])


if __name__ == "__main__":
    main()
