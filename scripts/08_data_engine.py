"""Stage 8 - Self-improving data engine: generate more training data exactly where the model is still blind.

Reads the ladder sensitivity analysis of a model (reports/evaluation_results.json from 06_evaluate.py on
--split val), finds the edit families whose just-noticeable level is not L1 (or that
are never detected), and generates new ladder Originals that contain ONLY those edit families. They are
written to data/metadata/ladder_engine.jsonl and 02_build_splits.py adds them to the TRAINING split only
(they are new documents, so val/test stay untouched). Retrain, re-evaluate, repeat - every round focuses
the data on the model's current blind spots. All labels are measured by the renderer; nothing is manual.

Usage: python scripts/08_data_engine.py --model finetuned_qlora [--results reports/evaluation_results.json]
                                        [--n 42] [--round 1] [--workers 0]
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prism.common import P, load_config, read_jsonl  # noqa: E402
from prism.dataset import (atomic_jsonl, canonical_hash, dataset_fingerprint, file_sha256, generation_fingerprint,
                           render_sample_once, split_fingerprint, validate_samples)  # noqa: E402
from prism.ladder import ORDINAL_KNOBS  # noqa: E402

CFG = None


def _init(cfg):
    global CFG
    CFG = cfg


def _make(i):
    return render_sample_once(CFG, CFG["_engine_start"] + i, f"engine_{CFG['_engine_key']}_{i:04d}")


def validation_source(path, model, cfg):
    """Adaptive training may use validation results only, with matching content provenance."""
    sidecar = os.path.join(os.path.dirname(path), "evaluation_provenance.json")
    if not os.path.exists(sidecar):
        raise ValueError("evaluation provenance is missing; run 06_evaluate.py --split val in the modified pipeline")
    with open(sidecar, encoding="utf-8") as handle:
        source = json.load(handle)
    if source.get("schema_version") != 1 or source.get("split") != "val" or model not in source.get("models", []):
        raise ValueError("data engine accepts only explicit validation-split provenance for the selected model; never test results")
    val = read_jsonl(P(cfg, "pairs_dir", "val.jsonl"))
    if source.get("split_fingerprint") != split_fingerprint(val, root=cfg["_root"]):
        raise ValueError("validation split/images differ from the evaluation report")
    if source.get("dataset_fingerprint") != dataset_fingerprint(cfg):
        raise ValueError("dataset changed since validation evaluation; rerun the validation evaluation")
    report_hash = file_sha256(path)
    if source.get("report_sha256") != report_hash:
        raise ValueError("evaluation report and its provenance do not match; rerun the validation evaluation")
    return source


def blind_spots(lad: dict, min_detection: float = 0.5) -> list:
    """Ordinal edit families whose level-1 edit is detected in fewer than `min_detection` of the Originals."""
    weak = []
    for knob in ORDINAL_KNOBS:
        j = lad.get("jnd", {})
        if knob in j and (j[knob] is None or j[knob]["level"] > 1):
            weak.append(knob)
    rows = [r for r in lad.get("rows", []) if r["artifact_type"] == "font_family"]
    if rows and min(r["detection_rate"] for r in rows) < min_detection:
        weak.append("font_family")
    return weak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--n", type=int, default=42, help="new ladder Originals to generate this round")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.n <= 0 or a.round <= 0 or a.workers < 0:
        raise ValueError("--n and --round must be positive; --workers must be nonnegative")
    path = a.results or os.path.join(P(cfg, "reports_dir"), "evaluation_results.json")
    source = validation_source(path, a.model, cfg)
    res = json.load(open(path))
    if a.model not in res or "ladder" not in res[a.model]:
        sys.exit(f"{path} has no ladder analysis for {a.model} - run 06_evaluate.py --split val first")
    weak = blind_spots(res[a.model]["ladder"])
    print(f"[engine] blind spots of {a.model}: {weak or 'none - every edit family detected at level 1'}")
    if not weak:
        return
    cfg["ladder"]["knobs"] = weak
    # Engine Originals never use held-out templates, collide with base IDs, or overwrite an old round.
    holdout = set(cfg["splits"].get("holdout_templates") or [])
    cfg["dataset"]["templates"] = [t for t in cfg["dataset"]["templates"] if t not in holdout]
    if "templates" in cfg["ladder"]:
        cfg["ladder"]["templates"] = [t for t in cfg["ladder"]["templates"] if t not in holdout]
    if not cfg["dataset"]["templates"] or cfg["ladder"].get("templates") == []:
        raise ValueError("no non-held-out templates are available for adaptive training")
    generation = generation_fingerprint(cfg)
    key = canonical_hash({"model": a.model, "base": cfg["model"]["base"], "round": a.round,
                          "n": a.n, "generation": generation, "source": source})[:20]
    cfg["_engine_key"] = key
    cfg["_engine_start"] = 1000000000 + int(key[:10], 16)
    cfg["_generation_fingerprint"] = generation
    out = P(cfg, "metadata_dir", "ladder_engine.jsonl")
    old = read_jsonl(out) if os.path.exists(out) else []
    for record in old:
        if record.get("engine_round") == a.round and record.get("engine_model") == a.model and record.get("engine_run") != key:
            raise ValueError("this model/round already exists with different inputs; use the next round number")
    with Pool(a.workers or os.cpu_count() or 1, initializer=_init, initargs=(cfg,)) as pool:
        recs = pool.map(_make, range(a.n))
    for r in recs:
        r["engine_round"] = a.round
        r["engine_targets"] = weak
        r["engine_model"] = a.model
        r["engine_run"] = key
        r["engine_source"] = source
    merged = {r["sample_id"]: r for r in old}
    if len(merged) != len(old):
        raise ValueError("existing engine metadata contains duplicate sample IDs")
    merged.update({r["sample_id"]: r for r in recs})
    base = read_jsonl(P(cfg, "metadata_dir", "ladder.jsonl"))
    validate_samples(base + list(merged.values()), root=cfg["_root"])
    atomic_jsonl(out, [merged[sid] for sid in sorted(merged)])
    n_pairs = sum(len(r["releases"]) for r in recs)
    print(f"[engine] round {a.round}: {len(recs)} Originals, {n_pairs} pairs focused on {weak} -> {out}")
    print("[engine] next: python scripts/02_build_splits.py  (adds them to train)  ->  retrain  ->  re-evaluate")


if __name__ == "__main__":
    main()
