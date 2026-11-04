"""Stage 1 - Generate the English dataset: the "24 generation" minute-change ladder (prism/ladder.py).

For each Original: 1 unchanged control + 6 edit families x 4 levels = 24 releases, each changing ONE text
block. Every release's 12 ground-truth scores are MEASURED by the renderer during generation and stored
in the metadata (no human annotation needed for these labels).

Outputs
  data/images/ladder_XXXX/{original,control_0,<knob>_<level>}.png
  data/metadata/ladder.jsonl           one record per Original (full metadata, OCR boxes, control params)
  data/metadata/ladder_metadata.csv    one row per Original-Release pair (easy to open in Excel)
  data/metadata/ladder_summary.json    per knob x level: mean measured score, render audit, monotonicity
  data/metadata/metadata_ladder.json   24 records per Original in the metadata_real.json schema (measured scores)

Usage:  python scripts/01_generate_dataset.py [--n 300] [--workers 8]
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prism.common import P, load_config, read_jsonl  # noqa: E402
from prism.dataset import atomic_json, atomic_jsonl, generation_fingerprint, render_sample_once, validate_samples  # noqa: E402
from prism.ladder import ladder_summary, to_metadata_real  # noqa: E402
from prism.metrics import CATEGORIES, METRIC_KEYS  # noqa: E402

CFG = None


def _init(cfg):
    global CFG
    CFG = cfg


def make_ladder(i: int) -> dict:
    return render_sample_once(CFG, i)


def generate_ladder(cfg, n, workers):
    if n <= 0 or workers <= 0:
        raise ValueError("sample and worker counts must be positive")
    t0 = time.time()
    md = P(cfg, "metadata_dir")
    os.makedirs(md, exist_ok=True)
    fingerprint = generation_fingerprint(cfg)
    state_path = os.path.join(md, "generation_state.json")
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as handle:
            if json.load(handle)["generation_fingerprint"] != fingerprint:
                raise ValueError("generation settings/code/fonts changed; existing images will not be overwritten; use a new dataset directory")
    old_path = os.path.join(md, "ladder.jsonl")
    old = read_jsonl(old_path) if os.path.exists(old_path) else []
    if old:
        validate_samples(old, root=cfg["_root"])
    existing = {r["sample_id"]: r for r in old}
    requested = {f"ladder_{i:04d}" for i in range(n)}
    if set(existing) - requested:
        raise ValueError("requested sample count would drop existing Originals; use a new dataset directory")
    atomic_json(state_path, {"generation_fingerprint": fingerprint, "label_source": "synthetic renderer measurements"})
    cfg = copy.deepcopy(cfg)
    cfg["_generation_fingerprint"] = fingerprint
    todo = [i for i in range(n) if f"ladder_{i:04d}" not in existing]
    with Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        for k, rec in enumerate(pool.imap(make_ladder, todo)):
            existing[rec["sample_id"]] = rec
            if (k + 1) % 5 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                print(f"[ladder] new/resumed {k + 1}/{len(todo)}, reused {len(old)} Originals "
                      f"({el:.0f}s)", flush=True)
    records = [existing[sid] for sid in sorted(requested)]
    integrity = validate_samples(records, root=cfg["_root"])
    # same schema as metadata_real.json (prism_iqa): 24 records per Original, measured metric_scores
    atomic_json(os.path.join(md, "metadata_ladder.json"), to_metadata_real(records))
    with open(os.path.join(md, "ladder_metadata.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["sample_id", "release_id", "template", "artifact_type", "level", "artifact_value", "primary_metric",
                     "target_block", "original_image_path", "release_image_path"] + METRIC_KEYS + CATEGORIES
                    + ["overall", "changed_pixels", "changed_pixels_outside_region"])
        for r in records:
            for rel in r["releases"]:
                cs = rel["category_scores"]
                wr.writerow([r["sample_id"], rel["release_id"], r["template"], rel["artifact_type"], rel["level"],
                             rel["artifact_value"], rel["primary_metric"], r["target_block"]["id"],
                             r["original_image_path"], rel["release_image_path"]]
                            + [rel["scores"][k] for k in METRIC_KEYS] + [cs[c] for c in CATEGORIES]
                            + [cs["overall"], rel["render_audit"]["changed_pixels"],
                               rel["render_audit"]["changed_pixels_outside_region"]])
    summ = ladder_summary(records)
    summ["seconds"] = round(time.time() - t0, 1)
    summ.update(integrity)
    summ["generation_fingerprint"] = fingerprint
    atomic_json(os.path.join(md, "ladder_summary.json"), summ)
    atomic_jsonl(os.path.join(md, "ladder.jsonl"), records)
    print(json.dumps({k: summ[k] for k in ("n_samples", "n_pairs", "monotonic_share")}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=None, help="ladder Originals (x25 pairs)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    n = a.n if a.n is not None else cfg["ladder"]["n_originals"]
    workers = a.workers if a.workers is not None else cfg["dataset"]["workers"]
    workers = workers or os.cpu_count() or 1
    generate_ladder(cfg, n, workers)


if __name__ == "__main__":
    main()
