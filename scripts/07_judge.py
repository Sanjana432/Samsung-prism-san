"""Stage 7 - Judge any original / edited image pair (GPU): scores, ACCEPT/REJECT, what is wrong, where, and the fix.

Works on images from outside the dataset - translated documents, AI-edited photos of documents, UI
screenshots, re-typeset scans. Images must show the same page at the same size (no automatic resizing,
which would erase minute changes).

Usage:
  python scripts/07_judge.py --original a.png --release b.png [--adapter outputs/best_adapter]
  python scripts/07_judge.py --pairs pairs.csv          # columns: original,release  -> reports/judge_results.jsonl
"""

import argparse
import csv
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prism.common import P, abspath, load_config  # noqa: E402
from prism.judge import verdict  # noqa: E402
from prism.prompts import TRAIN_PROMPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--original")
    ap.add_argument("--release")
    ap.add_argument("--pairs", help="CSV with columns original,release")
    ap.add_argument("--adapter", default="outputs/best_adapter")
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.pairs:
        with open(a.pairs, newline="") as f:
            pairs = [(r["original"], r["release"]) for r in csv.DictReader(f)]
    elif a.original and a.release:
        pairs = [(a.original, a.release)]
    else:
        sys.exit("give --original and --release, or --pairs")
    items = []
    for i, (o, r) in enumerate(pairs):
        so, sr = Image.open(abspath(o)).size, Image.open(abspath(r)).size
        if so != sr:
            sys.exit(f"pair {i}: sizes differ {so} vs {sr} - align/crop the pages first")
        items.append({"id": f"pair_{i:05d}", "sample_id": f"pair_{i:05d}", "release_id": "R",
                      "original_image_path": o, "release_image_path": r})
    from prism.vlm import load_model, load_processor, predict
    adapter = abspath(a.adapter) if a.adapter and a.adapter.lower() != "none" else None
    if adapter and not os.path.isfile(os.path.join(adapter, "adapter_config.json")):
        sys.exit("Requested trained adapter is missing. Train first, supply --adapter PATH, or use --adapter none for an explicit baseline.")
    model, proc = load_model(cfg, adapter), load_processor(cfg)
    rows = predict(model, proc, cfg, items, TRAIN_PROMPT)
    thr = a.threshold if a.threshold is not None else cfg["evaluation"].get("gate_threshold", 0.95)
    out = []
    for it, row in zip(items, rows):
        v = verdict(row["pred"], row.get("effective_raw", row.get("raw")), thr, cfg["scoring"]["category_weights"])
        v.update({"original": it["original_image_path"], "release": it["release_image_path"]})
        out.append(v)
        print(f"\n{it['release_image_path']}: {v['decision']}  (overall {v.get('overall', float('nan')):.3f})")
        for fl in v["flags"]:
            print(f"  - {fl['name']} {fl['score']:.2f}: {fl['observation'] or ''}\n      fix: {fl['fix']}")
    if len(out) > 1:
        path = os.path.join(P(cfg, "reports_dir"), "judge_results.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.writelines(json.dumps(v) + "\n" for v in out)
        print(f"\n[judge] {len(out)} verdicts -> {path}")


if __name__ == "__main__":
    main()
