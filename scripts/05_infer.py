"""Stage 5 - Inference scoring with the fine-tuned adapter (GPU).

CoT reasoning -> 12 raw scores -> validation gate (1 retry) -> predictions/<name>_<split>.jsonl.
Then runs the Python score engine on the predictions and writes the product report
(per Original: category scores, overall, release ranking, best release) to reports/release_report_<name>.xlsx.

Usage: python scripts/05_infer.py --adapter outputs/best_adapter --name finetuned_qlora --split test
"""

import argparse
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prism.common import P, abspath, load_config, read_jsonl  # noqa: E402
from prism.metrics import CATEGORIES, METRIC_KEYS  # noqa: E402
from prism.prompts import TRAIN_PROMPT  # noqa: E402
from prism.scoring import rank_releases  # noqa: E402


def release_report(cfg, rows, items, path):
    meta = {it["id"]: it for it in items}
    by_s = defaultdict(dict)
    for r in rows:
        if r["pred"] and meta[r["id"]]["release_id"] != "control_0":
            by_s[meta[r["id"]]["sample_id"]][meta[r["id"]]["release_id"]] = r["pred"]
    out = []
    for sid, rels in sorted(by_s.items()):
        ranking, cats = rank_releases(rels, cfg["scoring"]["category_weights"])
        for rid in sorted(rels):
            out.append({"sample_id": sid, "release_id": rid, **rels[rid],
                        **{f"cat_{c}": cats[rid][c] for c in CATEGORIES}, "overall": cats[rid]["overall"],
                        "rank": ranking.index(rid) + 1, "is_best_release": rid == ranking[0]})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(out).to_excel(path, index=False)
    print(f"[infer] release report -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--adapter", default="outputs/best_adapter")
    ap.add_argument("--name", default="finetuned_qlora")
    ap.add_argument("--split", default="test")
    ap.add_argument("--prompt", default=TRAIN_PROMPT)
    a = ap.parse_args()
    cfg = load_config(a.config)
    from prism.dataset import assert_dataset_ready
    assert_dataset_ready(cfg)
    from prism.vlm import load_model, load_processor, predict
    items = read_jsonl(P(cfg, "pairs_dir", f"{a.split}.jsonl"))
    if cfg["inference"]["test_limit"] and a.split == "test":
        items = items[: cfg["inference"]["test_limit"]]
    adapter = abspath(a.adapter) if a.adapter and a.adapter.lower() != "none" else None
    cfg["_adapter_path"] = adapter
    model, proc = load_model(cfg, adapter), load_processor(cfg)
    rows = predict(model, proc, cfg, items, a.prompt, os.path.join(P(cfg, "predictions_dir"), f"{a.name}_{a.split}.jsonl"))
    valid = sum(1 for r in rows if r["pred"])
    print(f"[infer] {a.name}: {valid}/{len(rows)} valid outputs")
    release_report(cfg, rows, items, os.path.join(P(cfg, "reports_dir"), f"release_report_{a.name}.xlsx"))


if __name__ == "__main__":
    main()
