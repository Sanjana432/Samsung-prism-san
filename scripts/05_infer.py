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
    """Show every expected edited release; never choose a winner from incomplete predictions."""
    from prism.validation import validate_scores
    meta = {item["id"]: item for item in items}
    if len(meta) != len(items):
        raise ValueError("duplicate expected release-report IDs")
    predictions = {}
    for row in rows:
        if row["id"] not in meta or row["id"] in predictions:
            raise ValueError("unknown/duplicate prediction in release report")
        predictions[row["id"]] = row
    expected = defaultdict(list)
    for item in items:
        if item["release_id"] != "control_0":
            expected[item["sample_id"]].append(item)
    out, coverage = [], []
    for sid, members in sorted(expected.items()):
        valid = {}
        for item in members:
            row = predictions.get(item["id"])
            score = validate_scores(row.get("pred"))[0] if row else None
            if score is not None:
                valid[item["release_id"]] = score
        complete = len(valid) == len(members)
        ranking, cats = rank_releases(valid, cfg["scoring"]["category_weights"]) if valid else ([], {})
        quality = {rid: (cats[rid]["overall"], cats[rid]["technical"]) for rid in ranking}
        coverage.append({"sample_id": sid, "expected_releases": len(members), "valid_releases": len(valid),
                         "complete_group": complete, "selection_status": "complete" if complete else "incomplete_unranked"})
        for item in sorted(members, key=lambda value: value["release_id"]):
            rid, pair_id = item["release_id"], item["id"]
            score = valid.get(rid)
            rank = (1 + sum(q > quality[rid] for q in quality.values())) if complete and score is not None else None
            out.append({"sample_id": sid, "release_id": rid, "pair_id": pair_id,
                        "prediction_status": "valid" if score is not None else ("invalid" if pair_id in predictions else "missing"),
                        "complete_group": complete, **{key: score[key] if score else None for key in METRIC_KEYS},
                        **{f"cat_{c}": cats[rid][c] if score else None for c in CATEGORIES},
                        "overall": cats[rid]["overall"] if score else None, "rank": rank,
                        "is_best_release": rank == 1})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(out).to_excel(writer, sheet_name="releases", index=False)
        pd.DataFrame(coverage).to_excel(writer, sheet_name="coverage", index=False)
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
