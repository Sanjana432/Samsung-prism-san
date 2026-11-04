"""Stage 3 - Zero-shot benchmark (GPU).

  a) Prompt study: P1-P4 on a fixed subset of the VALIDATION split (never test) with the untouched
     4-bit base model -> per-prompt valid rate, pooled MAE and per-metric MAE (metric diagnostics).
  b) The best prompt (lowest MAE + 0.5 * invalid-rate) is run zero-shot on the full TEST split
     -> predictions/baseline_zero_shot_test.jsonl  (the number QLoRA must beat).

Resumable: re-running skips pairs already predicted.
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prism.common import P, load_config, read_jsonl  # noqa: E402
from prism.metrics import METRIC_KEYS, METRIC_NAMES  # noqa: E402
from prism.prompts import PROMPTS  # noqa: E402
from prism.vlm import load_model, load_processor, predict  # noqa: E402


def score_rows(rows, gt):
    valid = [r for r in rows if r["pred"]]
    out = {"n": len(rows), "valid_rate": len(valid) / max(1, len(rows))}
    errs = {k: [abs(r["pred"][k] - gt[r["id"]][k]) for r in valid] for k in METRIC_KEYS}
    out["MAE_pooled"] = float(np.mean([e for v in errs.values() for e in v])) if valid else 1.0
    for k in METRIC_KEYS:
        out[f"MAE_{k}"] = float(np.mean(errs[k])) if errs[k] else float("nan")
    out["selection_score"] = out["MAE_pooled"] + 0.5 * (1 - out["valid_rate"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip_study", action="store_true", help="use P4 directly")
    a = ap.parse_args()
    cfg = load_config(a.config)
    from prism.dataset import assert_dataset_ready
    assert_dataset_ready(cfg)
    val = read_jsonl(P(cfg, "pairs_dir", "val.jsonl"))
    test = read_jsonl(P(cfg, "pairs_dir", "test.jsonl"))
    if cfg["inference"]["test_limit"]:
        test = test[: cfg["inference"]["test_limit"]]
    gt = {r["id"]: r["scores"] for r in val + test}
    if not val or not test:
        raise ValueError("Baseline needs nonempty validation and test splits; generate more Originals per template first")
    model, proc = load_model(cfg), load_processor(cfg)
    pred_dir, rep_dir = P(cfg, "predictions_dir"), P(cfg, "reports_dir")
    os.makedirs(rep_dir, exist_ok=True)

    best = "P4"
    if not a.skip_study:
        subset = sorted(val, key=lambda r: r["id"])
        random.Random(cfg["project_seed"]).shuffle(subset)
        subset = subset[: cfg["inference"]["prompt_study_n"]]
        table = []
        for pid in PROMPTS:
            print(f"[prompt study] {pid} on {len(subset)} val pairs", flush=True)
            rows = predict(model, proc, cfg, subset, pid, os.path.join(pred_dir, "prompt_study", f"{pid}_val.jsonl"))
            table.append({"prompt": pid, **score_rows(rows, gt)})
            print(f"  {pid}: valid={table[-1]['valid_rate']:.2f} MAE={table[-1]['MAE_pooled']:.4f}", flush=True)
        df = pd.DataFrame(table)
        df.to_csv(os.path.join(rep_dir, "prompt_study.csv"), index=False)
        best = df.sort_values("selection_score").iloc[0]["prompt"]
        diag = df.set_index("prompt")[[f"MAE_{k}" for k in METRIC_KEYS]].T
        diag.index = [METRIC_NAMES[k] for k in METRIC_KEYS]
        diag.to_csv(os.path.join(rep_dir, "metric_diagnostics_zero_shot.csv"))
        with open(os.path.join(rep_dir, "prompt_study.json"), "w") as f:
            json.dump({"selected_prompt": best, "rule": "min(MAE_pooled + 0.5*(1-valid_rate)) on validation subset",
                       "table": table}, f, indent=2)
        print(f"[prompt study] selected {best}", flush=True)

    print(f"[baseline] zero-shot {best} on {len(test)} test pairs", flush=True)
    rows = predict(model, proc, cfg, test, best, os.path.join(pred_dir, "baseline_zero_shot_test.jsonl"))
    s = score_rows(rows, gt)
    print(f"[baseline] valid={s['valid_rate']:.3f} pooled MAE={s['MAE_pooled']:.4f}")


if __name__ == "__main__":
    main()
