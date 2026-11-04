"""Stage 3 - Zero-shot benchmark (GPU).

  a) Prompt study: P1-P4 on a fixed subset of the VALIDATION split (never test) with the untouched
     4-bit base model -> per-prompt valid rate, pooled MAE and per-metric MAE (metric diagnostics).
  b) The best prompt (lowest MAE + 0.5 * invalid-rate) is run zero-shot on the full TEST split
     -> predictions/baseline_zero_shot_test.jsonl  (the number QLoRA must beat).

Resumable: re-running skips pairs already predicted.
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prism.common import P, load_config, read_jsonl  # noqa: E402
from prism.dataset import atomic_json  # noqa: E402
from prism.evaluation import score_errors  # noqa: E402
from prism.metrics import METRIC_KEYS, METRIC_NAMES  # noqa: E402
from prism.prompts import PROMPTS  # noqa: E402
from prism.vlm import load_model, load_processor, predict  # noqa: E402


def score_rows(rows, gt):
    """Fixed expected-pair denominator, strict scores and no invented accuracy on failures."""
    if not gt or any(score_errors(scores) for scores in gt.values()):
        raise ValueError("Prompt scoring needs nonempty valid expected ground-truth scores")
    predictions = {}
    for row in rows:
        identity = row.get("id")
        if identity not in gt or identity in predictions:
            raise ValueError("Prompt predictions contain an unknown or duplicate pair ID")
        predictions[identity] = row
    valid = [row for row in rows if not score_errors(row.get("pred"))]
    out = {"n": len(gt), "n_returned": len(rows), "n_valid": len(valid),
           "n_missing": len(gt) - len(rows), "n_invalid_outputs": len(rows) - len(valid),
           "valid_rate": len(valid) / len(gt), "eligible_for_selection": bool(valid)}
    errors = np.asarray([[abs(row["pred"][key] - gt[row["id"]][key]) for key in METRIC_KEYS]
                         for row in valid], dtype=float)
    out["MAE_pooled"] = float(errors.mean()) if valid else None
    for index, key in enumerate(METRIC_KEYS):
        out[f"MAE_{key}"] = float(errors[:, index].mean()) if valid else None
    # 1.0 is a selection penalty for an entirely invalid prompt, never a measured MAE.
    out["selection_score"] = (out["MAE_pooled"] if valid else 1.0) + 0.5 * (1 - out["valid_rate"])
    return out


def select_prompt(table):
    eligible = [row for row in table if row["eligible_for_selection"]]
    return min(eligible, key=lambda row: row["selection_score"])["prompt"] if eligible else None


def format_mae(value):
    return f"{value:.4f}" if value is not None else "n/a"


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
    if cfg["inference"]["test_limit"] < 0:
        raise ValueError("inference.test_limit must be nonnegative")
    if cfg["inference"]["test_limit"]:
        test = test[: cfg["inference"]["test_limit"]]
    if not val or not test:
        raise ValueError("Baseline needs nonempty validation and test splits; generate more Originals per template first")
    if not a.skip_study and cfg["inference"]["prompt_study_n"] < 1:
        raise ValueError("inference.prompt_study_n must be positive for prompt selection")
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
            table.append({"prompt": pid, **score_rows(rows, {item["id"]: item["scores"] for item in subset})})
            print(f"  {pid}: valid={table[-1]['valid_rate']:.2f} MAE={format_mae(table[-1]['MAE_pooled'])}", flush=True)
        df = pd.DataFrame(table)
        df.to_csv(os.path.join(rep_dir, "prompt_study.csv"), index=False)
        best = select_prompt(table)
        diag = df.set_index("prompt")[[f"MAE_{k}" for k in METRIC_KEYS]].T
        diag.index = [METRIC_NAMES[k] for k in METRIC_KEYS]
        diag.to_csv(os.path.join(rep_dir, "metric_diagnostics_zero_shot.csv"))
        atomic_json(os.path.join(rep_dir, "prompt_study.json"), {
            "selected_prompt": best, "status": "selected" if best else "failed_no_valid_prompt",
            "rule": "min(MAE_pooled + 0.5*(1-valid_rate)) among prompts with valid predictions on validation subset",
            "table": table})
        if best is None:
            raise RuntimeError("No prompt produced valid validation scores; inspect prompt_study.json before test inference")
        print(f"[prompt study] selected {best}", flush=True)

    print(f"[baseline] zero-shot {best} on {len(test)} test pairs", flush=True)
    rows = predict(model, proc, cfg, test, best, os.path.join(pred_dir, "baseline_zero_shot_test.jsonl"))
    s = score_rows(rows, {item["id"]: item["scores"] for item in test})
    print(f"[baseline] valid={s['valid_rate']:.3f} pooled MAE={format_mae(s['MAE_pooled'])}")


if __name__ == "__main__":
    main()
