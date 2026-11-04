"""Stage 6 - Agreement analytics + diagnostic audit + report.

Reads predictions/<model>_<split>.jsonl for each model and the ground truth in data/pairs/<split>.jsonl,
then writes reports/evaluation_report.md, reports/evaluation_metrics.xlsx, figures and ranking CSVs.

Usage: python scripts/06_evaluate.py --models baseline_zero_shot finetuned_qlora [--split test]
All ground truth is computed by the renderer during generation; there is no manual annotation input.
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prism.common import P, load_config, read_jsonl  # noqa: E402
from prism.evaluation import evaluate, score_errors  # noqa: E402
from prism.report import write_report  # noqa: E402


def load_rows(cfg, model, split, pred_dir=None):
    expected = read_jsonl(P(cfg, "pairs_dir", f"{split}.jsonl"))
    def indexed(items, source):
        result = {}
        for row in items:
            if not isinstance(row, dict):
                raise ValueError(f"{source}: each JSONL record must be an object")
            identity = row.get("id")
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"{source}: prediction/ground-truth id must be a nonempty string")
            if identity in result:
                raise ValueError(f"{source}: duplicate id {identity}")
            result[identity] = row
        return result
    gt = indexed(expected, "ground truth")
    pred_path = os.path.join(pred_dir or P(cfg, "predictions_dir"), f"{model}_{split}.jsonl")
    preds = indexed(read_jsonl(pred_path) if os.path.exists(pred_path) else [], pred_path)
    unknown = set(preds) - set(gt)
    if unknown:
        raise ValueError(f"{pred_path}: unknown prediction ids {sorted(unknown)[:10]}")
    manifest_path = pred_path + ".manifest.json"
    manifest, scope = None, "unverified: no prediction manifest"
    if os.path.exists(manifest_path):
        from prism.dataset import canonical_hash, split_fingerprint
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        identity = manifest.get("identity", {})
        if not isinstance(identity, dict) or not isinstance(identity.get("pairs_fingerprint"), str):
            raise ValueError(f"{manifest_path}: malformed prediction identity")
        if manifest.get("fingerprint") != canonical_hash(identity):
            raise ValueError(f"{manifest_path}: identity fingerprint mismatch")
        expected_sha = split_fingerprint(expected)
        if identity["pairs_fingerprint"] == expected_sha:
            scope = "full expected split"
        elif preds and identity["pairs_fingerprint"] == split_fingerprint([row for row in expected if row["id"] in preds]):
            scope = "verified partial subset; coverage denominator remains the full expected split"
        else:
            raise ValueError(f"{manifest_path}: predictions do not match the expected split or recorded subset")
    rows = []
    for pid, g in gt.items():
        p = preds.get(pid, {})
        errors = score_errors(p.get("pred")) if p.get("pred") is not None else []
        effective_raw = p.get("effective_raw") if "effective_raw" in p else (p.get("retry_raw") if p.get("retried") else p.get("raw"))
        rows.append({"id": pid, "sample_id": g["sample_id"], "release_id": g["release_id"], "template": g["template"],
                     "gt": g["scores"], "pred": None if errors else p.get("pred"), "retried": p.get("retried", False),
                     "missing_prediction": pid not in preds, "prediction_errors": errors or p.get("errors", []),
                     "prediction_provenance": manifest, "prediction_scope": scope,
                     "kind": g.get("kind", "ladder"), "target": g.get("target"), "raw": effective_raw,
                     "raw_initial": p.get("raw"),
                     **{k: g[k] for k in ("artifact_type", "level", "artifact_value", "primary_metric") if k in g}})
    return rows


def run_evaluation(cfg, models, split="test", pred_dir=None, out_dir=None, extra=None):
    from prism.dataset import atomic_json, dataset_fingerprint, file_sha256, split_fingerprint
    if split not in ("val", "test"):
        raise ValueError("held-out evaluation requires split='val' or split='test'")
    expected = read_jsonl(P(cfg, "pairs_dir", f"{split}.jsonl"))
    model_config = {key: value for key, value in cfg.items() if not key.startswith("_")}
    config_json = json.dumps(model_config, sort_keys=True, allow_nan=False).encode("utf-8")
    provenance = {"schema_version": 1, "split": split,
                  "split_fingerprint": split_fingerprint(expected),
                  "dataset_fingerprint": dataset_fingerprint(cfg), "models": list(models),
                  "base_model": cfg["model"]["base"], "model_config": cfg["model"],
                  "configuration_fingerprint": hashlib.sha256(config_json).hexdigest(),
                  "configuration": model_config}
    w = cfg["scoring"]["category_weights"]
    thr = cfg["evaluation"]["top_error_threshold"]
    rows_by, results = {}, {}
    for m in models:
        rows_by[m] = load_rows(cfg, m, split, pred_dir)
        results[m] = evaluate(rows_by[m], w, thr, cfg["evaluation"].get("gate_threshold", 0.95))
        results[m]["provenance"] = dict(provenance, prediction_name=m)
        manifest = rows_by[m][0].get("prediction_provenance") if rows_by[m] else None
        identity = manifest.get("identity", {}) if manifest else {}
        results[m]["provenance"].update({"prediction_identity": identity or None,
            "prediction_manifest_fingerprint": manifest.get("fingerprint") if manifest else None,
            "prediction_scope": rows_by[m][0].get("prediction_scope") if rows_by[m] else "empty split",
            "base_model": identity.get("model", {}).get("base") if identity else None,
            "model_config": identity.get("model") if identity else None})
        print(f"[evaluate] {m}: n={len(rows_by[m])} valid={results[m]['coverage']['valid_rate']:.3f} "
              f"MAE={results[m]['all_metrics_pooled'].get('MAE', float('nan')):.4f} "
              f"bestRelease={results[m]['ranking']['best_release_accuracy_RSA']:.3f}")
    out = out_dir or P(cfg, "reports_dir")
    path = write_report(results, rows_by, out, title=f"PRISM VLM evaluation - {split} split", extra=extra)
    provenance["prediction_identities"] = {name: results[name]["provenance"]["prediction_identity"] for name in models}
    provenance["report_sha256"] = file_sha256(os.path.join(out, "evaluation_results.json"))
    atomic_json(os.path.join(out, "evaluation_provenance.json"), provenance)
    print(f"[evaluate] report -> {path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", default=["baseline_zero_shot", "finetuned_qlora"])
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    cfg = load_config(a.config)
    models = a.models
    missing = {m for m in models if not os.path.exists(os.path.join(P(cfg, "predictions_dir"), f"{m}_{a.split}.jsonl"))}
    if missing:
        print(f"[evaluate] WARNING: missing prediction files for {sorted(missing)}; these models have zero coverage")
    run_evaluation(cfg, models, a.split,
                   extra={"Base model": cfg["model"]["base"], "Split": a.split,
                          "Ground truth": "pre-defined scores realised by the renderer (see data/metadata)"})


if __name__ == "__main__":
    main()
