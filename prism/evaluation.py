"""Agreement analytics + diagnostic audit.

Accuracy measures reported (per metric, per category, overall, per template):
  MAE, RMSE, PLCC (Pearson), SROCC (Spearman), bias (mean signed error),
  tolerance accuracy (|error| <= 0.05 / 0.10 / 0.15),
  quality-band accuracy + macro-F1 (excellent >= .90 / good .75-.90 / fair .50-.75 / poor < .50),
Ranking measures: best-release accuracy (RSA), exact-ranking accuracy, mean Kendall tau,
                  pairwise preference accuracy per metric.
Coverage: valid-output (parse) rate after the validation gate.
"""

from __future__ import annotations

import itertools
import math
from numbers import Real
import re
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from .metrics import CATEGORIES, METRIC_KEYS, METRIC_CATEGORY, QUALITY_BANDS, band_of
from .scoring import category_scores, rank_releases

TOLS = (0.05, 0.10, 0.15)


def score_errors(scores):
    """Strict evaluation boundary: exactly twelve finite real scores; booleans are not scores."""
    if not isinstance(scores, dict):
        return ["not_score_object"]
    errors = []
    if set(scores) != set(METRIC_KEYS):
        errors.append("metric_keys_mismatch")
    for key, value in scores.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            errors.append(f"non_numeric:{key}")
        elif not 0 <= value <= 1 or not math.isfinite(float(value)):
            errors.append(f"invalid_range:{key}")
    return errors


def _mean(values):
    return float(np.mean(values)) if values else float("nan")


def _safe_corr(fn, a, b):
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(fn(a, b)[0])


def regression_metrics(gt, pred) -> dict:
    gt, pred = np.asarray(gt, float), np.asarray(pred, float)
    if len(gt) == 0:
        return {"n": 0, **{key: float("nan") for key in
                ("MAE", "RMSE", "PLCC", "SROCC", "bias", "band_acc", "band_macro_F1",
                 "acc@0.05", "acc@0.10", "acc@0.15")}}
    err = pred - gt
    out = {
        "n": int(len(gt)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(math.sqrt(np.mean(err ** 2))),
        "PLCC": _safe_corr(pearsonr, gt, pred),
        "SROCC": _safe_corr(spearmanr, gt, pred),
        "bias": float(np.mean(err)),
    }
    for t in TOLS:
        out[f"acc@{t:.2f}"] = float(np.mean(np.abs(err) <= t + 1e-9))
    gb, pb = [band_of(x) for x in gt], [band_of(x) for x in pred]
    out["band_acc"] = float(np.mean([a == b for a, b in zip(gb, pb)]))
    f1s = []
    for name, _, _ in QUALITY_BANDS:
        tp = sum(a == name and b == name for a, b in zip(gb, pb))
        fp = sum(a != name and b == name for a, b in zip(gb, pb))
        fn = sum(a == name and b != name for a, b in zip(gb, pb))
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    out["band_macro_F1"] = float(np.mean(f1s)) if f1s else float("nan")
    return out


def evaluate(records: list, category_weights=None, top_error_threshold=0.30, gate_threshold=0.95) -> dict:
    """records: [{sample_id, release_id, template, gt: {12}, pred: {12} | None, raw?}]"""
    checked = []
    seen, seen_pairs = set(), set()
    for row in records:
        identity = row.get("id", (row["sample_id"], row["release_id"]))
        if identity in seen:
            raise ValueError(f"duplicate evaluation identity: {identity}")
        seen.add(identity)
        pair = (row["sample_id"], row["release_id"])
        if pair in seen_pairs:
            raise ValueError(f"duplicate sample/release pair: {pair}")
        seen_pairs.add(pair)
        if score_errors(row["gt"]):
            raise ValueError(f"invalid ground truth for {identity}: {score_errors(row['gt'])}")
        errors = score_errors(row.get("pred")) if row.get("pred") is not None else []
        checked.append(dict(row, pred=None if errors else row.get("pred"),
                            prediction_errors=list(row.get("prediction_errors", [])) + errors))
    records = checked
    n_all = len(records)
    valid = [r for r in records if r.get("pred")]
    res = {"coverage": {"n_total": n_all, "n_valid": len(valid),
                        "valid_rate": len(valid) / n_all if n_all else float("nan"),
                        "n_retried": sum(1 for r in records if r.get("retried")),
                        "n_invalid": n_all - len(valid),
                        "n_missing": sum(bool(r.get("missing_prediction")) for r in records),
                        "n_invalid_outputs": sum(not r.get("missing_prediction") and r.get("pred") is None for r in records)}}
    res["coverage"]["denominator"] = "all expected ground-truth pairs, including absent predictions"
    res["invalid_outputs"] = [{"id": r.get("id"), "sample_id": r["sample_id"], "release_id": r["release_id"],
                               "missing_prediction": bool(r.get("missing_prediction")),
                               "errors": r.get("prediction_errors", [])}
                              for r in records if r.get("pred") is None]

    res["per_metric"] = {k: regression_metrics([r["gt"][k] for r in valid], [r["pred"][k] for r in valid])
                         for k in METRIC_KEYS}
    gcat = [category_scores(r["gt"], category_weights) for r in valid]
    pcat = [category_scores(r["pred"], category_weights) for r in valid]
    res["per_category"] = {c: regression_metrics([g[c] for g in gcat], [p[c] for p in pcat])
                           for c in CATEGORIES + ["overall"]}
    res["all_metrics_pooled"] = regression_metrics([r["gt"][k] for r in valid for k in METRIC_KEYS],
                                                    [r["pred"][k] for r in valid for k in METRIC_KEYS])
    by_t = defaultdict(list)
    for r, g, p in zip(valid, gcat, pcat):
        by_t[r["template"]].append((g["overall"], p["overall"]))
    res["per_template_overall"] = {t: regression_metrics([a for a, _ in v], [b for _, b in v]) for t, v in sorted(by_t.items())}

    res["ranking"], res["ranking_rows"] = ranking_metrics(records, category_weights)

    # ---------------- diagnostic audit
    err_rows = []
    for r in valid:
        for k in METRIC_KEYS:
            e = r["pred"][k] - r["gt"][k]
            err_rows.append({"sample_id": r["sample_id"], "release_id": r["release_id"], "template": r["template"],
                             "metric": k, "category": METRIC_CATEGORY[k], "gt": r["gt"][k], "pred": r["pred"][k],
                             "error": round(e, 4), "abs_error": round(abs(e), 4)})
    top = sorted([e for e in err_rows if e["abs_error"] > top_error_threshold], key=lambda e: -e["abs_error"])
    res["top_errors"] = top
    buckets = {"untouched (gt>=0.98)": (0.98, 1.01), "minute (0.90-0.98)": (0.90, 0.98),
               "moderate (0.60-0.90)": (0.60, 0.90), "strong (<0.60)": (-0.01, 0.60)}
    sev_table = {}
    for k in METRIC_KEYS:
        sev_table[k] = {}
        for bname, (lo, hi) in buckets.items():
            es = [e["abs_error"] for e in err_rows if e["metric"] == k and lo <= e["gt"] < hi]
            expected_count = sum(lo <= r["gt"][k] < hi for r in records)
            sev_table[k][bname] = {"n": len(es), "n_expected": expected_count,
                                 "valid_rate": len(es) / expected_count if expected_count else float("nan"),
                                 "MAE": float(np.mean(es)) if es else float("nan")}
    res["error_by_severity"] = sev_table
    pattern = defaultdict(lambda: {"n": 0, "over": 0, "under": 0})
    for e in top:
        p = pattern[(e["metric"], e["template"])]
        p["n"] += 1
        p["over" if e["error"] > 0 else "under"] += 1
    res["failure_patterns"] = sorted(
        [{"metric": m, "template": t, **v, "direction": "over-scores (misses defect)" if v["over"] >= v["under"]
          else "under-scores (hallucinates defect)"} for (m, t), v in pattern.items()], key=lambda x: -x["n"])
    res["error_rows"] = err_rows
    if any(r.get("kind") == "ladder" for r in records):
        res["ladder"] = ladder_sensitivity(records, category_weights)
        res["explanations"] = explanation_check(records)
    res["quality_gate"] = quality_gate(records, gate_threshold)
    return res


def ranking_metrics(records, category_weights=None):
    """Evaluate complete expected release groups, without filename tie-breaking credit.

    Quality ordering remains (overall, technical), as in the product score engine. GT ties form
    equivalence classes. Predicted best ties receive their fraction of GT-best members as credit;
    a constant predictor therefore cannot receive automatic perfect selection accuracy.
    """
    by_sample = defaultdict(dict)
    for row in records:
        if row["release_id"] != "control_0":
            by_sample[row["sample_id"]][row["release_id"]] = row
    best, exact, taus, rows = [], [], [], []
    pair_hits = {k: [] for k in METRIC_KEYS + ["overall"]}
    pair_expected = dict.fromkeys(pair_hits, 0)
    n_expected = 0
    for sid, releases in sorted(by_sample.items()):
        if len(releases) < 2:
            continue
        n_expected += 1
        gt_rank, gt_cats = rank_releases({k: r["gt"] for k, r in releases.items()}, category_weights)
        available = {k: r for k, r in releases.items() if r.get("pred") is not None}
        pred_rank, pred_cats = rank_releases({k: r["pred"] for k, r in available.items()}, category_weights)
        quality = lambda c: (c["overall"], c["technical"])
        gt_best = {k for k in releases if quality(gt_cats[k]) == quality(gt_cats[gt_rank[0]])}
        complete = len(available) == len(releases)
        pred_best = {k for k in available if quality(pred_cats[k]) == quality(pred_cats[pred_rank[0]])} if available else set()
        credit = len(gt_best & pred_best) / len(pred_best) if complete else 0.0
        correctly_ordered = complete and all(
            quality(gt_cats[a]) == quality(gt_cats[b]) or
            (quality(gt_cats[a]) > quality(gt_cats[b])) == (quality(pred_cats[a]) > quality(pred_cats[b]))
            and quality(pred_cats[a]) != quality(pred_cats[b])
            for a, b in itertools.combinations(releases, 2))
        if complete:
            best.append(credit)
            exact.append(bool(correctly_ordered))
            if len(releases) >= 3:
                ids = sorted(releases)
                tau = _safe_corr(kendalltau, [gt_cats[k]["overall"] for k in ids], [pred_cats[k]["overall"] for k in ids])
                if math.isfinite(tau):
                    taus.append(tau)
        for a, b in itertools.combinations(releases, 2):
            for k in pair_hits:
                dg = (gt_cats[a]["overall"] - gt_cats[b]["overall"]) if k == "overall" else releases[a]["gt"][k] - releases[b]["gt"][k]
                if abs(dg) <= (0.02 if k == "overall" else 0.05):
                    continue
                pair_expected[k] += 1
                if a in available and b in available:
                    dp = (pred_cats[a]["overall"] - pred_cats[b]["overall"]) if k == "overall" else available[a]["pred"][k] - available[b]["pred"][k]
                    pair_hits[k].append(bool(np.sign(dg) == np.sign(dp)))
        rows.append({"sample_id": sid, "n_expected": len(releases), "n_valid": len(available), "complete": complete,
                     "gt_ranking": ">".join(gt_rank), "pred_ranking": ">".join(pred_rank),
                     "gt_best_set": ",".join(sorted(gt_best)), "pred_best_set": ",".join(sorted(pred_best)),
                     "best_credit": credit, "exact_order_correct": bool(correctly_ordered)})
    return {"n_samples": len(best), "n_samples_expected": n_expected, "n_samples_incomplete": n_expected - len(best),
            "complete_group_rate": len(best) / n_expected if n_expected else float("nan"),
            "best_release_accuracy_RSA": _mean(best),
            "best_release_accuracy_end_to_end": sum(best) / n_expected if n_expected else float("nan"),
            "exact_ranking_accuracy": _mean(exact),
            "exact_ranking_accuracy_end_to_end": sum(exact) / n_expected if n_expected else float("nan"),
            "mean_kendall_tau": _mean(taus), "n_kendall": len(taus),
            "tie_policy": "quality=(overall,technical); GT equivalence classes; best-set intersection / predicted-best-set size",
            "pairwise_preference_accuracy": {k: _mean(v) for k, v in pair_hits.items()},
            "pairwise_preference_accuracy_end_to_end": {k: sum(pair_hits[k]) / pair_expected[k] if pair_expected[k] else float("nan") for k in pair_hits},
            "pairwise_counts": {k: {"n_expected": pair_expected[k], "n_valid": len(pair_hits[k])} for k in pair_hits}}, rows


KNOB_PRIMARY = {"font_family": "font_similarity", "font_size": "font_similarity", "font_color": "font_similarity",
                "inpaint_color_mismatch": "background_fidelity", "justification": "alignment_preservation",
                "line_spacing": "reading_flow", "char_substitution": "ocr_accuracy",
                "ghost_residue": "ghost_text_residue", "texture_smoothing": "texture_continuity",
                "lighting_drop": "lighting_consistency", "local_compression": "authenticity",
                "object_shift": "object_preservation", "heading_shift": "visual_anchor_integrity"}


def quality_gate(records: list, threshold: float = 0.95) -> dict:
    """Valid-output confusion metrics plus end-to-end success, with abstentions explicit."""
    tp = fp = fn = tn = 0
    diag = []
    n_gt_reject = sum(min(r["gt"].values()) < threshold for r in records)
    for row in records:
        if row.get("pred") is None:
            continue
        g_rej = min(row["gt"].values()) < threshold
        p_rej = min(row["pred"].values()) < threshold
        tp += g_rej and p_rej
        fp += (not g_rej) and p_rej
        fn += g_rej and not p_rej
        tn += (not g_rej) and not p_rej
        if g_rej and p_rej:
            gmin, pmin = min(row["gt"].values()), min(row["pred"].values())
            gt_worst = {k for k in METRIC_KEYS if row["gt"][k] == gmin}
            pred_worst = {k for k in METRIC_KEYS if row["pred"][k] == pmin}
            diag.append(len(gt_worst & pred_worst) / len(pred_worst))
    n = tp + fp + fn + tn
    total = len(records)
    return {"threshold": threshold, "n": n, "n_total": total, "n_abstained": total - n,
            "decision_accuracy": (tp + tn) / n if n else float("nan"),
            "decision_accuracy_end_to_end": (tp + tn) / total if total else float("nan"),
            "reject_precision": tp / (tp + fp) if tp + fp else float("nan"),
            "reject_recall": tp / (tp + fn) if tp + fn else float("nan"),
            "reject_recall_end_to_end": tp / n_gt_reject if n_gt_reject else float("nan"),
            "false_reject_rate": fp / (fp + tn) if fp + tn else float("nan"),
            "diagnosis_accuracy": _mean(diag), "n_diagnoses": len(diag),
            "diagnosis_accuracy_end_to_end": sum(diag) / n_gt_reject if n_gt_reject else float("nan"),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


_POS = re.compile(r"\b(top|middle|bottom)(?:-(left|right))?\b")
_NUM = re.compile(r"(-?\d+(?:\.\d+)?)\s*(px|%)")


def _metric_line(text: str, metric: str):
    """The analysis line for a metric (lines are in METRIC_KEYS order, one per metric)."""
    if not text or "<analysis>" not in text or "</analysis>" not in text:
        return None
    body = text.split("<analysis>", 1)[1].split("</analysis>", 1)[0]
    lines = [l.strip() for l in body.splitlines() if l.strip().startswith("-")]
    return lines[METRIC_KEYS.index(metric)] if len(lines) == len(METRIC_KEYS) else None


def explanation_check(records: list) -> dict:
    """Count missing/invalid explanations as failures wherever ground truth is checkable."""
    loc_hits, num_hits, loc_cond, num_cond = [], [], [], []
    n_lines = eligible = missing_targets = 0
    for row in records:
        metric = row.get("primary_metric")
        if row.get("kind") != "ladder" or row.get("artifact_type") in (None, "control") or metric not in METRIC_KEYS:
            continue
        gt_line = _metric_line(row.get("target", ""), metric)
        if gt_line is None:
            missing_targets += 1
            continue
        eligible += 1
        pred_line = _metric_line(row.get("raw"), metric) if row.get("pred") is not None else None
        n_lines += pred_line is not None
        g_pos = _POS.search(gt_line)
        p_pos = _POS.search(pred_line) if pred_line else None
        if g_pos:
            hit = bool(p_pos) and p_pos.group(0) == g_pos.group(0)
            loc_hits.append(hit)
            if pred_line is not None:
                loc_cond.append(hit)
        g_num = _NUM.findall(gt_line.split("->")[0])
        p_num = _NUM.findall(pred_line.split("->")[0]) if pred_line else []
        if g_num:
            gv, gu = float(g_num[0][0]), g_num[0][1]
            same = [float(v) for v, unit in p_num if unit == gu]
            hit = bool(same) and abs(same[0] - gv) <= max(1.0, 0.25 * abs(gv))
            num_hits.append(hit)
            if pred_line is not None:
                num_cond.append(hit)
    return {"n_explanations": n_lines, "n_eligible": eligible, "n_missing_or_invalid": eligible - n_lines,
            "n_missing_targets": missing_targets, "explanation_coverage": n_lines / eligible if eligible else float("nan"),
            "location_accuracy": _mean(loc_hits), "n_location": len(loc_hits),
            "quantity_accuracy": _mean(num_hits), "n_quantity": len(num_hits),
            "location_accuracy_given_explanation": _mean(loc_cond),
            "quantity_accuracy_given_explanation": _mean(num_cond)}


def ladder_sensitivity(records: list, category_weights=None, false_alarm_below=0.98) -> dict:
    """Detection requires both a valid release and that Original's valid unchanged control.

    Conditional rates use available comparisons. End-to-end rates and JND use all expected
    releases, so dropping controls or difficult predictions cannot improve reported sensitivity.
    """
    by_s = defaultdict(dict)
    for row in records:
        if row.get("kind") == "ladder":
            by_s[row["sample_id"]][row["release_id"]] = row
    per, controls = {}, []
    n_expected_controls = 0
    for rels in by_s.values():
        c = rels.get("control_0")
        n_expected_controls += c is not None
        valid_control = c is not None and c.get("pred") is not None
        if valid_control:
            controls.append(c)
        for rid, row in rels.items():
            if rid == "control_0" or row.get("primary_metric") not in METRIC_KEYS:
                continue
            metric = row["primary_metric"]
            point = per.setdefault(rid, {"artifact_type": row["artifact_type"], "level": row["level"],
                "value": row.get("artifact_value"), "primary_metric": metric, "gt": [], "pred": [],
                "valid_gt": [], "det": [], "det_overall": [], "n_missing_controls": 0})
            point["gt"].append(row["gt"][metric])
            if not valid_control:
                point["n_missing_controls"] += 1
            if row.get("pred") is None:
                continue
            point["pred"].append(row["pred"][metric])
            point["valid_gt"].append(row["gt"][metric])
            if valid_control:
                point["det"].append(row["pred"][metric] < c["pred"][metric] - 1e-9)
                point["det_overall"].append(category_scores(row["pred"], category_weights)["overall"]
                    < category_scores(c["pred"], category_weights)["overall"] - 1e-9)
    rows = []
    for rid, point in sorted(per.items(), key=lambda item: (item[1]["artifact_type"], item[1]["level"])):
        n = len(point["gt"])
        rows.append({"release_id": rid, **{k: point[k] for k in ("artifact_type", "level", "value", "primary_metric")},
            "n": n, "n_valid": len(point["pred"]), "n_compared": len(point["det"]),
            "n_missing_controls": point["n_missing_controls"], "gt_mean": _mean(point["gt"]),
            "pred_mean": _mean(point["pred"]),
            "MAE": _mean([abs(g - p) for g, p in zip(point["valid_gt"], point["pred"])]),
            "detection_rate": _mean(point["det"]), "overall_detection_rate": _mean(point["det_overall"]),
            "detection_rate_end_to_end": sum(point["det"]) / n,
            "overall_detection_rate_end_to_end": sum(point["det_overall"]) / n})
    jnd, mono = {}, {}
    for knob in sorted({row["artifact_type"] for row in rows}):
        if knob == "font_family":
            continue
        levels = sorted((row for row in rows if row["artifact_type"] == knob), key=lambda row: row["level"])
        detected = [row for row in levels if row["detection_rate_end_to_end"] >= 0.5]
        jnd[knob] = {"level": detected[0]["level"], "value": detected[0]["value"]} if detected else None
        correlations, n_flat, n_incomplete = [], 0, 0
        for rels in by_s.values():
            expected = [row for row in rels.values() if row.get("artifact_type") == knob]
            if not expected:
                continue
            if len(expected) < 3 or any(row.get("pred") is None for row in expected):
                n_incomplete += 1
                continue
            pts = sorted((row["level"], row["pred"][row["primary_metric"]]) for row in expected)
            if np.std([score for _, score in pts]) <= 1e-9:
                n_flat += 1
            else:
                correlations.append(float(spearmanr([level for level, _ in pts], [score for _, score in pts])[0]))
        mono[knob] = {"mean_spearman": _mean(correlations), "n_with_variation": len(correlations),
                      "n_flat": n_flat, "n_incomplete": n_incomplete}
    leak = defaultdict(lambda: defaultdict(list))
    intended_mass, total_mass = defaultdict(float), defaultdict(float)
    unintended = defaultdict(lambda: defaultdict(list))
    attribution_counts = defaultdict(int)
    for rels in by_s.values():
        c = rels.get("control_0")
        if c is None or c.get("pred") is None:
            continue
        for rid, row in rels.items():
            if rid != "control_0" and row.get("pred") is not None:
                knob = row["artifact_type"]
                intended_metrics = {key for key in METRIC_KEYS if abs(row["gt"][key] - c["gt"][key]) > 1e-9}
                primary = row.get("primary_metric", KNOB_PRIMARY.get(knob))
                if primary in METRIC_KEYS:
                    intended_metrics.add(primary)
                drops = {key: max(0.0, c["pred"][key] - row["pred"][key]) for key in METRIC_KEYS}
                intended_mass[knob] += sum(drops[key] for key in intended_metrics)
                total_mass[knob] += sum(drops.values())
                attribution_counts[knob] += 1
                for key in METRIC_KEYS:
                    leak[knob][key].append(row["pred"][key] - c["pred"][key])
                    unintended[knob][key].append(0.0 if key in intended_metrics else drops[key])
    leakage = {knob: {key: _mean(values) for key, values in keys.items()} for knob, keys in leak.items()}
    precision = {}
    for knob in leakage:
        precision[knob] = intended_mass[knob] / total_mass[knob] if total_mass[knob] > 1e-9 else float("nan")
    fa = [any(c["pred"][key] < false_alarm_below for key in METRIC_KEYS) for c in controls]
    return {"n_originals": len(by_s), "rows": rows, "jnd": jnd, "monotonicity": mono,
            "jnd_denominator": "all expected releases; missing or invalid comparisons count as failures",
            "leakage": leakage, "attribution_precision": precision,
            "attribution_definition": "sum of positive predicted drops on per-pair GT-changed metrics plus primary / all positive predicted drops",
            "attribution_counts": dict(attribution_counts),
            "unintended_drop": {knob: {key: _mean(values) for key, values in keys.items()} for knob, keys in unintended.items()},
            "control": {"n": len(controls), "n_expected": n_expected_controls,
                        "n_originals_without_valid_control": len(by_s) - len(controls),
                        "false_alarm_rate": _mean(fa),
                        "mean_pred_overall": _mean([category_scores(c["pred"], category_weights)["overall"] for c in controls])}}
