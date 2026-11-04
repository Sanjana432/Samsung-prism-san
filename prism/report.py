"""Writes the evaluation outputs: Excel workbook, Markdown report, figures, release-ranking CSVs."""

from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import CATEGORIES, CATEGORY_NAMES, METRIC_KEYS, METRIC_NAMES, QUALITY_BANDS, band_of  # noqa: E402
from .evaluation import score_errors  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical order (fixed, never cycled)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
KEY = ["MAE", "RMSE", "PLCC", "SROCC", "acc@0.05", "acc@0.10", "acc@0.15", "band_acc", "band_macro_F1", "bias", "n"]


def _fmt(v, d=3):
    if v is None or (isinstance(v, (float, np.floating)) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def finite_json(value):
    """JSON has no NaN/Infinity literals: undefined statistics are exported as null."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _plot_mae(results, path):
    names = list(results)
    labels = [METRIC_NAMES[k] for k in METRIC_KEYS] + [CATEGORY_NAMES[c] for c in CATEGORIES] + ["Overall"]
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=SURFACE)
    _style(ax)
    h = 0.8 / len(names)
    y = np.arange(len(labels))
    for i, n in enumerate(names):
        r = results[n]
        vals = [r["per_metric"][k]["MAE"] for k in METRIC_KEYS] + [r["per_category"][c]["MAE"] for c in CATEGORIES + ["overall"]]
        ax.barh(y + i * h - 0.4 + h / 2, vals, height=h * 0.85, color=SERIES[i % len(SERIES)], label=n)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error (lower is better)", color=INK2)
    ax.set_title("MAE vs renderer ground truth (valid outputs only)", color=INK, loc="left", fontsize=12)
    if not any(result["coverage"]["n_valid"] for result in results.values()):
        ax.text(0.5, 0.5, "No valid predictions; MAE is undefined", transform=ax.transAxes, ha="center")
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_scatter(rows, name, path):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), facecolor=SURFACE)
    for ax, k in zip(axes.ravel(), METRIC_KEYS):
        _style(ax)
        ax.grid(color=GRID, linewidth=0.6)
        g = [r["gt"][k] for r in rows if r.get("pred")]
        p = [r["pred"][k] for r in rows if r.get("pred")]
        ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle="--")
        ax.scatter(g, p, s=10, color=SERIES[0], alpha=0.55, linewidths=0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(METRIC_NAMES[k], fontsize=10, color=INK, loc="left")
    fig.supxlabel("Ground truth (pre-defined)", color=INK2)
    fig.supylabel(f"Predicted ({name})", color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_band_confusion(rows, name, path):
    bands = [b[0] for b in QUALITY_BANDS]
    m = np.zeros((4, 4))
    for r in rows:
        if r.get("pred"):
            for k in METRIC_KEYS:
                m[bands.index(band_of(r["gt"][k])), bands.index(band_of(r["pred"][k]))] += 1
    norm = m / np.maximum(m.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.6), facecolor=SURFACE)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{norm[i, j]:.0%}\n({int(m[i, j])})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(bands)
    ax.set_yticks(range(4))
    ax.set_yticklabels(bands)
    ax.set_xlabel("Predicted band", color=INK2)
    ax.set_ylabel("Ground-truth band", color=INK2)
    ax.set_title(f"Quality-band confusion ({name}), all metrics", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_ladder(results, path):
    knobs = ["font_size", "font_color", "inpaint_color_mismatch", "justification", "line_spacing", "font_family"]
    fig, axes = plt.subplots(1, len(knobs), figsize=(3.0 * len(knobs), 3.0), sharey=True)
    for ax, knob in zip(axes, knobs):
        for n, r in results.items():
            rows = sorted((x for x in (r.get("ladder") or {}).get("rows", []) if x["artifact_type"] == knob),
                          key=lambda x: x["level"])
            if rows:
                ax.plot([x["level"] for x in rows], [x.get("detection_rate_end_to_end", x["detection_rate"]) for x in rows], marker="o", label=n)
        ax.axhline(0.5, color="#999", lw=0.8, ls="--")
        ax.set_title(knob, fontsize=9)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_ylim(-0.02, 1.02)
        _style(ax)
    axes[0].set_ylabel("detections / all expected releases")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(results: dict, rows_by_model: dict, out_dir: str, title: str = "PRISM VLM evaluation",
                 extra: dict | None = None):
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    names = list(results)
    if not names:
        raise ValueError("at least one model result is required")
    rows_by_model = {name: [dict(row, pred=None if score_errors(row.get("pred")) else row["pred"])
                           for row in rows_by_model[name]] for name in names}
    for key in ("split_fingerprint", "dataset_fingerprint"):
        fingerprints = {result.get("provenance", {}).get(key) for result in results.values()} - {None}
        if len(fingerprints) > 1:
            raise ValueError(f"cannot compare results with different {key}")
    indexed = {name: {row.get("id", (row["sample_id"], row["release_id"])): row for row in rows_by_model[name]
                      if row.get("pred") is not None} for name in names}
    common_ids = set.intersection(*(set(rows) for rows in indexed.values()))
    if any(indexed[name][identity]["gt"] != indexed[names[0]][identity]["gt"]
           for name in names for identity in common_ids):
        raise ValueError("cannot compare predictions against different ground-truth scores")
    paired_mae = {name: (float(np.mean([abs(indexed[name][identity]["pred"][key] - indexed[name][identity]["gt"][key])
                                      for identity in common_ids for key in METRIC_KEYS])) if common_ids else float("nan"))
                  for name in names}

    # ---------------- Excel
    summary, per_metric, per_cat, per_tpl, rank, pref, sev = [], [], [], [], [], [], []
    for n, r in results.items():
        o, pooled, rk = r["per_category"]["overall"], r["all_metrics_pooled"], r["ranking"]
        summary.append({"model": n, "valid_output_rate": r["coverage"]["valid_rate"], "n_pairs": r["coverage"]["n_total"],
                        "n_missing_predictions": r["coverage"].get("n_missing", 0),
                        "n_invalid_outputs": r["coverage"].get("n_invalid_outputs", 0),
                        "n_common_valid_pairs": len(common_ids), "MAE_common_valid_pairs": paired_mae[n],
                        "MAE_all_metrics": pooled["MAE"], "RMSE_all_metrics": pooled["RMSE"],
                        "PLCC_all_metrics": pooled["PLCC"], "SROCC_all_metrics": pooled["SROCC"],
                        "acc@0.10_all_metrics": pooled["acc@0.10"], "band_acc_all_metrics": pooled["band_acc"],
                        "MAE_overall_score": o["MAE"], "PLCC_overall_score": o["PLCC"], "SROCC_overall_score": o["SROCC"],
                        "best_release_accuracy_given_complete_group": rk["best_release_accuracy_RSA"],
                        "best_release_accuracy_end_to_end": rk.get("best_release_accuracy_end_to_end"),
                        "complete_group_rate": rk.get("complete_group_rate"),
                        "gate_decision_accuracy_end_to_end": r["quality_gate"].get("decision_accuracy_end_to_end"),
                        "exact_ranking_accuracy": rk["exact_ranking_accuracy"], "mean_kendall_tau": rk["mean_kendall_tau"]})
        for k in METRIC_KEYS:
            per_metric.append({"model": n, "metric": METRIC_NAMES[k], **{c: r["per_metric"][k].get(c) for c in KEY}})
            pref.append({"model": n, "metric": METRIC_NAMES[k], "pairwise_preference_accuracy": rk["pairwise_preference_accuracy"][k]})
            for b, v in r["error_by_severity"][k].items():
                sev.append({"model": n, "metric": METRIC_NAMES[k], "gt_bucket": b, **v})
        pref.append({"model": n, "metric": "Overall", "pairwise_preference_accuracy": rk["pairwise_preference_accuracy"]["overall"]})
        for c in CATEGORIES + ["overall"]:
            per_cat.append({"model": n, "category": CATEGORY_NAMES.get(c, "Overall"), **{x: r["per_category"][c].get(x) for x in KEY}})
        for t, v in r["per_template_overall"].items():
            per_tpl.append({"model": n, "template": t, **{x: v.get(x) for x in KEY}})
        rank.append({"model": n, **{k: v for k, v in rk.items() if k != "pairwise_preference_accuracy"}})
    xlsx = os.path.join(out_dir, "evaluation_metrics.xlsx")
    with pd.ExcelWriter(xlsx) as xw:
        pd.DataFrame(summary).to_excel(xw, sheet_name="summary", index=False)
        pd.DataFrame(per_metric).to_excel(xw, sheet_name="per_metric", index=False)
        pd.DataFrame(per_cat).to_excel(xw, sheet_name="per_category", index=False)
        pd.DataFrame(per_tpl).to_excel(xw, sheet_name="per_template", index=False)
        pd.DataFrame(rank).to_excel(xw, sheet_name="ranking", index=False)
        pd.DataFrame(pref).to_excel(xw, sheet_name="pairwise_preference", index=False)
        pd.DataFrame(sev).to_excel(xw, sheet_name="error_by_severity", index=False)
        pd.DataFrame([{"model": name, **result["coverage"]} for name, result in results.items()]).to_excel(xw, sheet_name="coverage", index=False)
        pd.DataFrame([{"model": name, **{key: value for key, value in result["quality_gate"].items() if key != "confusion"}}
                      for name, result in results.items()]).to_excel(xw, sheet_name="quality_gate", index=False)
        pd.DataFrame([{"model": name, **result.get("explanations", {})} for name, result in results.items()]).to_excel(xw, sheet_name="explanations", index=False)
        pd.DataFrame([{"model": name, **{key: result.get("provenance", {}).get(key) for key in
                                       ("split", "split_fingerprint", "dataset_fingerprint", "base_model", "configuration_fingerprint")}}
                      for name, result in results.items()]).to_excel(xw, sheet_name="provenance", index=False)
        for n, r in results.items():
            short = n[:18]
            pd.DataFrame(r["failure_patterns"]).to_excel(xw, sheet_name=f"patterns_{short}"[:31], index=False)
            pd.DataFrame(r["top_errors"][:500]).to_excel(xw, sheet_name=f"top_err_{short}"[:31], index=False)
            pd.DataFrame(r["ranking_rows"]).to_excel(xw, sheet_name=f"rank_{short}"[:31], index=False)
            pd.DataFrame(r.get("invalid_outputs", [])).to_excel(xw, sheet_name=f"invalid_{short}"[:31], index=False)
            if r.get("ladder"):
                pd.DataFrame(r["ladder"]["rows"]).to_excel(xw, sheet_name=f"ladder_{short}"[:31], index=False)
    for n, r in results.items():
        pd.DataFrame(r["ranking_rows"]).to_csv(os.path.join(out_dir, f"release_rankings_{n}.csv"), index=False)

    # ---------------- figures
    _plot_mae(results, os.path.join(fig_dir, "mae_per_metric.png"))
    if any(r.get("ladder") for r in results.values()):
        _plot_ladder(results, os.path.join(fig_dir, "ladder_sensitivity.png"))
    for n in names:
        _plot_scatter(rows_by_model[n], n, os.path.join(fig_dir, f"scatter_{n}.png"))
        _plot_band_confusion(rows_by_model[n], n, os.path.join(fig_dir, f"band_confusion_{n}.png"))

    # ---------------- Markdown
    L = [f"# {title}", ""]
    if extra:
        for k, v in extra.items():
            L.append(f"- **{k}:** {v}")
        L.append("")
    L += ["## Evaluated-pair accuracy and coverage", "",
          "Coverage uses every expected ground-truth pair, including missing and invalid outputs. Regression, band and correlation statistics are conditional on valid predictions; undefined values are n/a. Gate and selection end-to-end success count abstentions as failures.", "",
          "Release quality is ordered by (overall, technical), matching the score engine. Ground-truth ties are equivalence classes; best-selection credit is the share of the predicted best set that belongs to the ground-truth best set. Exact ranking ignores order within GT ties but requires strict ordering between different GT classes. Incomplete release groups receive zero end-to-end selection credit.", "",
          "| Measure | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    rowsdef = [
        ("Valid-output rate (validation gate)", lambda r: r["coverage"]["valid_rate"]),
        ("Missing predictions", lambda r: r["coverage"].get("n_missing", 0)),
        ("Complete release-group rate", lambda r: r["ranking"].get("complete_group_rate")),
        ("MAE, all 12 metrics pooled", lambda r: r["all_metrics_pooled"]["MAE"]),
        ("RMSE, all 12 metrics pooled", lambda r: r["all_metrics_pooled"]["RMSE"]),
        ("PLCC, all 12 metrics pooled", lambda r: r["all_metrics_pooled"]["PLCC"]),
        ("SROCC, all 12 metrics pooled", lambda r: r["all_metrics_pooled"]["SROCC"]),
        ("Within +-0.05", lambda r: r["all_metrics_pooled"]["acc@0.05"]),
        ("Within +-0.10", lambda r: r["all_metrics_pooled"]["acc@0.10"]),
        ("Within +-0.15", lambda r: r["all_metrics_pooled"]["acc@0.15"]),
        ("Quality-band accuracy", lambda r: r["all_metrics_pooled"]["band_acc"]),
        ("Quality-band macro-F1", lambda r: r["all_metrics_pooled"]["band_macro_F1"]),
        ("Overall-score MAE", lambda r: r["per_category"]["overall"]["MAE"]),
        ("Overall-score SROCC", lambda r: r["per_category"]["overall"]["SROCC"]),
        ("Best-set selection credit (complete groups only)", lambda r: r["ranking"]["best_release_accuracy_RSA"]),
        ("Best-set selection credit (all expected groups)", lambda r: r["ranking"].get("best_release_accuracy_end_to_end")),
        ("Exact ranking (complete groups only, GT ties allowed)", lambda r: r["ranking"]["exact_ranking_accuracy"]),
        ("Exact ranking (all expected groups)", lambda r: r["ranking"].get("exact_ranking_accuracy_end_to_end")),
        ("Mean Kendall tau (ranking)", lambda r: r["ranking"]["mean_kendall_tau"]),
    ]
    for label, fn in rowsdef:
        L.append(f"| {label} | " + " | ".join(_fmt(fn(results[n])) for n in names) + " |")
    if len(names) >= 2:
        a, b = names[0], names[-1]
        if common_ids:
            d = paired_mae[b] - paired_mae[a]
            L += ["", f"On the same {len(common_ids)} pairs valid for every compared model, pooled MAE changes by {d:+.3f} from `{a}` to `{b}`. This is a common-valid-subset comparison; inspect full-split coverage alongside it."]
        else:
            L += ["", "No paired MAE comparison is available: there are no pairs with valid predictions for every model."]
    for n in names:
        r = results[n]
        L += ["", f"## Per-metric accuracy - {n}", "",
              "| Metric | MAE | RMSE | PLCC | SROCC | +-0.10 | Band acc | Bias | Pairwise pref. acc |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k in METRIC_KEYS:
            m = r["per_metric"][k]
            L.append(f"| {METRIC_NAMES[k]} | {_fmt(m.get('MAE'))} | {_fmt(m.get('RMSE'))} | {_fmt(m.get('PLCC'))} | "
                     f"{_fmt(m.get('SROCC'))} | {_fmt(m.get('acc@0.10'))} | {_fmt(m.get('band_acc'))} | {_fmt(m.get('bias'))} | "
                     f"{_fmt(r['ranking']['pairwise_preference_accuracy'][k])} |")
        L += ["", "| Category | MAE | RMSE | PLCC | SROCC |", "|---|---|---|---|---|"]
        for c in CATEGORIES + ["overall"]:
            m = r["per_category"][c]
            L.append(f"| {CATEGORY_NAMES.get(c, 'Overall')} | {_fmt(m.get('MAE'))} | {_fmt(m.get('RMSE'))} | "
                     f"{_fmt(m.get('PLCC'))} | {_fmt(m.get('SROCC'))} |")
        L += ["", f"### Diagnostic audit - {n}", "",
              f"{len(r['top_errors'])} metric predictions have |error| > 0.30 "
              f"({len(r['top_errors']) / max(1, len(r['error_rows'])):.1%} of all).", ""]
        if r["failure_patterns"]:
            L += ["| Metric | Template | Count | Direction |", "|---|---|---|---|"]
            for p in r["failure_patterns"][:10]:
                L.append(f"| {METRIC_NAMES[p['metric']]} | {p['template']} | {p['n']} | {p['direction']} |")
        L += ["", "MAE by ground-truth severity bucket (does the model see *minute* changes?):", "",
              "| Metric | untouched | minute | moderate | strong |", "|---|---|---|---|---|"]
        for k in METRIC_KEYS:
            v = list(r["error_by_severity"][k].values())
            L.append(f"| {METRIC_NAMES[k]} | " + " | ".join(_fmt(x["MAE"]) for x in v) + " |")
    for n in names:
        lad = results[n].get("ladder")
        if not lad:
            continue
        c = lad["control"]
        L += ["", f"## Minute-change sensitivity (ladder) - {n}", "",
              f"{lad['n_originals']} expected Originals. Detection requires a valid release and a valid unchanged control. No synthetic control score is substituted. End-to-end rates and just-noticeable levels include all expected releases; missing comparisons count as failures. Detection = predicted primary-metric "
              f"score below the model's own prediction for the unchanged control. Controls: false-alarm rate "
              f"{_fmt(c['false_alarm_rate'])} (any metric < 0.98 on an identical image), mean overall {_fmt(c['mean_pred_overall'])}.", "",
              "| Edit | Level | Value | Primary metric | GT mean | Pred mean | MAE | Valid comparisons | Detection conditional | Detection end-to-end |", "|---|---|---|---|---|---|---|---|---|---|"]
        for row in lad["rows"]:
            L.append(f"| {row['artifact_type']} | {row['level']} | {row['value']} | {METRIC_NAMES[row['primary_metric']]} | "
                     f"{_fmt(row['gt_mean'])} | {_fmt(row['pred_mean'])} | {_fmt(row['MAE'])} | {row.get('n_compared', row['n'])}/{row['n']} | {_fmt(row['detection_rate'])} | {_fmt(row.get('detection_rate_end_to_end'))} |")
        prec = lad.get("attribution_precision", {})
        if prec:
            L += ["", "Attribution precision (positive predicted-drop mass on the metrics changed in each pair's ground truth, including its primary metric, divided by all positive predicted-drop mass). A bold font-family edit legitimately affects both Font Similarity and Semantic Emphasis. Missing comparisons are excluded and counted in coverage:", "",
                  "| Edit | Attribution precision | Largest wrongly-blamed metric |", "|---|---|---|"]
            for knob, pv in sorted(prec.items()):
                d = lad.get("unintended_drop", {}).get(knob, {})
                other = max(d, key=d.get, default=None)
                if other is not None and d[other] <= 1e-9:
                    other = None
                L.append(f"| {knob} | {_fmt(pv)} | {METRIC_NAMES.get(other, '-')} ({_fmt(d.get(other))}) |")
        L += ["", "| Edit | Just-noticeable level (>= 50% detected) | Mean Spearman(level, pred) | Flat Originals |", "|---|---|---|---|"]
        for knob, m in lad["monotonicity"].items():
            j = lad["jnd"].get(knob)
            jtxt = "not reached" if not j else "L{} ({})".format(j["level"], j["value"])
            L.append(f"| {knob} | {jtxt} | {_fmt(m['mean_spearman'])} | {m['n_flat']} |")
    L += ["", "## Quality gate (accept / reject) and explanation check", "",
          "| Measure | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    gate_rows = [("Gate threshold", lambda r: r["quality_gate"]["threshold"]),
                 ("Gate abstentions (missing/invalid outputs)", lambda r: r["quality_gate"].get("n_abstained")),
                 ("Decision accuracy, valid outputs only", lambda r: r["quality_gate"]["decision_accuracy"]),
                 ("Decision accuracy, all expected pairs", lambda r: r["quality_gate"].get("decision_accuracy_end_to_end")),
                 ("Reject precision", lambda r: r["quality_gate"]["reject_precision"]),
                 ("Reject recall", lambda r: r["quality_gate"]["reject_recall"]),
                 ("Reject recall, all expected GT rejects", lambda r: r["quality_gate"].get("reject_recall_end_to_end")),
                 ("False-reject rate", lambda r: r["quality_gate"]["false_reject_rate"]),
                 ("Diagnosis accuracy (right metric blamed)", lambda r: r["quality_gate"]["diagnosis_accuracy"]),
                 ("Diagnosis success, all expected GT rejects", lambda r: r["quality_gate"].get("diagnosis_accuracy_end_to_end")),
                 ("Explanation coverage", lambda r: r.get("explanations", {}).get("explanation_coverage")),
                 ("Explanation: location correct", lambda r: r.get("explanations", {}).get("location_accuracy")),
                 ("Explanation: quantity correct", lambda r: r.get("explanations", {}).get("quantity_accuracy"))]
    for label, fn in gate_rows:
        L.append(f"| {label} | " + " | ".join(_fmt(fn(results[n])) for n in names) + " |")
    L += ["", "## Files", "", "- `evaluation_metrics.xlsx` - every table above plus top errors and per-sample rankings",
          "- `release_rankings_<model>.csv` - per sample: ground-truth vs predicted ranking of the 24 edited releases and best release",
          "- `figures/` - MAE per metric, GT-vs-prediction scatter per metric, quality-band confusion"]
    with open(os.path.join(out_dir, "evaluation_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out_dir, "evaluation_results.json"), "w") as f:
        json.dump(finite_json({n: {k: v for k, v in r.items() if k not in ("error_rows",)} for n, r in results.items()}),
                  f, indent=1, allow_nan=False)
    return os.path.join(out_dir, "evaluation_report.md")
