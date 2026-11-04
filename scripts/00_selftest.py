"""CPU self-test of everything that does not need a GPU. Run after stages 01 + 02.

Checks: renderer identity (all-zero release == original), target->realised fidelity, validation
gate, score engine, and the full evaluation/report path using mock predictors
(oracle+noise, constant, oracle+noise with 10% malformed outputs).
Exit code 0 = all good.
"""

import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prism.common import P, load_config, read_jsonl, write_jsonl  # noqa: E402
from prism.metrics import METRIC_KEYS  # noqa: E402
from prism.perturb import sample_release_severities  # noqa: E402
from prism.reasoning import build_target  # noqa: E402
from prism.render import render  # noqa: E402
from prism.scoring import category_scores, rank_releases  # noqa: E402
from prism.templates import TEMPLATES, build_document  # noqa: E402
from prism.validation import validate_output  # noqa: E402

FAIL = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


def main(config=None):
    cfg = load_config(config)
    print("[1] renderer identity")
    for t in TEMPLATES:
        d = build_document(t, 7)
        a, _ = render(d)
        b, _ = render(d, {k: 0.0 for k in METRIC_KEYS}, seed=99)
        check(np.array_equal(np.asarray(a), np.asarray(b)), f"{t}: zero-severity release identical to original")

    print("[2] target -> realised fidelity (16 random releases)")
    errs = {k: [] for k in METRIC_KEYS}
    rng = np.random.default_rng(1)
    for i in range(16):
        t = list(TEMPLATES)[i % len(TEMPLATES)]
        d = build_document(t, 1000 + i)
        sev = sample_release_severities(["technical", "visual", "human_perception"][i % 3], rng, cfg)
        _, info = render(d, sev, seed=i)
        for k in METRIC_KEYS:
            errs[k].append(abs(sev[k] - info["realised_severity"][k]))
    for k in METRIC_KEYS:
        m = float(np.mean(errs[k]))
        # emphasis/OCR are count-quantised, so allow a looser bound
        check(m < (0.08 if k in ("semantic_emphasis", "ocr_accuracy", "font_similarity") else 0.03),
              f"{k}: mean |target - realised| = {m:.4f}")

    print("[3] validation gate")
    good = build_target({k: 0.1 for k in METRIC_KEYS}, _dummy_details())
    s, e = validate_output(good)
    check(s is not None and abs(s["ocr_accuracy"] - 0.9) < 1e-6, "parses CoT + JSON target")
    check(validate_output("no json here")[0] is None, "rejects text without JSON")
    bad = json.dumps({k: 1.4 for k in METRIC_KEYS})
    check(validate_output(bad)[0] is None, "rejects out-of-range values")
    miss = json.dumps({k: 0.5 for k in METRIC_KEYS[:-1]})
    check(validate_output(miss)[0] is None, "rejects missing keys")
    named = json.dumps({"OCR Accuracy": 0.5, **{k: 0.5 for k in METRIC_KEYS[1:]}})
    check(validate_output(named)[0] is not None, "accepts display-name keys")
    trailing = "{" + ", ".join(f'"{k}": 0.5' for k in METRIC_KEYS) + ",}"
    check(validate_output(trailing)[0] is not None, "tolerates a trailing comma")

    print("[4] score engine")
    cs = category_scores({k: 1.0 for k in METRIC_KEYS})
    check(abs(cs["overall"] - 1.0) < 1e-9, "all-ones -> overall 1.0")
    rk, _ = rank_releases({"R1": {k: 0.9 for k in METRIC_KEYS}, "R2": {k: 0.5 for k in METRIC_KEYS},
                           "R3": {k: 0.7 for k in METRIC_KEYS}})
    check(rk == ["R1", "R3", "R2"], f"ranking {rk}")

    print("[5] evaluation path with mock predictors")
    test_path = P(cfg, "pairs_dir", "test.jsonl")
    if not os.path.exists(test_path) or not read_jsonl(test_path):
        print("  SKIP  no data/pairs/test.jsonl - run stages 01 and 02 first")
    else:
        test = read_jsonl(test_path)
        pdir = os.path.join(P(cfg, "predictions_dir"), "selftest")
        r = np.random.default_rng(0)

        def noisy(sc, sd):
            return {k: float(np.clip(sc[k] + r.normal(0, sd), 0, 1)) for k in METRIC_KEYS}
        write_jsonl(os.path.join(pdir, "mock_oracle_noise05_test.jsonl"),
                    [{"id": x["id"], "pred": noisy(x["scores"], 0.05)} for x in test])
        write_jsonl(os.path.join(pdir, "mock_constant_test.jsonl"),
                    [{"id": x["id"], "pred": {k: 0.85 for k in METRIC_KEYS}} for x in test])
        rows = []
        for x in test:
            raw = build_target({k: 1 - v for k, v in noisy(x["scores"], 0.10).items()}, _dummy_details())
            if r.random() < 0.10:
                raw = raw[: len(raw) // 2]  # truncated / malformed output
            sc, _ = validate_output(raw)
            rows.append({"id": x["id"], "pred": sc, "raw": raw})
        write_jsonl(os.path.join(pdir, "mock_noise10_malformed_test.jsonl"), rows)
        spec = importlib.util.spec_from_file_location("ev", os.path.join(ROOT, "scripts", "06_evaluate.py"))
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        res = ev.run_evaluation(cfg, ["mock_constant", "mock_noise10_malformed", "mock_oracle_noise05"], "test",
                                pred_dir=pdir, out_dir=os.path.join(P(cfg, "reports_dir"), "selftest"),
                                extra={"NOTE": "SELF-TEST with mock predictors - not model results"})
        o, c, m = res["mock_oracle_noise05"], res["mock_constant"], res["mock_noise10_malformed"]
        # most ladder scores are exactly 1.0, where clipping at 1 halves the noise: expect 0.02-0.04
        check(0.015 < o["all_metrics_pooled"]["MAE"] < 0.045, f"oracle+N(0,.05) MAE 0.02-0.04 (got {o['all_metrics_pooled']['MAE']:.4f})")
        check(o["all_metrics_pooled"]["MAE"] < c["all_metrics_pooled"]["MAE"], "oracle beats constant predictor")
        check(0.8 < m["coverage"]["valid_rate"] < 0.97, f"malformed outputs caught by gate (valid rate {m['coverage']['valid_rate']:.2f})")
        check(os.path.exists(os.path.join(P(cfg, "reports_dir"), "selftest", "evaluation_metrics.xlsx")), "xlsx report written")

    print("[6] minute-change ladder (3 Originals, configured edit families)")
    import tempfile
    from prism.common import abspath
    from prism.evaluation import ladder_sensitivity
    from prism.ladder import KNOB_METRIC, ORDINAL_KNOBS, make_ladder_sample
    lcfg = dict(cfg, paths=dict(cfg["paths"], images_dir=os.path.relpath(tempfile.mkdtemp(prefix="prism_ladder_"), ROOT)))
    recs = [make_ladder_sample(lcfg, i) for i in (0, 9, 20)]
    for rec in recs:
        rels = {x["release_id"]: x for x in rec["releases"]}
        o = np.asarray(_open(abspath(rec["original_image_path"])))
        c = np.asarray(_open(abspath(rels["control_0"]["release_image_path"])))
        check(np.array_equal(o, c), f"{rec['template']}: control identical to Original")
        check(all(x["render_audit"]["changed_pixels_outside_region"] == 0 for x in rec["releases"]),
              f"{rec['template']}: no pixel changed outside the edited block")
        check(all(not x["render_audit"]["no_visible_change"] for x in rec["releases"] if x["release_id"] != "control_0"),
              f"{rec['template']}: every one of the {len(rec['releases']) - 1} edits changes pixels")
        for knob in [k for k in ORDINAL_KNOBS if k in rec["knobs"]]:
            s = [rels[f"{knob}_{lv}"]["scores"][KNOB_METRIC[knob]] for lv in range(1, 5)]
            check(all(a >= b for a, b in zip(s, s[1:])) and s[0] < 1.0, f"{rec['template']}: {knob} scores ordered {s}")
        side = [k for x in rec["releases"] if x["artifact_type"] in ORDINAL_KNOBS
                for k in METRIC_KEYS if k != x["primary_metric"] and x["scores"][k] != 1.0]
        check(not side, f"{rec['template']}: ordinal edits move only their own metric {sorted(set(side))}")
    spec = importlib.util.spec_from_file_location("chk", os.path.join(ROOT, "scripts", "check_ladder_scores.py"))
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    bad = [(x["release_id"], k) for r in recs for x in r["releases"]
           for k, s in chk.severities(x["control_parameters"]["local"], x["control_parameters"]["emph_total"]).items()
           if abs(round(1 - round(min(1.0, max(0.0, s)), 4), 2) - x["scores"][k]) > 0.0100001]
    check(not bad, f"every ladder score re-computes from its stored measurements {bad[:3]}")
    check(all(px for r in recs for x in r["releases"] for _, px in chk.pixel_checks(x) if x["artifact_type"] == "control"),
          "pixel evidence: controls are pixel-identical")
    tgt = lambda x: ("<analysis>" + chr(10) + x["reasoning"] + chr(10) + "</analysis>" + chr(10))  # noqa: E731
    lrows = [{"sample_id": r["sample_id"], "release_id": x["release_id"], "kind": "ladder", "artifact_type": x["artifact_type"],
              "level": x["level"], "artifact_value": x["artifact_value"], "primary_metric": x["primary_metric"],
              "gt": x["scores"], "pred": x["scores"], "target": tgt(x), "raw": tgt(x)} for r in recs for x in r["releases"]]
    from prism.evaluation import explanation_check, quality_gate
    from prism.judge import verdict
    ex = explanation_check(lrows)
    check(ex["location_accuracy"] == 1.0 and ex["quantity_accuracy"] == 1.0 and ex["n_quantity"] > 30,
          f"explanation check: a truthful explanation scores 1.0 ({ex['n_quantity']} quantities)")
    g = quality_gate(lrows, 0.95)
    check(g["decision_accuracy"] == 1.0 and g["diagnosis_accuracy"] == 1.0, "quality gate: oracle decides and diagnoses perfectly")
    import re
    def lie(t):  # every reported quantity x -> 3x + 5, every location swapped
        t = re.sub(r"(\d+(?:\.\d+)?)(\s*(?:px|%))", lambda m: f"{float(m.group(1)) * 3 + 5:g}{m.group(2)}", t)
        return t.replace("top", "TMP").replace("bottom", "top").replace("TMP", "bottom")
    lied = explanation_check([dict(x, raw=lie(x["raw"])) for x in lrows])
    check(lied["quantity_accuracy"] == 0.0 and lied["location_accuracy"] < 1.0,
          f"explanation check: catches wrong quantities and locations ({lied['quantity_accuracy']}, {lied['location_accuracy']:.2f})")
    v = verdict({k: (0.80 if k == "reading_flow" else 1.0) for k in METRIC_KEYS})
    check(v["decision"] == "REJECT" and v["flags"][0]["metric"] == "reading_flow", "judge verdict: flags the failing metric")
    oracle = ladder_sensitivity(lrows)
    check(all(v and v["level"] == 1 for v in oracle["jnd"].values()) and oracle["control"]["false_alarm_rate"] == 0,
          "ladder analysis: oracle detects every level-1 edit with no false alarms")
    const = ladder_sensitivity([dict(x, pred={k: 0.9 for k in METRIC_KEYS}) for x in lrows])
    check(all(v is None for v in const["jnd"].values()) and const["control"]["false_alarm_rate"] == 1,
          "ladder analysis: constant predictor detects nothing and always false-alarms")
    import shutil
    shutil.rmtree(os.path.join(ROOT, lcfg["paths"]["images_dir"]), ignore_errors=True)

    print("\nSELFTEST", "FAILED: " + "; ".join(FAIL) if FAIL else "PASSED")
    sys.exit(1 if FAIL else 0)


def _open(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def _dummy_details():
    return {"ocr_cer": 0.02, "ocr_examples": [["Total", "Tota1"]], "align_mean_shift_px": 4, "font_family_term": 0.0,
            "font_size_mult": 1.03, "obj_scale": 0.96, "obj_shift_px": 3, "obj_opacity": 0.93, "n_objects": 3,
            "bg_delta_e2000": 2.5, "tex_detail_loss": 0.1, "ghost_alpha": 0.05, "light_drop": 0.05,
            "light_angle_deg": 90, "line_spacing_mult": 1.06, "anchor_shrink": 0.03, "anchor_shift_px": 2,
            "emph_lost": 1, "emph_total": 8, "jpeg_quality": 88}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    main(parser.parse_args().config)
