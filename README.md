# Samsung PRISM 26VI05 - VLM image-quality evaluation pipeline (English)

> This is the improved `codex/smolvlm2-model-modified` branch. Start with
> [MODIFIED_PIPELINE.md](MODIFIED_PIPELINE.md) for the current run instructions,
> shared pipeline fixes, verification evidence and remaining GPU checks.

End-to-end pipeline: **dataset with pre-defined ground-truth scores -> pairwise JSONL -> zero-shot
benchmark -> QLoRA fine-tuning -> CoT inference -> validation gate -> Python score engine ->
agreement analytics -> diagnostic audit.**

## Sanjana's model comparison: SmolVLM2
This fork uses **`HuggingFaceTB/SmolVLM2-2.2B-Instruct`** (Apache-2.0, SmolLM2 + SigLIP),
an independent model family from the original Qwen2.5-VL experiment. See
[`MODEL_CHANGE.md`](MODEL_CHANGE.md) for the exact scope, verification and Colab instructions.
The model integration is combined with the shared data, training, prediction and evaluation fixes described in the guide.

## How to run
1. Open this model's notebook using the links in [MODIFIED_PIPELINE.md](MODIFIED_PIPELINE.md).
2. Choose a fresh Colab GPU runtime, supply the team's identical dataset when available, and run the cells in order.
3. Run the GPU smoke check before baseline, training and inference. Inspect the evaluation report and coverage together.

For local execution, use `python scripts/run_pipeline.py --cpu-only` for generation and CPU checks,
then `python scripts/run_pipeline.py --stages baseline train infer evaluate` for the model stages.
Add `--config configs/colab_runtime.yaml` when using the notebook's generated settings.
Full pretrained accuracy and peak GPU memory still require an actual Colab run.

## The dataset: the "24 generation" minute-change ladder (`prism/ladder.py`)
Every Original gets 1 unchanged control + **6 single-factor edit families x 4 minute levels = 24 releases**
(the prism_iqa "24 generation"). Each release changes ONE text block of the document; everything else is
rendered exactly as the Original (audited per release). The renderer **measures what it actually produced**
and stores all 12 metric scores as ground truth - no manual annotation.

| Edit family | Levels | Metric it moves |
|---|---|---|
| font_family | same class, other class, third class, other class bold | Font Similarity (+ Semantic Emphasis for bold) |
| font_size | x1.01, x1.025, x1.05, x1.10 | Font Similarity |
| font_color | +2, +5, +10, +20 grey levels | Font Similarity (CIEDE2000 of the ink) |
| inpaint_color_mismatch | 1, 3, 6, 12 | Background Fidelity (CIEDE2000 of the re-filled patch) |
| justification | 1, 2, 4, 8 px | Alignment Preservation |
| line_spacing | +1, +2, +4, +8 px | Reading Flow |

300 Originals (13 per template, all 23 templates) x 25 = **7,500 pairs** (7,800 images incl. the Originals). `ladder.knobs` can
add 7 more families (look-alike characters, ghost text, grain smoothing, soft shadow, local JPEG, object
shift, heading shift) so 11 of the 12 metrics get their own ladder.

**23 English templates:** invoice, receipt, certificate, application form, business letter,
restaurant menu, event flyer, mobile settings screen, bank statement, payslip, boarding pass, medical
prescription, resume, newsletter, nutrition label, ID card, shipping label, report card, utility bill,
email, web landing page, chat screenshot, presentation slide (incl. dark-mode and landscape variants) - all randomised (names, items, prices, colours,
fonts from 11 bundled open-licence families, paper texture, logos, stamps, QR codes, signatures, photos).
Drop your own photos into `assets/photos/` and the flyer/menu/form photo areas will use them.

### Metadata
- `data/metadata/ladder.jsonl` - one record per Original: template, seed, original path, `ocr_data`
  (line text + bounding boxes), the edited `target_block`, and per release: image path, `artifact_type`,
  `level`, ground-truth `scores`, `category_scores`, `score_calculation` (written-out formula per metric),
  `pixel_measurements`, `control_parameters`, `reasoning`, `render_audit`.
- `data/metadata/ladder_metadata.csv` - one row per Original-Release pair (opens in Excel);
  `metadata_ladder.json` - the same in the prism_iqa `metadata_real.json` schema; `ladder_summary.json`.
- `data/pairs/{train,val,test}.jsonl` - pairwise training records (Original + one release + CoT/JSON target),
  split 70/15/15 **by sample_id** (no leakage), stratified by template.
- `scripts/check_ladder_scores.py` re-computes every stored score from its measurements.
- `pilot/` - **5 images per artifact type** (each changes one metric only) with contact sheets in
  `pilot/sheets/` and `pilot_metadata.json` - a visual sanity check of each artifact type.

## Model pipeline (matches the Training & Evaluation slide)
1. Training inputs - pairwise JSONL, native SmolVLM2 processor (image tokens never truncated), 4-bit NF4 base.
2. QLoRA adapters - all-linear, r 16 / alpha 32 / dropout 0.05.
3. Optimisation - TRL SFTTrainer, cosine LR, warmup 5%, grad-accum 16, validation checkpointing.
4. Validation gate - JSON schema + numeric range check, one retry, failures logged (`prism/validation.py`).
5. Inference scoring - forced CoT (`<analysis>` one line per metric) then the 12 scores (`prism/vlm.py`).
6. Python engine - category averages, overall, ranking of the 24 releases, best release (`prism/scoring.py`).
7. Agreement analytics - MAE, RMSE, PLCC, SROCC (`prism/evaluation.py`).
8. Diagnostic audit - top errors (|error| > 0.30), failure patterns, error by severity.

Checkpoint selection uses **validation-set generation MAE**, not training loss.

## Accuracy measures reported
Per metric, per category, overall and per template: MAE, RMSE, PLCC, SROCC, bias, accuracy within
+-0.05 / +-0.10 / +-0.15, quality-band accuracy and macro-F1 (excellent/good/fair/poor). Ranking:
best-release accuracy, exact ranking accuracy and Kendall tau over the 24 edited releases of each Original,
pairwise preference accuracy. Ladder sensitivity: detection rate per level, just-noticeable level,
monotonicity, false alarms on the controls, attribution precision.
Coverage: valid-output rate after the gate. Baseline vs fine-tuned side by side.

## What the pipeline measures
- Four severity levels for each of the six default edit families, plus unchanged controls.
  Seven optional families are available in configuration.
- Synthetic measurement labels from the renderer, including colour distance, geometry and text changes.
- Missing-output coverage, control false alarms, severity ordering, attribution and release ranking,
  alongside MAE/RMSE/PLCC/SROCC. Explanations are checked against the known edited region.
- An optional validation-driven data engine, with test-set results excluded from targeting new training examples.
- Generalisation to 23 document/UI templates, with an optional held-out-template test.

These are implemented evaluation capabilities. Whether a pretrained or fine-tuned model detects
the smallest discrepancies must be established by its held-out results.

## Ground-truth labels
The renderer creates deterministic synthetic measurement labels for training and evaluation.
They describe controlled changes; they are not human quality judgments. Agreement with human
reviewers and real scans requires a separate held-out annotated dataset.

## Layout
```
configs/config.yaml        all settings
prism/                     library (templates, renderer, metrics, prompts, gate, scoring, evaluation, report, vlm)
scripts/00..08             stages (07 judge any pair, 08 data engine); gpu_smoke_test.py;
                           make_pilot_showcase.py; check_ladder_scores.py
assets/fonts               bundled open-licence fonts (DejaVu, Liberation, Carlito, Caladea, Poppins, FreeFont)
```
