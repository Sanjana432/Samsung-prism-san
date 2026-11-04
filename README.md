# Samsung PRISM 26VI05 - VLM image-quality evaluation pipeline (English)

End-to-end pipeline: **dataset with pre-defined ground-truth scores -> pairwise JSONL -> zero-shot
benchmark -> QLoRA fine-tuning -> CoT inference -> validation gate -> Python score engine ->
agreement analytics -> diagnostic audit.**

## Sanjana's model comparison: Gemma4 E2B
This branch uses **`google/gemma-4-E2B-it`** (Apache-2.0, Google Gemma4),
an independent model family from the original Qwen2.5-VL experiment. See
[`MODEL_CHANGE.md`](MODEL_CHANGE.md) for the exact scope, verification and Colab instructions.
Generation, ground-truth labels, splits, P1-P4 prompts, QLoRA hyperparameters and evaluation stay unchanged.

## How to run
- **With Claude Code (local GPU):** open this folder in Claude Code and say
  *"Follow CLAUDE.md and run the whole pipeline."* It handles setup, fixes and the final report.
- **Manually:** see the numbered steps in `CLAUDE.md`, or `./run_all.sh`.
- **Google Colab:** upload this zip to Drive, open `colab_run_pipeline.ipynb`, choose a GPU runtime, Run all.

Hardware: CUDA GPU; Colab uses inference batch 1 below 20 GB. Gemma4 E2B has 5.1B total parameters,
including large unquantized lookup tables. Actual T4 memory fit and runtime must be checked with
the GPU smoke test before long stages; no benchmark accuracy or GPU memory measurements are supplied.

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
1. Training inputs - pairwise JSONL, native Gemma4 processor (image tokens never truncated), 4-bit NF4 base.
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

## What makes PRISM different
1. **It measures the smallest change a VLM can see.** Each of 13 defect types comes in 4 minute levels, so
   the evaluation reports a just-noticeable level per defect, instead of hiding blindness inside an average
   (a model blind to every change below 0.05 still gets a pooled MAE of ~0.009).
2. **Ground truth with zero manual annotation that anyone can audit.** Every score is computed from what was
   drawn (dE2000, pixels, % of characters) and ships with its formula; one script re-computes all of them and
   cross-checks the pixels.
3. **It knows WHAT is wrong and WHERE, and whether its explanation is true.** Training targets name the
   region; the evaluation scores location and quantity in the model's own explanation.
4. **It does not confuse defects.** Single-factor edits give an edit x metric leakage matrix and an
   attribution-precision score - impossible with mixed-defect datasets.
5. **It acts, not just grades.** `07_judge.py` returns ACCEPT/REJECT, the failing metric, the region and the
   repair action for any original/edited pair; the evaluation reports gate accuracy and diagnosis accuracy.
6. **It improves itself.** `08_data_engine.py` generates new training data only for the defects the model
   still misses, round after round.
7. **It tests generalisation.** 23 document/UI types with an unseen-template test option; this fork
   runs the Gemma4 E2B comparison with fp16/SDPA and 4-bit linear weights; verify actual memory fit first.
8. **Sanity built in.** Unchanged controls measure false alarms; ordering across levels measures consistency.

## No manual annotation
Every ground-truth score is computed automatically by the renderer during generation (stage 1) and
used directly for training and evaluation. There is no annotation sheet, rater step or human-label input.

## Layout
```
configs/config.yaml        all settings
prism/                     library (templates, renderer, metrics, prompts, gate, scoring, evaluation, report, vlm)
scripts/00..08             stages (07 judge any pair, 08 data engine); gpu_smoke_test.py;
                           make_pilot_showcase.py; check_ladder_scores.py
assets/fonts               bundled open-licence fonts (DejaVu, Liberation, Carlito, Caladea, Poppins, FreeFont)
```
