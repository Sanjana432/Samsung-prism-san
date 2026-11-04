# PRISM improved pipeline

The four `*-modified` branches extend the model-only versions with common correctness,
reproducibility and recovery fixes. The earlier model-only branches remain available.
This is a software implementation; full pretrained GPU accuracy and peak T4 memory remain
to be measured on the same shared dataset.

| Model | Improved code | Colab |
|---|---|---|
| SmolVLM2-2.2B | [Branch](https://github.com/Sanjana432/Samsung-prism-san/tree/smolvlm2-model-modified) | [Run](https://colab.research.google.com/github/Sanjana432/Samsung-prism-san/blob/smolvlm2-model-modified/colab_run_pipeline.ipynb) |
| Gemma 4 E2B | [Branch](https://github.com/Sanjana432/Samsung-prism-san/tree/gemma4-model-modified) | [Run](https://colab.research.google.com/github/Sanjana432/Samsung-prism-san/blob/gemma4-model-modified/colab_run_pipeline.ipynb) |
| InternVL 3.5 2B | [Branch](https://github.com/Sanjana432/Samsung-prism-san/tree/internvl35-model-modified) | [Run](https://colab.research.google.com/github/Sanjana432/Samsung-prism-san/blob/internvl35-model-modified/colab_run_pipeline.ipynb) |
| Phi-3.5 Vision | [Branch](https://github.com/Sanjana432/Samsung-prism-san/tree/phi35-model-modified) | [Run](https://colab.research.google.com/github/Sanjana432/Samsung-prism-san/blob/phi35-model-modified/colab_run_pipeline.ipynb) |

## Run

Open the matching Colab notebook, use a fresh GPU runtime and run its cells in order. Each
notebook clones its own modified branch into `MyDrive/prism_ladder_<model>_modified`. It
writes a separate, ignored `configs/colab_runtime.yaml` so GPU settings do not dirty the
tracked configuration or obstruct later updates. Every stage is a checked subprocess:
an error stops the notebook instead of being followed by a misleading success report.

Use the team's identical `data/` folder, sample IDs, labels and split membership for every
model. Move one copy between workspaces if Drive space is limited; keep model outputs in
their own workspaces. Existing measured data can be validated and adopted by stage 02;
the runner preserves it rather than regenerating images. Its original count must match
`ladder.n_originals`. Generating a new dataset uses the same 24-release ladder and renderer.

From a local terminal or Colab command cell:

```bash
python scripts/run_pipeline.py --cpu-only
python scripts/run_pipeline.py --stages baseline train infer evaluate
```

The first command generates missing data, publishes/validates splits, independently
recomputes scores and runs CPU self-tests. It does not run or claim pretrained inference.
The second validates data, runs the GPU smoke check and then the model stages, each in
its own process so GPU allocations are released between stages.

To supply a separate configuration, add `--config configs/colab_runtime.yaml`. GPU smoke,
generation, split validation, self-tests, training and evaluation accept that same option.
The checked runner records progress/failure in `reports/pipeline_status.json`.

## Changes beyond the model swap

**Generation and data:** completed Original/release sets are staged with recovery receipts,
validated and atomically published. Reruns reuse verified images instead of overwriting
them. Controls must be pixel-identical; edits must change pixels inside the allowed region.
IDs, dimensions, complete severity ladders and finite ground-truth scores are checked.
Labels remain deterministic synthetic renderer measurements, not human judgments.
Configuration and JSONL text use explicit UTF-8 on Windows and Linux.

**Shared splits:** every Original stays in exactly one split. Duplicate Originals and pair
IDs are rejected; existing assignments are frozen. Validation runs before publication;
an interrupted publication marker prevents GPU work against partially replaced files.
Manifests fingerprint records and actual image bytes, so reused filenames cannot hide
changed data. Empty splits require an explicit CPU-pilot option, not a silent benchmark.

**Training:** all train/validation pairs are checked against the native processor's context
budget before model weights are loaded. Full answer tokens and the generation reserve
must fit, with no silent image-token truncation. Training manifests bind model/revision,
processor, source code, configuration, image content, targets and library versions.
Only complete compatible optimizer/RNG checkpoints are resumed. The best adapter is
selected on validation generation MAE plus invalid-output penalty; invalid validation
cannot manufacture a successful best checkpoint. Earlier adapters are preserved when
replacing an adapter. Validation restores model/cache state even after errors. LoRA-only
trainables and finite nonzero gradients are checked before updates.

**Prediction:** output manifests bind the exact images, model/revision, adapter weights,
processor, system/rubric prompts, decoding budget, code and library versions. Incompatible
or unidentified legacy predictions are not silently mixed into a new run. A crash-truncated
last JSONL line can be recovered; corruption in complete records fails explicitly. Saved
scores reject booleans, numeric strings, duplicate keys and non-finite values. The effective
retry text stays attached to its final scores for honest explanation evaluation.

**Evaluation:** missing outputs remain in the full ground-truth denominator. Invalid,
duplicate and unknown records are handled explicitly. Rankings are tie-aware and report
incomplete groups; missing controls cannot become an imaginary perfect reference.
The release spreadsheet includes every expected edited release and its valid/invalid/missing
status; an incomplete group has no declared best release.
Explanations that are missing are counted as missing rather than excluded from accuracy.
Legitimate multi-metric effects, such as a bold font-family edit, receive attribution credit.
Conditional results, end-to-end coverage and common-valid-subset comparisons are identified
separately. All-invalid reports still export Excel/figures, with strict JSON `null` for
undefined values rather than non-standard `NaN`. Reports include hash-linked provenance.

**Adaptive data engine:** new training examples can be targeted only from validation-set
results whose report and dataset fingerprints match. Test/unknown/stale reports are rejected.
Rounds use unique namespaces, append without replacing earlier rounds, and exclude held-out
templates. Adaptive rounds are optional and are not part of the default benchmark run.

## Read results correctly

The unchanged 12 scoring dimensions and 24-release stimulus design make model results
comparable. Pooled MAE can look good when most metrics are unchanged; inspect per-artifact
severity detection, unchanged-control false alarms, attribution, missing-output coverage and
ranking too. Record resolution/tiling/crops and hardware because native processors differ.
Training on synthetic formula labels does not establish agreement with human reviewers or
real-world scans; those require a separately held-out annotated evaluation set.

The model-specific `MODEL_CHANGE.md` documents the architecture and dependency choices.
Gemma E2B has about 5.1B total parameters, so its effective-size name is not a GPU-memory
promise. The GPU smoke test remains necessary on the actual Colab allocation for every model.

## Verification and recovery

Verified locally on 4 November 2026, with each model's own dependencies and real
processor-only snapshot:

| Branch model | Tests passed | Skipped |
|---|---:|---:|
| SmolVLM2 | 71 | 0 |
| Gemma 4 E2B | 72 | 0 |
| InternVL 3.5 | 71 | 0 |
| Phi-3.5 Vision | 71 | 0 |

The 285 checks across the four branches cover data integrity, resume identity, invalid
outputs, evaluation coverage/ranking, checkpoint selection and each native model integration.
Tiny randomly initialized real architectures exercise forward/backward and adapter reload;
the Gemma suite also runs an actual one-step TRL trainer check. These do not measure the
pretrained weights.

A separate CPU end-to-end pilot generated 3 Originals and 75 pairs (3 unchanged controls
plus 72 edited releases, 78 images). All 900 metric labels were independently reproduced
with zero mismatches; all 75 locality checks passed, all controls were pixel-identical and
all 72 edits changed pixels. Dataset validation, split publication, all 23 renderer-template
identity checks and mock-predictor Excel/figure reporting completed successfully. All
41 Python files and 10 notebook cells in each branch passed syntax/structure checks.
An additional UTF-8 metadata round-trip passed in each branch with Python UTF-8 mode disabled.
Generated pilot data and mock reports are ignored, not distributed as model results.

```bash
python -m unittest discover -s tests -v
python scripts/validate_dataset.py
```

Native model integration tests need the corresponding processor-only snapshot environment
variable documented in `MODEL_CHANGE.md`; without it those tests are explicitly skipped.
The tests use tiny randomly initialized model weights for backward/adapter checks, not
full pretrained checkpoints. CPU self-test figures use explicitly named mock predictors.

If a run rejects changed data, settings, source or checkpoints, preserve that run and choose
a new configured outputs/predictions directory. Do not disable the checks to combine results
from different experiments. If split publication was interrupted, rerun stage 02 with the
same intact inputs. Retain the shared images and provenance files with every reported result.
