# PRISM model branch: Google Gemma4 E2B

Model: [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it), Apache-2.0.
Branch: `gemma4-model`. This is a distinct Gemma architecture, independent of Qwen.

Gemma4 supports interleaved image pairs and native system instructions. The E2B version is a
resource-conscious candidate from the current Google model family. Its published name means
approximately 2.3B effective parameters, **5.1B total including embeddings**. It is not a 2B-memory model.
No ranking against the other PRISM models is claimed before the shared dataset is evaluated.

## What changes

- `configs/config.yaml`: Gemma4 checkpoint/local directory and native per-image soft-token budget.
- `prism/vlm.py`: native Gemma4 loader/processor, nested original-release images, native assistant
  header, and `enable_thinking=False`. The existing P4 observation instructions are retained.
- Training collator: mask prompt, padding and visual tokens using Gemma's own token IDs.
- Training preparation: PEFT casts frozen unquantized lookup tables to fp32 by default; this branch
  performs the equivalent freezing, nonquantized-parameter upcast and non-reentrant checkpointing
  while excluding `nn.Embedding` weights and tied references from that cast in the first place.
  This avoids both persistent and transient fp32 allocations for the large per-layer/input tables.
  Layer norms retain PEFT's fp32 treatment.
  The same helper is used in the real training stage and GPU smoke test.
- `prism/common.py`: Gemma4's combined processor configuration, Jinja chat template, and single/sharded weights.
- Model dependencies: Transformers 5.17.0, PEFT 0.21.0, TRL 1.13.0 and native torchvision processing.
  These newer APIs are required by Gemma4; the original TRL path and Trainer fallback are retained.
- Colab notebook and model notes identify the correct model branch and separate Drive workspace.

Generation, templates, the 24-release ladder, measured labels, P1-P4 wording, split assignment,
metric definitions, QLoRA r/alpha/dropout/all-linear targets, optimizer schedule, checkpoint selection,
validation retry rules, scoring and evaluation calculations stay the same. The other branches and
earlier local website are separate from this model integration.

## Access and run

1. [Open this branch in GitHub](https://github.com/Sanjana432/Samsung-prism-san/tree/gemma4-model).
2. [Open its notebook in Colab](https://colab.research.google.com/github/Sanjana432/Samsung-prism-san/blob/gemma4-model/colab_run_pipeline.ipynb).
3. Select a GPU runtime, then run setup. This branch uses `MyDrive/prism_ladder_gemma4`, so adapters,
   predictions and checkpoints cannot accidentally resume a different model's run.
4. Copy the team's shared `data/` directory with original/release images, metadata and exact split
   files into this workspace before dataset cells. Do not copy another model's outputs/predictions.
   The unchanged generator remains available when the shared dataset has not yet been generated.
5. Run `scripts/gpu_smoke_test.py` before long jobs. It runs real inference, checks the label mask,
   runs one 4-bit LoRA training step and reports peak GPU memory. Colab sets inference batch size 1
   below 20 GB; the native 280-soft-token per-image budget is retained.
6. Run the existing baseline, QLoRA, inference and evaluation stages. Compare the same test sample IDs
   and split membership across branches. Record native model preprocessing alongside results.

NF4 quantization applies to linear weights, not lookup tables. CUDA training memory also includes
vision activations, gradients, adapters and optimizer state. Preserving frozen embedding dtype
reduces one important cost, but **full pretrained T4 inference/training memory fit remains unverified**.
If a T4 still runs out of memory, use a larger GPU or record any lower vision budget explicitly;
reducing resolution can hide the minute discrepancies being studied.

## Verification

Verified on 4 November 2026: **15/15 integration tests passed**, including native processor/mask
checks, a real 768×1024 rendered pair with the complete 12-line target (**1,285/4,096 tokens**),
tiny Gemma4 forward/backward and adapter reload, and one actual TRL SFTTrainer optimizer step.
The CPU environment used Transformers 5.17.0, PEFT 0.21.0, TRL 1.13.0, PyTorch 2.14.0 and
torchvision 0.29.0. All non-model configuration sections match the original upstream commit.

Run `python -m unittest discover -s tests -v`. Set `GEMMA4_PROCESSOR_DIR` to a local copy of the
public processor files to enable native processor tests:

```python
import os
from huggingface_hub import snapshot_download
os.environ['GEMMA4_PROCESSOR_DIR'] = snapshot_download(
    'google/gemma-4-E2B-it', allow_patterns=['*.json', '*.jinja', '*.txt', '*.model'])
```

The tests check real paired-image processing, unchanged prompt text, inference-only decoding,
assistant/image/padding masks, overlength rejection, prediction retry/resume, local checkpoint
validation, frozen embedding dtype preservation, and forward/backward + adapter reload through
a tiny randomly initialized instance of the actual Gemma4 architecture. The latter uses a compact
test vocabulary to avoid allocating full 262K-vocabulary logits; native vocabulary processing is
checked separately. These are integration checks, not measurements of pretrained model accuracy.

Run the unchanged `python scripts/00_selftest.py` for renderer and deterministic scoring checks.
Full pretrained GPU inference, QLoRA training, accuracy and minute-change detection thresholds
must be measured on the actual Colab GPU and shared test set.
