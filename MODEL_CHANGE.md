# PRISM comparison: Phi-3.5 Vision

> This is the improved `phi35-model-modified` branch. Start with
> [MODIFIED_PIPELINE.md](MODIFIED_PIPELINE.md) for the current run instructions,
> shared pipeline fixes, verification evidence and remaining GPU checks.

This branch uses
[`microsoft/Phi-3.5-vision-instruct`](https://huggingface.co/microsoft/Phi-3.5-vision-instruct).
Microsoft releases this 4.2B multimodal model under MIT and explicitly supports optical character
recognition, document understanding and multiple-image comparison. This is a research candidate;
minute-change detection accuracy on PRISM must be established by the shared benchmark.

## Model-specific implementation

- Official Phi decoder plus CLIP vision encoder, distinct from the Qwen2.5-VL baseline.
- The official checkpoint uses custom Python code. The configuration, processor, tokenizer, code
  and weights are pinned to revision `12b77fb40b63a2c73c68243d3f767aab688a1b2a`; `trust_remote_code`
  is limited to the selected model. The downloaded configuration/model/processor sources were
  reviewed before the CPU tests loaded them.
- `transformers==4.46.3`, `trl==0.17.0` and `peft==0.15.2` retain the Cache interfaces used by
  the official model. Do not replace this with a latest-Transformers install in the same session.
- Native multi-image input uses four 336 px crops plus an overview per image, with Original first
  and Release second. Phi's processor encodes one prompt at a time; this branch pads the resulting
  text and concatenates vision features in pair order, preserving batch behavior.
- Negative image placeholder IDs stay negative in `input_ids` and have attention mask 1. They
  are masked to -100 only in training labels, together with prompt/padding positions. Integer
  `input_ids`, masks and image sizes are not cast to fp16.
- Inference, QLoRA training and the GPU smoke test use one common model-loading function:
  NF4/double quantization, native fp16 on T4 (bf16 only on supporting GPUs), eager attention.
  FlashAttention is not required.

The renderer, templates, 24-release design, metric definitions, P1-P4 wording and QLoRA hyperparameters retain the team experiment.
This improved branch also adds shared data, checkpoint, prediction and evaluation fixes in
[MODIFIED_PIPELINE.md](MODIFIED_PIPELINE.md).

## Access and run

1. [Open this branch in GitHub](https://github.com/Sanjana432/Samsung-prism-san/tree/phi35-model-modified)
   or [open its notebook in Colab](https://colab.research.google.com/github/Sanjana432/Samsung-prism-san/blob/phi35-model-modified/colab_run_pipeline.ipynb).
2. Choose a GPU runtime. The notebook clones this branch into `MyDrive/prism_ladder_phi35_modified`,
   keeping weights, adapters, predictions and reports separate from the other model versions.
3. Copy the team's identical `data/` folder (images, metadata and splits) into that workspace
   before running dataset cells if it already exists. Do not copy another model's `outputs/`
   or `predictions/`. Generation remains available with the same stimulus design and new recovery checks.
4. Install the branch's `requirements.txt` and run the existing dataset checks, then
   `scripts/gpu_smoke_test.py` before longer model stages. The smoke test measures real 4-bit
   inference, a LoRA backward step and peak CUDA memory. On T4 the notebook reduces inference
   batch size to 2; if memory is insufficient, use 1 first. Lowering crops can hide minute
   changes, so record that choice in comparisons.
5. Run the existing prompt study, zero-shot baseline, training, inference and evaluation cells.
   Compare scores using the same source IDs, releases, split manifest, prompt rubric and metrics.

## Initial model integration verification

The results below describe the original model integration. For the expanded modified-branch
regressions and CPU pilot, see [MODIFIED_PIPELINE.md](MODIFIED_PIPELINE.md).

Verified on 4 November 2026: **14/14 integration tests passed** with Transformers 4.46.3,
PEFT 0.15.2, TRL 0.17.0, CPU PyTorch 2.14.0 and torchvision 0.29.0. The real 768x1024
generated invoice with its full twelve-line reasoning target used **2384 tokens** (357 supervised)
at four crops per image, fitting the unchanged 4096-token limit. Batched two-pair logits matched
separate single-pair outputs, confirming that repeated -1/-2 image IDs do not mix samples.
All 31 Python files parse; notebook JSON and `git diff --check` pass; all non-model config sections
match the original `1ae046558905c0bba20198ea077fec2cb0ffc627` commit.

CPU integration tests use the real official tokenizer and image processor, plus a randomly
initialized reduced Phi decoder and one-layer CLIP encoder with the real HD image projection.
They test pair ordering, active negative image IDs, assistant-only labels, generation suffix
decoding, retry/resume, LoRA gradients and adapter reload. Reduced weights are not a pretrained
model and these checks cannot establish PRISM accuracy or full-model GPU memory requirements.

```bash
python -m unittest discover -s tests -v
```

Set `PHI35_PROCESSOR_DIR` to the local pinned metadata/code snapshot to enable native tests.
Download only the small files, without the checkpoint weights, in the test environment:

```python
import os
from huggingface_hub import snapshot_download
os.environ['PHI35_PROCESSOR_DIR'] = snapshot_download(
    'microsoft/Phi-3.5-vision-instruct', revision='12b77fb40b63a2c73c68243d3f767aab688a1b2a',
    allow_patterns=['*.json', '*.txt', '*.model', '*.py'])
```

Full 4-bit CUDA execution, end-to-end QLoRA and pretrained accuracy still require a Colab GPU run.
No trained checkpoint or measured PRISM model accuracy is included in this branch.
