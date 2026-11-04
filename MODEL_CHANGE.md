# Sanjana's PRISM comparison: InternVL3.5-2B

This branch uses [OpenGVLab/InternVL3_5-2B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF),
with the native Transformers InternVL implementation and Apache-2.0 weights.

## Why this model

InternVL provides a document-capable vision encoder and dynamic image tiling in a compact model.
It supports Original + Release in one conversation and native 4-bit loading.
It is a different VLM architecture from Qwen2.5-VL, but **its text decoder is Qwen3**. Therefore
this is an architecture-level comparison, not a claim that all four models have unrelated language
backbones. SmolVLM2, Gemma and Phi provide the more independent model-family comparisons.

Sources: [official checkpoint](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF),
[native Transformers documentation](https://huggingface.co/docs/transformers/v4.56.2/model_doc/internvl).
These capabilities motivate the choice; they do not establish minute-change accuracy on PRISM.

## Exact scope

Only the model integration, model dependencies and model run instructions change:
- Native `InternVLForConditionalGeneration` and `InternVLProcessor`.
- `transformers==4.56.2`; the existing `trl==0.17.0` and `peft==0.15.2` are retained.
- Native 448 px tiles, capped by `model.max_image_patches: 4`; 256 visual tokens per tile.
  The processor may add an overview when it splits an image. No extra ROI detector or scoring aid.
- Paired image ordering, model-specific assistant token masking, left padding for generation
  and right padding for supervised training, without image-token truncation.
- A separate model checkpoint directory and Drive workspace.

The templates, 24-release generation ladder, labels, P1-P4 prompt wording, target text, splits,
QLoRA r/alpha/dropout/all-linear targets, optimizer, training schedule, validation checkpoint
selection, output gate/retry/resume logic, 12 metrics, score engine and evaluation remain unchanged.
All non-model configuration sections match the original team configuration.
No website files exist in this upstream repository; the earlier PRISM site is unchanged.

## Open and run

[Open this branch's Colab notebook](https://colab.research.google.com/github/Sanjana432/Samsung-prism-san/blob/internvl35-model/colab_run_pipeline.ipynb).

1. Select a GPU runtime. The notebook clones branch `internvl35-model` into
   `MyDrive/prism_ladder_internvl35`.
2. Supply the exact same team `data/` directory (originals, releases, metadata and split files)
   used by the other model branches. Do not copy another model's adapters or predictions.
   The unchanged generator is available if the shared dataset has not yet been created.
3. Install requirements. On GPUs below 20 GB, the notebook uses four native patches/image and
   inference batch size 2. Run the GPU smoke test before long stages.
4. Run the existing prompt-study/baseline, QLoRA training, inference and evaluation cells.
   Outputs remain in this model's separate workspace, with the same filenames and schemas.
5. On GPU memory failure, lower inference batch size to 1 first. The smoke test suggests
   lowering `model.max_image_patches` to 2 or 1 if necessary. Record any such reduction because
   less image detail can hide the minute differences being studied.

Weights use 4-bit NF4/double quantization; T4 uses fp16 and SDPA without FlashAttention.
The actual full-checkpoint T4 memory requirement and training throughput need a GPU run.

## Software verification

All 14 integration tests passed locally with Transformers 4.56.2, PEFT 0.15.2, TRL 0.17.0,
CPU PyTorch 2.14.0 and torchvision 0.29.0. All Python source files and notebook JSON parse;
the non-model configuration and prediction validation/retry/resume function match the team baseline.
An actual generated 768x1024 Original/Release pair with its full twelve-line reasoning target
uses 3,280 tokens with the default four-patch cap, within the unchanged 4,096-token limit.

The integration tests use real checkpoint processor/tokenizer files and a tiny randomly initialized
InternVL + Qwen3 architecture with the same LoRA settings for CPU forward/backward, finite gradient,
adapter save/reload and generation checks. They test software compatibility, not pretrained accuracy.

```bash
python -m unittest discover -s tests -v
```

Set `INTERNVL_PROCESSOR_DIR` to a local processor-only snapshot to enable native tests.
They are explicitly skipped without it. To download these small files without model weights:

```python
import os
from huggingface_hub import snapshot_download
os.environ["INTERNVL_PROCESSOR_DIR"] = snapshot_download(
    "OpenGVLab/InternVL3_5-2B-HF", allow_patterns=["*.json", "*.txt", "*.jinja"])
```

Then run the unittest command in that environment. The full GPU smoke test checks real 4-bit
loading, inference, masking, a LoRA forward/backward step and peak memory on your actual runtime.

Full pretrained inference, QLoRA training and test-set accuracy have not been run locally: this
machine has CPU PyTorch. Do not treat the synthetic software test outputs as model results.
