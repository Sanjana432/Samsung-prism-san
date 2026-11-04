# Sanjana's PRISM model comparison: SmolVLM2

> This is the improved `smolvlm2-model-modified` branch. Start with
> [MODIFIED_PIPELINE.md](MODIFIED_PIPELINE.md) for the current run instructions,
> shared pipeline fixes, verification evidence and remaining GPU checks.

This branch replaces the Qwen2.5-VL-3B integration with
[`HuggingFaceTB/SmolVLM2-2.2B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct).
It keeps the team's experiment and evaluation pipeline in place.

## Why this model

- It is an Apache-2.0 model with a SmolLM2 language decoder and SigLIP vision encoder, providing
  a model family independent of Qwen for the team comparison.
- It accepts multiple images in one conversation and has native support in the repository's
  existing `transformers==4.51.3`, including PEFT/LoRA training.
- Its 2.2B size makes it a practical candidate for limited-memory Colab GPUs. The integration
  retains 4-bit NF4/double quantization, native fp16 on T4 and SDPA attention; FlashAttention is not required.
- This is a choice based on architecture, features and resource constraints. It is not a claim
  that SmolVLM2 outperforms Qwen, or that its minute-change accuracy has already been measured.

## Exact change scope

| File | Model-related change |
|---|---|
| `configs/config.yaml` | SmolVLM2 model ID, separate local checkpoint directory, native image-size setting |
| `prism/vlm.py` | Native model class, paired-image processing, template-derived assistant header, floating-point vision input dtype |
| `scripts/04_train_qlora.py` | Use the shared SmolVLM pair encoder and native assistant/image label mask |
| `prism/common.py` | Validate SmolVLM local checkpoint files; allow single or sharded safetensors weights |
| `requirements.txt` | Replace `qwen-vl-utils` with the SmolVLM processor's `num2words` dependency |
| `colab_run_pipeline.ipynb` | Clone this model branch into its own Drive workspace and use SmolVLM T4 image settings |
| `scripts/gpu_smoke_test.py` | Update the model-specific image-size recommendation |
| `tests/test_smolvlm.py` | Integration regression checks |
| `README.md`, `PIPELINE.md`, this file | Model identification and run instructions |

The renderer, templates, 24-release design, metric definitions, P1-P4 wording and QLoRA hyperparameters retain the team experiment.
This improved branch also changes integrity, checkpoint/resume safety, validation and evaluation behavior as documented in
[MODIFIED_PIPELINE.md](MODIFIED_PIPELINE.md).

## Run on Colab

1. [Open the model branch's notebook in Colab](https://colab.research.google.com/github/Sanjana432/Samsung-prism-san/blob/smolvlm2-model-modified/colab_run_pipeline.ipynb).
2. Select a GPU runtime. The setup downloads this branch into
   `MyDrive/prism_ladder_smolvlm2_modified`, separate from the Qwen experiment's `prism_ladder` folder.
3. Run the setup cells. On GPUs below 20 GB the notebook sets
   `model.image_longest_edge: 768` and `inference.batch_size: 2`.
4. Use the **same original images, generated releases, metadata and split files** as the team's
   Qwen comparison. Copy the existing `data/` directory into this model workspace before running
   the dataset cells when it is already available. Do not copy Qwen's `outputs/`, `predictions/`
   or model-specific reports. Dataset generation is still available, with the same seed and code,
   if the team has not generated its shared dataset yet.
5. Run `scripts/gpu_smoke_test.py` before the long model stages. It checks native processor input,
   answer masking, inference, one real 4-bit LoRA forward/backward step and peak GPU memory.
   If it runs out of memory, reduce inference batch size to 1 first; reducing image resolution
   can remove the minute discrepancies being studied and must be recorded in the comparison.
6. Run the existing prompt study/baseline, training, inference and evaluation cells. Stage names,
   output formats and the 12 quality metrics remain the same.

For a fair cross-model comparison, match the sample IDs, train/validation/test membership, prompts
and scoring rules, and record the model's preprocessing and training settings. The model's own
tiling is different from Qwen's pixel-budget processor; there is no added crop, detector or scoring aid.

## Initial model integration verification

The results below describe the original model integration. For the expanded modified-branch
regressions and CPU pilot, see [MODIFIED_PIPELINE.md](MODIFIED_PIPELINE.md).

Verified locally on 4 November 2026: **14/14 integration tests passed** with the actual public
processor files and `transformers==4.51.3`, `peft==0.15.2`, `trl==0.17.0` (CPU PyTorch 2.14.0).
All 31 Python source files parse, the notebook JSON is valid, `git diff --check` passes and all
non-model configuration sections match the original commit `1ae046558905c0bba20198ea077fec2cb0ffc627`.
The unchanged repository CPU self-test passed, with the saved-dataset report test skipped as below.
A real 768x1024 Original/Release pair with all 12 measured reasoning lines and JSON scores uses
2,461 of the 4,096 allowed training tokens with the default native image settings.

The integration suite uses the real public tokenizer, chat template and image processor, and a
tiny randomly initialized instance of the actual SmolVLM architecture for CPU training checks.
It does not use or measure the full pretrained checkpoint.

```bash
python -m unittest discover -s tests -v
```

Set `SMOLVLM_PROCESSOR_DIR` to a local copy of the model's processor/tokenizer files to enable all
native-processor tests. Without it, those tests are explicitly skipped. No model weights are needed:

```python
import os
from huggingface_hub import snapshot_download
os.environ['SMOLVLM_PROCESSOR_DIR'] = snapshot_download(
    'HuggingFaceTB/SmolVLM2-2.2B-Instruct', revision='482adb537c021c86670beed01cd58990d01e72e4',
    allow_patterns=['*.json', '*.txt', '*.model'])
```

Then launch the unittest command in the same environment. The repository's unchanged CPU self-test is:

```bash
python scripts/00_selftest.py
```

The CPU self-test covers all 23 template identity checks, measured-score fidelity, the validation
gate, score engine and 3 complete 24-release ladders. Its saved-dataset report test is skipped
when `data/pairs/test.jsonl` has not been generated. These are software checks, not model results.

Full pretrained inference, 4-bit CUDA memory use, full QLoRA training and benchmark accuracy must
still be verified on the Colab GPU. No accuracy figures or trained adapter are supplied by this change.
