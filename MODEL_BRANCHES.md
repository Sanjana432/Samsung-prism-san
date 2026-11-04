# PRISM: four model versions and how to use them

All four versions live in **one repository**: [Sanjana432/Samsung-prism-san](https://github.com/Sanjana432/Samsung-prism-san).
A branch is a separately named version of its code. You do not need four GitHub accounts or four
separate repositories. Open a model's link below to see and download that version.

| Version | Exact model checkpoint | License | Code | Notebook |
|---|---|---|---|---|
| SmolVLM2 | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | Apache-2.0 | [SmolVLM2 branch](https://github.com/Sanjana432/Samsung-prism-san/tree/smolvlm2-model) | [Open Colab](https://colab.research.google.com/github/Sanjana432/Samsung-prism-san/blob/smolvlm2-model/colab_run_pipeline.ipynb) |
| Gemma 4 E2B | `google/gemma-4-E2B-it` | Apache-2.0 | [Gemma branch](https://github.com/Sanjana432/Samsung-prism-san/tree/gemma4-model) | [Open Colab](https://colab.research.google.com/github/Sanjana432/Samsung-prism-san/blob/gemma4-model/colab_run_pipeline.ipynb) |
| InternVL 3.5 2B | `OpenGVLab/InternVL3_5-2B-HF` | Apache-2.0 | [InternVL branch](https://github.com/Sanjana432/Samsung-prism-san/tree/internvl35-model) | [Open Colab](https://colab.research.google.com/github/Sanjana432/Samsung-prism-san/blob/internvl35-model/colab_run_pipeline.ipynb) |
| Phi-3.5 Vision | `microsoft/Phi-3.5-vision-instruct` | MIT | [Phi branch](https://github.com/Sanjana432/Samsung-prism-san/tree/phi35-model) | [Open Colab](https://colab.research.google.com/github/Sanjana432/Samsung-prism-san/blob/phi35-model/colab_run_pipeline.ipynb) |

The original team's Qwen2.5-VL-3B pipeline remains on `main`. It is an additional baseline,
outside the four versions above. Each model branch contains the complete pipeline, not just a model file.

## Why these four

- **SmolVLM2** is the smaller SmolLM2/SigLIP comparison and a useful first integration to try.
- **Gemma 4 E2B** is the Google Gemma family (probably the name recalled as "Gamma"). It supports
  document/image understanding and an independent language backbone. "E2B" means effective size:
  it has about **5.1B total parameters**, so it is not a 2B-memory equivalent of SmolVLM2.
- **InternVL 3.5 2B** supplies a different vision architecture and native image tiling. Its text
  decoder is Qwen3: this is a distinct VLM, but not an independent Qwen-free language backbone.
- **Phi-3.5 Vision** offers Microsoft's Phi family and explicit multi-image comparison support.

These are practical candidates for this experiment, not a measured ranking of PRISM accuracy. Model
cards and general document benchmarks do not establish sensitivity to a one-pixel or tiny colour change.
That is what the team's common evaluation must measure.

Primary model documentation: [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct),
[Gemma 4](https://huggingface.co/google/gemma-4-E2B-it),
[InternVL](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF),
[Phi Vision](https://huggingface.co/microsoft/Phi-3.5-vision-instruct).

## Access the code without Git commands

1. Click the **Code** link for your model in the table.
2. GitHub displays the selected branch above the file list. Check that its name matches your model.
3. Open `configs/config.yaml`; its `model.base` identifies the exact checkpoint.
4. Open `MODEL_CHANGE.md` for that model's setup, dependencies and test limitations.
5. To download the complete code, choose GitHub's green **Code** button, then **Download ZIP**.
   Keep each extracted model in its own folder.

You can also use GitHub's branch selector to switch between these versions. They are alternatives;
merging the four model branches together is not necessary to run a comparison.

## Run on Google Colab

1. Use the matching **Open Colab** link above and save a copy of the notebook to your Drive.
2. Select **Runtime -> Change runtime type -> GPU**. Run one model per runtime.
3. Run that notebook's setup cells; they install its own dependency versions and use its own
   model-specific Drive folder. Use a fresh runtime when moving between model families.
4. Give each model the team's identical `data/` directory, including images, measured metadata
   and `data/pairs/{train,val,test}.jsonl`. Do not carry another model's checkpoints or predictions across.
   To avoid storing four dataset copies, move that same `data/` folder between model workspaces
   when switching runs; keep each model's own outputs and checkpoints in its original workspace.
5. Run the GPU smoke test before the long stages. Then run prompt study/baseline, fine-tuning,
   inference and evaluation in the existing order. Keep saved output/checkpoints to resume later.
6. The results are in each workspace's `reports/evaluation_report.md`,
   `reports/evaluation_metrics.xlsx` and `predictions/`.

The notebooks support the model-specific setup; **full pretrained GPU runs, peak T4 memory and
PRISM accuracy have not been measured locally**. Quantization reduces linear-weight memory but
does not make every model's embeddings and vision activations small. A successful smoke test on
your assigned GPU is required before committing to a long training run.

## Keep the comparison fair

Use the **same sample IDs and split membership** for every model. Having each student evaluate a
different random set of 300 images would mix model differences with dataset difficulty. If each
student runs 300 originals, use the same agreed 300 originals for the cross-model results.

Keep the generation ladder, scores, prompt-study procedure, optimization and evaluation rules the
same. Record the model, revision, processor resolution/tiling/crops, library versions and hardware.
Processors necessarily differ between model families; disclose those settings rather than claiming
identical visual token counts. Compare minute-change detection by artifact and severity, unchanged
control false alarms, attribution and ranking as well as MAE/RMSE/PLCC/SROCC.

Changing models does not establish high accuracy by itself. Report the measured results, including
failed or invalid outputs, using the existing validation gate and evaluation code.
