"""GPU smoke test (~5 min). Run BEFORE the long stages to catch environment/API problems early.

1. loads the 4-bit base model + processor
2. generates on 2 validation pairs with P4 and runs the validation gate
3. builds a training batch with the collator: checks label masking and sequence length
4. one forward + backward pass through the LoRA-wrapped model
5. prints peak VRAM and a recommendation for batch size / num_crops
"""

import argparse
import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prism.common import P, load_config, read_jsonl  # noqa: E402
from prism.vlm import compute_dtype, load_base_model, load_model, load_processor, predict  # noqa: E402

from prism.training import (assert_finite_training, assert_lora_trainables, atomic_json,
                            context_preflight, disable_training_cache, seed_training)  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    from prism.dataset import assert_dataset_ready
    assert_dataset_ready(cfg)
    seed_training(cfg["project_seed"])
    assert torch.cuda.is_available(), "No CUDA GPU visible to torch"
    g = torch.cuda.get_device_properties(0)
    print(f"[smoke] GPU {g.name}, {g.total_memory / 2**30:.1f} GB, bf16={torch.cuda.is_bf16_supported()}")
    val = read_jsonl(P(cfg, "pairs_dir", "val.jsonl"))[:2]
    train = read_jsonl(P(cfg, "pairs_dir", "train.jsonl"))[:2]
    if not train or not val:
        raise ValueError("GPU smoke test needs nonempty train and validation pairs")

    spec = importlib.util.spec_from_file_location("tr", os.path.join(ROOT, "scripts", "04_train_qlora.py"))
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    preflight_proc = load_processor(cfg, padding_side="right")
    context_preflight(train[:1], val, tr.Collator(preflight_proc, cfg), cfg)

    torch.cuda.reset_peak_memory_stats()
    model, proc = load_model(cfg), load_processor(cfg)
    rows = predict(model, proc, cfg, val, "P4", batch_size=cfg["inference"]["batch_size"], max_new_tokens=400)
    for r in rows:
        print(f"[smoke] gate valid={r['pred'] is not None} retried={r['retried']}  raw[:200]={r['raw'][:200]!r}")
    inference_peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[smoke] inference peak VRAM {inference_peak:.2f} GB")
    del model
    torch.cuda.empty_cache()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    proc = load_processor(cfg, padding_side="right")
    batch = tr.Collator(proc, cfg)(train[:1])
    n_lab = int((batch["labels"] != -100).sum())
    print(f"[smoke] seq len {batch['input_ids'].shape[1]} (max {cfg['model']['max_seq_len']}), supervised tokens {n_lab}")
    decoded = proc.tokenizer.decode(batch["input_ids"][0][batch["labels"][0] != -100])
    assert decoded.lstrip().startswith("<analysis>"), f"label mask wrong, supervised text starts with: {decoded[:80]!r}"
    print("[smoke] label mask OK - only the assistant answer is supervised")

    torch.cuda.reset_peak_memory_stats()
    q = cfg["qlora"]
    m = load_base_model(cfg)
    disable_training_cache(m)
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    m = get_peft_model(m, LoraConfig(r=q["r"], lora_alpha=q["alpha"], lora_dropout=q["dropout"],
                                     target_modules=q["target_modules"], task_type="CAUSAL_LM"))
    n_trainable = assert_lora_trainables(m)
    m.train()
    batch = {k: v.to(m.device) for k, v in batch.items()}
    with torch.autocast("cuda", dtype=compute_dtype()):
        loss = m(**batch).loss
    if not bool(torch.isfinite(loss.detach()).all()):
        raise RuntimeError("GPU smoke test produced non-finite loss")
    loss.backward()
    assert_finite_training(m, loss)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[smoke] train step OK, loss={loss.item():.3f}, peak VRAM {peak:.2f} GB")
    total = g.total_memory / 2**30
    if peak > 0.85 * total:
        print("[smoke] RECOMMEND: lower inference.batch_size to 1; if necessary lower model.num_crops to 1 and record it")
    elif total >= 40:
        print("[smoke] RECOMMEND: you can raise inference.batch_size to 8")
    atomic_json(os.path.join(P(cfg, "reports_dir"), "gpu_smoke_test.json"), {
        "model": cfg["model"], "gpu": g.name, "gpu_gib": total, "compute_dtype": str(compute_dtype()),
        "inference_peak_gib": inference_peak, "training_peak_gib": peak, "loss": float(loss.detach()),
        "lora_trainable_parameters": n_trainable, "supervised_tokens": n_lab,
        "n_predictions": len(rows), "n_predictions_valid": sum(row["pred"] is not None for row in rows),
        "status": "compatibility_checks_passed", "note": "Smoke test, not a benchmark accuracy result",
    })
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
