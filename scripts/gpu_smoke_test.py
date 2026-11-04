"""GPU smoke test (~5 min). Run BEFORE the long stages to catch environment/API problems early.

1. loads the 4-bit base model + processor
2. generates on 2 validation pairs with P4 and runs the validation gate
3. builds a training batch with the collator: checks label masking and sequence length
4. one forward + backward pass through the LoRA-wrapped model
5. prints peak VRAM and a recommendation for batch size / max_soft_tokens
"""

import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prism.common import P, load_config, read_jsonl  # noqa: E402
from prism.vlm import VLMClass, bnb_config, compute_dtype, load_model, load_processor, predict, prepare_training_model  # noqa: E402


def main():
    cfg = load_config()
    assert torch.cuda.is_available(), "No CUDA GPU visible to torch"
    g = torch.cuda.get_device_properties(0)
    print(f"[smoke] GPU {g.name}, {g.total_memory / 2**30:.1f} GB, bf16={torch.cuda.is_bf16_supported()}")
    val = read_jsonl(P(cfg, "pairs_dir", "val.jsonl"))[:2]
    train = read_jsonl(P(cfg, "pairs_dir", "train.jsonl"))[:2]

    torch.cuda.reset_peak_memory_stats()
    model, proc = load_model(cfg), load_processor(cfg)
    rows = predict(model, proc, cfg, val, "P4", batch_size=cfg["inference"]["batch_size"], max_new_tokens=400)
    for r in rows:
        print(f"[smoke] gate valid={r['pred'] is not None} retried={r['retried']}  raw[:200]={r['raw'][:200]!r}")
    print(f"[smoke] inference peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    del model
    torch.cuda.empty_cache()

    spec = importlib.util.spec_from_file_location("tr", os.path.join(ROOT, "scripts", "04_train_qlora.py"))
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    from peft import LoraConfig, get_peft_model
    proc = load_processor(cfg, padding_side="right")
    batch = tr.Collator(proc, cfg)(train[:1])
    n_lab = int((batch["labels"] != -100).sum())
    print(f"[smoke] seq len {batch['input_ids'].shape[1]} (max {cfg['model']['max_seq_len']}), supervised tokens {n_lab}")
    decoded = proc.tokenizer.decode(batch["input_ids"][0][batch["labels"][0] != -100])
    assert decoded.lstrip().startswith("<analysis>"), f"label mask wrong, supervised text starts with: {decoded[:80]!r}"
    print("[smoke] label mask OK - only the assistant answer is supervised")

    torch.cuda.reset_peak_memory_stats()
    q = cfg["qlora"]
    m = VLMClass.from_pretrained(cfg["model"]["base"], quantization_config=bnb_config(cfg), dtype=compute_dtype(),
                                 device_map="auto", attn_implementation="sdpa")
    m.config.use_cache = False
    m = prepare_training_model(m, cfg)
    m = get_peft_model(m, LoraConfig(r=q["r"], lora_alpha=q["alpha"], lora_dropout=q["dropout"],
                                     target_modules=q["target_modules"], task_type="CAUSAL_LM"))
    m.train()
    batch = {k: v.to(m.device) for k, v in batch.items()}
    with torch.autocast("cuda", dtype=compute_dtype()):
        loss = m(**batch).loss
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[smoke] train step OK, loss={loss.item():.3f}, peak VRAM {peak:.2f} GB")
    total = g.total_memory / 2**30
    if peak > 0.85 * total:
        print("[smoke] RECOMMEND: inference.batch_size=1; if needed lower model.max_soft_tokens to 140 and record the changed resolution")
    elif total >= 40:
        print("[smoke] RECOMMEND: you can raise inference.batch_size to 8")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
