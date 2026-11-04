"""Model-integration checks; no GPU or full model weights required.

Run: python -m unittest discover -s tests -v
Set GEMMA4_PROCESSOR_DIR to a local snapshot of the model's public processor files
to also run native-processor and tiny randomly initialised Gemma4 forward/backward tests.
These checks establish integration correctness, not the pretrained model's accuracy or T4 VRAM use.
"""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from prism.common import MODEL_FILES, ROOT, check_model_dir, load_config
from prism.metrics import METRIC_KEYS
from prism.prompts import PROMPTS, SYSTEM, TRAIN_PROMPT
from prism.validation import validate_output
from prism.vlm import assistant_marker, encode_pairs, generate, load_model, load_processor, predict, prepare_training_model


SCORES = {k: round(1 - i * 0.02, 2) for i, k in enumerate(METRIC_KEYS)}
ANSWER = "<analysis>\n- Comparison complete\n</analysis>\n" + json.dumps(SCORES)
PROCESSOR_DIR = os.environ.get("GEMMA4_PROCESSOR_DIR")


def training_module():
    spec = importlib.util.spec_from_file_location("prism_train", Path(ROOT) / "scripts/04_train_qlora.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        for name in MODEL_FILES:
            self.write(name, {})
        self.write("config.json", {"model_type": "gemma4"})
        self.write("tokenizer_config.json", {"chat_template": "example template"})

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value), encoding="utf-8")

    def test_single_weight_file_and_embedded_chat_template(self):
        (self.directory / "model.safetensors").touch()
        check_model_dir(str(self.directory))

    def test_sharded_weights_and_external_chat_template(self):
        self.write("tokenizer_config.json", {})
        (self.directory / "chat_template.jinja").write_text("example template", encoding="utf-8")
        self.write("model.safetensors.index.json", {"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}})
        (self.directory / "part1.safetensors").touch()
        with self.assertRaisesRegex(SystemExit, "part2.safetensors"):
            check_model_dir(str(self.directory))
        (self.directory / "part2.safetensors").touch()
        check_model_dir(str(self.directory))

    def test_missing_weights_and_tokenizer_reported(self):
        (self.directory / "tokenizer_config.json").unlink()
        with self.assertRaisesRegex(SystemExit, "tokenizer_config.json"):
            check_model_dir(str(self.directory))

    def test_qwen_checkpoint_rejected(self):
        self.write("config.json", {"model_type": "qwen2_5_vl"})
        with self.assertRaisesRegex(SystemExit, "not a Gemma4"):
            check_model_dir(str(self.directory))

    def test_no_cuda_fails_before_downloading_weights(self):
        with patch("prism.vlm.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(SystemExit, "CUDA GPU"):
                load_model(load_config())


class ValidationContractTests(unittest.TestCase):
    def test_existing_retry_and_resume_contract(self):
        items = [{"id": f"s{i}_r", "sample_id": f"s{i}", "release_id": "R"} for i in range(2)]
        cfg = load_config()
        with tempfile.TemporaryDirectory() as directory:
            out = os.path.join(directory, "predictions.jsonl")
            with patch("prism.vlm.generate", side_effect=[[ANSWER, "not JSON"], [ANSWER]]) as gen:
                rows = predict(None, None, cfg, items, "P4", out_path=out, batch_size=2)
                self.assertEqual(gen.call_count, 2)
            self.assertEqual([r["retried"] for r in rows], [False, True])
            self.assertTrue(all(r["pred"] == SCORES for r in rows))
            self.assertEqual(set(rows[0]), {"id", "sample_id", "release_id", "prompt_id", "pred", "raw", "retried", "errors"})
            with patch("prism.vlm.generate") as gen:
                resumed = predict(None, None, cfg, items, "P4", out_path=out)
                gen.assert_not_called()
            self.assertEqual(rows, resumed)


@unittest.skipUnless(PROCESSOR_DIR, "set GEMMA4_PROCESSOR_DIR for native processor tests")
class NativeGemma4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = load_config()
        cls.cfg["model"]["base"] = PROCESSOR_DIR
        cls.proc = load_processor(cls.cfg, padding_side="right")
        cls.train = training_module()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.items = []
        for i in range(2):
            paths = []
            for name, colour in (("original", (40 + i * 60, 20, 10)), ("release", (10, 30, 80 + i * 60))):
                path = Path(self.temp.name) / f"{i}_{name}.png"
                Image.new("RGB", (64 + i * 10, 80), colour).save(path)
                paths.append(str(path))
            self.items.append({"id": f"s{i}_r", "sample_id": f"s{i}", "release_id": "R",
                               "original_image_path": paths[0], "release_image_path": paths[1], "target": ANSWER})
        self.proc.tokenizer.padding_side = "right"

    def test_native_resolution_setting_and_assistant_header(self):
        self.assertEqual(self.proc.image_processor.max_soft_tokens, self.cfg["model"]["max_soft_tokens"])
        self.assertTrue(assistant_marker(self.proc))

    def test_pairs_remain_nested_ordered_and_padding_is_restored(self):
        with patch.object(self.proc.image_processor, "preprocess", wraps=self.proc.image_processor.preprocess) as vision:
            enc = encode_pairs(self.proc, self.items, PROMPTS[TRAIN_PROMPT])
            images = vision.call_args.kwargs.get("images", vision.call_args.args[0] if vision.call_args.args else None)
            self.assertEqual([len(pair) for pair in images], [2, 2])
            self.assertEqual(images[0][0].getpixel((0, 0)), (40, 20, 10))
            self.assertEqual(images[0][1].getpixel((0, 0)), (10, 30, 80))
            self.assertEqual(images[1][0].getpixel((0, 0)), (100, 20, 10))
        self.assertEqual(enc["pixel_values"].shape[0], 4)
        self.assertEqual(self.proc.tokenizer.padding_side, "right")
        self.assertEqual(enc["attention_mask"][:, -1].tolist(), [1, 1])
        text = self.proc.tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True)
        self.assertIn(SYSTEM, text)
        self.assertIn(PROMPTS[TRAIN_PROMPT], text)
        self.assertLess(text.index("ORIGINAL:"), text.index("RELEASE:"))

    def test_padding_restored_after_processor_failure(self):
        proc = MagicMock()
        proc.tokenizer.padding_side = "right"
        proc.apply_chat_template.return_value = "text"
        proc.side_effect = RuntimeError("image processing failed")
        with self.assertRaisesRegex(RuntimeError, "processing failed"):
            encode_pairs(proc, self.items, "compare")
        self.assertEqual(proc.tokenizer.padding_side, "right")

    def test_training_masks_prompt_images_and_padding_only(self):
        batch = self.train.Collator(self.proc, self.cfg)([self.items[0], dict(self.items[1], target=ANSWER + "\nExtra")])
        image_id = self.proc.tokenizer.convert_tokens_to_ids(self.proc.image_token)
        self.assertTrue(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))
        self.assertTrue(torch.all(batch["labels"][batch["input_ids"] == image_id] == -100))
        for i in range(2):
            supervised = self.proc.tokenizer.decode(batch["labels"][i][batch["labels"][i] != -100], skip_special_tokens=True)
            self.assertTrue(supervised.lstrip().startswith(ANSWER), repr(supervised[:100]))
            self.assertNotIn(SYSTEM, supervised)
            self.assertEqual(validate_output(supervised)[0], SCORES)

    def test_overlength_training_is_rejected_without_truncation(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["max_seq_len"] = 8
        with self.assertRaisesRegex(RuntimeError, "never truncate image tokens"):
            self.train.Collator(self.proc, cfg)(self.items[:1])

    def test_real_fullpage_renderer_target_fits_sequence_budget(self):
        from prism.ladder import make_ladder_sample
        cfg = copy.deepcopy(self.cfg)
        cfg["paths"]["images_dir"] = self.temp.name
        cfg["ladder"]["knobs"] = ["font_size"]
        sample = make_ladder_sample(cfg, 0)
        self.assertEqual(sample["image_size"], [768, 1024])
        release = next(r for r in sample["releases"] if r["release_id"] == "font_size_1")
        target = "<analysis>\n" + release["reasoning"] + "\n</analysis>\n" + json.dumps(release["scores"])
        self.assertEqual(len(release["reasoning"].splitlines()), 12)
        batch = self.train.Collator(self.proc, cfg)([{
            "original_image_path": sample["original_image_path"],
            "release_image_path": release["release_image_path"], "target": target}])
        length = batch["input_ids"].shape[1]
        print(f"[Gemma4 fullpage] 768x1024, 12-line renderer target, sequence={length}/{cfg['model']['max_seq_len']}")
        self.assertLessEqual(length, 4096)

    def test_generate_decodes_only_new_tokens_and_preserves_output_schema(self):
        model = MagicMock(device=torch.device("cpu"))
        def complete(**kwargs):
            ids = kwargs["input_ids"]
            answer = self.proc.tokenizer.encode(ANSWER, add_special_tokens=False)
            return torch.cat([ids, torch.tensor([answer] * ids.shape[0])], dim=1)
        model.generate.side_effect = complete
        with patch("prism.vlm.compute_dtype", return_value=torch.float32):
            raws = generate(model, self.proc, self.cfg, self.items, PROMPTS["P4"], 768)
        self.assertEqual(len(raws), 2)
        self.assertTrue(all(validate_output(raw)[0] == SCORES for raw in raws))
        self.assertTrue(all("ORIGINAL:" not in raw for raw in raws))
        kwargs = model.generate.call_args.kwargs
        self.assertEqual(kwargs["input_ids"].dtype, torch.long)
        self.assertFalse(kwargs["do_sample"])

    def test_tiny_real_model_lora_backward_adapter_reload_and_generation(self):
        """Run the actual Gemma4 architecture with small random weights; this is not an accuracy test."""
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig, Gemma4VisionConfig, PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers

        proc = copy.deepcopy(self.proc)
        # A compact test vocabulary keeps logits small. Native checkpoint tokenization is
        # tested separately above; the actual Gemma4 vision, PLE and decoder modules run here.
        specials = list(dict.fromkeys(["[UNK]"] + proc.tokenizer.all_special_tokens))
        backend = Tokenizer(models.WordLevel({token: i for i, token in enumerate(specials)}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token=proc.tokenizer.bos_token,
                                      eos_token=proc.tokenizer.eos_token, pad_token=proc.tokenizer.pad_token,
                                      additional_special_tokens=specials[1:])
        for name in ("image_token", "boi_token", "eoi_token", "audio_token", "boa_token", "eoa_token"):
            setattr(tok, name, getattr(proc.tokenizer, name))
        proc.tokenizer = tok
        proc.image_token_id = tok.convert_tokens_to_ids(proc.image_token)
        proc.image_processor.max_soft_tokens = 70
        proc.image_processor.patch_size = 2
        proc.image_processor.pooling_kernel_size = 2
        config = Gemma4Config(
            vision_config=Gemma4VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                              num_attention_heads=4, num_key_value_heads=4, head_dim=8,
                                              patch_size=2, pooling_kernel_size=2, position_embedding_size=256),
            text_config=Gemma4TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=8,
                                         vocab_size=len(tok), vocab_size_per_layer_input=len(tok), hidden_size_per_layer_input=4,
                                         layer_types=["sliding_attention", "full_attention"], max_position_embeddings=4096,
                                         pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id),
            image_token_id=proc.image_token_id, boi_token_id=tok.convert_tokens_to_ids(proc.boi_token),
            eoi_token_id=tok.convert_tokens_to_ids(proc.eoi_token),
        )
        model = Gemma4ForConditionalGeneration(config)
        model.config.use_cache = False
        model = prepare_training_model(model, self.cfg)
        q = self.cfg["qlora"]
        model = get_peft_model(model, LoraConfig(r=q["r"], lora_alpha=q["alpha"], lora_dropout=q["dropout"],
                                                target_modules=q["target_modules"], bias="none", task_type="CAUSAL_LM"))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        batch = self.train.Collator(proc, self.cfg)(self.items[:1])
        loss = model(**batch).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for name, p in model.named_parameters() if "lora_" in name and p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(torch.count_nonzero(g) > 0 for g in grads))
        with tempfile.TemporaryDirectory() as directory:
            from datasets import Dataset
            from trl import SFTConfig, SFTTrainer
            args = self.train.build_args(SFTConfig, output_dir=os.path.join(directory, "train"),
                                         max_steps=1, per_device_train_batch_size=1, use_cpu=True,
                                         report_to="none", optim="adamw_torch", save_strategy="no",
                                         remove_unused_columns=False, dataloader_num_workers=0,
                                         dataset_kwargs={"skip_prepare_dataset": True}, max_length=None,
                                         fp16=False, bf16=False, gradient_checkpointing=False)
            trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(self.items[:1]),
                                 data_collator=self.train.Collator(proc, self.cfg), processing_class=proc.tokenizer)
            trained = trainer.train()
            self.assertTrue(torch.isfinite(torch.tensor(trained.training_loss)))
            model.save_pretrained(directory)
            reloaded = PeftModel.from_pretrained(Gemma4ForConditionalGeneration(config), directory)
            reloaded.eval()
            with patch("prism.vlm.compute_dtype", return_value=torch.float32):
                raw = generate(reloaded, proc, self.cfg, self.items[:1], "Compare these two documents.", 2)
            self.assertEqual(len(raw), 1)
            self.assertIsInstance(raw[0], str)

    def test_kbit_preparation_preserves_frozen_lookup_dtype(self):
        model = torch.nn.Sequential(torch.nn.Embedding(8, 4), torch.nn.LayerNorm(4)).half()
        address = model[0].weight.data_ptr()
        cfg = copy.deepcopy(self.cfg)
        cfg["train"]["gradient_checkpointing"] = False
        prepared = prepare_training_model(model, cfg)
        self.assertEqual(prepared[0].weight.dtype, torch.float16)
        self.assertEqual(prepared[0].weight.data_ptr(), address)
        self.assertEqual(prepared[1].weight.dtype, torch.float32)
        self.assertTrue(all(not p.requires_grad for p in prepared.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
