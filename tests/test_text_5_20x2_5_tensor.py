"""Optional real CPU tensor checks; skips are explicit without dependencies."""
from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path

from test_text_5_20x2_5_static import BASE, CONVERT, SCHEDULE, TRAIN, load_file

HAVE_TENSORS = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors", "tokenizers"))


@unittest.skipUnless(HAVE_TENSORS, "Real tensor checks require torch, transformers, safetensors and tokenizers")
class TextTensorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        cls.torch = torch
        torch.set_num_threads(2)
        cls.model_module = load_file("text_52025_tensor_model", BASE / "recursive_model_5_20x2_5.py")
        cls.Model = cls.model_module.RecursiveLlama5_20x2_5ForCausalLM
        cls.Config = LlamaConfig
        cls.SourceModel = LlamaForCausalLM
        engine = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                                     **{f"t{i}": i for i in range(4, 41)}}, unk_token="[UNK]"))
        engine.pre_tokenizer = Whitespace()
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=engine, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")

    def source(self, tied=True):
        self.torch.manual_seed(123)
        config = self.Config(vocab_size=41, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=30, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                             tie_word_embeddings=tied, attention_dropout=0.0,
                             _attn_implementation="eager")
        return self.SourceModel(config)

    def converted(self, tied=True):
        source = self.source(tied)
        model = self.Model(CONVERT.build_target_config(source.config))
        model.load_state_dict(source.state_dict(), strict=True)
        model.tie_weights()
        return source, model

    def test_original_single_pass_parity_full_inheritance_and_recurrent_trace(self):
        source, model = self.converted()
        self.assertGreater(CONVERT.assert_weights_equal(source, model), 200)
        result, _ = CONVERT.audit_forward(source, model)
        self.assertTrue(result["source_single_pass_parity"])
        self.assertEqual(result["recursive_trace"], list(SCHEDULE))
        self.assertEqual(len(model.model.layers), 30)
        self.assertEqual(len(list(model.parameters())), len(list(source.parameters())))

    def test_middle_gradient_is_sum_of_both_executions(self):
        torch = self.torch
        _, shared = self.converted(False)
        unrolled = copy.deepcopy(shared)
        unrolled.model.layers = torch.nn.ModuleList([copy.deepcopy(shared.model.layers[i]) for i in SCHEDULE])
        unrolled.model.logical_to_physical = tuple(range(50))
        ids = torch.tensor([[3, 5, 7, 11]])
        for model in (shared, unrolled):
            logits = model(input_ids=ids, use_cache=False).logits
            loss, _ = TRAIN._causal_loss_sum(logits, ids)
            loss.backward()
        for physical, layer in enumerate(shared.model.layers):
            logical = [i for i, index in enumerate(SCHEDULE) if index == physical]
            for name, parameter in layer.named_parameters():
                expected = sum(dict(unrolled.model.layers[i].named_parameters())[name].grad for i in logical)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                torch.testing.assert_close(parameter.grad, expected, rtol=3e-4, atol=1e-6)
        self.assertEqual(sum(i == 5 for i in SCHEDULE), 2)

    def test_distinct_fifty_cache_slots_and_cached_logits_match_recompute(self):
        torch = self.torch
        _, model = self.converted()
        model.eval()
        ids = torch.tensor([[3, 4, 5, 6, 7]])
        with torch.no_grad():
            prefix = model(input_ids=ids[:, :-1], use_cache=True)
            cache = prefix.past_key_values
            self.assertEqual(len(cache), 50)
            self.assertEqual([cache.get_seq_length(i) for i in range(50)], [4] * 50)
            cached = model(input_ids=ids[:, -1:], attention_mask=torch.ones_like(ids), past_key_values=cache, use_cache=True)
            full = model(input_ids=ids, use_cache=False)
            torch.testing.assert_close(cached.logits[:, -1], full.logits[:, -1], rtol=2e-4, atol=1e-6)
            self.assertEqual([cache.get_seq_length(i) for i in range(50)], [5] * 50)

    def test_save_reload_tied_and_untied_logits(self):
        torch = self.torch
        for tied in (True, False):
            with self.subTest(tied=tied), tempfile.TemporaryDirectory() as directory:
                _, model = self.converted(tied)
                model.eval()
                model.save_pretrained(directory, safe_serialization=True)
                restored, loading = self.Model.from_pretrained(directory, local_files_only=True,
                                                               attn_implementation="eager", output_loading_info=True)
                self.assertFalse(loading["missing_keys"])
                self.assertFalse(loading["unexpected_keys"])
                CONVERT.assert_weights_equal(model, restored)
                CONVERT.assert_tied_weights(restored)
                ids = torch.tensor([[3, 4, 5]])
                with torch.no_grad():
                    torch.testing.assert_close(model(input_ids=ids, use_cache=False).logits,
                                               restored(input_ids=ids, use_cache=False).logits, rtol=0, atol=0)

    def test_full_converter_with_saved_local_assets_and_existing_output_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir, target_dir = root / "SmolLM2", root / "converted"
            source = self.source()
            source.save_pretrained(source_dir, safe_serialization=True)
            self.tokenizer.save_pretrained(source_dir)
            args = CONVERT.parse_args(["--source-checkpoint", str(source_dir), "--output-dir", str(target_dir), "--threads", "2"])
            metadata = CONVERT.convert(args)
            self.assertEqual(metadata["status"], "PASS")
            self.assertEqual(metadata["logical_layer_count"], 50)
            self.assertTrue(metadata["parameter_audit"]["reload_weights_logits_tokenizer_equal"])
            with self.assertRaises(FileExistsError):
                CONVERT.convert(args)

    def test_actual_sixteen_microbatch_gradient_matches_global_token_mean(self):
        torch = self.torch
        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scores = torch.nn.Embedding(41, 41)

            def forward(self, input_ids, **kwargs):
                return types.SimpleNamespace(logits=self.scores(input_ids))
        torch.manual_seed(71)
        accumulated = TinyLM()
        direct = copy.deepcopy(accumulated)
        batches = [TRAIN.collate_dynamic_padding([[3, 4, 5, 6, 7], [8, 9][:1 + i % 2],
                                                 [10, 11, 12], [13, 14, 15, 16][:2 + i % 3]], pad_token_id=0)
                   for i in range(16)]
        config = TRAIN.Stage4Config(world_size=1, micro_batch_size=4, gradient_accumulation_steps=16)
        complete, report = TRAIN._training_window(accumulated, iter(batches), config=config, device=torch.device("cpu"), rank=0)
        total, count = 0, 0
        for batch in batches:
            loss, tokens = TRAIN._causal_loss_sum(direct(input_ids=batch["input_ids"]).logits, batch["labels"])
            total = total + loss
            count = count + tokens
        (total / count).backward()
        self.assertTrue(complete)
        self.assertEqual(report["window_global_valid_token_count"], count.item())
        torch.testing.assert_close(accumulated.scores.weight.grad, direct.scores.weight.grad, rtol=1e-5, atol=1e-7)

    def test_checkpoint_optimizer_scheduler_and_rank_cursor_restore(self):
        torch = self.torch
        _, model = self.converted()
        config = TRAIN._parse_args(["--gate", "FORMAL"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr, betas=config.optimizer_betas,
                                     eps=config.optimizer_eps, weight_decay=config.weight_decay)
        scheduler = TRAIN._load_scheduler(optimizer, 925, 18488)
        ids = torch.tensor([[3, 4, 5]])
        loss, _ = TRAIN._causal_loss_sum(model(input_ids=ids, use_cache=False).logits, ids)
        loss.backward()
        optimizer.step()
        scheduler.step()
        cursors = {str(rank): {"epoch": 1, "shard_index": rank, "shard_microbatches_seen": 16,
                               "samples_seen": 64, "microbatches_seen": 16} for rank in range(8)}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-step-000001"
            TRAIN.save_complete_checkpoint(checkpoint, model=model, tokenizer=self.tokenizer, optimizer=optimizer,
                                           scheduler=scheduler, optimizer_step=1, cumulative_samples=64, cumulative_tokens=10,
                                           data_cursors_by_rank=cursors, manifest={"world_size": 8}, config=config, report={}, rank=0)
            fresh_opt = torch.optim.AdamW(model.parameters(), lr=config.max_lr)
            fresh_sched = TRAIN._load_scheduler(fresh_opt, 925, 18488)
            state = TRAIN._restore_training_state(checkpoint, optimizer=fresh_opt, scheduler=fresh_sched, config=config, rank=7)
            self.assertEqual(state[:3], (1, 64, 10))
            self.assertEqual(state[3]["shard_index"], 7)
            self.assertEqual(fresh_sched.state_dict()["last_epoch"], scheduler.state_dict()["last_epoch"])
            self.assertEqual(fresh_opt.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])
            for left, right in zip(optimizer.state.values(), fresh_opt.state.values()):
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)

    def test_wrong_geometry_and_conflicting_schedule_rejected(self):
        source = self.source()
        for key, value in (("num_hidden_layers", 30), ("recursive_layer_count", 20),
                           ("recursive_middle_layer_count", 10), ("recursive_loops_scope", "full_stack"),
                           ("logical_to_physical_schedule", list(range(30)))):
            config = CONVERT.build_target_config(source.config)
            setattr(config, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.Model(config)


if __name__ == "__main__":
    unittest.main()
