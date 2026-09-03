"""Static and dependency-light contracts for 5-10xpoisson-parcae."""

from __future__ import annotations

import importlib.util
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "code" / "RSmol" / "recursive_model_5_10xpoisson_parcae.py"
STAGE4 = ROOT / "code" / "RSmol" / "scripts" / "train_stage4_5_10xpoisson_parcae_ddp.py"
CONVERTER = ROOT / "code" / "RSmol" / "scripts" / "convert_stepwise_5_10xpoisson_parcae.py"
STAGE1 = ROOT / "code" / "RSmol" / "scripts" / "audit_stage1_5_10xpoisson_parcae.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PoissonParcaeStaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_text = MODEL.read_text(encoding="utf-8")
        cls.stage4_text = STAGE4.read_text(encoding="utf-8")
        cls.converter_text = CONVERTER.read_text(encoding="utf-8")
        cls.stage1_text = STAGE1.read_text(encoding="utf-8")
        cls.stage4 = load(STAGE4, "stage4_5_10xpoisson_parcae_static")

    def test_exact_truncated_poisson_contract(self):
        self.assertEqual(self.stage4.POISSON_SUPPORT, tuple(range(4, 11)))
        self.assertAlmostEqual(sum(self.stage4.POISSON_PROBABILITIES), 1.0, places=14)
        self.assertAlmostEqual(self.stage4.POISSON_NORMALIZATION_Z, 0.8197137896443656, places=14)
        self.assertEqual(self.stage4.SAMPLING_POLICY, "truncated_poisson")
        self.assertIn("exp(-7)", self.model_text)

    def test_exact_injection_constants_without_clamp(self):
        import math
        target_product = -math.log(math.sqrt(1.0 / 5.0))
        # inverse-softplus(target_product) -> softplus -> target_product,
        # hence A=1 gives the prescribed initial decay exactly.
        inverse_softplus = math.log(math.expm1(target_product))
        recovered = math.log1p(math.exp(inverse_softplus))
        self.assertAlmostEqual(recovered, target_product, places=14)
        self.assertAlmostEqual(math.exp(-recovered), math.sqrt(1.0 / 5.0), places=14)
        self.assertNotIn("clamp", self.model_text.lower())
        self.assertIn("dt = F.softplus(self.dt_bias)", self.model_text)
        self.assertIn("A = torch.exp(self.A_log)", self.model_text)

    def test_per_microbatch_per_sequence_private_sampler_contract(self):
        for marker in ("torch.Generator(device=\"cpu\")", "optimizer_step", "microbatch_index", "batch_size", "torch.multinomial", "replacement=True", "SAMPLER_VERSION"):
            self.assertIn(marker, self.stage4_text + self.model_text)
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is unavailable in the dependency-light checkout")
        before = random.getstate()
        a = self.stage4.sample_middle_loop_counts(11, 2, 4, 9, 8)
        b = self.stage4.sample_middle_loop_counts(11, 2, 4, 9, 8)
        self.assertEqual(a.tolist(), b.tolist())
        self.assertTrue(all(4 <= int(x) <= 10 for x in a.tolist()))
        self.assertEqual(random.getstate(), before)

    def test_schedule_and_local_alignment(self):
        for T, depth in zip(range(4, 11), range(50, 111, 10)):
            schedule = self.stage4.build_schedule(T)
            self.assertEqual(len(schedule), depth)
            self.assertEqual(schedule[:5], tuple(range(5)))
            self.assertEqual(schedule[-5:], tuple(range(15, 20)))
        for marker in ("left_alignment_tau", "tau_i", "no_op_mask", "local_tmax", "torch.where", "middle_loop_counts"):
            self.assertIn(marker, self.model_text + self.stage1_text)

    def test_parcae_norm_injection_and_selective_bptt(self):
        for marker in ("class PreludeNorm", "pn_e = self.prelude_norm(e)", "PN(e)", "A_log", "dt_bias", "self.B", "softplus", "torch.exp", "u_t = Abar_h + Bbar_PN_e", "Abar(h_t) + Bbar(PN(e))", "MiddleBlockStack", "identity", "functional_call", "detached_parameters"):
            self.assertIn(marker, self.model_text)
        # Initialization may use a local no-grad/inference context.  The
        # selective-BPTT contract only forbids wrapping the recursive early
        # calls (or h_candidate/functional_call) in no_grad, because that
        # would sever the hidden-input path into the prefix.
        model_class_region = self.model_text.split("class RecursiveLlama5_10xpoisson_parcaeModel", 1)[1]
        forward_region = model_class_region.split("    def forward(", 1)[1]
        early_call_region = forward_region.split("        hidden_states = h", 1)[0]
        self.assertNotIn("torch.no_grad()", early_call_region)
        detached_region = self.model_text.split("def _detached_recurrent_call", 1)[1].split("class RecursiveLlama", 1)[0]
        self.assertNotIn("with torch.no_grad()", detached_region)
        self.assertNotIn("with torch.no_grad():\n                h_candidate", forward_region)
        self.assertIn("h_candidate = _detached_recurrent_call", forward_region)
        self.assertIn("h_candidate = self.recurrent", forward_region)
        for marker in ("early_hidden_gradient_norms", "early_parameter_gradient_edges_absent", "exact_parameter_gradient_tail", "last_four_injection_middle_parameter_grads", "prefix_layers_with_grad", "suffix_layers_with_grad"):
            self.assertIn(marker, self.stage1_text)

    def test_independent_namespace_and_additive_pn_contract(self):
        for text in (self.model_text, self.stage4_text, self.converter_text, self.stage1_text):
            legacy_model = "recursive_model_" + "5_10x" + "r_5"
            legacy_stage4 = "train_stage4_" + "5_10x" + "r_5"
            self.assertNotIn(legacy_model, text)
            self.assertNotIn(legacy_stage4, text)
        self.assertIn("PN(e) = PreludeNorm(prefix(x))", self.model_text)
        self.assertIn("u_t   = Abar(h_t) + Bbar(PN(e))", self.model_text)
        self.assertIn("pn_e = self.prelude_norm(e)", self.model_text)
        self.assertIn("prelude_norm_calls", self.model_text)
        self.assertIn("recursive_injection_formula", self.converter_text + self.stage4_text)

    def test_metadata_and_atomic_converter(self):
        for marker in ("SAMPLING_POLICY = \"truncated_poisson\"", "SAMPLER_VERSION", "POISSON_LAMBDA", "POISSON_SUPPORT", "POISSON_NORMALIZATION_Z", "recursive_sampler_version", "recursive_poisson_lambda", "recursive_poisson_support", "recursive_poisson_probabilities", "recursive_poisson_normalization_z", "recursive_state_init_std", "recursive_embedding_scale", "recursive_prelude_norm", "recursive_injection_formula", "recursive_injection_no_weight_decay", "conversion_metadata.json", "checkpoint_complete.json", "tempfile.mkdtemp", "staging.replace(output)", "allow-overwrite", "FORBIDDEN_CHECKOUT"):
            self.assertIn(marker, self.converter_text + self.stage4_text)

    def test_stage4_resume_ddp_and_optimizer_wiring(self):
        for marker in ("--resume-from", "_load_resume_state", "_resume_sampler_sequence_audit", "_resume_step_hint", "rng_states_by_rank", "optimizer_group_audit", "build_optimizer_param_groups", "find_unused_parameters=False", "all_rank_depth_summaries", "rank_local_tmax", "gradient_shape_fingerprint", "global_window_loss_sum", "all_reduce(global_window_loss_sum", "FORMAL data ended before target", "no_global_tmax_broadcast", "torchrun", "_gradient_audit_passes", "all_physical_layers_finite_nonzero", "all_gradients_finite_nonzero", "total_grad_norm must be finite and >0", "total_grad_norm_finite_nonzero", "error_if_nonfinite=False", "_collective_check", "_resume_configuration_mismatches", "checkpoint_seed", "current_seed"):
            self.assertIn(marker, self.stage4_text + (ROOT / "code" / "RSmol" / "scripts" / "smoke_recursive_5_10xpoisson_parcae.sh").read_text(encoding="utf-8"))

    def test_stage4_real_data_dynamic_padding_epoch_and_formal_bounds(self):
        for marker in ("class DistributedParquetStream", "collate_dynamic_padding", "add_special_tokens=False", "truncation=True", "max_length=int(context_length)", "max_length=1024", "valid_mask", "pending_token_ids", "epoch_rollover_policy", "fixed_hashed_rank_shard_order", "raw_row_counts_are_not_training_capacity", "_preaudit_dataset(config.data_dir, tokenizer=tokenizer", "row_index", "stream.restore_cursor", "checkpoint_contains_tokenizer", "tokenizer.save_pretrained(staging / \"tokenizer\")", "tokenizer_payload", "model.safetensors.index.json", "pytorch_model.bin.index.json", "_validate_checkpoint_artifacts", "cursor_policy", "data_dir", "gradient_accumulation_steps", "checkpoint_seed", "current_seed", "_resume_configuration_mismatches"):
            self.assertIn(marker, self.stage4_text)
        self.assertNotIn("add_special_tokens=True", self.stage4_text)
        self.assertNotIn("truncation=False", self.stage4_text)
        self.assertNotIn("formal_remaining_raw_rows", self.stage4_text)
        for marker in ("_validate_formal_runtime_configuration", "config.world_size", "WORLD_SIZE=8", "gradient_accumulation_steps", "micro_batch_size", "context_length", "max_microbatches"):
            self.assertIn(marker, self.stage4_text)
        for marker in ("formal_audits", "report_policy", "audit_point", "_collective_check", "_collective_error_guard", "broadcast_object_list", "_close_process_group(barrier=False)"):
            self.assertIn(marker, self.stage4_text)

    def test_gradient_audit_is_fail_closed_not_serialization_only(self):
        valid = {
            "all_physical_layers_finite_nonzero": True,
            "all_gradients_finite_nonzero": True,
            "all_parameters_finite": True,
            "groups": {"injection": {"A_log": {"valid": True}, "dt_bias": {"valid": True}, "B": {"valid": True}}},
        }
        self.assertTrue(self.stage4._gradient_audit_passes(valid))
        for key in ("all_physical_layers_finite_nonzero", "all_gradients_finite_nonzero", "all_parameters_finite"):
            invalid = dict(valid)
            invalid[key] = False
            self.assertFalse(self.stage4._gradient_audit_passes(invalid))
        invalid_injection = dict(valid)
        invalid_injection["groups"] = {"injection": {"A_log": {"valid": False}}}
        self.assertFalse(self.stage4._gradient_audit_passes(invalid_injection))

    def test_resume_identity_and_formal_reporting_contract(self):
        current = self.stage4.Stage4Config(
            seed=17,
            world_size=8,
            micro_batch_size=8,
            gradient_accumulation_steps=16,
            context_length=1024,
            data_dir=Path("/data/poisson-parcae"),
        )
        saved = {
            "seed": 17,
            "world_size": 8,
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 16,
            "context_length": 1024,
            "data_dir": "/data/poisson-parcae",
        }
        self.assertEqual(self.stage4._resume_configuration_mismatches(saved, current), {})
        for field, value in (("seed", 18), ("world_size", 1), ("micro_batch_size", 4), ("gradient_accumulation_steps", 8), ("context_length", 512), ("data_dir", "/other")):
            changed = dict(saved)
            changed[field] = value
            self.assertIn(field, self.stage4._resume_configuration_mismatches(changed, current))
        self.assertIn("compact scalar metric per optimizer step", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn("run_stage4_5_10xpoisson_parcae_3090.sh", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn("RSMOL_5_10XPOISSON_PARCAE_SOURCE_CHECKPOINT=/hpc_stor03", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn("scripts/*.sh", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn("total_grad_norm_finite_nonzero", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn('"total_grad_norm": float(total_grad_norm.detach().item())', self.stage4_text)
        self.assertNotIn('"total_grad_norm": float(total_grad_norm.detach().item()) if audit_point else None', self.stage4_text)

    def test_cursor_identity_rejects_resume_environment_changes(self):
        stream = self.stage4.DistributedParquetStream.__new__(self.stage4.DistributedParquetStream)
        stream.data_dir = Path("/data/poisson-parcae").resolve()
        stream.rank = 2
        stream.world_size = 8
        stream.seed = 17
        stream.batch_size = 8
        stream.context_length = 1024
        stream.gradient_accumulation_steps = 16
        stream.cursor_policy = self.stage4.DistributedParquetStream.cursor_policy
        stream.rank_shards = tuple()
        stream.pending_token_ids = []
        cursor = {
            "rank": 2,
            "world_size": 8,
            "seed": 17,
            "cursor_policy": stream.cursor_policy,
            "data_dir": str(stream.data_dir),
            "batch_size": 8,
            "context_length": 1024,
            "gradient_accumulation_steps": 16,
            "epoch": 0,
            "shard_index": 0,
            "row_index": 0,
            "pending_token_ids": [],
            "samples_seen": 0,
            "microbatches_seen": 0,
        }
        stream.restore_cursor(cursor)
        for field, value in (("rank", 3), ("world_size", 4), ("seed", 18), ("cursor_policy", "other"), ("data_dir", "/other"), ("batch_size", 4), ("context_length", 512), ("gradient_accumulation_steps", 8)):
            changed = dict(cursor)
            changed[field] = value
            with self.assertRaises(ValueError):
                stream.restore_cursor(changed)

    def test_stage1_executes_scalar_cache_generation_reload_audits(self):
        for marker in ("scalar_inference_all_r", "default_r7", "cache_contract", "generation_contract", "reload_contract", "register_forward_hook", "_expected_scalar_trace", "use_cache=False", "use_cache=True", "get_seq_length", "past_key_values", "fork_rng", "model.generate", "max_new_tokens", "AutoModelForCausalLM.from_pretrained"):
            self.assertIn(marker, self.stage1_text)
        self.assertIn("all_lengths_expected", self.stage1_text)
        self.assertIn("cache_r_mismatch_rejected", self.stage1_text)
        self.assertIn("resolved_depths", self.stage1_text)
        self.assertIn("self.post_init()\n        # The outer causal-LM", self.model_text)
        self.assertIn("self.model.recurrent.injection.reset_parameters()", self.model_text)

    def test_formal_arithmetic(self):
        self.assertEqual(self.stage4.DEFAULT_FORMAL_OPTIMIZER_STEPS, 9244)
        self.assertEqual(self.stage4.DEFAULT_FORMAL_WARMUP_STEPS, 463)
        self.assertEqual(self.stage4.DEFAULT_FORMAL_LOCAL_MICROBATCHES, 9244 * 16)
        self.assertEqual(self.stage4.DEFAULT_FORMAL_SAMPLES_PER_RANK, 9244 * 16 * 8)
        self.assertEqual(self.stage4.DEFAULT_FORMAL_GLOBAL_SAMPLES, 9244 * 16 * 8 * 8)


if __name__ == "__main__":
    unittest.main()
