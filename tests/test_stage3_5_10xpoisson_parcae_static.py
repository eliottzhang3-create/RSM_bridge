"""Dependency-light contracts for isolated 5-10xpoisson-Parcae Stage 3."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3_5_10xpoisson_parcae.py"
RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3_5_10xpoisson_parcae.sh"
SUBMIT = ROOT / "code" / "RSmol" / "run_stage3_eval_5_10xpoisson_parcae_5090.sh"
MODEL = ROOT / "code" / "RSmol" / "recursive_model_5_10xpoisson_parcae.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("stage3_5_10xpoisson_parcae_static", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage3ParcaeStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_evaluator()
        cls.source = EVALUATOR.read_text(encoding="utf-8")
        cls.runtime = RUNTIME.read_text(encoding="utf-8")
        cls.submit = SUBMIT.read_text(encoding="utf-8")
        cls.model_source = MODEL.read_text(encoding="utf-8")

    def _config(self, vocab_size=3):
        return {
            "model_type": "llama",
            "vocab_size": vocab_size,
            "architectures": ["RecursiveLlama5_10xpoisson_parcaeForCausalLM"],
            "num_hidden_layers": 110,
            "recursive_source_num_hidden_layers": 30,
            "recursive_source_layer_count": 30,
            "recursive_layer_count": 20,
            "recursive_loops": 10,
            "recursive_loops_scope": "middle_only",
            "recursive_min_middle_loops": 4,
            "recursive_max_middle_loops": 10,
            "recursive_default_inference_middle_loops": 7,
            "recursive_parameter_gradient_tail_loops": 4,
            "recursive_prefix_layer_count": 5,
            "recursive_middle_layer_count": 10,
            "recursive_suffix_layer_count": 5,
            "recursive_min_logical_layer_count": 50,
            "recursive_max_logical_layer_count": 110,
            "recursive_mapping_policy": "explicit_5_10xpoisson_parcae_source_layers",
            "recursive_sampling_policy": "truncated_poisson",
            "recursive_prelude_norm": "LlamaRMSNorm",
            "recursive_state_init": "like-init",
            "recursive_learned_h0": False,
            "recursive_training_loop_mode": "per_local_microbatch_per_sequence_truncated_poisson",
            "recursive_local_tmax": True,
            "recursive_noop_left_alignment": True,
            "recursive_injection_no_weight_decay": True,
            "recursive_B_init": "identity",
            "recursive_injection_formula": "h*decay + dt*(PN(e) @ B.T)",
            "recursive_poisson_lambda": 7.0,
            "recursive_poisson_support": list(range(4, 11)),
            "recursive_poisson_normalization_z": self.module.POISSON_NORMALIZATION_Z,
            "recursive_poisson_Z": self.module.POISSON_NORMALIZATION_Z,
            "recursive_poisson_probabilities": list(self.module.POISSON_PROBABILITIES),
            "recursive_source_layer_indices_0based": list(self.module.SOURCE_MAPPING_0BASED),
            "logical_to_physical": list(self.module._expected_schedule(10)),
            "recursive_logical_to_physical": list(self.module._expected_schedule(10)),
            "logical_to_physical_schedule": list(self.module._expected_schedule(10)),
            "recursive_logical_to_physical_schedule": list(self.module._expected_schedule(10)),
            "recursive_sampler_version": self.module.SAMPLER_VERSION,
            "recursive_sampler_key": self.module.SAMPLER_KEY,
            "recursive_backward_policy": self.module.BACKWARD_POLICY,
            "recursive_injection_init": "parcae_exact_ssm_decay_sqrt_1_over_5_identity_B_no_weight_decay",
            "recursive_state_init_std": 0.02,
            "recursive_embedding_scale": 1.0,
            "recursive_ssm_decay": self.module.DEFAULT_SSM_DECAY,
            "recursive_initial_decay": self.module.DEFAULT_SSM_DECAY,
            "recursive_target_product": self.module.DEFAULT_TARGET_PRODUCT,
            "recursive_initial_dt": self.module.DEFAULT_TARGET_PRODUCT,
        }

    def test_exact_parcae_contract_constants(self):
        self.assertEqual(self.module.ARCHITECTURE_CONTRACT, "logical_50_110_physical_20_5_10xpoisson_parcae_tail4")
        self.assertEqual(self.module.PHYSICAL_LAYER_COUNT, 20)
        self.assertEqual((self.module.MIN_LOGICAL_LAYER_COUNT, self.module.MAX_LOGICAL_LAYER_COUNT), (50, 110))
        self.assertEqual(tuple(self.module.POISSON_SUPPORT), tuple(range(4, 11)))
        self.assertAlmostEqual(self.module.POISSON_NORMALIZATION_Z, 0.8197137896443656, places=14)
        self.assertEqual(self.module.DEFAULT_INFERENCE_MIDDLE_LOOPS, 7)
        self.assertEqual(self.module.PARAMETER_GRADIENT_TAIL_LOOPS, 4)

    def test_artifact_contract_accepts_stage4_nested_tokenizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary).resolve()
            (model / "tokenizer").mkdir()
            (model / "config.json").write_text(json.dumps(self._config()), encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"placeholder")
            (model / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer" / "tokenizer.json").write_text('{"model":{"vocab":{"a":0,"b":1,"c":2}}}', encoding="utf-8")
            result = self.module.inspect_model_artifacts_5_10xpoisson_parcae(model)
            self.assertEqual(result["model_label"], self.module.MODEL_LABEL)
            self.assertEqual(result["tokenizer_path"], str(model / "tokenizer"))
            self.assertEqual(result["recursive_audit"]["logical_depth_range"], [50, 110])
            self.assertTrue(result["vocab_compatible"])

    def test_artifact_contract_rejects_wrong_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary).resolve()
            (model / "tokenizer").mkdir()
            config = self._config()
            config["recursive_poisson_support"] = [7]
            (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"placeholder")
            (model / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer" / "tokenizer.json").write_text('{"model":{"vocab":{"a":0,"b":1,"c":2}}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Poisson support"):
                self.module.inspect_model_artifacts_5_10xpoisson_parcae(model)

    def test_protocol_is_isolated_and_offline(self):
        for marker in (
            '"code.RSmol.recursive_model_5_10xpoisson_parcae"',
            "register_auto_class()",
            "AutoModelForCausalLM.from_pretrained",
            "local_files_only=True",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "TaskManager(include_path=str(overlay_dir))",
            "num_fewshot=5 if task in {\"mmlu\", \"gsm8k\"} else None",
            "lm_eval_results.json",
            "log_samples.json",
            "summary.json",
            "summary.csv",
            "audit_report.json",
            "run_config.json",
        ):
            self.assertIn(marker, self.source)
        self.assertNotIn("recursive_model_5_10_5", self.source)
        self.assertNotIn("evaluate_stage3_5_10_5", self.source)

    def test_preflight_contract_markers(self):
        for marker in (
            "default_T7",
            "scalar_inference",
            "T_values",
            "validate_cache_incremental",
            "incremental",
            "validate_generation",
            "validate_save_reload",
            "finite_logits",
            "prelude_norm",
            "like-init",
            "parameter_gradient_tail_loops",
            "PN(e)",
            "B_identity",
            "early_parameter_gradient_edges_absent",
            "torch.inference_mode",
        ):
            self.assertIn(marker, self.source)
        self.assertNotIn("device_map=", self.source)
        self.assertIn("h*decay + dt*(PN(e) @ B.T)", self.source)

    def test_runtime_and_submit_namespace_resources(self):
        for marker in (
            "RSMOL_STAGE3_5_10XPOISSON_PARCAE_MODEL",
            "RSMOL_STAGE3_5_10XPOISSON_PARCAE_BENCHMARK_ROOT",
            "RSMOL_STAGE3_5_10XPOISSON_PARCAE_OUTPUT_DIR",
            "RSMOL_STAGE3_5_10XPOISSON_PARCAE_DEVICE",
            "RSMOL_STAGE3_5_10XPOISSON_PARCAE_DTYPE",
            "RSMOL_STAGE3_5_10XPOISSON_PARCAE_BATCH_SIZE",
            "RSMOL_STAGE3_5_10XPOISSON_PARCAE_TASKS",
            "RSMOL_STAGE3_5_10XPOISSON_PARCAE_CACHE_ROOT",
            "RSMOL_STAGE3_5_10XPOISSON_PARCAE_LOG_ROOT",
            "RSMOL_STAGE3_5_10XPOISSON_PARCAE_VALIDATION_ONLY",
            "RSMOL_STAGE3_5_10XPOISSON_PARCAE_SMOKE",
            "RSMOL_STAGE3_5_10XPOISSON_PARCAE_LIMIT",
            "evaluate_stage3_5_10xpoisson_parcae.py",
        ):
            self.assertIn(marker, self.runtime + self.submit)
        self.assertIn("conda activate", self.runtime)
        self.assertIn("bash scripts/evaluate_stage3_5_10xpoisson_parcae.sh", self.submit)
        self.assertIn("vc submit", self.submit)
        self.assertIn("-p pdgpu-5090", self.submit)
        self.assertIn("-c 8 -m 32G -g 1 -n 1", self.submit)
        self.assertIn("docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1", self.submit)
        self.assertNotIn("evaluate_stage3.py", self.submit)

    def test_benchmark_tasks_and_defaults(self):
        self.assertEqual(tuple(self.module.STAGE3_TASKS), ("hellaswag", "mmlu", "gsm8k", "arc_easy", "arc_challenge"))
        self.assertEqual(self.module.DEFAULT_MODEL.name, "checkpoint-009244")
        self.assertEqual(self.module.DEFAULT_BENCHMARK_ROOT.as_posix(), "/hpc_stor03/sjtu_home/jinwei.zhang/data/eval_datasets")
        config = self.module.parse_args(["--output-dir", "/tmp/parcae-eval", "--validation-only", "--smoke", "--tasks", "mmlu"])
        self.assertEqual(config.tasks, ("mmlu",))
        self.assertEqual(config.limit, 2)
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.dtype, "bfloat16")


if __name__ == "__main__":
    unittest.main()
