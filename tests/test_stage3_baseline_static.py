"""Dependency-light contracts for isolated baseline Stage 3 evaluation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3_baseline.py"
FALCON_RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3_baseline_falcon90m.sh"
TINY_RUNTIME = ROOT / "code" / "RSmol" / "scripts" / "evaluate_stage3_baseline_tinybrainbot100m.sh"
FALCON_SUBMIT = ROOT / "code" / "RSmol" / "run_stage3_eval_falcon90m_4090.sh"
TINY_SUBMIT = ROOT / "code" / "RSmol" / "run_stage3_eval_tinybrainbot100m_4090.sh"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("stage3_baseline_static", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage3BaselineStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_evaluator()
        cls.source = EVALUATOR.read_text(encoding="utf-8")
        cls.falcon_runtime = FALCON_RUNTIME.read_text(encoding="utf-8")
        cls.tiny_runtime = TINY_RUNTIME.read_text(encoding="utf-8")
        cls.falcon_submit = FALCON_SUBMIT.read_text(encoding="utf-8")
        cls.tiny_submit = TINY_SUBMIT.read_text(encoding="utf-8")

    def _model_dir(self, root: Path, key: str, *, gguf_only: bool = False) -> Path:
        model = root / key
        model.mkdir()
        if key == "falcon90m":
            config = {
                "architectures": ["FalconH1ForCausalLM"],
                "model_type": "falcon_h1",
                "transformers_version": "4.57.0",
                "hidden_size": 512,
                "intermediate_size": 768,
                "num_hidden_layers": 24,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "vocab_size": 32768,
                "max_position_embeddings": 262144,
                "tie_word_embeddings": True,
            }
        else:
            config = {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 768,
                "intermediate_size": 2048,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "num_key_value_heads": 4,
                "vocab_size": 32000,
                "max_position_embeddings": 1024,
                "tie_word_embeddings": True,
            }
        (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (model / "tokenizer.model").write_bytes(b"placeholder")
        if gguf_only:
            (model / "model-f16.gguf").write_bytes(b"placeholder")
        else:
            (model / "model.safetensors").write_bytes(b"placeholder")
        return model

    def test_contracts_are_explicit_and_registry_isolated(self):
        self.assertEqual(
            self.module.MODEL_CONTRACTS["falcon90m"]["architecture"],
            "FalconH1ForCausalLM",
        )
        self.assertEqual(
            self.module.MODEL_CONTRACTS["tinybrainbot100m"]["dimensions"]["hidden_size"],
            768,
        )
        self.assertIn("hellaswag", self.module.STAGE3_TASKS)
        self.assertIn("prepare_local_task_overlays", self.source)
        self.assertIn("local_files_only=True", self.source)
        self.assertIn("trust_remote_code=False", self.source)
        self.assertNotIn("register_auto_class", self.source)
        self.assertNotIn("recursive_model import", self.source)
        self.assertNotIn("device_map=", self.source)

    def test_falcon_artifact_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = self._model_dir(Path(temporary), "falcon90m")
            report = self.module.inspect_model_artifacts(model, "falcon90m")
            self.assertEqual(report["config_contract"]["expected_architecture"], "FalconH1ForCausalLM")
            self.assertTrue(all(report["config_contract"]["checks"].values()))

    def test_tinybrainbot_hf_weight_wins_and_gguf_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = self._model_dir(Path(temporary), "tinybrainbot100m")
            (model / "tinybrainbot-100m-v3-base-f16.gguf").write_bytes(b"placeholder")
            report = self.module.inspect_model_artifacts(model, "tinybrainbot100m")
            self.assertEqual(report["weight_format"], "HF safetensors/bin")
            self.assertEqual(report["ignored_gguf_files"], ["tinybrainbot-100m-v3-base-f16.gguf"])
            self.assertEqual(report["model_files"], ["model.safetensors"])

    def test_gguf_only_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = self._model_dir(Path(temporary), "tinybrainbot100m", gguf_only=True)
            with self.assertRaisesRegex(FileNotFoundError, "no HF safetensors/bin"):
                self.module.inspect_model_artifacts(model, "tinybrainbot100m")

    def test_wrong_architecture_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = self._model_dir(Path(temporary), "falcon90m")
            config = json.loads((model / "config.json").read_text(encoding="utf-8"))
            config["architectures"] = ["LlamaForCausalLM"]
            (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                self.module.inspect_model_artifacts(model, "falcon90m")

    def test_defaults_and_validation_flags(self):
        config = self.module.parse_args(
            ["--model-key", "tinybrainbot100m", "--output-dir", "/tmp/baseline", "--smoke", "--tasks", "mmlu"]
        )
        self.assertEqual(config.model_path, self.module.DEFAULT_MODEL_PATHS["tinybrainbot100m"])
        self.assertEqual(config.tasks, ("mmlu",))
        self.assertEqual(config.limit, 2)
        self.assertEqual(config.device, "cuda:0")

    def test_remote_wrappers_use_rsmol_and_one_4090_gpu(self):
        for runtime in (self.falcon_runtime, self.tiny_runtime):
            self.assertIn("source \"$USER_CONDA_BASE/etc/profile.d/conda.sh\"", runtime)
            self.assertIn('conda activate "$RSMOL_BASELINE_CONDA_ENV"', runtime)
            self.assertIn("HF_HUB_OFFLINE=1", runtime)
            self.assertIn("evaluate_stage3_baseline.py", runtime)
            self.assertIn("--tasks", runtime)
            self.assertIn("cuda:0", runtime)
        self.assertIn(
            'RSMOL_BASELINE_CONDA_ENV="${RSMOL_BASELINE_CONDA_ENV:-swift_start}"',
            self.falcon_runtime,
        )
        self.assertIn(
            'RSMOL_BASELINE_CONDA_ENV="${RSMOL_BASELINE_CONDA_ENV:-rsmol}"',
            self.tiny_runtime,
        )
        for submit in (self.falcon_submit, self.tiny_submit):
            self.assertIn("vc submit", submit)
            self.assertIn("-p pdgpu-4090", submit)
            self.assertIn("-c 8 -m 32G -g 1 -n 1", submit)
            self.assertIn("bash scripts/evaluate_stage3_baseline_", submit)
            self.assertIn("RSMOL_BASELINE_CONDA_ENV", submit)
            self.assertNotIn("python", submit)


if __name__ == "__main__":
    unittest.main()
