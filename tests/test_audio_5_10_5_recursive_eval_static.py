"""Dependency-light checks for the isolated fixed-recursive MMAU/MMAR route."""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code/RSmol/scripts"
ROUTE = ROOT / "code/RSmol"
MMAU = SCRIPTS / "evaluate_mmau_test_mini_audio_5_10_5_recursive_mellow.py"
MMAR = SCRIPTS / "evaluate_mmar_audio_5_10_5_recursive_mellow.py"
MMAU_SHELL = SCRIPTS / "evaluate_mmau_test_mini_audio_5_10_5_recursive_mellow.sh"
MMAR_SHELL = SCRIPTS / "evaluate_mmar_audio_5_10_5_recursive_mellow.sh"
MMAU_SUBMIT = ROUTE / "run_mmau_test_mini_audio_5_10_5_recursive_mellow_5090.sh"
MMAR_SUBMIT = ROUTE / "run_mmar_audio_5_10_5_recursive_mellow_5090.sh"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedRecursiveEvaluationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(SCRIPTS))
        try:
            cls.mmau = load_module(MMAU, "fixed_recursive_mmau_test")
            cls.mmar = load_module(MMAR, "fixed_recursive_mmar_test")
        finally:
            sys.path.remove(str(SCRIPTS))

    def test_full_defaults_and_route_specific_checkpoint(self) -> None:
        mmau = self.mmau.parse_args(["--output-dir", "/tmp/recursive-mmau"])
        mmar = self.mmar.parse_args(["--output-dir", "/tmp/recursive-mmar"])
        for args in (mmau, mmar):
            self.assertEqual(args.mode, "full")
            self.assertEqual(args.checkpoint, Path(self.mmau.DEFAULT_CHECKPOINT))
            self.assertEqual(args.max_new_tokens, 32)
            self.assertTrue(str(args.checkpoint).endswith("checkpoint-037810"))
            self.assertIn("audio_5_10_5_recursive_mellow/partition_formal_eos_v2_10epochs_20260919", str(args.checkpoint).replace("\\", "/"))
        self.assertEqual(mmau.max_prompt_tokens, 129)
        self.assertEqual(mmar.max_prompt_tokens, 606)
        with self.assertRaises(SystemExit):
            self.mmau.parse_args(["--output-dir", "/tmp/mmau", "--max-new-tokens", "5"])
        with self.assertRaises(SystemExit):
            self.mmar.parse_args(["--output-dir", "/tmp/mmar", "--max-new-tokens", "5"])

    def test_exact_recursive_checkpoint_and_generation_are_isolated(self) -> None:
        text = MMAU.read_text(encoding="utf-8")
        tree = ast.parse(text)
        self.assertIn(self.mmau.PARTITION_CONTRACT, text)
        self.assertIn("audio_recursive_5_10_5_partition_config.json", text)
        self.assertIn("validate_recursive_5_10_5", text)
        self.assertIn("_audio_state_hashes", text)
        self.assertIn("_load_training_state_metadata", text)
        self.assertIn("mmap=True", text)
        self.assertIn("checkpoint_complete.json", text)
        self.assertIn("text_model_runtime_contract", text)
        self.assertIn("forbidden_custom_parameter_names", text)
        self.assertIn("use_cache=False", text)
        self.assertIn("logits_to_keep=1", text)
        self.assertIn("trace != expected_trace", text)
        self.assertIn("for generation_step in range(token_budget)", text)
        self.assertIn("finally:\n        for handle in handles:\n            handle.remove()", text)
        self.assertIn("skip_second_prefix=True", text)
        self.assertIn('"prefix_token_count": AUDIO_SINGLE_PREFIX_TOKENS', text)
        self.assertIn('"audio2_prefix_materialized": False', text)
        self.assertIn('"prefix_layout": "audio1 + separator1"', text)
        self.assertNotIn("model.mesh_model", text)
        self.assertNotIn("validate_original_smollm2", text)
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({"_audit_partition_checkpoint", "_load_runtime_model", "_build_compact_audio_prefix", "_greedy_decode_recursive", "_run_model_generation"} <= functions)

    def test_official_pipeline_is_shared_but_backend_is_not(self) -> None:
        mmau = MMAU.read_text(encoding="utf-8")
        mmar = MMAR.read_text(encoding="utf-8")
        self.assertIn("return official.parse_args", mmau)
        self.assertIn("report = official.run(", mmau)
        self.assertIn("load_runtime_model=_load_runtime_model", mmau)
        self.assertIn("run_model_generation=_run_model_generation", mmau)
        self.assertIn("evaluate_mmar_5_10x2_5_mesh_mellow as official", mmar)
        self.assertIn("load_runtime_model=recursive._load_runtime_model", mmar)
        self.assertIn("run_model_generation=recursive._run_model_generation", mmar)
        self.assertNotIn("evaluate_mmau_test_mini_audio_smollm2 as smollm2", mmar)
        self.assertEqual(
            self.mmau.build_fixed_order_prompt("Which sound?", ["one", "two"]),
            "Which sound? a) one b) two",
        )
        self.assertEqual(
            self.mmau.prepare_model_output_for_official_scorer("c) plausible"),
            "plausible",
        )

    def test_backend_exception_invalidates_score(self) -> None:
        for module in (self.mmau, self.mmar):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                args = SimpleNamespace(output_dir=output)
                report = {"status": "PASS", "records": {"skip_reasons": {"sample_exception": 1}}}
                with patch.object(module.official, "run", return_value=report):
                    result = module.run(args)
                self.assertEqual(result["status"], "FAILED")
                self.assertIn("recursive generation failures", result["fatal_error"]["error"])
                self.assertEqual(json.loads((output / "evaluation_report.json").read_text(encoding="utf-8"))["status"], "FAILED")

    def test_full_submit_wrappers_are_separate_and_single_gpu(self) -> None:
        for path in (MMAU, MMAR, MMAU_SHELL, MMAR_SHELL, MMAU_SUBMIT, MMAR_SUBMIT):
            self.assertTrue(path.is_file(), path)
        for shell, submit, benchmark in (
            (MMAU_SHELL, MMAU_SUBMIT, "MMAU"),
            (MMAR_SHELL, MMAR_SUBMIT, "MMAR"),
        ):
            inner = shell.read_text(encoding="utf-8")
            outer = submit.read_text(encoding="utf-8")
            self.assertIn("conda activate rsmol", inner)
            self.assertIn("-p pdgpu-5090", outer)
            self.assertIn("-c 32 -m 256G -g 1 -n 1", outer)
            self.assertIn("--mode full", outer)
            self.assertIn("--max-new-tokens 32", outer)
            self.assertIn("--run-official-evaluation", outer)
            self.assertIn("audio_5_10_5_recursive_mellow/", outer)
            self.assertIn("checkpoint-037810", outer)
            self.assertIn("recursive", outer)
            self.assertNotIn("audio_smollm2_135m_mellow/", outer)
            self.assertNotIn("audio_5_10x2_5_mesh_mellow/", outer)
            if benchmark == "MMAU":
                self.assertIn("test_mini.parquet", outer)
                self.assertIn("mmau-test-mini.json", outer)
                self.assertIn("--max-prompt-tokens 129", outer)
            else:
                self.assertIn("MMAR-meta.json", outer)
                self.assertIn("mmar-audio", outer)
                self.assertIn("--max-prompt-tokens 606", outer)


if __name__ == "__main__":
    unittest.main()
