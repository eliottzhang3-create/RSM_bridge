"""Dependency-light contracts for isolated SmolLM2 generation and MMAU eval."""
from __future__ import annotations

import importlib.util
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "code" / "RSmol" / "scripts"
GEN = SCRIPT_DIR / "generate_audio_smollm2_checkpoint_reasonaqa.py"
GEN_SH = SCRIPT_DIR / "generate_audio_smollm2_checkpoint_reasonaqa.sh"
GEN_SUBMIT = ROOT / "code" / "RSmol" / "run_audio_smollm2_checkpoint_reasonaqa_generation_3090.sh"
EVAL = SCRIPT_DIR / "evaluate_mmau_test_mini_audio_smollm2.py"
EVAL_SH = SCRIPT_DIR / "evaluate_mmau_test_mini_audio_smollm2.sh"
EVAL_SUBMIT = ROOT / "code" / "RSmol" / "run_mmau_test_mini_audio_smollm2_5090.sh"
MMAR = SCRIPT_DIR / "evaluate_mmar_audio_smollm2.py"
MMAR_SH = SCRIPT_DIR / "evaluate_mmar_audio_smollm2.sh"
MMAR_SUBMIT = ROOT / "code" / "RSmol" / "run_mmar_audio_smollm2_5090.sh"


def load_eval():
    spec = importlib.util.spec_from_file_location("audio_smollm2_mmau_static_eval", EVAL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SmolLM2GenerationMMAUStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluator = load_eval()

    def test_isolated_files_and_defaults(self) -> None:
        for path in (GEN, GEN_SH, GEN_SUBMIT, EVAL, EVAL_SH, EVAL_SUBMIT, MMAR, MMAR_SH, MMAR_SUBMIT):
            self.assertTrue(path.is_file(), path)
        generation = GEN.read_text(encoding="utf-8")
        evaluator = EVAL.read_text(encoding="utf-8")
        self.assertIn("audio_smollm2_135m_mellow/formal_20260911_v1/checkpoint-011343", generation)
        self.assertIn("formal_30epochs_20260923_v1", evaluator)
        self.assertIn("checkpoint-113430", evaluator)
        self.assertIn("audio_smollm2_shared_store_config.json", evaluator)
        self.assertIn("mellow_author_reply_raw_generation_choice_label_scoring_v1", evaluator)
        self.assertIn("evaluate_mmau_test_mini_5_10x2_5_mesh_mellow", evaluator)
        self.assertNotIn("generate_audio_checkpoint_reasonaqa", generation)

    def test_standard_architecture_and_prefix_contract(self) -> None:
        generation = GEN.read_text(encoding="utf-8")
        evaluator = EVAL.read_text(encoding="utf-8")
        for marker in (
            "ORIGINAL_SMOLLM2_CONTRACT",
            "SMOLLM2_HIDDEN_SIZE",
            "AUDIO_TOKENS_PER_CLIP",
            "AUDIO_PREFIX_TOKENS",
            "independent_decoder_layers",
            "has_router_parameters",
            "has_memory_parameters",
            "_audit_saved_checkpoint",
            "_load_model",
            "model.text_model",
            "use_cache=False",
            "logits_to_keep=1",
            "audio1 + separator + audio2 + separator + prompt + generated_tokens",
        ):
            if marker != "_audit_saved_checkpoint":
                self.assertIn(marker, generation + evaluator)
        self.assertIn('"architecture_contract": ORIGINAL_SMOLLM2_CONTRACT', generation)
        self.assertIn("_audit_shared_store_checkpoint", evaluator)
        self.assertIn("SHARED_STORE_AUDIO_PREFIX_TOKENS = 260", evaluator)
        self.assertIn("skip_second_prefix=False", evaluator)
        self.assertIn("htsat_audio1_embedding_reused_for_slot2", evaluator)
        self.assertNotIn("write_routers", generation + evaluator)
        self.assertNotIn("read_routers", generation + evaluator)
        self.assertNotIn("model.mesh_model", generation + evaluator)

    def test_generation_args_lock_protocol(self) -> None:
        args = self.evaluator.parse_args(["--output-dir", "/tmp/mmau", "--mode", "full"])
        self.assertEqual(args.mode, "full")
        self.assertEqual(args.max_new_tokens, 300)
        self.assertEqual(args.max_prompt_tokens, 129)
        self.assertEqual(args.checkpoint, Path(self.evaluator.DEFAULT_CHECKPOINT))
        with self.assertRaises(SystemExit):
            self.evaluator.parse_args(["--output-dir", "/tmp/mmau", "--max-new-tokens", "5"])

    def test_prompt_and_scorer_prediction_match_current_official_contract(self) -> None:
        prompt = self.evaluator.official.build_mellow_author_reply_prompt(
            "Which one?", ["first", "second"]
        )
        self.assertEqual(prompt, "which one? a) first b) second")
        evaluator = EVAL.read_text(encoding="utf-8")
        self.assertNotIn("parse_model_output", evaluator)
        self.assertIn("run_model_generation=_run_mmau_author_reply_generation", evaluator)
        self.assertIn("prepare_model_output_for_official_scorer", evaluator)

    def test_official_evaluation_uses_input_and_prediction_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            evaluator = output / "evaluation.py"
            evaluator.write_text(
                "import argparse, json\n"
                "p=argparse.ArgumentParser(); p.add_argument('--input', required=True)\n"
                "a=p.parse_args(); d=json.load(open(a.input)); assert all('model_output' in x for x in d)\n"
                "print(f'Total Accuracy: 0.00% over {len(d)} samples')\n",
                encoding="utf-8",
            )
            (output / "predictions_fixed_order.json").write_text("[{\"model_output\": \"A\"}]", encoding="utf-8")
            result = self.evaluator._run_official_evaluation(
                types.SimpleNamespace(run_official_evaluation=True, evaluation_script=evaluator), output, 1
            )
            self.assertEqual(result["status"], "PASS")
            text = (output / "official_evaluation.txt").read_text(encoding="utf-8")
            self.assertIn("--input", text)

    def test_submit_wrappers_are_isolated(self) -> None:
        gen_submit = GEN_SUBMIT.read_text(encoding="utf-8")
        eval_submit = EVAL_SUBMIT.read_text(encoding="utf-8")
        mmar_submit = MMAR_SUBMIT.read_text(encoding="utf-8")
        self.assertIn("pdgpu-3090", gen_submit)
        self.assertIn("pdgpu-5090", eval_submit)
        self.assertIn("-g 1", gen_submit)
        self.assertIn("-g 1", eval_submit)
        self.assertIn("test_mini.parquet", eval_submit)
        self.assertIn("mmau-test-mini.json", eval_submit)
        self.assertIn("evaluation.py", eval_submit)
        self.assertIn("pdgpu-5090", mmar_submit)
        self.assertIn("MMAR-meta.json", mmar_submit)
        self.assertIn("mmar-audio", mmar_submit)
        self.assertIn("checkpoint-113430", eval_submit + mmar_submit)
        self.assertIn("formal_30epochs_20260923_v1", eval_submit + mmar_submit)
        self.assertIn("audio_smollm2", gen_submit + eval_submit)
        self.assertNotIn("5_10x2_5_mesh_mellow", gen_submit + eval_submit)
        self.assertIn("--max-new-tokens 300", eval_submit)
        self.assertIn("--dtype fp32", eval_submit)
        self.assertIn("-c 8 -m 32G", eval_submit + mmar_submit)
        self.assertIn("--max-new-tokens 32", mmar_submit)

    def test_mmar_adapter_uses_the_same_baseline_backend_and_official_protocol(self) -> None:
        text = MMAR.read_text(encoding="utf-8")
        self.assertIn("smollm2._load_runtime_model", text)
        self.assertIn("smollm2._run_model_generation", text)
        self.assertIn("mmar_audio_smollm2_shared_store_fixed260_dual_scoring", text)
        self.assertIn("write_choice_label_prefix_evaluation", text)
        self.assertIn("prepare_model_output_for_official_scorer", text)
        self.assertNotIn("parse_model_output", text)


if __name__ == "__main__":
    unittest.main()
