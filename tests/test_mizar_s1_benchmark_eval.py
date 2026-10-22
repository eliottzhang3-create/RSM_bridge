"""CPU-only contracts for Mizar Stage 1 benchmark evaluation."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code" / "RSmol" / "scripts"
SOURCE = SCRIPTS / "evaluate_mizar_s1_benchmark.py"
RUNTIME_SOURCE = SCRIPTS / "mizar_s1_eval_runtime.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MizarS1BenchmarkEvalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("evaluate_mizar_s1_benchmark", SOURCE)
        cls.runtime = load_module("mizar_s1_eval_runtime", RUNTIME_SOURCE)

    def test_fixed_protocol_is_training_aligned_and_raw(self):
        protocol = self.runtime.PROTOCOL
        self.assertEqual(protocol["prefix_layout"], [129, 129, 1, 256])
        self.assertEqual(protocol["prefix_tokens"], 515)
        self.assertEqual(protocol["answer_start_position"], 515)
        self.assertEqual(protocol["first_answer_logit_position"], 514)
        self.assertEqual(protocol["max_new_tokens"], 300)
        self.assertFalse(protocol["do_sample"])
        self.assertIsNone(protocol["attention_mask"])
        self.assertEqual(protocol["prediction_preprocessing"],
                         "none_including_eos_no_strip_no_label_extraction")

    def test_benchmark_defaults_use_separate_full_evaluation_directories(self):
        mmau = self.module.parse_args(["--benchmark", "mmau"])
        mmar = self.module.parse_args(["--benchmark", "mmar"])
        self.assertEqual(mmau.parquet.name, "test_mini.parquet")
        self.assertEqual(mmau.metadata_json.name, "mmau-test-mini.json")
        self.assertEqual(mmau.audio_root.name, "test-mini-audios")
        self.assertEqual(mmar.metadata_json.name, "MMAR-meta.json")
        self.assertEqual(mmar.audio_root.name, "mmar-audio")
        self.assertIn("mmau_test_mini_step122937", str(mmau.output_dir))
        self.assertIn("mmar_step122937", str(mmar.output_dir))

    def test_scorer_key_is_taken_from_official_script(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.py"
            path.write_text("output_key = 'answer_prediction'\n", encoding="utf-8")
            self.assertEqual(self.module.scorer_prediction_key(path), "answer_prediction")
            path.write_text("output_key = 'model_output'\n", encoding="utf-8")
            self.assertEqual(self.module.scorer_prediction_key(path), "model_output")
            path.write_text("output_key = 'selected_option'\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.module.scorer_prediction_key(path)

    def test_materialization_preserves_raw_text_and_official_order(self):
        records = [
            {"id": "a", "answer": "one", "choices": ["one", "two"]},
            {"id": "b", "answer": "two", "choices": ["one", "two"]},
        ]
        raw = "  b) two <|endoftext|>  "
        state = [{"id": "b", "row_index": 1, "status": "generated", "generated_text": raw}]
        result = self.module._materialize(state, records, "answer_prediction", "mmar")
        self.assertEqual([row["id"] for row in result], ["a", "b"])
        self.assertEqual(result[1]["answer_prediction"], raw)
        self.assertEqual(result[0]["answer_prediction"], "")

    def test_output_identity_refuses_cross_checkpoint_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = type("Args", (), {"output_dir": output})()
            self.module.claim_output(args, {"checkpoint": "one"})
            self.module.claim_output(args, {"checkpoint": "one"})
            with self.assertRaises(RuntimeError):
                self.module.claim_output(args, {"checkpoint": "two"})

    def test_mmau_audio_directory_is_optional_but_mmar_audio_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.ckpt"
            runtime = root / "runtime.json"
            route = root / "route"
            metadata = root / "metadata.json"
            scorer = root / "evaluation.py"
            parquet = root / "test_mini.parquet"
            audio_root = root / "missing-audios"
            checkpoint.touch()
            runtime.touch()
            route.mkdir()
            metadata.touch()
            scorer.touch()
            parquet.touch()
            args = type("Args", (), {
                "benchmark": "mmau", "checkpoint": checkpoint,
                "runtime_config": runtime, "route_root": route,
                "metadata_json": metadata, "evaluation_script": scorer,
                "parquet": parquet, "audio_root": audio_root,
            })()
            self.module._validate_input_paths(args)

            args.benchmark = "mmar"
            with self.assertRaisesRegex(FileNotFoundError, "audio root not found"):
                self.module._validate_input_paths(args)

    def test_submission_wrappers_request_one_gpu_and_official_full_runner(self):
        mmau_submit = (ROOT / "code/RSmol/run_mizar_s1_mmau_eval_5090.sh").read_text(encoding="utf-8")
        mmar_submit = (ROOT / "code/RSmol/run_mizar_s1_mmar_eval_5090.sh").read_text(encoding="utf-8")
        inner = (SCRIPTS / "evaluate_mizar_s1_benchmark.sh").read_text(encoding="utf-8")
        for content, benchmark in ((mmau_submit, "mmau"), (mmar_submit, "mmar")):
            self.assertIn("-p pdgpu-5090", content)
            self.assertIn("-g 1", content)
            self.assertIn(f"--benchmark {benchmark}", content)
            self.assertIn("model--step-122937.ckpt", content)
            self.assertIn("vc submit", content)
        self.assertIn("conda activate mellow_c8204d8", inner)

    def test_inference_path_has_no_legacy_audio_or_prefix_contract(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("prefer_official_audio_file=True", source)
        self.assertIn("center_crop_before_split_or_right_pad_to20s", source)
        self.assertIn("_prepare_mmar", source)
        self.assertIn("_run_official_evaluation", source) if "_run_official_evaluation" in source else self.assertIn("run_official_scorer", source)
        self.assertNotIn("prepare_model_output_for_official_scorer", source)
        self.assertNotIn("max_new_tokens 32", source)
        runtime_source = RUNTIME_SOURCE.read_text(encoding="utf-8")
        self.assertIn("attention_mask=None, labels=None, use_cache=False", runtime_source)
        self.assertIn("skip_special_tokens=False", runtime_source)
        self.assertNotIn("split(tokenizer.eos_token", runtime_source)


if __name__ == "__main__":
    unittest.main()
