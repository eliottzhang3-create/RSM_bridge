"""Dependency-light contracts for the official MMAR evaluator."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code" / "RSmol" / "scripts"
SOURCE = SCRIPTS / "evaluate_mmar_5_10x2_5_mesh_mellow.py"
INNER_SH = SCRIPTS / "evaluate_mmar_5_10x2_5_mesh_mellow.sh"
SUBMIT_SH = ROOT / "code" / "RSmol" / "run_mmar_5_10x2_5_mesh_mellow_5090.sh"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("mmar_audio_mesh_evaluator", SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


class MMAREvaluatorStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_loads_official_json_and_jsonl_shapes(self) -> None:
        records = [{"id": "0"}, {"id": "1"}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "meta.json"
            jsonl_path = root / "meta.jsonl"
            json_path.write_text(json.dumps(records), encoding="utf-8")
            jsonl_path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
            self.assertEqual(self.module.load_mmar_records(json_path), records)
            self.assertEqual(self.module.load_mmar_records(jsonl_path), records)

    def test_audio_path_is_confined_to_extracted_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = (root / "audio" / "clip.wav").resolve()
            self.assertEqual(self.module.resolve_mmar_audio_path(root, "./audio/clip.wav"), expected)
            with self.assertRaises(ValueError):
                self.module.resolve_mmar_audio_path(root, "../clip.wav")
            with self.assertRaises(ValueError):
                self.module.resolve_mmar_audio_path(root, expected)

    def test_prediction_file_keeps_official_order_and_denominator(self) -> None:
        official = [
            {"id": "0", "answer": "a", "choices": ["a", "b"]},
            {"id": "1", "answer": "b", "choices": ["a", "b"]},
        ]
        state = [
            {"id": "1", "row_index": 1, "status": "generated", "official_prediction": "b) b"},
        ]
        predictions = self.module.materialize_predictions(
            state,
            official,
            full=True,
            output_key="model_prediction",
        )
        self.assertEqual([item["id"] for item in predictions], ["0", "1"])
        self.assertEqual([item["model_prediction"] for item in predictions], ["", "b) b"])
        self.assertTrue(all("answer_prediction" not in item for item in predictions))

    def test_leading_abcd_label_is_removed_for_detected_official_key(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("prepare_model_output_for_official_scorer", source)
        self.assertIn('report["official_evaluation"]["prediction_key"] = prediction_key', source)
        self.assertNotIn("parse_model_output", source)
        self.assertNotIn("selected_option", source)

    def test_choice_label_prefix_score_keeps_full_denominator(self) -> None:
        predictions = [
            {
                "id": "0",
                "answer": "one",
                "choices": ["one", "two"],
                "modality": "audio",
                "category": "test",
                "model_prediction": "a) one",
            },
            {
                "id": "1",
                "answer": " unmatched answer ",
                "choices": ["one", "two"],
                "modality": "audio",
                "category": "test",
                "model_prediction": "a) one",
            },
        ]
        score = self.module.evaluate_choice_label_prefix_predictions(
            predictions, output_key="model_prediction"
        )
        self.assertEqual(score["total"]["total"], 2)
        self.assertEqual(score["total"]["correct"], 1)
        self.assertEqual(score["record_errors"]["total"], 1)
        self.assertEqual(
            score["record_errors"]["policy"],
            "record_retained_and_counted_incorrect",
        )

    def test_reasonaqa_prompt_supports_six_mmar_choices_without_prefix(self) -> None:
        prompt = self.module.common.build_fixed_order_prompt(
            "What is heard?",
            ["one", "two", "three", "four", "five", "six"],
        )
        self.assertEqual(
            prompt,
            "What is heard? a) one b) two c) three d) four e) five f) six",
        )
        self.assertNotIn("Choices:", prompt)
        self.assertEqual(
            self.module.common.build_fixed_order_prompt("Binary?", ["yes", "no", ""]),
            "Binary? a) yes b) no c)",
        )

    def test_official_scorer_is_audited_by_semantics_not_only_bytes(self) -> None:
        for output_key in ("answer_prediction", "model_prediction"):
            source = (
                "import re\n"
                "def string_match(answer, prediction, choices):\n"
                "    answer_tokens = set()\n"
                "    prediction_tokens = set()\n"
                "    incorrect_tokens = set()\n"
                "    cond1 = answer_tokens.issubset(prediction_tokens)\n"
                "    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)\n"
                "    return cond1 and cond2\n"
                f"output_key = {output_key!r}\n"
                "modality_metrics = {}\n"
                "category_metrics = {}\n"
                "print('Total Accuracy:')\n"
            )
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "evaluation.py"
                path.write_text(source, encoding="utf-8")
                audit = self.module._audit_mmar_scorer_semantics(path)
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["output_key"], output_key)

    def test_hf_model_prediction_file_is_consumed_by_official_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scorer = root / "evaluation.py"
            scorer.write_text(
                "import argparse, json\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--input', required=True)\n"
                "args = parser.parse_args()\n"
                "records = json.load(open(args.input, encoding='utf-8'))\n"
                "assert all('model_prediction' in item for item in records)\n"
                "print(f'Total Accuracy: 0.00% over {len(records)} samples')\n",
                encoding="utf-8",
            )
            predictions = [
                {"id": "0", "model_prediction": "a) one"},
                {"id": "1", "model_prediction": "b) two"},
            ]
            (root / "predictions_official.json").write_text(
                json.dumps(predictions),
                encoding="utf-8",
            )
            result = self.module._run_official_evaluation(
                SimpleNamespace(
                    output_dir=root,
                    evaluation_script=scorer,
                    run_official_evaluation=True,
                ),
                predictions,
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["reported_total"], 2)

    def test_inference_only_is_not_reported_as_comparable_pass(self) -> None:
        self.assertEqual(
            self.module.common.overall_evaluation_status("PASS", "NOT_REQUESTED"),
            self.module.common.INFERENCE_ONLY_STATUS,
        )

    def test_smoke_directory_can_resume_as_full(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = dict(
                output_dir=root,
                checkpoint=Path("/checkpoint"),
                metadata_json=Path("/data/MMAR-meta.json"),
                audio_root=Path("/data/mmar-audio"),
                evaluation_script=Path("/data/code/evaluation.py"),
                htsat_checkpoint=Path("/models/htsat.ckpt"),
                mellow_root=Path("/code/mellow"),
                max_prompt_tokens=self.module.MMAR_MAX_PROMPT_TOKENS,
                max_new_tokens=self.module.common.DEFAULT_MAX_NEW_TOKENS,
            )
            self.module._ensure_output_dir(SimpleNamespace(mode="smoke", **common))
            self.module._ensure_output_dir(SimpleNamespace(mode="full", **common))
            config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["mode"], "full")
            self.assertEqual(config["mode_history"], ["smoke", "full"])

    def test_dataset_override_derives_all_official_artifact_paths(self) -> None:
        args = self.module.parse_args([
            "--dataset-dir", "/custom/MMAR",
            "--output-dir", "/tmp/mmar-output",
        ])
        self.assertEqual(args.metadata_json, Path("/custom/MMAR/MMAR-meta.json"))
        self.assertEqual(args.audio_root, Path("/custom/MMAR/mmar-audio"))
        self.assertEqual(args.evaluation_script, Path("/custom/MMAR/code/evaluation.py"))

    def test_source_and_wrappers_lock_official_contract(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        for marker in (
            "detected_from_official_evaluation.py",
            "MMAR_CORE_CANONICAL_SHA256",
            "MMAR_ID_SET_SHA256",
            "MMAR_EVALUATION_SHA256",
            "MMAR_HF_EVALUATION_SHA256",
            "first 10 seconds",
            "compact single-audio prefix",
            "official MMAR order",
            "prompt_format",
        ):
            self.assertIn(marker, source)
        self.assertNotIn("MMAR contains skipped rows", source)
        self.assertIn("skipped_rows_scored_as_incorrect", source)
        self.assertIn("checkpoint-037810", self.module.DEFAULT_CHECKPOINT)
        common_source = Path(self.module.common.__file__).read_text(encoding="utf-8")
        self.assertIn("skipped.jsonl", common_source)
        self.assertIn("progress.jsonl", common_source)
        self.assertIn("conda activate rsmol", INNER_SH.read_text(encoding="utf-8"))
        submit = SUBMIT_SH.read_text(encoding="utf-8")
        self.assertIn("vc submit", submit)
        self.assertIn("-p pdgpu-5090", submit)
        self.assertIn("-g 1", submit)
        self.assertIn("MMAR-meta.json", submit)
        self.assertIn("code/evaluation.py", submit)
        self.assertIn("--max-prompt-tokens 606", submit)
        self.assertIn("--max-new-tokens 32", submit)


if __name__ == "__main__":
    unittest.main()
