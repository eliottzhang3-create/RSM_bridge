"""Dependency-light contracts for the MMAU test-mini evaluator."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "code" / "RSmol" / "scripts" / "evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.py"
INNER_SH = ROOT / "code" / "RSmol" / "scripts" / "evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.sh"
SUBMIT_SH = ROOT / "code" / "RSmol" / "run_mmau_test_mini_5_10x2_5_mesh_mellow_5090.sh"


def load_module():
    spec = importlib.util.spec_from_file_location("mmau_test_mini_evaluator", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MMAUEvaluatorStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_parser_accepts_only_conservative_choice_forms(self) -> None:
        choices = ["red", "red and blue", "Green", "A third option"]
        self.assertEqual(self.module.parse_model_output("A", choices)["selected_option"], "red")
        self.assertEqual(self.module.parse_model_output("(b)", choices)["selected_option"], "red and blue")
        self.assertEqual(self.module.parse_model_output("C.", choices)["selected_option"], "Green")
        self.assertEqual(self.module.parse_model_output("[D]", choices)["selected_option"], "A third option")
        self.assertEqual(self.module.parse_model_output("(B)red and blue", choices)["selected_option"], "red and blue")
        self.assertEqual(self.module.parse_model_output(" RED ", choices)["selected_option"], "red")
        self.assertEqual(self.module.parse_model_output("red and blue", choices)["selected_option"], "red and blue")
        self.assertEqual(self.module.parse_model_output("blue", choices)["selected_option"], "")
        self.assertEqual(self.module.parse_model_output("The answer is A", choices)["selected_option"], "")
        self.assertEqual(self.module.parse_model_output("X", ["one", "two", "three", "X"])["selected_option"], "X")

    def test_parser_rejects_ambiguous_duplicate_text(self) -> None:
        result = self.module.parse_model_output("same", ["same", "same"])
        self.assertEqual(result["selected_option"], "")
        self.assertEqual(result["parse_status"], "unparseable")
        self.assertEqual(
            self.module.parse_model_output("same long", ["same", "same long"])["selected_option"],
            "same long",
        )

    def test_parser_extracts_answer_from_start_of_repetitive_generation(self) -> None:
        first = self.module.parse_model_output(
            "B) A goat Cd) A birdB) A goat Cd) A bird",
            ["A human", "A goat", "A car", "A bird"],
        )
        self.assertEqual(first["selected_option"], "A goat")
        self.assertEqual(first["parse_method"], "leading_label")

        second = self.module.parse_model_output(
            "D) Unusual soundD) Unusual sound) Unusual sound is a sound that lacks context",
            ["Loudness", "Frequency range", "Duration", "Unusual sound"],
        )
        self.assertEqual(second["selected_option"], "Unusual sound")
        self.assertEqual(second["parse_method"], "leading_label")

        text_first = self.module.parse_model_output(
            "Unusual soundD) Unusual sound continues",
            ["Loudness", "Frequency range", "Duration", "Unusual sound"],
        )
        self.assertEqual(text_first["selected_option"], "Unusual sound")
        self.assertEqual(text_first["parse_method"], "leading_full_text")

    def test_parser_does_not_search_for_an_answer_inside_explanation(self) -> None:
        choices = ["Man", "Woman", "Child", "Robot"]
        self.assertEqual(
            self.module.parse_model_output("The answer is (A) Man", choices)["selected_option"],
            "",
        )
        self.assertEqual(self.module.parse_model_output("Mango", choices)["selected_option"], "")

    def test_fixed_order_prompt_and_choice_alignment(self) -> None:
        prompt = self.module.build_fixed_order_prompt("Which one?", ["first", "second"])
        self.assertIn("(A) first", prompt)
        self.assertIn("(B) second", prompt)
        self.assertEqual(
            prompt,
            "Answer the following multiple-choice question based on the audio. Which one? "
            "Choices: (A) first (B) second",
        )
        self.assertTrue(self.module.choices_match_fixed_order(["(A) first", "B. second"], ["first", "second"]))
        self.assertFalse(self.module.choices_match_fixed_order(["second", "first"], ["first", "second"]))
        self.assertFalse(self.module.choices_match_fixed_order(["(B) first", "A. second"], ["first", "second"]))

    def test_metadata_row_uses_other_attributes_id(self) -> None:
        sample_id = "row-1"
        row = {
            "instruction": "Which one?",
            "choices": ["(A) first", "(B) second"],
            "other_attributes": {"id": sample_id},
        }
        sample_id_from_row, source = self.module.extract_row_id(row)
        self.assertEqual(sample_id_from_row, sample_id)
        self.assertEqual(source, "other_attributes.id")
        self.assertEqual(self.module.extract_row_question(row), "Which one?")
        self.assertEqual(self.module.extract_row_choices(row), ["(A) first", "(B) second"])
        self.assertEqual(self.module.extract_row_answer({"answer": "first"}), "first")
        self.assertEqual(
            self.module._normalize_text(self.module._strip_choice_label("(A) first")),
            "first",
        )

    def test_metadata_id_and_fields_accept_json_encoded_structs(self) -> None:
        row = {
            "instruction": "Which one?",
            "choices": '["(A) first", "(B) second"]',
            "answer": "(A) first",
            "other_attributes": '{"id": "json-row", "task": "sound", "difficulty": "easy"}',
        }
        self.assertEqual(self.module.extract_row_id(row), ("json-row", "other_attributes.id"))
        self.assertEqual(self.module.extract_row_choices(row), ["(A) first", "(B) second"])
        self.assertEqual(self.module._coerce_choices("['(A) first', '(B) second']"), ["(A) first", "(B) second"])
        self.assertEqual(self.module._as_mapping(row["other_attributes"])["id"], "json-row")
        self.assertEqual(
            self.module.extract_record_field(self.module._as_mapping(row["other_attributes"]), "task"),
            "sound",
        )

    def test_audio_payload_keeps_parent_sampling_rate_for_raw_array(self) -> None:
        payload, source = self.module.extract_audio_payload(
            {
                "context": {"audio": [0.0, 0.25, -0.25], "sampling_rate": 22050},
            }
        )
        self.assertEqual(source, "context.audio")
        self.assertEqual(payload["sampling_rate"], 22050)
        self.assertEqual(payload["array"], [0.0, 0.25, -0.25])
        payload, _ = self.module.extract_audio_payload(
            {"context": {"audio": [0.0, 0.5]}, "sampling_rate": 16000}
        )
        self.assertEqual(payload["sampling_rate"], 16000)
        self.assertEqual(self.module.extract_audio_payload({"audio_id": "clip-1"}), (None, "missing"))

    def test_label_stripping_allows_no_space_after_explicit_label(self) -> None:
        self.assertEqual(self.module._strip_choice_label("(A)first"), "first")
        self.assertTrue(
            self.module.choices_match_fixed_order(["(A)first", "[B]second"], ["first", "second"])
        )

    def test_parquet_iterator_stops_at_physical_row_five(self) -> None:
        class FakeBatch:
            def __init__(self, rows):
                self.rows = rows

            def to_pylist(self):
                return self.rows

        class FakeParquetFile:
            def __init__(self, path):
                self.path = path

            def iter_batches(self, **kwargs):
                self.kwargs = kwargs
                yield FakeBatch([{"id": index} for index in range(3)])
                yield FakeBatch([{"id": index} for index in range(3, 9)])

        fake_pyarrow = types.ModuleType("pyarrow")
        fake_parquet = types.ModuleType("pyarrow.parquet")
        fake_parquet.ParquetFile = FakeParquetFile
        fake_pyarrow.parquet = fake_parquet
        with mock.patch.dict(
            sys.modules,
            {"pyarrow": fake_pyarrow, "pyarrow.parquet": fake_parquet},
        ):
            rows = list(self.module.iter_parquet_rows(Path("unused.parquet"), batch_size=2, limit=5))
        self.assertEqual([index for index, _ in rows], [0, 1, 2, 3, 4])
        self.assertEqual([row["id"] for _, row in rows], [0, 1, 2, 3, 4])

    def test_official_evaluation_uses_model_output_and_input_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            evaluator = output / "evaluation.py"
            evaluator.write_text(
                "import argparse, json\n"
                "p=argparse.ArgumentParser(); p.add_argument('--input', required=True)\n"
                "a=p.parse_args(); d=json.load(open(a.input)); assert all('model_output' in x for x in d)\n"
                "print('Total Accuracy: 0.00%% over %d samples' % len(d))\n",
                encoding="utf-8",
            )
            predictions = output / "predictions_fixed_order.json"
            predictions.write_text(json.dumps([{"answer": "x", "model_output": ""}]), encoding="utf-8")
            args = types.SimpleNamespace(
                run_official_evaluation=True,
                evaluation_script=evaluator,
            )
            result = self.module._run_official_evaluation(args, output, 1)
            self.assertEqual(result["status"], "PASS")
            report = (output / "official_evaluation.txt").read_text(encoding="utf-8")
            self.assertIn("--input", report)
            self.assertIn("Total Accuracy", report)

    def test_counts_report_audio_crop_and_padding_statistics(self) -> None:
        counts = self.module._counts(
            types.SimpleNamespace(
                state={
                    "0": {
                        "status": "parsed",
                        "id": "long",
                        "audio_original_duration_seconds": 12.0,
                        "audio_was_cropped": True,
                        "audio_was_padded": False,
                    },
                    "1": {
                        "status": "unparseable",
                        "id": "short",
                        "audio_original_duration_seconds": 1.0,
                        "audio_was_cropped": False,
                        "audio_was_padded": True,
                    },
                    "2": {"status": "skipped", "reason": "audio_decode_failed"},
                }
            )
        )
        self.assertEqual(counts["generation_completed"], 2)
        self.assertEqual(counts["audio"]["over_ten_seconds"], 1)
        self.assertEqual(counts["audio"]["cropped_ids"], ["long"])
        self.assertEqual(counts["audio"]["padded_records"], 1)

    def test_resume_store_deduplicates_and_materializes_in_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.module.ProgressStore(output) as store:
                store.append_raw(
                    {
                        "status": "unparseable",
                        "row_index": 4,
                        "id": "id-4",
                        "model_output": "",
                        "official_record": {"id": "id-4", "question": "q4", "choices": ["x", "y"], "answer": "x", "task": "sound", "difficulty": "easy"},
                    }
                )
                store.append_raw(
                    {
                        "status": "parsed",
                        "row_index": 1,
                        "id": "id-1",
                        "model_output": "x",
                        "official_record": {"id": "id-1", "question": "q1", "choices": ["x", "y"], "answer": "x", "task": "sound", "difficulty": "easy"},
                    }
                )
                store.append_skip({"status": "skipped", "row_index": 2, "id": "id-2", "stage": "audio", "reason": "audio_missing"})
            with self.module.ProgressStore(output) as recovered:
                self.assertTrue(recovered.has_terminal(1, "id-1"))
                self.assertTrue(recovered.has_terminal(2, "id-2"))
                self.assertTrue(recovered.has_terminal(4, "id-4"))
                predictions = self.module.materialize_predictions(recovered.state.values())
            self.assertEqual([item["id"] for item in predictions], ["id-1", "id-4"])
            self.assertEqual(predictions[0]["model_output"], "x")
            self.assertEqual(predictions[1]["model_output"], "")
            self.assertNotIn("official_record", predictions[0])

    def test_resume_recovers_raw_record_when_progress_append_was_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "raw_generations.jsonl").write_text(
                '{"status":"parsed","row_index":0,"id":"id-0","model_output":"x",'
                '"official_record":{"id":"id-0","choices":["x","y"]}}\n',
                encoding="utf-8",
            )
            (output / "progress.jsonl").write_text(
                '{"status":"parsed","record":{"status":"parsed","row_index":1,"id":"id-1",'
                '"model_output":"y","official_record":{"id":"id-1","choices":["x","y"]}}}\n',
                encoding="utf-8",
            )
            with self.module.ProgressStore(output) as recovered:
                self.assertTrue(recovered.has_terminal(0, "id-0"))
                self.assertTrue(recovered.has_terminal(1, "id-1"))

    def test_source_and_wrappers_lock_protocol(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        for marker in (
            "ParquetFile",
            "iter_batches",
            "use_threads=False",
            "SCRIPT_DIR",
            "for import_root in (SCRIPT_DIR, ROOT)",
            "other_attributes.id",
            "fixed-order",
            "SMOKE_ROWS = 5",
            "model_output",
            "DEFAULT_MAX_NEW_TOKENS",
            "use_cache=False",
            "logical_trace",
            "skipped.jsonl",
            "progress.jsonl",
            "official_evaluation.txt",
            "_run_official_evaluation",
        ):
            self.assertIn(marker, text)
        self.assertIn('"permutation_majority_vote": False', text)
        self.assertNotIn("audio_id_path", text)
        self.assertIn("No model_prediction", text)
        self.assertIn("conda activate rsmol", INNER_SH.read_text(encoding="utf-8"))
        submit = SUBMIT_SH.read_text(encoding="utf-8")
        self.assertIn("vc submit", submit)
        self.assertIn("-p pdgpu-5090", submit)
        self.assertIn("-g 1", submit)
        self.assertIn("test_mini.parquet", submit)
        self.assertIn("mmau-test-mini.json", submit)

    def test_smoke_output_can_be_promoted_to_full(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            common = dict(
                output_dir=output,
                checkpoint=Path("/ckpt"),
                dataset_dir=Path("/data"),
                parquet=Path("/data/test_mini.parquet"),
                metadata_json=Path("/data/mmau-test-mini.json"),
                evaluation_script=Path("/data/evaluation.py"),
                htsat_checkpoint=Path("/models/htsat.ckpt"),
                mellow_root=Path("/code/mellow"),
                max_prompt_tokens=129,
                max_new_tokens=16,
            )
            self.module._ensure_output_dir(types.SimpleNamespace(mode="smoke", **common))
            self.module._ensure_output_dir(types.SimpleNamespace(mode="full", **common))
            config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["mode"], "full")
            with self.assertRaises(RuntimeError):
                self.module._ensure_output_dir(types.SimpleNamespace(mode="smoke", **common))


if __name__ == "__main__":
    unittest.main()
