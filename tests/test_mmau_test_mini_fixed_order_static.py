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

    def test_model_output_preparser_only_removes_leading_abcd_label(self) -> None:
        self.assertTrue(hasattr(self.module, "prepare_model_output_for_official_scorer"))
        source = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("selected_option", source)
        self.assertNotIn("parse_method", source)
        self.assertIn("prepare_model_output_for_official_scorer", source)
        prepare = self.module.prepare_model_output_for_official_scorer
        self.assertEqual(prepare("a) Wind and stream"), "Wind and stream")
        self.assertEqual(prepare("  D) answer"), "answer")
        self.assertEqual(prepare("e) fifth choice"), "e) fifth choice")
        self.assertEqual(prepare("The answer is a) first"), "The answer is a) first")

    def test_fixed_order_prompt_and_choice_alignment(self) -> None:
        prompt = self.module.build_fixed_order_prompt("Which one?", ["first", "second"])
        self.assertIn("a) first", prompt)
        self.assertIn("b) second", prompt)
        self.assertEqual(
            prompt,
            "Which one? a) first b) second",
        )
        self.assertNotIn("Choices:", prompt)
        self.assertTrue(self.module.choices_match_fixed_order(["(A) first", "B. second"], ["first", "second"]))
        self.assertFalse(self.module.choices_match_fixed_order(["second", "first"], ["first", "second"]))
        self.assertFalse(self.module.choices_match_fixed_order(["(B) first", "A. second"], ["first", "second"]))

    def test_mellow_author_reply_prompt_and_choice_label_metric(self) -> None:
        prompt = self.module.build_mellow_author_reply_prompt(
            "Which Sound!",
            ["Dog", "CAT"],
        )
        self.assertEqual(prompt, "which sound? a) dog b) cat")
        labeled = self.module.mellow_author_reply_labeled_answer("CAT", ["Dog", "CAT"])
        self.assertEqual(labeled, "b) cat")
        self.assertTrue(self.module.mellow_author_reply_choice_is_correct("b) anything", labeled))
        self.assertFalse(self.module.mellow_author_reply_choice_is_correct(" b) anything", labeled))
        self.assertFalse(self.module.mellow_author_reply_choice_is_correct("cat", labeled))

    def test_mellow_author_reply_scorer_keeps_complete_denominator(self) -> None:
        score = self.module.evaluate_mellow_author_reply_predictions([
            {
                "id": "sound-1",
                "task": "sound",
                "difficulty": "easy",
                "choices": ["Dog", "Cat"],
                "answer": "Cat",
                "model_output": "b) cat",
            },
            {
                "id": "music-1",
                "task": "music",
                "difficulty": "hard",
                "choices": ["Piano", "Guitar"],
                "answer": "Piano",
                "model_output": "",
            },
        ])
        self.assertEqual(score["total"], {"correct": 1, "total": 2, "accuracy_percent": 50.0})
        self.assertEqual(score["task"]["sound"]["total"], 1)
        self.assertEqual(score["task"]["music"]["total"], 1)
        self.assertEqual(score["record_errors"]["total"], 0)

    def test_malformed_answer_mapping_is_counted_wrong_instead_of_raising(self) -> None:
        score = self.module.evaluate_mellow_author_reply_predictions([
            {
                "id": "speech-leading-space",
                "task": "speech",
                "difficulty": "medium",
                "choices": ["Exact answer.", "Other answer."],
                "answer": " Exact answer.",
                "model_output": "a) exact answer",
            }
        ])
        self.assertEqual(score["total"], {"correct": 0, "total": 1, "accuracy_percent": 0.0})
        self.assertEqual(score["record_errors"]["total"], 1)
        self.assertEqual(
            score["record_errors"]["counts"],
            {"answer_not_exact_choice": 1},
        )
        self.assertEqual(
            score["rows"][0]["scoring_error"]["policy"],
            "counted_incorrect_without_shrinking_denominator",
        )

    def test_mellow_author_reply_uses_id_wav_and_smoke_materializes_skips(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('filename = f"{sample_id}.wav"', source)
        self.assertIn("official_records[:SMOKE_ROWS]", source)

    def test_mellow_author_reply_audio_segment_repeats_and_random_crops(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is unavailable in the dependency-light local test environment")

        short = torch.tensor([[1.0, 2.0, 3.0]])
        repeated, audit = self.module.mellow_author_reply_audio_segment(
            short,
            target_rate=1,
            seconds=5,
        )
        self.assertEqual(repeated.tolist(), [[1.0, 2.0, 3.0, 1.0, 2.0]])
        self.assertEqual(audit["policy"], "repeat_then_trim")

        class FakeRng:
            @staticmethod
            def randrange(value):
                self.assertEqual(value, 2)
                return 1

        long = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
        cropped, audit = self.module.mellow_author_reply_audio_segment(
            long,
            target_rate=1,
            seconds=3,
            rng=FakeRng(),
        )
        self.assertEqual(cropped.tolist(), [[1.0, 2.0, 3.0]])
        self.assertEqual(audit["crop_start"], 1)

    def test_real_choice_prefixes_are_not_misread_as_labels(self) -> None:
        choices = [
            "F. Scott Fitzgerald",
            "J.D. Salinger",
            "E-guitar",
            "E-bass",
            "B:maj/1",
        ]
        self.assertTrue(self.module.choices_match_fixed_order(choices, choices))
        self.assertEqual(self.module._strip_choice_label("F. Scott Fitzgerald"), "F. Scott Fitzgerald")
        self.assertEqual(self.module._strip_choice_label("E-guitar"), "E-guitar")
        self.assertEqual(self.module._strip_choice_label("B:maj/1"), "B:maj/1")
        labeled = [f"({chr(ord('A') + index)}) {choice}" for index, choice in enumerate(choices)]
        self.assertTrue(self.module.choices_match_fixed_order(labeled, choices))

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
            self.assertEqual(result["reported_total"], 1)
            self.assertEqual(result["total_accuracy_percent"], 0.0)
            report = (output / "official_evaluation.txt").read_text(encoding="utf-8")
            self.assertIn("--input", report)
            self.assertIn("Total Accuracy", report)

    def test_inference_only_is_not_reported_as_comparable_pass(self) -> None:
        self.assertEqual(
            self.module.overall_evaluation_status("PASS", "NOT_REQUESTED"),
            self.module.INFERENCE_ONLY_STATUS,
        )
        self.assertEqual(
            self.module.overall_evaluation_status("PASS", "PASS"),
            "PASS",
        )
        self.assertEqual(
            self.module.overall_evaluation_status("FAILED", "PASS"),
            "FAILED",
        )

    def test_pipeline_failure_overwrites_stale_smoke_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            log = output / "official_evaluation.txt"
            log.write_text("Total Accuracy: 20.00% over 5 samples\n", encoding="utf-8")
            args = types.SimpleNamespace(run_official_evaluation=True, mode="full")
            error = RuntimeError("formal coverage mismatch")
            result = self.module._block_official_evaluation(args, output, 1000, error)
            self.assertEqual(result["status"], "BLOCKED_BY_PIPELINE_FAILURE")
            content = log.read_text(encoding="utf-8")
            self.assertNotIn("over 5 samples", content)
            self.assertIn("prediction_count: 1000", content)
            self.assertIn("formal coverage mismatch", content)

    def test_counts_report_audio_crop_and_padding_statistics(self) -> None:
        counts = self.module._counts(
            types.SimpleNamespace(
                state={
                    "0": {
                        "status": "generated",
                        "id": "long",
                        "audio_original_duration_seconds": 12.0,
                        "audio_was_cropped": True,
                        "audio_was_padded": False,
                    },
                    "1": {
                        "status": "generated",
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
                        "status": "generated",
                        "row_index": 4,
                        "id": "id-4",
                        "model_output": "",
                        "official_record": {"id": "id-4", "question": "q4", "choices": ["x", "y"], "answer": "x", "task": "sound", "difficulty": "easy"},
                    }
                )
                store.append_raw(
                    {
                        "status": "generated",
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

    def test_full_materialization_keeps_official_denominator_and_order(self) -> None:
        state = [
            {
                "status": "generated",
                "row_index": 1,
                "id": "id-1",
                "model_output": "b) second",
            }
        ]
        official = [
            {"id": "id-0", "choices": ["first", "second"], "answer": "first"},
            {"id": "id-1", "choices": ["first", "second"], "answer": "second"},
        ]
        predictions = self.module.materialize_predictions(state, official)
        self.assertEqual([item["id"] for item in predictions], ["id-0", "id-1"])
        self.assertEqual([item["model_output"] for item in predictions], ["", "b) second"])

    def test_resume_recovers_raw_record_when_progress_append_was_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "raw_generations.jsonl").write_text(
                '{"status":"generated","row_index":0,"id":"id-0","model_output":"x",'
                '"official_record":{"id":"id-0","choices":["x","y"]}}\n',
                encoding="utf-8",
            )
            (output / "progress.jsonl").write_text(
                '{"status":"generated","record":{"status":"generated","row_index":1,"id":"id-1",'
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
            "MMAU-v05.15.25",
            "MMAU_METADATA_CANONICAL_SHA256",
            "compact_single_audio_prefix",
            "reasonaqa_lowercase_labels_no_choices_prefix_v1",
        ):
            self.assertIn(marker, text)
        self.assertIn('"permutation_majority_vote": False', text)
        self.assertNotIn("audio_id_path", text)
        self.assertNotIn("leading_full_text", text)
        self.assertNotIn("contains skipped rows", text)
        self.assertIn("skipped_rows_scored_as_incorrect", text)
        self.assertIn("checkpoint-037810", text)
        self.assertIn("conda activate rsmol", INNER_SH.read_text(encoding="utf-8"))
        submit = SUBMIT_SH.read_text(encoding="utf-8")
        self.assertIn("vc submit", submit)
        self.assertIn("-p pdgpu-5090", submit)
        self.assertIn("-g 1", submit)
        self.assertIn("test_mini.parquet", submit)
        self.assertIn("mmau-test-mini.json", submit)
        self.assertIn("--max-new-tokens 32", submit)

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
                max_new_tokens=32,
            )
            self.module._ensure_output_dir(types.SimpleNamespace(mode="smoke", **common))
            self.module._ensure_output_dir(types.SimpleNamespace(mode="full", **common))
            config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["mode"], "full")
            with self.assertRaises(RuntimeError):
                self.module._ensure_output_dir(types.SimpleNamespace(mode="smoke", **common))


if __name__ == "__main__":
    unittest.main()
