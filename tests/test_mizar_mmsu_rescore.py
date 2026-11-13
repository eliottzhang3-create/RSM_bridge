"""MMSU real scorer compatibility, CPU-only recovery and future inference flow."""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import test_mizar_s1_ced_mmsu_eval as legacy
import test_mizar_s1_ced_mmsu_shell as shell
import test_mizar_mmsu_selected5 as presets

ROOT, RSMOL, SCORER = legacy.ROOT, shell.RSMOL, legacy.SCORER
sys.path.insert(0, str(RSMOL))
from mizar_s1_ced_mmsu_eval import parsing, rescore, scoring, storage

LAUNCHER = "rescore_mizar_mmsu_selected5.sh"


class ParsingTests(unittest.TestCase):
    def test_actual_lowercase_responses_match_answer_text_regardless_of_letter(self):
        cases = [("a. Philippines <|endoftext|>", "Philippines", True),
                 ("a. new Zealand <|endoftext|>", "New Zealand", True),
                 ("d. Hong Kong <|endoftext|>", "Hong Kong", True),
                 ("a. Australia <|endoftext|>", "Australia", True),
                 ("d. India <|endoftext|>", "India", True),
                 ("(c) INDIA", "India", True), ("[d] india", "India", True),
                 (" a.   new\n\tZEALAND <|endoftext|> ", "New  Zealand", True),
                 ("A", "India", False), ("The answer is A.", "India", False),
                 ("a. China", "India", False), ("", "India", False),
                 ("<|endoftext|>", "India", False), ("None", "India", False)]
        for raw, target, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(parsing.parse_response(raw, target)["matched"], expected)

    def test_contains_complete_text_with_boundaries_and_declared_negation_behavior(self):
        self.assertTrue(parsing.parse_response("I think Hong Kong is the location.", "Hong Kong")["matched"])
        self.assertFalse(parsing.parse_response("The sound is noise.", "no")["matched"])
        self.assertFalse(parsing.parse_response("The value is 13.", "3")["matched"])
        self.assertFalse(parsing.parse_response("a. Australia", "a")["matched"])
        self.assertFalse(parsing.parse_response("Hello world", "Hello, world")["matched"])
        # The requested containment metric is explicitly lexical, not semantic.
        self.assertTrue(parsing.parse_response("It is not India; maybe China.", "India")["matched"])

    def test_5000_normalized_official_scorer_keeps_invalid_wrong_and_full_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = [{"official_record": legacy.record(i), "generation": legacy.generation("a. India <|endoftext|>")}
                     for i in range(5000)]
            saved[0]["generation"] = legacy.generation("I cannot decide")
            saved[1]["generation"] = legacy.generation("")
            saved[2]["generation"] = legacy.generation("c. China <|endoftext|>")
            saved[3]["generation"] = legacy.generation("d. India <|endoftext|>")
            saved[4]["official_record"]["answer_gt"] = "China"
            saved[4]["generation"] = legacy.generation("a. China <|endoftext|>")
            saved[5]["official_record"]["choice_d"] = "India"
            raw = root / "predictions_official.jsonl"
            scoring.write_predictions(raw, saved)
            original = raw.read_bytes()
            parsed, score = parsing.score_records(parsing.read_complete_jsonl(raw), SCORER, root)
            self.assertEqual(score["status"], "PASS")
            self.assertEqual(score["actual_denominator"], 5000)
            self.assertAlmostEqual(score["overall_accuracy"], 4997 / 5000)
            self.assertEqual((parsed["matched_rows"], parsed["nonmatching_rows"]), (4997, 3))
            self.assertEqual(raw.read_bytes(), original)
            rows = parsing.read_complete_jsonl(root / "predictions_normalized.jsonl")
            self.assertEqual([r["response"] for r in rows[:6]], ["", "", "", "A", "C", "A"])
            self.assertEqual(parsed["multiple_gt_choice_rows"], 1)
            self.assertEqual(set(rows[0]), set(legacy.dataset.METADATA) | {"response"})
            self.assertIsNone(rows[0]["choice_b"])

    def test_all_invalid_are_counted_wrong_not_dropped_or_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [{**legacy.record(i), "response": "No explicit answer"} for i in range(5000)]
            parsed, score = parsing.score_records(rows, SCORER, Path(directory))
            self.assertEqual((parsed["nonmatching_rows"], score["actual_denominator"], score["overall_accuracy"]), (5000, 5000, 0))
            self.assertEqual(score["status"], "PASS")


class OfflineTests(unittest.TestCase):
    def fixture(self, directory, *, raw=True):
        source = Path(directory) / "existing output's MMSU"
        source.mkdir()
        saved = []
        for index in range(5000):
            record = legacy.record(index)
            generation = legacy.generation("a. India <|endoftext|>")
            saved.append({"row_index": index, "id": record["id"], "official_record": record,
                          "record_sha256": storage.object_sha(record), "generation": generation,
                          "generation_sha256": storage.object_sha(generation)})
        if raw:
            with (source / "raw_generations.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                for value in saved: stream.write(json.dumps(value) + "\n")
        scoring.write_predictions(source / "predictions_official.jsonl", saved)
        for name in ("run_config.json", "evaluation_report.json", "model_audit.json"):
            (source / name).write_text(json.dumps({"fixture": True, "status": "FAILED"}), encoding="utf-8")
        args = rescore.parse_args(["--source-output-dir", str(source), "--evaluation-script", str(SCORER)])
        return args, saved

    def test_full_hash_validated_offline_run_and_repeat_preserve_every_source_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory)
            before = {p.name: p.read_bytes() for p in args.source_output_dir.iterdir() if p.is_file()}
            report = rescore.run(args)
            self.assertEqual(report["status"], "PASS", report["fatal_error"])
            self.assertFalse(report["gpu_inference"])
            self.assertEqual(report["official_evaluation"]["overall_accuracy"], 1)
            self.assertEqual(report["source"]["kind"], "hash_validated_raw_generations")
            repeated = rescore.run(args)
            self.assertEqual(repeated["status"], "PASS", repeated["fatal_error"])
            for name, value in before.items():
                self.assertEqual((args.source_output_dir / name).read_bytes(), value)

    def test_prediction_only_fallback_when_raw_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory, raw=False)
            report = rescore.run(args)
            self.assertEqual(report["status"], "PASS", report["fatal_error"])
            self.assertEqual(report["source"]["kind"], "raw_predictions_only")

    def test_broken_raw_hash_never_falls_back_to_other_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            args, saved = self.fixture(directory)
            raw = args.source_output_dir / "raw_generations.jsonl"
            saved[0]["generation"]["generated_text"] = "C"
            raw.write_text("".join(json.dumps(row) + "\n" for row in saved), encoding="utf-8")
            original = raw.read_bytes()
            report = rescore.run(args)
            self.assertEqual(report["status"], "FAILED")
            self.assertIn("identity/content mismatch", report["fatal_error"]["error"])
            self.assertEqual(raw.read_bytes(), original)
            self.assertFalse((args.output_dir / "evaluation_report.json").exists())

    def test_incomplete_corrupt_duplicate_and_nontext_sources_refused_without_mutation(self):
        for kind in ("short", "tail", "corrupt", "duplicate", "nontext", "disagree"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                args, saved = self.fixture(directory)
                raw = args.source_output_dir / "raw_generations.jsonl"
                if kind in ("short", "tail", "corrupt"):
                    raw.write_bytes(raw.read_bytes().split(b"\n", 1)[0] + (b"\n" if kind == "short" else b""))
                    if kind == "corrupt": raw.write_bytes(b"{bad json}\n")
                elif kind in ("duplicate", "nontext"):
                    raw.unlink()
                    rows = parsing.read_complete_jsonl(args.source_output_dir / "predictions_official.jsonl")
                    if kind == "duplicate": rows[1]["id"] = rows[0]["id"]
                    else: rows[0]["response"] = None
                    (args.source_output_dir / "predictions_official.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                else:
                    saved[0]["generation"]["generated_text"] = "C"
                    saved[0]["generation_sha256"] = storage.object_sha(saved[0]["generation"])
                    raw.write_text("".join(json.dumps(row) + "\n" for row in saved), encoding="utf-8")
                before = {p.name: p.read_bytes() for p in args.source_output_dir.iterdir() if p.is_file()}
                report = rescore.run(args)
                self.assertEqual(report["status"], "FAILED")
                for name, value in before.items(): self.assertEqual((args.source_output_dir / name).read_bytes(), value)

    def test_output_identity_change_and_source_alias_do_not_overwrite_previous_report(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory)
            self.assertEqual(rescore.run(args)["status"], "PASS")
            original = (args.output_dir / "evaluation_report.json").read_bytes()
            (args.source_output_dir / "run_config.json").write_text('{"changed": true}', encoding="utf-8")
            report = rescore.run(args)
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual((args.output_dir / "evaluation_report.json").read_bytes(), original)
            with self.assertRaises(SystemExit):
                rescore.parse_args(["--source-output-dir", str(args.source_output_dir),
                                    "--output-dir", str(args.source_output_dir)])

    def test_module_import_does_not_load_runtime_torch_parquet_or_audio(self):
        code = ('import sys; from mizar_s1_ced_mmsu_eval import rescore; '
                'assert not any(n in sys.modules for n in '
                '("torch","transformers","pyarrow","soundfile","mizar_s1_ced_mmsu_eval.runtime"))')
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=RSMOL, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class FutureWorkflowTests(unittest.TestCase):
    fixture = legacy.WorkflowTests.fixture
    execute = legacy.WorkflowTests.execute

    def test_future_full_evaluation_uses_same_parser_without_changing_raw_response(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rows, loaded = self.fixture(directory)
            generate = Mock(side_effect=lambda *a, **k: legacy.generation("a. India <|endoftext|>"))
            report, _, _, _ = self.execute(args, rows, loaded, generate)
            self.assertEqual(report["status"], "PASS", report["fatal_error"])
            self.assertEqual(report["response_parsing"]["matched_rows"], 5000)
            self.assertEqual(report["official_evaluation"]["actual_denominator"], 5000)
            self.assertEqual(report["official_evaluation"]["response_policy"], parsing.VERSION)
            raw = parsing.read_complete_jsonl(args.output_dir / "raw_generations.jsonl")
            self.assertEqual(raw[0]["generation"]["generated_text"], "a. India <|endoftext|>")
            legacy_predictions = parsing.read_complete_jsonl(args.output_dir / "predictions_official.jsonl")
            self.assertEqual(legacy_predictions[0]["response"], "a. India <|endoftext|>")


class ShellTests(unittest.TestCase):
    setUp = shell.ShellTests.setUp

    def fixture(self, directory):
        path, rs, env = shell.ShellTests.fixture(self, directory)
        shutil.copy2(RSMOL / LAUNCHER, rs / LAUNCHER)
        # The shared fixture copies the full package, including real rescore.sh.
        return path, rs, env

    def launch(self, rs, env, args):
        return subprocess.run([self.bash, str(rs / LAUNCHER), *args], cwd=rs, env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)

    def test_five_real_compute_shells_and_cpu_parser_match_original_submission_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            result = self.launch(rs, env, ["all", "--check-args"])
            self.assertEqual(result.returncode, 0, result.stderr)
            configs = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(configs), 5)
            for target, config in zip(presets.TARGETS, configs):
                source = (presets.OUTPUT_ROOT + "/mizar_official_stage3_q4_seed20260905_matched_ced_v1/mmsu"
                          if target == "official_mizar" else
                          presets.OUTPUT_ROOT + "/" + presets.TRAINED[target][1] + "/eval/mmsu_step17313")
                self.assertTrue(config["source_output_dir"].replace("\\", "/").endswith(source))
                self.assertTrue(config["output_dir"].replace("\\", "/").endswith(source + "/" + rescore.SUBDIRECTORY))
            self.assertFalse((path / "submission.bin").exists())

    def test_cpu_shell_really_scores_spaced_paths_and_never_submits(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            source_root = path / "fixture source"
            source_root.mkdir()
            args, _ = OfflineTests.fixture(self, source_root)
            result = subprocess.run([self.bash, str(rs / "mizar_s1_ced_mmsu_eval/rescore.sh"),
                "--source-output-dir", str(args.source_output_dir), "--evaluation-script", str(SCORER)],
                cwd=rs, env=env, capture_output=True, text=True, encoding="utf-8", timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["official_evaluation"]["actual_denominator"], 5000)
            self.assertFalse((path / "submission.bin").exists())

    def test_batch_attempts_every_target_after_one_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            compute = rs / "mizar_s1_ced_mmsu_eval/rescore.sh"
            compute.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$MMSU_VC_CAPTURE"\n'
                '[[ "$*" != *"mizar_official_stage3"* ]]\n', encoding="utf-8", newline="\n")
            result = self.launch(rs, env, ["all"])
            self.assertEqual(result.returncode, 1)
            self.assertEqual(len((path / "submission.bin").read_text().splitlines()), 5)
            self.assertIn("continuing", result.stderr)

    def test_invalid_batch_arguments_fail_before_any_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            for args in ([], ["unknown"], ["all", "--output-dir", "/same"], ["base", "--checkpoint", "/wrong"],
                         ["base", "--evaluation-script"], ["base", "--output-dir="]):
                result = self.launch(rs, env, args)
                self.assertEqual(result.returncode, 2, (args, result.stderr))
            self.assertFalse((path / "submission.bin").exists())

    def test_shell_syntax(self):
        for path in (RSMOL / LAUNCHER, RSMOL / "mizar_s1_ced_mmsu_eval/rescore.sh"):
            result = subprocess.run([self.bash, "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
