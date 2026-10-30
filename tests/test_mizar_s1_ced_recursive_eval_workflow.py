"""Run the full evaluator control flow with 1000 synthetic benchmark rows."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/RSmol/scripts"))
import evaluate_mizar_s1_ced_recursive_benchmark as runner


class WorkflowTests(unittest.TestCase):
    def fixture(self, directory, benchmark="mmar"):
        root = Path(directory)
        scorer = root / "evaluation.py"
        scorer.write_text('output_key = "model_output"\n', encoding="utf-8")
        args = runner.parse_args(["--benchmark", benchmark, "--eval-variant", "recursive_5_10_5",
                                  "--evaluation-script", str(scorer), "--output-dir", str(root / "eval")])
        records = [{"id": str(i), "question": "Question?", "choices": ["One", "Two"],
                    "answer": "One", "audio_path": "fixture.wav"} for i in range(1000)]
        runtime = (None, None, None, None, None, {"checkpoint_step": 17313}, {})
        identity = {"benchmark": benchmark, "eval_variant": args.eval_variant,
                    "checkpoint": "fixture", "prediction_key": "model_output"}
        return args, records, runtime, identity

    def execute(self, args, records, runtime, identity, generate):
        def prepare(row, **_):
            return {**row, "prompt": "question? a) one b) two", "official_record": dict(row)}
        pq = SimpleNamespace(ParquetFile=lambda _: SimpleNamespace(metadata=SimpleNamespace(num_rows=1000)))
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            stack.enter_context(patch.object(runner, "_validate_input_paths"))
            stack.enter_context(patch.object(runner, "_identity", return_value=dict(identity)))
            stack.enter_context(patch.object(runner, "_metadata_records", return_value=(records, {"status": "PASS"}, "model_output")))
            stack.enter_context(patch.object(runner, "_call_model_loader", return_value=runtime))
            stack.enter_context(patch.object(runner, "_prepare_mmar", side_effect=lambda row, *_: prepare(row)))
            stack.enter_context(patch.object(runner.mmau, "_prepare_metadata_row", side_effect=prepare))
            stack.enter_context(patch.object(runner.mmau, "build_metadata_index", return_value={}))
            stack.enter_context(patch.object(runner.mmau, "iter_parquet_rows", side_effect=lambda *_a, **_k: enumerate(records)))
            stack.enter_context(patch.object(runner.mmau, "extract_row_id", side_effect=lambda row: (row["id"], None)))
            stack.enter_context(patch.object(runner, "make_mmau_audio_decoder", return_value=object()))
            stack.enter_context(patch.dict(sys.modules, {"pyarrow": SimpleNamespace(parquet=pq), "pyarrow.parquet": pq}))
            stack.enter_context(patch.object(runner, "generate", side_effect=generate))
            scorer = stack.enter_context(patch.object(runner, "run_official_scorer", return_value={"status": "PASS", "reported_rows": 1000}))
            report = runner.run(args)
        return report, scorer

    @staticmethod
    def generation(*_):
        return {"generated_text": "  A) One <|endoftext|>  ", "generated_token_count": 3}

    def test_full_1000_rows_for_both_benchmarks_preserve_raw_predictions(self):
        for benchmark in ("mmau", "mmar"):
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                args, records, runtime, identity = self.fixture(directory, benchmark)
                generate = Mock(side_effect=self.generation)
                report, scorer = self.execute(args, records, runtime, identity, generate)
                self.assertEqual(report["status"], "PASS", report.get("fatal_error"))
                self.assertEqual(generate.call_count, 1000)
                self.assertEqual(report["records"]["generated"], 1000)
                self.assertEqual(report["inference_coverage"]["skipped"], 0)
                scorer.assert_called_once()
                filename = "predictions_fixed_order.json" if benchmark == "mmau" else "predictions_official.json"
                predictions = json.loads((args.output_dir / filename).read_text(encoding="utf-8"))
                self.assertEqual([p["id"] for p in predictions], [str(i) for i in range(1000)])
                self.assertTrue(all(p["model_output"] == self.generation()["generated_text"] for p in predictions))

    def test_interrupted_run_resumes_without_repeating_completed_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            args, records, runtime, identity = self.fixture(directory)
            generated = Mock(side_effect=[self.generation(), self.generation(), RuntimeError("interrupted")])
            failed, scorer = self.execute(args, records, runtime, identity, generated)
            self.assertEqual(failed["status"], "FAILED")
            scorer.assert_not_called()
            self.assertEqual(failed["records"]["generated"], 2)
            raw = args.output_dir / "raw_generations.jsonl"
            original = raw.read_bytes()
            resumed = Mock(side_effect=self.generation)
            passed, scorer = self.execute(args, records, runtime, identity, resumed)
            self.assertEqual(passed["status"], "PASS", passed.get("fatal_error"))
            self.assertEqual(resumed.call_count, 998)
            self.assertEqual(resumed.call_args_list[0].args[1]["id"], "2")
            self.assertTrue(raw.read_bytes().startswith(original))
            self.assertEqual(len(raw.read_text(encoding="utf-8").splitlines()), 1000)
            scorer.assert_called_once()

    def test_skipped_record_blocks_successful_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            args, records, runtime, identity = self.fixture(directory)
            failed, _ = self.execute(args, records, runtime, identity, Mock(side_effect=RuntimeError("stop")))
            self.assertEqual(failed["status"], "FAILED")
            with runner.mmau.ProgressStore(args.output_dir) as store:
                store.append_skip({"row_index": 0, "id": "0", "status": "skipped", "reason": "fixture"})
            report, scorer = self.execute(args, records, runtime, identity, self.generation)
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["records"]["skipped"], 1)
            scorer.assert_not_called()

    def test_incomplete_traversal_or_wrong_loaded_step_blocks_scoring(self):
        for kind in ("rows", "step"):
            with tempfile.TemporaryDirectory() as directory:
                args, records, runtime, identity = self.fixture(directory)
                if kind == "rows": records = records[:-1]
                else: runtime[5]["checkpoint_step"] = 20
                generate = Mock(side_effect=self.generation)
                report, scorer = self.execute(args, records, runtime, identity, generate)
                self.assertEqual(report["status"], "FAILED")
                generate.assert_not_called()
                scorer.assert_not_called()

    def test_output_identity_rejects_other_variant_checkpoint_protocol_and_unowned_files(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _, identity = self.fixture(directory)
            runner.claim_output(args, identity)
            runner.claim_output(args, identity)
            for field in ("eval_variant", "checkpoint", "benchmark"):
                with self.assertRaises(RuntimeError):
                    runner.claim_output(args, {**identity, field: "changed"})
            with self.assertRaises(RuntimeError):
                runner.claim_output(args, {**identity, "generation": {"use_cache": True}})
            args.output_dir = Path(directory) / "unowned"
            args.output_dir.mkdir()
            (args.output_dir / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError): runner.claim_output(args, identity)

    def test_official_scorer_must_succeed_and_confirm_full_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _, _ = self.fixture(directory)
            args.output_dir.mkdir()
            prediction = args.output_dir / "predictions.json"
            prediction.write_text("[]", encoding="utf-8")
            for returncode, rows, expected in ((0, 1000, "PASS"), (0, 999, "FAILED"), (1, 1000, "FAILED")):
                completed = SimpleNamespace(returncode=returncode, stdout=f"Total Accuracy: 50.00% over {rows} samples", stderr="")
                with patch.object(runner.subprocess, "run", return_value=completed) as execute:
                    result = runner.run_official_scorer(args, prediction, 1000)
                self.assertEqual(result["status"], expected)
                self.assertEqual(execute.call_args.args[0], [sys.executable, str(args.evaluation_script), "--input", str(prediction)])


if __name__ == "__main__":
    unittest.main()
