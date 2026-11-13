"""MMSU correctness and workflow tests. Model inference is an explicit fixture."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/RSmol"))
from mizar_s1_ced_mmsu_eval import dataset, evaluate, runtime, scoring, storage

SCORER = ROOT / "tests/fixtures/mmsu/mmsu_evaluation.py"


def record(index=0):
    return dict(id=f"sample-{index}", task_name="test-task", question="Which Country?",
                choice_a="India", choice_b=None, choice_c="China", choice_d="Japan",
                answer_gt="India", category="Perception", **{"sub-category": "Linguistics",
                "sub-sub-category": "Semantics", "linguistics_sub_discipline": "test"})


def generation(text="  A<|endoftext|>  "):
    return dict(generated_text=text, generated_text_raw=text, generated_token_count=2,
                generated_token_ids=[3, 0], prompt_truncated=True,
                prompt_original_token_count=280, audio_right_pad_samples=32000)


class ProtocolTests(unittest.TestCase):
    def test_official_prompt_preserves_case_null_slots_and_excludes_labels(self):
        row = record()
        expected = ("Choose the most suitable answer from options A, B, C, and D. "
                    "You must respond with only A, B, C, or D.\n\nQuestion: Which Country?\n\n"
                    "A. India\nB. None\nC. China\nD. Japan")
        self.assertEqual(dataset.prompt(row), expected)
        row["answer_gt"], row["category"], row["task_name"] = "secret", "secret2", "secret3"
        self.assertEqual(dataset.prompt(row), expected)

    def test_metadata_validation_and_exact_text_answers(self):
        dataset.validate_record(record())
        for key, value in (("answer_gt", "india"), ("question", None), ("choice_a", 42), ("id", True)):
            row = record()
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                dataset.validate_record(row)

    def test_global_metadata_count_duplicates_and_null_choices(self):
        rows = [record(i) for i in range(5000)]
        actual, audit = dataset.audit_records(enumerate(rows))
        self.assertEqual(actual, rows)
        self.assertEqual(audit["null_choices"], {"choice_b": 5000})
        self.assertEqual(audit["task_count"], 1)
        self.assertEqual(audit["multiple_matching_choices"], 0)
        with self.assertRaisesRegex(ValueError, "5000 metadata"):
            dataset.audit_records(enumerate(rows[:-1]))
        rows[1667] = rows[0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            dataset.audit_records(enumerate(rows))
        with self.assertRaisesRegex(ValueError, "noncontiguous"):
            dataset.audit_records([(1, rows[0])])

    def test_shard_paths_are_exact_and_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "data").mkdir()
            for name in reversed(dataset.SHARD_NAMES):
                (path / "data" / name).touch()
            self.assertEqual([p.name for p in dataset.shard_paths(path)], list(dataset.SHARD_NAMES))
            (path / "data/extra.parquet").touch()
            with self.assertRaisesRegex(ValueError, "exactly"):
                dataset.shard_paths(path)

    def test_generation_settings_only_change_mmsu_prompt_and_scorer(self):
        differences = {key for key in runtime.PROTOCOL if runtime.PROTOCOL[key] != runtime.base.PROTOCOL[key]}
        self.assertEqual(differences, {"version", "prompt_policy", "scoring"})
        self.assertEqual(runtime.PROTOCOL["max_new_tokens"], 300)
        self.assertIs(runtime.PROTOCOL["use_cache"], False)
        original = copy.deepcopy(runtime.base.PROTOCOL)
        self.assertEqual(original["prompt_policy"], "fixed_official_choice_order_lowercase_then_training_tokenize")

    def test_all_existing_variants_reuse_the_actual_loader(self):
        for variant in runtime.VARIANTS:
            selected, _, route = runtime.adapter(variant)
            is_official_recursive = variant in runtime.multilayer.ROUTE_SPECS and runtime.multilayer.variant_spec(variant).get("official")
            self.assertTrue((route / ("runtime.py" if is_official_recursive else "models/mellow.py")).is_file(), variant)
            args = SimpleNamespace(eval_variant=variant)
            with self.subTest(variant=variant), patch.object(selected, "load_runtime", return_value="loaded") as loader:
                self.assertEqual(runtime.load_runtime(args), "loaded")
                loader.assert_called_once_with(args)

    def test_route_context_is_passed_to_real_adapter_api(self):
        for variant in runtime.VARIANTS:
            _, family, _ = runtime.adapter(variant)
            context, prefix, audio = object(), object(), object()
            model = SimpleNamespace(generate_prefix_inference=Mock(return_value=(prefix, audio, context)))
            selected = SimpleNamespace(greedy_decode=Mock(return_value=generation()),
                generate_prefix_inference=Mock(return_value=(prefix, audio, context)),
                sample_context=Mock(return_value={"sample": context}))
            inputs = {"input": {"attention_mask": "sample mask"}}
            loaded = (model, "tokenizer", None, None, None, {"test": "audit"}, {})
            with self.subTest(variant=variant):
                actual, result = runtime.prefix_and_decode(selected, family, variant, loaded, inputs)
                self.assertIs(actual, prefix)
                self.assertEqual(result, generation())
                expected = ({"eval_variant": variant, "context": {"sample": context}} if family == "multilayer"
                            else {"eval_variant": variant, "audio_information": context} if family == "recursive" else {})
                selected.greedy_decode.assert_called_once_with(model, "tokenizer", prefix, **expected)
                if family == "multilayer":
                    selected.generate_prefix_inference.assert_called_once_with(model, inputs, eval_variant=variant, audit=loaded[5])
                    selected.sample_context.assert_called_once_with(variant, audio, context, inputs["input"])
                else:
                    model.generate_prefix_inference.assert_called_once_with(inputs)

    def test_explicit_target_arguments_no_historical_checkpoint_defaults(self):
        for variant in runtime.VARIANTS:
            args = (["--eval-variant", variant, "--checkpoint", "model.ckpt", "--output-dir", "out",
                     "--model-dir", "assets"] if variant == "official_mizar" else
                    ["--eval-variant", variant, "--run-root", "run", "--expected-step", "456",
                     "--expected-geometry", "8,4,4"])
            with self.subTest(variant=variant):
                parsed = evaluate.parse_args(args)
                if variant != "official_mizar":
                    self.assertTrue(str(parsed.checkpoint).endswith("model--step-456.ckpt"))
                    self.assertEqual(parsed.expected_geometry, (8, 4, 4))
                self.assertTrue(str(parsed.dataset_dir).endswith("MMSU data"))
        with contextlib.redirect_stderr(io.StringIO()):
            for args in ([], ["--eval-variant", "base", "--run-root", "run"],
                         ["--eval-variant", "base", "--run-root", "run", "--expected-step", "17313"]):
                with self.assertRaises(SystemExit):
                    evaluate.parse_args(args)

    def test_preserved_mmau_mmar_sources_with_explicit_selected4_extensions(self):
        audit = json.loads((ROOT / "code/RSmol/mizar_s1_ced_mmsu_eval/reference_audit.json").read_text())
        self.assertGreaterEqual(len(audit["files"]), 28)
        from mizar_selected4_eval.audit import effective_reference_files
        for path, hashes in effective_reference_files(ROOT).items():
            self.assertEqual(storage.text_sha(ROOT / path), hashes["lf_sha256"], path)

    def test_real_run_identity_binds_checkpoint_runtime_shards_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route = root / "route"
            route.mkdir()
            (route / "contract.py").write_text("# fixture route source\n")
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"checkpoint fixture")
            config = root / "runtime.json"
            config.write_text("{}")
            args = evaluate.parse_args(["--eval-variant", "base", "--checkpoint", str(checkpoint),
                "--runtime-config", str(config), "--route-root", str(route), "--output-dir", str(root / "out"),
                "--expected-step", "99", "--expected-geometry", "8,4,4", "--evaluation-script", str(SCORER)])
            paths = [root / name for name in dataset.SHARD_NAMES]
            for path in paths:
                path.write_bytes(b"shard fixture")
            original = evaluate.identity(args, paths)
            for path in (checkpoint, config, paths[1], route / "contract.py"):
                before = path.read_bytes()
                path.write_bytes(before + b" changed")
                self.assertNotEqual(evaluate.identity(args, paths)["identity_sha256"], original["identity_sha256"])
                path.write_bytes(before)
            self.assertEqual(evaluate.identity(args, paths), original)
            text_sha = storage.text_sha
            with patch.object(storage, "text_sha", side_effect=lambda path:
                    "changed evaluator" if Path(path).name == "evaluate.py" else text_sha(path)):
                self.assertNotEqual(evaluate.identity(args, paths)["identity_sha256"], original["identity_sha256"])


class StoreTests(unittest.TestCase):
    def test_durable_raw_recovery_partial_tail_and_missing_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows = [record(i) for i in range(3)]
            store = storage.GenerationStore(path, rows)
            store.append(0, rows[0], generation())
            prefix = store.path.read_bytes()
            store.progress.unlink()
            with store.path.open("ab") as stream:
                stream.write(b'{"row_index":1,"incomplete":')
            recovered = storage.GenerationStore(path, rows)
            self.assertEqual(len(recovered.state), 1)
            self.assertEqual(recovered.path.read_bytes(), prefix)
            recovered.append(1, rows[1], generation())
            self.assertEqual(len(storage.GenerationStore(path, rows).state), 2)

    def test_complete_corrupt_line_or_saved_content_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows = [record()]
            store = storage.GenerationStore(path, rows)
            store.append(0, rows[0], generation())
            valid = store.path.read_bytes()
            store.path.write_bytes(valid + b'invalid-json\n')
            with self.assertRaisesRegex(ValueError, "corrupt complete"):
                storage.GenerationStore(path, rows)
            store.path.write_bytes(valid)
            altered = json.loads(valid)
            altered["generation"]["generated_text"] = "B"
            store.path.write_text(json.dumps(altered) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                storage.GenerationStore(path, rows)

    def test_changed_identity_and_nonempty_unowned_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            storage.claim_output(path, {"checkpoint": "first"})
            storage.claim_output(path, {"checkpoint": "first"})
            with self.assertRaises(ValueError):
                storage.claim_output(path, {"checkpoint": "changed"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "unowned.txt").write_text("do not overwrite")
            with self.assertRaises(FileExistsError):
                storage.claim_output(path, {})

    def test_process_lock_rejects_second_owner_then_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with storage.output_lock(path):
                with self.assertRaisesRegex(RuntimeError, "in use"):
                    with storage.output_lock(path):
                        self.fail("concurrent output owner")
            with storage.output_lock(path):
                pass


class ScorerTests(unittest.TestCase):
    def test_unmodified_official_fixture_and_format_semantics(self):
        self.assertEqual(scoring.audit_scorer(SCORER)["sha256"], scoring.SCORER_LF_SHA256)
        import runpy
        function = runpy.run_path(str(SCORER))["calculate_accuracy_per_task_and_category"]
        cases = [("A", 1, 1), ("A<|endoftext|>", 1, 1), ("a", 0, 0),
                 ("The answer is A.", 1, 1), ("The answer is A.<|endoftext|>", 0, 0),
                 ("", 0, 1), ("None", 0, 1), ("B", 0, 1)]
        for text, correct, total in cases:
            with self.subTest(text=text), contextlib.redirect_stdout(io.StringIO()):
                _, groups, accuracy, count = function([{**record(), "response": text}])
                self.assertEqual(count, total)
                self.assertEqual(sum(g["total_correct"] for g in groups.values()), correct)
                self.assertEqual(accuracy, correct / total if total else 0)

    def test_scorer_change_rejected_but_crlf_transfer_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scorer.py"
            path.write_bytes(SCORER.read_bytes().replace(b"\n", b"\r\n"))
            scoring.audit_scorer(path)
            with path.open("ab") as stream:
                stream.write(b"# changed\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                scoring.audit_scorer(path)

    def test_cli_and_function_keep_actual_denominator_and_raw_response(self):
        for excluded in (0, 2):
            with self.subTest(excluded=excluded), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                saved = [{"official_record": record(i), "generation": generation("a" if i < excluded else "A<|endoftext|>")}
                         for i in range(5000)]
                predictions = path / "predictions.jsonl"
                scoring.write_predictions(predictions, saved)
                result = scoring.run_scorer(SCORER, predictions, path)
                self.assertEqual(result["execution_status"], "PASS")
                self.assertEqual(result["actual_denominator"], 5000 - excluded)
                self.assertEqual(result["excluded_rows"], excluded)
                self.assertEqual(result["overall_accuracy"], 1)
                self.assertEqual(result["status"], "FAILED" if excluded else "PASS")
                row = json.loads(predictions.read_text(encoding="utf-8").splitlines()[-1])
                self.assertIsNone(row["choice_b"])
                self.assertEqual(row["response"], "A<|endoftext|>")
                self.assertTrue((path / "official_evaluation.txt").is_file())

    def test_incomplete_predictions_or_nonzero_cli_return_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            predictions = path / "predictions.jsonl"
            scoring.write_predictions(predictions, [{"official_record": record(), "generation": generation()}])
            with self.assertRaisesRegex(ValueError, "5000 unique"):
                scoring.run_scorer(SCORER, predictions, path)
            saved = [{"official_record": record(i), "generation": generation()} for i in range(5000)]
            scoring.write_predictions(predictions, saved)
            failure = SimpleNamespace(returncode=2, stdout="", stderr="official scorer failed")
            with patch.object(scoring.subprocess, "run", return_value=failure):
                result = scoring.run_scorer(SCORER, predictions, path)
            self.assertEqual(result["execution_status"], "FAILED")
            self.assertEqual(result["status"], "FAILED")


class WorkflowTests(unittest.TestCase):
    def fixture(self, directory):
        path = Path(directory)
        args = evaluate.parse_args(["--eval-variant", "base", "--run-root", str(path / "run"),
                                   "--expected-step", "99", "--expected-geometry", "8,8,4",
                                   "--evaluation-script", str(SCORER)])
        rows = [record(i) for i in range(5000)]
        loaded = (None, None, None, None, None, {"status": "PASS", "fixture": True}, {})
        return args, rows, loaded

    def execute(self, args, rows, loaded, generate, *, run_identity=None, stream=None, preflight_error=None):
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            stack.enter_context(patch.object(dataset, "shard_paths", return_value=[]))
            stack.enter_context(patch.object(evaluate, "identity", return_value=run_identity or {"fixture": "first"}))
            stack.enter_context(patch.object(dataset, "preflight", side_effect=preflight_error,
                                            return_value=(rows, {"status": "PASS", "rows": len(rows)})))
            stack.enter_context(patch.object(dataset, "iter_rows", side_effect=lambda *a, **k:
                iter(stream) if stream is not None else enumerate({**row, "audio": {"bytes": b"fixture"}} for row in rows)))
            loader = stack.enter_context(patch.object(runtime, "load_runtime", return_value=loaded))
            decoder = stack.enter_context(patch.object(runtime, "decode_audio", return_value=(object(), {})))
            stack.enter_context(patch.object(runtime, "generate", side_effect=generate))
            # Keep file content/ordering real; fsync latency is unrelated to control-flow coverage.
            stack.enter_context(patch.object(storage.os, "fsync"))
            scorer = stack.enter_context(patch.object(scoring, "run_scorer", wraps=scoring.run_scorer))
            report = evaluate.run(args)
        return report, loader, decoder, scorer

    def test_full_5000_flow_then_rescore_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rows, loaded = self.fixture(directory)
            generate = Mock(side_effect=lambda *a, **k: generation())
            report, loader, decoder, scorer = self.execute(args, rows, loaded, generate)
            self.assertEqual(report["status"], "PASS", report["fatal_error"])
            self.assertEqual(generate.call_count, 5000)
            self.assertEqual(decoder.call_count, 5000)
            self.assertEqual(report["records"]["prompt_truncated_rows"], 5000)
            self.assertEqual(report["official_evaluation"]["actual_denominator"], 5000)
            before = (args.output_dir / "raw_generations.jsonl").read_bytes()
            report, loader, decoder, scorer = self.execute(args, rows, loaded, Mock(side_effect=AssertionError("repeat")))
            self.assertEqual(report["status"], "PASS", report["fatal_error"])
            loader.assert_not_called()
            decoder.assert_not_called()
            scorer.assert_called_once()
            self.assertEqual((args.output_dir / "raw_generations.jsonl").read_bytes(), before)

    def test_interruption_resume_only_unfinished_rows_and_no_scoring_partial_run(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rows, loaded = self.fixture(directory)
            report, _, _, scorer = self.execute(args, rows, loaded,
                Mock(side_effect=[generation(), generation(), RuntimeError("interrupted")]))
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["records"]["generated"], 2)
            scorer.assert_not_called()
            raw = args.output_dir / "raw_generations.jsonl"
            prior = raw.read_bytes()
            resumed = Mock(side_effect=lambda *a, **k: generation())
            report, _, decoder, scorer = self.execute(args, rows, loaded, resumed)
            self.assertEqual(report["status"], "PASS", report["fatal_error"])
            self.assertEqual(resumed.call_count, 4998)
            self.assertEqual(decoder.call_count, 4998)
            self.assertTrue(raw.read_bytes().startswith(prior))
            self.assertEqual(len(raw.read_text(encoding="utf-8").splitlines()), 5000)

    def test_changed_identity_does_not_overwrite_previous_report(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rows, loaded = self.fixture(directory)
            self.execute(args, rows, loaded, Mock(side_effect=RuntimeError("stop")))
            prior = (args.output_dir / "evaluation_report.json").read_bytes()
            report, loader, _, scorer = self.execute(args, rows, loaded, Mock(), run_identity={"fixture": "changed"})
            self.assertEqual(report["status"], "FAILED")
            self.assertFalse(report["output_claimed"])
            loader.assert_not_called()
            scorer.assert_not_called()
            self.assertEqual((args.output_dir / "evaluation_report.json").read_bytes(), prior)

    def test_preflight_and_stream_errors_never_score(self):
        for failure in ("metadata", "short_stream", "changed_row"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                args, rows, loaded = self.fixture(directory)
                stream = [(0, {**rows[0], "audio": {}})]
                if failure == "changed_row":
                    stream[0][1]["answer_gt"] = "changed"
                report, loader, _, scorer = self.execute(args, rows, loaded, Mock(return_value=generation()),
                    preflight_error=ValueError("bad metadata") if failure == "metadata" else None,
                    stream=stream)
                self.assertEqual(report["status"], "FAILED")
                scorer.assert_not_called()
                if failure == "metadata":
                    loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
