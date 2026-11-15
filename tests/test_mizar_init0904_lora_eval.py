"""Final init0904 audio state, serial8to4 execution and three real submissions."""
from __future__ import annotations

import copy
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import test_mizar_selected4_eval as helpers
import test_mizar_s1_ced_mmsu_eval as mmsu_tests
import test_mizar_s1_ced_multilayer_eval_workflow as workflow
from mizar_selected4_eval.audit import before_init0904_extension, init0904_reference, source_audit

ROOT, RSMOL = helpers.ROOT, helpers.RSMOL
VARIANT = "recursive_5_10_5_init0904_layer8_4_audio_reinject_loop_lora_r8"
PARENT = "recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8"
DIRECTORY = "mizar_s1_ced_small_init0904_layer8_4_audio_reinject_loop_lora_r8"
RUN = helpers.OUTPUT_ROOT + DIRECTORY + "_gbs256_5090/formal_20261113_221451337688842"
adapter, runner, mmsu = helpers.multi.adapter, helpers.multi.runner, helpers.mmsu


class RuntimeTests(unittest.TestCase):
    select = helpers.RuntimeTests.select
    fixture = helpers.RuntimeTests.fixture
    resign = helpers.RuntimeTests.resign
    validate = helpers.RuntimeTests.validate

    def setUp(self):
        self.select(VARIANT)

    def test_exact_completed_contract_and_three_benchmark_paths(self):
        contract, saved, config = self.fixture()
        self.assertIs(self.validate(saved, config), contract)
        self.assertEqual(self.route.ROUTE, "ced_small_mizar_s1_fixed20_gbs256_init0904_layer8_4_audio_reinject_loop_lora_r8_v1")
        self.assertEqual(self.spec["route_root"].name, DIRECTORY)
        self.assertEqual(self.spec["run_root"].as_posix(), RUN)
        self.assertEqual(self.route.DEFAULT_TEXT_MODEL_DIR, helpers.OUTPUT_ROOT +
            "stage4_5_10_5/formal-maxlr8e-4-resume4500-20260904_154711/checkpoint-step-009244")
        self.assertEqual(self.spec["interaction_layers"], (8, 4))
        self.assertEqual(self.spec["new_parameters"], 4406019)
        for benchmark in ("mmau", "mmar", "mmsu"):
            args = ["--eval-variant", VARIANT]
            if benchmark == "mmsu":
                parsed = mmsu.parse_args(args + ["--run-root", RUN, "--expected-step", "17313",
                                               "--expected-geometry", "8,8,4"])
            else:
                parsed = runner.parse_args(args + ["--benchmark", benchmark])
            folder = "mmau_test_mini" if benchmark == "mmau" else benchmark
            self.assertEqual(parsed.checkpoint, Path(RUN + "/formal/checkpoints/model--step-17313.ckpt").resolve())
            self.assertEqual(parsed.runtime_config, Path(RUN + "/formal/runtime.json").resolve())
            self.assertEqual(parsed.output_dir, Path(RUN + f"/eval/{folder}_step17313").resolve())
            self.assertEqual(parsed.route_root, self.spec["route_root"].resolve())
            self.assertEqual(parsed.expected_geometry, (8, 8, 4))

    def test_wrong_route_source_step_geometry_and_trained_branch_contracts_rejected(self):
        _, saved, config = self.fixture()
        for variant in (PARENT, "recursive_5_10_5_init0904", "recursive_5_10_5_text_epoch1_layer4_8_audio_reinject_loop_lora_r8"):
            with self.subTest(other=variant), self.assertRaises(ValueError):
                self.validate(saved, config, variant)
        for field, value in (("total_step", 100), ("contract_sha256", "wrong")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate({**saved, field: value}, config)
        with self.assertRaises(ValueError):
            adapter.validate_checkpoint(saved, config, self.route, 17313, (8, 4, 4), eval_variant=VARIANT)
        for field in ("code_digest", "recursive_model_source_sha256"):
            with patch.object(self.route, field, return_value="wrong"), self.assertRaises(ValueError):
                self.validate(saved, config)
        for field in adapter.branch_contracts(self.route, self.spec):
            bad, current = copy.deepcopy(saved), copy.deepcopy(config)
            bad["contract"][field]["incorrect"] = True
            self.resign(bad, current)
            with self.subTest(branch=field), self.assertRaises(ValueError):
                self.validate(bad, current)

    def test_strict_full_audio_state_loaded_before_audit_and_freeze(self):
        with patch.object(helpers, "BY_VARIANT", {VARIANT: None}):
            helpers.RuntimeTests.test_full_final_state_strict_load_audit_then_freeze_no_adapter_reset(self)

    def test_execution_hooks_require_input_then_ced8_input_then_ced4_and_both_loras(self):
        with patch.object(helpers.multi, "VARIANTS", (VARIANT,)):
            helpers.multi.MultilayerEvaluationTests.test_trace_hooks_check_physical_calls_branches_routers_and_query32(self)

    def test_mmsu_keeps_each_samples_pair_without_swapping_ced4_ced8(self):
        self.assertIs(helpers.runtime.adapter(VARIANT)[0], adapter)
        model = Mock()
        pairs = [(object(), object()), (object(), object())]
        prefixes = [object(), object()]
        model.generate_prefix_inference.side_effect = [(p, {}, a) for p, a in zip(prefixes, pairs)]
        loaded = (model, object(), None, None, None, {}, None)
        with patch.object(adapter, "greedy_decode", return_value={"generated_text": " raw EOS "}) as decode:
            for i in range(2):
                prefix, result = helpers.runtime.prefix_and_decode(adapter, "multilayer", VARIANT, loaded, {"input": {}})
                self.assertIs(prefix, prefixes[i])
                self.assertIs(decode.call_args.kwargs["context"]["audio_information"], pairs[i])
                self.assertEqual(result["generated_text"], " raw EOS ")
        self.assertEqual(model.generate_prefix_inference.call_count, 2)

    def test_existing_model_forward_and_data_sources_are_identical_to_supported_parent(self):
        parent = adapter.variant_spec(PARENT)["route_root"]
        for relative in ("models/ced.py", "models/decoder.py", "models/mellow.py", "models/recursive_text.py",
                         "models/recursive_layer_inject.py", "models/loop_lora.py", "data/fixed20.py",
                         "data/audiotext_dataset.py", "train.py", "optim.py", "write_config.py"):
            with self.subTest(source=relative):
                self.assertEqual((self.spec["route_root"] / relative).read_bytes().replace(b"\r\n", b"\n"),
                                 (parent / relative).read_bytes().replace(b"\r\n", b"\n"))

    def test_source_audit_preserves_historical_evaluator_and_training_bytes(self):
        audit = source_audit(ROOT)
        self.assertEqual(audit["status"], "PASS", audit)
        for path, record in init0904_reference(ROOT)["authorized_evaluator_extensions"].items():
            source = (ROOT / path).read_text(encoding="utf-8")
            before_init0904_extension(ROOT, path, source)
            with self.assertRaises(ValueError):
                before_init0904_extension(ROOT, path, source + "# unauthorized change\n")
            with self.assertRaises(ValueError):
                before_init0904_extension(ROOT, path, source.replace(record["inserted_block"], ""))


class DecodeTests(helpers.multi.NumpyDecodeTests):
    def test_eos_full_recomputation_explicit_context_and_raw_text(self):
        with patch.object(helpers.multi, "VARIANTS", (VARIANT,)):
            super().test_eos_full_recomputation_explicit_context_and_raw_text()

    def test_both_benchmarks_keep_300_token_limit_and_finite_context(self):
        with patch.object(helpers.multi, "VARIANTS", (VARIANT,)):
            super().test_both_benchmarks_keep_300_token_limit_and_finite_context()


class WorkflowTests(unittest.TestCase):
    fixture = mmsu_tests.WorkflowTests.fixture
    execute = mmsu_tests.WorkflowTests.execute

    def test_1000_rows_of_both_benchmarks_keep_original_order_and_raw_outputs(self):
        with patch.dict(adapter.ROUTE_SPECS, {VARIANT: adapter.variant_spec(VARIANT)}, clear=True):
            workflow.WorkflowTests().test_full_1000_rows_for_both_benchmarks_preserve_raw_predictions()

    def test_5000_mmsu_rows_v2_scoring_and_resume_keep_raw_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rows, loaded = self.fixture(directory)
            args.eval_variant, args.expected_step = VARIANT, 17313
            args.route_root = adapter.variant_spec(VARIANT)["route_root"]
            raw = "  b. india <|endoftext|>  "
            generation = lambda *_a, **_k: mmsu_tests.generation(raw)
            report, _, _, scorer = self.execute(args, rows, loaded,
                Mock(side_effect=[generation(), generation(), RuntimeError("interrupted")]))
            self.assertEqual(report["records"]["generated"], 2)
            scorer.assert_not_called()
            generate = Mock(side_effect=generation)
            report, _, _, _ = self.execute(args, rows, loaded, generate)
            self.assertEqual(report["status"], "PASS", report.get("fatal_error"))
            self.assertEqual(generate.call_count, 4998)
            self.assertEqual(report["response_parsing"]["matched_rows"], 5000)
            self.assertEqual(report["official_evaluation"]["actual_denominator"], 5000)
            predictions = [json.loads(line) for line in (args.output_dir / "predictions_official.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["response"] == raw for row in predictions))


class SubmissionTests(unittest.TestCase):
    setUp = helpers.shell.ShellTests.setUp

    def test_three_real_bash_submissions_bind_checkpoint_runtime_geometry_and_resources(self):
        for benchmark in ("mmau", "mmar", "mmsu"):
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                path, rs, env = helpers.shell.ShellTests.fixture(self, directory)
                name = f"run_mizar_s1_ced_small_{benchmark}_eval_5090.sh"
                shutil.copy2(RSMOL / name, rs / name)
                # Other checkpoints in a run cannot participate in exact path selection.
                env = {k: v for k, v in env.items() if not k.startswith("RSMOL_MIZAR_CED_")}
                env["RSMOL_MIZAR_CED_EVAL_VARIANT"] = VARIANT
                args = (["--eval-variant", VARIANT, "--run-root", RUN,
                         "--expected-step", "17313", "--expected-geometry", "8,8,4"]
                        if benchmark == "mmsu" else [])
                result = subprocess.run([self.bash, str(rs / name), *args], cwd=rs, env=env,
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)
                self.assertEqual(result.returncode, 0, result.stderr)
                submission = (path / "submission.bin").read_bytes().decode().split("\0")[:-1]
                for flag, value in (("-p", "pdgpu-5090"), ("-c", "8"), ("-m", "32G"), ("-g", "1"), ("-n", "1")):
                    self.assertEqual(submission[submission.index(flag) + 1], value)
                command = shlex.split(submission[submission.index("--cmd") + 1])
                parser = mmsu if benchmark == "mmsu" else runner
                parsed = parser.parse_args(command[2:])
                self.assertEqual(parsed.eval_variant, VARIANT)
                self.assertEqual(parsed.checkpoint, Path(RUN + "/formal/checkpoints/model--step-17313.ckpt").resolve())
                self.assertEqual(parsed.runtime_config, Path(RUN + "/formal/runtime.json").resolve())
                self.assertEqual(parsed.expected_step, 17313)
                self.assertEqual(parsed.expected_geometry, (8, 8, 4))
                self.assertEqual(parsed.route_root.name, DIRECTORY)
                self.assertEqual(command[1], "mizar_s1_ced_mmsu_eval/run.sh" if benchmark == "mmsu" else
                                 "scripts/evaluate_mizar_s1_ced_multilayer_benchmark.sh")


if __name__ == "__main__":
    unittest.main()
