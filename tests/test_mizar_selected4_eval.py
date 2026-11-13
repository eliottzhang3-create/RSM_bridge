"""Four explicit final audio checkpoints across three existing benchmarks."""
from __future__ import annotations

import copy
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import test_mizar_s1_ced_mmsu_shell as shell
import test_mizar_s1_ced_recursive_eval as plain
import test_mizar_s1_ced_multilayer_eval as multi

ROOT, RSMOL = shell.ROOT, shell.RSMOL
sys.path.insert(0, str(RSMOL))
from mizar_s1_ced_mmsu_eval import evaluate as mmsu, runtime
from mizar_selected4_eval.audit import source_audit

LAUNCHER = "run_mizar_selected4_benchmarks_5090.sh"
OUTPUT_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
# Independent expectations, including the intentionally different run order.
TARGETS = {
    "init0904": ("recursive_5_10_5_init0904", "recursive",
        "mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_init0904",
        "mizar_s1_ced_small_gbs256_recursive_5_10_5_init0904_5090/formal_20261111_130037935014521"),
    "text_epoch1": ("recursive_5_10_5_text_epoch1", "recursive", "mizar_s1_ced_small_text_epoch1",
        "mizar_s1_ced_small_text_epoch1_gbs256_5090/formal_20261112_103104688616376"),
    "text_epoch1_serial48_lora_r8": (
        "recursive_5_10_5_text_epoch1_layer4_8_audio_reinject_loop_lora_r8", "multilayer",
        "mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8",
        "mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_gbs256_5090/formal_20261112_103717262469368"),
    "text_lora_epoch1_serial48_r8": ("recursive_5_10_5_ced48_text_epoch1_loop_lora_r8", "multilayer",
        "mizar_s1_ced48_text_epoch1_loop_lora_r8",
        "mizar_s1_ced48_text_epoch1_loop_lora_r8_5090/formal_20261112_111357380584730"),
}
BY_VARIANT = {value[0]: value for value in TARGETS.values()}


class RuntimeTests(unittest.TestCase):
    def select(self, variant):
        self.variant = variant
        self.runtime, family, _ = runtime.adapter(variant)
        self.runner = plain.runner if family == "recursive" else multi.runner
        self.spec = self.runtime.variant_spec(variant)
        self.route = plain.load_module("selected4_contract_" + variant, self.spec["route_root"] / "contract.py")

    def fixture(self):
        contract, saved, config = plain.BASE.CedBenchmarkEvaluationTest.fixture(self)
        contract["recursive_text_checkpoint"] = {"fixture_only": True}
        self.resign(saved, config)
        return contract, saved, config

    resign = plain.BASE.CedBenchmarkEvaluationTest.resign

    def validate(self, saved, config, variant=None):
        selected = self.runtime if variant is None else runtime.adapter(variant)[0]
        return selected.validate_checkpoint(saved, config, self.route, 17313, (8, 8, 4),
                                             eval_variant=variant or self.variant)

    def test_exact_training_identity_and_all_twelve_parser_defaults(self):
        outputs = set()
        for variant, family, directory, run in TARGETS.values():
            self.select(variant)
            contract, saved, config = self.fixture()
            self.assertIs(self.validate(saved, config), contract)
            self.assertEqual(self.spec["route"], self.route.ROUTE)
            self.assertEqual(self.spec["route_root"].name, directory)
            self.assertEqual(self.spec["run_root"].as_posix(), OUTPUT_ROOT + run)
            for benchmark in ("mmau", "mmar", "mmsu"):
                args = ["--eval-variant", variant]
                if benchmark == "mmsu":
                    parsed = mmsu.parse_args(args + ["--run-root", OUTPUT_ROOT + run,
                        "--expected-step", "17313", "--expected-geometry", "8,8,4"])
                else:
                    parsed = self.runner.parse_args(args + ["--benchmark", benchmark])
                self.assertEqual(parsed.checkpoint, Path(OUTPUT_ROOT + run + "/formal/checkpoints/model--step-17313.ckpt").resolve())
                self.assertEqual(parsed.runtime_config, Path(OUTPUT_ROOT + run + "/formal/runtime.json").resolve())
                self.assertEqual(parsed.expected_geometry, (8, 8, 4))
                self.assertEqual(parsed.expected_step, 17313)
                self.assertEqual(parsed.route_root, self.spec["route_root"].resolve())
                outputs.add(parsed.output_dir)
        self.assertEqual(len(outputs), 12)

    def test_cross_route_and_runtime_mismatches_fail_even_with_matching_shapes(self):
        for variant in BY_VARIANT:
            self.select(variant)
            _, saved, config = self.fixture()
            others = set(BY_VARIANT) - {variant}
            others |= {"recursive_5_10_5", "recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8"}
            for other in others:
                with self.subTest(saved=variant, selected=other), self.assertRaises(ValueError):
                    self.validate(saved, config, other)
            with self.assertRaises(ValueError):
                self.validate(saved, {**config, "route": "other"})
            for field in ("code_digest", "recursive_model_source_sha256"):
                with patch.object(self.route, field, return_value="changed"), self.assertRaises(ValueError):
                    self.validate(saved, config)

    def test_text_and_loop_lora_provenance_cannot_be_substituted(self):
        for variant, family, *_ in TARGETS.values():
            self.select(variant)
            for key, value in (("text_model_contract", "wrong"), ("text_model_architecture", "wrong"),
                               ("text_model_schedule", list(range(30)))):
                contract, saved, config = self.fixture()
                contract[key] = value
                self.resign(saved, config)
                with self.assertRaises(ValueError): self.validate(saved, config)
            if family != "multilayer":
                continue
            contract, saved, config = self.fixture()
            self.assertEqual(contract["layer_injection"]["ced_layers_1based"], [4, 8])
            for key in self.runtime.branch_contracts(self.route, self.spec):
                bad = copy.deepcopy(saved)
                bad["contract"][key]["incorrect"] = True
                current = copy.deepcopy(config)
                self.resign(bad, current)
                with self.assertRaises(ValueError): self.validate(bad, current)
            inherited = self.spec["inherited_lora_parameters"]
            self.assertEqual(inherited, 1628160 if "ced48" in variant else 0)
            if inherited:
                self.assertEqual(self.route.LOOP_LORA_CONTRACT["initialization"],
                                 "strict_pretrained_A_B_without_post_load_reset")
                self.assertEqual(self.route.TEXT_SOURCE_ROUTE, "mizar_text_5_10x2_5_loop_lora_r8_gbs1024_v1")

    def test_full_final_state_strict_load_audit_then_freeze_no_adapter_reset(self):
        for variant in BY_VARIANT:
            self.select(variant)
            for ddp in (False, True):
                with self.subTest(variant=variant, ddp=ddp), tempfile.TemporaryDirectory() as directory:
                    state = {"audio_encoder.base.mapper.weight": [2.5],
                             "caption_decoder.lm.model.layers.5.self_attn.q_proj.weight": [3.5]}
                    if self.spec.get("loop_lora"):
                        state.update({f"caption_decoder.lm.model.layers.5.self_attn.q_proj.lora_{ab}.{loop}":
                                      [7.5 + loop] for ab in ("A", "B") for loop in (0, 1)})
                    serialized = {"module." + k: v for k, v in state.items()} if ddp else state
                    args, model, modules, packages = plain.RecursiveCedEvaluationTests.loader_fixture(self, directory, serialized)
                    sequence = []
                    model.load_state_dict.side_effect = lambda *a, **k: sequence.append("final_state")
                    model.requires_grad_.side_effect = lambda *a: sequence.append("freeze")
                    def audit(*a, **k):
                        sequence.append("audit")
                        model.load_state_dict.assert_called_once_with(state, strict=True)
                        return {"status": "PASS"}
                    with (patch.dict(sys.modules, packages),
                          patch.object(self.runtime, "import_training_route", return_value=modules),
                          patch.object(self.runtime, "audit_assets", return_value={"status": "PASS"}),
                          patch.object(self.runtime, "audit_recursive_route", side_effect=audit),
                          patch.object(self.runtime, "install_recursive_trace_audit", return_value={})):
                        result = self.runtime.load_runtime(args)
                    self.assertEqual(sequence, ["final_state", "audit", "freeze"])
                    self.assertEqual(result[5]["state_dict_source"], "training_checkpoint_state_dict")
                    model.caption_decoder.lm.model.enable_loop_lora.assert_not_called()
                    model.caption_decoder.lm.resize_token_embeddings.assert_not_called()

    def test_real_trace_hooks_enforce_serial48_order_and_both_loop_adapters(self):
        variants = tuple(v for v, family, *_ in TARGETS.values() if family == "multilayer")
        with patch.object(multi, "VARIANTS", variants):
            multi.MultilayerEvaluationTests.test_trace_hooks_check_physical_calls_branches_routers_and_query32(self)

    def test_plain_routes_keep_existing_greedy_and_raw_eos(self):
        variants = tuple(v for v, family, *_ in TARGETS.values() if family == "recursive")
        with patch.object(plain, "VARIANTS", variants):
            self.decode_fixture = plain.RecursiveCedEvaluationTests.decode_fixture.__get__(self)
            self.decode = plain.RecursiveCedEvaluationTests.decode.__get__(self)
            plain.RecursiveCedEvaluationTests.test_greedy_eos_recompute_raw_text_and_feature_order_for_all_variants(self)

    def test_actual_serial_and_config_validators_report_fresh_vs_inherited_parameters(self):
        for variant, family, *_ in TARGETS.values():
            if family != "multilayer":
                continue
            self.select(variant)
            spec = self.spec
            self.spec = {**spec, "layer_order": (4, 8), "reinjection": True}
            model, contract, modules = plain.RecursiveCedEvaluationTests.structure_fixture(self)
            self.spec = spec
            lm, core = model.caption_decoder.lm, model.caption_decoder.lm.model
            model.caption_decoder.text_model_contract = spec["text_contract"]
            core.gradient_checkpointing = False
            core.audio_linear.parameters = lambda: [types.SimpleNamespace(numel=lambda: 332352)]
            core.audio_gate.parameters = lambda: [types.SimpleNamespace(numel=lambda: 577)]
            core.named_parameters = lambda: [("layers.5.self_attn.q_proj.lora_A.0",
                types.SimpleNamespace(numel=lambda: 1628160))]
            if spec["inherited_lora_parameters"]:
                config = lm.config.to_dict()
                config.update(architectures=["MizarTextLoopLoRAR8ForCausalLM"], vocab_size=49152,
                    tie_word_embeddings=True, text_route=self.route.TEXT_SOURCE_ROUTE,
                    loop_lora_contract=copy.deepcopy(self.route.TEXT_LORA_CONTRACT))
            lora = types.SimpleNamespace(__file__=str(spec["route_root"] / "models/loop_lora.py"),
                TARGETS=multi.adapter.LOOP_LORA_TARGETS,
                validate_loop_lora=Mock(return_value=dict(spec["loop_lora_config"])))
            modules["models.loop_lora"] = lora
            with (patch.dict(sys.modules, modules), patch.object(self.route, "recursive_checkpoint_contract",
                    return_value=contract["recursive_text_checkpoint"])):
                result = multi.adapter.audit_recursive_route(self.route, contract, model, eval_variant=variant)
            inherited = spec["inherited_lora_parameters"]
            self.assertEqual(result["branch_parameter_count"], 4406019)
            self.assertEqual(result["fresh_audio_parameter_count"], 2777859)
            self.assertEqual(result["new_parameter_count"], 4406019 - inherited)
            self.assertEqual(result["inherited_lora_parameter_count"], inherited)
            self.assertEqual(result["fresh_lora_parameter_count"], 1628160 - inherited)
            self.assertEqual(result["evaluated_weights"], "complete_final_audio_checkpoint_state_dict")

    def test_historical_baseline_training_and_generation_functions_preserved(self):
        result = source_audit(ROOT)
        self.assertEqual(result["status"], "PASS", result)
        self.assertGreaterEqual(result["preserved_training_files"], 105)


class SubmissionTests(unittest.TestCase):
    setUp = shell.ShellTests.setUp

    def test_submission_shell_syntax(self):
        for name in (LAUNCHER, "run_mizar_s1_ced_small_mmau_eval_5090.sh", "run_mizar_s1_ced_small_mmar_eval_5090.sh"):
            result = subprocess.run([self.bash, "-n", str(RSMOL / name)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def fixture(self, directory):
        path, rs, env = shell.ShellTests.fixture(self, directory)
        for name in (LAUNCHER, "run_mizar_s1_ced_small_mmau_eval_5090.sh", "run_mizar_s1_ced_small_mmar_eval_5090.sh"):
            shutil.copy2(RSMOL / name, rs / name)
        startup = Path(env["BASH_ENV"])
        startup.write_text('vc() { printf "%s\\0" "$@" >> "$MMSU_VC_CAPTURE"; '
            '[[ "$*" != *"${SELECTED4_FAIL_PATTERN:-NEVER_MATCH}"* ]]; }\nexport -f vc\n',
            encoding="utf-8", newline="\n")
        return path, rs, env

    def launch(self, rs, env, args):
        return subprocess.run([self.bash, str(rs / LAUNCHER), *args], cwd=rs, env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)

    def submissions(self, path):
        values = (path / "submission.bin").read_bytes().decode().split("\0")[:-1]
        indices = [i for i, value in enumerate(values) if value == "submit"] + [len(values)]
        return [values[a:b] for a, b in zip(indices, indices[1:])]

    def assert_binding(self, submission, target, benchmark):
        for flag, value in (("-p", "pdgpu-5090"), ("-c", "8"), ("-m", "32G"), ("-g", "1"), ("-n", "1"),
                            ("-i", "docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1")):
            self.assertEqual(submission[submission.index(flag) + 1], value)
        self.assertLessEqual(len(submission[submission.index("-j") + 1]), 60)
        command = shlex.split(submission[submission.index("--cmd") + 1])
        variant, family, directory, run = TARGETS[target]
        options = dict(zip(command[2::2], command[3::2]))
        self.assertEqual(options["--eval-variant"], variant)
        self.assertEqual(options["--checkpoint"], OUTPUT_ROOT + run + "/formal/checkpoints/model--step-17313.ckpt")
        self.assertEqual(options["--runtime-config"], OUTPUT_ROOT + run + "/formal/runtime.json")
        self.assertTrue(options["--route-root"].endswith("/" + directory))
        self.assertEqual(options["--expected-step"], "17313")
        self.assertEqual(options["--expected-geometry"], "8,8,4")
        folder = "mmau_test_mini" if benchmark == "mmau" else benchmark
        self.assertEqual(options["--output-dir"], OUTPUT_ROOT + run + "/eval/" + folder + "_step17313")
        selected = mmsu if benchmark == "mmsu" else plain.runner if family == "recursive" else multi.runner
        parsed = selected.parse_args(command[2:])
        self.assertEqual(parsed.eval_variant, variant)
        self.assertEqual(parsed.expected_geometry, (8, 8, 4))
        if benchmark != "mmsu":
            self.assertEqual(command[1], f"scripts/evaluate_mizar_s1_ced_{family}_benchmark.sh")
            self.assertEqual(parsed.benchmark, benchmark)
        return command, parsed

    def test_all_twelve_real_submissions_and_parsers_with_stale_model_environment(self):
        for target in TARGETS:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                path, rs, env = self.fixture(directory)
                for field in ("EVAL_VARIANT", "RUN_ROOT", "ROUTE_ROOT", "CHECKPOINT", "RUNTIME_CONFIG",
                              "EXPECTED_STEP", "EXPECTED_GEOMETRY", "MMAU_OUTPUT_DIR", "MMAR_OUTPUT_DIR"):
                    env["RSMOL_MIZAR_CED_" + field] = "WRONG_STALE_VALUE"
                result = self.launch(rs, env, [target, "all"])
                self.assertEqual(result.returncode, 0, result.stderr)
                submissions = self.submissions(path)
                self.assertEqual(len(submissions), 3)
                for submission, benchmark in zip(submissions, ("mmau", "mmar", "mmsu")):
                    self.assert_binding(submission, target, benchmark)

    def test_check_args_never_submits_and_shows_three_fixed_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            for target in TARGETS:
                result = self.launch(rs, env, [target, "all", "--check-args"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(result.stdout.splitlines()), 3)
                self.assertIn(TARGETS[target][3], result.stdout)
                self.assertFalse((path / "submission.bin").exists())

    def test_all_attempts_remaining_benchmarks_after_submission_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            env["SELECTED4_FAIL_PATTERN"] = "evaluate_mizar_s1_ced_recursive_benchmark.sh --benchmark mmau"
            result = self.launch(rs, env, ["init0904", "all"])
            self.assertEqual(result.returncode, 1)
            submissions = self.submissions(path)
            self.assertEqual(len(submissions), 3)
            self.assertIn("continuing other benchmarks", result.stderr)

    def test_single_benchmark_io_paths_with_spaces_and_quotes(self):
        for benchmark in ("mmau", "mmar", "mmsu"):
            with tempfile.TemporaryDirectory() as directory:
                path, rs, env = self.fixture(directory)
                dataset, output = "/dataset's spaced dir", "/output's spaced dir"
                result = self.launch(rs, env, ["text_lora_epoch1_serial48_r8", benchmark,
                    "--dataset-dir", dataset, "--output-dir=" + output, "--parquet-batch-size", "3"])
                self.assertEqual(result.returncode, 0, result.stderr)
                command = shlex.split(self.submissions(path)[0][-1])
                selected = mmsu if benchmark == "mmsu" else multi.runner
                parsed = selected.parse_args(command[2:])
                self.assertEqual(parsed.dataset_dir, Path(dataset).resolve())
                self.assertEqual(parsed.output_dir, Path(output).resolve())
                self.assertEqual(parsed.parquet_batch_size, 3)

    def test_invalid_options_fail_before_any_submission(self):
        bad = [[], ["unknown", "all"], ["init0904", "unknown"], ["init0904", "all", "--output-dir", "/wrong"],
               ["init0904", "mmau", "--checkpoint", "/wrong"], ["text_epoch1", "mmsu", "--expected-step=1"],
               ["text_epoch1", "mmar", "--mmsu-root", "/wrong"], ["init0904", "mmau", "--dataset-dir"],
               ["init0904", "mmsu", "--parquet-batch-size=0"], ["init0904", "all", "--check-args=true"]]
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            for args in bad:
                result = self.launch(rs, env, args)
                self.assertEqual(result.returncode, 2, (args, result.stderr))
                self.assertFalse((path / "submission.bin").exists())


if __name__ == "__main__":
    unittest.main()
