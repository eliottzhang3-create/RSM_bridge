"""Exercise all 24 submissions with fake vc and the real compute wrapper."""
from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ced_shell_helpers", ROOT / "tests/test_mizar_s1_ced_recursive_mesh_eval_shell.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
SPECS = {
    "recursive_5_10_5_layer36912_crossattn": ("recursive_5_10_5", "layer36912_crossattn", "20261030_140459065784533", "ca4"),
    "recursive_5_10_5_layer36912_query32_crossattn": ("recursive_5_10_5", "layer36912_query32_crossattn", "20261030_162028139130855", "q32"),
    "recursive_mesh_layer4_8_inject": ("recursive_mesh", "layer4_8_inject", None, "m48i"),
    "recursive_mesh_layer4_8_listen": ("recursive_mesh", "layer4_8_listen", "20261030_172450546025072", "m48l"),
    "recursive_mesh_layer36912_listen": ("recursive_mesh", "layer36912_listen", "20261101_113133327944502", "m4l"),
    "recursive_mesh_layer4_8_audio7slot_split3": ("recursive_mesh", "layer4_8_audio7slot_split3", "20261101_170211458358358", "m7s3"),
    "recursive_mesh_layer4_8_audio7slot_split3_inject": ("recursive_mesh", "layer4_8_audio7slot_split3_inject", "20261102_002211827036990", "m7s3i"),
    "recursive_mesh_layer8_4_inject": ("recursive_mesh", "layer8_4_inject", "20261102_112040402999384", "m84i"),
    "recursive_mesh_layer4_8_audio7slot_split3_prompt_mix": ("recursive_mesh", "layer4_8_audio7slot_split3_prompt_mix", "20261102_132232677352730", "m7pm"),
    "recursive_5_10_5_layer8_4_inject_audio_reinject": ("recursive_5_10_5", "layer8_4_inject_audio_reinject", "20261102_112121702891801", "r84ri"),
    "recursive_5_10_5_layer8_4_inject_audio_reinject_loop_lora_r8": ("recursive_5_10_5", "layer8_4_inject_audio_reinject_loop_lora_r8", "20261103_165410438279511", "r84l8"),
    "recursive_mesh_5_10x4_5": ("recursive_mesh_5_10x4_5", "", "20261103_093224843011222", "mx4"),
}


def expected_step(variant):
    return 34626 if variant == "recursive_mesh_5_10x4_5" else 17313


class MultilayerCedShellTests(unittest.TestCase):
    run_launcher = helpers.MeshCedEvaluationShellTests.run_launcher
    assert_resources = helpers.MeshCedEvaluationShellTests.assert_resources

    def setUp(self):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        self.bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
        if not self.bash: self.skipTest("Bash is unavailable")

    def test_twenty_two_submissions_bind_variant_checkpoint_runtime_route_and_resources(self):
        for variant, (family, suffix, stamp, tag) in SPECS.items():
            for benchmark in ("mmau", "mmar"):
                with self.subTest(variant=variant, benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                    overrides = {} if stamp else {"RSMOL_MIZAR_CED_RUN_ROOT": "/actual completed run"}
                    result, submission = self.run_launcher(directory, benchmark, variant, overrides)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assert_resources(submission)
                    command = shlex.split(submission[submission.index("--cmd") + 1])
                    self.assertEqual(command[:2], ["bash", "scripts/evaluate_mizar_s1_ced_multilayer_benchmark.sh"])
                    options = dict(zip(command[2::2], command[3::2]))
                    gbs = 128 if variant == "recursive_mesh_5_10x4_5" else 256
                    step = expected_step(variant)
                    run = (f"/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs{gbs}_{variant}_5090/formal_{stamp}"
                           if stamp else "/actual completed run")
                    self.assertEqual(options["--eval-variant"], variant)
                    self.assertEqual(options["--checkpoint"], run + f"/formal/checkpoints/model--step-{step}.ckpt")
                    self.assertEqual(options["--runtime-config"], run + "/formal/runtime.json")
                    route = (f"mizar_stage1_fixed20_gbs{gbs}_3epochs_{family}_ced_small" + (f"_{suffix}" if suffix else ""))
                    self.assertTrue(options["--route-root"].endswith("/" + route))
                    folder = f"mmau_test_mini_step{step}" if benchmark == "mmau" else f"mmar_step{step}"
                    self.assertEqual(options["--output-dir"], run + "/eval/" + folder)
                    self.assertEqual(options["--expected-step"], str(step))
                    self.assertEqual(options["--expected-geometry"], "8,4,4" if gbs == 128 else "8,8,4")
                    self.assertEqual(options["--benchmark"], benchmark)
                    self.assertTrue(submission[submission.index("-j") + 1].startswith(f"mizar-ced-multi-{tag}-{benchmark}-"))
                    if benchmark == "mmau": self.assertEqual(options["--parquet-batch-size"], "8")

    def test_path_overrides_and_spaces_survive_for_each_variant(self):
        for variant in SPECS:
            for benchmark in ("mmau", "mmar"):
                with tempfile.TemporaryDirectory() as directory:
                    overrides = {"RSMOL_MIZAR_CED_RUN_ROOT": "/custom run's root",
                        "RSMOL_MIZAR_CED_CHECKPOINT": "/custom weights/model's step.ckpt",
                        "RSMOL_MIZAR_CED_ROUTE_ROOT": "/custom route",
                        "RSMOL_MIZAR_CED_RUNTIME_CONFIG": "/custom runtime/config.json",
                        f"RSMOL_MIZAR_CED_{benchmark.upper()}_OUTPUT_DIR": "/custom output",
                        f"RSMOL_{benchmark.upper()}_DATASET_DIR": "/custom dataset"}
                    result, submission = self.run_launcher(directory, benchmark, variant, overrides,
                        ("--metadata-json", "/metadata path/records.json"))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    command = shlex.split(submission[submission.index("--cmd") + 1])
                    options = dict(zip(command[2::2], command[3::2]))
                    for option, variable in (("--checkpoint", "RSMOL_MIZAR_CED_CHECKPOINT"),
                        ("--runtime-config", "RSMOL_MIZAR_CED_RUNTIME_CONFIG"), ("--route-root", "RSMOL_MIZAR_CED_ROUTE_ROOT"),
                        ("--output-dir", f"RSMOL_MIZAR_CED_{benchmark.upper()}_OUTPUT_DIR"),
                        ("--dataset-dir", f"RSMOL_{benchmark.upper()}_DATASET_DIR")):
                        self.assertEqual(options[option], overrides[variable])
                    self.assertEqual(options["--metadata-json"], "/metadata path/records.json")

    def test_run_root_override_derives_all_three_paths(self):
        for variant in SPECS:
            with tempfile.TemporaryDirectory() as directory:
                result, submission = self.run_launcher(directory, "mmar", variant,
                    {"RSMOL_MIZAR_CED_RUN_ROOT": "/another run"})
                self.assertEqual(result.returncode, 0, result.stderr)
                command = shlex.split(submission[submission.index("--cmd") + 1])
                options = dict(zip(command[2::2], command[3::2]))
                self.assertEqual(options["--checkpoint"], f"/another run/formal/checkpoints/model--step-{expected_step(variant)}.ckpt")
                self.assertEqual(options["--runtime-config"], "/another run/formal/runtime.json")
                self.assertEqual(options["--output-dir"], f"/another run/eval/mmar_step{expected_step(variant)}")

    def test_unknown_variant_never_calls_vc(self):
        for benchmark in ("mmau", "mmar"):
            with tempfile.TemporaryDirectory() as directory:
                result, submission = self.run_launcher(directory, benchmark, "recursive_5_10_5_typo")
                self.assertEqual(result.returncode, 2)
                self.assertIsNone(submission)

    def test_pending_routes_reject_missing_run_before_submitting_and_accept_cli_paths(self):
        for variant in (name for name, value in SPECS.items() if value[2] is None):
            for benchmark in ("mmau", "mmar"):
                with tempfile.TemporaryDirectory() as directory:
                    result, submission = self.run_launcher(directory, benchmark, variant)
                    self.assertEqual(result.returncode, 2)
                    self.assertIsNone(submission)
                    self.assertIn("No completed default run", result.stderr)
                for extra in (("--run-root", "/actual new run"),
                        ("--checkpoint", "/actual new run/formal/checkpoints/model--step-17313.ckpt"),
                        ("--checkpoint", "/independent weights.ckpt", "--runtime-config", "/independent runtime.json",
                         "--output-dir", "/independent output")):
                    with tempfile.TemporaryDirectory() as directory:
                        result, submission = self.run_launcher(directory, benchmark, variant, extra=extra)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assert_resources(submission)
                        command = shlex.split(submission[submission.index("--cmd") + 1])
                        options = dict(zip(command[2::2], command[3::2]))
                        self.assertNotIn("/formal/checkpoints/model--step-17313.ckpt", [options["--checkpoint"]])
                        if "--runtime-config" not in extra:
                            self.assertEqual(options["--runtime-config"], "/actual new run/formal/runtime.json")

    def test_compute_wrapper_environment_arguments_and_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts, binary = root / "scripts", root / "bin"
            scripts.mkdir(); binary.mkdir()
            source = helpers.RSMOL / "scripts/evaluate_mizar_s1_ced_multilayer_benchmark.sh"
            wrapper = scripts / source.name
            wrapper.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
            conda = root / "conda base/etc/profile.d"
            conda.mkdir(parents=True)
            (conda / "conda.sh").write_text('conda() { [[ "$1" == activate && "$2" == mellow_c8204d8 ]]; }\n', encoding="utf-8", newline="\n")
            executable = binary / "python"
            executable.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$RSM_TEST_CAPTURE"\n'
                'printf \'%s\\n\' "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE" "$TOKENIZERS_PARALLELISM" "$PWD" > "$RSM_TEST_ENV"\n'
                'exit "$RSM_TEST_EXIT"\n', encoding="utf-8", newline="\n")
            executable.chmod(0o755)
            capture, captured_env = root / "args", root / "env"
            env = {**os.environ, "RSMOL_CONDA_BASE": helpers.bash_path(root / "conda base"),
                "RSM_TEST_CAPTURE": helpers.bash_path(capture), "RSM_TEST_ENV": helpers.bash_path(captured_env),
                "PATH": f"{helpers.bash_path(binary)}:/usr/bin:/bin"}
            for exit_code in (0, 7):
                env["RSM_TEST_EXIT"] = str(exit_code)
                result = subprocess.run([self.bash, helpers.bash_path(wrapper), "--eval-variant", "recursive_5_10_5_layer36912_crossattn"],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, exit_code, result.stderr)
                self.assertEqual(capture.read_text(encoding="utf-8").splitlines(),
                    ["-u", "scripts/evaluate_mizar_s1_ced_multilayer_benchmark.py", "--eval-variant", "recursive_5_10_5_layer36912_crossattn"])
                self.assertEqual(captured_env.read_text(encoding="utf-8").splitlines(), ["1", "1", "false", helpers.bash_path(root)])

    def test_checkpoint_override_infers_matching_run_for_all_variants(self):
        for variant in SPECS:
            step = expected_step(variant)
            checkpoint = f"/replacement run/formal/checkpoints/model--step-{step}.ckpt"
            for benchmark in ("mmau", "mmar"):
                for source in ("cli", "env"):
                    with self.subTest(variant=variant, benchmark=benchmark, source=source), tempfile.TemporaryDirectory() as directory:
                        env = {"RSMOL_MIZAR_CED_CHECKPOINT": checkpoint} if source == "env" else {}
                        extra = ("--checkpoint", checkpoint) if source == "cli" else ()
                        result, submission = self.run_launcher(directory, benchmark, variant, env, extra)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        command = shlex.split(submission[submission.index("--cmd") + 1])
                        options = dict(zip(command[2::2], command[3::2]))
                        self.assertEqual(options["--checkpoint"], checkpoint)
                        self.assertEqual(options["--runtime-config"], "/replacement run/formal/runtime.json")
                        folder = f"mmau_test_mini_step{step}" if benchmark == "mmau" else f"mmar_step{step}"
                        self.assertEqual(options["--output-dir"], "/replacement run/eval/" + folder)

    def test_cli_equals_and_missing_path_values_before_submission(self):
        for benchmark in ("mmau", "mmar"):
            for extra in (("--run-root=/equals run",),
                          ("--checkpoint=/equals run/formal/checkpoints/model--step-17313.ckpt",)):
                with tempfile.TemporaryDirectory() as directory:
                    result, submission = self.run_launcher(directory, benchmark, "recursive_mesh_layer4_8_inject", extra=extra)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    command = shlex.split(submission[submission.index("--cmd") + 1])
                    self.assertEqual(command[command.index("--runtime-config") + 1], "/equals run/formal/runtime.json")
                    self.assertIn(extra[0], command)
            for extra in (("--run-root",), ("--run-root=",), ("--checkpoint", "--output-dir", "/output")):
                with tempfile.TemporaryDirectory() as directory:
                    result, submission = self.run_launcher(directory, benchmark, "recursive_mesh_layer4_8_inject", extra=extra)
                    self.assertEqual(result.returncode, 2)
                    self.assertIsNone(submission)
                    self.assertIn("missing value", result.stderr)

    def test_step_and_geometry_overrides_derive_paths_before_submission(self):
        for variant in ("recursive_mesh_5_10x4_5", "recursive_mesh_layer4_8_audio7slot_split3_prompt_mix"):
            for benchmark in ("mmau", "mmar"):
                for source in ("env", "cli", "equals"):
                    with self.subTest(variant=variant, benchmark=benchmark, source=source), tempfile.TemporaryDirectory() as directory:
                        env = {"RSMOL_MIZAR_CED_EXPECTED_STEP": "100", "RSMOL_MIZAR_CED_EXPECTED_GEOMETRY": "4,8,4"}
                        extra = () if source == "env" else (("--expected-step", "200", "--expected-geometry", "4,4,4")
                            if source == "cli" else ("--expected-step=200", "--expected-geometry=4,4,4"))
                        result, submission = self.run_launcher(directory, benchmark, variant, env, extra)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        command = shlex.split(submission[submission.index("--cmd") + 1])
                        step, geometry = ("100", "4,8,4") if source == "env" else ("200", "4,4,4")
                        # Generated named arguments precede unchanged user arguments.
                        self.assertEqual(command[command.index("--expected-step") + 1], step)
                        self.assertEqual(command[command.index("--expected-geometry") + 1], geometry)
                        self.assertTrue(command[command.index("--checkpoint") + 1].endswith(f"model--step-{step}.ckpt"))
                        self.assertTrue(command[command.index("--output-dir") + 1].endswith("step" + step))


if __name__ == "__main__":
    unittest.main()
