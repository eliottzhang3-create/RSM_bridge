"""Exercise all eight submissions with fake vc and the real compute wrapper."""
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
    "recursive_5_10_5": ("", "20261028_160823559722647", "plain"),
    "recursive_5_10_5_layer8_4_inject": ("_layer8_4_inject", "20261030_000807991327585", "layer84"),
    "recursive_5_10_5_layer4_8_inject": ("_layer4_8_inject", "20261029_210116592758429", "layer48"),
    "recursive_5_10_5_audio_reinject": ("_audio_reinject", "20261029_110929631468450", "reinject"),
}


class RecursiveCedShellTests(unittest.TestCase):
    run_launcher = helpers.MeshCedEvaluationShellTests.run_launcher
    assert_resources = helpers.MeshCedEvaluationShellTests.assert_resources

    def setUp(self):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        self.bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
        if not self.bash: self.skipTest("Bash is unavailable")

    def test_eight_submissions_bind_variant_checkpoint_runtime_route_and_resources(self):
        for variant, (suffix, stamp, tag) in SPECS.items():
            for benchmark in ("mmau", "mmar"):
                with self.subTest(variant=variant, benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                    result, submission = self.run_launcher(directory, benchmark, variant)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assert_resources(submission)
                    command = shlex.split(submission[submission.index("--cmd") + 1])
                    self.assertEqual(command[:2], ["bash", "scripts/evaluate_mizar_s1_ced_recursive_benchmark.sh"])
                    options = dict(zip(command[2::2], command[3::2]))
                    run = f"/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_ced_small_gbs256_recursive_5_10_5{suffix}_5090/formal_{stamp}"
                    self.assertEqual(options["--eval-variant"], variant)
                    self.assertEqual(options["--checkpoint"], run + "/formal/checkpoints/model--step-17313.ckpt")
                    self.assertEqual(options["--runtime-config"], run + "/formal/runtime.json")
                    self.assertTrue(options["--route-root"].endswith("/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small" + suffix))
                    folder = "mmau_test_mini_step17313" if benchmark == "mmau" else "mmar_step17313"
                    self.assertEqual(options["--output-dir"], run + "/eval/" + folder)
                    self.assertEqual(options["--expected-step"], "17313")
                    self.assertEqual(options["--expected-geometry"], "8,8,4")
                    self.assertEqual(options["--benchmark"], benchmark)
                    self.assertTrue(submission[submission.index("-j") + 1].startswith(f"mizar-ced-r515-{tag}-{benchmark}-"))
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
                self.assertEqual(options["--checkpoint"], "/another run/formal/checkpoints/model--step-17313.ckpt")
                self.assertEqual(options["--runtime-config"], "/another run/formal/runtime.json")
                self.assertEqual(options["--output-dir"], "/another run/eval/mmar_step17313")

    def test_unknown_variant_never_calls_vc(self):
        for benchmark in ("mmau", "mmar"):
            with tempfile.TemporaryDirectory() as directory:
                result, submission = self.run_launcher(directory, benchmark, "recursive_5_10_5_typo")
                self.assertEqual(result.returncode, 2)
                self.assertIsNone(submission)

    def test_compute_wrapper_environment_arguments_and_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts, binary = root / "scripts", root / "bin"
            scripts.mkdir(); binary.mkdir()
            source = helpers.RSMOL / "scripts/evaluate_mizar_s1_ced_recursive_benchmark.sh"
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
                result = subprocess.run([self.bash, helpers.bash_path(wrapper), "--eval-variant", "recursive_5_10_5"],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, exit_code, result.stderr)
                self.assertEqual(capture.read_text(encoding="utf-8").splitlines(),
                    ["-u", "scripts/evaluate_mizar_s1_ced_recursive_benchmark.py", "--eval-variant", "recursive_5_10_5"])
                self.assertEqual(captured_env.read_text(encoding="utf-8").splitlines(), ["1", "1", "false", helpers.bash_path(root)])


if __name__ == "__main__":
    unittest.main()
