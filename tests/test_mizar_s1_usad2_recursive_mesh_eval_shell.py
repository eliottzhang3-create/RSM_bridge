"""Execute the real USAD2 launchers with fake vc/Python, without submitting jobs."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code/RSmol"
BASE_RUN = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_usad2_small_gbs256_5090/formal_20261027_115510113068168"
MESH_RUN = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_usad2_small_gbs256_recursive_mesh_5090/formal_20261027_160151646726596"


def bash_path(path):
    value = Path(path).resolve().as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


class MeshUSAD2EvaluationShellTests(unittest.TestCase):
    def setUp(self):
        git_bash = "C:/Program Files/Git/bin/bash.exe"
        self.bash = git_bash if os.name == "nt" and Path(git_bash).is_file() else shutil.which("bash")
        self.bash = self.bash or git_bash
        if not Path(self.bash).is_file():
            self.skipTest("Bash is unavailable")

    def run_launcher(self, directory, benchmark, variant=None, overrides=None, extra=()):
        root = Path(directory)
        binary = root / "bin"
        binary.mkdir(exist_ok=True)
        capture = root / "vc_args"
        capture.unlink(missing_ok=True)
        stub = binary / "vc"
        stub.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$RSM_TEST_CAPTURE"\n',
                        encoding="utf-8", newline="\n")
        stub.chmod(0o755)
        source = RSMOL / f"run_mizar_s1_usad2_small_{benchmark}_eval_5090.sh"
        submit = root / source.name
        submit.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
        env = {key: value for key, value in os.environ.items() if not key.startswith("RSMOL_")}
        env.update(RSM_TEST_CAPTURE=bash_path(capture), PATH=f"{bash_path(binary)}:/usr/bin:/bin")
        if variant is not None:
            env["RSMOL_MIZAR_USAD2_EVAL_VARIANT"] = variant
        env.update(overrides or {})
        result = subprocess.run([self.bash, bash_path(submit), *extra], env=env,
                                capture_output=True, text=True)
        values = capture.read_text(encoding="utf-8").splitlines() if capture.is_file() else None
        return result, values

    def assert_resources(self, submission):
        self.assertEqual(submission[0], "submit")
        for flag, value in (("-p", "pdgpu-5090"), ("-c", "8"), ("-m", "32G"),
                            ("-g", "1"), ("-n", "1"),
                            ("-i", "docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1")):
            self.assertEqual(submission[submission.index(flag) + 1], value)
        self.assertLessEqual(len(submission[submission.index("-j") + 1]), 60)

    def test_both_benchmarks_keep_base_defaults_and_select_mesh_explicitly(self):
        for benchmark in ("mmau", "mmar"):
            for variant in (None, "base", "recursive_mesh"):
                with self.subTest(benchmark=benchmark, variant=variant), tempfile.TemporaryDirectory() as directory:
                    result, submission = self.run_launcher(directory, benchmark, variant)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assert_resources(submission)
                    mesh = variant == "recursive_mesh"
                    run = MESH_RUN if mesh else BASE_RUN
                    script = "evaluate_mizar_s1_usad2_recursive_mesh_benchmark.sh" if mesh else "evaluate_mizar_s1_usad2_benchmark.sh"
                    command = shlex.split(submission[submission.index("--cmd") + 1])
                    self.assertEqual(command[:2], ["bash", "scripts/" + script])
                    options = dict(zip(command[2::2], command[3::2]))
                    self.assertEqual(options["--checkpoint"], run + "/formal/checkpoints/model--step-17313.ckpt")
                    self.assertEqual(options["--runtime-config"], run + "/formal/runtime.json")
                    route = "mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_usad2_small" if mesh else "mizar_stage1_fixed20_gbs256_3epochs_usad2_small"
                    self.assertTrue(options["--route-root"].endswith("/" + route))
                    folder = "mmau_test_mini_step17313" if benchmark == "mmau" else "mmar_step17313"
                    self.assertEqual(options["--output-dir"], run + "/eval/" + folder)
                    self.assertEqual(options["--benchmark"], benchmark)
                    self.assertEqual(options["--expected-step"], "17313")
                    self.assertEqual(options["--expected-geometry"], "8,8,4")
                    if benchmark == "mmau":
                        self.assertEqual(options["--parquet-batch-size"], "8")
                    job = submission[submission.index("-j") + 1]
                    prefix = "mizar-s1-usad2-mesh-" if mesh else "mizar-s1-usad2-"
                    self.assertTrue(job.startswith(prefix + benchmark + "-"))
                    log = submission[submission.index("JOB=1:1") + 1]
                    expected_log = f"mizar_s1_usad2_small_{'recursive_mesh_' if mesh else ''}{benchmark}_5090."
                    self.assertIn(expected_log, log)

    def test_mesh_all_path_overrides_and_extra_arguments_survive_shell_quoting(self):
        for benchmark in ("mmau", "mmar"):
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                overrides = {
                    "RSMOL_MIZAR_USAD2_RUN_ROOT": "/custom run's root",
                    "RSMOL_MIZAR_USAD2_CHECKPOINT": "/checkpoint path/model's step.ckpt",
                    "RSMOL_MIZAR_USAD2_RUNTIME_CONFIG": "/runtime path/config.json",
                    "RSMOL_MIZAR_USAD2_ROUTE_ROOT": "/route path/mesh",
                    "RSMOL_MIZAR_USAD2_EXPECTED_STEP": "8656",
                    "RSMOL_MIZAR_USAD2_EXPECTED_GEOMETRY": "8,8,4",
                    f"RSMOL_{benchmark.upper()}_DATASET_DIR": "/dataset path/official",
                    f"RSMOL_MIZAR_USAD2_{benchmark.upper()}_OUTPUT_DIR": "/output path/eval",
                }
                result, submission = self.run_launcher(directory, benchmark, "recursive_mesh", overrides,
                                                        ("--metadata-json", "/metadata path/records.json"))
                self.assertEqual(result.returncode, 0, result.stderr)
                command = shlex.split(submission[submission.index("--cmd") + 1])
                options = dict(zip(command[2::2], command[3::2]))
                for option, variable in (("--checkpoint", "RSMOL_MIZAR_USAD2_CHECKPOINT"),
                                         ("--runtime-config", "RSMOL_MIZAR_USAD2_RUNTIME_CONFIG"),
                                         ("--route-root", "RSMOL_MIZAR_USAD2_ROUTE_ROOT"),
                                         ("--expected-step", "RSMOL_MIZAR_USAD2_EXPECTED_STEP"),
                                         ("--expected-geometry", "RSMOL_MIZAR_USAD2_EXPECTED_GEOMETRY"),
                                         ("--dataset-dir", f"RSMOL_{benchmark.upper()}_DATASET_DIR"),
                                         ("--output-dir", f"RSMOL_MIZAR_USAD2_{benchmark.upper()}_OUTPUT_DIR")):
                    self.assertEqual(options[option], overrides[variable])
                self.assertEqual(options["--metadata-json"], "/metadata path/records.json")

    def test_unknown_variant_is_rejected_before_vc(self):
        for benchmark in ("mmau", "mmar"):
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                result, submission = self.run_launcher(directory, benchmark, "recursive_mseh")
                self.assertEqual(result.returncode, 2)
                self.assertIsNone(submission)
                self.assertIn("unsupported RSMOL_MIZAR_USAD2_EVAL_VARIANT", result.stderr)

    def test_mesh_run_root_override_derives_matching_checkpoint_runtime_and_output(self):
        for benchmark in ("mmau", "mmar"):
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                run = "/another mesh run"
                result, submission = self.run_launcher(directory, benchmark, "recursive_mesh",
                                                        {"RSMOL_MIZAR_USAD2_RUN_ROOT": run})
                self.assertEqual(result.returncode, 0, result.stderr)
                command = shlex.split(submission[submission.index("--cmd") + 1])
                options = dict(zip(command[2::2], command[3::2]))
                self.assertEqual(options["--checkpoint"], run + "/formal/checkpoints/model--step-17313.ckpt")
                self.assertEqual(options["--runtime-config"], run + "/formal/runtime.json")
                folder = "mmau_test_mini_step17313" if benchmark == "mmau" else "mmar_step17313"
                self.assertEqual(options["--output-dir"], run + "/eval/" + folder)

    def test_compute_wrapper_activates_same_environment_and_propagates_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts, binary = root / "scripts", root / "bin"
            scripts.mkdir()
            binary.mkdir()
            source = RSMOL / "scripts/evaluate_mizar_s1_usad2_recursive_mesh_benchmark.sh"
            wrapper = scripts / source.name
            wrapper.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
            conda = root / "conda base/etc/profile.d"
            conda.mkdir(parents=True)
            (conda / "conda.sh").write_text(
                'conda() { [[ "$1" == activate && "$2" == mellow_c8204d8 ]]; }\n', encoding="utf-8", newline="\n")
            python = binary / "python"
            python.write_text(
                '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$RSM_TEST_CAPTURE"\n'
                'printf \'%s\\n\' "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE" "$TOKENIZERS_PARALLELISM" "$PWD" > "$RSM_TEST_ENV"\n'
                'exit "$RSM_TEST_EXIT"\n', encoding="utf-8", newline="\n")
            python.chmod(0o755)
            capture, captured_env = root / "args", root / "env"
            env = {**os.environ, "RSMOL_CONDA_BASE": bash_path(root / "conda base"),
                   "RSM_TEST_CAPTURE": bash_path(capture), "RSM_TEST_ENV": bash_path(captured_env),
                   "PATH": f"{bash_path(binary)}:/usr/bin:/bin"}
            for exit_code in (0, 7):
                env["RSM_TEST_EXIT"] = str(exit_code)
                result = subprocess.run([self.bash, bash_path(wrapper), "--benchmark", "mmar",
                                         "--checkpoint", "/checkpoint path/model.ckpt"],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, exit_code, result.stderr)
                self.assertEqual(capture.read_text(encoding="utf-8").splitlines(),
                                 ["-u", "scripts/evaluate_mizar_s1_usad2_recursive_mesh_benchmark.py",
                                  "--benchmark", "mmar", "--checkpoint", "/checkpoint path/model.ckpt"])
                actual_env = captured_env.read_text(encoding="utf-8").splitlines()
                self.assertEqual(actual_env[:3], ["1", "1", "false"])
                self.assertEqual(actual_env[3], bash_path(root))


if __name__ == "__main__":
    unittest.main()
