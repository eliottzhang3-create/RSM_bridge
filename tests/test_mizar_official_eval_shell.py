"""Real Bash submission/compute wrappers using local vc and Python stubs."""
from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RSMOL = ROOT / "code/RSmol"
sys.path.insert(0, str(RSMOL))
from mizar_official_eval import evaluate as runner

spec = importlib.util.spec_from_file_location("ced_shell_helpers", ROOT / "tests/test_mizar_s1_ced_recursive_mesh_eval_shell.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
bash_path = helpers.bash_path


class OfficialShellTests(unittest.TestCase):
    def setUp(self):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        self.bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
        if not self.bash:
            self.skipTest("Bash unavailable")

    assert_resources = helpers.MeshCedEvaluationShellTests.assert_resources

    def run_launcher(self, directory, benchmark, overrides=None, extra=()):
        target = Path(directory) / "mizar_official_eval"
        target.mkdir(exist_ok=True)
        for name in ("submit_5090.sh", "run.sh"):
            (target / name).write_bytes((RSMOL / "mizar_official_eval" / name).read_bytes().replace(b"\r\n", b"\n"))
        return helpers.MeshCedEvaluationShellTests.run_launcher(
            self, directory, benchmark, "official_mizar", overrides, extra)

    def test_both_existing_entries_submit_isolated_official_evaluation(self):
        for benchmark, folder in (("mmau", "mmau_test_mini"), ("mmar", "mmar")):
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                result, submission = self.run_launcher(directory, benchmark)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assert_resources(submission)
                command = shlex.split(submission[submission.index("--cmd") + 1])
                self.assertEqual(command[:2], ["bash", "mizar_official_eval/run.sh"])
                args = runner.parse_args(command[2:])
                self.assertEqual(args.benchmark, benchmark)
                self.assertEqual(args.model_dir, runner.runtime_adapter.DEFAULT_MODEL_DIR.resolve())
                self.assertEqual(args.output_dir, (runner.OUTPUT_ROOT / folder).resolve())
                self.assertNotIn("--expected-step", command)
                self.assertNotIn("--runtime-config", command)
                self.assertIn(f"mizar_official_{benchmark}_5090.", submission[submission.index("JOB=1:1") + 1])

    def test_environment_and_cli_paths_survive_quoting(self):
        for benchmark in ("mmau", "mmar"):
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as directory:
                overrides = {
                    "RSMOL_MIZAR_OFFICIAL_MODEL_DIR": "/model's assets/Mizar159M",
                    "RSMOL_MIZAR_OFFICIAL_CHECKPOINT": "/final checkpoint/model's.ckpt",
                    f"RSMOL_{benchmark.upper()}_DATASET_DIR": "/official dataset",
                    f"RSMOL_MIZAR_OFFICIAL_{benchmark.upper()}_OUTPUT_DIR": "/official output",
                    "RSMOL_MIZAR_CED_CHECKPOINT": "/unrelated old training.ckpt",
                    "RSMOL_MIZAR_CED_RUN_ROOT": "/unrelated old run",
                }
                result, submission = self.run_launcher(directory, benchmark, overrides,
                    ("--metadata-json", "/metadata dir/record's.json", "--output-dir", "/CLI override/eval"))
                self.assertEqual(result.returncode, 0, result.stderr)
                command = shlex.split(submission[submission.index("--cmd") + 1])
                args = runner.parse_args(command[2:])
                self.assertEqual(args.model_dir, Path(overrides["RSMOL_MIZAR_OFFICIAL_MODEL_DIR"]).resolve())
                self.assertEqual(args.checkpoint, Path(overrides["RSMOL_MIZAR_OFFICIAL_CHECKPOINT"]).resolve())
                self.assertEqual(args.dataset_dir, Path("/official dataset").resolve())
                self.assertEqual(args.metadata_json, Path("/metadata dir/record's.json").resolve())
                self.assertEqual(args.output_dir, Path("/CLI override/eval").resolve())
                self.assertNotIn("/unrelated old training.ckpt", command)

    def test_cli_model_override_derives_checkpoint_from_effective_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result, submission = self.run_launcher(directory, "mmau", extra=("--model-dir", "/another model"))
            self.assertEqual(result.returncode, 0, result.stderr)
            command = shlex.split(submission[submission.index("--cmd") + 1])
            args = runner.parse_args(command[2:])
            self.assertEqual(args.checkpoint, Path("/another model/checkpoints/final/stage3_q4_seed20260905.ckpt").resolve())

    def test_compute_wrapper_offline_environment_and_exit_propagation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper_dir, binary = root / "mizar_official_eval", root / "bin"
            wrapper_dir.mkdir()
            binary.mkdir()
            wrapper = wrapper_dir / "run.sh"
            wrapper.write_bytes((RSMOL / "mizar_official_eval/run.sh").read_bytes().replace(b"\r\n", b"\n"))
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
                                         "--model-dir", "/model's assets"],
                                        env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, exit_code, result.stderr)
                self.assertEqual(capture.read_text(encoding="utf-8").splitlines(),
                                 ["-u", "mizar_official_eval/evaluate.py", "--benchmark", "mmar",
                                  "--model-dir", "/model's assets"])
                self.assertEqual(captured_env.read_text(encoding="utf-8").splitlines(),
                                 ["1", "1", "false", bash_path(root)])

    def test_original_launcher_branches_are_byte_identical_after_removing_official_case(self):
        # git supplies the pre-change sources during development; the unified
        # regression suite independently exercises all old routes after upload.
        for benchmark in ("mmau", "mmar"):
            name = f"code/RSmol/run_mizar_s1_ced_small_{benchmark}_eval_5090.sh"
            prior = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT,
                                   capture_output=True, check=True).stdout.replace(b"\r\n", b"\n")
            current = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
            case = (f'  official_mizar)\n    exec bash "$SCRIPT_DIR/mizar_official_eval/submit_5090.sh" {benchmark} "$@"\n    ;;\n').encode()
            # HEAD may already include this extension when testing a committed checkout.
            self.assertEqual(current.replace(case, b""), prior.replace(case, b""))


if __name__ == "__main__":
    unittest.main()
