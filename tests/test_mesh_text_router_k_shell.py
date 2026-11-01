"""Exercise submission and compute wrappers without a real submission or CUDA."""
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


def bash_path(path):
    value = Path(path).resolve().as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


class RouterShellTests(unittest.TestCase):
    def setUp(self):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        self.bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
        if not self.bash: self.skipTest("requires Bash")

    def launcher(self, root, extra=(), check_exit=0, vc_exit=0):
        binary = root / "bin"; binary.mkdir()
        python_args, vc_args = root / "python_args", root / "vc_args"
        for name, capture, exit_code in (("python", python_args, check_exit), ("vc", vc_args, vc_exit)):
            stub = binary / name
            stub.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > {shlex.quote(bash_path(capture))}\nexit {exit_code}\n",
                            encoding="utf-8", newline="\n")
            stub.chmod(0o755)
        source = (RSMOL / "run_mesh_text_router_k_5090.sh").read_text(encoding="utf-8")
        # Only redirect the hardcoded external output ROOT into this test's temp dir.
        source = source.replace("ROOT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mesh_text_router_k_5090",
                                "ROOT=" + shlex.quote(bash_path(root / "remote output")))
        submit = root / "submit.sh"; submit.write_text(source, encoding="utf-8", newline="\n")
        env = {**os.environ, "PATH": f"{bash_path(binary)}:/usr/bin:/bin"}
        result = subprocess.run([self.bash, bash_path(submit), *extra], env=env, text=True, capture_output=True)
        return result, (python_args.read_text().splitlines() if python_args.exists() else None), \
            (vc_args.read_text().splitlines() if vc_args.exists() else None)

    def test_default_resources_original_checkpoint_fixed100_and_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, check, submit = self.launcher(root, ("--prompts-jsonl", "/user input/prompts.jsonl"))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--check-inputs", check)
            self.assertEqual(submit[0], "submit")
            for key, value in (("-p", "pdgpu-5090"), ("-c", "8"), ("-m", "32G"), ("-g", "1"), ("-n", "1")):
                self.assertEqual(submit[submit.index(key) + 1], value)
            self.assertLessEqual(len(submit[submit.index("-j") + 1]), 60)
            command = shlex.split(submit[submit.index("--cmd") + 1])
            self.assertEqual(command[:2], ["bash", "scripts/calibrate_mesh_text_router_k.sh"])
            options = dict(zip(command[2::2], command[3::2]))
            self.assertEqual(options["--max-new-tokens"], "100")
            self.assertEqual(options["--new-slot-mass-cap"], "0.005")
            self.assertEqual(options["--k-margin"], "1.0")
            self.assertIn("formal_round2_lr2e-4_2e-5_resume5000_20260908", options["--text-model-dir"])
            self.assertIn("run_", options["--output-dir"])

    def test_spaces_apostrophes_and_overrides_survive_bash_quoting(self):
        with tempfile.TemporaryDirectory() as directory:
            result, check, submit = self.launcher(Path(directory), (
                "--prompts-jsonl", "/user's input/prompts.jsonl", "--text-model-dir", "/model weights/original",
                "--output-dir", "/new output/run", "--new-slot-mass-cap", "0.001", "--k-margin", "2.0",
                "--max-new-tokens", "100", "--seed", "7"))
            self.assertEqual(result.returncode, 0, result.stderr)
            command = shlex.split(submit[submit.index("--cmd") + 1])
            options = dict(zip(command[2::2], command[3::2]))
            self.assertEqual(options["--prompts-jsonl"], "/user's input/prompts.jsonl")
            self.assertEqual(options["--text-model-dir"], "/model weights/original")
            self.assertEqual(options["--output-dir"], "/new output/run")
            self.assertEqual(options["--seed"], "7")
            self.assertEqual(options["--new-slot-mass-cap"], "0.001")
            self.assertEqual(options["--k-margin"], "2.0")

    def test_bad_input_check_or_fixed_horizon_or_missing_prompts_never_submits(self):
        cases = [(("--prompts-jsonl", "/prompts"), 3), (("--prompts-jsonl", "/prompts", "--max-new-tokens", "99"), 0),
                 ((), 0), (("--prompts-jsonl", "relative.jsonl"), 0)]
        for args, check_exit in cases:
            with tempfile.TemporaryDirectory() as directory:
                result, check, submit = self.launcher(Path(directory), args, check_exit)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(submit)

    def test_submit_failure_propagates_without_success_message(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self.launcher(Path(directory), ("--prompts-jsonl", "/prompts"), vc_exit=7)
            self.assertEqual(result.returncode, 7)
            self.assertNotIn("Submitted", result.stdout)

    def test_real_compute_wrapper_activates_environment_offline_and_propagates_exit(self):
        for exit_code in (0, 9):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                scripts, binary = root / "scripts", root / "bin"
                scripts.mkdir(); binary.mkdir()
                wrapper = scripts / "calibrate_mesh_text_router_k.sh"
                wrapper.write_text((RSMOL / "scripts" / wrapper.name).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
                conda = root / "conda base/etc/profile.d"; conda.mkdir(parents=True)
                (conda / "conda.sh").write_text('conda() { [[ "$1" == activate && "$2" == mellow_c8204d8 ]]; }\n', encoding="utf-8", newline="\n")
                capture = root / "capture"
                stub = binary / "python"
                stub.write_text("#!/usr/bin/env bash\n"
                    '[[ "$HF_HUB_OFFLINE" == 1 && "$TRANSFORMERS_OFFLINE" == 1 && "$HF_DATASETS_OFFLINE" == 1 && "$TOKENIZERS_PARALLELISM" == false ]] || exit 19\n'
                    f"printf '%s\\n' \"$@\" > {shlex.quote(bash_path(capture))}\nexit {exit_code}\n", encoding="utf-8", newline="\n")
                stub.chmod(0o755)
                result = subprocess.run([self.bash, bash_path(wrapper), "--prompts-jsonl", "/a prompt path"],
                    env={**os.environ, "RSMOL_CONDA_BASE": bash_path(root / "conda base"), "PATH": f"{bash_path(binary)}:/usr/bin:/bin"},
                    text=True, capture_output=True)
                self.assertEqual(result.returncode, exit_code, result.stderr)
                self.assertEqual(capture.read_text().splitlines(), ["-B", "-u", "scripts/calibrate_mesh_text_router_k.py", "--prompts-jsonl", "/a prompt path"])


if __name__ == "__main__":
    unittest.main()
