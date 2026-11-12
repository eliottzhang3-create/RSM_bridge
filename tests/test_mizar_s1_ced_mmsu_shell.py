"""Run the actual launcher and compute wrapper with a fake vc/conda only."""
from __future__ import annotations

import json
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


class ShellTests(unittest.TestCase):
    def setUp(self):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        self.bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
        if not self.bash:
            self.skipTest("Bash unavailable")

    def fixture(self, directory):
        path = Path(directory)
        # Copy only the isolated implementation and existing light runtime imports.
        # Parsing does not read production assets or import torch.
        rs = path / "RSmol"
        rs.mkdir()
        package = rs / "mizar_s1_ced_mmsu_eval"
        shutil.copytree(RSMOL / package.name, package)
        shutil.copy2(RSMOL / "run_mizar_s1_ced_small_mmsu_eval_5090.sh", rs)
        (rs / "scripts").mkdir()
        for source in (RSMOL / "scripts").glob("mizar_s1_ced_*eval_runtime.py"):
            shutil.copy2(source, rs / "scripts" / source.name)
        shutil.copytree(RSMOL / "mizar_official_eval", rs / "mizar_official_eval")
        conda = path / "fake conda/etc/profile.d"
        conda.mkdir(parents=True)
        (conda / "conda.sh").write_text(
            'conda() { [[ "$*" == "activate mellow_c8204d8" ]]; }\n'
            'export -f conda\n', encoding="utf-8", newline="\n")
        fake_bin = path / "bin"
        fake_bin.mkdir()
        python = fake_bin / "python"
        python.write_text('#!/usr/bin/env bash\n'
            'printf "%s\\0" "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE" "$TOKENIZERS_PARALLELISM" "$PYTHONIOENCODING" > "$MMSU_ENV_CAPTURE"\n'
            'exec "$MMSU_TEST_PYTHON" "$@"\n', encoding="utf-8", newline="\n")
        python.chmod(0o755)
        startup = path / "startup.sh"
        startup.write_text('vc() { printf "%s\\0" "$@" > "$MMSU_VC_CAPTURE"; }\nexport -f vc\n',
                           encoding="utf-8", newline="\n")
        env = dict(os.environ, RSMOL_CONDA_BASE=(path / "fake conda").as_posix(),
                   MMSU_TEST_PYTHON=Path(sys.executable).as_posix(),
                   MMSU_VC_CAPTURE=(path / "submission.bin").as_posix(),
                   MMSU_ENV_CAPTURE=(path / "environment.bin").as_posix(),
                   BASH_ENV=startup.as_posix(), MSYS_NO_PATHCONV="1", MSYS2_ARG_CONV_EXCL="*")
        # Set PATH inside Bash: its separator is ':' even on Windows.
        with (conda / "conda.sh").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write('export PATH="$MMSU_FAKE_BIN:$PATH"\n')
        location = fake_bin.as_posix()
        env["MMSU_FAKE_BIN"] = ("/" + location[0].lower() + location[2:]
                                 if os.name == "nt" else location)
        return path, rs, env

    def launch(self, rs, env, args):
        return subprocess.run([self.bash, str(rs / "run_mizar_s1_ced_small_mmsu_eval_5090.sh"), *args],
            cwd=rs, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)

    def test_resources_arguments_spaces_and_offline_environment(self):
        for variant in ("base", "recursive_5_10_5_layer4_8_inject", "recursive_mesh_layer36912_listen", "official_mizar"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                path, rs, env = self.fixture(directory)
                args = ["--eval-variant", variant, "--checkpoint", "/weights dir/model's.ckpt",
                        "--output-dir", "/output dir/MMSU", "--mmsu-root", "/data dir/MMSU"]
                if variant == "official_mizar":
                    args.extend(["--model-dir", "/official assets"])
                else:
                    args.extend(["--runtime-config", "/runtime dir/runtime.json", "--expected-step", "456",
                                 "--expected-geometry", "8,4,4"])
                result = self.launch(rs, env, args)
                self.assertEqual(result.returncode, 0, result.stderr)
                submission = (path / "submission.bin").read_bytes().decode().split("\0")[:-1]
                for option, value in (("-p", "pdgpu-5090"), ("-g", "1"), ("-c", "8"),
                        ("-m", "32G"), ("-n", "1"), ("-i", "docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1")):
                    self.assertEqual(submission[submission.index(option) + 1], value)
                command = shlex.split(submission[submission.index("--cmd") + 1])
                self.assertEqual(command[:2], ["bash", "mizar_s1_ced_mmsu_eval/run.sh"])
                self.assertEqual(command[2:], args)
                self.assertEqual((path / "environment.bin").read_bytes().split(b"\0")[:-1],
                                 [b"1", b"1", b"false", b"utf-8"])
                # Execute the real compute shell again in explicit argument-only mode.
                compute = subprocess.run([self.bash, str(rs / command[1]), "--check-args", *command[2:]],
                    cwd=rs, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
                self.assertEqual(compute.returncode, 0, compute.stderr)
                config = json.loads(compute.stdout)
                self.assertTrue(config["dataset_dir"].replace("\\", "/").endswith("/MMSU/MMSU data"))
                self.assertTrue(config["evaluation_script"].replace("\\", "/").endswith("/MMSU/MMSU code/mmsu_evaluation.py"))

    def test_missing_targets_fail_before_vc_and_check_args_never_submits(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            result = self.launch(rs, env, ["--eval-variant", "base"])
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((path / "submission.bin").exists())
            result = self.launch(rs, env, ["--check-args", "--eval-variant=base", "--run-root=/chosen run",
                                          "--expected-step=42", "--expected-geometry=8,8,4"])
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(result.stdout)
            self.assertEqual(config["expected_step"], 42)
            self.assertFalse((path / "submission.bin").exists())


if __name__ == "__main__":
    unittest.main()
