"""Execute real Bash wrappers with fake conda, vc and torchrun executables."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "code/RSmol"
BASH = str(Path("C:/Program Files/Git/bin/bash.exe")) if os.name == "nt" else shutil.which("bash")

DRIVER = r'''
import json, os, subprocess, sys
from pathlib import Path
tool, args = sys.argv[1], sys.argv[2:]
record = {"tool": tool, "args": args,
          "env": {key: os.environ.get(key) for key in (
              "RSMOL_5_20X2_5_OUTPUT_DIR", "RSMOL_5_20X2_5_MODEL_DIR", "RSMOL_5_20X2_5_RESUME_FROM",
              "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "CONDA_DEFAULT_ENV")}}
with open(os.environ["MOCK_CALLS"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(record) + "\n")
if tool == "python" and any("prepare_text_5_20x2_5.py" in a for a in args):
    sys.exit(int(os.environ.get("MOCK_FAIL_PREPARE", "0")))
if tool == "vc" and os.environ.get("MOCK_RUN_JOB") == "1":
    command = args[args.index("--cmd") + 1]
    environment = {key: value for key, value in os.environ.items() if not key.startswith("RSMOL_5_20X2_5_")}
    sys.exit(subprocess.run([os.environ["MOCK_BASH"], "-c", command], env=environment).returncode)
'''


class TextShellTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="text52025-")
        self.addCleanup(self.temp.cleanup)
        self.tmp = Path(self.temp.name)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        driver = self.tmp / "driver.py"
        driver.write_text(DRIVER, encoding="utf-8", newline="\n")
        for name in ("python", "vc", "torchrun"):
            tool = self.bin / name
            tool.write_text('#!/bin/bash\nexec "$MOCK_PYTHON" "$MOCK_DRIVER" ' + name + ' "$@"\n', encoding="utf-8", newline="\n")
            tool.chmod(0o755)
        conda = self.tmp / "conda base"
        profile = conda / "etc/profile.d"
        profile.mkdir(parents=True)
        (profile / "conda.sh").write_text('conda() { export CONDA_DEFAULT_ENV="$2"; }\n', encoding="utf-8", newline="\n")
        self.model = self.tmp / "model dir"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}", encoding="utf-8")
        self.audit = self.tmp / "audit.json"
        self.audit.write_text("{}", encoding="utf-8")
        self.calls_path = self.tmp / "calls.jsonl"
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        MOCK_PYTHON=Path(sys.executable).as_posix(), MOCK_DRIVER=driver.as_posix(),
                        MOCK_CALLS=str(self.calls_path), MOCK_BASH=BASH, MOCK_RUN_JOB="1",
                        RSMOL_5_20X2_5_MODEL_DIR=self.model.as_posix(),
                        RSMOL_5_20X2_5_CONDA_BASE=conda.as_posix(),
                        RSMOL_5_20X2_5_TASK_TMP=(self.tmp / "runtime tmp").as_posix(),
                        RSMOL_STAGE4_GATE="D", RSMOL_STAGE4_MICRO_BATCH_SIZE="8",
                        RSMOL_STAGE4_MAX_LR="0.002", RSMOL_STAGE4_MAX_OPTIMIZER_STEPS="10")
        # Keep Linux job-command strings literal when a Windows mock process
        # receives --cmd from Git Bash. The remote Linux launcher needs no such setting.
        self.env.update(MSYS_NO_PATHCONV="1", MSYS2_ARG_CONV_EXCL="*")

    def run_script(self, name, args=()):
        return subprocess.run([BASH, (BASE / name).as_posix(), *map(str, args)], env=self.env,
                              cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=40)

    def calls(self):
        return [json.loads(line) for line in self.calls_path.read_text(encoding="utf-8").splitlines()]

    def options(self, args):
        return {name: args[i + 1] for i, name in enumerate(args[:-1]) if name.startswith("--")}

    def test_shell_syntax_and_lf(self):
        for name in ("scripts/convert_stepwise_5_20x2_5.sh", "scripts/train_stage4_5_20x2_5_ddp.sh",
                     "run_stage4_5_20x2_5_5090.sh", "run_stage4_5_20x2_5_training_5090.sh"):
            with self.subTest(file=name):
                self.assertNotIn(b"\r", (BASE / name).read_bytes())
                result = subprocess.run([BASH, "-n", (BASE / name).as_posix()], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_fresh_formal_submission_then_runtime_with_escaped_paths(self):
        output = (self.tmp / "formal output;$literal" / "formal").as_posix()
        result = self.run_script("run_stage4_5_20x2_5_training_5090.sh", [
            "--model-dir", self.model.as_posix(), "--data-dir", (self.tmp / "data").as_posix(),
            "--output-dir", output, "--audit-report", self.audit.as_posix()])
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.calls()
        self.assertEqual([c["tool"] for c in calls], ["vc", "torchrun"])
        vc = calls[0]["args"]
        self.assertEqual(vc[vc.index("-p") + 1], "pdgpu-5090")
        for flag, value in (("-c", "32"), ("-m", "256G"), ("-g", "8"), ("-n", "1")):
            self.assertEqual(vc[vc.index(flag) + 1], value)
        self.assertIn("sjtu_wumengyue-mhl:0.0.1", vc[vc.index("-i") + 1])
        runtime = self.options(calls[1]["args"])
        for name, value in (("--gate", "FORMAL"), ("--micro-batch-size", "4"),
                            ("--gradient-accumulation-steps", "16"), ("--max-optimizer-steps", "18488"),
                            ("--scheduler-total-steps", "18488"), ("--warmup-steps", "925"),
                            ("--max-lr", "8e-4"), ("--min-lr", "8e-5")):
            self.assertEqual(runtime[name], value)
        self.assertEqual(Path(runtime["--output-dir"]), Path(output))
        self.assertEqual(Path(runtime["--audit-report"]), self.audit)

    def test_cpu_conversion_never_submits_gpu(self):
        result = self.run_script("scripts/convert_stepwise_5_20x2_5.sh")
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.calls()
        self.assertEqual([c["tool"] for c in calls], ["python"])
        self.assertIn("convert_stepwise_5_20x2_5.py", " ".join(calls[0]["args"]))
        self.assertEqual(calls[0]["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(self.options(calls[0]["args"])["--threads"], "8")

    def test_default_submission_needs_no_audit_or_preflight(self):
        self.env["MOCK_FAIL_PREPARE"] = "23"
        result = self.run_script("run_stage4_5_20x2_5_training_5090.sh")
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.calls()
        self.assertEqual([c["tool"] for c in calls], ["vc", "torchrun"])
        self.assertNotIn("--audit-report", calls[-1]["args"])
        self.assertNotIn("prepare_text_5_20x2_5", str(calls))
        self.assertNotIn("audit_stage4_dataset", str(calls))
        self.assertEqual(self.options(calls[-1]["args"])["--gate"], "FORMAL")

    def test_missing_optional_report_does_not_block_submission_or_runtime(self):
        missing = self.tmp / "old mount" / "missing-audit.json"
        result = self.run_script("run_stage4_5_20x2_5_training_5090.sh", ["--audit-report", missing.as_posix()])
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.calls()
        self.assertEqual([c["tool"] for c in calls], ["vc", "torchrun"])
        self.assertEqual(Path(self.options(calls[-1]["args"])["--audit-report"]), missing)

    def test_resume_alias_forwards_complete_checkpoint(self):
        resume = self.tmp / "checkpoint-step-000500"
        resume.mkdir()
        for name in ("config.json", "training_state.pt", "checkpoint_complete.json"):
            (resume / name).write_text("{}", encoding="utf-8")
        result = self.run_script("run_stage4_5_20x2_5_5090.sh", ["--resume-from", resume.as_posix()])
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.calls()
        runtime = self.options(calls[-1]["args"])
        self.assertEqual(Path(runtime["--resume-from"]), resume)
        self.assertEqual(runtime["--gate"], "FORMAL")

    def test_dry_run_never_preflights_or_submits(self):
        result = self.run_script("run_stage4_5_20x2_5_training_5090.sh", ["--dry-run"])
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.calls()
        self.assertEqual([c["tool"] for c in calls], ["python"])
        self.assertIn("--dry-run", calls[0]["args"])
        self.assertEqual(self.options(calls[0]["args"])["--gate"], "FORMAL")

    def test_unknown_options_fail_before_work(self):
        result = self.run_script("run_stage4_5_20x2_5_training_5090.sh", ["--micro-batch-size", "8"])
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.calls_path.exists())


if __name__ == "__main__":
    unittest.main()
