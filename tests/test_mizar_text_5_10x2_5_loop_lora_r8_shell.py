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
              "RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_OUTPUT_DIR", "RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_MODEL_DIR", "RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_RESUME_FROM",
              "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "CONDA_DEFAULT_ENV")}}
with open(os.environ["MOCK_CALLS"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(record) + "\n")
if tool == "python" and any("prepare_text_mizar_text_5_10x2_5_loop_lora_r8.py" in a for a in args):
    sys.exit(int(os.environ.get("MOCK_FAIL_PREPARE", "0")))
if tool == "torchrun":
    sys.exit(int(os.environ.get("MOCK_FAIL_TRAIN", "0")))
if tool == "vc" and os.environ.get("MOCK_RUN_JOB") == "1":
    command = args[args.index("--cmd") + 1]
    environment = {key: value for key, value in os.environ.items() if not key.startswith("RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_")}
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
                        RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_MODEL_DIR=self.model.as_posix(),
                        RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_CONDA_BASE=conda.as_posix(),
                        RSMOL_MIZAR_TEXT_5_10X2_5_LOOP_LORA_R8_TASK_TMP=(self.tmp / "runtime tmp").as_posix(),
                        RSMOL_STAGE4_GATE="D", RSMOL_STAGE4_MICRO_BATCH_SIZE="8",
                        RSMOL_STAGE4_MAX_LR="0.002", RSMOL_STAGE4_MAX_OPTIMIZER_STEPS="10",
                        RSMOL_MIZAR_TEXT_5_10X2_5_MODEL_DIR="/old-route/invalid-model",
                        RSMOL_MIZAR_TEXT_5_10X2_5_OUTPUT_DIR="/old-route/invalid-output",
                        RSMOL_MIZAR_TEXT_5_10X2_5_RESUME_FROM="/old-route/invalid-resume",
                        RSMOL_MIZAR_TEXT_5_10X2_5_MAX_LR="0.1",
                        RSMOL_MIZAR_TEXT_5_10X2_5_MIN_LR="0.01")
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
        for name in ("scripts/convert_mizar_text_5_10x2_5_loop_lora_r8.sh", "scripts/train_stage4_mizar_text_5_10x2_5_loop_lora_r8_ddp.sh", "run_stage4_mizar_text_5_10x2_5_loop_lora_r8_training_5090.sh"):
            with self.subTest(file=name):
                self.assertNotIn(b"\r", (BASE / name).read_bytes())
                result = subprocess.run([BASH, "-n", (BASE / name).as_posix()], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_fresh_formal_submission_then_runtime_with_escaped_paths(self):
        output = (self.tmp / "formal output;$literal" / "formal").as_posix()
        result = self.run_script("run_stage4_mizar_text_5_10x2_5_loop_lora_r8_training_5090.sh", [
            "--model-dir", self.model.as_posix(), "--data-dir", (self.tmp / "data").as_posix(),
            "--output-dir", output, "--max-lr", "8e-4", "--min-lr", "8e-5"])
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.calls()
        self.assertEqual([c["tool"] for c in calls], ["vc", "torchrun"])
        vc = calls[0]["args"]
        self.assertEqual(vc[vc.index("-p") + 1], "pdgpu-5090")
        for flag, value in (("-c", "32"), ("-m", "256G"), ("-g", "8"), ("-n", "1")):
            self.assertEqual(vc[vc.index(flag) + 1], value)
        self.assertIn("sjtu_wumengyue-mhl:0.0.1", vc[vc.index("-i") + 1])
        runtime = self.options(calls[1]["args"])
        for name, value in (("--gate", "FORMAL"), ("--micro-batch-size", "8"),
                            ("--gradient-accumulation-steps", "16"), ("--max-optimizer-steps", "9244"),
                            ("--scheduler-total-steps", "9244"), ("--warmup-steps", "463"),
                            ("--max-lr", "8e-4"), ("--min-lr", "8e-5")):
            self.assertEqual(runtime[name], value)
        self.assertEqual(Path(runtime["--output-dir"]), Path(output))

    def test_cpu_conversion_never_submits_gpu(self):
        result = self.run_script("scripts/convert_mizar_text_5_10x2_5_loop_lora_r8.sh")
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.calls()
        self.assertEqual([c["tool"] for c in calls], ["python"])
        self.assertIn("convert_mizar_text_5_10x2_5_loop_lora_r8.py", " ".join(calls[0]["args"]))
        self.assertEqual(calls[0]["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(self.options(calls[0]["args"])["--threads"], "8")

    def test_default_submission_needs_no_audit_or_preflight(self):
        self.env["MOCK_FAIL_PREPARE"] = "23"
        result = self.run_script("run_stage4_mizar_text_5_10x2_5_loop_lora_r8_training_5090.sh")
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.calls()
        self.assertEqual([c["tool"] for c in calls], ["vc", "torchrun"])
        self.assertNotIn("--audit-report", calls[-1]["args"])
        self.assertNotIn("prepare_text_mizar_text_5_10x2_5_loop_lora_r8", str(calls))
        self.assertNotIn("audit_stage4_dataset", str(calls))
        self.assertEqual(self.options(calls[-1]["args"])["--gate"], "FORMAL")


    def test_resume_alias_forwards_complete_checkpoint(self):
        resume = self.tmp / "checkpoint-step-000500"
        resume.mkdir()
        for name in ("config.json", "training_state.pt", "checkpoint_complete.json"):
            (resume / name).write_text("{}", encoding="utf-8")
        result = self.run_script("run_stage4_mizar_text_5_10x2_5_loop_lora_r8_training_5090.sh", ["--resume-from", resume.as_posix()])
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.calls()
        runtime = self.options(calls[-1]["args"])
        self.assertEqual(Path(runtime["--resume-from"]), resume)
        self.assertEqual(runtime["--gate"], "FORMAL")

    def test_epoch2_weights_only_and_explicit_lower_lrs(self):
        final = self.tmp / "checkpoint-step-009244"
        final.mkdir()
        (final / "config.json").write_text("{}")
        result = self.run_script("run_stage4_mizar_text_5_10x2_5_loop_lora_r8_training_5090.sh", [
            "--model-dir", final.as_posix(), "--max-lr", "2e-4", "--min-lr", "2e-5"])
        self.assertEqual(result.returncode, 0, result.stdout)
        runtime = self.options(self.calls()[-1]["args"])
        self.assertEqual(Path(runtime["--model-path"]), final)
        self.assertEqual(runtime["--learning-rate"], "2e-4")
        self.assertEqual(runtime["--max-lr"], "2e-4")
        self.assertEqual(runtime["--min-lr"], "2e-5")
        self.assertNotIn("--resume-from", runtime)

    def test_min_lr_omitted_is_derived_in_python_parser(self):
        result = self.run_script("run_stage4_mizar_text_5_10x2_5_loop_lora_r8_training_5090.sh", ["--max-lr", "2e-4"])
        self.assertEqual(result.returncode, 0, result.stdout)
        runtime = self.options(self.calls()[-1]["args"])
        self.assertEqual(runtime["--max-lr"], "2e-4")
        self.assertNotIn("--min-lr", runtime)

    def test_dry_run_never_preflights_or_submits(self):
        result = self.run_script("run_stage4_mizar_text_5_10x2_5_loop_lora_r8_training_5090.sh", ["--dry-run"])
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = self.calls()
        self.assertEqual([c["tool"] for c in calls], ["python"])
        self.assertIn("--dry-run", calls[0]["args"])
        self.assertEqual(self.options(calls[0]["args"])["--gate"], "FORMAL")

    def test_unknown_options_fail_before_work(self):
        result = self.run_script("run_stage4_mizar_text_5_10x2_5_loop_lora_r8_training_5090.sh", ["--micro-batch-size", "4"])
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.calls_path.exists())

    def test_training_failure_propagates_through_submit_chain(self):
        self.env["MOCK_FAIL_TRAIN"] = "17"
        result = self.run_script("run_stage4_mizar_text_5_10x2_5_loop_lora_r8_training_5090.sh")
        self.assertEqual(result.returncode, 17, result.stdout)


if __name__ == "__main__":
    unittest.main()
