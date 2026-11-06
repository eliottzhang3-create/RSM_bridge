"""Execute real Bash with explicit fake scheduler/trainer processes; no jobs or GPUs."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "code/RSmol/run_mizar_official_final_s1_gbs128_5_20x2_5_5090.sh"
RUN = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_official_final_s1_gbs128_5_20x2_5/run.sh"
BASH = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing/bash")


def quoted(value):
    return shlex.quote(Path(value).as_posix())


@unittest.skipUnless(BASH.is_file(), "Bash unavailable")
class ShellTests(unittest.TestCase):
    def run_bash(self, harness, *args, env=None):
        return subprocess.run([str(BASH), str(harness), *map(str, args)], capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=20,
                              env=dict(os.environ, PYTHONUTF8="1", **(env or {})))

    def submission_fixture(self, tmp):
        root = Path(tmp)
        model = root / "model with spaces"
        for d in (model, model/"tokenizer", model/"ced-small", root/"index"):
            d.mkdir(parents=True, exist_ok=True)
        for f in (model/"manifest.json", model/"model.ckpt", root/"manifest.json", root/"paths.tsv"):
            f.write_text("fixture")
        script = root/"launch.sh"
        text = LAUNCHER.read_text().replace(
            "/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json", (root/"manifest.json").as_posix())
        script.write_text(text, newline="\n")
        harness = root/"harness.sh"
        harness.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
python() {{
  case "$1" in
    */check_model.py) echo AUDIT >> {quoted(root/'events')} ;;
    */prepare_index.py)
      echo INDEX >> {quoted(root/'events')}
      while [[ $# -gt 0 ]]; do
        if [[ "$1" == --path-output ]]; then printf '%s\\n' {quoted(root/'index')} > "$2"; return; fi
        shift
      done
      return 91 ;;
    *) return 92 ;;
  esac
}}
vc() {{ printf '%s\\0' "$@" > {quoted(root/'vc.args')}; }}
source {quoted(script)} "$@"
''', newline="\n")
        return root, model, harness

    def test_fresh_submit_has_correct_resources_one_model_argument_and_no_prerequisite_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, model, harness = self.submission_fixture(tmp)
            output = root/"new output"
            result = self.run_bash(harness, "--model-dir", model.as_posix(), "--resolved-paths",
                                   (root/"paths.tsv").as_posix(), "--output-dir", output.as_posix(), "--stage-workers", "3")
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            args = (root/"vc.args").read_text().strip("\0").split("\0")
            for option, expected in (("-p", "pdgpu-5090"), ("-c", "32"), ("-m", "256G"), ("-g", "8"), ("-n", "1")):
                self.assertEqual(args[args.index(option)+1], expected)
            command = shlex.split(args[args.index("--cmd")+1])
            self.assertEqual(command[2], "formal")
            self.assertEqual(command[-1], "3")
            self.assertEqual(sum("model with spaces" in a for a in command), 1)
            self.assertEqual((root/"events").read_text().splitlines(), ["AUDIT", "INDEX"])
            self.assertIn("5_20x2_5", command[1])
            self.assertNotIn("CUDA_VISIBLE_DEVICES", LAUNCHER.read_text())

    def test_resume_infers_original_output_and_rejects_other_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, model, harness = self.submission_fixture(tmp)
            output = root/"old output"
            checkpoint = output/"formal/checkpoints/model--step-5771.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("shell fixture; not a tensor checkpoint")
            common = ["--model-dir", model.as_posix(), "--resolved-paths", (root/"paths.tsv").as_posix(),
                      "--resume-checkpoint", checkpoint.as_posix()]
            result = self.run_bash(harness, *common)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            args = (root/"vc.args").read_text().split("\0")
            command = shlex.split(args[args.index("--cmd")+1])
            self.assertTrue(any(a.endswith("old output") for a in command), command)
            (root/"vc.args").unlink()
            result = self.run_bash(harness, *common, "--output-dir", (root/"wrong").as_posix())
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("original output", result.stderr)
            self.assertFalse((root/"vc.args").exists())

    def test_nonempty_fresh_output_and_invalid_workers_never_submit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, model, harness = self.submission_fixture(tmp)
            output = root/"existing"
            output.mkdir()
            (output/"keep.txt").write_text("must survive")
            common = ["--model-dir", model.as_posix(), "--resolved-paths", (root/"paths.tsv").as_posix()]
            for suffix in (["--stage-workers", "0"], ["--output-dir", output.as_posix()]):
                result = self.run_bash(harness, *common, *suffix)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root/"vc.args").exists())
            self.assertEqual((output/"keep.txt").read_text(), "must survive")

    def test_compute_starts_torchrun_before_copier_completion_and_reaps_before_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conda = root/"conda/etc/profile.d"
            conda.mkdir(parents=True)
            (conda/"conda.sh").write_text("conda() { :; }\n")
            script = root/"run.sh"
            script.write_text(RUN.read_text().replace("/dev/shm/", root.as_posix()+"/"), newline="\n")
            harness = root/"harness.sh"
            harness.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
export MELLOW_CONDA_BASE={quoted(root/'conda')}
event() {{ echo "$1" >> {quoted(root/'events')}; }}
python() {{
  case "$1" in
    stage_data.py)
      [[ "$*" == *--background* ]] || return 90
      event COPY_START
      touch {quoted(root/'metadata-ready')}
      for ((i=0; i<500; i++)); do
        if [[ -e {quoted(root/'training-started')} ]]; then
          event COPY_COMPLETE
          touch {quoted(root/'copy-complete')}
          return
        fi
        sleep .01
      done
      return 91 ;;
    -m)
      for ((i=0; i<500; i++)); do
        [[ ! -e {quoted(root/'copy-complete')} ]] || return 92
        if [[ -e {quoted(root/'metadata-ready')} ]]; then event METADATA_READY; return; fi
        sleep .01
      done
      return 93 ;;
    write_config.py)
      [[ ! -e {quoted(root/'copy-complete')} ]] || return 94
      event CONFIG ;;
    */stage_data.py)
      [[ "$*" == *--cleanup* && -e {quoted(root/'copy-complete')} ]] || return 95
      event CLEANUP ;;
    -) {quoted(sys.executable)} "$@" ;;
    *) return 96 ;;
  esac
}}
torchrun() {{
  [[ "$*" == *--nproc_per_node=8* ]] || return 97
  [[ ! -e {quoted(root/'copy-complete')} ]] || return 98
  event TRAIN_START
  touch {quoted(root/'training-started')}
  printf '%s\\n' '{{"status":"PASS","optimizer_steps":34626}}' > {quoted(root/'out/formal/training_report.json')}
}}
source {quoted(script)} formal {quoted(root/'paths.tsv')} {quoted(root/'out')} '' {quoted(root/'model')} {quoted(root/'index')} 4
''', newline="\n")
            (root/"paths.tsv").write_text("fixture")
            (root/"model").mkdir()
            result = self.run_bash(harness)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            events = (root/"events").read_text().splitlines()
            self.assertEqual(events, ["COPY_START", "METADATA_READY", "CONFIG", "TRAIN_START", "COPY_COMPLETE", "CLEANUP"])


if __name__ == "__main__":
    unittest.main()
