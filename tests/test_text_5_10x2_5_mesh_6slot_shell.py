"""Run the real Bash entrypoints against local vc/conda/torchrun stand-ins."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code/RSmol"
PREFIX = "RSMOL_5_10X2_5_MESH_6SLOT_"
BASH = shutil.which("bash")
if BASH is None and Path("C:/Program Files/Git/bin/bash.exe").is_file():
    BASH = "C:/Program Files/Git/bin/bash.exe"


def posix(path):
    path = Path(path).resolve()
    value = path.as_posix()
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


@unittest.skipUnless(BASH, "Bash is unavailable")
class SixSlotShellTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="six slot shell ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.code = self.root / "code/RSmol"
        (self.code / "scripts").mkdir(parents=True)
        for source in CODE.glob("run_stage4_5_10x2_5_mesh_6slot*.sh"):
            shutil.copyfile(source, self.code / source.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.capture = self.root / "args.bin"
        self.env_capture = self.root / "environment.bin"
        self.env = {key:value for key,value in os.environ.items() if not key.startswith(PREFIX)}
        self.env.update(PATH=str(self.bin)+os.pathsep+self.env.get("PATH",""), CAPTURE=posix(self.capture), ENV_CAPTURE=posix(self.env_capture))

    def executable(self, name, body):
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"+body, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def run_script(self, path):
        return subprocess.run([BASH,posix(path)], cwd=self.root, env=self.env, capture_output=True, text=True, timeout=30)

    def captured_args(self):
        return self.capture.read_bytes().decode().rstrip("\0").split("\0")

    def prepare_submission(self):
        self.executable("vc", 'printf "%s\\0" "$@" > "$CAPTURE"\n')
        (self.code / "scripts/train_stage4_5_10x2_5_mesh_6slot_ddp.sh").write_text('env -0 > "$ENV_CAPTURE"\n', encoding="utf-8", newline="\n")

    def remote_environment(self, args):
        command = args[args.index("--cmd")+1]
        result = subprocess.run([BASH,"-c",command], cwd=self.code, env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        return dict(item.split("=",1) for item in self.env_capture.read_bytes().decode().split("\0") if "=" in item)

    def prepare_runtime(self):
        source = (CODE / "scripts/train_stage4_5_10x2_5_mesh_6slot_ddp.sh").read_text(encoding="utf-8")
        conda = self.root / "conda/etc/profile.d"
        conda.mkdir(parents=True)
        (conda / "conda.sh").write_text('conda() { :; }\n',encoding="utf-8",newline="\n")
        source = source.replace("USER_CONDA_BASE=/hpc_stor03/sjtu_home/jinwei.zhang/env/miniconda3", "USER_CONDA_BASE="+shlex.quote(posix(self.root / "conda")))
        target = self.code / "scripts/train_stage4_5_10x2_5_mesh_6slot_ddp.sh"
        target.write_text(source,encoding="utf-8",newline="\n")
        self.executable("torchrun", 'printf "%s\\0" "$@" > "$CAPTURE"\n')
        data = self.root / "parquet data"
        data.mkdir()
        self.env[PREFIX+"DATA_DIR"] = posix(data)
        self.env[PREFIX+"OUTPUT_DIR"] = posix(self.root / "training output")
        return target

    def test_all_bash_sources_parse_and_use_lf(self):
        paths = [*CODE.glob("run_stage4_5_10x2_5_mesh_6slot*.sh"), * (CODE / "scripts").glob("*5_10x2_5_mesh_6slot*.sh")]
        self.assertEqual(len(paths),6)
        for path in paths:
            self.assertNotIn(b"\r",path.read_bytes(),path)
            result = subprocess.run([BASH,"-n",posix(path)],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_formal_submission_resources_quoting_and_fresh_initialization(self):
        self.prepare_submission()
        value = "output with spaces ' and $(false) `false`"
        self.env.update({PREFIX+"OUTPUT_DIR":value,PREFIX+"MODEL_DIR":"isolated six-slot model",PREFIX+"RESUME_FROM":"stale smoke checkpoint"})
        result = self.run_script(self.code / "run_stage4_5_10x2_5_mesh_6slot_formal_3090.sh")
        self.assertEqual(result.returncode,0,result.stderr)
        args = self.captured_args()
        for flag, expected in (("-p","pdgpu-3090"),("-c","32"),("-m","256G"),("-g","8"),("-n","1")):
            self.assertEqual(args[args.index(flag)+1],expected)
        self.assertLessEqual(len(args[args.index("-j")+1]),60)
        remote = self.remote_environment(args)
        self.assertEqual(remote[PREFIX+"OUTPUT_DIR"],value)
        self.assertEqual(remote[PREFIX+"MODEL_DIR"],"isolated six-slot model")
        self.assertEqual(remote[PREFIX+"STAGE4_GATE"],"FORMAL")
        self.assertEqual(remote[PREFIX+"RESUME_FROM"],"")

    def test_explicit_formal_resume_is_forwarded(self):
        self.prepare_submission()
        self.env[PREFIX+"FORMAL_RESUME_FROM"] = "/formal run/checkpoint-000500"
        result = self.run_script(self.code / "run_stage4_5_10x2_5_mesh_6slot_formal_3090.sh")
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(self.remote_environment(self.captured_args())[PREFIX+"RESUME_FROM"],"/formal run/checkpoint-000500")

    def test_formal_runtime_arguments_cover_full_epoch(self):
        target = self.prepare_runtime()
        result = self.run_script(target)
        self.assertEqual(result.returncode,0,result.stderr)
        args = self.captured_args()
        expected = {"--gate":"FORMAL","--micro-batch-size":"8","--gradient-accumulation-steps":"16","--context-length":"1024","--max-optimizer-steps":"9244","--scheduler-total-steps":"9244","--warmup-steps":"463","--epochs":"1","--max-lr":"8e-4","--min-lr":"8e-5","--save-every":"500"}
        for flag,value in expected.items():
            self.assertEqual(args[args.index(flag)+1],value,flag)
        self.assertTrue(args[args.index("--model-path")+1].endswith("SmolLM2-5-10x2-5-mesh-6slot"))

    def test_runtime_rejects_nonempty_output(self):
        target = self.prepare_runtime()
        output = self.root / "training output"
        output.mkdir()
        (output / "keep.txt").write_text("existing run",encoding="utf-8")
        result = self.run_script(target)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("refusing nonempty",result.stderr)
        self.assertFalse(self.capture.exists())

    def test_smoke_and_resume_wrappers_keep_diagnostic_gates(self):
        self.prepare_submission()
        smoke = self.run_script(self.code / "run_stage4_5_10x2_5_mesh_6slot_smoke_3090.sh")
        self.assertEqual(smoke.returncode,0,smoke.stderr)
        self.assertEqual(self.remote_environment(self.captured_args())[PREFIX+"STAGE4_GATE"],"D")
        checkpoint = self.root / "checkpoint-000010"
        checkpoint.mkdir()
        for name in ("checkpoint_complete.json","checkpoint_manifest.json","training_state.pt"):
            (checkpoint / name).write_text("fixture",encoding="utf-8")
        self.env[PREFIX+"RESUME_FROM"] = posix(checkpoint)
        resume = self.run_script(self.code / "run_stage4_5_10x2_5_mesh_6slot_resume_3090.sh")
        self.assertEqual(resume.returncode,0,resume.stderr)
        remote = self.remote_environment(self.captured_args())
        self.assertEqual(remote[PREFIX+"STAGE4_GATE"],"E")
        self.assertEqual(remote[PREFIX+"RESUME_FROM"],posix(checkpoint))


if __name__ == "__main__":
    unittest.main()
