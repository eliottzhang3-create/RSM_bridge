"""Dry-run and stub-vc tests: never submit to the actual queue."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT/'code/RSmol/run_stage4_mizar_text_7_14x2_7_mesh_training_5090.sh'
RUNTIME = ROOT/'code/RSmol/scripts/train_stage4_mizar_text_7_14x2_7_mesh_ddp.sh'
CONVERTER = ROOT/'code/RSmol/scripts/convert_mizar_text_7_14x2_7_mesh.sh'
PREFIX = 'RSMOL_MIZAR_TEXT_7_14X2_7_MESH_'
INITIAL_MODEL = '/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x2_7_mesh/formal_20261117_202454_1706253/checkpoint-001000'
BASH = os.environ.get('CODEX_BASH') or ('C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash'))


@unittest.skipUnless(BASH and Path(BASH).is_file(), 'Bash is unavailable')
class ShellTests(unittest.TestCase):
    def run_bash(self, *args, env=None):
        clean = {k:v for k,v in os.environ.items() if not k.startswith(PREFIX)}
        clean.update(env or {})
        return subprocess.run([BASH,*map(str,args)],cwd=ROOT,env=clean,text=True,
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)

    def test_syntax_and_default_formal_dry_run(self):
        for path in (ENTRY,RUNTIME,CONVERTER):
            result = self.run_bash('-n',path)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertNotIn(b'\r',path.read_bytes())
        result = self.run_bash(ENTRY,'--dry-run')
        self.assertEqual(result.returncode,0,result.stderr)
        for expected in ('--gate FORMAL','--nproc_per_node=8','--micro-batch-size 8',
                         '--gradient-accumulation-steps 16','--max-optimizer-steps 9244',
                         '--warmup-steps 463','--save-every 500',
                         '--max-lr 6e-4','--min-lr 6e-5','--model-path '+INITIAL_MODEL,
                         'formal_reinit1000_lr6e4_'):
            self.assertIn(expected,result.stdout)
        self.assertNotIn('--resume-from',result.stdout)
        self.assertNotIn('--max-lr 8e-4',result.stdout)

    def test_explicit_model_override_remains_weights_only(self):
        result = self.run_bash(ENTRY,'--dry-run','--model-dir','/external/another/checkpoint-002000')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--model-path /external/another/checkpoint-002000',result.stdout)
        self.assertNotIn('--resume-from',result.stdout)

    def test_explicit_memory_fallback_and_invalid_batch(self):
        result = self.run_bash(ENTRY,'--dry-run','--micro-batch-size','4','--gradient-accumulation-steps','32')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--micro-batch-size 4',result.stdout)
        self.assertIn('--gradient-accumulation-steps 32',result.stdout)
        result = self.run_bash(ENTRY,'--dry-run','--micro-batch-size','4')
        self.assertNotEqual(result.returncode,0)

    def test_short_and_resume_contracts(self):
        result = self.run_bash(ENTRY,'--dry-run','--gate','D')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--max-optimizer-steps 10',result.stdout)
        self.assertIn('--scheduler-total-steps 9244',result.stdout)
        result = self.run_bash(ENTRY,'--dry-run','--gate','E')
        self.assertNotEqual(result.returncode,0)
        result = self.run_bash(ENTRY,'--dry-run','--resume-from','/external/run/checkpoint-005000')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--resume-from /external/run/checkpoint-005000',result.stdout)

    def test_stub_vc_receives_isolated_environment_and_resource_arguments(self):
        # Export a Bash function so even PATH resolution cannot reach real vc.
        with tempfile.TemporaryDirectory() as tmp:
            command = 'vc() { printf "VC_ARG:%s\\n" "$@"; }; export -f vc; bash "$TEST_ENTRY" --output-dir "/external/new run"'
            result = self.run_bash('-c',command,env={
                'TEST_ENTRY':ENTRY.as_posix(), PREFIX+'JOB_LOG_DIR':Path(tmp).as_posix(),
                'RSMOL_5_10X2_5_MESH_4SLOT_RESUME_FROM':'/foreign/checkpoint',
                'RSMOL_MIZAR_TEXT_5_10X2_5_MODEL_DIR':'/foreign/model',
            })
            self.assertEqual(result.returncode,0,result.stderr)
            arguments = [line.removeprefix('VC_ARG:') for line in result.stdout.splitlines() if line.startswith('VC_ARG:')]
            self.assertEqual(arguments[:4],['submit','-p','pdgpu-5090','-i'])
            self.assertIn('256G',arguments)
            cmd = arguments[arguments.index('--cmd')+1]
            self.assertIn(PREFIX+'RESUME_FROM=',cmd)
            self.assertIn(PREFIX+'MODEL_DIR=',cmd)
            self.assertIn(INITIAL_MODEL,cmd)
            self.assertNotIn('/foreign',cmd)
            self.assertIn('/external/new\\ run',cmd)


if __name__ == '__main__':
    unittest.main()
