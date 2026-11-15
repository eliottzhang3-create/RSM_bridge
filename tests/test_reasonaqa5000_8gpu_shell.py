"""Actual Bash submission/compute entry points with fake vc/conda, no GPU jobs."""
from __future__ import annotations

import contextlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code/RSmol'), str(ROOT / 'tests')]
from mizar_s1_ced_reasonaqa5000_8gpu_eval import config, submit
from mizar_s1_ced_reasonaqa5000_eval import inspect_checkpoint
import test_mizar_s1_ced_mmsu_shell as mmsu_shell
from test_reasonaqa5000_eval import write_pool


class ShellTests(unittest.TestCase):
    def setUp(self):
        git = Path('C:/Program Files/Git/bin/bash.exe')
        self.bash = str(git) if os.name == 'nt' and git.is_file() else shutil.which('bash')
        if not self.bash:
            self.skipTest('Bash unavailable')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path, self.rs, self.env = mmsu_shell.ShellTests().fixture(self.temp.name)
        for name in ('mizar_s1_ced_reasonaqa5000_eval', config.PACKAGE.name):
            shutil.copytree(config.ROOT / name, self.rs / name)
        for name in ('generate_reasonaqa_test5000.sh', 'run_reasonaqa5000_eval_8gpu_3090.sh'):
            shutil.copy2(config.ROOT / name, self.rs / name)
        Path(self.env['BASH_ENV']).write_text(
            'vc() { local prev="" arg; for arg in "$@"; do '
            'if [[ "$prev" == -j && ${#arg} -gt 60 ]]; then echo "job exceeds 60" >&2; return 64; fi; '
            'prev="$arg"; done; printf "%s\\0" "$@" >> "$MMSU_VC_CAPTURE"; printf "\\0" >> "$MMSU_VC_CAPTURE"; }\n'
            'export -f vc\n', encoding='utf-8', newline='\n')
        self.env['PATH'] = str(Path(self.bash).parent) + os.pathsep + self.env['PATH']

    def launch(self, script, *argv):
        return subprocess.run([self.bash, str(self.rs / script), *argv], cwd=self.rs, env=self.env,
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)

    def test_actual_all_thirty_check_args_show_eight_gpu_3090_and_new_outputs(self):
        result = self.launch('run_reasonaqa5000_eval_8gpu_3090.sh', 'all', '--check-args')
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([row['target'] for row in rows], list(config.TARGETS))
        self.assertEqual(len({row['output_dir'] for row in rows}), 30)
        for row in rows:
            args = row['vc_args']
            for flag, value in (('-p', 'pdgpu-3090'), ('-g', '8'), ('-c', '64'), ('-m', '256G'), ('-n', '1')):
                self.assertEqual(args[args.index(flag) + 1], value)
            self.assertLessEqual(len(args[args.index('-j') + 1]), 60)
            self.assertTrue(row['output_dir'].endswith('_8gpu_v1'))
            self.assertEqual(row['parallel_evaluation']['rows_per_rank'], 625)
        self.assertFalse((self.path / 'submission.bin').exists())

    def test_generate_then_actual_submit_and_compute_parser_keep_spaces_and_quotes(self):
        source = write_pool(self.path)
        bundle = self.path / 'sample bundle'
        result = self.launch('generate_reasonaqa_test5000.sh', '--test-json', str(source), '--output-dir', str(bundle))
        self.assertEqual(result.returncode, 0, result.stderr)
        for directory, name in (('caps', 'test/example.wav'), ('clotho', 'evaluation/Rain fall.wav'), ('aqa', 'dripping.wav')):
            path = self.path / directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'path fixture only')
        argv = ['r515_inject84', '--manifest', str(bundle / 'manifest.json'), '--output-dir', "/output dir/model's eight GPU eval",
                '--audiocaps-root', str(self.path / 'caps'), '--clotho-root', str(self.path / 'clotho'),
                '--clotho-aqa-root', str(self.path / 'aqa')]
        result = self.launch('run_reasonaqa5000_eval_8gpu_3090.sh', *argv)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])['submitted'], 1)
        captured = (self.path / 'submission.bin').read_bytes().decode().split('\0\0')[0].split('\0')
        self.assertEqual(captured[captured.index('-p') + 1], 'pdgpu-3090')
        self.assertEqual(captured[captured.index('-g') + 1], '8')
        self.assertLessEqual(len(captured[captured.index('-j') + 1]), 60)
        command = shlex.split(captured[captured.index('--cmd') + 1])
        self.assertEqual(command[:3], ['bash', 'mizar_s1_ced_reasonaqa5000_8gpu_eval/run.sh', 'evaluate'])
        self.assertIn("/output dir/model's eight GPU eval", [item.replace('\\', '/') for item in command])
        compute = self.launch(command[1], 'evaluate', '--check-args', *command[3:])
        self.assertEqual(compute.returncode, 0, compute.stderr)
        self.assertEqual(json.loads(compute.stdout)['parallel_evaluation'], config.PARALLEL)
        self.assertEqual((self.path / 'environment.bin').read_bytes().split(b'\0')[:-1], [b'1', b'1', b'false', b'utf-8'])

    def test_excluded_targets_and_shared_output_rejected_before_submission(self):
        for argv in (('official_mizar',), ('base',), ('all', '--output-dir', '/shared'),
                     ('init0904', '--manifest', str(self.path / 'missing.json'))):
            with self.subTest(argv=argv):
                result = self.launch('run_reasonaqa5000_eval_8gpu_3090.sh', *argv)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.path / 'submission.bin').exists())


class BatchTests(unittest.TestCase):
    def test_thirty_independent_jobs_and_candidate_failure_continue_other_twenty_nine(self):
        for candidate_error, expected in ((None, 30), (ValueError('candidate mismatch'), 29)):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                report = Path(directory) / 'submission.json'
                with mock.patch.object(submit.sample, 'load_manifest', return_value={'records': [], 'manifest_sha256': 'fixture'}), \
                     mock.patch.object(submit.dataset, 'prepare_records'), \
                     mock.patch.object(inspect_checkpoint, 'inspect', side_effect=candidate_error,
                                       return_value={'status': 'PASS_CPU_CONTRACT_ONLY'}) as inspect, \
                     mock.patch.object(submit.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'fixture', '')) as vc, \
                     contextlib.redirect_stdout(io.StringIO()):
                    status = submit.main(['all', '--submission-report', str(report)])
                self.assertEqual(status, 0 if expected == 30 else 1)
                self.assertEqual(vc.call_count, expected)
                inspect.assert_called_once()
                saved = json.loads(report.read_text())
                self.assertEqual(saved['submitted'], expected)
                self.assertEqual(saved['targets'][-1]['status'], 'SUBMITTED')
                for item in saved['targets']:
                    if item['status'] == 'SUBMITTED':
                        args = item['vc_args']
                        self.assertEqual(args[args.index('-g') + 1], '8')


if __name__ == '__main__':
    unittest.main()
