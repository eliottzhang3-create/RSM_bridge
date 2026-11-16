"""Real Bash entry points and Python dispatcher, with vc/conda explicitly faked."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import test_mizar_s1_ced_mmsu_shell as original_shell

RSMOL = original_shell.RSMOL


class ShellTests(unittest.TestCase):
    def setUp(self):
        self.helper = original_shell.ShellTests()
        self.helper.setUp()
        self.bash = self.helper.bash

    def fixture(self, directory):
        path, rs, env = self.helper.fixture(directory)
        shutil.copytree(RSMOL / 'mizar_s1_ced_mmsu_remaining_eval', rs / 'mizar_s1_ced_mmsu_remaining_eval')
        shutil.copy2(RSMOL / 'run_mizar_mmsu_remaining_4090.sh', rs)
        if os.name == 'nt':
            env['PATH'] = str(Path(self.bash).parent) + os.pathsep + env['PATH']
        return path, rs, env

    def launch(self, rs, env, args, package=False):
        entry = rs / ('mizar_s1_ced_mmsu_remaining_eval/run.sh' if package else 'run_mizar_mmsu_remaining_4090.sh')
        return subprocess.run([self.bash, str(entry), *args], cwd=rs, env=env,
                              capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=40)

    def test_actual_all27_check_args_resources_and_exclusions_without_writes_or_vc(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            result = self.launch(rs, env, ['all', '--check-args'])
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(rows), 27)
            from mizar_s1_ced_mmsu_remaining_eval import config
            self.assertEqual({row['eval_variant'] for row in rows}, set(config.VARIANT_TARGETS))
            self.assertTrue(set(config.EXCLUDED).isdisjoint(row['eval_variant'] for row in rows))
            for row in rows:
                command = row['vc_args']
                for key, value in (('-p', 'pdgpu-4090'), ('-g', '1'), ('-c', '8'), ('-m', '32G'), ('-n', '1')):
                    self.assertEqual(command[command.index(key) + 1], value)
                self.assertLessEqual(len(command[command.index('-j') + 1]), 52)
            self.assertFalse((path / 'submission.bin').exists())
            self.assertFalse((rs / 'log').exists())

    def test_real_single_submit_quoted_paths_original_compute_and_persistent_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            options = ['ca36912', '--mmsu-root', "/MMSU data's root", '--output-dir', str(path / 'chosen output'),
                       '--ledger-dir', str(path / 'ledger')]
            result = self.launch(rs, env, [*options, '--submission-report', str(path / 'first.json')])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            captured = (path / 'submission.bin').read_bytes()
            command = captured.decode().split('\0')[:-1]
            for key, value in (('-p', 'pdgpu-4090'), ('-g', '1'), ('-c', '8'), ('-m', '32G'), ('-n', '1')):
                self.assertEqual(command[command.index(key) + 1], value)
            compute = shlex.split(command[command.index('--cmd') + 1])
            self.assertEqual(compute[:2], ['bash', 'mizar_s1_ced_mmsu_eval/run.sh'])
            self.assertEqual(compute[compute.index('--mmsu-root') + 1].replace('\\', '/'), "/MMSU data's root")
            checked = subprocess.run([self.bash, str(rs / compute[1]), *compute[2:], '--check-args'],
                                     cwd=rs, env=env, capture_output=True, text=True, encoding='utf-8', timeout=30)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            resolved = json.loads(checked.stdout)
            self.assertTrue(resolved['checkpoint'].replace('\\', '/').endswith(
                'formal_20261030_140459065784533/formal/checkpoints/model--step-17313.ckpt'))
            self.assertTrue(resolved['runtime_config'].replace('\\', '/').endswith(
                'formal_20261030_140459065784533/formal/runtime.json'))
            self.assertEqual(resolved['expected_geometry'], [8, 8, 4])
            self.assertEqual((path / 'environment.bin').read_bytes().split(b'\0')[:-1],
                             [b'1', b'1', b'false', b'utf-8'])
            stamp = (path / 'submission.bin').stat().st_mtime_ns
            result = self.launch(rs, env, [*options, '--submission-report', str(path / 'second.json')])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            second = json.loads((path / 'second.json').read_text())
            self.assertEqual(second['targets'][0]['status'], 'SKIPPED_SUBMITTED')
            self.assertEqual((path / 'submission.bin').stat().st_mtime_ns, stamp)
            self.assertFalse((path / 'chosen output').exists())

    def test_invalid_targets_and_inspect_never_call_vc(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rs, env = self.fixture(directory)
            for options in (['base'], ['official_mizar'], ['all', '--retry-submitted'],
                            ['ca36912', '--checkpoint', '/wrong']):
                result = self.launch(rs, env, options)
                self.assertNotEqual(result.returncode, 0)
            result = self.launch(rs, env, ['inspect', 'ca36912', '--output-dir', str(path / 'absent'),
                                          '--submission-report', str(path / 'inspect.json')], package=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads((path / 'inspect.json').read_text())['targets'][0]['status'], 'NEW')
            self.assertFalse((path / 'absent').exists())
            self.assertFalse((rs / 'log/mmsu_remaining4090_admissions').exists())
            self.assertFalse((path / 'submission.bin').exists())


if __name__ == '__main__':
    unittest.main()
