"""Real generation/submission wrappers with synthetic JSON/audio paths and fake vc/conda."""
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
from mizar_s1_ced_reasonaqa5000_eval import config, submit, inspect_checkpoint
import test_mizar_s1_ced_mmsu_shell as old_shell
from test_reasonaqa5000_eval import write_pool


class JobNameTests(unittest.TestCase):
    def test_all_thirty_actual_commands_fit_vc_limit_and_keep_distinct_targets(self):
        items = submit.planned(submit.parse_args(['all']))
        self.assertEqual(len(items), 30)
        for tag in ('20261115_233059_123456_12345',
                    '20261115_233059_999999_12345678901234567890'):
            jobs = []
            for name, argv, resolved in items:
                args = submit.command(name, argv, tag)
                job = args[args.index('-j') + 1]
                self.assertLessEqual(len(job), 60, (name, job))
                self.assertRegex(job, r'^[a-zA-Z0-9][a-zA-Z0-9-]*$')
                self.assertEqual(args[args.index('-p') + 1], 'pdgpu-4090')
                jobs.append(job)
            self.assertEqual(len(set(jobs)), 30)

    def test_shared_long_target_prefixes_and_different_batches_remain_distinct(self):
        prefix = 'serial84_attn_shared_mlp_shared_modality_' * 2
        first, second = prefix + 'r8', prefix + 'r16'
        jobs = [submit.job_name(name, tag) for name, tag in
                ((first, 'batch1'), (second, 'batch1'), (first, 'batch2'))]
        self.assertEqual(len(set(jobs)), 3)
        self.assertTrue(all(len(job) <= 60 for job in jobs))
        self.assertEqual(jobs[0], submit.job_name(first, 'batch1'))


class ShellTests(unittest.TestCase):
    def setUp(self):
        git = Path('C:/Program Files/Git/bin/bash.exe')
        self.bash = str(git) if os.name == 'nt' and git.is_file() else shutil.which('bash')
        if not self.bash:
            self.skipTest('Bash unavailable')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path, self.rs, self.env = old_shell.ShellTests().fixture(self.temp.name)
        shutil.copytree(config.PACKAGE, self.rs / config.PACKAGE.name)
        for name in ('generate_reasonaqa_test5000.sh', 'run_reasonaqa5000_eval_5090.sh'):
            shutil.copy2(config.ROOT / name, self.rs / name)
        Path(self.env['BASH_ENV']).write_text('vc() { printf "%s\\0" "$@" >> "$MMSU_VC_CAPTURE"; printf "\\0" >> "$MMSU_VC_CAPTURE"; }\nexport -f vc\n',
                                             encoding='utf-8', newline='\n')
        git_bin = str(Path(self.bash).parent)
        self.env['PATH'] = git_bin + os.pathsep + self.env['PATH']

    def launch(self, script, *argv):
        return subprocess.run([self.bash, str(self.rs / script), *argv], cwd=self.rs, env=self.env,
                              capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)

    def test_all_thirty_real_parsers_exclusions_exact_paths_and_steps_without_assets(self):
        result = self.launch('run_reasonaqa5000_eval_5090.sh', 'all', '--check-args')
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([r['target'] for r in rows], list(config.TARGETS))
        self.assertEqual(len({r['output_dir'] for r in rows}), 30)
        self.assertFalse({r['eval_variant'] for r in rows} & set(config.EXCLUDED))
        for row in rows:
            spec = config.TARGETS[row['target']]
            self.assertEqual(row['expected_step'], spec['expected_step'])
            self.assertEqual(row['expected_geometry'], list(spec['expected_geometry']))
            self.assertTrue(row['checkpoint'].replace('\\', '/').endswith(
                spec['run_root'].as_posix().split('outputs/RSmol/')[-1] + f'/formal/checkpoints/model--step-{spec["expected_step"]}.ckpt'))
        self.assertFalse((self.path / 'submission.bin').exists())
        self.assertEqual((self.path / 'environment.bin').read_bytes().split(b'\0')[:-1], [b'1', b'1', b'false', b'utf-8'])

    def test_actual_dataset_generator_then_real_vc_submission_with_spaces(self):
        source = write_pool(self.path)
        bundle = self.path / 'sample bundle'
        made = self.launch('generate_reasonaqa_test5000.sh', '--test-json', str(source), '--output-dir', str(bundle))
        self.assertEqual(made.returncode, 0, made.stderr)
        self.assertEqual(json.loads(made.stdout)['sample_rows'], 5000)
        for root, name in [('caps', 'test/example.wav'), ('clotho', 'evaluation/Rain fall.wav'), ('aqa', 'dripping.wav')]:
            path = self.path / root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'audio path fixture')
        args = ['r515_inject84', '--manifest', str(bundle / 'manifest.json'),
                '--audiocaps-root', str(self.path / 'caps'), '--clotho-root', str(self.path / 'clotho'),
                '--clotho-aqa-root', str(self.path / 'aqa'), '--output-dir', "/output dir/model's eval"]
        result = self.launch('run_reasonaqa5000_eval_5090.sh', *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(summary['submitted'], 1)
        data = (self.path / 'submission.bin').read_bytes().decode().split('\0\0')[0].split('\0')
        for option, value in (('-p', 'pdgpu-4090'), ('-g', '1'), ('-c', '8'), ('-m', '32G'), ('-n', '1'),
                              ('-i', 'docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1')):
            self.assertEqual(data[data.index(option) + 1], value)
        self.assertLessEqual(len(data[data.index('-j') + 1]), 60)
        command = shlex.split(data[data.index('--cmd') + 1])
        self.assertEqual(command[:3], ['bash', 'mizar_s1_ced_reasonaqa5000_eval/run.sh', 'evaluate'])
        self.assertIn("/output dir/model's eval", [part.replace('\\', '/') for part in command])
        self.assertIn(str(bundle / 'manifest.json'), command)
        # Follow the actual submitted command through the real compute shell without GPU execution.
        compute = self.launch(command[1], 'evaluate', '--check-args', *command[3:])
        self.assertEqual(compute.returncode, 0, compute.stderr)
        self.assertEqual(json.loads(compute.stdout)['expected_step'], 17313)

    def test_rejects_excluded_targets_overrides_and_missing_manifest_before_vc(self):
        for argv in [('official_mizar',), ('base',), ('serial84_lora_r8',),
                     ('all', '--output-dir', '/shared'), ('r515_inject84', '--checkpoint', '/wrong'),
                     ('r515_inject84', '--manifest', str(self.path / 'missing.json'))]:
            with self.subTest(argv=argv):
                result = self.launch('run_reasonaqa5000_eval_5090.sh', *argv)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.path / 'submission.bin').exists())


class BatchTests(unittest.TestCase):
    def test_all_thirty_submission_commands_and_candidate_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'submissions.json'
            with mock.patch.object(submit.sample, 'load_manifest', return_value={'records': [], 'manifest_sha256': 'fixture'}), \
                 mock.patch.object(submit.dataset, 'prepare_records'), \
                 mock.patch.object(inspect_checkpoint, 'inspect', return_value={'status': 'PASS_CPU_CONTRACT_ONLY'}) as candidate, \
                 mock.patch.object(submit.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'fixture job', '')) as vc, \
                 contextlib.redirect_stdout(io.StringIO()):
                code = submit.main(['all', '--submission-report', str(report)])
            self.assertEqual(code, 0)
            self.assertEqual(vc.call_count, 30)
            candidate.assert_called_once()
            records = json.loads(report.read_text())
            self.assertEqual(records['submitted'], 30)
            self.assertEqual({item['target'] for item in records['targets']}, set(config.TARGETS))
            for item in records['targets']:
                args = item['vc_args']
                command = shlex.split(args[args.index('--cmd') + 1])
                self.assertEqual(command[command.index('--target') + 1], item['target'])
                self.assertEqual(args[args.index('-p') + 1], 'pdgpu-4090')
                self.assertLessEqual(len(args[args.index('-j') + 1]), 60)

    def test_candidate_and_vc_failures_continue_remaining_jobs_and_return_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'submissions.json'
            count = 0
            def fake_vc(*args, **kwargs):
                nonlocal count
                count += 1
                return subprocess.CompletedProcess([], 5 if count == 1 else 0, '', 'fixture refusal' if count == 1 else '')
            with mock.patch.object(submit.sample, 'load_manifest', return_value={'records': [], 'manifest_sha256': 'fixture'}), \
                 mock.patch.object(submit.dataset, 'prepare_records'), \
                 mock.patch.object(inspect_checkpoint, 'inspect', side_effect=ValueError('candidate wrong route')), \
                 mock.patch.object(submit.subprocess, 'run', side_effect=fake_vc), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = submit.main(['all', '--submission-report', str(report)])
            self.assertEqual(code, 1)
            self.assertEqual(count, 29)
            saved = json.loads(report.read_text())
            self.assertEqual(saved['failed'], 2)
            self.assertEqual(saved['submitted'], 28)
            self.assertEqual(saved['targets'][-1]['status'], 'SUBMITTED')
            candidate = next(item for item in saved['targets'] if item['target'] == 'mesh_inject48')
            self.assertFalse(candidate['vc_attempted'])
            self.assertIn('wrong route', candidate['error'])


if __name__ == '__main__':
    unittest.main()
