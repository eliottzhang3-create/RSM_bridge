"""Execute real ReasonAQA Bash entrypoints; only vc/conda and assets are fixtures."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code/RSmol'), str(ROOT / 'tests')]
from mizar_s1_ced_reasonaqa_eval import config
import test_mizar_s1_ced_mmsu_shell as legacy_shell


class ReasonAQAShellTests(unittest.TestCase):
    def setUp(self):
        git_bash = Path('C:/Program Files/Git/bin/bash.exe')
        self.bash = str(git_bash) if os.name == 'nt' and git_bash.is_file() else shutil.which('bash')
        if not self.bash:
            self.skipTest('Bash unavailable')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path, self.rs, self.env = legacy_shell.ShellTests().fixture(self.temporary.name)
        shutil.copytree(config.ROOT / 'mizar_s1_ced_reasonaqa_eval', self.rs / 'mizar_s1_ced_reasonaqa_eval')
        for name in ('run_mizar_reasonaqa_selected7_5090.sh', 'run_mizar_s1_ced_small_reasonaqa_eval_5090.sh'):
            shutil.copy2(config.ROOT / name, self.rs / name)
        # Append capture rather than replacing so all seven real submissions are visible.
        Path(self.env['BASH_ENV']).write_text('vc() { printf "%s\\0" "$@" >> "$MMSU_VC_CAPTURE"; printf "\\0" >> "$MMSU_VC_CAPTURE"; }\nexport -f vc\n',
                                             encoding='utf-8', newline='\n')

    def launch(self, *args):
        return subprocess.run([self.bash, str(self.rs / 'run_mizar_reasonaqa_selected7_5090.sh'), *args],
            cwd=self.rs, env=self.env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=45)

    def captures(self):
        data = (self.path / 'submission.bin').read_bytes().decode()
        return [part.split('\0') for part in data.split('\0\0') if part]

    def test_all_seven_exact_targets_resources_and_quoted_options(self):
        result = self.launch('all', '--test-json', "/data dir/test's.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        self.assertEqual([args['target'] for args in parsed], list(config.TARGETS))
        submissions = self.captures()
        self.assertEqual(len(submissions), 7)
        for alias, submit, args in zip(config.TARGETS, submissions, parsed):
            for option, value in (('-p', 'pdgpu-5090'), ('-g', '1'), ('-c', '8'), ('-m', '32G'), ('-n', '1'),
                                  ('-i', 'docker.v2.aispeech.com/sjtu/sjtu_wumengyue-mhl:0.0.1')):
                self.assertEqual(submit[submit.index(option) + 1], value)
            command = shlex.split(submit[submit.index('--cmd') + 1])
            self.assertEqual(command, ['bash', 'mizar_s1_ced_reasonaqa_eval/run.sh', '--target', alias,
                                       '--test-json', "/data dir/test's.json"])
            self.assertEqual(args['expected_step'], 17313)
            self.assertEqual(args['expected_geometry'], [8, 8, 4])
            normalized = args['checkpoint'].replace('\\', '/')
            expected = config.TARGETS[alias]['run_root'].as_posix().split('outputs/RSmol/')[-1]
            self.assertTrue(normalized.endswith(expected + '/formal/checkpoints/model--step-17313.ckpt'))
        self.assertEqual((self.path / 'environment.bin').read_bytes().split(b'\0')[:-1],
                         [b'1', b'1', b'false', b'utf-8'])

    def test_check_only_no_submit_no_assets_and_all_outputs_independent(self):
        result = self.launch('all', '--check-args')
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(rows), 7)
        self.assertEqual(len({r['output_dir'] for r in rows}), 7)
        self.assertFalse((self.path / 'submission.bin').exists())

    def test_single_custom_output_and_identity_override_rejections(self):
        result = self.launch('base', '--output-dir', '/output dir/reasonaqa', '--clotho-aqa-root=/audio dir/aqa')
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(self.captures()[0][self.captures()[0].index('--cmd') + 1])
        self.assertIn('/output dir/reasonaqa', command)
        for args in [('all', '--output-dir', '/same'), ('base', '--checkpoint', '/wrong'),
                     ('base', '--eval-variant=base'), ('unknown',), ('base', '--test-json'), ('base', '--test-json=')]:
            with self.subTest(args=args):
                (self.path / 'submission.bin').unlink(missing_ok=True)
                result = self.launch(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.path / 'submission.bin').exists())


if __name__ == '__main__':
    unittest.main()
