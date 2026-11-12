"""Run the real Bash assignment through production startup/cleanup guards.

POSIX filesystem queries and rmtree are explicit fixtures on Windows. No
real /dev/shm directory is created or removed; guard bodies are unchanged.
"""
import ast
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import subprocess
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_ced_small_huginn_text_epoch1"


class HuginnStagingRootTests(unittest.TestCase):
    def scope(self, *, present=False, symlinks=(), resolutions=None, owner=None):
        class PosixPath(PurePosixPath):
            def is_symlink(self):
                return str(self) in symlinks

            def resolve(self):
                value = (resolutions or {}).get(str(self), posixpath.normpath(str(self)))
                return type(self)(value)

            def exists(self):
                return present

            def read_text(self, **kwargs):
                return json.dumps(owner)

        tree = ast.parse((ROUTE / "stage_data.py").read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"validate_stage_root", "cleanup"}]
        self.assertEqual(len(nodes), 2)
        scope = {"Path": PosixPath, "json": json, "shutil": Mock()}
        exec(compile(ast.Module(nodes, type_ignores=[]), "actual-staging-guards", "exec"), scope)
        return scope

    def shell_root(self, token="formal_fixture_123"):
        bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
        self.assertTrue(bash and Path(bash).is_file(), "Bash is required for this regression")
        assignments = re.findall(r'^\s*(STAGE_ROOT="/dev/shm/[^"]+")\s*$',
                                 (ROUTE / "run.sh").read_text(encoding="utf-8"), re.M)
        self.assertEqual(len(assignments), 1)
        script = 'set -eu\nSTAGE_TOKEN=' + token + '\n' + assignments[0]
        script += '\nprintf "%s\\n" "$STAGE_ROOT"\n'
        result = subprocess.run([bash, "-c", script], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_actual_shell_root_passes_startup_and_cleanup_and_keeps_unique_tokens(self):
        roots = []
        for token in ("formal_fixture_123", "formal_fixture_456"):
            root = self.shell_root(token)
            roots.append(root)
            self.assertIn("huginn_text_epoch1_", root)
            scope = self.scope()
            self.assertEqual(str(scope["validate_stage_root"](root)), root)
            scope["cleanup"](root, token)
            scope["shutil"].rmtree.assert_not_called()
            scope = self.scope(present=True, owner={"token": token, "root": root})
            scope["cleanup"](root, token)
            scope["shutil"].rmtree.assert_called_once_with(scope["Path"](root))
        self.assertEqual(len(set(roots)), len(roots))

    def test_previous_isolated_name_reproduces_both_guard_failures(self):
        root = "/dev/shm/mizar_s1_ced_small_huginn_text_epoch1_formal_fixture_123"
        scope = self.scope()
        with self.assertRaisesRegex(ValueError, "nonsymlink direct child"):
            scope["validate_stage_root"](root)
        with self.assertRaisesRegex(ValueError, "nonsymlink direct child"):
            scope["cleanup"](root, "formal_fixture_123")
        scope["shutil"].rmtree.assert_not_called()

    def test_unsafe_paths_symlinks_and_resolved_aliases_remain_rejected(self):
        root = "/dev/shm/mizar_s1_gbs256_huginn_fixture"
        cases = (
            ("relative/mizar_s1_gbs256_fixture", {}),
            ("/tmp/mizar_s1_gbs256_fixture", {}),
            (root + "/nested", {}),
            ("/dev/shm/wrong_prefix_fixture", {}),
            ("/dev/shm/mizar_s1_gbs256_parent/../mizar_s1_gbs256_fixture", {}),
            (root, {"symlinks": (root,)}),
            (root, {"resolutions": {root: "/tmp/alias"}}),
        )
        for value, options in cases:
            with self.subTest(root=value, options=options):
                scope = self.scope(**options)
                with self.assertRaises(ValueError):
                    scope["validate_stage_root"](value)
                with self.assertRaises(ValueError):
                    scope["cleanup"](value, "fixture")
                scope["shutil"].rmtree.assert_not_called()

    def test_cleanup_requires_matching_token_and_root(self):
        root = self.shell_root()
        for owner in ({"token": "other", "root": root},
                      {"token": "fixture", "root": "/dev/shm/mizar_s1_gbs256_other"}):
            with self.subTest(owner=owner):
                scope = self.scope(present=True, owner=owner)
                with self.assertRaisesRegex(ValueError, "does not own"):
                    scope["cleanup"](root, "fixture")
                scope["shutil"].rmtree.assert_not_called()


if __name__ == "__main__":
    unittest.main()
