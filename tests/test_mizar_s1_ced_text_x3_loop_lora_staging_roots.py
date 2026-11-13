"""Exercise run.sh's actual tmpfs assignment against staging's POSIX guards.

Only Path filesystem queries and rmtree are fixtures on Windows. The shell
assignment, validate_stage_root and cleanup bodies come from production files.
No /dev/shm directory is created and no real directory is removed.
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
FAMILY = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine"
ROUTES = (FAMILY / "mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8",)


class StagingRootTests(unittest.TestCase):
    def scope(self, route, *, present=False, symlinks=(), resolutions=None, owner=None):
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

        tree = ast.parse((route / "stage_data.py").read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"validate_stage_root", "cleanup"}]
        self.assertEqual(len(nodes), 2)
        scope = {"Path": PosixPath, "json": json, "shutil": Mock()}
        exec(compile(ast.Module(nodes, type_ignores=[]), "actual-staging-guards", "exec"), scope)
        return scope

    def shell_root(self, route):
        bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
        self.assertTrue(bash and Path(bash).is_file(), "Bash is required for this regression")
        assignments = re.findall(r'^\s*(STAGE_ROOT="/dev/shm/[^"]+")\s*$',
                                 (route / "run.sh").read_text(encoding="utf-8"), re.M)
        self.assertEqual(len(assignments), 1)
        script = 'set -eu\nSTAGE_TOKEN=formal_fixture_123\n' + assignments[0]
        script += '\nprintf "%s\\n" "$STAGE_ROOT"\n'
        # Prevent Git Bash from mapping /dev/shm into a Windows path on output.
        result = subprocess.run([bash, "-c", script], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_real_shell_roots_pass_startup_and_cleanup_guards_and_remain_distinct(self):
        roots = []
        for route in ROUTES:
            with self.subTest(route=route.name):
                root = self.shell_root(route)
                roots.append(root)
                scope = self.scope(route)
                self.assertEqual(str(scope["validate_stage_root"](root)), root)
                scope["cleanup"](root, "formal_fixture_123")  # absent root is safe
                scope["shutil"].rmtree.assert_not_called()

                owner = {"token": "formal_fixture_123", "root": root}
                scope = self.scope(route, present=True, owner=owner)
                scope["cleanup"](root, owner["token"])
                scope["shutil"].rmtree.assert_called_once_with(scope["Path"](root))
        self.assertEqual(len(set(roots)), len(ROUTES))

    def test_old_isolated_names_reproduce_reported_startup_and_cleanup_failure(self):
        roots = (
            "/dev/shm/mizar_s1_ced_small_text_epoch1_formal_fixture",
            "/dev/shm/mizar_s1_ced_text_e1_in48_loop_lora_r8_formal_fixture",
        )
        for route in ROUTES:
            scope = self.scope(route)
            for root in roots:
                with self.subTest(route=route.name, root=root):
                    with self.assertRaisesRegex(ValueError, "nonsymlink direct child"):
                        scope["validate_stage_root"](root)
                    with self.assertRaises(ValueError):
                        scope["cleanup"](root, "fixture")
                    scope["shutil"].rmtree.assert_not_called()

    def test_unsafe_paths_symlinks_and_resolved_aliases_remain_rejected(self):
        root = "/dev/shm/mizar_s1_gbs256_fixture"
        for route in ROUTES:
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
                with self.subTest(route=route.name, root=value, options=options):
                    scope = self.scope(route, **options)
                    with self.assertRaises(ValueError):
                        scope["validate_stage_root"](value)
                    with self.assertRaises(ValueError):
                        scope["cleanup"](value, "fixture")
                    scope["shutil"].rmtree.assert_not_called()

    def test_cleanup_still_requires_matching_token_and_root(self):
        for route in ROUTES:
            root = self.shell_root(route)
            for owner in ({"token": "other", "root": root},
                          {"token": "fixture", "root": "/dev/shm/mizar_s1_gbs256_other"}):
                with self.subTest(route=route.name, owner=owner):
                    scope = self.scope(route, present=True, owner=owner)
                    with self.assertRaisesRegex(ValueError, "does not own"):
                        scope["cleanup"](root, "fixture")
                    scope["shutil"].rmtree.assert_not_called()


if __name__ == "__main__":
    unittest.main()

