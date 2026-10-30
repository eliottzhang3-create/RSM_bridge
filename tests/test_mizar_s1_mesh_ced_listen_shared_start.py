"""Reuse the mother's config, concurrent staging and cleanup behavioral fixtures."""
import ast
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer36912_listen"
sys.path.insert(0, str(ROUTE))
import write_config
from contract import CED_ASSET_FILES, CED_EXPECTED_CONFIG, SOURCE_COUNTS, write_json

source = (ROOT / "tests/test_mizar_s1_recursive_mesh_ced_small_shared_start.py").read_text(encoding="utf-8")
tree = ast.parse(source)
fixture = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fixture")
original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CEDSharedStartTests")
original.name = "ListenSharedStartTests"
original.body = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in {
    "test_formal_config_is_created_with_zero_staged_files",
    "test_runner_starts_real_training_during_copy_and_cleans_on_failure"}]
exec(compile(ast.Module(body=[fixture, original], type_ignores=[]), "<mother-shared-start-listen>", "exec"), globals())

if __name__ == "__main__":
    unittest.main()
