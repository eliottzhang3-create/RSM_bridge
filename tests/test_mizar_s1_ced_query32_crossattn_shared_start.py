"""Run the parent's config and producer-cleanup fixtures against the new route."""
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
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer36912_query32_crossattn"
sys.path.insert(0,str(ROUTE))
import write_config
from contract import CED_ASSET_FILES, CED_EXPECTED_CONFIG, SOURCE_COUNTS, write_json

parent = ast.parse((ROOT / "tests/test_mizar_s1_recursive_5_10_5_ced_small_shared_start.py").read_text(encoding="utf-8"))
fixture = next(node for node in parent.body if isinstance(node,ast.FunctionDef) and node.name=="fixture")
original = next(node for node in parent.body if isinstance(node,ast.ClassDef) and node.name=="CEDSharedStartTests")
methods = {"test_formal_config_is_created_with_zero_staged_files",
           "test_runner_starts_real_training_during_copy_and_cleans_on_failure"}
original.body = [node for node in original.body if isinstance(node,ast.FunctionDef) and node.name in methods]
original.name = "CrossAttentionSharedStartTests"
exec(compile(ast.Module(body=[fixture,original],type_ignores=[]),"<parent-fixtures-new-route>","exec"),globals())


if __name__ == "__main__":
    unittest.main()
