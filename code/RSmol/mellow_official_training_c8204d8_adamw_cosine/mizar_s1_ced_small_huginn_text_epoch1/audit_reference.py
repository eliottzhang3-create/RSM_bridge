"""Read-only local comparison with the exact CED-Small audio parent.

This audit uses source/recipe checks only. Production asset loading, Torch
autograd and eight-rank training require the remote runtime. It is never
invoked by the formal launcher.
"""
import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path

import contract

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small")


def find_function(tree, name):
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one function {name}, got {len(matches)}")
    return matches[0]


def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "actual-parent-function", "exec"), scope)


def audit_parent():
    changed = {".gitignore", "contract.py", "models/recursive_text.py", "write_config.py", "run.sh",
               "audit_reference.py", "README.md", "reference_audit.json"}
    identical = []
    for source in sorted(PARENT.rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        name = source.relative_to(PARENT).as_posix()
        if name in changed:
            continue
        if source.read_bytes() != (ROOT / name).read_bytes():
            raise ValueError(f"Parent implementation changed: {name}")
        identical.append(name)
    original_huginn = (contract.RSMOL_ROOT / "recursive_model_mizar_text_5_10x2_5_huginn.py").read_text(encoding="utf-8")
    expected_huginn = original_huginn.replace(
        "from code.RSmol.mizar_text_5_10x2_5_huginn.contract import",
        "from mizar_text_5_10x2_5_huginn.contract import")
    if expected_huginn != (ROOT / "models/huginn.py").read_text(encoding="utf-8"):
        raise ValueError("Isolated Huginn differs from the trained text implementation")
    spec = importlib.util.spec_from_file_location("_huginn_audio_parent_contract", PARENT / "contract.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    prepared = dict(rows=1477492, source_rows=contract.SOURCE_COUNTS,
                    rows_canonical_sha256="synthetic-rows", mapping_sha256="synthetic-mapping")
    actual = contract.semantic_contract(prepared, "synthetic-text", "synthetic-ced")
    original = old.semantic_contract(prepared, "synthetic-text", "synthetic-ced")
    allowed = {"route", "code_sha256", "recursive_model_source_sha256", "recursive_model_sources",
               "text_model_contract", "text_model_architecture", "huginn"}
    for key in set(actual) | set(original):
        if key not in allowed and actual.get(key) != original.get(key):
            raise ValueError(f"Audio recipe changed: {key}")
    return {"byte_identical_files": identical, "semantic_changes_allowed": sorted(allowed),
            "audio_recipe": "identical to exact parent except Huginn identity and initialization",
            "huginn_model": "identical to text model except dependency import path"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "PASS_LOCAL_SOURCE_ONLY", "comparison": audit_parent(),
              "code_sha256": contract.code_digest(),
              "remote_assets_read": False, "gpu_training_verified": False}
    contract.write_json(args.output, result)


if __name__ == "__main__":
    main()
