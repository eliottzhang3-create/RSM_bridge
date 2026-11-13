"""Read-only local comparison with the exact CED-Small audio parent.

This audit uses source/recipe checks only. Production asset loading, Torch
autograd and eight-rank training require the remote runtime. It is never
invoked by the formal launcher.
"""
import argparse
import ast
import copy
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


def verify_training_delta():
    """Remove only diagnostics and compare the entire original training AST."""
    class DiagnosticsOnly(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None if node.module == "gradient_diagnostics" else node

        def visit_If(self, node):
            if ast.unparse(node.test) in ("not gradient_audit['passed']", "tracker.last_anomaly is not None"):
                return None
            return self.generic_visit(node)

        def visit_Dict(self, node):
            node = self.generic_visit(node)
            pairs = [(k, v) for k, v in zip(node.keys, node.values)
                     if not (isinstance(k, ast.Constant) and k.value == "audio_boundary_normalization")]
            node.keys, node.values = [p[0] for p in pairs], [p[1] for p in pairs]
            return node

        def visit_Constant(self, node):
            if node.value == "nonfinite/emergency gradient norm; optimizer update aborted; see gradient_failure_rank*_step*.json":
                node.value = "nonfinite/emergency gradient norm; optimizer update aborted"
            return node

    original = ast.parse((PARENT / "train.py").read_text(encoding="utf-8"))
    actual = DiagnosticsOnly().visit(ast.parse((ROOT / "train.py").read_text(encoding="utf-8")))
    if ast.dump(original) != ast.dump(actual):
        raise ValueError("Training changed beyond boundary-norm metadata and failure diagnostics")


def verify_optimizer_delta():
    """The same FP32 norm arithmetic, clipping policy and optimizer groups."""
    original = ast.parse((PARENT / "optim.py").read_text(encoding="utf-8"))
    old_return = find_function(original, "track_and_clip_").body[-1]
    actual = ast.parse((ROOT / "optim.py").read_text(encoding="utf-8"))
    method = find_function(actual, "track_and_clip_")
    assignment = next(n for n in method.body if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == "total_norm")
    if ast.dump(assignment.value) != ast.dump(old_return.value.elts[0]):
        raise ValueError("GradNorm norm arithmetic changed")
    if ast.unparse(method.body[-1]) != "return (total_norm, scale)":
        raise ValueError("Unexpected GradNorm return")
    method.body[-1] = copy.deepcopy(old_return)

    class DiagnosticsOnly(ast.NodeTransformer):
        def visit_Import(self, node):
            return None if [n.name for n in node.names] == ["math"] else node

        def visit_ImportFrom(self, node):
            return None if node.module == "gradient_diagnostics" else node

        def visit_Assign(self, node):
            if ast.unparse(node.targets[0]) in ("self.last_anomaly", "total_norm"):
                return None
            return node

        def visit_If(self, node):
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "inspect_gradients" for n in ast.walk(node)):
                return None
            return self.generic_visit(node)

    if ast.dump(original) != ast.dump(DiagnosticsOnly().visit(actual)):
        raise ValueError("Optimizer/GradNorm changed beyond failure-only diagnostics")


def verify_huginn_delta():
    """Verify inherited Huginn computation apart from the explicit norm hooks."""
    original = (contract.RSMOL_ROOT / "recursive_model_mizar_text_5_10x2_5_huginn.py").read_text(encoding="utf-8")
    original = original.replace("from code.RSmol.mizar_text_5_10x2_5_huginn.contract import",
                                "from mizar_text_5_10x2_5_huginn.contract import")
    before = ast.parse(original)
    after = ast.parse((ROOT / "models/huginn.py").read_text(encoding="utf-8"))

    class NormHooksOnly(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None if node.module == "contract" else node

        def visit_FunctionDef(self, node):
            return None if node.name == "enable_recurrent_boundary_norms" else self.generic_visit(node)

        def visit_If(self, node):
            text = ast.unparse(node.test)
            if (text == "getattr(config, 'huginn_audio_boundary_norms', None) is not None"
                    or text in ("logical_index in (5, 15) and hasattr(self, 'recurrent_input_norm')",
                                "logical_index in (14, 24) and hasattr(self, 'recurrent_output_norm')")):
                return None
            return self.generic_visit(node)

    # The top-level architecture docstring describes the added audio norms.
    before.body = before.body[1:]
    after.body = after.body[1:]
    if ast.dump(before) != ast.dump(NormHooksOnly().visit(after)):
        raise ValueError("Huginn computation changed beyond audio boundary norms")


def audit_parent():
    changed = {".gitignore", "contract.py", "models/recursive_text.py", "write_config.py", "run.sh",
               "audit_reference.py", "README.md", "reference_audit.json", "train.py", "optim.py"}
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
    verify_training_delta()
    verify_optimizer_delta()
    verify_huginn_delta()
    spec = importlib.util.spec_from_file_location("_huginn_audio_parent_contract", PARENT / "contract.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    prepared = dict(rows=1477492, source_rows=contract.SOURCE_COUNTS,
                    rows_canonical_sha256="synthetic-rows", mapping_sha256="synthetic-mapping")
    actual = contract.semantic_contract(prepared, "synthetic-text", "synthetic-ced")
    original = old.semantic_contract(prepared, "synthetic-text", "synthetic-ced")
    allowed = {"route", "code_sha256", "recursive_model_source_sha256", "recursive_model_sources",
               "text_model_contract", "text_model_architecture", "huginn", "audio_boundary_normalization"}
    for key in set(actual) | set(original):
        if key not in allowed and actual.get(key) != original.get(key):
            raise ValueError(f"Audio recipe changed: {key}")
    return {"byte_identical_files": identical, "semantic_changes_allowed": sorted(allowed),
            "training_ast_preserved_except_diagnostics": True, "optimizer_policy_ast_preserved": True,
            "huginn_ast_preserved_except_boundary_norms": True,
            "audio_recipe": "identical to exact parent except Huginn identity and initialization",
            "huginn_model": "text model with shared input/output audio RMSNorms"}


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
