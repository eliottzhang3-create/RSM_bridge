"""Audit the isolated split3+gates route against its unmodified split3 parent."""
import argparse
import ast
import copy
import difflib
import hashlib
from pathlib import Path

from contract import SOURCE_COUNTS, code_digest, semantic_contract, write_json

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.with_name(ROOT.name.removesuffix("_inject"))


def tree(path):
    text = path.read_text(encoding="utf-8-sig")
    return ast.parse(text.replace("audio7slot_split3_inject", "audio7slot_split3"))


def find_function(module, name):
    matches = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one function {name}, got {len(matches)}")
    return matches[0]


def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "split3-inject-audit", "exec"), scope)


def equal(a, b, label):
    if ast.dump(a) != ast.dump(b):
        diff = "\n".join(difflib.unified_diff(ast.unparse(b).splitlines(), ast.unparse(a).splitlines()))
        raise ValueError(f"unexpected split3 difference: {label}\n{diff}")


def audit_parent():
    changed = {"contract.py", "optim.py", "train.py", "write_config.py", "audit_reference.py",
               "models/recursive_text.py", "models/recursive_mesh_listen.py"}
    unchanged = []
    for path in sorted(ROOT.rglob("*")):
        name = path.relative_to(ROOT).as_posix()
        if path.suffix not in {".py", ".sh"} or name in changed:
            continue
        expected = (PARENT/name).read_bytes().replace(b"\r\n", b"\n")
        actual = path.read_bytes().replace(b"\r\n", b"\n").replace(b"audio7slot_split3_inject", b"audio7slot_split3")
        if actual != expected:
            raise ValueError(f"unexpected data/mapper/interface/startup change: {name}")
        unchanged.append(name)
    # The entire train loop, optimizer steps, checkpoints, diagnostics schedule,
    # and restoration remain identical; only the reported new count differs.
    actual = ast.parse((ROOT/"train.py").read_text(encoding="utf-8")
                       .replace("audio7slot_split3_inject", "audio7slot_split3")
                       .replace("2469164", "2468010"))
    expected = tree(PARENT/"train.py")
    unchanged_functions = []
    for function in expected.body:
        if isinstance(function, ast.FunctionDef) and function.name not in {"validate_trainable_model", "first_gradient_audit"}:
            equal(find_function(actual, function.name), function, "train:"+function.name)
            unchanged_functions.append(function.name)
    equal(find_function(tree(ROOT/"contract.py"), "validate_resume_payload"),
          find_function(tree(PARENT/"contract.py"), "validate_resume_payload"), "full resume validator")
    scope = {"__file__": str(PARENT/"contract.py")}
    execute(tree(PARENT/"contract.py").body, scope)
    prepared = dict(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS,
                    rows_canonical_sha256="rows", mapping_sha256="mapping")
    old, new = scope["semantic_contract"](prepared, "text", "ced"), semantic_contract(prepared, "text", "ced")
    for value in (old, new):
        for key in ("route", "code_sha256", "recursive_model_source_sha256", "audio_listen"):
            value.pop(key, None)
    if old != new:
        raise ValueError("common training contract changed")
    return {"identical_files_normalized_lf": unchanged,
            "identical_training_functions": unchanged_functions, "common_recipe_equal": True}


def audit_split3_copy():
    actual, expected = tree(ROOT/"models/recursive_mesh_listen.py"), tree(PARENT/"models/recursive_mesh_listen.py")
    # Remove only the two direct residual calls and per-forward audit resets.
    for module in (actual, expected):
        core = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "MeshLlamaModel")
        if module is actual:
            forward = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
            class StripInjection(ast.NodeTransformer):
                def visit_Assign(self, node):
                    if any(ast.unparse(t) in {"self.last_injection_trace", "self.last_injection_audit"} for t in node.targets):
                        return None
                    if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "self.inject_audio":
                        return None
                    return self.generic_visit(node)
                def visit_If(self, node):
                    if ast.unparse(node.test) == "loop == 0":
                        if len(node.body) != 1 or "self.inject_audio" not in ast.unparse(node.body[0]):
                            raise ValueError("unexpected loop-specific change")
                        return None
                    return self.generic_visit(node)
            StripInjection().visit(forward)
        core.body = [n for n in core.body if not isinstance(n, ast.FunctionDef)
                     or n.name not in {"enable_audio_listen", "inject_audio"}]
    equal(actual, expected, "entire backbone except gate registration, injection, audit resets")
    return {"backbone_except_explicit_gates_ast_equal": True,
            "copy_source_sha256": {p.relative_to(PARENT).as_posix(): hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                for p in sorted(PARENT.rglob("*")) if p.suffix in {".py", ".sh"}}}


def audit():
    return {"status": "PASS", "scope": "local source audit; no real assets or GPU training",
            "route_code_sha256": code_digest(), "split3_parent": str(PARENT),
            "parent_parity": audit_parent(), "backbone_parity": audit_split3_copy()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"reference_audit.json")
    args = parser.parse_args()
    result = audit()
    write_json(args.output, result)
    print(result["status"], args.output)
