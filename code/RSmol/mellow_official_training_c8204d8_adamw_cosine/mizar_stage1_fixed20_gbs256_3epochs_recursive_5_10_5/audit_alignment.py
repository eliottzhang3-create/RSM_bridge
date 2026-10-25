"""CPU/stdlib comparison against the repaired MeSH Mizar training route."""
from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path


ROUTE_ROOT = Path(__file__).resolve().parent
MOTHER_ROOT = ROUTE_ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh")
ALLOWED_DIFFERENCES = {
    "README.md", "contract.py", "models/recursive_text.py", "models/decoder.py",
    "train.py", "write_config.py", "run.sh", "NOTICE", "runtime_checks.py",
}
ADDITIONS = {"audit_alignment.py", "audit_tokenizer.py"}


def load_contract(root):
    spec = importlib.util.spec_from_file_location(f"contract_{root.name}", root / "contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_identity(source):
    return source.replace("recursive_5_10_5", "recursive_mesh").replace(
        "recursive-5-10-5", "recursive-mesh").replace(
        "recursive 5-10-5 non-MeSH", "recursive MeSH").replace(
        "5-10-5 non-MeSH recursive checkpoint", "5-10x2-5 MeSH recursive checkpoint")


def contract_training_ast(source):
    tree = ast.parse(source)
    identity = {"ROUTE", "TEXT_MODEL_CONTRACT", "TEXT_MODEL_ARCHITECTURE",
                "DEFAULT_TEXT_MODEL_DIR", "TEXT_MODEL_SOURCE_MAPPING"}
    model_audits = {"recursive_checkpoint_contract", "recursive_model_source_sha256",
                    "validate_recursive_config_values"}
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id in identity for target in node.targets))
                 and not (isinstance(node, ast.FunctionDef) and node.name in model_audits)]
    return ast.dump(tree, include_attributes=False)


def audit_alignment(route_root=ROUTE_ROOT, mother_root=MOTHER_ROOT):
    def files(root):
        return {p.relative_to(root).as_posix(): p for p in root.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}

    mother, route = files(mother_root), files(route_root)
    problems, identical = [], []
    missing, unexpected = set(mother) - set(route), set(route) - set(mother) - ADDITIONS
    if missing or unexpected:
        problems.append(f"file set changed: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for name in sorted(set(mother) & set(route) - ALLOWED_DIFFERENCES):
        if mother[name].read_bytes() != route[name].read_bytes():
            problems.append(f"training file differs: {name}")
        else:
            identical.append(name)
    for name in ("train.py", "models/decoder.py", "write_config.py", "run.sh"):
        old = mother[name].read_text(encoding="utf-8")
        new = normalize_identity(route[name].read_text(encoding="utf-8"))
        if old != new:
            problems.append(f"non-model training source changed: {name}")
    old_runtime = mother["runtime_checks.py"].read_text(encoding="utf-8")
    new_runtime = route["runtime_checks.py"].read_text(encoding="utf-8")
    if old_runtime.replace("recursive-mesh-fixture", "recursive-5-10-5-fixture").replace(
            "logical_30_physical_20_5_10x2_5", "logical_30_physical_20_5_10_5_loops_2_no_mesh").replace(
            "is_recursive_mesh", "is_recursive_5_10_5") != new_runtime:
        problems.append("runtime_checks.py changed outside the model fixture identity")
    old_contract, new_contract = load_contract(mother_root), load_contract(route_root)
    if contract_training_ast(mother["contract.py"].read_text(encoding="utf-8")) != \
            contract_training_ast(route["contract.py"].read_text(encoding="utf-8")):
        problems.append("training constants/functions changed in contract.py")
    rsmol_root = route_root.parents[1]
    old_outer = (rsmol_root / "run_mizar_s1_gbs256_recursive_mesh_5090.sh").read_text(encoding="utf-8")
    new_outer = (rsmol_root / "run_mizar_s1_gbs256_recursive_5_10_5_5090.sh").read_text(encoding="utf-8")
    new_outer = normalize_identity(new_outer)
    old_outer = re.sub(r"^TEXT_MODEL_DIR=.*$", "TEXT_MODEL_DIR=<checkpoint>", old_outer, flags=re.MULTILINE)
    new_outer = re.sub(r"^TEXT_MODEL_DIR=.*$", "TEXT_MODEL_DIR=<checkpoint>", new_outer, flags=re.MULTILINE)
    if old_outer != new_outer:
        problems.append("submission behavior changed beyond checkpoint/route identity")
    if new_contract.ROUTE == old_contract.ROUTE:
        problems.append("route identity is not isolated")
    if new_contract.geometry(1477492) != old_contract.geometry(1477492):
        problems.append("training geometry differs")
    return {"status": "FAIL" if problems else "PASS", "mother": str(mother_root),
            "route": str(route_root), "byte_identical_files": identical,
            "identity_only_files": ["train.py", "models/decoder.py", "write_config.py", "run.sh", "outer launcher"],
            "model_specific_files": ["contract.py: identity/model audit only", "models/recursive_text.py"],
            "problems": problems}


if __name__ == "__main__":
    report = audit_alignment()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "PASS" else 1)
