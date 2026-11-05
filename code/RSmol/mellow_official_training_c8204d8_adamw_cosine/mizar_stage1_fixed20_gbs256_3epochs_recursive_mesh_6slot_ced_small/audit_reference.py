"""Standard-library source/recipe audit against the unchanged five-slot CED parent."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from contract import SOURCE_COUNTS, code_digest, geometry, semantic_contract, write_json

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
PARENT = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small")
STAGE_PREFIX = "mizar_s1_gbs256_recursive_mesh_6slot_ced_small_"


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def find_function(module, name):
    matches = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one function named {name}, got {len(matches)}")
    return matches[0]


def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "six-slot-reference-extract", "exec"), scope)


def normalized_bytes(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


def module_contract(path):
    scope = {"__file__": str(path)}
    execute(tree(path).body, scope)
    return scope


def protected_originals():
    expected = json.loads((ROOT / "original_files_sha256.json").read_text(encoding="utf-8"))
    for name, checksum in expected.items():
        if hashlib.sha256(normalized_bytes(REPO / name)).hexdigest() != checksum:
            raise ValueError(f"protected original changed: {name}")
    return len(expected)


def audit_parent():
    allowed = {"README.md", "reference_audit.json", "audit_reference.py", "contract.py",
               "models/recursive_text.py", "train.py", "run.sh", "stage_data.py"}
    unchanged = []
    for path in sorted(PARENT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        name = path.relative_to(PARENT).as_posix()
        if name not in allowed:
            if normalized_bytes(ROOT / name) != normalized_bytes(path):
                raise ValueError(f"copied parent file changed: {name}")
            unchanged.append(name)
    new, old = tree(ROOT / "train.py"), tree(PARENT / "train.py")
    new.body = [n for n in new.body if not isinstance(n, ast.FunctionDef)
                or n.name != "recursive_execution_audit"]
    for index, node in enumerate(new.body):
        if isinstance(node, ast.FunctionDef) and node.name == "first_gradient_audit":
            new.body[index] = copy.deepcopy(find_function(old, node.name))
    if ast.dump(new) != ast.dump(old):
        raise ValueError("trainer changed beyond the first-gradient architecture audit")
    expected = tree(PARENT / "contract.py")
    source = (ROOT / "contract.py").read_text(encoding="utf-8").replace(
        '"memory_slots": 6,', '"memory_slots": 5,').replace(
        '"recursive_model_5_10x2_5_mesh_6slot.py"', '"recursive_model_5_10x2_5_mesh.py"')
    actual = ast.parse(source)
    for index, node in enumerate(actual.body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {
                "ROUTE", "TEXT_MODEL_CONTRACT", "TEXT_MODEL_ARCHITECTURE"} for t in node.targets):
            name = node.targets[0].id
            actual.body[index] = copy.deepcopy(next(n for n in expected.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name))
    if ast.dump(actual) != ast.dump(expected):
        raise ValueError("contract changed beyond route, six-slot geometry and model source")
    stage = (ROOT / "stage_data.py").read_text(encoding="utf-8").replace(STAGE_PREFIX, "mizar_s1_gbs256_")
    if ast.dump(ast.parse(stage)) != ast.dump(tree(PARENT / "stage_data.py")):
        raise ValueError("staging changed beyond the isolated tmpfs prefix")
    run = (ROOT / "run.sh").read_text(encoding="utf-8").replace(
        "run_mizar_s1_gbs256_recursive_mesh_6slot_ced_small_3090.sh",
        "run_mizar_s1_gbs256_recursive_mesh_ced_small_5090.sh").replace(
        STAGE_PREFIX, "mizar_s1_gbs256_recursive_mesh_ced_small_")
    if run != (PARENT / "run.sh").read_text(encoding="utf-8"):
        raise ValueError("compute launcher changed beyond isolated names")
    return {"byte_identical_files": unchanged,
            "optimizer_epoch_loop": "identical including gradient diagnostic",
            "whole_trainer": "identical except read-only first-gradient architecture audit",
            "contract": "identical except route, six-slot geometry and model source",
            "staging": "identical except isolated tmpfs prefix"}


def audit_recipe():
    parent = module_contract(PARENT / "contract.py")
    prepared = dict(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS,
                    rows_canonical_sha256="synthetic_rows", mapping_sha256="synthetic_mapping")
    expected = parent["semantic_contract"](prepared, "synthetic_text", "synthetic_ced")
    actual = semantic_contract(prepared, "synthetic_text", "synthetic_ced")
    allowed = {"route", "code_sha256", "recursive_model_source_sha256",
               "text_model_contract", "text_model_architecture"}
    changed = {k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k)}
    if changed != allowed:
        raise ValueError(f"unexpected recipe differences: {changed ^ allowed}")
    return {"changed_fields": sorted(changed), "identical_fields": sorted(set(expected) - changed)}


def audit():
    parent_audit = audit_parent()
    parent_contract = module_contract(PARENT / "contract.py")
    evidence = json.loads((PARENT / "reference_audit.json").read_text(encoding="utf-8"))
    if (evidence["route_code_sha256"] != parent_contract["code_digest"]()
            or evidence["official_mapper_class_asts_equal"] != ["CEDFrequencyMerger", "CEDTemporalMapper"]):
        raise ValueError("parent official mapper evidence is stale")
    shape = geometry(sum(SOURCE_COUNTS.values()))
    return {"status": "PASS", "scope": "local source/stdlib only; no real assets or GPU training",
            "route_code_sha256": code_digest(), "geometry": shape,
            "protected_original_files": protected_originals(), "parent_route_audit": parent_audit,
            "recipe": audit_recipe(), "official_mapper_lineage": {
                "parent_source_identity_verified": True,
                "official_mapper_class_asts_equal": evidence["official_mapper_class_asts_equal"],
                "mapper": "normalized-LF identical to the audited parent"},
            "checkpoint_steps": [round(shape["total_steps"] * i / 6) for i in range(1, 7)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reference_audit.json")
    args = parser.parse_args()
    result = audit()
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
