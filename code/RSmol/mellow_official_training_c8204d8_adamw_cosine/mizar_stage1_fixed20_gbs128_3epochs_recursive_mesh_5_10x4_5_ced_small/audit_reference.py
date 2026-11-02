"""Source/recipe audit against the unchanged CED parent and established x4 loader.

Standard library only: this does not prove pretrained loading or GPU training.
"""
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
X4 = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_5_10x4_5")


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def find_function(module, name):
    matches = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one function named {name}, got {len(matches)}")
    return matches[0]


def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "x4-ced-reference-extract", "exec"), scope)


def normalized_bytes(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


def protected_originals():
    expected = json.loads((ROOT / "original_files_sha256.json").read_text(encoding="utf-8"))
    for name, checksum in expected.items():
        if hashlib.sha256(normalized_bytes(REPO / name)).hexdigest() != checksum:
            raise ValueError(f"protected original changed: {name}")
    return len(expected)


def module_contract(path):
    scope = {"__file__": str(path)}
    execute(tree(path).body, scope)
    return scope


def audit_parent():
    unchanged = [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "data").glob("*.py"))]
    unchanged += ["prepare_index.py", "utils/audio_io.py"]
    for name in unchanged + ["optim.py", "check_ced_assets.py", "models/audio.py", "models/ced.py",
                              "models/ced_offline.py", "models/mellow.py", "models/decoder.py"]:
        if normalized_bytes(ROOT / name) != normalized_bytes(PARENT / name):
            raise ValueError(f"CED parent implementation changed: {name}")
    identical_functions = {
        "contract.py": ("geometry", "lr_factor", "data_order", "rank_seed", "raw_references",
                        "ced_label_metadata", "validate_ced_config", "validate_ced_assets",
                        "validate_checkpoint_tokenizer", "validate_tokenizer_contract", "semantic_contract"),
        "train.py": ("move_batch", "rng_state", "restore_rng", "answer_loss", "save_checkpoint",
                     "reload_audit", "validate_trainable_model", "role_snapshots", "probe_sources"),
    }
    for name, functions in identical_functions.items():
        for function in functions:
            if ast.dump(find_function(tree(ROOT / name), function)) != ast.dump(find_function(tree(PARENT / name), function)):
                raise ValueError(f"CED parent function differs: {name}:{function}")
    loops = []
    for source in (ROOT, PARENT):
        train = find_function(tree(source / "train.py"), "train")
        loops.append(ast.dump(next(n for n in train.body if isinstance(n, ast.For))))
    if loops[0] != loops[1]:
        raise ValueError("CED epoch/update/microbatch loop changed")
    stage = normalized_bytes(ROOT / "stage_data.py").decode("utf-8").replace(
        "mizar_s1_gbs128_recursive_mesh_5_10x4_5_ced_small_", "mizar_s1_gbs256_").replace("GBS128", "GBS256")
    if ast.dump(ast.parse(stage)) != ast.dump(tree(PARENT / "stage_data.py")):
        raise ValueError("staging changed beyond isolated tmpfs naming")
    new, old = tree(ROOT / "train.py"), tree(PARENT / "train.py")
    new.body = [n for n in new.body if not isinstance(n, ast.FunctionDef) or n.name != "recursive_execution_audit"]
    for index, node in enumerate(new.body):
        if isinstance(node, ast.FunctionDef) and node.name in {"first_gradient_audit", "inspect_tokenizer"}:
            new.body[index] = copy.deepcopy(find_function(old, node.name))

    class RestoreMessages(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = node.value.replace("GBS128", "GBS256").replace(
                    "8 GPUs x microbatch4 x accumulation4 (global batch 128)",
                    "8 GPUs x microbatch8 x accumulation4 (global batch 256)")
            return node

    if ast.dump(RestoreMessages().visit(new)) != ast.dump(old):
        raise ValueError("trainer changed beyond architecture diagnostics and GBS messages")
    return {"byte_identical_files": unchanged, "identical_function_asts": identical_functions,
            "optimizer_epoch_loop": "identical including gradient diagnostic",
            "whole_trainer": "identical except GBS messages and architecture diagnostics",
            "staging": "identical except route-specific tmpfs prefix and message"}


def audit_recipe():
    parent = module_contract(PARENT / "contract.py")
    prepared = dict(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS,
                    rows_canonical_sha256="synthetic_rows", mapping_sha256="synthetic_mapping")
    expected = parent["semantic_contract"](prepared, "synthetic_text", "synthetic_ced")
    actual = semantic_contract(prepared, "synthetic_text", "synthetic_ced")
    allowed = {"route", "code_sha256", "recursive_model_source_sha256", "geometry",
               "text_model_contract", "text_model_architecture", "text_model_schedule"}
    changed = {k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k)}
    if changed != allowed:
        raise ValueError(f"unexpected recipe differences: {changed ^ allowed}")
    return {"changed_fields": sorted(changed), "identical_fields": sorted(set(expected) - changed)}


def audit():
    parent = audit_parent()
    if normalized_bytes(ROOT / "models/recursive_text.py") != normalized_bytes(X4 / "models/recursive_text.py"):
        raise ValueError("x4 strict loader differs from the established x4 route")
    x4 = module_contract(X4 / "contract.py")
    for function in ("recursive_checkpoint_contract", "recursive_model_source_sha256"):
        if ast.dump(find_function(tree(ROOT / "contract.py"), function)) != ast.dump(find_function(tree(X4 / "contract.py"), function)):
            raise ValueError(f"x4 model contract differs: {function}")
    shape = geometry(sum(SOURCE_COUNTS.values()))
    if shape != x4["geometry"](sum(SOURCE_COUNTS.values())):
        raise ValueError("GBS128 geometry differs from the established x4 route")
    parent_contract = module_contract(PARENT / "contract.py")
    parent_report = json.loads((PARENT / "reference_audit.json").read_text(encoding="utf-8"))
    if (parent_report["route_code_sha256"] != parent_contract["code_digest"]()
            or parent_report["official_mapper_class_asts_equal"] != ["CEDFrequencyMerger", "CEDTemporalMapper"]):
        raise ValueError("CED parent official-mapper evidence is stale")
    official_lineage = {"parent_source_identity_verified": True,
                        "official_mapper_class_asts_equal": parent_report["official_mapper_class_asts_equal"],
                        "mapper_file": "normalized-LF identical to the audited CED parent"}
    return {"status": "PASS", "scope": "local source/stdlib audit only; no real assets/tensors/GPU training",
            "route_code_sha256": code_digest(), "geometry": shape,
            "protected_original_files": protected_originals(), "parent_route_audit": parent,
            "recipe": audit_recipe(), "x4_loader": "normalized-LF identical",
            "ced_parent_baseline_audit": official_lineage,
            "checkpoint_steps": [round(shape["total_steps"] * i / 6) for i in range(1, 7)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reference_audit.json")
    args = parser.parse_args()
    result = audit()
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
