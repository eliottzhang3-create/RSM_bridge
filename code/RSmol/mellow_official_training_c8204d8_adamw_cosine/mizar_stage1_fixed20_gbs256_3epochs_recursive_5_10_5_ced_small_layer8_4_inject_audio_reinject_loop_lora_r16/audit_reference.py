"""Exact r8 -> r16/alpha32 source audit; all other training behavior is inherited."""
import ast
import copy
import hashlib
import json
from pathlib import Path

from contract import (LOOP_LORA_CONTRACT, SERIAL_INJECTION_CONTRACT, SOURCE_COUNTS,
                      code_digest, semantic_contract, write_json)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
PARENT = ROOT.with_name(ROOT.name.replace("loop_lora_r16", "loop_lora_r8"))
BASELINE = REPO / "audit_reports/mizar_s1_ced_loop_lora_r16_parent_baseline.json"
LAUNCHER = REPO / "code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r16_5090.sh"


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def find_function(module, name):
    found = [node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(found) != 1:
        raise ValueError(f"expected one function {name}, got {len(found)}")
    return found[0]


def execute(nodes, scope):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(module, "rank16-source-audit", "exec"), scope)


def equal(actual, expected, label):
    if ast.dump(actual) != ast.dump(expected):
        raise ValueError("unexpected inherited semantic difference: " + label)


def protected_parents():
    files = json.loads(BASELINE.read_text(encoding="utf-8"))["files"]
    for name, expected in files.items():
        raw = (REPO / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("protected original changed: " + name)
    return {"files": len(files), "all_unchanged": True}


def expected_source(name):
    """Only rank, alpha, count, and explicit isolated names are allowed to differ."""
    source = (PARENT / name).read_text(encoding="utf-8-sig")
    source = source.replace("loop_lora_r8", "loop_lora_r16")
    if name == "models/loop_lora.py":
        old = "RANK, ALPHA, LOOPS = 8, 16, 2"
        if source.count(old) != 1:
            raise ValueError("rank8 parent constants changed")
        source = source.replace(old, "RANK, ALPHA, LOOPS = 16, 32, 2")
    elif name == "contract.py":
        source = source.replace('"rank": 8, "alpha": 16', '"rank": 16, "alpha": 32')
        source = source.replace('"new_parameters": 1628160', '"new_parameters": 3256320')
    elif name == "write_config.py":
        source = source.replace("r8 alpha16", "r16 alpha32").replace("1628160 new parameters", "3256320 new parameters")
    elif name == "runtime_checks.py":
        source = source.replace("validate_mizar_s1_ced_loop_lora.py", "validate_mizar_s1_ced_loop_lora_r16.py")
    return source


def audit():
    inventory = lambda path: {p.relative_to(path).as_posix() for p in path.rglob("*")
                              if p.suffix in {".py", ".sh"} and "__pycache__" not in p.parts}
    if inventory(ROOT) != inventory(PARENT):
        raise ValueError("rank16 must contain the complete rank8 source inventory")
    checked, identical = [], []
    semantic_changes = {"models/loop_lora.py", "contract.py", "write_config.py", "runtime_checks.py"}
    for name in sorted(inventory(ROOT) - {"audit_reference.py"}):
        if expected_source(name) != (ROOT / name).read_text(encoding="utf-8-sig"):
            raise ValueError("unexpected difference from rank8: " + name)
        checked.append(name)
        if name not in semantic_changes:
            identical.append(name)
    for name in (".gitignore", "LICENSE", "NOTICE", "licenses/Mizar-ClearBSD.txt"):
        if (ROOT/name).read_bytes() != (PARENT/name).read_bytes():
            raise ValueError("copied license/ignore file changed: " + name)
    for name in ("train.py", "optim.py", "models/recursive_layer_inject.py", "models/recursive_text.py"):
        equal(tree(ROOT/name), tree(PARENT/name), "complete " + name)
    expected_launcher = LAUNCHER.with_name(LAUNCHER.name.replace("_r16_", "_r8_")).read_text(encoding="utf-8")
    expected_launcher = expected_launcher.replace("loop_lora_r8", "loop_lora_r16")
    expected_launcher = expected_launcher.replace("mizar-ced-in84-lr8-g256-", "mizar-ced-in84-lr16-g256-")
    expected_launcher = expected_launcher.replace("LoRA r8 alpha16", "LoRA r16 alpha32")
    if expected_launcher != LAUNCHER.read_text(encoding="utf-8"):
        raise ValueError("launcher differs beyond isolated names and r16/alpha32 message")
    scope = {"__file__": str(PARENT / "contract.py")}
    execute(tree(PARENT / "contract.py").body, scope)
    prepared = dict(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS,
                    rows_canonical_sha256="synthetic_rows", mapping_sha256="synthetic_mapping")
    old = scope["semantic_contract"](prepared, "text", "ced")
    new = semantic_contract(prepared, "text", "ced")
    if (old["loop_lora"]["rank"], old["loop_lora"]["alpha"], old["loop_lora"]["new_parameters"]) != (8, 16, 1628160):
        raise ValueError("unexpected rank8 parent configuration")
    if (new["loop_lora"]["rank"], new["loop_lora"]["alpha"], new["loop_lora"]["new_parameters"]) != (16, 32, 3256320):
        raise ValueError("unexpected rank16 configuration")
    for value in (old, new):
        for key in ("route", "code_sha256"):
            value.pop(key)
        for key in ("rank", "alpha", "new_parameters"):
            value["loop_lora"].pop(key)
    if new != old:
        raise ValueError("rank16 recipe differs beyond rank/alpha/count and isolated identity")
    return {"status": "PASS", "scope": "source audit only; no tensors or remote execution",
            "route_code_sha256": code_digest(), "protected_originals": protected_parents(),
            "checked_copied_sources": checked, "identical_source_files_except_route_names": identical,
            "rank_changes": {"rank": [8, 16], "alpha": [16, 32], "new_parameters": [1628160, 3256320]},
            "training_parity": {"entire_train_function_ast_equal_after_explicit_audits": True,
                                "entire_train_module_ast_equal": True, "entire_optimizer_module_ast_equal": True},
            "backbone_parity": {"entire_parent_backbone_ast_equal_after_explicit_additions": True,
                                "entire_backbone_and_loader_ast_equal": True},
            "common_recipe_equal": True, "launcher_resources_and_arguments_equal": True,
            "loop_lora": dict(LOOP_LORA_CONTRACT), "serial_injection": dict(SERIAL_INJECTION_CONTRACT)}


if __name__ == "__main__":
    result = audit()
    write_json(ROOT / "reference_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
