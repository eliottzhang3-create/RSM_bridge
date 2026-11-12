"""Exact source/recipe audit against the unchanged layer4/8 loop-LoRA parent.

Optional local check only; formal submission does not invoke this module.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

from contract import (DEFAULT_TEXT_MODEL_DIR, LOOP_LORA_CONTRACT, ROUTE,
                      SERIAL_INJECTION_CONTRACT, SOURCE_COUNTS,
                      code_digest, semantic_contract, write_json)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
PARENT = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8")
PARENT_ENTRY = "run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8_5090.sh"
ENTRY = "run_mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_5090.sh"
PARENT_ROUTE = "ced_small_mizar_s1_fixed20_gbs256_recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8_v1"
PARENT_TEXT = ("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/"
               "formal-epoch2-continue-20260902_184936/checkpoint-step-009244")
EXPECTED_TEXT = ("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
                 "mizar_text_5_10x2_5_gbs1024/epoch1/checkpoint-step-009244")
EXPECTED_ROUTE = "ced_small_mizar_s1_fixed20_gbs256_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_v1"
PARENT_OUTPUT = "mizar_s1_ced_small_gbs256_recursive_5_10_5_layer4_8_inject_audio_reinject_loop_lora_r8_5090"
OUTPUT = "mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_gbs256_5090"


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def find_function(module, name):
    found = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise ValueError(f"expected one function {name}, got {len(found)}")
    return found[0]


def execute(nodes, scope):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(module, "epoch1-loop-lora-source-audit", "exec"), scope)


def text(path):
    return path.read_text(encoding="utf-8")


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError("unexpected source or recipe difference: " + label)


def protected_parents():
    files = json.loads(text(ROOT / "source_guard.json"))
    for name, expected in files.items():
        raw = (REPO / name).read_bytes().replace(b"\r\n", b"\n")
        require_equal(hashlib.sha256(raw).hexdigest(), expected, name)
    return {"files": len(files), "all_unchanged": True}


def audit():
    originals = protected_parents()
    inventory = lambda path: {p.relative_to(path).as_posix() for p in path.rglob("*")
                              if p.suffix in {".py", ".sh"} and "__pycache__" not in p.parts}
    require_equal(inventory(ROOT), inventory(PARENT), "complete source inventory")
    changed = {"audit_reference.py", "contract.py", "run.sh"}
    identical = []
    for name in sorted(inventory(PARENT) - changed):
        require_equal((ROOT / name).read_bytes(), (PARENT / name).read_bytes(), name)
        identical.append(name)
    require_equal(DEFAULT_TEXT_MODEL_DIR, EXPECTED_TEXT, "default epoch1 checkpoint")
    require_equal(ROUTE, EXPECTED_ROUTE, "isolated route")
    expected = text(PARENT / "contract.py").replace(PARENT_ROUTE, EXPECTED_ROUTE).replace(
        '"/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/"\n'
        '    "formal-epoch2-continue-20260902_184936/checkpoint-step-009244"',
        '"' + EXPECTED_TEXT + '"')
    require_equal(text(ROOT / "contract.py"), expected, "contract identity changes only")
    expected = text(PARENT / "run.sh").replace(PARENT_ENTRY, ENTRY).replace(
        "/dev/shm/mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8_",
        "/dev/shm/mizar_s1_gbs256_ced_text_e1_in48_loop_lora_r8_")
    require_equal(text(ROOT / "run.sh"), expected, "runtime help and tmpfs only")
    require_equal(text(ROOT / ".gitignore"), text(PARENT / ".gitignore") + "!source_guard.json\n",
                  "Git includes the source guard")
    expected = text(REPO / "code/RSmol" / PARENT_ENTRY).replace(PARENT.name, ROOT.name).replace(
        PARENT_TEXT, EXPECTED_TEXT).replace(PARENT_OUTPUT, OUTPUT).replace(
        "mizar-ced-in48-lr8-g256-", "mizar-ced-e1-in48-lr8-g256-").replace(
        "mizar_s1_gbs256_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8.",
        "mizar_s1_ced_small_text_epoch1_layer4_8_audio_reinject_loop_lora_r8_5090.").replace(
        "Submitted Mizar S1 / CED-Small GBS256 /",
        "Submitted Mizar S1 / Mizar text epoch1 / CED-Small GBS256 /")
    require_equal(text(REPO / "code/RSmol" / ENTRY), expected, "launcher isolation only")
    scope = {"__file__": str(PARENT / "contract.py")}
    execute(tree(PARENT / "contract.py").body, scope)
    prepared = dict(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS,
                    rows_canonical_sha256="synthetic_rows", mapping_sha256="synthetic_mapping")
    old = scope["semantic_contract"](prepared, EXPECTED_TEXT, "ced")
    new = semantic_contract(prepared, EXPECTED_TEXT, "ced")
    for value in (old, new):
        for key in ("route", "code_sha256"):
            value.pop(key)
    require_equal(new, old, "entire inherited training/model/data recipe")
    return {
        "status": "PASS", "scope": "local exact source/recipe audit; no production assets or GPU",
        "route_code_sha256": code_digest(), "protected_originals": originals,
        "byte_identical_sources": identical,
        "training_parity": {"entire_train_file_byte_identical": "train.py" in identical},
        "backbone_parity": {"entire_backbone_file_byte_identical": "models/recursive_layer_inject.py" in identical},
        "common_recipe_equal": True,
        "loop_lora": LOOP_LORA_CONTRACT, "serial_injection": SERIAL_INJECTION_CONTRACT,
    }


if __name__ == "__main__":
    result = audit()
    write_json(ROOT / "reference_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

