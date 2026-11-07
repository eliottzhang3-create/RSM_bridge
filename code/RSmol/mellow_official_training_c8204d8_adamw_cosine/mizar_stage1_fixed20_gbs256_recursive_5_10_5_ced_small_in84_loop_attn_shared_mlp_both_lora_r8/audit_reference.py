"""Audit isolated shared-plus-modality MLP LoRA against its unchanged mother."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from contract import SOURCE_COUNTS, LOOP_LORA_CONTRACT, code_digest, semantic_contract, write_json

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
PARENT = ROOT.with_name("mizar_stage1_fixed20_gbs256_recursive_5_10_5_ced_small_in84_loop_attn_shared_mlp_modality_lora_r8")
BASELINE = REPO/"audit_reports/mizar_s1_attn_shared_mlp_both_lora_r8_parent_baseline.json"
LAUNCHER = REPO/"code/RSmol/run_mizar_s1_gbs256_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_attn_shared_mlp_shared_modality_lora_r8_5090.sh"


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def find_function(module, name):
    found = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise ValueError(f"expected one function {name}, got {len(found)}")
    return found[0]


def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[])),
                 "shared-attention-modality-mlp-source-audit", "exec"), scope)


def equal(actual, expected, label):
    if ast.dump(actual) != ast.dump(expected):
        raise ValueError("unexpected inherited difference: " + label)


def protected_parents():
    files = json.loads(BASELINE.read_text(encoding="utf-8"))["files"]
    for name, expected in files.items():
        raw = (REPO/name).read_bytes().replace(bytes([13, 10]), bytes([10]))
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("protected original changed: " + name)
    return {"files": len(files), "all_unchanged": True}


def isolated_names(source):
    return (source.replace(PARENT.name, ROOT.name)
            .replace("mizar_s1_attn_shared_mlp_modality_lora_r8", "mizar_s1_attn_shared_mlp_both_lora_r8")
            .replace("loop_attn_shared_mlp_modality_lora_r8", "loop_attn_shared_mlp_shared_modality_lora_r8")
            .replace("mizar-ced-in84-asmm8-g256-", "mizar-ced-in84-asmb8-g256-"))


def audit_training():
    expected = (PARENT/"train.py").read_text(encoding="utf-8")
    for loop in (1, 2):
        needle = f'"loop{loop}_attention_shared_lora_A", "loop{loop}_attention_shared_lora_B",'
        expected = expected.replace(needle, needle+f'\n                   "loop{loop}_mlp_shared_lora_A", "loop{loop}_mlp_shared_lora_B",')
    expected = expected.replace(
        'families = {"attention_shared": ("shared", "shared"), "mlp_audio": ("modality", "audio"),\n                "mlp_text": ("modality", "text")}',
        'families = {"attention_shared": ("shared", "shared"), "mlp_shared": ("shared_modality", "shared"),\n                "mlp_audio": ("shared_modality", "audio"), "mlp_text": ("shared_modality", "text")}')
    if expected != (ROOT/"train.py").read_text(encoding="utf-8"):
        raise ValueError("trainer differs beyond four role names and shared MLP gradient family")
    equal(find_function(tree(ROOT/"train.py"), "train"), find_function(tree(PARENT/"train.py"), "train"), "complete training/update loop")
    return {"entire_train_function_ast_equal_after_explicit_audits": True,
            "entire_train_module_equal_after_role_and_gradient_audit_changes": True}


def audit():
    inventory = lambda p: {f.relative_to(p).as_posix() for f in p.rglob("*")
                           if f.suffix in {".py", ".sh"} and "__pycache__" not in f.parts}
    if inventory(ROOT) != inventory(PARENT):
        raise ValueError("copied source inventory differs")
    changed = {"audit_reference.py", "contract.py", "models/loop_lora.py", "optim.py", "train.py", "write_config.py", "runtime_checks.py"}
    identical = []
    for name in sorted(inventory(PARENT)-changed):
        if (ROOT/name).read_text(encoding="utf-8") != isolated_names((PARENT/name).read_text(encoding="utf-8")):
            raise ValueError("unexpected copied source difference: " + name)
        identical.append(name)
    if (ROOT/"runtime_checks.py").read_text(encoding="utf-8") != isolated_names((PARENT/"runtime_checks.py").read_text(encoding="utf-8")):
        raise ValueError("optional tensor entry differs beyond route name")
    for name in (".gitignore", "LICENSE", "NOTICE", "licenses/Mizar-ClearBSD.txt"):
        if (ROOT/name).read_bytes() != (PARENT/name).read_bytes():
            raise ValueError("copied license/ignore differs: " + name)
    expected_launcher = isolated_names(LAUNCHER.with_name(LAUNCHER.name.replace("mlp_shared_modality", "mlp_modality")).read_text(encoding="utf-8"))
    expected_launcher = expected_launcher.replace("Attention-shared / MLP-modality loop-specific LoRA r8 alpha16", "Attention-shared r8 alpha16 / MLP-shared-plus-modality r8 alpha8 loop-specific LoRA")
    if expected_launcher != LAUNCHER.read_text(encoding="utf-8"):
        raise ValueError("formal launcher differs beyond identity and architecture message")
    old_contract = ast.parse(isolated_names((PARENT/"contract.py").read_text(encoding="utf-8")))
    new_contract = tree(ROOT/"contract.py")
    for module in (old_contract, new_contract):
        for node in module.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LOOP_LORA_CONTRACT" for t in node.targets):
                node.value = ast.Constant(value=None)
    equal(old_contract, new_contract, "entire contract except LoRA and isolated identity")
    for name in ("models/recursive_layer_inject.py", "models/recursive_text.py"):
        equal(tree(ROOT/name), tree(PARENT/name), "complete backbone/loader")
    equal(find_function(tree(ROOT/"models/loop_lora.py"), "call_with_loop_lora"), find_function(tree(PARENT/"models/loop_lora.py"), "call_with_loop_lora"), "exception-safe round selection")
    old, new = [tree(p/"optim.py") for p in (PARENT, ROOT)]
    equal(next(n for n in old.body if isinstance(n, ast.ClassDef) and n.name == "GradNormTracker"), next(n for n in new.body if isinstance(n, ast.ClassDef) and n.name == "GradNormTracker"), "GradNormTracker")
    opts = [find_function(t, "build_optimizer") for t in (old, new)]
    for node in opts:
        node.body = [n for n in node.body if not isinstance(n, ast.If)]
    equal(*opts, "optimizer and scheduler except role coverage guard")
    scope = {"__file__": str(PARENT/"contract.py")}
    execute(tree(PARENT/"contract.py").body, scope)
    prepared = dict(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS, rows_canonical_sha256="rows", mapping_sha256="mapping")
    recipes = [scope["semantic_contract"](prepared, "text", "ced"), semantic_contract(prepared, "text", "ced")]
    for recipe in recipes:
        for key in ("route", "code_sha256", "loop_lora"):
            recipe.pop(key, None)
    if recipes[0] != recipes[1]:
        raise ValueError("non-LoRA training recipe differs")
    return {"status": "PASS", "scope": "local source audit; no real assets or GPU execution",
            "route_code_sha256": code_digest(), "protected_originals": protected_parents(),
            "checked_copied_sources": sorted(inventory(ROOT)), "identical_source_files_except_route_names": identical,
            "training_parity": audit_training(), "backbone_parity": {"entire_backbone_ast_equal_after_metadata": True, "entire_parent_backbone_ast_equal": True},
            "common_recipe_equal": True, "launcher_resources_and_arguments_equal": True, "loop_lora": LOOP_LORA_CONTRACT}


if __name__ == "__main__":
    result = audit()
    write_json(ROOT/"reference_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
