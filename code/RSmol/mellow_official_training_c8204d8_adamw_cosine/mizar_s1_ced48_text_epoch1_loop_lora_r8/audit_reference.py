"""Read-only inheritance audit against the exact CED4->8 audio parent."""
import ast
import copy
import hashlib
import json
from pathlib import Path

from contract import (LOOP_LORA_CONTRACT, SOURCE_COUNTS, code_digest, semantic_contract, write_json)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
PARENT = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8")


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def find_function(module, name):
    found = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise ValueError(f"expected one function {name}, got {len(found)}")
    return found[0]


def execute(nodes, scope):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(module, "ced48-pretrained-lora-audit", "exec"), scope)


def equal(actual, expected, label):
    if ast.dump(actual) != ast.dump(expected):
        raise ValueError("unexpected inherited semantic difference: " + label)


def protected_parents():
    files = json.loads((ROOT / "source_guard.json").read_text(encoding="utf-8"))
    for name, expected in files.items():
        raw = (REPO / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("protected original changed: " + name)
    return {"files": len(files), "all_unchanged": True}


def audit_training():
    old, new = [find_function(tree(p / "train.py"), "train") for p in (PARENT, ROOT)]
    # Restore the parent-only assertion that initial LoRA B must be zero.
    # It cannot apply to a trained checkpoint. No update/data/save logic changes.
    parent_guard = next(n for n in ast.walk(old) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name) and n.func.id == "collective_require"
                        and n.args and "segment_start_step != 0" in ast.unparse(n.args[0]))
    new_guard = next(n for n in ast.walk(new) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "collective_require"
                     and n.args and "segment_start_step != 0" in ast.unparse(n.args[0]))
    expected_guard = ast.parse("gradient_audit['passed'] and (segment_start_step != 0 or all(gradient_audit['closed_gates'].values()))", mode="eval").body
    equal(new_guard.args[0], expected_guard, "only fresh audio gates must start closed")
    new_guard.args[0] = copy.deepcopy(parent_guard.args[0])
    removed = []
    class AuditKeys(ast.NodeTransformer):
        def visit_Dict(self, node):
            pairs = []
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in {
                        "text_initialization", "inherited_lora_parameter_count", "fresh_audio_parameter_count"}:
                    removed.append(key.value)
                    continue
                pairs.append((key, value))
            node.keys, node.values = [k for k,_ in pairs], [v for _,v in pairs]
            return self.generic_visit(node)
    AuditKeys().visit(new)
    if len(removed) != 3:
        raise ValueError(f"unexpected training audit additions: {removed}")
    equal(new, old, "entire train function after explicit initialization audits")
    return {"entire_train_function_preserved_after_explicit_audits": True,
            "changed_first_step_condition": "only fresh audio gates must be zero; trained B retained",
            "added_report_keys": removed}


def audit_backbone():
    old, new = [tree(p / "models/recursive_layer_inject.py") for p in (PARENT, ROOT)]
    installs = []
    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value) == "self.model.enable_loop_lora()":
                installs.append(node)
                return None
            return self.generic_visit(node)
        def visit_FunctionDef(self, node):
            if node.name == "enable_loop_lora":
                node.body[0] = copy.deepcopy(find_function(old, "enable_loop_lora").body[0])
            return self.generic_visit(node)
    Strip().visit(new)
    if len(installs) != 1:
        raise ValueError("LoRA slots must be installed exactly once in LM constructor")
    equal(new, old, "entire model except moving LoRA installation before HF load")
    return {"model_forward_injection_attention_rope_unchanged": True,
            "one_constructor_install_before_hf_weight_load": True}


def audit():
    expected_ignore = (PARENT / ".gitignore").read_text().strip() + "\n!source_guard.json\n!parent_reference_audit.json"
    if (ROOT / ".gitignore").read_text().strip() != expected_ignore:
        raise ValueError("only the small source/history JSON manifests may be added to gitignore exceptions")
    inventory = lambda root: {p.relative_to(root).as_posix() for p in root.rglob("*")
                              if p.suffix in {".py", ".sh"} and "__pycache__" not in p.parts}
    if inventory(ROOT) != inventory(PARENT):
        raise ValueError("missing or unexpected copied route sources")
    changed = {"audit_reference.py", "contract.py", "models/loop_lora.py", "models/recursive_layer_inject.py",
               "models/recursive_text.py", "train.py", "write_config.py", "run.sh", "runtime_checks.py"}
    identical = []
    for name in sorted(inventory(PARENT) - changed):
        if (PARENT / name).read_bytes() != (ROOT / name).read_bytes():
            raise ValueError("unexpected copied source difference: " + name)
        identical.append(name)
    old, new = [tree(p / "train.py") for p in (PARENT, ROOT)]
    for method in ("first_gradient_audit", "answer_loss", "move_batch", "rng_state", "restore_rng",
                   "save_checkpoint", "reload_audit", "inspect_tokenizer", "probe_sources", "main"):
        equal(find_function(new, method), find_function(old, method), method)
    old_contract = {"__file__": str(PARENT / "contract.py")}
    execute(tree(PARENT / "contract.py").body, old_contract)
    prepared = dict(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS,
                    rows_canonical_sha256="synthetic_rows", mapping_sha256="synthetic_mapping")
    a, b = old_contract["semantic_contract"](prepared,"text","ced"), semantic_contract(prepared,"text","ced")
    for value in (a,b):
        for name in ("route", "code_sha256", "recursive_model_source_sha256", "text_model_contract", "text_model_architecture"):
            value.pop(name)
        for name in ("A_init", "B_init", "initialization_rng", "initialization", "inherited_parameters", "fresh_parameters"):
            value["loop_lora"].pop(name, None)
    if a != b:
        raise ValueError("audio recipe differs beyond pretrained text/LoRA identity")
    return {"status": "PASS", "scope": "source audit only; no production tensors/jobs",
            "code_sha256": code_digest(), "protected_originals": protected_parents(),
            "byte_identical_sources": identical, "audio_semantic_recipe_equal": True,
            "training_parity": audit_training(), "backbone_parity": audit_backbone(),
            "loop_lora": LOOP_LORA_CONTRACT}


if __name__ == "__main__":
    result = audit()
    write_json(ROOT / "reference_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
