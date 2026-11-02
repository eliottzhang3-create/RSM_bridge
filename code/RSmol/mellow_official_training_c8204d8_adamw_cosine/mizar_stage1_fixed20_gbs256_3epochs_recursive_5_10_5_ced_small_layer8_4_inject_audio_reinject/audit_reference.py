"""Source and training parity audit; no assets, tensors, CUDA or job submission."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

from contract import (AUDIO_REINJECTION, SERIAL_INJECTION_CONTRACT, SOURCE_COUNTS,
                      code_digest, semantic_contract, write_json)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
PARENT = ROOT.with_name(ROOT.name.removesuffix("_audio_reinject"))
INPUT_PARENT = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_audio_reinject")
BASELINE = REPO / "audit_reports/mizar_s1_ced_layer84_audio_reinject_parent_baseline.json"


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def find_function(module, name):
    matches = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one function {name}, got {len(matches)}")
    return matches[0]


def execute(nodes, scope):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(module, "serial-audit", "exec"), scope)


def equal(a, b, label):
    if ast.dump(a) != ast.dump(b):
        raise ValueError(f"unexpected inherited semantic difference: {label}")


def protected_parents():
    files = json.loads(BASELINE.read_text(encoding="utf-8"))["files"]
    for name, expected in files.items():
        if hashlib.sha256((REPO / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError(f"protected original changed: {name}")
    return {"files": len(files), "all_unchanged": True}


def audit_backbone():
    original, actual = [tree(p / "models/recursive_layer_inject.py") for p in (PARENT, ROOT)]
    core = next(n for n in actual.body if isinstance(n, ast.ClassDef) and n.name == "RecursiveLlama5_10_5Model")
    core.body = [n for n in core.body if not isinstance(n, ast.FunctionDef)
                 or n.name not in {"enable_audio_reinjection", "inject_input_audio", "reinjection_diagnostics"}]
    counts = {"guard": 0, "assign": 0, "trace": 0}

    class Strip(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.unparse(node.test) == "not hasattr(self, 'audio_linear')":
                counts["guard"] += 1
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            target = ast.unparse(node.targets[0]) if len(node.targets) == 1 else ""
            if target in {"audio_input_information", "self.last_reinject_trace",
                          "self.last_reinject_stats", "self.last_serial_injection_trace"}:
                counts["assign"] += 1
                return None
            if (target == "hidden_states" and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == "self.inject_input_audio"):
                counts["assign"] += 1
                return None
            return self.generic_visit(node)

        def visit_AugAssign(self, node):
            if ast.unparse(node.target) == "self.last_serial_injection_trace":
                counts["trace"] += 1
                return None
            return self.generic_visit(node)

    Strip().visit(core)
    if counts != {"guard": 1, "assign": 5, "trace": 2}:
        raise ValueError(f"unexpected serial backbone additions: {counts}")
    equal(actual, original, "entire 8->4 model after removing input residual additions")
    for new_name, old_name in (("enable_audio_reinjection", "enable_audio_reinjection"),
                               ("inject_input_audio", "inject_audio"),
                               ("reinjection_diagnostics", "reinjection_diagnostics")):
        actual_fn = find_function(tree(ROOT / "models/recursive_layer_inject.py"), new_name)
        actual_fn.name = old_name
        equal(actual_fn, find_function(tree(INPUT_PARENT / "models/recursive_reinject.py"), old_name),
              "original input branch:" + old_name)
    return {"entire_parent_backbone_ast_equal": True, "original_input_branch_methods_equal": True,
            "explicit_added_nodes": counts}


def audit_training():
    old = find_function(tree(PARENT / "train.py"), "train")
    new = find_function(tree(ROOT / "train.py"), "train")
    opened = next(n for n in ast.walk(old) if isinstance(n, ast.If) and ast.unparse(n.test) == "step == 1")
    counts = {"contract_guard": 0, "collect": 0, "diagnostics": 0, "rank_contract": 0,
              "new_role_predicate": 0, "opened_audit": 0, "end_roles": 0, "audit_keys": 0}
    old_role = ast.parse('parameter_role(name).startswith("layer")', mode="eval").body

    class Strip(ast.NodeTransformer):
        def visit_If(self, node):
            test = ast.unparse(node.test)
            if "contract.get('serial_audio_injection')" in test:
                counts["contract_guard"] += 1
                return None
            if test == "model.caption_decoder.lm.model.collect_reinject_stats":
                counts["diagnostics"] += 1
                return None
            if test == "step == 1":
                counts["opened_audit"] += 1
                return copy.deepcopy(opened)
            return self.generic_visit(node)

        def visit_Assign(self, node):
            target = ast.unparse(node.targets[0]) if len(node.targets) == 1 else ""
            if target == "model.caption_decoder.lm.model.collect_reinject_stats":
                counts["collect"] += 1
                return None
            if target == "rank_check['serial_audio_injection']":
                counts["rank_contract"] += 1
                return None
            return self.generic_visit(node)

        def visit_Compare(self, node):
            if ast.unparse(node) == "parameter_role(name) in NEW_ROLES":
                counts["new_role_predicate"] += 1
                return copy.deepcopy(old_role)
            return self.generic_visit(node)

        def visit_comprehension(self, node):
            if isinstance(node.iter, ast.Name) and node.iter.id == "NEW_ROLES":
                counts["end_roles"] += 1
                node.iter = ast.Name(id="TRAINABLE_ROLES", ctx=ast.Load())
                node.ifs.append(ast.parse('role.startswith("layer")', mode="eval").body)
            return self.generic_visit(node)

        def visit_Dict(self, node):
            keep = []
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in {"audio_reinjection", "serial_audio_injection"}:
                    counts["audit_keys"] += 1
                    continue
                if isinstance(key, ast.Constant) and key.value == "new_parameter_count":
                    value = ast.Constant(value=2444930)
                keep.append((key, value))
            node.keys, node.values = [k for k, _ in keep], [v for _, v in keep]
            return self.generic_visit(node)

        def visit_Constant(self, node):
            if node.value == "each input projection/gate and intermediate MLP/gate must change during training":
                return ast.Constant(value="each new CED layer MLP/gate must change during training")
            return node

    new = Strip().visit(new)
    expected = {"contract_guard": 1, "collect": 1, "diagnostics": 1, "rank_contract": 1,
                "new_role_predicate": 2, "opened_audit": 1, "end_roles": 1, "audit_keys": 2}
    if counts != expected:
        raise ValueError(f"unexpected trainer additions: {counts}")
    equal(new, old, "complete train function including data/epoch/update/optimizer/resume")
    return {"complete_train_function_ast_equal_after_explicit_audits": True, "explicit_audit_nodes": counts}


def audit_parent():
    changed = {"audit_reference.py", "contract.py", "models/recursive_layer_inject.py",
               "models/recursive_text.py", "optim.py", "runtime_checks.py", "train.py", "write_config.py"}
    inventory = lambda root: {p.relative_to(root).as_posix() for p in root.rglob("*")
                              if p.suffix in {".py", ".sh"} and "__pycache__" not in p.parts}
    if inventory(ROOT) != inventory(PARENT):
        raise ValueError("route must contain the complete copied parent source inventory")
    identical = []
    for name in sorted(inventory(ROOT) - changed):
        old, new = [(p / name).read_text(encoding="utf-8-sig") for p in (PARENT, ROOT)]
        expected = old.replace("layer8_4_inject", "layer8_4_inject_audio_reinject").replace("layer84", "layer84_audio_reinject")
        if new != expected:
            raise ValueError("unexpected copied-file change: " + name)
        identical.append(name)
    functions = {
        "train.py": ("answer_loss", "move_batch", "rng_state", "restore_rng", "save_checkpoint",
                     "reload_audit", "inspect_tokenizer", "audit_checkpoint_tokenizer", "main"),
        "contract.py": ("geometry", "data_order", "rank_seed", "raw_references", "lr_factor",
                        "validate_recursive_config_values", "recursive_checkpoint_contract",
                        "validate_tokenizer_contract", "validate_checkpoint_tokenizer", "validate_ced_assets"),
        "models/recursive_text.py": ("validate_recursive_config", "validate_loaded_model", "validate_layer_injection"),
    }
    for name, methods in functions.items():
        for method in methods:
            equal(find_function(tree(ROOT / name), method), find_function(tree(PARENT / name), method), name + ":" + method)
    for filename, class_name in (("optim.py", "GradNormTracker"),):
        nodes = [next(n for n in tree(p / filename).body if isinstance(n, ast.ClassDef) and n.name == class_name)
                 for p in (ROOT, PARENT)]
        equal(*nodes, class_name)
    opts = [find_function(tree(p / "optim.py"), "build_optimizer") for p in (ROOT, PARENT)]
    for node in opts:
        node.body = [n for n in node.body if not isinstance(n, ast.If)]
    equal(*opts, "AdamW/LambdaLR/group construction after role coverage guard")
    scope = {"__file__": str(PARENT / "contract.py")}
    execute(tree(PARENT / "contract.py").body, scope)
    prepared = dict(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS,
                    rows_canonical_sha256="synthetic_rows", mapping_sha256="synthetic_mapping")
    old, new = scope["semantic_contract"](prepared, "text", "ced"), semantic_contract(prepared, "text", "ced")
    for value in (old, new):
        for key in ("route", "code_sha256", "recursive_model_source_sha256",
                    "audio_reinjection", "serial_audio_injection"):
            value.pop(key, None)
    if new != old:
        raise ValueError("inherited training recipe differs from 8->4 parent")
    input_scope = {"__file__": str(INPUT_PARENT / "contract.py")}
    execute(tree(INPUT_PARENT / "contract.py").body, input_scope)
    if AUDIO_REINJECTION != input_scope["AUDIO_REINJECTION"]:
        raise ValueError("original shared input reinjection contract changed")
    return {"identical_source_files": identical, "inherited_function_asts": functions,
            "common_recipe_equal": True, "input_contract_equal": True}


def audit():
    return {"status": "PASS", "scope": "source audit only; no tensor/remote validation",
            "route_code_sha256": code_digest(), "protected_originals": protected_parents(),
            "parent_parity": audit_parent(), "backbone_parity": audit_backbone(),
            "training_parity": audit_training(), "serial_contract": SERIAL_INJECTION_CONTRACT}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reference_audit.json")
    args = parser.parse_args()
    result = audit()
    write_json(args.output, result)
    print(result["status"], args.output)
