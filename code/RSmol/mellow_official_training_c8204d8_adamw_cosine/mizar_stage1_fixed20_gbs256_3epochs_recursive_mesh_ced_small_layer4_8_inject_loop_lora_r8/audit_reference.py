"""Audit the entire copied trainer/backbone against the unchanged serial parent."""
import ast
import copy
import hashlib
import json
from pathlib import Path

from contract import (LOOP_LORA_CONTRACT, INJECTION_CONTRACT, SOURCE_COUNTS,
                      code_digest, semantic_contract, write_json)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
PARENT = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject")
BASELINE = REPO / "audit_reports/mizar_s1_mesh_ced_loop_lora_parent_baseline.json"


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def find_function(module, name):
    found = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise ValueError(f"expected one function {name}, got {len(found)}")
    return found[0]


def execute(nodes, scope):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(module, "loop-lora-source-audit", "exec"), scope)


def equal(actual, expected, label):
    if ast.dump(actual) != ast.dump(expected):
        raise ValueError("unexpected inherited semantic difference: " + label)


def protected_parents():
    files = json.loads(BASELINE.read_text(encoding="utf-8"))["files"]
    for name, expected in files.items():
        raw = (REPO / name).read_bytes().replace(bytes([13,10]), bytes([10]))
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("protected original changed: " + name)
    return {"files": len(files), "all_unchanged": True}


def audit_backbone():
    old, new = [tree(p / "models/recursive_layer_inject.py") for p in (PARENT, ROOT)]
    counts = dict(imports=0, methods=0, arguments=0, guards=0, trace=0, middle=0, branch=0, calls=0)
    class Strip(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == "models.loop_lora":
                counts["imports"] += 1
                return None
            return node
        def visit_FunctionDef(self, node):
            if node.name == "enable_loop_lora":
                counts["methods"] += 1
                return None
            if node.name == "_run_stack":
                index = next(i for i,a in enumerate(node.args.kwonlyargs) if a.arg == "loop_index")
                node.args.kwonlyargs.pop(index)
                node.args.kw_defaults.pop(index)
                counts["arguments"] += 1
            return self.generic_visit(node)
        def visit_If(self, node):
            test = ast.unparse(node.test)
            if test in {"not getattr(self, 'loop_lora_enabled', False)", "self.gradient_checkpointing"} or test.startswith("middle != "):
                counts["guards"] += 1
                return None
            if test == "loop_index is not None":
                counts["branch"] += 1
                return node.orelse
            return self.generic_visit(node)
        def visit_Assign(self, node):
            name = ast.unparse(node.targets[0]) if len(node.targets) == 1 else ""
            if name == "self.last_loop_lora_trace":
                counts["trace"] += 1
                return None
            if name == "middle":
                counts["middle"] += 1
                return None
            return self.generic_visit(node)
        def visit_Call(self, node):
            for kw in list(node.keywords):
                if kw.arg == "loop_index":
                    counts["calls"] += 1
                    node.keywords.remove(kw)
            return self.generic_visit(node)
    Strip().visit(new)
    expected = dict(imports=1, methods=1, arguments=1, guards=3, trace=1, middle=1, branch=1, calls=1)
    if counts != expected:
        raise ValueError(f"unexpected MeSH backbone changes: {counts}")
    equal(new, old, "entire MeSH backbone including memory/router queries after explicit LoRA additions")
    return {"entire_parent_backbone_ast_equal_after_explicit_additions": True, "additions": counts}


def audit_training():
    old, new = [find_function(tree(p / "train.py"), "train") for p in (PARENT, ROOT)]
    counts = dict(guard=0, assignments=0, audit_keys=0, fresh=0, opened=0, messages=0, role_predicates=0, end_roles=0)
    class Strip(ast.NodeTransformer):
        def visit_If(self, node):
            if "contract.get('loop_lora')" in ast.unparse(node.test):
                counts["guard"] += 1
                return None
            return self.generic_visit(node)
        def visit_Assign(self, node):
            target = ast.unparse(node.targets[0]) if len(node.targets) == 1 else ""
            if target in {"record['loop_lora_trace']", "rank_check['loop_lora']"}:
                counts["assignments"] += 1
                return None
            return self.generic_visit(node)
        def visit_Dict(self, node):
            pairs = []
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "loop_lora":
                    counts["audit_keys"] += 1
                    continue
                if isinstance(key, ast.Constant) and key.value == "new_parameter_count":
                    value = ast.Constant(value=2444930)
                pairs.append((key, value))
            node.keys, node.values = [k for k, _ in pairs], [v for _, v in pairs]
            return self.generic_visit(node)
        def visit_BoolOp(self, node):
            node = self.generic_visit(node)
            if (isinstance(node.op, ast.And) and len(node.values) == 2
                    and ast.unparse(node.values[-1]) == "all(gradient_audit['closed_lora_B'].values())"):
                counts["fresh"] += 1
                return node.values[0]
            if (isinstance(node.op, ast.And)
                    and ast.unparse(node.values[-1]) == "not any(opened_gate_audit['closed_lora_B'].values())"):
                counts["opened"] += 1
                node.values.pop()
            return node
        def visit_Compare(self, node):
            if ast.unparse(node) == "parameter_role(name) in NEW_ROLES":
                counts["role_predicates"] += 1
                return ast.parse('parameter_role(name).startswith("layer")', mode="eval").body
            return self.generic_visit(node)
        def visit_comprehension(self, node):
            if isinstance(node.iter, ast.Name) and node.iter.id == "NEW_ROLES":
                counts["end_roles"] += 1
                node.iter = ast.Name(id="TRAINABLE_ROLES", ctx=ast.Load())
                node.ifs.append(ast.parse('role.startswith("layer")', mode="eval").body)
            return self.generic_visit(node)
        def visit_Constant(self, node):
            messages = {
                "both CED MLPs and both loop LoRA A/B pairs must receive gradients after the first update":
                    "both CED MLPs must receive gradients after the first gate update",
                "each CED MLP/gate and both loop LoRA A/B pairs must change during training":
                    "each new CED layer MLP/gate must change during training",
            }
            if isinstance(node.value, str) and node.value in messages:
                counts["messages"] += 1
                return ast.Constant(value=messages[node.value])
            return node
    Strip().visit(new)
    if counts != dict(guard=1, assignments=2, audit_keys=1, fresh=1, opened=1, messages=2, role_predicates=2, end_roles=1):
        raise ValueError(f"unexpected trainer changes: {counts}")
    equal(new, old, "complete train function including data/FP32/DDP/GA/update/save/resume")
    return {"entire_train_function_ast_equal_after_explicit_audits": True, "additions": counts}


def audit():
    inventory = lambda path: {p.relative_to(path).as_posix() for p in path.rglob("*")
                              if p.suffix in {".py", ".sh"} and "__pycache__" not in p.parts}
    if inventory(ROOT) != inventory(PARENT) | {"models/loop_lora.py"}:
        raise ValueError("missing copied parent files or unexpected new sources")
    changed = {"audit_reference.py", "contract.py", "models/recursive_layer_inject.py",
               "models/recursive_text.py", "optim.py", "runtime_checks.py", "train.py", "write_config.py"}
    identical = []
    for name in sorted(inventory(PARENT) - changed):
        expected = (PARENT / name).read_text(encoding="utf-8-sig").replace(
            "recursive_mesh_layer4_8_inject", "recursive_mesh_layer4_8_inject_loop_lora_r8").replace(
            "recursive_mesh_ced_small_layer4_8_inject", "recursive_mesh_ced_small_layer4_8_inject_loop_lora_r8")
        if expected != (ROOT / name).read_text(encoding="utf-8-sig"):
            raise ValueError("unexpected copied-file change: " + name)
        identical.append(name)
    functions = {
        "train.py": ("answer_loss", "move_batch", "rng_state", "restore_rng", "save_checkpoint",
                     "reload_audit", "inspect_tokenizer", "audit_checkpoint_tokenizer", "probe_sources", "main"),
        "contract.py": ("geometry", "data_order", "rank_seed", "raw_references", "lr_factor",
                        "recursive_checkpoint_contract", "validate_ced_assets",
                        "validate_tokenizer_contract", "validate_checkpoint_tokenizer"),
    }
    for name, methods in functions.items():
        for method in methods:
            equal(find_function(tree(ROOT / name), method), find_function(tree(PARENT / name), method), name + ":" + method)
    trackers = [next(n for n in tree(p / "optim.py").body if isinstance(n, ast.ClassDef) and n.name == "GradNormTracker")
                for p in (ROOT, PARENT)]
    equal(*trackers, "entire GradNormTracker")
    opts = [find_function(tree(p / "optim.py"), "build_optimizer") for p in (ROOT, PARENT)]
    for node in opts:
        node.body = [n for n in node.body if not isinstance(n, ast.If)]
    equal(*opts, "entire optimizer/scheduler except role coverage guard")
    scope = {"__file__": str(PARENT / "contract.py")}
    execute(tree(PARENT / "contract.py").body, scope)
    prepared = dict(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS,
                    rows_canonical_sha256="synthetic_rows", mapping_sha256="synthetic_mapping")
    old, new = scope["semantic_contract"](prepared, "text", "ced"), semantic_contract(prepared, "text", "ced")
    for value in (old, new):
        for key in ("route", "code_sha256", "recursive_model_source_sha256", "loop_lora"):
            value.pop(key, None)
        value["adamw"].pop("loop_lora_lr", None)
    if new != old:
        raise ValueError("inherited recipe differs from serial parent")
    return {"status": "PASS", "scope": "source audit only; no tensors or remote execution",
            "route_code_sha256": code_digest(), "protected_originals": protected_parents(),
            "identical_source_files_except_route_names": identical, "preserved_functions": functions,
            "training_parity": audit_training(), "backbone_parity": audit_backbone(),
            "common_recipe_equal": True, "loop_lora": LOOP_LORA_CONTRACT, "layer_injection": INJECTION_CONTRACT}


if __name__ == "__main__":
    result = audit()
    write_json(ROOT / "reference_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
