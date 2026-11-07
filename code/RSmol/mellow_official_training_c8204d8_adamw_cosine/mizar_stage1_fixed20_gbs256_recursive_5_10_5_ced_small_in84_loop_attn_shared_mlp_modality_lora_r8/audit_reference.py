"""Audit isolated shared-Attention/modality-MLP LoRA against its unchanged rank8 mother."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from contract import SOURCE_COUNTS, LOOP_LORA_CONTRACT, code_digest, semantic_contract, write_json

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
PARENT = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer8_4_inject_audio_reinject_loop_lora_r8")
BASELINE = REPO/"audit_reports/mizar_s1_attn_shared_mlp_modality_lora_r8_parent_baseline.json"


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


def audit_backbone():
    old, new = [tree(p/"models/recursive_layer_inject.py") for p in (PARENT, ROOT)]
    additions = []
    class Strip(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == "models.loop_lora":
                for alias in list(node.names):
                    if alias.name == "reset_loop_lora_trace":
                        node.names.remove(alias)
                        additions.append("reset_import")
            return node
        def visit_Expr(self, node):
            if ast.unparse(node) == "reset_loop_lora_trace(self)":
                additions.append("metadata_reset")
                return None
            return self.generic_visit(node)
        def visit_Assign(self, node):
            if ast.unparse(node.targets[0]) == "(self.last_batch_size, self.last_sequence_length)":
                additions.append("shape_metadata")
                return None
            return self.generic_visit(node)
    Strip().visit(new)
    if additions != ["reset_import", "metadata_reset", "shape_metadata"]:
        raise ValueError(f"unexpected backbone additions: {additions}")
    equal(new, old, "entire backbone except small invocation metadata")
    return {"entire_backbone_ast_equal_after_metadata": True, "additions": additions}


def audit_training():
    old, new = [find_function(tree(p/"train.py"), "train") for p in (PARENT, ROOT)]
    additions = []
    messages = {
        "input projection, both CED MLPs and all shared/modality LoRA branches must receive gradients after the first update":
            "input projection, both CED MLPs and both LoRA A/B pairs must receive gradients after the first update",
        "each serial audio branch and all shared/modality LoRA branches must change during training":
            "each serial audio branch and both loop LoRA A/B pairs must change during training"}
    class Strip(ast.NodeTransformer):
        def visit_Assign(self, node):
            if ast.unparse(node.targets[0]) == "record['lora_execution']":
                additions.append("execution_diagnostic")
                return None
            return self.generic_visit(node)
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in messages:
                additions.append("message")
                return ast.Constant(value=messages[node.value])
            return node
    Strip().visit(new)
    if additions != ["message", "execution_diagnostic", "message"]:
        raise ValueError(f"unexpected trainer additions: {additions}")
    equal(new, old, "entire training loop/update/FP32/loss/DDP/staging/save/resume")
    return {"entire_train_function_ast_equal_after_explicit_audits": True, "additions": additions}


def audit():
    inventory = lambda p: {f.relative_to(p).as_posix() for f in p.rglob("*")
                           if f.suffix in {".py", ".sh"} and "__pycache__" not in f.parts}
    if inventory(ROOT) != inventory(PARENT):
        raise ValueError("copied source inventory differs")
    changed = {"audit_reference.py", "contract.py", "models/recursive_layer_inject.py", "models/loop_lora.py",
               "optim.py", "runtime_checks.py", "train.py", "write_config.py"}
    identical = []
    for name in sorted(inventory(PARENT)-changed):
        expected = (PARENT/name).read_text(encoding="utf-8-sig").replace(
            "loop_lora_r8", "loop_attn_shared_mlp_modality_lora_r8").replace(
            PARENT.name.replace("loop_lora_r8", "loop_attn_shared_mlp_modality_lora_r8"), ROOT.name)
        if (ROOT/name).read_text(encoding="utf-8-sig") != expected:
            raise ValueError("unexpected copied source difference: " + name)
        identical.append(name)
    functions = {
        "train.py": ("collective_require", "rng_state", "restore_rng", "tensor_digest", "role_snapshots",
                     "weights_changed", "move_batch", "answer_loss", "inspect_tokenizer", "audit_checkpoint_tokenizer",
                     "probe_sources", "save_checkpoint", "reload_audit", "main"),
        "contract.py": ("geometry", "lr_factor", "data_order", "rank_seed", "raw_references",
                        "recursive_checkpoint_contract", "validate_recursive_config_values", "validate_ced_assets",
                        "validate_tokenizer_contract", "validate_checkpoint_tokenizer", "code_digest")}
    for name, methods in functions.items():
        for method in methods:
            equal(find_function(tree(ROOT/name), method), find_function(tree(PARENT/name), method), name+":"+method)
    old, new = [tree(p/"optim.py") for p in (PARENT, ROOT)]
    equal(next(n for n in old.body if isinstance(n, ast.ClassDef) and n.name == "GradNormTracker"),
          next(n for n in new.body if isinstance(n, ast.ClassDef) and n.name == "GradNormTracker"), "GradNormTracker")
    opts = [find_function(t, "build_optimizer") for t in (old, new)]
    for node in opts:
        node.body = [n for n in node.body if not isinstance(n, ast.If)]
    equal(*opts, "optimizer/scheduler except role coverage guard")
    equal(find_function(tree(ROOT/"models/loop_lora.py"), "call_with_loop_lora"),
          find_function(tree(PARENT/"models/loop_lora.py"), "call_with_loop_lora"), "exception-safe loop selection")
    scope = {"__file__": str(PARENT/"contract.py")}
    execute(tree(PARENT/"contract.py").body, scope)
    prepared = dict(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS,
                    rows_canonical_sha256="rows", mapping_sha256="mapping")
    recipes = [scope["semantic_contract"](prepared, "text", "ced"), semantic_contract(prepared, "text", "ced")]
    for recipe in recipes:
        for key in ("route", "code_sha256", "recursive_model_source_sha256", "loop_lora"):
            recipe.pop(key, None)
    if recipes[0] != recipes[1]:
        raise ValueError("non-LoRA training recipe differs")
    return {"status": "PASS", "scope": "local source audit; no real assets or GPU execution",
            "route_code_sha256": code_digest(), "protected_originals": protected_parents(),
            "checked_copied_sources": sorted(inventory(ROOT)), "identical_source_files_except_route_names": identical,
            "preserved_functions": functions, "training_parity": audit_training(), "backbone_parity": audit_backbone(),
            "common_recipe_equal": True, "loop_lora": LOOP_LORA_CONTRACT}


if __name__ == "__main__":
    result = audit()
    write_json(ROOT/"reference_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
