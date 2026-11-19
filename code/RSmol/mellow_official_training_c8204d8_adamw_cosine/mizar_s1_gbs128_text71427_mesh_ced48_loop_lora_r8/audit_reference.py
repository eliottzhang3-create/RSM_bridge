"""Offline source inheritance audit; no model assets or GPU execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from contract import SOURCE_COUNTS, code_digest, semantic_contract, write_json

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
PARENT = ROOT.with_name("mizar_s1_gbs128_text71427_mesh_ced_small")
REFERENCE = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_inject_loop_lora_r8")


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def find_function(module, name):
    found = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise ValueError(f"expected one function {name}, got {len(found)}")
    return found[0]


def execute(nodes, scope):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(module, "text71427-ced48-source-audit", "exec"), scope)


def equal(a, b, label):
    if ast.dump(a) != ast.dump(b):
        raise ValueError("unexpected inherited source change: " + label)


def protected_parents():
    files = json.loads((ROOT / "parent_snapshot.json").read_text(encoding="utf-8"))["sha256_normalized_lf"]
    for name, expected in files.items():
        raw = (REPO / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("protected source changed: " + name)
    return {"files": len(files), "all_unchanged": True}


def audit_backbone():
    old = tree(REPO / "code/RSmol/recursive_model_mizar_text_7_14x2_7_mesh.py")
    new = tree(ROOT / "models/recursive_layer_inject.py")

    class Strip(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None if node.module == "models.loop_lora" else node

        def visit_FunctionDef(self, node):
            if node.name in {"enable_loop_lora", "enable_layer_injection", "inject_audio", "injection_diagnostics"}:
                return None
            for index in reversed(range(len(node.args.args))):
                if node.args.args[index].arg == "audio_information":
                    default_index = index - (len(node.args.args) - len(node.args.defaults))
                    node.args.args.pop(index)
                    node.args.defaults.pop(default_index)
            for index in reversed(range(len(node.args.kwonlyargs))):
                if node.args.kwonlyargs[index].arg in {"audio_information", "loop_index"}:
                    node.args.kwonlyargs.pop(index)
                    node.args.kw_defaults.pop(index)
            return self.generic_visit(node)

        def visit_If(self, node):
            test = ast.unparse(node.test)
            if (test in {"not getattr(self, 'loop_lora_enabled', False)", "self.gradient_checkpointing",
                         "not hasattr(self, 'audio_gates')", "loop == 0"}
                    or test.startswith(("inputs_embeds is None or input_ids", "inputs_embeds.shape[1] <",
                                        "not isinstance(audio_information", "middle !="))):
                return None
            if test == "loop_index is not None":
                return node.orelse
            return self.generic_visit(node)

        def visit_For(self, node):
            if ast.unparse(node.iter) == "audio_information":
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            target = ast.unparse(node.targets[0])
            if target in {"expected", "middle", "self.last_injection_trace", "self.last_injection_stats", "self.last_loop_lora_trace"}:
                return None
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "self.inject_audio":
                return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            node.keywords = [k for k in node.keywords if k.arg not in {"audio_information", "loop_index"}]
            return self.generic_visit(node)

    equal(Strip().visit(new), old, "entire 28-layer backbone after explicit audio/LoRA additions")
    return {"entire_parent_backbone_ast_equal_after_explicit_additions": True}


def audit_training():
    old, new = [find_function(tree(p / "train.py"), "train") for p in (PARENT, ROOT)]

    class Strip(ast.NodeTransformer):
        def visit_If(self, node):
            test = ast.unparse(node.test)
            if ("contract.get('layer_injection')" in test or test in {"step == 1", "step - segment_start_step >= 2",
                    "model.caption_decoder.lm.model.collect_injection_stats"}):
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            target = ast.unparse(node.targets[0])
            if target in {"opened_gate_audit", "segment_start_step", "injection_initial_sha",
                          "model.caption_decoder.lm.model.collect_injection_stats"} or target.startswith("rank_check['") and any(
                    role in target for role in ("opened_gate_gradient", "layer_injection", "loop_lora", "injection_initial", "injection_final")):
                return None
            return self.generic_visit(node)

        def visit_Dict(self, node):
            removed = {"layer_injection", "loop_lora", "new_parameter_count", "injection_initial_sha256"}
            pairs = [(k, v) for k, v in zip(node.keys, node.values) if not isinstance(k, ast.Constant) or k.value not in removed]
            node.keys, node.values = [k for k, _ in pairs], [v for _, v in pairs]
            return self.generic_visit(node)

        def visit_Call(self, node):
            if ast.unparse(node.func) == "collective_require" and node.args and isinstance(node.args[0], ast.BoolOp):
                if ast.unparse(node.args[0]).startswith("gradient_audit['passed'] and"):
                    node.args[0] = ast.parse("gradient_audit['passed']", mode="eval").body
            return self.generic_visit(node)

    equal(Strip().visit(new), old, "entire GBS128 train function after explicit branch diagnostics")
    return {"entire_train_function_ast_equal_after_explicit_audits": True}


def audit():
    identical = []
    for name in ["prepare_index.py", "stage_data.py", "utils/audio_io.py", "models/audio.py", "models/ced_offline.py"] + [p.relative_to(PARENT).as_posix() for p in sorted((PARENT / "data").glob("*.py"))]:
        if (ROOT / name).read_text(encoding="utf-8-sig") != (PARENT / name).read_text(encoding="utf-8-sig"):
            raise ValueError("parent data/staging changed: " + name)
        identical.append(name)
    for name in ("models/ced.py", "models/decoder.py", "models/mellow.py"):
        equal(tree(ROOT / name), tree(REFERENCE / name), name)
    for name, methods in {
        "train.py": ("answer_loss", "move_batch", "rng_state", "restore_rng", "save_checkpoint", "reload_audit",
                     "recursive_forward_audit", "recursive_backward_audit", "router_gradient_audit", "inspect_tokenizer", "probe_sources"),
        "contract.py": ("geometry", "lr_factor", "data_order", "rank_seed", "raw_references", "recursive_checkpoint_contract", "recursive_model_source_sha256"),
        "models/recursive_text.py": ("validate_recursive_config", "validate_parameter_audit", "validate_loading_info", "audit_checkpoint_tensors"),
    }.items():
        for method in methods:
            equal(find_function(tree(ROOT / name), method), find_function(tree(PARENT / name), method), name + ":" + method)
    equal(next(n for n in tree(ROOT / "optim.py").body if isinstance(n, ast.ClassDef)),
          next(n for n in tree(PARENT / "optim.py").body if isinstance(n, ast.ClassDef)), "GradNormTracker")
    scope = {"__file__": str(PARENT / "contract.py")}
    execute(tree(PARENT / "contract.py").body, scope)
    prepared = dict(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS,
                    rows_canonical_sha256="synthetic_rows", mapping_sha256="synthetic_mapping")
    before, after = scope["semantic_contract"](prepared, "text", "ced"), semantic_contract(prepared, "text", "ced")
    for recipe in (before, after):
        for key in ("route", "code_sha256", "layer_injection", "loop_lora"):
            recipe.pop(key, None)
        recipe["adamw"].pop("new_parameters_lr", None)
        recipe["adamw"].pop("loop_lora_lr", None)
    if before != after:
        raise ValueError("inherited GBS128 semantic recipe changed")
    return {"status": "PASS", "route_code_sha256": code_digest(),
            "scope": "local source inheritance only; no production tensors or GPU execution",
            "protected_originals": protected_parents(), "identical_source_files_except_route_names": identical,
            "training_parity": audit_training(), "backbone_parity": audit_backbone(), "common_recipe_equal": True}


audit_parent = audit

if __name__ == "__main__":
    result = audit()
    write_json(ROOT / "reference_audit.json", result)
    print(json.dumps(result, indent=2))
