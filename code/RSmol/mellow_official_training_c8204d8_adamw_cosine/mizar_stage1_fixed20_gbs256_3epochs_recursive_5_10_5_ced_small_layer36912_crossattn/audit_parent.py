"""Read-only parent parity audit. This does not load tensors or remote assets."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / "mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small"
BACKBONE = ROOT.parents[1] / "recursive_model_5_10_5.py"


def source(path):
    return path.read_text(encoding="utf-8")


def tree(path):
    return ast.parse(source(path))


def function(module, name):
    return next(node for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)


def dump(node):
    return ast.dump(node, include_attributes=False)


class RemoveCrossAttention(ast.NodeTransformer):
    """Remove only the explicit new arguments, guards, adapter call and trace."""
    def visit_FunctionDef(self, node):
        if node.name == "enable_audio_cross_attention":
            return None
        if node.name == "forward":
            for arguments, defaults in ((node.args.args, node.args.defaults),
                                         (node.args.kwonlyargs, node.args.kw_defaults)):
                for index in range(len(arguments) - 1, -1, -1):
                    if arguments[index].arg in {"audio_grids", "prompt_mask"}:
                        default_index = index - (len(arguments) - len(defaults))
                        arguments.pop(index)
                        if default_index >= 0:
                            defaults.pop(default_index)
        return self.generic_visit(node)

    def visit_If(self, node):
        expression = ast.unparse(node.test)
        if expression in {'not hasattr(self, \'audio_crossattn\')',
                           'use_cache or past_key_values is not None',
                           'inputs_embeds is None or input_ids is not None',
                           'logical_index in (5, 15)'}:
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(ast.unparse(target) in {"audio_bank", "self.last_crossattn_trace"} for target in node.targets):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [keyword for keyword in node.keywords if keyword.arg not in {"audio_grids", "prompt_mask"}]
        return self.generic_visit(node)


class RemoveTrainingDiagnostics(ast.NodeTransformer):
    """Allow only the new diagnostics inside the original optimizer update loop."""
    def visit_Assign(self, node):
        targets = {ast.unparse(target) for target in node.targets}
        if targets & {"adapter.collect_diagnostics"}:
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if ast.unparse(node.test) in {"step >= 1 and opened_gate_audit is None",
                                     "step + 1 == opened_gate_audit_deadline",
                                     "adapter.collect_diagnostics"}:
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "collective_require"
                and node.value.args and ast.unparse(node.value.args[0]) ==
                "model.caption_decoder.lm.model.last_crossattn_trace == (5, 15)"):
            return None
        return self.generic_visit(node)


def audit():
    unchanged = ["models/audio.py", "models/ced_offline.py", "utils/__init__.py", "utils/io.py",
                 "check_ced_assets.py", "prepare_index.py", "stage_data.py"]
    # Discover the actual complete data/utils inventory, rather than omitting new files.
    unchanged = sorted(set(unchanged) | {
        str(path.relative_to(PARENT)).replace("\\", "/")
        for folder in ("data", "utils") for path in (PARENT / folder).rglob("*.py")})
    unchanged = [name for name in unchanged if (PARENT / name).is_file()]
    inventory = {}
    for name in unchanged:
        before, after = source(PARENT / name), source(ROOT / name)
        if before != after:
            raise AssertionError(f"parent data/runtime implementation changed: {name}")
        inventory[name] = hashlib.sha256(before.encode("utf-8")).hexdigest()
    base, candidate = tree(BACKBONE), tree(ROOT / "models/recursive_crossattn.py")
    for name in ("LogicalSlotCacheView", "RecursiveLlama5_10_5Model", "RecursiveLlamaForCausalLM"):
        stripped = RemoveCrossAttention().visit(copy.deepcopy(function(candidate, name)))
        if dump(function(base, name)) != dump(stripped):
            raise AssertionError(f"backbone changed beyond explicit cross attention: {name}")
    for node in base.body:
        if isinstance(node, ast.FunctionDef) and dump(node) != dump(function(candidate, node.name)):
            raise AssertionError(f"original recursive helper changed: {node.name}")
    original_train = function(tree(PARENT / "train.py"), "train")
    candidate_train = function(tree(ROOT / "train.py"), "train")
    original_loop = next(node for node in original_train.body if isinstance(node, ast.For))
    candidate_loop = next(node for node in candidate_train.body if isinstance(node, ast.For))
    if dump(original_loop) != dump(RemoveTrainingDiagnostics().visit(copy.deepcopy(candidate_loop))):
        raise AssertionError("original epoch/data/optimizer update loop changed beyond cross attention diagnostics")
    trainer_equal = ["collective_require", "rng_state", "restore_rng", "tensor_digest", "weights_changed",
                     "move_batch", "answer_loss", "inspect_tokenizer", "audit_checkpoint_tokenizer",
                     "probe_sources", "save_checkpoint", "reload_audit", "main"]
    for name in trainer_equal:
        if dump(function(tree(PARENT / "train.py"), name)) != dump(function(tree(ROOT / "train.py"), name)):
            raise AssertionError(f"original training helper changed: {name}")
    for name in ("CEDFrequencyMerger", "CEDTemporalMapper"):
        if dump(function(tree(PARENT / "models/ced.py"), name)) != dump(function(tree(ROOT / "models/ced.py"), name)):
            raise AssertionError(f"original mapper changed: {name}")
    for name in ("_waveform_to_mel", "_prepare_ced_input", "train"):
        base_class = function(tree(PARENT / "models/ced.py"), "CEDSmallWrapper")
        new_class = function(tree(ROOT / "models/ced.py"), "CEDSmallWrapper")
        if dump(function(base_class, name)) != dump(function(new_class, name)):
            raise AssertionError(f"CED preprocessing/mode changed: {name}")
    for name in ("GradNormTracker",):
        if dump(function(tree(PARENT / "optim.py"), name)) != dump(function(tree(ROOT / "optim.py"), name)):
            raise AssertionError("GradNormTracker changed")
    base_contract, new_contract = tree(PARENT / "contract.py"), tree(ROOT / "contract.py")
    for name in ("geometry", "lr_factor", "data_order", "rank_seed", "raw_references"):
        if dump(function(base_contract, name)) != dump(function(new_contract, name)):
            raise AssertionError(f"original training recipe changed: {name}")
    import contract
    return {"status": "PASS", "scope": "source/AST parity only; not tensor, weight-loading or GPU evidence",
            "parent": PARENT.name, "route_code_sha256": contract.code_digest(),
            "identical_files": inventory, "backbone_classes_ast_equal_after_additions": True,
            "complete_epoch_update_loop_ast_equal_after_diagnostics": True,
            "unchanged_training_helpers": trainer_equal, "original_mapper_ast_equal": True,
            "original_preprocessing_and_gradnorm_ast_equal": True}


if __name__ == "__main__":
    result = audit()
    (ROOT / "reference_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
