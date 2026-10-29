"""Audit unchanged parent semantics; this is not tensor or remote GPU validation."""
import argparse
import ast
import copy
import hashlib
from pathlib import Path

from contract import SOURCE_COUNTS, code_digest, semantic_contract, write_json

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small")
BACKBONE = ROOT.parents[1] / "recursive_model_5_10_5.py"


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def find_function(module, name):
    matches = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one function {name}, got {len(matches)}")
    return matches[0]


def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "layer84-audit", "exec"), scope)


def equal(a, b, label):
    if ast.dump(a) != ast.dump(b):
        raise ValueError(f"unexpected parent semantic difference: {label}")


def audit_backbone():
    original, actual = tree(BACKBONE), tree(ROOT / "models/recursive_layer_inject.py")
    actual.body[0] = copy.deepcopy(original.body[0])
    core = next(n for n in actual.body if isinstance(n, ast.ClassDef) and n.name == "RecursiveLlama5_10_5Model")
    core.body = [n for n in core.body if not isinstance(n, ast.FunctionDef)
                 or n.name not in {"enable_layer_injection", "inject_audio", "injection_diagnostics"}]
    forward = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    guards = {"not hasattr(self, 'audio_gates')",
              "inputs_embeds is None or input_ids is not None or use_cache or (past_key_values is not None)",
              "inputs_embeds.shape[1] < self.layer_inject_tokens",
              "not isinstance(audio_information, (tuple, list)) or len(audio_information) != 2",
              "logical_index in self.layer_inject_indices"}
    guards = {ast.dump(ast.parse(value, mode="eval").body) for value in guards}
    counts = {"assign": 0, "guard": 0, "for": 0}

    class Strip(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and ast.unparse(node.targets[0]) in {
                    "expected", "self.last_injection_trace", "self.last_injection_stats"}:
                counts["assign"] += 1
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            if ast.dump(node.test) in guards:
                counts["guard"] += 1
                return None
            return self.generic_visit(node)

        def visit_For(self, node):
            if ast.unparse(node.iter) == "audio_information":
                counts["for"] += 1
                return None
            return self.generic_visit(node)

    Strip().visit(forward)
    if counts != {"assign": 3, "guard": 5, "for": 1}:
        raise ValueError(f"unexpected injection delta: {counts}")
    for node in ast.walk(actual):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            index = next(i for i, arg in enumerate(node.args.args) if arg.arg == "audio_information")
            default_index = index - (len(node.args.args) - len(node.args.defaults))
            node.args.args.pop(index)
            node.args.defaults.pop(default_index)
        if isinstance(node, ast.Call):
            node.keywords = [kw for kw in node.keywords if kw.arg != "audio_information"]
    equal(actual, original, "entire recursive backbone after removing explicit additions")
    return {"entire_backbone_ast_equal_after_explicit_additions": True, "added_nodes": counts}


def audit_parent():
    files = ["prepare_index.py", "stage_data.py", "check_ced_assets.py", "utils/audio_io.py",
             "models/audio.py", "models/ced_offline.py"]
    files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "data").glob("*.py"))]
    for name in files:
        if (ROOT/name).read_bytes().replace(b"\r\n", b"\n") != (PARENT/name).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"parent data/staging file changed: {name}")
    functions = {
        "train.py": ("answer_loss", "move_batch", "rng_state", "restore_rng", "save_checkpoint",
                     "reload_audit", "inspect_tokenizer", "audit_checkpoint_tokenizer", "main"),
        "contract.py": ("geometry", "data_order", "rank_seed", "raw_references", "lr_factor",
                        "validate_recursive_config_values", "recursive_checkpoint_contract",
                        "validate_tokenizer_contract", "validate_checkpoint_tokenizer", "validate_ced_assets"),
        "optim.py": ("_clip_grad", "track_and_clip_", "state_dict", "load_state_dict"),
        "models/ced.py": ("_waveform_to_mel", "_prepare_ced_input", "_hidden_to_temporal_grid", "train"),
        "models/recursive_text.py": ("validate_recursive_config", "validate_loaded_model"),
        "models/decoder.py": ("_make_prefix", "generate_prefix_inference"),
    }
    for name, methods in functions.items():
        for method in methods:
            equal(find_function(tree(ROOT/name), method), find_function(tree(PARENT/name), method), name+":"+method)
    for cls in ("CEDFrequencyMerger", "CEDTemporalMapper"):
        a, b = [next(n for n in tree(p/"models/ced.py").body if isinstance(n, ast.ClassDef) and n.name == cls)
                for p in (ROOT, PARENT)]
        equal(a, b, cls)
    old_loop = next(n for n in find_function(tree(PARENT/"train.py"), "train").body if isinstance(n, ast.For))
    new_loop = next(n for n in find_function(tree(ROOT/"train.py"), "train").body if isinstance(n, ast.For))
    old_gradient = next(n for n in ast.walk(old_loop) if isinstance(n, ast.If) and ast.unparse(n.test) == "gradient_audit is None")

    class StripDiagnostics(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(ast.unparse(t) == "model.caption_decoder.lm.model.collect_injection_stats" for t in node.targets):
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            if ast.unparse(node.test) in {"step == 1", "model.caption_decoder.lm.model.collect_injection_stats"}:
                return None
            if ast.unparse(node.test) == "gradient_audit is None":
                return copy.deepcopy(old_gradient)
            return self.generic_visit(node)

    equal(StripDiagnostics().visit(new_loop), old_loop, "complete sampling/staging/optimizer update loop")
    scope = {"__file__": str(PARENT/"contract.py")}
    execute(tree(PARENT/"contract.py").body, scope)
    prepared = dict(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS,
                    rows_canonical_sha256="synthetic_rows", mapping_sha256="synthetic_mapping")
    old = scope["semantic_contract"](prepared, "text", "ced")
    new = semantic_contract(prepared, "text", "ced")
    for key in ("route", "code_sha256", "recursive_model_source_sha256", "layer_injection"):
        old.pop(key, None)
        new.pop(key, None)
    new["adamw"].pop("new_parameters_lr")
    if new != old:
        raise ValueError("common training contract differs from mother")
    return {"identical_files_normalized_lf": files, "identical_function_asts": functions,
            "optimizer_epoch_loop": "identical except explicit diagnostic additions"}


def audit_order_swap():
    """Compare the full copied implementation with the immediate 4->8 parent."""
    parent = ROOT.with_name(ROOT.name.replace("layer8_4_inject", "layer4_8_inject"))
    replacements = {
        "contract.py": [('"ced_layers_1based": [4, 8]', '"ced_layers_1based": [8, 4]')],
        "models/recursive_layer_inject.py": [
            ("branch = self.layer_inject_indices.index(logical_index)",
             "branch = 1 - self.layer_inject_indices.index(logical_index)")],
        "models/recursive_text.py": [('"ced_layers": [4, 8]', '"ced_layers": [8, 4]')],
        "train.py": [('audit["trace"] == [[5, 4], [15, 8]]',
                      'audit["trace"] == [[5, 8], [15, 4]]')],
        "write_config.py": [
            ("CED block4/block8 outputs -> independent 1536-576-576 MLPs -> independent zero-initialized tanh gates before logical layers5/15; new parameters maxLR6e-4",
             "CED block8 then block4 outputs -> their independent 1536-576-576 MLPs and zero-initialized tanh gates before logical layers5/15 respectively; explicit input pair stays (A4,A8); new parameters maxLR6e-4")],
    }
    def source_files(root):
        return {p.relative_to(root).as_posix() for p in root.rglob("*")
                if p.suffix in {".py", ".sh"} and "__pycache__" not in p.parts}
    files = source_files(parent)
    if source_files(ROOT) != files:
        raise ValueError("8->4 route does not contain the complete parent source inventory")
    for name in sorted(files):
        expected = (parent/name).read_text(encoding="utf-8-sig")
        expected = expected.replace("layer4_8_inject", "layer8_4_inject").replace("layer48", "layer84")
        actual = (ROOT/name).read_text(encoding="utf-8-sig")
        for old, new in replacements.get(name, ()):
            if expected.count(old) != 1:
                raise ValueError(f"parent order-swap anchor changed: {name}")
            expected = expected.replace(old, new)
        if name.endswith(".py"):
            expected_tree, actual_tree = ast.parse(expected), ast.parse(actual)
            if name == "audit_reference.py":
                # Only the new parity audit and its report entry are added here.
                for module in (expected_tree, actual_tree):
                    module.body = [n for n in module.body if not isinstance(n, ast.FunctionDef)
                                   or n.name not in {"audit", "audit_order_swap"}]
            equal(actual_tree, expected_tree, "4->8 parent copy:" + name)
        elif actual != expected:
            raise ValueError(f"unexpected copied launcher difference: {name}")
    return {"parent_route": parent.name, "compared_source_files": sorted(files),
            "allowed_semantic_changes": sorted(replacements),
            "injection_trace": [[5, 8], [15, 4]],
            "feature_pair": ["A4", "A8"], "module_registration_order_unchanged": True}


def audit():
    return {"status": "PASS", "scope": "stdlib source audit; not tensor or remote GPU validation",
            "route_code_sha256": code_digest(), "parent_parity": audit_parent(), "backbone_parity": audit_backbone(),
            "order_swap_parity": audit_order_swap(),
            "parent_file_sha256": {p.relative_to(PARENT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(PARENT.rglob("*")) if p.suffix in {".py", ".sh", ".md"}}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"reference_audit.json")
    args = parser.parse_args()
    result = audit()
    write_json(args.output, result)
    print(result["status"], args.output)
