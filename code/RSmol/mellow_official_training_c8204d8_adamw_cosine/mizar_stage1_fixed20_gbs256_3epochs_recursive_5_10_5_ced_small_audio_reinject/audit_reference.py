"""Stdlib parity audit against the copied CED non-MeSH parent; no GPU claims.

The inherited official-reference report is provenance only. This audit executes
no tensor model and does not certify remote loading, training or decoding.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

from contract import AUDIO_REINJECTION, SOURCE_COUNTS, code_digest, semantic_contract, write_json

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small")
BACKBONE = ROOT.parents[1] / "recursive_model_5_10_5.py"


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def find_function(module, name):
    matches = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one function named {name}, got {len(matches)}")
    return matches[0]


def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "parity-audit", "exec"), scope)


def normalized_bytes(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


def equal_ast(a, b, label):
    if ast.dump(a) != ast.dump(b):
        raise ValueError(f"parent implementation changed outside audio reinjection: {label}")


def audit_backbone():
    original, actual = tree(BACKBONE), tree(ROOT / "models/recursive_reinject.py")
    actual.body[0] = copy.deepcopy(original.body[0])  # module documentation only
    core = next(n for n in actual.body if isinstance(n, ast.ClassDef) and n.name == "RecursiveLlama5_10_5Model")
    core.body = [n for n in core.body if not isinstance(n, ast.FunctionDef)
                 or n.name not in {"enable_audio_reinjection", "inject_audio", "reinjection_diagnostics"}]
    forward = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    added_assignments = {"audio_information", "self.last_reinject_trace", "self.last_reinject_stats"}
    added_tests = {
        "not hasattr(self, 'audio_linear')", "input_ids is not None or inputs_embeds is None",
        "use_cache or past_key_values is not None", "inputs_embeds.shape[1] < self.audio_reinject_tokens",
        "logical_index in self.audio_reinject_indices",
    }
    counts = {"assignments": 0, "guards": 0}
    def strip(node):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                kept = []
                for item in value:
                    if isinstance(item, ast.Assign) and len(item.targets) == 1 and ast.unparse(item.targets[0]) in added_assignments:
                        counts["assignments"] += 1
                        continue
                    if isinstance(item, ast.If) and ast.unparse(item.test) in added_tests:
                        counts["guards"] += 1
                        continue
                    if isinstance(item, ast.AST):
                        strip(item)
                    kept.append(item)
                setattr(node, field, kept)
            elif isinstance(value, ast.AST):
                strip(value)
    strip(forward)
    if counts != {"assignments": 3, "guards": 5}:
        raise ValueError(f"unexpected backbone injection delta: {counts}")
    equal_ast(actual, original, "entire recursive implementation after removing explicit injection additions")
    return {"baseline": str(BACKBONE.relative_to(ROOT.parents[3])).replace("\\", "/"),
            "entire_backbone_ast_equal_after_explicit_additions": True, "added_forward_nodes": counts}


def audit_parent():
    files = ["prepare_index.py", "stage_data.py", "check_ced_assets.py", "utils/audio_io.py",
             "models/audio.py", "models/ced.py", "models/ced_offline.py", "models/mellow.py", "models/decoder.py"]
    files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "data").glob("*.py"))]
    for name in files:
        if normalized_bytes(ROOT / name) != normalized_bytes(PARENT / name):
            raise ValueError(f"parent audio/data implementation changed: {name}")
    pairs = {"train.py": ("answer_loss", "move_batch", "rng_state", "restore_rng", "probe_sources",
                          "save_checkpoint", "reload_audit", "inspect_tokenizer", "audit_checkpoint_tokenizer", "main"),
             "contract.py": ("geometry", "data_order", "rank_seed", "raw_references", "lr_factor",
                             "validate_recursive_config_values", "recursive_checkpoint_contract",
                             "validate_tokenizer_contract", "validate_checkpoint_tokenizer",
                             "ced_label_metadata", "validate_ced_config", "validate_ced_assets", "code_digest"),
             "optim.py": ("_clip_grad", "track_and_clip_", "state_dict", "load_state_dict"),
             "models/recursive_text.py": ("validate_recursive_config", "validate_loaded_model")}
    for name, functions in pairs.items():
        for function in functions:
            equal_ast(find_function(tree(ROOT/name), function), find_function(tree(PARENT/name), function), name+":"+function)
    def epoch_loop(path, strip):
        function = find_function(tree(path), "train")
        loop = copy.deepcopy(next(n for n in function.body if isinstance(n, ast.For)))
        if strip:
            for node in ast.walk(loop):
                for field, value in ast.iter_fields(node):
                    if isinstance(value, list):
                        setattr(node, field, [item for item in value if not (
                            isinstance(item, ast.Assign) and any(ast.unparse(t) == "model.caption_decoder.lm.model.collect_reinject_stats" for t in item.targets)
                            or isinstance(item, ast.If) and ast.unparse(item.test) in {
                                "step == 1", "model.caption_decoder.lm.model.collect_reinject_stats"})])
        return loop
    equal_ast(epoch_loop(ROOT/"train.py", True), epoch_loop(PARENT/"train.py", False), "complete epoch/update/microbatch loop")
    # Optimizer construction changes only the new role assignment and accepted roles.
    old = find_function(tree(PARENT/"optim.py"), "build_optimizer")
    new = find_function(tree(ROOT/"optim.py"), "build_optimizer")
    def optimizer_nodes(function):
        return [n for n in ast.walk(function) if isinstance(n, ast.Assign) and any(ast.unparse(t) == "lr" for t in n.targets)], [n for n in ast.walk(function) if isinstance(n, ast.If) and "for key in groups" in ast.unparse(n.test)]
    (old_lr, old_roles), (new_lr, new_roles) = optimizer_nodes(old), optimizer_nodes(new)
    if any(len(x) != 1 for x in (old_lr, old_roles, new_lr, new_roles)):
        raise ValueError("optimizer delta could not be identified")
    new_lr[0].value = copy.deepcopy(old_lr[0].value)
    new_roles[0].test, new_roles[0].body = copy.deepcopy(old_roles[0].test), copy.deepcopy(old_roles[0].body)
    equal_ast(new, old, "AdamW construction, decay and LambdaLR")
    old_scope = {"__file__": str(PARENT/"contract.py")}
    execute(tree(PARENT/"contract.py").body, old_scope)
    prepared = dict(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS,
                    rows_canonical_sha256="synthetic_rows", mapping_sha256="synthetic_mapping")
    expected = old_scope["semantic_contract"](prepared, "same_text", "same_ced")
    actual = semantic_contract(prepared, "same_text", "same_ced")
    ignored = {"route", "code_sha256", "recursive_model_source_sha256", "audio_reinjection"}
    common = sorted(set(expected) - ignored)
    if set(actual) - set(expected) != {"audio_reinjection"}:
        raise ValueError("unexpected new recipe fields")
    for key in common:
        if actual[key] != expected[key]:
            raise ValueError(f"common training contract changed: {key}")
    if actual["audio_reinjection"] != AUDIO_REINJECTION:
        raise ValueError("audio reinjection contract mismatch")
    return {"identical_files_normalized_lf": files, "identical_function_asts": pairs,
            "identical_common_contract_fields": common,
            "optimizer_epoch_loop": "AST-identical after removing only reinjection diagnostics",
            "optimizer": "same original groups/LR/decay/AdamW/LambdaLR; two explicit new roles LR6e-4"}


def audit():
    inherited = json.loads((PARENT / "reference_audit.json").read_text(encoding="utf-8"))
    return {"status": "PASS", "route_code_sha256": code_digest(), "parent_route": PARENT.name,
            "parent_parity": audit_parent(), "backbone_parity": audit_backbone(),
            "audio_reinjection": AUDIO_REINJECTION,
            "parent_code_sha256": {p.relative_to(PARENT).as_posix(): hashlib.sha256(normalized_bytes(p)).hexdigest()
                                   for p in sorted(PARENT.rglob("*")) if p.suffix in {".py", ".sh"}},
            "inherited_official_reference": {"status": inherited["status"],
                "route_code_sha256": inherited["route_code_sha256"],
                "report_sha256": hashlib.sha256((PARENT/"reference_audit.json").read_bytes()).hexdigest(),
                "scope": "parent historical stdlib evidence; not rerun here"},
            "scope": "local stdlib file/AST/config comparisons only; not tensor, weight-loading or GPU training PASS"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))
