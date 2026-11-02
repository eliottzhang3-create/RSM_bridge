"""Source parity audit against the untouched split3 training route."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
from contract import SOURCE_COUNTS, code_digest, semantic_contract, write_json

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.with_name(ROOT.name.removesuffix("_prompt_mix"))
REPO = ROOT.parents[3]

def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))

def find_function(module, name):
    matches = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one function {name}, got {len(matches)}")
    return matches[0]

def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "mix-audit", "exec"), scope)

def equal(actual, expected, label):
    if ast.dump(actual) != ast.dump(expected):
        diff = "\n".join(difflib.unified_diff(ast.unparse(expected).splitlines(), ast.unparse(actual).splitlines()))
        raise ValueError(f"unexpected semantic difference: {label}\n{diff}")

def normalized(name):
    return (ROOT/name).read_text(encoding="utf-8-sig").replace("audio7slot_split3_prompt_mix", "audio7slot_split3")

def audit_protected():
    files = json.loads((ROOT/"protected_sources.json").read_text(encoding="utf-8"))
    for name, expected in files.items():
        actual = hashlib.sha256((REPO/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError(f"protected original changed: {name}")
    return {"original_files": len(files), "all_normalized_sha256_match": True}

def audit_parent():
    changed = {"models/decoder.py", "models/mellow.py", "models/recursive_mesh_listen.py",
               "models/recursive_text.py", "optim.py", "contract.py", "train.py",
               "write_config.py", "audit_reference.py"}
    files = []
    for path in sorted(PARENT.rglob("*")):
        name = path.relative_to(PARENT).as_posix()
        if path.suffix in {".py", ".sh"} and name not in changed:
            if normalized(name) != path.read_text(encoding="utf-8-sig"):
                raise ValueError(f"copied data/recipe file differs: {name}")
            files.append(name)
    source = normalized("train.py").replace('"new_parameter_count": 2802093', '"new_parameter_count": 2468010')
    class StripMixAudit(ast.NodeTransformer):
        def visit_Assign(self, node):
            targets = [ast.unparse(t) for t in node.targets]
            if targets == ["mix"] or targets == ["record['prompt_mix_rank0_last_microbatch']"]:
                return None
            return self.generic_visit(node)
        def visit_If(self, node):
            if any(isinstance(n, ast.Name) and n.id == "mix" for n in ast.walk(node.test)):
                return None
            return self.generic_visit(node)
        def visit_Dict(self, node):
            pairs = [(k, v) for k, v in zip(node.keys, node.values)
                     if not isinstance(k, ast.Constant) or k.value != "prompt_mix"]
            node.keys, node.values = [x[0] for x in pairs], [x[1] for x in pairs]
            return self.generic_visit(node)
    equal(StripMixAudit().visit(ast.parse(source)), tree(PARENT/"train.py"), "entire trainer")
    for name in ("build_optimizer", "_clip_grad", "track_and_clip_", "state_dict", "load_state_dict"):
        equal(find_function(tree(ROOT/"optim.py"), name), find_function(tree(PARENT/"optim.py"), name), name)
    a, b = ast.parse(normalized("contract.py")), tree(PARENT/"contract.py")
    for fn in [n for n in b.body if isinstance(n, ast.FunctionDef) and n.name != "semantic_contract"]:
        equal(find_function(a, fn.name), fn, "contract:"+fn.name)
    scope = {"__file__": str(PARENT/"contract.py")}
    execute(tree(PARENT/"contract.py").body, scope)
    prepared = dict(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS,
                    rows_canonical_sha256="rows", mapping_sha256="mapping")
    old = scope["semantic_contract"](prepared, "text", "ced")
    new = semantic_contract(prepared, "text", "ced")
    for key in ("route", "code_sha256", "recursive_model_source_sha256", "audio_listen", "prompt_mix"):
        old.pop(key, None); new.pop(key, None)
    if old != new:
        raise ValueError("common data/training recipe changed")
    return {"identical_files_normalized_lf": files, "entire_trainer_parity": True,
            "optimizer_scheduler_resume_parity": True, "common_training_contract_equal": True}

def audit_backbone():
    # Only explicit-seed interface, initial contents and seed diagnostics differ.
    source = normalized("models/recursive_mesh_listen.py").replace("audio_memory_seeds", "audio_information")
    swaps = {
        "len(audio_information) != 3": "len(audio_information) != 2",
        "explicit (weighted_E,weighted_A4,weighted_A8) audio_information": "explicit (A4,A8) audio_information",
        "weighted E/A4/A8 shape, dtype and device": "A4/A8 shape, dtype and device",
        "audio_memory = torch.cat((audio_information[0].unsqueeze(1), zeros_audio,\n            audio_information[1].unsqueeze(1), audio_information[2].unsqueeze(1)), dim=1)":
        "audio_memory = torch.cat((hidden[:, :n].unsqueeze(1), zeros_audio,\n            audio_information[0].unsqueeze(1), audio_information[1].unsqueeze(1)), dim=1)",
        '("weighted_e_rms", "weighted_a4_rms", "weighted_a8_rms")': '("a4_rms", "a8_rms")',
        "seed_read = sum(a * audio_read[..., slot:slot + 1]\n                                for slot, a in zip((0, 5, 6), self._diagnostic_audio_seeds))":
        "seed_read = sum(a * audio_read[..., i + 5:i + 6]\n                                for i, a in enumerate(self._diagnostic_audio_seeds))",
    }
    for actual, expected in swaps.items():
        if source.count(actual) != 1:
            raise ValueError(f"explicit seed delta missing or repeated: {actual}")
        source = source.replace(actual, expected)
    equal(ast.parse(source), tree(PARENT/"models/recursive_mesh_listen.py"), "entire split3 backbone")
    source = normalized("models/recursive_text.py").replace(
        '["alpha_E*E", 0, 0, 0, 0, "alpha_4*A4", "alpha_8*A8"]',
        '["input_embedding", 0, 0, 0, 0, "A4", "A8"]')
    equal(ast.parse(source), tree(PARENT/"models/recursive_text.py"), "entire strict loader")
    return {"entire_model_after_explicit_seed_deltas_equal": True,
            "all_six_split3_initializers_unchanged": True, "strict_loader_parity": True}

def audit():
    return {"status": "PASS", "scope": "source parity; no tensor/assets/GPU claim",
            "route_code_sha256": code_digest(), "protected": audit_protected(),
            "parent_parity": audit_parent(), "backbone_parity": audit_backbone()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"reference_audit.json")
    args = parser.parse_args()
    result = audit()
    write_json(args.output, result)
    print(result["status"], args.output)
