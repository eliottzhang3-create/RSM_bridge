"""Executable parity audit against the exact MeSH CED mother (no ML imports)."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small")
ROOT = HERE.parents[3]


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def function(node, name):
    return next(n for n in node.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def same(left, right, label):
    if ast.dump(left, include_attributes=False) != ast.dump(right, include_attributes=False):
        raise AssertionError(f"mother AST parity failed: {label}")


class StripListen(ast.NodeTransformer):
    """Remove ONLY enumerated listen additions; preserve every mother operation."""
    def visit_Assign(self, node):
        targets = [ast.unparse(t) for t in node.targets]
        if any(t in {"audio_bank", "audio_grids", "prompt_mask", "audio_listen", "listen_gradient_state",
                     "self.audio_listen", "audio_listen.collect_diagnostics", "audio_listen.gradient_audit_state",
                     "model.caption_decoder.lm.model.audio_listen.gradient_audit_state",
                     "audit['audio_listen']", "record['audio_listen']"} for t in targets):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        text = ast.unparse(node.test)
        if text in {"self.audio_listen is not None", "audio_grids is not None or prompt_mask is not None",
                    "loop == 0 and self.audio_listen is not None", "listen_gradient_state is None"}:
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == "enable_audio_listen":
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id in {"audit_listen_update", "validate_resume_payload"}:
                return None
            if (isinstance(call.func, ast.Name) and call.func.id == "collective_require"
                    and call.args and ast.unparse(call.args[0]) == "listen_gradient_state['complete']"):
                return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [k for k in node.keywords if k.arg not in {"audio_grids", "prompt_mask"}]
        return self.generic_visit(node)

    def visit_Dict(self, node):
        pairs = [(k, v) for k, v in zip(node.keys, node.values)
                 if not isinstance(k, ast.Constant) or k.value not in {"audio_listen", "listen_gradient_state"}]
        node.keys, node.values = [k for k, _ in pairs], [v for _, v in pairs]
        return self.generic_visit(node)


def audit_parent():
    changed = {"README.md", "audit_reference.py", "reference_audit.json", "runtime_checks.py", "contract.py",
               "train.py", "optim.py", "run.sh", "write_config.py", "models/ced.py", "models/decoder.py",
               "models/mellow.py", "models/recursive_text.py"}
    identical = []
    inventory = {}
    for path in sorted(PARENT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        name = path.relative_to(PARENT).as_posix()
        inventory[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if name not in changed:
            if path.read_bytes() != (HERE / name).read_bytes():
                raise AssertionError(f"mother file changed: {name}")
            identical.append(name)
    checks = []
    old, new = tree(PARENT / "contract.py"), tree(HERE / "contract.py")
    old_constants = {n.targets[0].id: n for n in old.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    new_constants = {n.targets[0].id: n for n in new.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    for name, value in old_constants.items():
        if name != "ROUTE":
            same(value, new_constants[name], f"constant {name}")
    for node in old.body:
        if isinstance(node, ast.FunctionDef) and node.name != "recursive_model_source_sha256":
            candidate = copy.deepcopy(function(new, node.name))
            same(node, StripListen().visit(candidate), f"contract.{node.name}")
    checks.append("all mother constants and contract functions, except explicit architecture/source identity")
    old, new = tree(PARENT / "train.py"), tree(HERE / "train.py")
    for node in old.body:
        if isinstance(node, ast.FunctionDef) and node.name not in {"validate_trainable_model", "first_gradient_audit"}:
            same(node, StripListen().visit(copy.deepcopy(function(new, node.name))), f"train.{node.name}")
    checks.append("entire training/update loop, loss, sampler, RNG, checkpoint, probes and precision after removing enumerated additions")
    old_opt = tree(PARENT / "optim.py")
    new_text = (HERE / "optim.py").read_text(encoding="utf-8")
    new_text = new_text.replace('lr = 3e-4 if role == "default" else 6e-4', 'lr = 6e-4 if role == "map" else 3e-4')
    new_text = new_text.replace('{"map", "default", "listen_projections", "listen_gates", "listen_routers", "listen_amplitudes"}', '{"map", "default"}')
    new_opt = ast.parse(new_text)
    for name in ("build_optimizer", "GradNormTracker"):
        same(function(old_opt, name), function(new_opt, name), f"optim.{name}")
    checks.append("optimizer/scheduler/GradNorm implementation; only added parameter roles receive mapper LR")
    shared = ROOT / "code/RSmol/recursive_model_5_10x2_5_mesh.py"
    old_mesh, new_mesh = tree(shared), tree(HERE / "models/recursive_mesh_listen.py")
    same(old_mesh, StripListen().visit(copy.deepcopy(new_mesh)), "complete original MeSH implementation")
    checks.append("all original MeSH layers, six routers, queries, memory initialization, read/write and schedule")
    old_ced, new_ced = tree(PARENT / "models/ced.py"), tree(HERE / "models/ced.py")
    for name in ("CEDFrequencyMerger", "CEDTemporalMapper"):
        same(function(old_ced, name), function(new_ced, name), name)
    old_wrapper, new_wrapper = function(old_ced, "CEDSmallWrapper"), function(new_ced, "CEDSmallWrapper")
    for name in ("train", "_waveform_to_mel", "_prepare_ced_input", "_hidden_to_temporal_grid"):
        same(function(old_wrapper, name), function(new_wrapper, name), f"CED.{name}")
    # The only constructor addition is a read-only 12-block geometry check.
    constructor = copy.deepcopy(function(new_wrapper, "__init__"))
    constructor.body = [n for n in constructor.body if not (isinstance(n, ast.If) and
                         ast.unparse(n.test).startswith("not hasattr(self.ced, 'blocks')"))]
    same(function(old_wrapper, "__init__"), constructor, "CED constructor and mapper initialization RNG")
    checks.append("official mapper classes, frozen encoder construction, waveform/mel and temporal ordering")
    for name in ("models/decoder.py", "models/mellow.py"):
        old_tree, candidate = tree(PARENT / name), StripListen().visit(copy.deepcopy(tree(HERE / name)))
        if name.endswith("decoder.py"):
            forward = function(function(candidate, "DecoderModel"), "forward")
            forward.args.args = [arg for arg in forward.args.args if arg.arg != "audio_grids"]
            forward.args.defaults = forward.args.defaults[:-1]
        else:
            forward = function(function(candidate, "Mellow"), "forward")
            forward.body[1].targets[0].elts[-1] = ast.Name(id="_", ctx=ast.Store())
        same(old_tree, candidate, name)
    checks.append("original prefix, embedding concatenation, LM call and audio wrapper")
    old_name, new_name = PARENT.name, HERE.name
    runner = (HERE / "run.sh").read_text(encoding="utf-8").replace("ced_small_layer36912_listen", "ced_small")
    if runner != (PARENT / "run.sh").read_text(encoding="utf-8"):
        raise AssertionError("run.sh changed beyond isolated identity/path")
    launch_name = "run_mizar_s1_gbs256_recursive_mesh_ced_small"
    launcher = (ROOT / f"code/RSmol/{launch_name}_layer36912_listen_5090.sh").read_text(encoding="utf-8")
    launcher = launcher.replace(new_name, old_name).replace("recursive_mesh_layer36912_listen_5090", "recursive_mesh_5090")
    launcher = launcher.replace("ced_small_layer36912_listen.", "ced_small.").replace("mizar-ced-mesh-listen-", "mizar-s1-ced-mesh-gbs256-")
    launcher = launcher.replace("recursive MeSH four-depth listen gamma0.3:", "recursive MeSH:")
    if launcher != (ROOT / f"code/RSmol/{launch_name}_5090.sh").read_text(encoding="utf-8"):
        raise AssertionError("launcher changed beyond isolated identity/path")
    checks.append("formal submission, resources, shared-start staging and cleanup; paths/identity only")
    spec = importlib.util.spec_from_file_location("listen_audit_contract", HERE / "contract.py")
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    return {"status": "PASS", "scope": "source parity only; no real model weights or GPU evidence",
            "mother_route": PARENT.name, "route": HERE.name, "byte_identical_files": identical,
            "semantic_checks": checks, "mother_file_sha256": inventory,
            "shared_mesh_sha256": hashlib.sha256(shared.read_bytes()).hexdigest(),
            "geometry": contract.geometry(1477492), "audio_listen": contract.audio_listen_contract()}


if __name__ == "__main__":
    report = audit_parent()
    (HERE / "reference_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "mother_file_sha256"}, ensure_ascii=False))
