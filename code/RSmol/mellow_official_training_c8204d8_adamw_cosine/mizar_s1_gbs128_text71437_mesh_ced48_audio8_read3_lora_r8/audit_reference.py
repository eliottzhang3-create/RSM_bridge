"""Read-only two-parent source audit; no model assets or GPU required."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
BASE = ROOT.with_name("mizar_s1_gbs128_text71437_mesh_ced_small")
SOURCE = ROOT.with_name("mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_lora_r8")


def tree(path):
    return ast.parse(Path(path).read_text(encoding="utf-8-sig"))


def find_function(module, name):
    matches = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one function {name}: {len(matches)}")
    return matches[0]


def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "audio8-read3-source", "exec"), scope)


def normalized(path):
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def equal_node(left, right, label):
    if ast.dump(left, include_attributes=False) != ast.dump(right, include_attributes=False):
        raise ValueError(f"inherited numerical implementation changed: {label}")


def audit():
    snapshot = json.loads((ROOT / "protected_sources.json").read_text(encoding="utf-8"))["sha256_normalized_lf"]
    for name, expected in snapshot.items():
        if hashlib.sha256(normalized(REPO / name)).hexdigest() != expected:
            raise ValueError(f"protected source changed: {name}")
    files = ["prepare_index.py", "stage_data.py", "check_ced_assets.py", "check_text_assets.py",
             "models/audio.py", "models/ced_offline.py", "utils/audio_io.py"]
    files += [p.relative_to(BASE).as_posix() for p in sorted((BASE / "data").glob("*.py"))]
    for name in files:
        if normalized(ROOT / name) != normalized(BASE / name):
            raise ValueError(f"three-loop data/staging implementation changed: {name}")
    for name in ("models/ced.py", "models/mellow.py", "models/decoder.py"):
        if normalized(ROOT / name) != normalized(SOURCE / name):
            raise ValueError(f"read2 CED/mapper/interface implementation changed: {name}")
    functions = {
        "contract.py": ["geometry", "lr_factor", "data_order", "rank_seed", "raw_references",
                        "validate_ced_assets", "validate_ced_config", "validate_checkpoint_tokenizer",
                        "validate_tokenizer_contract", "recursive_checkpoint_contract",
                        "recursive_model_source_sha256"],
        "train.py": ["answer_loss", "move_batch", "rng_state", "restore_rng", "inspect_tokenizer",
                     "audit_checkpoint_tokenizer", "collective_require", "tensor_digest", "weights_changed"],
        "optim.py": ["track_and_clip_", "_clip_grad"],
        "write_config.py": ["file_sha256", "asset_identity"],
    }
    for name, names in functions.items():
        for fn in names:
            equal_node(find_function(tree(ROOT / name), fn), find_function(tree(BASE / name), fn), f"{name}:{fn}")
    shared = REPO / "code/RSmol/recursive_model_mizar_text_7_14x3_7_mesh.py"
    private = ROOT / "models/recursive_audio8_read3.py"
    helpers = ("_cfg", "_assert_llama_api", "make_dynamic_cache", "_cache_seq_length", "_validate_cache",
               "_causal_mask", "_call_layer", "_init_router", "_write", "_read")
    for name in helpers:
        equal_node(find_function(tree(private), name), find_function(tree(shared), name), f"backbone:{name}")
    # The entire production epoch/update loop must match the read2 parent.
    def epoch_loop(path):
        fn = find_function(tree(path), "train")
        return next(n for n in fn.body if isinstance(n, ast.For) and ast.unparse(n.target) == "epoch")
    old_epoch = ast.parse(ast.unparse(epoch_loop(SOURCE / "train.py")).replace(
        "both recursive passes require finite nonzero gradients",
        "all three recursive passes require finite nonzero gradients")).body[0]
    equal_node(epoch_loop(ROOT / "train.py"), old_epoch, "complete_epoch_update_loop")
    return {"status": "PASS_LOCAL_SOURCE", "protected_files": len(snapshot),
            "unchanged_data_staging_files": files, "numeric_functions_equal": functions,
            "backbone_helpers_equal": list(helpers), "complete_epoch_update_loop_equal": True,
            "gpu_training_verified": False}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
