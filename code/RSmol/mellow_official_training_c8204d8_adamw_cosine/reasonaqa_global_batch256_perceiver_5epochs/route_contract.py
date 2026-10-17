"""Identity of the formal30 text-backbone replacement (no ML imports)."""

import hashlib
import json
from pathlib import Path

ROUTE_CONTRACT = "mellow_formal30_gbs256_5_10x2_5_mesh_v1"
TEXT_CONTRACT = "logical_30_physical_20_5_10x2_5"
BASELINE_TEXT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2")
MESH_TEXT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/"
    "formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244"
)
LOGICAL_TO_PHYSICAL = tuple(range(15)) + tuple(range(5, 15)) + tuple(range(15, 20))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_mesh_config(config):
    raw = config if isinstance(config, dict) else config.to_dict()
    expected = {
        "hidden_size": 576, "num_hidden_layers": 30,
        "recursive_layer_count": 20, "recursive_loops": 2,
        "mesh_memory_slots": 5, "mesh_router_count": 6,
        "mesh_architecture_contract": TEXT_CONTRACT,
    }
    for field, value in expected.items():
        if raw.get(field) != value:
            raise ValueError(f"mesh {field}: expected {value!r}, got {raw.get(field)!r}")
    architecture = raw.get("model_architecture_contract")
    if architecture and architecture not in {TEXT_CONTRACT, TEXT_CONTRACT + "_mesh"}:
        raise ValueError(f"unexpected model architecture contract: {architecture!r}")
    schedule = raw.get("logical_to_physical") or raw.get("logical_to_physical_schedule", ())
    if tuple(schedule) != LOGICAL_TO_PHYSICAL:
        raise ValueError(f"unexpected recursive schedule: {schedule!r}")
    return raw


def validate_sources(mesh_path, baseline_path):
    for supplied, expected in ((mesh_path, MESH_TEXT), (baseline_path, BASELINE_TEXT)):
        if Path(supplied).resolve() != expected.resolve():
            raise ValueError(f"this comparison fixes source {expected}; got {supplied}")
        if not (Path(supplied) / "config.json").is_file():
            raise FileNotFoundError(f"missing text config: {supplied}")
    validate_mesh_config(json.loads((Path(mesh_path) / "config.json").read_text(encoding="utf-8")))
