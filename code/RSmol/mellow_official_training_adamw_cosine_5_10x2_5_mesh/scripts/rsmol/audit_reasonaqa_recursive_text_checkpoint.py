#!/usr/bin/env python3
"""CPU-only audit for the 5-10x2-5 MeSH text initialization directory.

The audio trainer loads this artifact with ``from_pretrained`` and creates a
fresh audio bridge, optimizer, scheduler, and RNG state.  This audit checks
the Hugging Face directory contract, recursive MeSH config fields, tokenizer
presence, and a deterministic state-dict structure without allocating a
model or touching CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/"
    "formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244"
)
CONTRACT = "rsmol_5_10x2_5_mesh_text_checkpoint_audit_v1"
EXPECTED_CONFIG = {
    "hidden_size": 576,
    "num_hidden_layers": 30,
    "recursive_layer_count": 20,
    "recursive_loops": 2,
    "memory_slots": 5,
    "router_count": 6,
}
TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "merges.txt",
}
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{__import__('os').getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _regular_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _load_state(path: Path) -> dict[str, Any]:
    """Load a safetensors or torch state dict only when the runtime supports it."""
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("safetensors is required to inspect *.safetensors") from exc
        state = load_file(str(path), device="cpu")
    else:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required to inspect binary model weights") from exc
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError(f"weight file does not contain a non-empty state dict: {path}")
    return state


def _state_structure(state: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256()
    dtypes: Counter[str] = Counter()
    total = 0
    names: list[str] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str):
            raise ValueError(f"state dict contains a non-string key: {name!r}")
        if not hasattr(value, "shape") or not hasattr(value, "dtype") or not hasattr(value, "numel"):
            raise ValueError(f"state dict entry is not a tensor: {name!r}")
        shape = tuple(int(dim) for dim in value.shape)
        dtype = str(value.dtype)
        names.append(name)
        total += int(value.numel())
        dtypes[dtype] += 1
        digest.update(name.encode("utf-8"))
        digest.update(dtype.encode("ascii"))
        digest.update(repr(shape).encode("ascii"))
    return {
        "tensor_count": len(names),
        "total_elements": total,
        "dtype_distribution": dict(sorted(dtypes.items())),
        "first_keys": names[:5],
        "last_keys": names[-5:],
        "structural_sha256": digest.hexdigest(),
        "router_tensor_count": sum("write_routers." in n or "read_routers." in n for n in names),
    }


def audit(checkpoint: Path, *, hash_files: bool = True) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint must be a Hugging Face directory: {checkpoint}")
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise ValueError(f"checkpoint is missing config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.json must contain an object")

    files = _regular_files(checkpoint)
    relative_files = [str(path.relative_to(checkpoint).as_posix()) for path in files]
    weight_files = [path for path in files if path.suffix in WEIGHT_SUFFIXES and ("model" in path.name or "pytorch" in path.name)]
    if not weight_files:
        weight_files = [path for path in files if path.suffix in {".safetensors", ".bin"}]
    if not weight_files:
        raise ValueError("checkpoint contains no recognized model weight file")
    states = [_load_state(path) for path in weight_files]
    state: dict[str, Any] = {}
    for shard in states:
        overlap = sorted(set(state).intersection(shard))
        if overlap:
            raise ValueError(f"duplicate tensor keys across weight files: {overlap[:5]}")
        state.update(shard)
    structure = _state_structure(state)

    def config_value(key: str) -> Any:
        if key in config:
            return config[key]
        return config.get(f"mesh_{key}")

    config_checks = {key: {"expected": expected, "actual": config_value(key), "passed": config_value(key) == expected} for key, expected in EXPECTED_CONFIG.items()}
    expected_schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    schedule = config.get("logical_to_physical") or config.get("logical_to_physical_schedule")
    schedule_check = {"expected": expected_schedule, "actual": schedule, "passed": list(schedule or ()) == expected_schedule}
    architecture = config.get("architecture_contract", config.get("mesh_architecture_contract"))
    architecture_passed = architecture in {
        # The converter writes this exact value to mesh_architecture_contract.
        "logical_30_physical_20_5_10x2_5",
        # Keep compatibility with artifacts that used the longer label.
        "logical_30_physical_20_5_10x2_5_mesh",
    }
    tokenizer_files = sorted(name for name in relative_files if Path(name).name in TOKENIZER_FILES)
    tokenizer_config = {}
    tokenizer_config_path = checkpoint / "tokenizer_config.json"
    if tokenizer_config_path.is_file():
        raw_tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        if isinstance(raw_tokenizer_config, dict):
            tokenizer_config = raw_tokenizer_config
    file_inventory = []
    for path in files:
        item = {"path": str(path.relative_to(checkpoint).as_posix()), "bytes": path.stat().st_size}
        if hash_files:
            item["sha256"] = sha256_file(path)
        file_inventory.append(item)

    failures: list[str] = []
    failures.extend(key for key, result in config_checks.items() if not result["passed"])
    if not schedule_check["passed"]:
        failures.append("logical_to_physical")
    if not architecture_passed:
        failures.append("architecture_contract")
    if not tokenizer_files:
        failures.append("tokenizer_files")
    status = "PASS" if not failures else "FAIL"
    return {
        "status": status,
        "contract": CONTRACT,
        "gpu_required": False,
        "checkpoint": str(checkpoint),
        "initialization_semantics": "load recursive text state_dict only; create fresh audio bridge/optimizer/scheduler/RNG",
        "config": {"model_type": config.get("model_type"), "architecture_contract": architecture, "checks": {**config_checks, "logical_to_physical": schedule_check}},
        "tokenizer": {"files": tokenizer_files, "config": tokenizer_config, "eos_token_id": config.get("eos_token_id", tokenizer_config.get("eos_token_id")), "pad_token_id": config.get("pad_token_id", tokenizer_config.get("pad_token_id"))},
        "weights": {"files": [str(path.relative_to(checkpoint).as_posix()) for path in weight_files], "structure": structure},
        "file_count": len(file_inventory),
        "file_inventory": file_inventory,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--skip-file-hashes", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    report = audit(checkpoint, hash_files=not args.skip_file_hashes)
    output = args.output_report.expanduser().resolve() if args.output_report else checkpoint / "recursive_text_checkpoint_audit.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
