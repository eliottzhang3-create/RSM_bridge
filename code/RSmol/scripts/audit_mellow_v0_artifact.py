#!/usr/bin/env python3
"""Strict offline artifact preflight for the released full Mellow-v0 model.

This audit instantiates the native Mellow architecture from the local GitHub
checkout, points its SmolLM2 decoder at the local Hugging Face model, and
strictly loads every tensor from the released ``v0.ckpt`` on CPU.  It performs
no benchmark inference and does not use the example audio files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_MELLOW_SOURCE_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/mellow-main"
)
DEFAULT_MELLOW_SNAPSHOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/Mellow-v0/"
    "models--soham97--mellow/snapshots/83672db0dae28764e283210d5bb732621e903d8a"
)
DEFAULT_MELLOW_CHECKPOINT = DEFAULT_MELLOW_SNAPSHOT / "v0.ckpt"
DEFAULT_BASE_SMOLLM2 = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2")
DEFAULT_REPORT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0/preflight/"
    "mellow_v0_artifact_preflight.json"
)

ARTIFACT_CONTRACT = "native_mellow_v0_full_checkpoint_offline_v1"
MODEL_ARGUMENTS = {
    "audioenc_name": "HTSAT",
    "d_in": 768,
    "prefix_length": 389,
    "d_out": 576,
}
SOURCE_PREFIXES = {
    "htsat": "audio_encoder.base.htsat.",
    "c2l": "audio_encoder.base.c2l.",
    "projection": "audio_encoder.projection.",
    "text_decoder": "caption_decoder.lm.",
}
REQUIRED_SOURCE_FILES = (
    "mellow/__init__.py",
    "mellow/wrapper.py",
    "mellow/config/v0.yaml",
    "mellow/config/v0_local.yaml",
    "mellow/model/model.py",
    "mellow/model/mellow.py",
    "mellow/model/decoder.py",
    "mellow/model/audio.py",
    "mellow/model/htsat.py",
    "mellow/model/config.py",
)
REQUIRED_SMOLLM2_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_state_dict(payload: Any) -> dict[str, Any]:
    import torch

    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping) or not payload:
        raise RuntimeError("Mellow-v0 checkpoint has no non-empty state dictionary")
    state: dict[str, Any] = {}
    for raw_key, value in payload.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module."):]
        if key in state:
            raise RuntimeError(f"duplicate state key after module-prefix removal: {key}")
        if not torch.is_tensor(value):
            raise RuntimeError(f"checkpoint value is not a tensor: {key}")
        state[key] = value.detach().cpu()
    return state


def _load_state(checkpoint: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    return _canonical_state_dict(payload)


def _source_inventory(root: Path) -> dict[str, str]:
    missing = [name for name in REQUIRED_SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Mellow source checkout is incomplete: {missing}")
    return {name: _sha256(root / name) for name in REQUIRED_SOURCE_FILES}


def _smollm2_inventory(root: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_SMOLLM2_FILES if not (root / name).is_file()]
    weights = sorted([*root.glob("model*.safetensors"), *root.glob("pytorch_model*.bin")])
    if missing or not weights:
        raise FileNotFoundError(
            f"local SmolLM2 is incomplete: missing={missing} weights={[p.name for p in weights]}"
        )
    return {
        "required_file_sha256": {name: _sha256(root / name) for name in REQUIRED_SMOLLM2_FILES},
        "weight_files": [path.name for path in weights],
    }


def _instantiate_native_model(source_root: Path, base_smollm2: Path) -> tuple[Any, Any]:
    from transformers import AutoTokenizer

    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from mellow.model.model import get_model_class  # type: ignore[import-not-found]

    model_class = get_model_class("Mellow")
    model = model_class(
        audioenc_name=MODEL_ARGUMENTS["audioenc_name"],
        d_in=MODEL_ARGUMENTS["d_in"],
        text_decoder=str(base_smollm2),
        prefix_length=MODEL_ARGUMENTS["prefix_length"],
        d_out=MODEL_ARGUMENTS["d_out"],
    )
    tokenizer = AutoTokenizer.from_pretrained(base_smollm2, local_files_only=True)
    original_size = len(tokenizer)
    tokenizer.add_special_tokens({"pad_token": "!"})
    if len(tokenizer) != original_size:
        raise RuntimeError("Mellow pad token unexpectedly expanded the local SmolLM2 vocabulary")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None or int(pad_token_id) < 0 or int(pad_token_id) >= len(tokenizer):
        raise RuntimeError(
            "native Mellow pad token is outside the local SmolLM2 vocabulary: "
            f"pad_token_id={pad_token_id!r} vocabulary={len(tokenizer)}"
        )
    # Native Mellow uses the tokenizer's '!' token for right-padding prompts,
    # but decoder.py independently hardcodes id 0 as the inter-audio separator.
    # They are distinct contracts and are not required to be the same token.
    separator_id = 0
    if separator_id >= len(tokenizer):
        raise RuntimeError("native Mellow separator token id 0 is outside the vocabulary")
    return model, tokenizer


def audit(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    source_root = args.mellow_source_root.resolve()
    snapshot = args.mellow_snapshot.resolve()
    checkpoint = args.mellow_checkpoint.resolve()
    base_smollm2 = args.base_smollm2.resolve()
    if checkpoint.parent != snapshot:
        raise RuntimeError(
            f"Mellow checkpoint must be inside the selected snapshot: {checkpoint} vs {snapshot}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Mellow-v0 checkpoint not found: {checkpoint}")
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"Mellow Hugging Face snapshot lacks config.json: {snapshot}")

    source_hashes = _source_inventory(source_root)
    smollm2_inventory = _smollm2_inventory(base_smollm2)
    state = _load_state(checkpoint)
    group_counts = {
        name: sum(key.startswith(prefix) for key in state)
        for name, prefix in SOURCE_PREFIXES.items()
    }
    empty_groups = [name for name, count in group_counts.items() if count <= 0]
    known_prefixes = tuple(SOURCE_PREFIXES.values())
    unknown_keys = sorted(key for key in state if not key.startswith(known_prefixes))
    if empty_groups or unknown_keys:
        raise RuntimeError(
            f"Mellow-v0 state inventory mismatch: empty_groups={empty_groups} "
            f"unknown_keys={unknown_keys[:20]}"
        )

    model, tokenizer = _instantiate_native_model(source_root, base_smollm2)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"strict state load reported incompatibilities: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    if any(module.training for module in model.modules()):
        raise RuntimeError("native Mellow model did not enter complete eval mode")
    runtime_state = model.state_dict()
    state_key_mismatch = {
        "missing": sorted(set(runtime_state) - set(state)),
        "unexpected": sorted(set(state) - set(runtime_state)),
    }
    shape_mismatches = {
        key: {
            "checkpoint": list(state[key].shape),
            "runtime": list(runtime_state[key].shape),
        }
        for key in sorted(set(state) & set(runtime_state))
        if tuple(state[key].shape) != tuple(runtime_state[key].shape)
    }
    dtype_mismatches = {
        key: {
            "checkpoint": str(state[key].dtype),
            "runtime": str(runtime_state[key].dtype),
        }
        for key in sorted(set(state) & set(runtime_state))
        if state[key].dtype != runtime_state[key].dtype
    }
    if state_key_mismatch["missing"] or state_key_mismatch["unexpected"]:
        raise RuntimeError(f"runtime state key mismatch after strict load: {state_key_mismatch}")
    if shape_mismatches or dtype_mismatches:
        raise RuntimeError(
            "runtime state metadata mismatch after strict load: "
            f"shapes={shape_mismatches} dtypes={dtype_mismatches}"
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    state_parameter_count = sum(value.numel() for value in state.values())

    return {
        "status": "PASS",
        "stage": "mellow_v0_artifact_preflight",
        "artifact_contract": ARTIFACT_CONTRACT,
        "offline": {
            "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
            "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
        },
        "mellow_source_root": str(source_root),
        "mellow_source_sha256": source_hashes,
        "mellow_snapshot": str(snapshot),
        "snapshot_config_sha256": _sha256(snapshot / "config.json"),
        "mellow_checkpoint": str(checkpoint),
        "mellow_checkpoint_sha256": _sha256(checkpoint),
        "base_smollm2": str(base_smollm2),
        "base_smollm2_inventory": smollm2_inventory,
        "model_arguments": {**MODEL_ARGUMENTS, "text_decoder": str(base_smollm2)},
        "state_tensor_count": len(state),
        "state_parameter_count": state_parameter_count,
        "group_tensor_counts": group_counts,
        "strict_state_dict_load": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "runtime_parameter_count": parameter_count,
        "runtime_state_tensor_count": len(runtime_state),
        "runtime_state_shapes_match": True,
        "runtime_state_dtypes_match": True,
        "tokenizer": {
            "size": len(tokenizer),
            "pad_token": tokenizer.pad_token,
            "pad_token_id": int(tokenizer.pad_token_id),
            "separator_token": tokenizer.convert_ids_to_tokens(0),
            "separator_token_id": 0,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": int(tokenizer.eos_token_id),
        },
        "example_audio_check": "NOT_REQUESTED",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mellow-source-root", type=Path, default=DEFAULT_MELLOW_SOURCE_ROOT)
    parser.add_argument("--mellow-snapshot", type=Path, default=DEFAULT_MELLOW_SNAPSHOT)
    parser.add_argument("--mellow-checkpoint", type=Path, default=DEFAULT_MELLOW_CHECKPOINT)
    parser.add_argument("--base-smollm2", type=Path, default=DEFAULT_BASE_SMOLLM2)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit(args)
    except Exception as exc:
        import traceback

        report = {
            "status": "FAILED",
            "stage": "mellow_v0_artifact_preflight",
            "artifact_contract": ARTIFACT_CONTRACT,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    _write_json(args.report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
