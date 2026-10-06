#!/usr/bin/env python3
"""Build the fresh Mellow-v0 -> 5-10x2-5 MeSH initialization artifacts.

This is a CPU-only conversion for the official Mellow ReasonAQA route.  It
creates two artifacts:

* a standalone 5-10x2-5 MeSH text directory; and
* a full ``--init-model-checkpoint`` containing the Mellow state-dict layout.

The text decoder tensors are copied from Mellow-v0 using the audited
30-to-20 physical-layer mapping.  The six router tensors are retained from a
fresh MeSH model construction, so their initialization follows the MeSH
router contract.  Mellow's HTSAT, c2l, and projection tensors are copied
unchanged into the full initialization checkpoint.  No optimizer, scheduler,
RNG, or training progress is included.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
RSMOL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(RSMOL_ROOT) not in sys.path:
    sys.path.insert(0, str(RSMOL_ROOT))

from audit_mellow_v0_artifact import (  # noqa: E402
    DEFAULT_BASE_SMOLLM2,
    DEFAULT_MELLOW_CHECKPOINT,
    DEFAULT_MELLOW_SNAPSHOT,
    DEFAULT_MELLOW_SOURCE_ROOT,
    SOURCE_PREFIXES,
    _instantiate_native_model,
    _load_state,
    _sha256,
    _smollm2_inventory,
    _source_inventory,
)
from audit_mellow_v0_mesh_initialization import (  # noqa: E402
    EXPECTED_DISCARDED_SOURCE_LAYERS,
    TEXT_PREFIX,
    _source_key_for_target,
)
from convert_stepwise_5_10x2_5_mesh import _target_config  # noqa: E402
from recursive_model_5_10x2_5_mesh import (  # noqa: E402
    LOGICAL_TO_PHYSICAL,
    MEMORY_SLOT_COUNT,
    MAPPING_POLICY,
    PHYSICAL_LAYER_COUNT,
    ROUTER_PARAMETER_COUNT,
    SOURCE_LAYER_INDICES_0BASED,
    RecursiveLlamaForCausalLM,
    parameter_audit,
    register_auto_class,
)


ARTIFACT_CONTRACT = "mellow_v0_to_5_10x2_5_mesh_initialization_v1"
DEFAULT_OUTPUT_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/converted/"
    "mellow_v0_5_10x2_5_mesh_initialization"
)
DEFAULT_TEXT_OUTPUT = DEFAULT_OUTPUT_ROOT / "text_model"
DEFAULT_INIT_CHECKPOINT = DEFAULT_OUTPUT_ROOT / "mellow_v0_5_10x2_5_mesh_init.pt"
DEFAULT_OFFICIAL_ROUTE_ROOT = RSMOL_ROOT / "mellow_official_training_adamw_cosine_5_10x2_5_mesh"
DEFAULT_HTSAT_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT")
DEFAULT_REPORT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0/preflight/"
    "mellow_v0_5_10x2_5_mesh_initialization.json"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _tensor_digest(value: Any) -> str:
    import torch

    if not torch.is_tensor(value):
        raise TypeError(f"expected tensor, got {type(value).__name__}")
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _extract_group(state: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key.startswith(prefix)}


def _copy_tokenizer(base_smollm2: Path, target: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_smollm2, local_files_only=True)
    original_size = len(tokenizer)
    tokenizer.add_special_tokens({"pad_token": "!"})
    if len(tokenizer) != original_size:
        raise RuntimeError("adding Mellow's '!' pad token changed the vocabulary size")
    if tokenizer.pad_token_id is None or int(tokenizer.pad_token_id) != 17:
        raise RuntimeError(f"unexpected Mellow pad token contract: {tokenizer.pad_token_id!r}")
    target.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(target)
    return {
        "size": len(tokenizer),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": int(tokenizer.eos_token_id),
        "saved_files": sorted(path.name for path in target.iterdir() if path.is_file()),
    }


def _build_target_text(
    source_config: Any,
    source_state: Mapping[str, Any],
    checkpoint: Path,
    text_output: Path,
    base_smollm2: Path,
    seed: int,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import torch

    target_config = _target_config(
        source_config,
        {"kind": "native_mellow_v0_text", "mapping": SOURCE_LAYER_INDICES_0BASED},
        checkpoint,
    )
    torch.manual_seed(seed)
    register_auto_class()
    target_model = RecursiveLlamaForCausalLM(target_config)
    floating = next(
        value for value in source_state.values()
        if torch.is_tensor(value) and value.is_floating_point()
    )
    target_model.to(dtype=floating.dtype)
    target_state = target_model.state_dict()
    inherited: dict[str, Any] = {}
    missing: list[str] = []
    shape_mismatches: dict[str, Any] = {}
    dtype_mismatches: dict[str, Any] = {}
    for target_key, target_value in target_state.items():
        if "write_routers." in target_key or "read_routers." in target_key:
            continue
        source_key = _source_key_for_target(target_key)
        source_value = source_state.get(source_key)
        if source_value is None:
            missing.append(target_key)
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            shape_mismatches[target_key] = {
                "source_key": source_key,
                "source_shape": list(source_value.shape),
                "target_shape": list(target_value.shape),
            }
        if source_value.dtype != target_value.dtype:
            dtype_mismatches[target_key] = {
                "source_key": source_key,
                "source_dtype": str(source_value.dtype),
                "target_dtype": str(target_value.dtype),
            }
        inherited[target_key] = source_value.detach().clone()
    if missing or shape_mismatches or dtype_mismatches:
        raise RuntimeError(
            "Mellow-to-MeSH initialization mapping failed: "
            f"missing={missing[:8]} shapes={list(shape_mismatches)[:8]} "
            f"dtypes={list(dtype_mismatches)[:8]}"
        )
    incompatible = target_model.load_state_dict(inherited, strict=False)
    expected_router_keys = sorted(
        key for key in target_state
        if "write_routers." in key or "read_routers." in key
    )
    if sorted(incompatible.missing_keys) != expected_router_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "target initialization load produced unexpected keys: "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    target_model.tie_weights()
    loaded_state = target_model.state_dict()
    for target_key, source_value in inherited.items():
        if not torch.equal(loaded_state[target_key].cpu(), source_value.cpu()):
            raise RuntimeError(f"inherited value changed after target load: {target_key}")

    text_output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{text_output.name}.staging-", dir=text_output.parent))
    try:
        target_model.save_pretrained(staging, safe_serialization=True)
        tokenizer_audit = _copy_tokenizer(base_smollm2, staging)
        metadata = {
            "status": "ok",
            "conversion": "Mellow-v0-5-10x2-5-MeSH-text",
            "source_checkpoint": str(checkpoint.resolve()),
            "source_layer_indices_0based": list(SOURCE_LAYER_INDICES_0BASED),
            "logical_to_physical": list(LOGICAL_TO_PHYSICAL),
            "mapping_policy": MAPPING_POLICY,
            "logical_layer_count": 30,
            "physical_layer_count": PHYSICAL_LAYER_COUNT,
            "recursive_loops": 2,
            "memory_slots": MEMORY_SLOT_COUNT,
            "router_parameter_count": ROUTER_PARAMETER_COUNT,
            "discarded_source_layer_indices_0based": list(EXPECTED_DISCARDED_SOURCE_LAYERS),
            "router_initialization": "retained from fresh MeSH construction",
            "parameter_audit": parameter_audit(target_model),
            "tokenizer": tokenizer_audit,
            "seed": seed,
        }
        (staging / "mellow_v0_mesh_text_initialization.json").write_text(
            json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if text_output.exists():
            raise FileExistsError(f"refusing to overwrite text output: {text_output}")
        staging.replace(text_output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target_model, metadata, tokenizer_audit


def _build_full_init_checkpoint(
    target_model: Any,
    source_state: Mapping[str, Any],
    output_path: Path,
    text_output: Path,
    checkpoint: Path,
    source_root: Path,
    snapshot: Path,
    official_route_root: Path,
    htsat_root: Path,
    seed: int,
) -> dict[str, Any]:
    import torch

    target_state = target_model.state_dict()
    full_state: OrderedDict[str, Any] = OrderedDict()
    # Copy the three audio groups directly.  The source text group is
    # deliberately omitted here because the target text group is rebuilt
    # below with the 30-to-20 layer mapping and fresh routers.
    for name in ("htsat", "c2l", "projection"):
        prefix = SOURCE_PREFIXES[name]
        group = _extract_group(source_state, prefix)
        if not group:
            raise RuntimeError(f"Mellow-v0 source group is empty: {name} ({prefix})")
        for key, value in group.items():
            full_state[key] = value.detach().cpu().clone()
    for target_key, value in target_state.items():
        full_state[f"{TEXT_PREFIX}{target_key}"] = value.detach().cpu().clone()

    expected_keys = {
        key for key in source_state
        if any(key.startswith(SOURCE_PREFIXES[name]) for name in ("htsat", "c2l", "projection"))
    }
    expected_keys.update(f"{TEXT_PREFIX}{key}" for key in target_state)
    if set(full_state) != expected_keys:
        raise RuntimeError(
            "initialization checkpoint key inventory mismatch: "
            f"missing={sorted(expected_keys - set(full_state))[:8]} "
            f"unexpected={sorted(set(full_state) - expected_keys)[:8]}"
        )

    # Validate against the exact official route class before publishing.  The
    # native Mellow audit proves the source checkpoint; this second load proves
    # that the state-dict layout is accepted by the trainer the user will run.
    route_root = official_route_root.resolve()
    if not (route_root / "models/mellow.py").is_file():
        raise FileNotFoundError(f"official Mellow route checkout not found: {route_root}")
    if not (htsat_root / "HTSAT_AudioSet_Saved_1.ckpt").is_file():
        raise FileNotFoundError(f"official HTSAT initialization checkpoint not found: {htsat_root}")
    if str(route_root) not in sys.path:
        sys.path.insert(0, str(route_root))
    from models.mellow import Mellow  # type: ignore[import-not-found]

    official_model = Mellow(
        audioenc_name="HTSAT",
        d_in=768,
        text_decoder=str(text_output),
        prefix_length=389,
        freeze_text_decoder_weights=False,
        d_out=576,
        use_pretrained_audioencoder=True,
        freeze_audio_encoder_weights=True,
        pretrained_audioencoder_path=str(htsat_root),
    )
    official_keys = set(official_model.state_dict())
    if official_keys != set(full_state):
        raise RuntimeError(
            "official Mellow state-dict layout mismatch: "
            f"missing={sorted(official_keys - set(full_state))[:12]} "
            f"unexpected={sorted(set(full_state) - official_keys)[:12]}"
        )
    incompatible = official_model.load_state_dict(full_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "official Mellow strict initialization load mismatch: "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    del official_model
    gc.collect()

    metadata = {
        "status": "complete",
        "artifact_contract": ARTIFACT_CONTRACT,
        "artifact_kind": "official_mellow_model_weight_only_initialization",
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "mellow_source_root": str(source_root.resolve()),
        "mellow_snapshot": str(snapshot.resolve()),
        "text_model_output": str(text_output.resolve()),
        "official_route_root": str(route_root),
        "htsat_constructor_checkpoint": str((htsat_root / "HTSAT_AudioSet_Saved_1.ckpt").resolve()),
        "text_model_contract": "logical_30_physical_20_5_10x2_5",
        "text_architecture_contract": "logical_30_physical_20_5_10x2_5_mesh",
        "logical_to_physical": list(LOGICAL_TO_PHYSICAL),
        "source_layer_indices_0based": list(SOURCE_LAYER_INDICES_0BASED),
        "discarded_source_layer_indices_0based": list(EXPECTED_DISCARDED_SOURCE_LAYERS),
        "memory_slots": MEMORY_SLOT_COUNT,
        "router_parameter_count": ROUTER_PARAMETER_COUNT,
        "router_initialization": "fresh MeSH truncated-normal initialization",
        "audio_initialization": {
            "htsat": "copied from Mellow-v0; official route freezes HTSAT",
            "c2l": "copied from Mellow-v0; official route keeps c2l trainable",
            "projection": "copied from Mellow-v0; official route keeps bridge trainable",
        },
        "state_tensor_count": len(full_state),
        "state_parameter_count": int(sum(value.numel() for value in full_state.values())),
        "state_group_tensor_counts": {
            name: len(_extract_group(full_state, prefix))
            for name, prefix in SOURCE_PREFIXES.items()
        } | {"text_decoder": len(target_state)},
        "seed": seed,
        "optimizer_scheduler_rng_included": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    payload = {"state_dict": full_state, "metadata": metadata}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite initialization checkpoint: {output_path}")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    return metadata


def _build(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.set_grad_enabled(False)
    source_root = args.mellow_source_root.expanduser().resolve()
    snapshot = args.mellow_snapshot.expanduser().resolve()
    checkpoint = args.mellow_checkpoint.expanduser().resolve()
    base_smollm2 = args.base_smollm2.expanduser().resolve()
    text_output = args.text_output.expanduser().resolve()
    init_checkpoint = args.init_checkpoint.expanduser().resolve()
    if not source_root.is_dir() or not (source_root / "mellow/config/v0_local.yaml").is_file():
        raise FileNotFoundError(f"complete Mellow source checkout not found: {source_root}")
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"Mellow snapshot not found: {snapshot}")
    if not checkpoint.is_file() or checkpoint.parent != snapshot:
        raise FileNotFoundError(f"Mellow-v0 checkpoint must be v0.ckpt inside snapshot: {checkpoint}")
    if not base_smollm2.is_dir():
        raise FileNotFoundError(f"base SmolLM2 directory not found: {base_smollm2}")
    _source_inventory(source_root)
    _smollm2_inventory(base_smollm2)
    source_state = _load_state(checkpoint)
    native_model, native_tokenizer = _instantiate_native_model(source_root, base_smollm2)
    native_model.load_state_dict(source_state, strict=True)
    source_config = native_model.caption_decoder.lm.config
    source_geometry = {
        "hidden_size": int(getattr(source_config, "hidden_size", -1)),
        "vocab_size": int(getattr(source_config, "vocab_size", -1)),
        "num_hidden_layers": int(getattr(source_config, "num_hidden_layers", -1)),
    }
    if source_geometry != {"hidden_size": 576, "vocab_size": 49152, "num_hidden_layers": 30}:
        raise RuntimeError(f"unexpected Mellow-v0 source geometry: {source_geometry}")
    del native_model, native_tokenizer
    gc.collect()

    target_model, text_metadata, tokenizer_metadata = _build_target_text(
        source_config,
        source_state,
        checkpoint,
        text_output,
        base_smollm2,
        args.seed,
    )
    init_metadata = _build_full_init_checkpoint(
        target_model,
        source_state,
        init_checkpoint,
        text_output,
        checkpoint,
        source_root,
        snapshot,
        args.official_route_root,
        args.htsat_root,
        args.seed,
    )
    return {
        "status": "PASS",
        "stage": "mellow_v0_to_5_10x2_5_mesh_initialization",
        "artifact_contract": ARTIFACT_CONTRACT,
        "text_output": str(text_output),
        "init_checkpoint": str(init_checkpoint),
        "text_metadata": text_metadata,
        "init_metadata": init_metadata,
        "tokenizer": tokenizer_metadata,
        "cuda_used": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mellow-source-root", type=Path, default=DEFAULT_MELLOW_SOURCE_ROOT)
    parser.add_argument("--mellow-snapshot", type=Path, default=DEFAULT_MELLOW_SNAPSHOT)
    parser.add_argument("--mellow-checkpoint", type=Path, default=DEFAULT_MELLOW_CHECKPOINT)
    parser.add_argument("--base-smollm2", type=Path, default=DEFAULT_BASE_SMOLLM2)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_OUTPUT)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--official-route-root", type=Path, default=DEFAULT_OFFICIAL_ROUTE_ROOT)
    parser.add_argument("--htsat-root", type=Path, default=DEFAULT_HTSAT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = _build(args)
    except Exception as exc:
        import traceback

        report = {
            "status": "FAILED",
            "stage": "mellow_v0_to_5_10x2_5_mesh_initialization",
            "artifact_contract": ARTIFACT_CONTRACT,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "cuda_used": False,
        }
    _write_json(args.report_path.expanduser().resolve(), report)
    if report.get("status") == "PASS":
        text_metadata = report.get("text_metadata", {})
        init_metadata = report.get("init_metadata", {})
        console_report = {
            "status": report.get("status"),
            "stage": report.get("stage"),
            "artifact_contract": report.get("artifact_contract"),
            "text_output": report.get("text_output"),
            "init_checkpoint": report.get("init_checkpoint"),
            "text_geometry": {
                key: text_metadata.get(key)
                for key in (
                    "logical_layer_count",
                    "physical_layer_count",
                    "recursive_loops",
                    "memory_slots",
                    "router_parameter_count",
                    "discarded_source_layer_indices_0based",
                )
            },
            "init_state": {
                key: init_metadata.get(key)
                for key in (
                    "state_tensor_count",
                    "state_parameter_count",
                    "state_group_tensor_counts",
                    "audio_initialization",
                    "router_initialization",
                    "optimizer_scheduler_rng_included",
                )
            },
            "report_path": str(args.report_path.expanduser().resolve()),
            "cuda_used": False,
        }
        print(json.dumps(_json_safe(console_report), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(_json_safe(report), ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
