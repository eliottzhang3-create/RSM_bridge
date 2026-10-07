#!/usr/bin/env python3
"""Build the fresh Mellow-v0 -> 5-10x2-5 MeSH initialization artifacts.

This is a CPU-only conversion for the official Mellow ReasonAQA route.  It
creates two artifacts:

* a standalone 5-10x2-5 MeSH text directory; and
* a full ``--init-model-checkpoint`` containing the Mellow state-dict layout.

The text decoder tensors are copied from Mellow-v0 using the audited
30-to-20 physical-layer mapping. Routers can either use the original fresh
MeSH initialization or the audited epoch-17 router source. Mellow's HTSAT,
c2l, and projection tensors are copied unchanged. No optimizer, scheduler,
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
from audit_mellow_v0_mesh_router_source import (  # noqa: E402
    BASE as ROUTER_BASELINE,
    ROUTER_KEYS,
    audit as audit_router_source,
    load as load_router_checkpoint,
    sha256_tensor,
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
EPOCH17_ARTIFACT_CONTRACT = "mellow_v0_to_5_10x2_5_mesh_epoch17_router_initialization_v1"
DEFAULT_OUTPUT_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/converted/"
    "mellow_v0_5_10x2_5_mesh_initialization"
)
DEFAULT_TEXT_OUTPUT = DEFAULT_OUTPUT_ROOT / "text_model"
DEFAULT_INIT_CHECKPOINT = DEFAULT_OUTPUT_ROOT / "mellow_v0_5_10x2_5_mesh_init.pt"
DEFAULT_OFFICIAL_ROUTE_ROOT = RSMOL_ROOT / "mellow_official_training_adamw_cosine_5_10x2_5_mesh"
DEFAULT_TWO_STAGE_ROUTE_ROOT = RSMOL_ROOT / "mellow_official_training_mellow_v0_two_stage_adamw_cosine_5_10x2_5_mesh"
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
    router_tensors: Mapping[str, Any] | None = None,
    router_source_path: Path | None = None,
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
    if router_tensors is not None:
        expected_full_keys = {f"{TEXT_PREFIX}{key}" for key in expected_router_keys}
        if set(router_tensors) != expected_full_keys:
            raise RuntimeError("audited router source key set differs from target MeSH routers")
        with torch.no_grad():
            for key in expected_router_keys:
                source_value = router_tensors[f"{TEXT_PREFIX}{key}"]
                target_value = target_model.state_dict()[key]
                if source_value.shape != target_value.shape or source_value.dtype != target_value.dtype:
                    raise RuntimeError(f"router shape/dtype mismatch: {key}")
                if not torch.isfinite(source_value).all().item():
                    raise RuntimeError(f"non-finite router source: {key}")
                target_value.copy_(source_value)
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
            "router_initialization": "epoch-17 checkpoint" if router_tensors is not None else "retained from fresh MeSH construction",
            "router_source_checkpoint": str(router_source_path) if router_source_path else None,
            "parameter_audit": parameter_audit(target_model),
            "tokenizer": tokenizer_audit,
            "seed": seed,
        }
        (staging / "mellow_v0_mesh_text_initialization.json").write_text(
            json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if router_tensors is not None:
            # Verify the serialized text model, including tied embedding aliases.
            disk_model = RecursiveLlamaForCausalLM.from_pretrained(staging, local_files_only=True)
            disk_state = disk_model.state_dict()
            memory_state = target_model.state_dict()
            if set(disk_state) != set(memory_state):
                raise RuntimeError("serialized text model key inventory differs from target")
            for key in memory_state:
                if not torch.equal(disk_state[key].cpu(), memory_state[key].cpu()):
                    raise RuntimeError(f"serialized text model tensor differs: {key}")
            del disk_model, disk_state
            gc.collect()
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
    router_source_path: Path | None = None,
    router_audit: Mapping[str, Any] | None = None,
    baseline_state: Mapping[str, Any] | None = None,
    router_tensors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    target_state = target_model.state_dict()
    full_state: OrderedDict[str, Any] = OrderedDict()
    # Copy the three audio groups directly.  The source text group is
    # deliberately omitted here because the target text group is rebuilt
    # below with the 30-to-20 layer mapping and selected routers.
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
        "artifact_contract": EPOCH17_ARTIFACT_CONTRACT if router_source_path else ARTIFACT_CONTRACT,
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
        "router_initialization": "epoch-17 checkpoint" if router_source_path else "fresh MeSH truncated-normal initialization",
        "router_source_checkpoint": str(router_source_path) if router_source_path else None,
        "router_source_sha256": router_audit.get("source_sha256") if router_audit else None,
        "router_source_audit_contract": router_audit.get("artifact_contract") if router_audit else None,
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
    if baseline_state is not None:
        if len(full_state) != 401 or len(baseline_state) != 401:
            raise RuntimeError("epoch-17 initialization requires exactly 401 model tensors")
        metadata["verified_tensor_count"] = len(full_state)
        metadata["verified_router_tensor_count"] = len(ROUTER_KEYS)
        metadata["verified_unchanged_tensor_count"] = len(full_state) - len(ROUTER_KEYS)
    payload = {"state_dict": full_state, "metadata": metadata}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite initialization checkpoint: {output_path}")
    try:
        torch.save(payload, temporary)
        if baseline_state is not None:
            if router_tensors is None or set(router_tensors) != ROUTER_KEYS:
                raise RuntimeError("router source tensor inventory is incomplete")
            disk_payload = load_router_checkpoint(temporary)
            disk_state = disk_payload.get("state_dict")
            disk_metadata = disk_payload.get("metadata")
            if not isinstance(disk_metadata, Mapping) or disk_metadata != metadata:
                raise RuntimeError("serialized full initialization metadata differs")
            if not isinstance(disk_state, Mapping) or set(disk_state) != set(baseline_state):
                raise RuntimeError("serialized full initialization key inventory differs from baseline")
            for key, baseline_value in baseline_state.items():
                expected = router_tensors[key] if key in ROUTER_KEYS else baseline_value
                actual = disk_state[key]
                if not torch.is_tensor(actual) or actual.shape != expected.shape or actual.dtype != expected.dtype:
                    raise RuntimeError(f"serialized tensor shape/dtype differs: {key}")
                if not torch.equal(actual, expected):
                    raise RuntimeError(f"serialized tensor value differs: {key}")
                if actual.is_floating_point() and not torch.isfinite(actual).all().item():
                    raise RuntimeError(f"serialized tensor is non-finite: {key}")
            disk_payload = None
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
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
    router_source_path = None
    router_audit = None
    router_tensors = None
    baseline_state = None
    if args.router_source_checkpoint is not None:
        if args.seed != 0:
            raise ValueError("epoch-17 transfer requires seed=0 to reproduce the audited baseline")
        router_source_path = args.router_source_checkpoint.expanduser().resolve(strict=True)
        baseline_path = args.baseline_init.expanduser().resolve(strict=True)
        if baseline_path != ROUTER_BASELINE.expanduser().resolve(strict=True):
            raise ValueError("epoch-17 mode requires the original Mellow-v0 baseline initialization")
        old_root = DEFAULT_OUTPUT_ROOT.resolve(strict=True)
        output_root = text_output.parent
        if output_root == old_root or output_root.is_relative_to(old_root) or old_root.is_relative_to(output_root):
            raise ValueError("epoch-17 output root must be separate from the original initialization root")
        if (text_output.parent != init_checkpoint.parent or text_output.name != "text_model"
                or init_checkpoint.name != DEFAULT_INIT_CHECKPOINT.name):
            raise ValueError("epoch-17 outputs must be <new-root>/text_model and <new-root>/mellow_v0_5_10x2_5_mesh_init.pt")
        if text_output.exists() or init_checkpoint.exists():
            raise FileExistsError("refusing existing epoch-17 output; choose a fresh directory")
        router_audit = audit_router_source(router_source_path, baseline_path)
        if router_audit["source_sha256"] != args.expected_router_source_sha256:
            raise ValueError("router source SHA256 differs from the audited epoch-17 checkpoint")
        if router_audit["baseline_sha256"] != args.expected_baseline_sha256:
            raise ValueError("baseline SHA256 differs from the audited Mellow-v0 initialization")
        router_source = load_router_checkpoint(router_source_path)
        baseline = load_router_checkpoint(baseline_path)
        baseline_metadata = baseline["metadata"]
        if int(baseline_metadata.get("seed", -1)) != args.seed:
            raise ValueError("baseline initialization seed does not match conversion seed")
        if baseline_metadata.get("source_checkpoint_sha256") != _sha256(checkpoint):
            raise ValueError("Mellow-v0 source checkpoint differs from baseline initialization")
        router_tensors = {key: router_source["state_dict"][key].detach().cpu().clone() for key in ROUTER_KEYS}
        for key, value in router_tensors.items():
            if sha256_tensor(value) != router_audit["router_tensors"][key]["sha256"]:
                raise RuntimeError(f"router source changed after audit: {key}")
        baseline_state = baseline["state_dict"]
        del router_source, baseline
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
        router_tensors,
        router_source_path,
    )
    route_root = DEFAULT_TWO_STAGE_ROUTE_ROOT if router_source_path is not None else args.official_route_root
    if router_source_path is not None and args.official_route_root != DEFAULT_OFFICIAL_ROUTE_ROOT:
        if args.official_route_root.expanduser().resolve() != DEFAULT_TWO_STAGE_ROUTE_ROOT.resolve():
            raise ValueError("epoch-17 mode must validate against the isolated two-stage route")
    init_metadata = _build_full_init_checkpoint(
        target_model,
        source_state,
        init_checkpoint,
        text_output,
        checkpoint,
        source_root,
        snapshot,
        route_root,
        args.htsat_root,
        args.seed,
        router_source_path,
        router_audit,
        baseline_state,
        router_tensors,
    )
    return {
        "status": "PASS",
        "stage": "mellow_v0_to_5_10x2_5_mesh_initialization",
        "artifact_contract": EPOCH17_ARTIFACT_CONTRACT if router_source_path else ARTIFACT_CONTRACT,
        "text_output": str(text_output),
        "init_checkpoint": str(init_checkpoint),
        "text_metadata": text_metadata,
        "init_metadata": init_metadata,
        "tokenizer": tokenizer_metadata,
        "router_source_audit": {key: router_audit[key] for key in (
            "source_sha256", "baseline_sha256", "router_tensor_count",
            "routers_different_from_baseline",
        )} if router_audit else None,
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
    parser.add_argument("--router-source-checkpoint", type=Path, default=None)
    parser.add_argument("--baseline-init", type=Path, default=ROUTER_BASELINE)
    parser.add_argument("--expected-router-source-sha256", default="3eeed406c6125056d3bc2be4b9d136085a1ded7864e174f18fa83fefb657e4b9")
    parser.add_argument("--expected-baseline-sha256", default="8af13e26e26c3ab9019ea69de431c6823584c657dcc9e3007ea19f1d43b6d3ef")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = args.report_path.expanduser().resolve()
    if args.router_source_checkpoint is not None:
        if report_path == DEFAULT_REPORT.resolve() or report_path.exists():
            raise SystemExit("epoch-17 mode requires a new --report-path; existing reports will not be overwritten")
    try:
        report = _build(args)
    except Exception as exc:
        import traceback

        report = {
            "status": "FAILED",
            "stage": "mellow_v0_to_5_10x2_5_mesh_initialization",
            "artifact_contract": EPOCH17_ARTIFACT_CONTRACT if args.router_source_checkpoint is not None else ARTIFACT_CONTRACT,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "cuda_used": False,
        }
    _write_json(report_path, report)
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
                    "router_source_checkpoint",
                    "router_source_sha256",
                    "verified_tensor_count",
                    "verified_router_tensor_count",
                    "verified_unchanged_tensor_count",
                    "optimizer_scheduler_rng_included",
                )
            },
            "report_path": str(report_path),
            "cuda_used": False,
        }
        print(json.dumps(_json_safe(console_report), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(_json_safe(report), ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
