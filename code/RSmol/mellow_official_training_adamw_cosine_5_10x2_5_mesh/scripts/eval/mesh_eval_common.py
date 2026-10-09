"""Strict runtime/checkpoint adapter for the 5-10x2-5 MeSH eval route."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


ROUTE_CONTRACT = "mellow_official_adamw_cosine_5_10x2_5_mesh_v1"
TEXT_CONTRACT = "logical_30_physical_20_5_10x2_5"
ROUTE_NAME = "mellow_official_training_adamw_cosine_5_10x2_5_mesh"
DEFAULT_CHECKPOINT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "mellow_official_reasonaqa_adamw_cosine_5_10x2_5_mesh_5090/formal_30epochs_20261005_175504/"
    "checkpoints/mellow_adamw_cosine_reasonaqa_mesh_formal_20_20261005_095511181778133/"
    "model--epo-30.ckpt"
)
DEFAULT_RUNTIME = DEFAULT_CHECKPOINT.parents[2] / "runtime_30epochs.yaml"
DEFAULT_ROUTE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = DEFAULT_CHECKPOINT.parents[2] / "eval"


def configure_args(
    args: Any,
    *,
    parser: Any,
    output_name: str,
    dataset_dir: Path,
    metadata_name: str = "mmau-test-mini.json",
    audio_subdir: str = "test-mini-audios",
    evaluation_relative: str = "evaluation.py",
) -> Any:
    args.training_branch = ROUTE_NAME
    args.training_checkpoint = (args.checkpoint_file or DEFAULT_CHECKPOINT).expanduser().resolve(strict=True)
    args.runtime_config = (args.runtime_config or DEFAULT_RUNTIME).expanduser().resolve(strict=True)
    args.route_root = (args.route_root or DEFAULT_ROUTE_ROOT).expanduser().resolve(strict=True)
    args.output_dir = (args.output_dir or DEFAULT_OUTPUT_ROOT / output_name).expanduser().resolve()
    args.dataset_dir = Path(args.dataset_dir or dataset_dir).expanduser().resolve(strict=True)
    parquet = getattr(args, "parquet", None)
    args.parquet = (parquet or args.dataset_dir / "test_mini.parquet").resolve()
    args.metadata_json = (args.metadata_json or args.dataset_dir / metadata_name).resolve()
    args.evaluation_script = (args.evaluation_script or args.dataset_dir / evaluation_relative).resolve()
    args.audio_root = (args.audio_root or args.dataset_dir / audio_subdir).resolve()
    args.checkpoint = args.training_checkpoint.parent
    args.mellow_root = args.route_root
    args.htsat_checkpoint = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT")
    return args


def validate_checkpoint(args: Any, *, expected_epochs: int = 30) -> dict[str, Any]:
    checkpoint = torch.load(args.training_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("state_dict"), Mapping):
        raise RuntimeError("target must be a schema-v2 full training checkpoint with state_dict")
    required = {"schema_version", "route_contract", "text_model_contract", "epoch_completed", "num_epochs", "state_dict"}
    missing = sorted(required.difference(checkpoint))
    if checkpoint.get("schema_version") != 2 or missing:
        raise RuntimeError(f"invalid MeSH checkpoint schema or fields: missing={missing}")
    if checkpoint.get("route_contract") != ROUTE_CONTRACT:
        raise RuntimeError(f"checkpoint route contract mismatch: {checkpoint.get('route_contract')!r}")
    if checkpoint.get("text_model_contract") != TEXT_CONTRACT:
        raise RuntimeError(f"checkpoint text contract mismatch: {checkpoint.get('text_model_contract')!r}")
    if int(checkpoint.get("epoch_completed", -1)) != expected_epochs or int(checkpoint.get("num_epochs", -1)) != expected_epochs:
        raise RuntimeError("target checkpoint must be a completed 30-epoch MeSH checkpoint")
    runtime = yaml.safe_load(args.runtime_config.read_text(encoding="utf-8")) or {}
    if runtime.get("route_contract") != ROUTE_CONTRACT or runtime.get("text_model_contract") != TEXT_CONTRACT:
        raise RuntimeError("runtime config route/text contract does not match the MeSH route")
    state = checkpoint["state_dict"]
    nonfinite = [name for name, value in state.items() if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all().item()]
    if nonfinite:
        raise RuntimeError(f"checkpoint contains non-finite tensors: {nonfinite[:8]}")
    return {
        "schema_version": checkpoint["schema_version"],
        "route_contract": checkpoint["route_contract"],
        "text_model_contract": checkpoint["text_model_contract"],
        "epoch_completed": checkpoint["epoch_completed"],
        "num_epochs": checkpoint["num_epochs"],
        "state_tensor_count": len(state),
    }
