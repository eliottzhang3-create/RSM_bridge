#!/usr/bin/env python3
"""Write a resolved runtime config for the isolated Mellow-v0 two-stage route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ROUTE_CONTRACT = "mellow_v0_official_adamw_cosine_two_stage_5_10x2_5_mesh_v1"
TEXT_CONTRACT = "logical_30_physical_20_5_10x2_5"
MELLOW_INIT_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/converted/"
    "mellow_v0_5_10x2_5_mesh_initialization"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--data-json", "--smoke-json", dest="data_json", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--training-stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--text-model-dir", type=Path, default=None)
    parser.add_argument("--mellow-init-root", type=Path, default=MELLOW_INIT_ROOT)
    parser.add_argument("--htsat-root", type=Path, default=Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT"))
    parser.add_argument("--init-model-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--max-epochs-this-run", type=int, default=0)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--router-max-lr", type=float, default=1e-3)
    parser.add_argument("--router-min-lr", type=float, default=1e-4)
    parser.add_argument("--other-max-lr", type=float, default=1e-4)
    parser.add_argument("--other-min-lr", type=float, default=1e-5)
    parser.add_argument("--all-max-lr", type=float, default=1e-3)
    parser.add_argument("--all-min-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _check_mesh_text(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"mesh text model directory does not exist: {path}")
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"unable to read mesh text config: {path}") from exc
    geometry = (
        int(config.get("hidden_size", -1)),
        int(config.get("num_hidden_layers", -1)),
        int(config.get("recursive_layer_count", -1)),
        int(config.get("recursive_loops", -1)),
        int(config.get("mesh_memory_slots", -1)),
        int(config.get("mesh_router_count", -1)),
    )
    if geometry != (576, 30, 20, 2, 5, 6):
        raise ValueError(f"Mellow-v0 mesh text geometry mismatch: {geometry!r}")
    if config.get("mesh_architecture_contract") != TEXT_CONTRACT:
        raise ValueError("Mellow-v0 mesh text contract mismatch")
    expected_schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    schedule = config.get("logical_to_physical") or config.get("logical_to_physical_schedule", ())
    if list(schedule) != expected_schedule:
        raise ValueError("Mellow-v0 mesh text logical-to-physical schedule mismatch")


def main() -> int:
    args = parse_args()
    if args.num_epochs < 1 or args.max_epochs_this_run < 0 or args.max_optimizer_steps < 0:
        raise ValueError("epoch and optimizer-step limits must be non-negative, with num_epochs >= 1")
    if args.batch_size != 8 or args.gradient_accumulation_steps != 4:
        raise ValueError("the isolated route requires per-rank batch 8 and accumulation 4")
    if not 0.0 < args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be between 0 and 1")
    resolved_stage_root = args.stage_root.resolve()
    if not resolved_stage_root.is_relative_to(Path("/dev/shm")):
        raise ValueError(f"training stage root must be under /dev/shm: {resolved_stage_root}")
    if not (resolved_stage_root / ".mellow_stage" / "READY.json").is_file():
        raise ValueError("training stage is not READY")
    for path, label in ((args.data_json, "training metadata"), (args.htsat_root, "HTSAT root")):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    text_model = (args.text_model_dir or (args.mellow_init_root / "text_model")).expanduser().resolve(strict=True)
    _check_mesh_text(text_model)
    init_checkpoint = args.init_model_checkpoint
    if args.training_stage == "stage1" and init_checkpoint is None and args.resume_checkpoint is None:
        init_checkpoint = args.mellow_init_root / "mellow_v0_5_10x2_5_mesh_init.pt"
    init_path = str(init_checkpoint.expanduser().resolve()) if init_checkpoint else ""
    resume_path = str(args.resume_checkpoint.expanduser().resolve()) if args.resume_checkpoint else ""
    if init_path and not Path(init_path).is_file():
        raise FileNotFoundError(f"initialization checkpoint does not exist: {init_path}")
    if resume_path and not Path(resume_path).is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
    if init_path and resume_path:
        raise ValueError("--init-model-checkpoint and --resume-checkpoint are mutually exclusive")
    if args.training_stage == "stage2" and not init_path and not resume_path:
        raise ValueError("stage2 requires a stage1 initialization checkpoint or a stage2 resume checkpoint")

    if args.training_stage == "stage1":
        parameter_groups = {
            "routers": {"max_lr": args.router_max_lr, "min_lr": args.router_min_lr},
            "other": {"max_lr": args.other_max_lr, "min_lr": args.other_min_lr},
        }
    else:
        parameter_groups = {"all": {"max_lr": args.all_max_lr, "min_lr": args.all_min_lr}}
    for name, bounds in parameter_groups.items():
        if not 0.0 <= float(bounds["min_lr"]) <= float(bounds["max_lr"]):
            raise ValueError(f"invalid learning-rate bounds for {name}: {bounds!r}")

    config: dict[str, Any] = {
        "mode": "train",
        "gpu": True,
        "route_contract": ROUTE_CONTRACT,
        "text_model_contract": TEXT_CONTRACT,
        "training_stage": args.training_stage,
        "resume_checkpoint": resume_path,
        "init_model_checkpoint": init_path,
        "data": {
            "datapath": str(resolved_stage_root),
            "datafiles": [str(args.data_json.resolve())],
            "sampling_rate": 32000,
            "segment_seconds": 10,
            "tokenizer_type": str(text_model),
            "op_text_len": 250,
            "ip_text_len": 129,
        },
        "model": {
            "encoder": {
                "audioenc_name": "HTSAT",
                "transformer_embed_dim": 768,
                "out_emb": 768,
                "d_proj": 576,
                "use_pretrained_audioencoder": True,
                "freeze_audio_encoder_weights": True,
                "pretrained_audioencoder_path": str(args.htsat_root.resolve()),
            },
            "decoder": {
                "text_decoder": str(text_model),
                "prefix_length": 40,
                "total_prefix_length": 389,
                "freeze_gpt_weights": False,
            },
            "model_type": "Mellow",
            "input_channels": 1,
            "output_channels": 1,
            "resume_checkpoint": resume_path,
            "init_model_checkpoint": init_path,
            "inference_window": 5,
        },
        "train": {
            "optimizer": {
                "optimizer_type": "AdamW",
                "betas": [0.9, 0.95],
                "learning_rate": max(float(v["max_lr"]) for v in parameter_groups.values()),
                "parameter_groups": parameter_groups,
                "weight_decay": 1e-4,
                "warmup_ratio": args.warmup_ratio,
                "scheduler": "step_cosine_warmup",
            },
            "max_grad_norm": 0.5,
            "emergency_stop_grad_norm": 1e6,
            "num_nodes": 1,
            "num_workers": args.num_workers,
            "persistent_data_workers": True,
            "loss_type": "sisdr_wav",
            "sync_batchnorm": True,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_epochs": args.num_epochs,
            "max_epochs_this_run": args.max_epochs_this_run,
            "max_optimizer_steps": args.max_optimizer_steps,
            "log_step": 1,
            "sav_per_num_epochs": 1,
            "random_seed": 1234,
            "mixed_precision": {"use_mixed_precision": False, "mixed_precision_dtype": "float16"},
        },
        "save_dir": str(args.save_dir.resolve()),
        "loss_reduction": "global_token_mean",
    }
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    print(f"wrote runtime config: {args.output_config}", flush=True)
    print(f"training_stage={args.training_stage} text_model={text_model} init={init_path or '<none>'} resume={resume_path or '<none>'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
