#!/usr/bin/env python3
"""Write the fully resolved official-Mellow runtime YAML for one job."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--smoke-json", "--data-json", dest="data_json", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument(
        "--text-model-dir",
        type=Path,
        default=Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2"),
    )
    parser.add_argument(
        "--htsat-root",
        type=Path,
        default=Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--max-epochs-this-run", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--init-model-checkpoint", type=Path, default=None)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_epochs < 1:
        raise ValueError("--num-epochs must be at least 1")
    if args.max_epochs_this_run < 0:
        raise ValueError("--max-epochs-this-run must be non-negative")
    if args.max_optimizer_steps < 0:
        raise ValueError("--max-optimizer-steps must be non-negative")
    resolved_stage_root = args.stage_root.resolve()
    shm_root = Path("/dev/shm").resolve()
    if not resolved_stage_root.is_relative_to(shm_root):
        raise ValueError(f"training stage root must be under /dev/shm: {resolved_stage_root}")
    if not (resolved_stage_root / ".mellow_stage" / "READY.json").is_file():
        raise ValueError("training stage is not READY")
    for path, label in (
        (resolved_stage_root, "stage root"),
        (args.data_json, "training metadata"),
        (args.text_model_dir, "text model"),
        (args.htsat_root, "HTSAT root"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if args.batch_size != 4 or args.gradient_accumulation_steps != 1:
        raise ValueError("the isolated route requires per-rank batch 4 and accumulation 1")
    resume_checkpoint = str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else ""
    if resume_checkpoint and not Path(resume_checkpoint).is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_checkpoint}")
    init_model_checkpoint = str(args.init_model_checkpoint.resolve()) if args.init_model_checkpoint else ""
    if init_model_checkpoint and not Path(init_model_checkpoint).is_file():
        raise FileNotFoundError(f"initialization checkpoint does not exist: {init_model_checkpoint}")
    if resume_checkpoint and init_model_checkpoint:
        raise ValueError("--resume-checkpoint and --init-model-checkpoint are mutually exclusive")
    if not 0.0 < args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be between 0 and 1")
    if not 0.0 <= args.min_lr <= args.max_lr:
        raise ValueError("--min-lr must be between 0 and --max-lr")

    config: dict[str, Any] = {
        "mode": "train",
        "gpu": True,
        "resume_checkpoint": resume_checkpoint,
        "init_model_checkpoint": init_model_checkpoint,
        "data": {
            "datapath": str(resolved_stage_root),
            "datafiles": [str(args.data_json.resolve())],
            "sampling_rate": 32000,
            "segment_seconds": 10,
            "tokenizer_type": str(args.text_model_dir.resolve()),
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
                "text_decoder": str(args.text_model_dir.resolve()),
                "prefix_length": 40,
                "total_prefix_length": 389,
                "freeze_gpt_weights": False,
            },
            "model_type": "Mellow",
            "input_channels": 1,
            "output_channels": 1,
            "resume_checkpoint": resume_checkpoint,
            "init_model_checkpoint": init_model_checkpoint,
            "inference_window": 5,
        },
        "train": {
            "optimizer": {
                "optimizer_type": "AdamW",
                "betas": [0.9, 0.95],
                "learning_rate": args.max_lr,
                "max_lr": args.max_lr,
                "min_lr": args.min_lr,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": 1e-4,
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
            "mixed_precision": {
                "use_mixed_precision": False,
                "mixed_precision_dtype": "float16",
            },
        },
        "save_dir": str(args.save_dir.resolve()),
    }

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    print(f"wrote runtime config: {args.output_config}", flush=True)
    print(
        f"effective global batch requires world_size=8: "
        f"{args.batch_size} x 8 x {args.gradient_accumulation_steps} = "
        f"{args.batch_size * 8 * args.gradient_accumulation_steps}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
