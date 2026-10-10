#!/usr/bin/env python3
"""Evaluate any completed gbs256 official-Mellow checkpoint on MMAU test-mini."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import yaml

import evaluate_mmau_test_mini_mellow_official_training as official_training
import evaluate_mellow_official_training_common as adapter


DEFAULT_DATASET = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini")
ROUTE_ROOTS = {
    "global_token_mean": "reasonaqa_global_batch256_lr_ablation_5epochs",
    "equal_microbatch_token_mean": "reasonaqa_global_batch256_old_microbatch_mean_5epochs",
}
ROUTE_PARENT = Path(__file__).resolve().parent.parent / "mellow_official_training_c8204d8_adamw_cosine"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--route-root", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--evaluation-script", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--parquet-batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=adapter.PROMPT_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--dtype", choices=("fp32",), default="fp32")
    args = parser.parse_args(argv)
    args.checkpoint_file = args.checkpoint_file.expanduser().resolve(strict=True)
    args.runtime_config = args.runtime_config.expanduser().resolve(strict=True)
    args.dataset_dir = args.dataset_dir.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.route_root is not None:
        args.route_root = args.route_root.expanduser().resolve(strict=True)
    for name in ("parquet", "metadata_json", "evaluation_script", "audio_root"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve(strict=True))
    if args.max_prompt_tokens != adapter.PROMPT_TOKENS:
        parser.error(f"--max-prompt-tokens is fixed at {adapter.PROMPT_TOKENS}")
    if args.max_new_tokens != 300:
        parser.error("--max-new-tokens is fixed at 300 for MMAU")
    if args.parquet_batch_size <= 0:
        parser.error("--parquet-batch-size must be positive")
    return args


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("checkpoint must be a mapping")
    required = {
        "schema_version", "state_dict", "epoch_completed", "num_epochs",
        "batch_geometry", "loss_reduction", "optimizer_contract", "scheduler", "total_step",
    }
    missing = sorted(required.difference(checkpoint))
    if missing or checkpoint.get("schema_version") != 2:
        raise RuntimeError(
            f"expected schema-v2 official Mellow checkpoint; "
            f"schema={checkpoint.get('schema_version')!r}, missing={missing}"
        )
    if not isinstance(checkpoint["state_dict"], dict) or not checkpoint["state_dict"]:
        raise RuntimeError("checkpoint state_dict must be a non-empty mapping")
    if int(checkpoint["epoch_completed"]) != 5 or int(checkpoint["num_epochs"]) != 5:
        raise RuntimeError("gbs256 MMAU evaluator requires a completed five-epoch checkpoint")
    expected_geometry = {
        "per_rank_batch_size": 8,
        "world_size": 8,
        "gradient_accumulation_steps": 4,
    }
    if checkpoint["batch_geometry"] != expected_geometry:
        raise RuntimeError(
            f"checkpoint batch geometry mismatch: expected={expected_geometry!r}, "
            f"actual={checkpoint['batch_geometry']!r}"
        )
    loss_reduction = str(checkpoint["loss_reduction"])
    if loss_reduction not in ROUTE_ROOTS:
        raise RuntimeError(f"unsupported gbs256 loss reduction: {loss_reduction!r}")
    optimizer = checkpoint["optimizer_contract"]
    if optimizer.get("type") != "AdamW" or tuple(optimizer.get("betas", ())) != (0.9, 0.95):
        raise RuntimeError(f"unexpected optimizer contract: {optimizer!r}")
    scheduler = checkpoint["scheduler"]
    if not isinstance(scheduler, dict) or scheduler.get("scheduler_type") != "step_cosine_warmup":
        raise RuntimeError(f"unexpected scheduler contract: {scheduler!r}")
    max_lr = float(scheduler.get("max_lr", 0.0))
    min_lr = float(scheduler.get("min_lr", 0.0))
    if not math.isfinite(max_lr) or not math.isfinite(min_lr) or not (0.0 < min_lr <= max_lr):
        raise RuntimeError(f"invalid scheduler learning-rate bounds: {scheduler!r}")
    if not math.isclose(min_lr, 0.1 * max_lr, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("checkpoint scheduler min_lr must be 0.1 * max_lr")
    total_steps = int(scheduler.get("total_steps", -1))
    warmup_steps = int(scheduler.get("warmup_steps", -1))
    if (
        total_steps <= 0 or total_steps % 5 != 0
        or total_steps != int(checkpoint["total_step"])
        or int(scheduler.get("last_step", -1)) != total_steps
        or warmup_steps != math.ceil(total_steps * 0.05)
    ):
        raise RuntimeError(f"checkpoint scheduler step contract mismatch: {scheduler!r}")
    return {
        "schema_version": 2,
        "epoch_completed": 5,
        "num_epochs": 5,
        "batch_geometry": expected_geometry,
        "effective_global_batch": 256,
        "loss_reduction": loss_reduction,
        "optimizer_contract": optimizer,
        "scheduler": {
            "scheduler_type": scheduler["scheduler_type"],
            "max_lr": max_lr,
            "min_lr": min_lr,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
        },
        "state_tensor_count": len(checkpoint["state_dict"]),
    }


def _validate_runtime(path: Path, checkpoint_contract: dict[str, Any]) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise RuntimeError("runtime config must be a mapping")
    model = config.get("model") or {}
    encoder = model.get("encoder") or {}
    decoder = model.get("decoder") or {}
    train = config.get("train") or {}
    expected = {
        "model_type": "Mellow", "audioenc_name": "HTSAT", "d_proj": 576,
        "prefix_length": 40, "total_prefix_length": 389, "ip_text_len": 129,
        "batch_size": 8, "gradient_accumulation_steps": 4, "num_epochs": 5,
    }
    actual = {
        "model_type": model.get("model_type"), "audioenc_name": encoder.get("audioenc_name"),
        "d_proj": encoder.get("d_proj"), "prefix_length": decoder.get("prefix_length"),
        "total_prefix_length": decoder.get("total_prefix_length"),
        "ip_text_len": (config.get("data") or {}).get("ip_text_len"),
        "batch_size": train.get("batch_size"),
        "gradient_accumulation_steps": train.get("gradient_accumulation_steps"),
        "num_epochs": train.get("num_epochs"),
    }
    mismatches = {key: {"expected": value, "actual": actual[key]} for key, value in expected.items() if actual[key] != value}
    if mismatches:
        raise RuntimeError(f"runtime config contract mismatch: {mismatches}")
    optimizer = train.get("optimizer") or {}
    scheduler = checkpoint_contract["scheduler"]
    if optimizer.get("optimizer_type") != "AdamW" or tuple(optimizer.get("betas", ())) != (0.9, 0.95):
        raise RuntimeError("runtime config optimizer must be AdamW with betas (0.9, 0.95)")
    for name in ("max_lr", "min_lr"):
        runtime_value = float(optimizer.get(name, float("nan")))
        if not math.isclose(runtime_value, scheduler[name], rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"runtime config {name} differs from checkpoint scheduler")
    if not math.isclose(float(optimizer.get("warmup_ratio", float("nan"))), 0.05, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("runtime config warmup ratio must be 0.05")
    if not math.isclose(float(optimizer.get("weight_decay", float("nan"))), 1e-4, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("runtime config weight decay must be 1e-4")
    if not math.isclose(float(optimizer.get("learning_rate", float("nan"))), scheduler["max_lr"], rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("runtime config learning_rate differs from checkpoint scheduler max_lr")
    if not str(decoder.get("text_decoder", "")).endswith("/SmolLM2"):
        raise RuntimeError("runtime config must use the standard SmolLM2 decoder")
    if not str(encoder.get("pretrained_audioencoder_path", "")).endswith("/HTSAT"):
        raise RuntimeError("runtime config must use the standard HTSAT root")
    return {
        "model_type": actual["model_type"], "audioenc_name": actual["audioenc_name"],
        "text_decoder": str(decoder.get("text_decoder")),
        "htsat_root": str(encoder.get("pretrained_audioencoder_path")),
        "batch_geometry": {"per_rank_batch_size": 8, "world_size": 8, "gradient_accumulation_steps": 4},
        "num_epochs": 5, "optimizer": train.get("optimizer") or {},
    }


def _delegate_args(args: argparse.Namespace) -> argparse.Namespace:
    delegate_argv = [
        "--route", "adamw_cosine", "--mode", args.mode,
        "--checkpoint-file", str(args.checkpoint_file), "--runtime-config", str(args.runtime_config),
        "--route-root", str(args.route_root), "--dataset-dir", str(args.dataset_dir),
        "--output-dir", str(args.output_dir), "--parquet-batch-size", str(args.parquet_batch_size),
        "--max-prompt-tokens", str(args.max_prompt_tokens), "--max-new-tokens", str(args.max_new_tokens),
        "--dtype", args.dtype, "--run-official-evaluation",
    ]
    for option, value in (("--parquet", args.parquet), ("--metadata-json", args.metadata_json), ("--evaluation-script", args.evaluation_script), ("--audio-root", args.audio_root)):
        if value is not None:
            delegate_argv.extend([option, str(value)])
    delegated = official_training.parse_args(delegate_argv)
    delegated.training_branch = str(args.route_root.name)
    return delegated


def _claim_output(args: argparse.Namespace) -> None:
    """Protect the resumable evaluation store from a different checkpoint."""
    identity = {
        "checkpoint_file": str(args.checkpoint_file),
        "checkpoint_sha256": adapter.sha256_file(args.checkpoint_file),
        "runtime_config": str(args.runtime_config),
        "runtime_config_sha256": adapter.sha256_file(args.runtime_config),
        "route_root": str(args.route_root),
        "dataset_dir": str(args.dataset_dir),
    }
    marker = args.output_dir / "gbs256_eval_identity.json"
    if marker.is_file():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous != identity:
            raise RuntimeError(f"evaluation output belongs to a different checkpoint or config: {marker}")
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"evaluation output is nonempty and has no gbs256 identity marker: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(identity, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_contract = _load_checkpoint(args.checkpoint_file)
    expected_route = (ROUTE_PARENT / ROUTE_ROOTS[checkpoint_contract["loss_reduction"]]).resolve(strict=True)
    if args.route_root is not None and args.route_root != expected_route:
        raise RuntimeError(f"route root does not match loss reduction: expected={expected_route}, actual={args.route_root}")
    args.route_root = expected_route
    run_root = args.checkpoint_file.parent.parent.parent
    if args.runtime_config.parent != run_root:
        raise RuntimeError(f"runtime config must belong to checkpoint run {run_root}")
    runtime_contract = _validate_runtime(args.runtime_config, checkpoint_contract)
    _claim_output(args)
    report = official_training.run(_delegate_args(args))
    report["gbs256_checkpoint_contract"] = checkpoint_contract
    report["gbs256_runtime_contract"] = runtime_contract
    report["gbs256_route_root"] = str(args.route_root)
    adapter.write_json(args.output_dir / "evaluation_report.json", report)
    print(json.dumps({"status": report.get("status"), "mode": report.get("mode"), "loss_reduction": checkpoint_contract["loss_reduction"], "primary_comparison_score": report.get("primary_comparison_score", {}), "report": str(args.output_dir / "evaluation_report.json")}, ensure_ascii=False, default=official_training.official._json_default))
    comparable = bool((report.get("primary_comparison_score") or {}).get("comparable"))
    return 0 if report.get("status") == "PASS" and (args.mode != "full" or comparable) else 1


if __name__ == "__main__":
    raise SystemExit(main())
