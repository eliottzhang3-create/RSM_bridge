#!/usr/bin/env python3
"""Isolated Stage 4 DDP trainer for MeSH 5-10x2-5.

The implementation keeps the fixed parquet manifest/cursor and token-weighted
gradient contract used by the project while making the MeSH model and its six
routers explicit.  It is intentionally self-contained so the older training
scripts remain byte-for-byte untouched.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from recursive_model_5_10x2_5_mesh import (  # noqa: E402
    LOGICAL_TO_PHYSICAL,
    MEMORY_SLOT_COUNT,
    RecursiveLlamaForCausalLM,
    parameter_audit,
    register_auto_class,
)

MODEL_ARCHITECTURE_CONTRACT = "logical_30_physical_20_5_10x2_5_mesh"
DATA_ROOT_DEFAULT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset/data")
OUTPUT_ROOT_DEFAULT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh")
DEFAULT_WORLD_SIZE = 8
DEFAULT_MICRO_BATCH_SIZE = 8
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 16
DEFAULT_CONTEXT_LENGTH = 1024
DEFAULT_FORMAL_OPTIMIZER_STEPS = 9244
DEFAULT_FORMAL_WARMUP_STEPS = 463
DEFAULT_MAX_LR = 8e-4
DEFAULT_MIN_LR = 8e-5
DEFAULT_SAVE_EVERY = 500
DEFAULT_CHECKPOINT_RETENTION = 3
DEFAULT_ADAMW_BETAS = (0.9, 0.95)
DEFAULT_ADAMW_WEIGHT_DECAY = 0.1
DEFAULT_ADAMW_EPS = 1e-8
DEFAULT_ADAMW_AMSGRAD = False


@dataclass
class Stage4Config:
    gate: str = "D"
    model_path: Path | None = None
    tokenizer_path: Path | None = None
    data_dir: Path = DATA_ROOT_DEFAULT
    output_dir: Path = OUTPUT_ROOT_DEFAULT
    report_path: Path | None = None
    resume_from: Path | None = None
    world_size: int = DEFAULT_WORLD_SIZE
    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    context_length: int = DEFAULT_CONTEXT_LENGTH
    max_optimizer_steps: int = 10
    scheduler_total_steps: int = DEFAULT_FORMAL_OPTIMIZER_STEPS
    warmup_steps: int = DEFAULT_FORMAL_WARMUP_STEPS
    max_lr: float = DEFAULT_MAX_LR
    min_lr: float = DEFAULT_MIN_LR
    save_every: int = DEFAULT_SAVE_EVERY
    checkpoint_retention: int = DEFAULT_CHECKPOINT_RETENTION
    seed: int = 0
    num_workers: int = 0
    max_microbatches: int | None = None


def _env(name: str, default: Any) -> Any:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _parse_args(argv: list[str] | None = None) -> Stage4Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("A", "D", "E", "FORMAL"), default=_env("RSMOL_5_10X2_5_MESH_STAGE4_GATE", "D"))
    parser.add_argument("--model-path", type=Path, default=Path(_env("RSMOL_5_10X2_5_MESH_MODEL_DIR", "")) if _env("RSMOL_5_10X2_5_MESH_MODEL_DIR", "") else None)
    parser.add_argument("--tokenizer-path", type=Path, default=Path(_env("RSMOL_5_10X2_5_MESH_TOKENIZER_PATH", "")) if _env("RSMOL_5_10X2_5_MESH_TOKENIZER_PATH", "") else None)
    parser.add_argument("--data-dir", type=Path, default=Path(_env("RSMOL_5_10X2_5_MESH_DATA_DIR", str(DATA_ROOT_DEFAULT))))
    parser.add_argument("--output-dir", type=Path, default=Path(_env("RSMOL_5_10X2_5_MESH_OUTPUT_DIR", str(OUTPUT_ROOT_DEFAULT))))
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=Path(_env("RSMOL_5_10X2_5_MESH_RESUME_FROM", "")) if _env("RSMOL_5_10X2_5_MESH_RESUME_FROM", "") else None)
    parser.add_argument("--world-size", type=int, default=int(_env("RSMOL_5_10X2_5_MESH_WORLD_SIZE", DEFAULT_WORLD_SIZE)))
    parser.add_argument("--micro-batch-size", type=int, default=int(_env("RSMOL_5_10X2_5_MESH_MICRO_BATCH_SIZE", DEFAULT_MICRO_BATCH_SIZE)))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=int(_env("RSMOL_5_10X2_5_MESH_GRADIENT_ACCUMULATION_STEPS", DEFAULT_GRADIENT_ACCUMULATION_STEPS)))
    parser.add_argument("--context-length", type=int, default=int(_env("RSMOL_5_10X2_5_MESH_CONTEXT_LENGTH", DEFAULT_CONTEXT_LENGTH)))
    parser.add_argument("--max-optimizer-steps", type=int, default=None)
    parser.add_argument("--scheduler-total-steps", type=int, default=int(_env("RSMOL_5_10X2_5_MESH_SCHEDULER_TOTAL_STEPS", DEFAULT_FORMAL_OPTIMIZER_STEPS)))
    parser.add_argument("--warmup-steps", type=int, default=int(_env("RSMOL_5_10X2_5_MESH_WARMUP_STEPS", DEFAULT_FORMAL_WARMUP_STEPS)))
    parser.add_argument("--max-lr", type=float, default=float(_env("RSMOL_5_10X2_5_MESH_MAX_LR", DEFAULT_MAX_LR)))
    parser.add_argument("--min-lr", type=float, default=float(_env("RSMOL_5_10X2_5_MESH_MIN_LR", DEFAULT_MIN_LR)))
    parser.add_argument("--save-every", type=int, default=int(_env("RSMOL_5_10X2_5_MESH_SAVE_EVERY", DEFAULT_SAVE_EVERY)))
    parser.add_argument("--seed", type=int, default=int(_env("RSMOL_5_10X2_5_MESH_SEED", 0)))
    parser.add_argument("--max-microbatches", type=int, default=None)
    args = parser.parse_args(argv)
    gate = str(args.gate).upper()
    max_steps = args.max_optimizer_steps if args.max_optimizer_steps is not None else (DEFAULT_FORMAL_OPTIMIZER_STEPS if gate == "FORMAL" else (2 if gate == "E" else (10 if gate == "D" else 2)))
    if gate == "FORMAL":
        if (args.world_size, args.micro_batch_size, args.gradient_accumulation_steps, max_steps, args.scheduler_total_steps, args.warmup_steps) != (8, 8, 16, 9244, 9244, 463):
            raise ValueError("FORMAL requires 8 ranks, microbatch=8, GA=16, 9244 steps, scheduler_total_steps=9244, warmup=463")
        if not math.isclose(args.max_lr, 8e-4) or not math.isclose(args.min_lr, 8e-5):
            raise ValueError("FORMAL requires max_lr=8e-4 and min_lr=8e-5")
        if args.save_every != 500:
            raise ValueError("FORMAL requires save_every=500")
    if gate == "E" and args.resume_from is None:
        raise ValueError("Gate E requires --resume-from")
    return Stage4Config(gate=gate, model_path=args.model_path, tokenizer_path=args.tokenizer_path, data_dir=args.data_dir, output_dir=args.output_dir, report_path=args.report_path, resume_from=args.resume_from, world_size=args.world_size, micro_batch_size=args.micro_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, context_length=args.context_length, max_optimizer_steps=max_steps, scheduler_total_steps=args.scheduler_total_steps, warmup_steps=args.warmup_steps, max_lr=args.max_lr, min_lr=args.min_lr, save_every=args.save_every, checkpoint_retention=DEFAULT_CHECKPOINT_RETENTION, seed=args.seed, max_microbatches=args.max_microbatches)


def token_weighted_gradient_scale(*, world_size: int, global_window_tokens: int, gradient_accumulation_steps: int) -> float:
    if global_window_tokens <= 0:
        raise ValueError("global token count must be positive")
    return float(world_size * gradient_accumulation_steps / global_window_tokens)


def _dist_setup(config: Stage4Config) -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    detected_world = int(os.environ.get("WORLD_SIZE", str(config.world_size)))
    if detected_world > 1:
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, rank=rank, world_size=detected_world)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, detected_world, device


def _seed(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _manifest(data_dir: Path) -> list[Path]:
    candidates = [data_dir]
    nested_data = data_dir / "data"
    if nested_data != data_dir:
        candidates.append(nested_data)
    selected_dir: Path | None = None
    paths: list[Path] = []
    for candidate in candidates:
        candidate_paths = sorted(candidate.glob("*.parquet"))
        if candidate_paths:
            selected_dir = candidate
            paths = candidate_paths
            break
    if not paths:
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"no parquet shards found; checked: {checked}")
    if selected_dir is None:
        raise AssertionError("manifest selected directory was not recorded")
    return paths


class DistributedParquetStream:
    """Small deterministic row cursor over fixed parquet shard order."""

    cursor_policy = "fixed_sorted_shards_rank_round_robin_row_cursor"

    def __init__(self, paths: list[Path], tokenizer: Any, *, rank: int, world_size: int, batch_size: int, context_length: int, pad_token_id: int, seed: int = 0) -> None:
        self.paths = paths
        self.local_paths = [p for i, p in enumerate(paths) if i % world_size == rank]
        self.tokenizer = tokenizer
        self.rank = rank
        self.batch_size = batch_size
        self.context_length = context_length
        self.pad_token_id = pad_token_id
        self.shard_index = 0
        self.row_offset = 0
        self.microbatches_seen = 0
        self.seed = seed

    def cursor(self) -> dict[str, Any]:
        return {"rank": self.rank, "shard_index": self.shard_index, "row_offset": self.row_offset, "microbatches_seen": self.microbatches_seen, "policy": self.cursor_policy}

    def restore_cursor(self, value: dict[str, Any] | None) -> None:
        if value:
            self.shard_index = int(value.get("shard_index", 0))
            self.row_offset = int(value.get("row_offset", 0))
            self.microbatches_seen = int(value.get("microbatches_seen", 0))

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        import pyarrow.parquet as pq
        while self.shard_index < len(self.local_paths):
            path = self.local_paths[self.shard_index]
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=self.batch_size, columns=["text"], use_threads=False):
                texts = [str(x or "") for x in batch.column("text").to_pylist()]
                self.row_offset += len(texts)
                encoded = self.tokenizer(texts, max_length=self.context_length, truncation=True, padding=True, return_tensors="pt", add_special_tokens=True)
                ids = encoded["input_ids"].long()
                mask = encoded.get("attention_mask", ids.ne(self.pad_token_id)).long()
                labels = ids.clone()
                labels[mask == 0] = self.pad_token_id
                self.microbatches_seen += 1
                yield {"input_ids": ids, "attention_mask": mask, "labels": labels, "valid_mask": mask.bool()}
            self.shard_index += 1
            self.row_offset = 0


def _synthetic_stream(tokenizer: Any, *, batch_size: int, context_length: int, vocab_size: int, pad_token_id: int, seed: int) -> Iterator[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    while True:
        ids = torch.randint(0, vocab_size, (batch_size, min(context_length, 32)), generator=generator)
        mask = torch.ones_like(ids)
        yield {"input_ids": ids, "attention_mask": mask, "labels": ids.clone(), "valid_mask": mask.bool()}


def _cosine_lr(step: int, config: Stage4Config) -> float:
    if step < config.warmup_steps:
        return config.max_lr * max(0.0, float(step + 1) / max(1, config.warmup_steps))
    progress = min(1.0, max(0.0, float(step - config.warmup_steps) / max(1, config.scheduler_total_steps - config.warmup_steps)))
    return config.min_lr + 0.5 * (config.max_lr - config.min_lr) * (1.0 + math.cos(math.pi * progress))


def _set_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _optimizer(model: torch.nn.Module, config: Stage4Config) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": DEFAULT_ADAMW_WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}], lr=config.max_lr, betas=DEFAULT_ADAMW_BETAS, eps=DEFAULT_ADAMW_EPS, amsgrad=DEFAULT_ADAMW_AMSGRAD)
    router_names = [name for name, _ in model.named_parameters() if ".write_routers." in name or ".read_routers." in name]
    return optimizer, {"groups": [{"name": "decay", "parameter_count": len(decay)}, {"name": "no_decay", "parameter_count": len(no_decay)}], "router_parameters": router_names, "router_parameters_in_optimizer": len(router_names) == 12}


def _checkpoint(model: torch.nn.Module, tokenizer: Any, optimizer: torch.optim.Optimizer, scheduler_step: int, config: Stage4Config, rank: int, state: dict[str, Any]) -> Path | None:
    if rank != 0:
        return None
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"checkpoint-{scheduler_step:06d}"
    temporary = output / f".checkpoint-{scheduler_step:06d}.staging"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": {"type": "cosine_warmup", "step": scheduler_step, "max_lr": config.max_lr, "min_lr": config.min_lr, "warmup_steps": config.warmup_steps, "total_steps": config.scheduler_total_steps}, "optimizer_step": scheduler_step, "configuration": asdict(config), "manifest": state.get("manifest", []), "data_cursors_by_rank": state.get("data_cursors_by_rank", {}), "rng_state": torch.get_rng_state(), "rng_states_by_rank": state.get("rng_states_by_rank", {}), "checkpoint_contract": "model_config_tokenizer_optimizer_scheduler_step_data_cursors_rng_manifest"}, temporary / "training_state.pt")
    checkpoint_metadata = {"architecture_contract": MODEL_ARCHITECTURE_CONTRACT, "optimizer_step": scheduler_step, "router_parameters_in_optimizer": True, "memory_slots": MEMORY_SLOT_COUNT, "logical_to_physical": list(LOGICAL_TO_PHYSICAL)}
    (temporary / "mesh_checkpoint_metadata.json").write_text(json.dumps(checkpoint_metadata, indent=2) + "\n", encoding="utf-8")
    data_manifest = {"architecture_contract": MODEL_ARCHITECTURE_CONTRACT, "optimizer_step": scheduler_step, "data_shards": state.get("manifest", []), "data_cursors_by_rank": state.get("data_cursors_by_rank", {})}
    (temporary / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2, default=str) + "\n", encoding="utf-8")
    required = ["config.json", "training_state.pt", "mesh_checkpoint_metadata.json", "data_manifest.json"]
    required += [path.name for path in temporary.iterdir() if path.name.startswith("model") and path.is_file()]
    tokenizer_files = [name for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model", "vocab.json", "merges.txt") if (temporary / name).is_file()]
    if not tokenizer_files:
        raise RuntimeError("checkpoint tokenizer save produced no tokenizer files")
    required += tokenizer_files
    required += ["checkpoint_manifest.json", "checkpoint_complete.json"]
    manifest = {"status": "complete", "architecture_contract": MODEL_ARCHITECTURE_CONTRACT, "optimizer_step": scheduler_step, "files": sorted(set(required))}
    (temporary / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    marker = dict(manifest)
    marker["complete_marker"] = True
    marker["required_contract"] = "model_config_tokenizer_optimizer_scheduler_step_data_cursors_rng_manifest"
    (temporary / "checkpoint_complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    checkpoints = sorted(output.glob("checkpoint-*"))
    for stale in checkpoints[:-config.checkpoint_retention]:
        shutil.rmtree(stale, ignore_errors=True)
    return destination


def _load_checkpoint_state(path: Path) -> dict[str, Any]:
    _validate_checkpoint_complete(path)
    return torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)


def _validate_checkpoint_complete(path: Path) -> None:
    marker_path = path / "checkpoint_complete.json"
    manifest_path = path / "checkpoint_manifest.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"checkpoint is incomplete: missing complete/manifest marker under {path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete" or marker.get("complete_marker") is not True:
        raise ValueError("checkpoint complete marker is not valid")
    if manifest.get("status") != "complete" or manifest.get("architecture_contract") != MODEL_ARCHITECTURE_CONTRACT:
        raise ValueError("checkpoint manifest is not complete or architecture-matched")
    missing = [name for name in manifest.get("files", []) if not (path / name).is_file()]
    if missing:
        raise ValueError(f"checkpoint manifest lists missing files: {missing}")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    required_state = ("optimizer", "scheduler", "optimizer_step", "data_cursors_by_rank", "rng_state", "rng_states_by_rank", "manifest", "checkpoint_contract")
    missing_state = [key for key in required_state if key not in state]
    if missing_state:
        raise ValueError(f"checkpoint training_state missing fields: {missing_state}")


def run_training(config: Stage4Config) -> dict[str, Any]:
    rank, world_size, device = _dist_setup(config)
    _seed(config.seed, rank)
    report: dict[str, Any] = {"status": "FAIL", "gate": config.gate, "configuration": asdict(config), "architecture_contract": MODEL_ARCHITECTURE_CONTRACT, "world_size": world_size, "rank": rank, "device": str(device), "checks": [], "hard_failures": []}
    try:
        model_path = config.resume_from or config.model_path
        if model_path is None:
            raise ValueError("Stage 4 requires --model-path or --resume-from")
        if config.resume_from is not None:
            _validate_checkpoint_complete(config.resume_from)
        register_auto_class()
        from transformers import AutoTokenizer
        model = RecursiveLlamaForCausalLM.from_pretrained(model_path, local_files_only=True)
        tokenizer_path = config.tokenizer_path or model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.to(device)
        model.model.routing_stats_mode = True
        optimizer, optimizer_group_audit = _optimizer(model, config)
        resume_state = _load_checkpoint_state(config.resume_from) if config.resume_from else None
        optimizer_step = int(resume_state.get("optimizer_step", 0)) if resume_state else 0
        if config.gate == "E":
            config.max_optimizer_steps = optimizer_step + 2
        if resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
            saved_rng = resume_state.get("rng_states_by_rank", {}).get(str(rank))
            if saved_rng is None and rank == 0:
                saved_rng = resume_state.get("rng_state")
            if saved_rng is not None:
                torch.set_rng_state(saved_rng)
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" and world_size > 1 else None, find_unused_parameters=False) if world_size > 1 else model
        manifest = _manifest(config.data_dir) if config.gate != "A" else []
        stream_obj = DistributedParquetStream(manifest, tokenizer, rank=rank, world_size=world_size, batch_size=config.micro_batch_size, context_length=config.context_length, pad_token_id=int(tokenizer.pad_token_id), seed=config.seed) if manifest else None
        if resume_state and stream_obj is not None:
            stream_obj.restore_cursor(resume_state.get("data_cursors_by_rank", {}).get(str(rank)))
        stream: Iterator[dict[str, torch.Tensor]] = iter(stream_obj) if stream_obj is not None else _synthetic_stream(tokenizer, batch_size=config.micro_batch_size, context_length=config.context_length, vocab_size=int(model.config.vocab_size), pad_token_id=int(tokenizer.pad_token_id), seed=config.seed + rank)
        metrics: list[dict[str, Any]] = []
        last_checkpoint: str | None = None
        while optimizer_step < config.max_optimizer_steps:
            optimizer.zero_grad(set_to_none=True)
            total_tokens = torch.zeros((), dtype=torch.float64, device=device)
            total_loss = torch.zeros((), dtype=torch.float64, device=device)
            last_batch = None
            for micro in range(config.gradient_accumulation_steps):
                if config.max_microbatches is not None and stream_obj is not None and stream_obj.microbatches_seen >= config.max_microbatches:
                    break
                batch = next(stream)
                last_batch = batch
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                valid = batch["valid_mask"].to(device)[:, 1:].bool()
                amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
                sync = contextlib.nullcontext() if not hasattr(ddp_model, "no_sync") or micro == config.gradient_accumulation_steps - 1 else ddp_model.no_sync()
                with sync, amp:
                    out = ddp_model(input_ids=ids, attention_mask=mask, use_cache=False)
                    logits = out.logits
                    token_losses = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), reduction="none").reshape_as(valid)
                    loss_sum = token_losses.masked_select(valid).sum()
                    token_count = valid.sum().to(torch.float64)
                    (loss_sum / config.gradient_accumulation_steps).backward()
                total_loss += loss_sum.detach().double()
                total_tokens += token_count
            if last_batch is None:
                raise RuntimeError("no batch available for accumulation window")
            global_tokens = total_tokens.clone()
            if world_size > 1:
                dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM)
            scale = token_weighted_gradient_scale(world_size=world_size, global_window_tokens=int(global_tokens.item()), gradient_accumulation_steps=config.gradient_accumulation_steps)
            for parameter in ddp_model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0, error_if_nonfinite=False)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("nonfinite gradient norm")
            optimizer_step += 1
            _set_lr(optimizer, _cosine_lr(optimizer_step - 1, config))
            optimizer.step()
            routing_owner = ddp_model.module.model if hasattr(ddp_model, "module") else ddp_model.model
            router_stats = routing_owner.last_routing_stats
            routing_audit_due = optimizer_step == 1 or optimizer_step % 10 == 0 or optimizer_step == config.max_optimizer_steps
            routing_audit_ok = True
            routing_audit_error = None
            if routing_audit_due:
                if set(router_stats) != {"write_pre", "read_pre", "write_0", "read_0", "write_1", "read_1"}:
                    routing_audit_ok = False
                    routing_audit_error = "six router statistics were not produced"
                else:
                    for router_name, router_stat in router_stats.items():
                        probabilities = torch.tensor(router_stat.get("slot_probabilities", []), dtype=torch.float64)
                        entropy = float(router_stat.get("mean_entropy", float("nan")))
                        if probabilities.numel() != MEMORY_SLOT_COUNT or not torch.isfinite(probabilities).all() or not math.isfinite(entropy) or not math.isclose(float(probabilities.sum()), 1.0, rel_tol=1e-4, abs_tol=1e-4):
                            routing_audit_ok = False
                            routing_audit_error = f"nonfinite or invalid routing statistics for {router_name}"
                            break
                        if float(probabilities.max()) >= 0.9999 and entropy <= 1e-3:
                            routing_audit_ok = False
                            routing_audit_error = f"router collapse detected for {router_name}"
                            break
                if not routing_audit_ok:
                    raise RuntimeError(routing_audit_error or "routing audit failed")
            local_report = {"optimizer_step": optimizer_step, "loss": float(total_loss.item() / max(1, total_tokens.item())), "local_valid_tokens": int(total_tokens.item()), "global_valid_tokens": int(global_tokens.item()), "learning_rate": float(optimizer.param_groups[0]["lr"]), "grad_norm": float(grad_norm.item()), "router_parameters_in_optimizer": bool(optimizer_group_audit["router_parameters_in_optimizer"]), "routing_stats": {"memory_slots": MEMORY_SLOT_COUNT, "loss_auxiliary": False, "audit_due": routing_audit_due, "audit_passed": routing_audit_ok if routing_audit_due else None, "routers": router_stats}}
            metrics.append(local_report)
            if optimizer_step % config.save_every == 0 or optimizer_step == config.max_optimizer_steps:
                cursors = {str(rank): stream_obj.cursor()} if stream_obj is not None else {str(rank): {"synthetic": True}}
                if world_size > 1:
                    gathered: list[Any] = [None for _ in range(world_size)]
                    dist.all_gather_object(gathered, cursors)
                    cursors = {str(i): value.get(str(i), value) for i, value in enumerate(gathered)}
                rng_states: dict[str, Any] = {str(rank): torch.get_rng_state()}
                if world_size > 1:
                    gathered_rng: list[Any] = [None for _ in range(world_size)]
                    dist.all_gather_object(gathered_rng, torch.get_rng_state())
                    rng_states = {str(i): value for i, value in enumerate(gathered_rng)}
                last_checkpoint_path = _checkpoint(ddp_model, tokenizer, optimizer, optimizer_step, config, rank, {"manifest": [str(p) for p in manifest], "data_cursors_by_rank": cursors, "rng_states_by_rank": rng_states})
                if last_checkpoint_path is not None:
                    last_checkpoint = str(last_checkpoint_path)
            if config.gate in {"D", "E"} and optimizer_step >= config.max_optimizer_steps:
                break
        if config.gate == "FORMAL" and optimizer_step != DEFAULT_FORMAL_OPTIMIZER_STEPS:
            raise RuntimeError(f"FORMAL stopped at {optimizer_step}, expected {DEFAULT_FORMAL_OPTIMIZER_STEPS}")
        report.update({"status": "PASS", "configuration": asdict(config), "optimizer_steps": optimizer_step, "formal_optimizer_steps": DEFAULT_FORMAL_OPTIMIZER_STEPS, "warmup_steps": DEFAULT_FORMAL_WARMUP_STEPS, "metrics": metrics, "manifest": [str(p) for p in manifest], "data_cursors_by_rank": {str(rank): stream_obj.cursor() if stream_obj is not None else {"synthetic": True}}, "optimizer_group_audit": optimizer_group_audit, "checkpoint_contract": "model_config_tokenizer_optimizer_scheduler_step_data_cursors_rng_manifest", "checkpoint_retention": config.checkpoint_retention, "final_checkpoint": last_checkpoint, "logical_to_physical": list(LOGICAL_TO_PHYSICAL), "memory_slots": MEMORY_SLOT_COUNT, "use_cache": False})
    except Exception as exc:
        report["hard_failures"].append({"error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if rank == 0:
        report_path = config.report_path or (config.output_dir / f"stage4_gate_{config.gate}_audit.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        report = run_training(config)
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("status") == "PASS" else 1
    except Exception as exc:
        print(f"[result] status=FAIL error={exc!r}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
