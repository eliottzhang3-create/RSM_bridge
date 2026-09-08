#!/usr/bin/env python3
"""DDP trainer and 8-GPU smoke gates for ReasonAQA + MeSH audio."""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_5_10x2_5_mesh_mellow.data import ReasonAQADataset, collate_reasonaqa  # noqa: E402
from audio_5_10x2_5_mesh_mellow.model import AudioMeshConfig, AudioMeshModel, _load_mellow_wrapper, write_config  # noqa: E402
from recursive_model_5_10x2_5_mesh import RecursiveLlamaForCausalLM, register_auto_class  # noqa: E402


DEFAULT_MESH = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_resume_000500_nonfatal_router_20260907_115533/checkpoint-009244"
DEFAULT_HTSAT = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt"
DEFAULT_MELLOW = "/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", choices=("STAGE5", "STAGE7", "FORMAL"), default="STAGE5")
    p.add_argument("--model-path", type=Path, default=Path(DEFAULT_MESH))
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--tokenizer-path", type=Path)
    p.add_argument("--htsat-checkpoint", type=Path, default=Path(DEFAULT_HTSAT))
    p.add_argument("--mellow-root", type=Path, default=Path(DEFAULT_MELLOW))
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--report-path", type=Path)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers per DDP rank; 0 keeps loading in the rank process")
    return p.parse_args(argv)


def _init_dist(args: argparse.Namespace) -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return rank, world, device


def _seed(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _load_model(args: argparse.Namespace, device: torch.device) -> tuple[AudioMeshModel, Any]:
    register_auto_class()
    from transformers import AutoTokenizer
    model_path = args.resume_from / "mesh_model" if args.resume_from else args.model_path
    tokenizer_path = args.tokenizer_path or (args.resume_from / "tokenizer" if args.resume_from else model_path)
    mesh = RecursiveLlamaForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper, htsat, provenance = _load_mellow_wrapper(args.mellow_root, args.htsat_checkpoint, device)
    model = AudioMeshModel(mesh.to(device), tokenizer, wrapper, htsat, AudioMeshConfig())
    if args.resume_from:
        audio_state = torch.load(args.resume_from / "audio_bridge.pt", map_location=device, weights_only=False)
        model.bridge.load_state_dict(audio_state["bridge"])
        c2l = getattr(model.htsat_wrapper, "c2l", None)
        if c2l is not None and audio_state.get("c2l"):
            c2l.load_state_dict(audio_state["c2l"])
    model._audio_provenance = provenance
    return model.to(device), tokenizer


def _trainable_state(model: AudioMeshModel) -> dict[str, Any]:
    c2l = getattr(model.htsat_wrapper, "c2l", None)
    return {"bridge": model.bridge.state_dict(), "c2l": c2l.state_dict() if c2l is not None else {}}


def _save_checkpoint(path: Path, model: AudioMeshModel, tokenizer: Any, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR, step: int, epoch: int, batch_in_epoch: int, args: argparse.Namespace, manifest_hash: str, rng_states_by_rank: dict[str, Any], total_steps: int) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.mkdir(parents=True, exist_ok=True)
    model.mesh_model.save_pretrained(temporary / "mesh_model", safe_serialization=False)
    tokenizer.save_pretrained(temporary / "tokenizer")
    torch.save(_trainable_state(model), temporary / "audio_bridge.pt")
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "global_step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch, "rng_states_by_rank": rng_states_by_rank}, temporary / "training_state.pt")
    config = {"architecture_contract": "logical_30_physical_20_5_10x2_5_mesh_audio_mellow", "manifest_sha256": manifest_hash, "htsat_checkpoint": str(args.htsat_checkpoint), "mesh_model_path": str(args.model_path), "epochs": args.epochs, "max_lr": args.max_lr, "min_lr": args.min_lr, "warmup_steps": args.warmup_steps, "total_steps": total_steps, "global_step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch, "world_size": args.world_size, "micro_batch_size": args.micro_batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps}
    (temporary / "audio_mesh_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (temporary / "checkpoint_complete.json").write_text(json.dumps({"status": "complete", "global_step": step, "required": ["mesh_model", "tokenizer", "audio_bridge.pt", "training_state.pt", "audio_mesh_config.json"]}, indent=2) + "\n", encoding="utf-8")
    if path.exists():
        import shutil
        shutil.rmtree(path)
    temporary.replace(path)


def _load_training_state(path: Path, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR) -> dict[str, Any]:
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state


def _audit_saved_checkpoint(path: Path) -> dict[str, Any]:
    required = ("mesh_model/config.json", "audio_bridge.pt", "training_state.pt", "audio_mesh_config.json", "checkpoint_complete.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"composite checkpoint missing files: {missing}")
    marker = json.loads((path / "checkpoint_complete.json").read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise RuntimeError("composite checkpoint completion marker is invalid")
    training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    for key in ("optimizer", "scheduler", "global_step", "epoch", "batch_in_epoch", "rng_states_by_rank"):
        if key not in training:
            raise RuntimeError(f"composite checkpoint training_state missing {key}")
    audio = torch.load(path / "audio_bridge.pt", map_location="cpu", weights_only=False)
    if "bridge" not in audio or "c2l" not in audio:
        raise RuntimeError("composite checkpoint missing bridge/c2l state")
    if not training["rng_states_by_rank"]:
        raise RuntimeError("composite checkpoint has no per-rank RNG states")
    return {"passed": True, "path": str(path), "global_step": int(training["global_step"]), "epoch": int(training["epoch"]), "batch_in_epoch": int(training["batch_in_epoch"]), "rng_ranks": sorted(training["rng_states_by_rank"]), "required_files": list(required)}


def _router_stats(model: AudioMeshModel) -> dict[str, Any]:
    owner = model.mesh_model.model
    return getattr(owner, "last_routing_stats", {})


def _make_scheduler(optimizer: torch.optim.Optimizer, *, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    if max_lr <= 0 or min_lr < 0 or min_lr > max_lr:
        raise ValueError("learning rates must satisfy 0 <= min_lr <= max_lr and max_lr > 0")
    def scale(step: int) -> float:
        if step < warmup_steps:
            return min(1.0, float(step + 1) / max(1, warmup_steps))
        progress = min(1.0, max(0.0, float(step - warmup_steps) / max(1, total_steps - warmup_steps)))
        ratio = min_lr / max_lr
        return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state().cpu(),
        "cuda": torch.cuda.get_rng_state(device).cpu(),
        "python": random.getstate(),
    }


def _restore_rng_state(state: dict[str, Any] | None, device: torch.device) -> None:
    if not state:
        return
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"], device=device)
    if state.get("python") is not None:
        random.setstate(state["python"])


def _gather_rng_states(rank: int, world: int, device: torch.device) -> dict[str, Any]:
    local = _rng_state(device)
    gathered: list[Any] = [None for _ in range(world)]
    if world > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered[0] = local
    return {str(index): value for index, value in enumerate(gathered)}


def _skip_batches(data_iter: Any, count: int) -> None:
    for _ in range(max(0, int(count))):
        next(data_iter)


def _actual_resume_audit(path: Path, args: argparse.Namespace, batch_cpu: dict[str, Any], device: torch.device, expected_step: int, expected_lr: float, rank: int) -> dict[str, Any]:
    """Reload the complete composite checkpoint and execute one real batch."""
    saved_rng = _rng_state(device)
    reload_args = copy.copy(args)
    reload_args.resume_from = path
    reload_model, reload_tokenizer = _load_model(reload_args, device)
    try:
        saved_config = json.loads((path / "audio_mesh_config.json").read_text(encoding="utf-8"))
        optimizer = torch.optim.AdamW([p for p in reload_model.parameters() if p.requires_grad], lr=float(saved_config["max_lr"]), betas=(0.9, 0.95), weight_decay=0.1)
        scheduler = _make_scheduler(optimizer, max_lr=float(saved_config["max_lr"]), min_lr=float(saved_config.get("min_lr", 0.0)), warmup_steps=int(saved_config["warmup_steps"]), total_steps=max(1, int(saved_config.get("total_steps", expected_step))))
        state = _load_training_state(path, optimizer, scheduler)
        if int(state["global_step"]) != int(expected_step):
            raise RuntimeError(f"reloaded global step mismatch: {state['global_step']} != {expected_step}")
        loaded_lr = float(optimizer.param_groups[0]["lr"])
        if not math.isclose(loaded_lr, float(expected_lr), rel_tol=1e-6, abs_tol=1e-10):
            raise RuntimeError(f"reloaded learning-rate mismatch: {loaded_lr} != {expected_lr}")
        rank_state = state["rng_states_by_rank"].get(str(rank)) or state["rng_states_by_rank"].get("0")
        _restore_rng_state(rank_state, device)
        reload_model.train()
        moved = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch_cpu.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = reload_model(**{key: value for key, value in moved.items() if key not in {"row_indices", "audio2_reused"}})
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("reloaded checkpoint produced a nonfinite loss")
        labels = reload_model.last_labels
        prefix_length = int(reload_model.last_prefix_length or 0)
        prompt_length = int(moved["prompt_ids"].shape[1])
        if labels is None or bool((labels[:, :prefix_length + prompt_length] != -100).any()):
            raise RuntimeError("reloaded checkpoint violated answer-only label mask")
        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(reload_model.parameters(), 0.5, error_if_nonfinite=True)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("reloaded checkpoint produced a nonfinite gradient")
        return {"passed": True, "global_step": int(state["global_step"]), "epoch": int(state["epoch"]), "batch_in_epoch": int(state["batch_in_epoch"]), "learning_rate": loaded_lr, "loss": float(output.loss.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu()), "optimizer_state_loaded": bool(optimizer.state_dict()["state"]), "scheduler_last_epoch": int(scheduler.last_epoch), "forward_backward": True, "answer_only_labels": True}
    finally:
        del reload_model
        gc.collect()
        torch.cuda.empty_cache()
        _restore_rng_state(saved_rng, device)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank, world, device = _init_dist(args)
    _seed(args.seed, rank)
    report: dict[str, Any] = {"stage": f"{args.gate.lower()}_audio_5_10x2_5_mesh_mellow", "status": "FAIL", "configuration": vars(args), "rank": rank, "world_size": world, "checks": [], "warnings": [], "hard_failures": []}
    try:
        if world != args.world_size:
            raise RuntimeError(f"world size mismatch: launcher={world} requested={args.world_size}")
        model, tokenizer = _load_model(args, device)
        dataset = ReasonAQADataset(args.train_manifest, tokenizer)
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=False, drop_last=True)
        loader = DataLoader(dataset, batch_size=args.micro_batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=lambda rows: collate_reasonaqa(rows, tokenizer))
        steps_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
        formal_steps = steps_epoch * args.epochs
        max_steps = args.max_steps or (2 if args.gate == "STAGE5" else 10 if args.gate == "STAGE7" else formal_steps)
        total_steps = formal_steps if args.gate == "FORMAL" else max_steps
        required_warmup = math.ceil(total_steps * 0.05)
        if args.gate == "FORMAL" and args.warmup_steps is not None and args.warmup_steps != required_warmup:
            raise ValueError(f"FORMAL warmup must equal ceil(actual_total_steps*0.05)={required_warmup}, got {args.warmup_steps}")
        args.warmup_steps = required_warmup
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.max_lr, betas=(0.9, 0.95), weight_decay=0.1)
        scheduler = _make_scheduler(optimizer, max_lr=args.max_lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps, total_steps=total_steps)
        start_step, start_epoch, start_batch_in_epoch = 0, 0, 0
        if args.resume_from:
            state = _load_training_state(args.resume_from, optimizer, scheduler)
            start_step = int(state["global_step"])
            start_epoch = int(state.get("epoch", 0))
            start_batch_in_epoch = int(state.get("batch_in_epoch", 0))
            rank_rng = state.get("rng_states_by_rank", {}).get(str(rank)) or state.get("rng_states_by_rank", {}).get("0")
            _restore_rng_state(rank_rng, device)
        ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False) if world > 1 else model
        metrics: list[dict[str, Any]] = []
        optimizer_step = start_step
        epoch = start_epoch
        batch_in_epoch = start_batch_in_epoch
        if batch_in_epoch >= len(loader):
            epoch += batch_in_epoch // len(loader)
            batch_in_epoch = batch_in_epoch % len(loader)
        while optimizer_step < max_steps:
            sampler.set_epoch(epoch)
            data_iter = iter(loader)
            _skip_batches(data_iter, batch_in_epoch)
            for _ in range(batch_in_epoch, steps_epoch):
                if optimizer_step >= max_steps:
                    break
                step_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                optimizer.zero_grad(set_to_none=True)
                for micro in range(args.gradient_accumulation_steps):
                    batch = next(data_iter)
                    if micro == args.gradient_accumulation_steps - 1:
                        batch_cpu = {key: (value.detach().cpu().clone() if torch.is_tensor(value) else value) for key, value in batch.items()}
                    batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
                    sync = contextlib.nullcontext() if not hasattr(ddp, "no_sync") or micro == args.gradient_accumulation_steps - 1 else ddp.no_sync()
                    with sync:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            output = ddp(**{k: v for k, v in batch.items() if k not in {"row_indices", "audio2_reused"}})
                        if output.loss is None or not torch.isfinite(output.loss):
                            raise RuntimeError("nonfinite audio MeSH loss")
                        (output.loss / args.gradient_accumulation_steps).backward()
                owner = ddp.module if hasattr(ddp, "module") else ddp
                grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 0.5, error_if_nonfinite=True)
                optimizer_step += 1
                optimizer.step()
                scheduler.step()
                elapsed = max(time.perf_counter() - step_started, 1e-9)
                global_samples = int(args.micro_batch_size * world * args.gradient_accumulation_steps)
                answer_tokens = torch.tensor(int(batch["answer_attention_mask"].sum().item()), dtype=torch.long, device=device)
                if world > 1:
                    dist.all_reduce(answer_tokens, op=dist.ReduceOp.SUM)
                item = {"step": optimizer_step, "total_steps": max_steps, "progress_percent": 100.0 * optimizer_step / max(1, max_steps), "epoch": epoch, "batch_in_epoch": batch_in_epoch, "steps_per_epoch": steps_epoch, "loss": float(output.loss.detach().cpu()), "lr": float(optimizer.param_groups[0]["lr"]), "grad_norm": float(grad_norm), "effective_answer_tokens": int(answer_tokens.item()), "step_time_seconds": elapsed, "samples_per_second": global_samples / elapsed, "audio_seconds_per_second": global_samples * 20.0 / elapsed, "gpu_memory_allocated_gib": float(torch.cuda.memory_allocated(device) / 1024**3), "gpu_memory_reserved_gib": float(torch.cuda.memory_reserved(device) / 1024**3), "gpu_memory_max_allocated_gib": float(torch.cuda.max_memory_allocated(device) / 1024**3), "gpu_memory_max_reserved_gib": float(torch.cuda.max_memory_reserved(device) / 1024**3), "router_stats": _router_stats(owner)}
                metrics.append(item)
                batch_in_epoch += args.gradient_accumulation_steps
                if rank == 0 and (optimizer_step % 10 == 0 or optimizer_step == max_steps):
                    memory = torch.cuda.memory_allocated(device) / 1024**3
                    print(f"[audio-train] step={optimizer_step}/{max_steps} progress={item['progress_percent']:.2f}% epoch={epoch + 1}/{args.epochs if args.gate == 'FORMAL' else '?'} batch={batch_in_epoch}/{len(loader)} loss={item['loss']:.6f} lr={item['lr']:.8g} step_s={item['step_time_seconds']:.3f} samples/s={item['samples_per_second']:.2f} audio_s/s={item['audio_seconds_per_second']:.2f} answer_tokens={item['effective_answer_tokens']} gpu_alloc_gib={item['gpu_memory_allocated_gib']:.3f} gpu_reserved_gib={item['gpu_memory_reserved_gib']:.3f} gpu_max_alloc_gib={item['gpu_memory_max_allocated_gib']:.3f} gpu_max_reserved_gib={item['gpu_memory_max_reserved_gib']:.3f} router_stats={item['router_stats']}", flush=True)
                save_due = (args.gate == "FORMAL" and (optimizer_step % max(1, args.save_every) == 0 or optimizer_step == max_steps)) or (args.gate == "STAGE7" and optimizer_step >= 10)
                if save_due:
                    out = args.output_dir / f"checkpoint-{optimizer_step:06d}"
                    rng_states = _gather_rng_states(rank, world, device)
                    if rank == 0:
                        _save_checkpoint(out, owner, tokenizer, optimizer, scheduler, optimizer_step, epoch, batch_in_epoch, args, hashlib.sha256(args.train_manifest.read_bytes()).hexdigest(), rng_states, total_steps)
                    if world > 1:
                        dist.barrier()
                    if rank == 0:
                        report.setdefault("checkpoints", []).append(str(out))
                        if args.gate == "STAGE7":
                            artifact_audit = _audit_saved_checkpoint(out)
                            resume_audit = _actual_resume_audit(out, args, batch_cpu, device, optimizer_step, float(optimizer.param_groups[0]["lr"]), rank)
                            report["checkpoint_reload_audit"] = {"artifact": artifact_audit, "actual_resume": resume_audit}
                    if world > 1:
                        dist.barrier()
                    if optimizer_step >= max_steps:
                        break
                if optimizer_step >= max_steps:
                    break
            if batch_in_epoch >= len(loader):
                epoch += batch_in_epoch // len(loader)
                batch_in_epoch = batch_in_epoch % len(loader)
        report.update({"status": "PASS", "optimizer_steps": optimizer_step, "steps_per_epoch": steps_epoch, "total_formal_steps": formal_steps, "warmup_steps": args.warmup_steps, "metrics": metrics if rank == 0 else [], "ddp_broadcast_buffers": False, "router_policy": "warning_only", "model_trainable_audit": (ddp.module if hasattr(ddp, "module") else ddp).trainable_parameter_audit(), "resume_position": {"epoch": epoch, "batch_in_epoch": batch_in_epoch}, "checkpoints": report.get("checkpoints", [])})
    except Exception as exc:
        report["hard_failures"].append({"error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.report_path or args.output_dir / f"{args.gate.lower()}_audit.json"
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({"stage": report["stage"], "status": report["status"], "summary": {"steps": report.get("optimizer_steps"), "hard_failures": len(report.get("hard_failures", []))}, "report": str(args.report_path or args.output_dir)}, default=str))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
