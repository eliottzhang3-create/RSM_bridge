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
import shutil
import tempfile
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
from audio_5_10x2_5_mesh_mellow.model import (  # noqa: E402
    AUDIO_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    ARCHITECTURE_CONTRACT,
    MAPPER_CONTRACT,
    MESH_HIDDEN_SIZE,
    AudioMeshConfig,
    AudioMeshModel,
    _load_mellow_wrapper,
    write_config,
)
from recursive_model_5_10x2_5_mesh import RecursiveLlamaForCausalLM, register_auto_class  # noqa: E402


DEFAULT_MESH = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244"
DEFAULT_HTSAT = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt"
DEFAULT_MELLOW = "/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", choices=("STAGE5", "STAGE7", "FORMAL"), default="STAGE5")
    # ``--mesh-checkpoint`` is the public name used by the repository
    # submission wrappers and operator commands; keep ``--model-path`` as a
    # backward-compatible alias for older invocations.
    p.add_argument("--model-path", "--mesh-checkpoint", dest="model_path", type=Path, default=Path(DEFAULT_MESH))
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
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--checkpoint-retention", type=int, default=4)
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
        if not isinstance(audio_state.get("bridge"), dict) or not audio_state["bridge"]:
            raise RuntimeError("resume checkpoint has no non-empty bridge state")
        if not isinstance(audio_state.get("c2l"), dict) or not audio_state["c2l"]:
            raise RuntimeError("resume checkpoint has no non-empty c2l state")
        model.bridge.load_state_dict(audio_state["bridge"], strict=True)
        c2l = getattr(model.htsat_wrapper, "c2l", None)
        if c2l is None:
            raise RuntimeError("resume checkpoint requires wrapper.c2l")
        c2l.load_state_dict(audio_state["c2l"], strict=True)
    model._audio_provenance = provenance
    return model.to(device), tokenizer


def _trainable_state(model: AudioMeshModel) -> dict[str, Any]:
    c2l = getattr(model.htsat_wrapper, "c2l", None)
    return {"bridge": model.bridge.state_dict(), "c2l": c2l.state_dict() if c2l is not None else {}}


def _save_checkpoint(path: Path, model: AudioMeshModel, tokenizer: Any, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR, step: int, epoch: int, batch_in_epoch: int, args: argparse.Namespace, manifest_hash: str, rng_states_by_rank: dict[str, Any], total_steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {path}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)))
    published = False
    try:
        model.mesh_model.save_pretrained(temporary / "mesh_model", safe_serialization=False)
        tokenizer.save_pretrained(temporary / "tokenizer")
        torch.save(_trainable_state(model), temporary / "audio_bridge.pt")
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "global_step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch, "rng_states_by_rank": rng_states_by_rank}, temporary / "training_state.pt")
        provenance = getattr(model, "_audio_provenance", {})
        if int(model.mesh_model.config.hidden_size) != MESH_HIDDEN_SIZE:
            raise RuntimeError("refusing to save checkpoint with non-contract MeSH hidden size")
        if model.last_audio_tokens_per_clip is not None and tuple(model.last_audio_tokens_per_clip) != (AUDIO_TOKENS_PER_CLIP, AUDIO_TOKENS_PER_CLIP):
            raise RuntimeError("refusing to save checkpoint with non-contract audio token count")
        config = {
            "architecture_contract": ARCHITECTURE_CONTRACT,
            "mapper_contract": MAPPER_CONTRACT,
            "mapper_initialization": "random_c2l_and_xavier_projection",
            "mesh_hidden_size": MESH_HIDDEN_SIZE,
            "audio_tokens_per_clip": AUDIO_TOKENS_PER_CLIP,
            "audio_prefix_tokens_with_separators": AUDIO_PREFIX_TOKENS,
            "manifest_sha256": manifest_hash,
            "htsat_checkpoint": str(args.htsat_checkpoint.resolve()),
            "mellow_root": str(args.mellow_root.resolve()),
            "mellow_provenance": provenance,
            "mesh_model_path": str(args.model_path),
            "epochs": args.epochs,
            "max_lr": args.max_lr,
            "min_lr": args.min_lr,
            "warmup_steps": args.warmup_steps,
            "total_steps": total_steps,
            "global_step": step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "world_size": args.world_size,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "save_every": args.save_every,
            "checkpoint_retention": args.checkpoint_retention,
        }
        (temporary / "audio_mesh_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (temporary / "checkpoint_complete.json").write_text(json.dumps({"status": "complete", "global_step": step, "required": ["mesh_model", "tokenizer", "audio_bridge.pt", "training_state.pt", "audio_mesh_config.json"]}, indent=2) + "\n", encoding="utf-8")
        required = (temporary / "mesh_model" / "config.json", temporary / "tokenizer", temporary / "audio_bridge.pt", temporary / "training_state.pt", temporary / "audio_mesh_config.json", temporary / "checkpoint_complete.json")
        if any(not item.exists() for item in required):
            raise RuntimeError("refusing to publish incomplete checkpoint")
        temporary.replace(path)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def _prune_checkpoints(output_dir: Path, retention: int) -> list[str]:
    """Keep only the newest complete checkpoints inside this exact run dir."""
    if retention <= 0:
        raise ValueError("checkpoint_retention must be positive")
    output_resolved = output_dir.resolve()
    # Remove only our own interrupted staging directories; incomplete
    # ``checkpoint-*`` directories are never treated as complete or deleted.
    for temporary in output_dir.glob(".checkpoint-*.tmp"):
        if temporary.is_dir() and temporary.resolve().parent == output_resolved:
            shutil.rmtree(temporary, ignore_errors=True)
    complete: list[tuple[int, Path]] = []
    for candidate in output_dir.glob("checkpoint-*"):
        suffix = candidate.name.removeprefix("checkpoint-")
        if not candidate.is_dir() or not suffix.isdigit():
            continue
        resolved = candidate.resolve()
        if resolved.parent != output_resolved:
            raise RuntimeError(f"refusing to prune checkpoint outside output directory: {resolved}")
        marker = candidate / "checkpoint_complete.json"
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("status") == "complete":
            complete.append((int(suffix), candidate))
    complete.sort(key=lambda item: item[0])
    for _, stale in complete[:-retention]:
        shutil.rmtree(stale)
    return [str(path) for _, path in complete[-retention:]]


def _load_training_state(path: Path, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR) -> dict[str, Any]:
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state


def _validate_resume_artifacts(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"resume checkpoint directory not found: {path}")
    _audit_saved_checkpoint(path)
    marker = json.loads((path / "checkpoint_complete.json").read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise RuntimeError("resume checkpoint marker is not complete")
    config = json.loads((path / "audio_mesh_config.json").read_text(encoding="utf-8"))
    for key in ("architecture_contract", "mapper_contract", "mesh_hidden_size", "audio_tokens_per_clip", "audio_prefix_tokens_with_separators", "manifest_sha256", "htsat_checkpoint", "mellow_root", "mellow_provenance", "world_size", "micro_batch_size", "gradient_accumulation_steps", "epochs", "max_lr", "min_lr", "warmup_steps", "total_steps", "save_every", "checkpoint_retention"):
        if key not in config:
            raise RuntimeError(f"resume checkpoint config missing {key}")
    if config["architecture_contract"] != ARCHITECTURE_CONTRACT or config["mapper_contract"] != MAPPER_CONTRACT:
        raise RuntimeError("resume checkpoint architecture/mapper contract mismatch")
    if int(config["mesh_hidden_size"]) != MESH_HIDDEN_SIZE or int(config["audio_tokens_per_clip"]) != AUDIO_TOKENS_PER_CLIP or int(config["audio_prefix_tokens_with_separators"]) != AUDIO_PREFIX_TOKENS:
        raise RuntimeError("resume checkpoint audio shape contract mismatch")
    audio = torch.load(path / "audio_bridge.pt", map_location="cpu", weights_only=False)
    if not isinstance(audio.get("bridge"), dict) or not audio["bridge"] or not isinstance(audio.get("c2l"), dict) or not audio["c2l"]:
        raise RuntimeError("resume checkpoint must contain non-empty bridge and c2l state")
    return config


def _audit_saved_checkpoint(path: Path) -> dict[str, Any]:
    required = ("mesh_model/config.json", "tokenizer/tokenizer_config.json", "audio_bridge.pt", "training_state.pt", "audio_mesh_config.json", "checkpoint_complete.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"composite checkpoint missing files: {missing}")
    marker = json.loads((path / "checkpoint_complete.json").read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise RuntimeError("composite checkpoint completion marker is invalid")
    training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    config = json.loads((path / "audio_mesh_config.json").read_text(encoding="utf-8"))
    for key in ("optimizer", "scheduler", "global_step", "epoch", "batch_in_epoch", "rng_states_by_rank"):
        if key not in training:
            raise RuntimeError(f"composite checkpoint training_state missing {key}")
    audio = torch.load(path / "audio_bridge.pt", map_location="cpu", weights_only=False)
    if "bridge" not in audio or "c2l" not in audio or not audio["bridge"] or not audio["c2l"]:
        raise RuntimeError("composite checkpoint missing bridge/c2l state")
    if not training["rng_states_by_rank"]:
        raise RuntimeError("composite checkpoint has no per-rank RNG states")
    expected_rng = {str(index) for index in range(int(config.get("world_size", 0)))}
    actual_rng = {str(key) for key in training["rng_states_by_rank"]}
    if actual_rng != expected_rng:
        raise RuntimeError(f"composite checkpoint RNG ranks mismatch: expected={sorted(expected_rng)} actual={sorted(actual_rng)}")
    return {"passed": True, "path": str(path), "global_step": int(training["global_step"]), "epoch": int(training["epoch"]), "batch_in_epoch": int(training["batch_in_epoch"]), "rng_ranks": sorted(actual_rng), "required_files": list(required)}


def _router_stats(model: AudioMeshModel) -> dict[str, Any]:
    owner = model.mesh_model.model
    return getattr(owner, "last_routing_stats", {})


def _mesh_runtime_gradient_audit(model: AudioMeshModel) -> dict[str, Any]:
    """Validate one forward/backward traversed both MeSH middle loops."""
    mesh_owner = model.mesh_model.model
    trace = list(getattr(mesh_owner, "last_forward_trace", []))
    expected_trace = [
        *({"logical_index": i, "physical_index": i} for i in range(5)),
        *({"logical_index": 5 + i, "physical_index": 5 + i} for i in range(10)),
        *({"logical_index": 15 + i, "physical_index": 5 + i} for i in range(10)),
        *({"logical_index": 25 + i, "physical_index": 15 + i} for i in range(5)),
    ]
    trace_ok = trace == expected_trace
    input_refs = list(getattr(mesh_owner, "last_core_input_refs", []))
    output_refs = list(getattr(mesh_owner, "last_core_output_refs", []))
    loop_input_grads = [
        ref.grad is not None and torch.isfinite(ref.grad).all() for ref in input_refs
    ]
    loop_output_grads = [
        ref.grad is not None and torch.isfinite(ref.grad).all() for ref in output_refs
    ]
    router_grads = {
        name: any(p.grad is not None and torch.isfinite(p.grad).all() for p in router.parameters())
        for group, routers in (("write", mesh_owner.write_routers), ("read", mesh_owner.read_routers))
        for name, router in ((f"{group}_{index}", router) for index, router in enumerate(routers))
    }
    core_grads = [
        any(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
        for layer in mesh_owner.layers[5:15]
    ]
    result = {
        "trace_length": len(trace),
        "expected_trace_length": 30,
        "trace_matches_5_10_10_5": trace_ok,
        "loop_input_finite_gradients": loop_input_grads,
        "loop_output_finite_gradients": loop_output_grads,
        "both_middle_loops_have_finite_gradients": len(input_refs) == 2 and len(output_refs) == 2 and all(loop_input_grads) and all(loop_output_grads),
        "router_stats_nonempty": len(_router_stats(model)) == 6,
        "router_stats_names": sorted(_router_stats(model)),
        "router_finite_gradients": router_grads,
        "all_router_gradients_finite": all(router_grads.values()),
        "middle_core_layer_gradients_finite": core_grads,
        "all_middle_core_gradients_finite": all(core_grads),
    }
    if not all((result["trace_matches_5_10_10_5"], result["both_middle_loops_have_finite_gradients"], result["router_stats_nonempty"], result["all_router_gradients_finite"], result["all_middle_core_gradients_finite"])):
        raise RuntimeError(f"MeSH runtime gradient/router/trace audit failed: {result}")
    return result


def _make_scheduler(optimizer: torch.optim.Optimizer, *, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    if max_lr <= 0 or min_lr < 0 or min_lr > max_lr:
        raise ValueError("learning rates must satisfy 0 <= min_lr <= max_lr and max_lr > 0")
    def scale(step: int) -> float:
        if step < warmup_steps:
            return min(1.0, float(step + 1) / max(1, warmup_steps))
        # LambdaLR's ``step`` is zero at construction and increments after an
        # optimizer update.  The LR used by update N is therefore represented
        # by scheduler index N-1; use ``step + 1`` for the contract's update
        # index so the final update really uses min_lr.
        progress = min(1.0, max(0.0, float(step + 1 - warmup_steps) / max(1, total_steps - warmup_steps)))
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
        reload_model.mesh_model.model.routing_stats_mode = True
        reload_model.mesh_model.model.gradient_audit_mode = True
        moved = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch_cpu.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = reload_model(**{key: value for key, value in moved.items() if key not in {"row_indices", "audio2_reused"}})
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("reloaded checkpoint produced a nonfinite loss")
        labels = reload_model.last_labels
        prefix_length = int(reload_model.last_prefix_length or 0)
        text_ids = moved["text_ids"]
        prompt_lengths = moved["prompt_lengths"]
        answer_lengths = moved["answer_lengths"]
        if labels is None:
            raise RuntimeError("reloaded checkpoint did not expose labels")
        for row_index in range(text_ids.shape[0]):
            prompt_length = int(prompt_lengths[row_index].item())
            answer_length = int(answer_lengths[row_index].item())
            answer_start = prefix_length + prompt_length
            answer_end = answer_start + answer_length
            if bool((labels[row_index, :answer_start] != -100).any()) or bool((labels[row_index, answer_end:] != -100).any()):
                raise RuntimeError("reloaded checkpoint violated answer-only label mask")
            if not torch.equal(labels[row_index, answer_start:answer_end], text_ids[row_index, prompt_length:prompt_length + answer_length]):
                raise RuntimeError("reloaded checkpoint answer labels are misaligned with unified text")
        output.loss.backward()
        runtime_audit = _mesh_runtime_gradient_audit(reload_model)
        trainable_audit = reload_model.trainable_parameter_audit()
        if not trainable_audit["training_mode_contract"]:
            raise RuntimeError(f"reloaded checkpoint training mode contract failed: {trainable_audit}")
        bridge_grad = {
            name: p.grad is not None and torch.isfinite(p.grad).all()
            for name, p in reload_model.bridge.named_parameters()
        }
        c2l = getattr(reload_model.htsat_wrapper, "c2l", None)
        c2l_grad = {
            name: p.grad is not None and torch.isfinite(p.grad).all()
            for name, p in c2l.named_parameters()
        } if c2l is not None else {}
        if not bridge_grad or not all(bridge_grad.values()) or not c2l_grad or not all(c2l_grad.values()):
            raise RuntimeError(f"reloaded checkpoint audio gradients failed: bridge={bridge_grad} c2l={c2l_grad}")
        grad_norm = torch.nn.utils.clip_grad_norm_(reload_model.parameters(), 0.5, error_if_nonfinite=True)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("reloaded checkpoint produced a nonfinite gradient")
        return {"passed": True, "global_step": int(state["global_step"]), "epoch": int(state["epoch"]), "batch_in_epoch": int(state["batch_in_epoch"]), "learning_rate": loaded_lr, "loss": float(output.loss.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu()), "optimizer_state_loaded": bool(optimizer.state_dict()["state"]), "scheduler_last_epoch": int(scheduler.last_epoch), "forward_backward": True, "answer_only_labels": True, "training_mode_contract": trainable_audit, "audio_gradients": {"bridge": bridge_grad, "c2l": c2l_grad}, "mesh_runtime_gradient_audit": runtime_audit}
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
        if args.save_every <= 0 or args.checkpoint_retention <= 0:
            raise ValueError("save_every and checkpoint_retention must be positive")
        if args.gate == "FORMAL":
            canonical = {
                "world_size": 8,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "epochs": 3,
                "max_lr": 1e-3,
                "min_lr": 0.0,
                "save_every": 1000,
                "checkpoint_retention": 4,
            }
            for key, expected in canonical.items():
                if getattr(args, key) != expected:
                    raise ValueError(f"FORMAL requires {key}={expected}, got {getattr(args, key)}")
            if args.max_steps is not None:
                raise ValueError("FORMAL does not accept --max-steps; use STAGE5/STAGE7 for bounded smoke")
        elif args.max_steps is not None and args.max_steps <= 0:
            raise ValueError("bounded smoke --max-steps must be positive")
        model, tokenizer = _load_model(args, device)
        model.train()
        if not model.trainable_parameter_audit()["training_mode_contract"]:
            raise RuntimeError(f"audio training mode contract failed: {model.trainable_parameter_audit()}")
        model.mesh_model.model.routing_stats_mode = True
        model.mesh_model.model.gradient_audit_mode = True
        dataset = ReasonAQADataset(args.train_manifest, tokenizer)
        # Shuffle deterministically per epoch; set_epoch(epoch) below changes
        # the permutation while DistributedSampler keeps rank partitions
        # disjoint and equally sized.
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        loader = DataLoader(dataset, batch_size=args.micro_batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=lambda rows: collate_reasonaqa(rows, tokenizer))
        if args.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        # Only complete accumulation windows become optimizer steps.  This
        # keeps every optimizer step at the configured effective batch size;
        # a short tail of micro-batches is intentionally dropped at the epoch
        # boundary and will be reshuffled into the next epoch.
        steps_epoch = len(loader) // args.gradient_accumulation_steps
        dropped_microbatches = len(loader) % args.gradient_accumulation_steps
        if steps_epoch <= 0:
            raise ValueError(f"loader has {len(loader)} batches, fewer than one accumulation window of {args.gradient_accumulation_steps}")
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
            saved_config = _validate_resume_artifacts(args.resume_from)
            # Smoke resume may extend the bounded run (20 -> 22) while
            # retaining the original checkpoint's optimizer/scheduler
            # contract.  FORMAL never permits this; its total schedule must
            # remain the canonical three-epoch value.
            if args.gate != "FORMAL" and int(saved_config["total_steps"]) <= int(max_steps):
                total_steps = int(saved_config["total_steps"])
                args.warmup_steps = int(saved_config["warmup_steps"])
                scheduler = _make_scheduler(
                    optimizer,
                    max_lr=float(saved_config["max_lr"]),
                    min_lr=float(saved_config["min_lr"]),
                    warmup_steps=int(saved_config["warmup_steps"]),
                    total_steps=max(1, int(saved_config["total_steps"])),
                )
            saved_manifest_hash = str(saved_config["manifest_sha256"])
            current_manifest_hash = hashlib.sha256(args.train_manifest.read_bytes()).hexdigest()
            if saved_manifest_hash != current_manifest_hash:
                raise RuntimeError("resume manifest SHA256 mismatch")
            for key in ("world_size", "micro_batch_size", "gradient_accumulation_steps", "epochs", "max_lr", "min_lr", "save_every", "checkpoint_retention"):
                if str(saved_config[key]) != str(getattr(args, key)):
                    raise RuntimeError(f"resume {key} mismatch: saved={saved_config[key]} current={getattr(args, key)}")
            if int(saved_config["warmup_steps"]) != int(args.warmup_steps):
                raise RuntimeError(f"resume warmup_steps mismatch: saved={saved_config['warmup_steps']} current={args.warmup_steps}")
            if args.gate == "FORMAL" and int(saved_config["total_steps"]) != int(total_steps):
                raise RuntimeError(f"FORMAL resume total_steps mismatch: saved={saved_config['total_steps']} current={total_steps}")
            if args.gate != "FORMAL" and int(saved_config["total_steps"]) > int(total_steps):
                raise RuntimeError(f"resume smoke cannot shorten saved schedule: saved={saved_config['total_steps']} current={total_steps}")
            if Path(str(saved_config["htsat_checkpoint"])).resolve() != args.htsat_checkpoint.resolve():
                raise RuntimeError("resume HTSAT checkpoint mismatch")
            if Path(str(saved_config["mellow_root"])).resolve() != args.mellow_root.resolve():
                raise RuntimeError("resume Mellow root mismatch")
            if not saved_config.get("mellow_provenance"):
                raise RuntimeError("resume checkpoint has no Mellow provenance")
            if getattr(model, "_audio_provenance", {}).get("mellow_htsat_sha256") != saved_config["mellow_provenance"].get("mellow_htsat_sha256"):
                raise RuntimeError("resume Mellow provenance mismatch")
            state = _load_training_state(args.resume_from, optimizer, scheduler)
            start_step = int(state["global_step"])
            start_epoch = int(state.get("epoch", 0))
            start_batch_in_epoch = int(state.get("batch_in_epoch", 0))
            if start_batch_in_epoch % args.gradient_accumulation_steps != 0:
                raise RuntimeError("resume batch cursor is not aligned to the requested gradient accumulation steps")
            rank_rng = state.get("rng_states_by_rank", {}).get(str(rank)) or state.get("rng_states_by_rank", {}).get("0")
            if str(rank) not in state.get("rng_states_by_rank", {}):
                raise RuntimeError(f"resume checkpoint has no RNG state for rank {rank}")
            _restore_rng_state(rank_rng, device)
        ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False) if world > 1 else model
        metrics: list[dict[str, Any]] = []
        runtime_gradient_audit: dict[str, Any] | None = None
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
            completed_optimizer_steps = batch_in_epoch // args.gradient_accumulation_steps
            for _ in range(completed_optimizer_steps, steps_epoch):
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
                if runtime_gradient_audit is None:
                    runtime_gradient_audit = _mesh_runtime_gradient_audit(owner)
                    owner.mesh_model.model.gradient_audit_mode = False
                lr_before_optimizer_step = float(optimizer.param_groups[0]["lr"])
                optimizer_step += 1
                optimizer.step()
                scheduler.step()
                elapsed = max(time.perf_counter() - step_started, 1e-9)
                global_samples = int(args.micro_batch_size * world * args.gradient_accumulation_steps)
                answer_tokens = torch.tensor(int(batch["answer_attention_mask"].sum().item()), dtype=torch.long, device=device)
                if world > 1:
                    dist.all_reduce(answer_tokens, op=dist.ReduceOp.SUM)
                item = {"step": optimizer_step, "total_steps": max_steps, "progress_percent": 100.0 * optimizer_step / max(1, max_steps), "epoch": epoch, "batch_in_epoch": batch_in_epoch, "steps_per_epoch": steps_epoch, "loss": float(output.loss.detach().cpu()), "lr": float(optimizer.param_groups[0]["lr"]), "lr_before_optimizer_step": lr_before_optimizer_step, "grad_norm": float(grad_norm), "effective_answer_tokens": int(answer_tokens.item()), "step_time_seconds": elapsed, "samples_per_second": global_samples / elapsed, "audio_seconds_per_second": global_samples * 20.0 / elapsed, "gpu_memory_allocated_gib": float(torch.cuda.memory_allocated(device) / 1024**3), "gpu_memory_reserved_gib": float(torch.cuda.memory_reserved(device) / 1024**3), "gpu_memory_max_allocated_gib": float(torch.cuda.max_memory_allocated(device) / 1024**3), "gpu_memory_max_reserved_gib": float(torch.cuda.max_memory_reserved(device) / 1024**3), "router_stats": _router_stats(owner)}
                metrics.append(item)
                batch_in_epoch += args.gradient_accumulation_steps
                if rank == 0 and (optimizer_step % 10 == 0 or optimizer_step == max_steps):
                    memory = torch.cuda.memory_allocated(device) / 1024**3
                    print(f"[audio-train] step={optimizer_step}/{max_steps} progress={item['progress_percent']:.2f}% epoch={epoch + 1}/{args.epochs if args.gate == 'FORMAL' else '?'} batch={batch_in_epoch}/{len(loader)} loss={item['loss']:.6f} lr={item['lr']:.8g} step_s={item['step_time_seconds']:.3f} samples/s={item['samples_per_second']:.2f} audio_s/s={item['audio_seconds_per_second']:.2f} answer_tokens={item['effective_answer_tokens']} gpu_alloc_gib={item['gpu_memory_allocated_gib']:.3f} gpu_reserved_gib={item['gpu_memory_reserved_gib']:.3f} gpu_max_alloc_gib={item['gpu_memory_max_allocated_gib']:.3f} gpu_max_reserved_gib={item['gpu_memory_max_reserved_gib']:.3f} router_stats={item['router_stats']}", flush=True)
                save_due = (args.gate == "FORMAL" and (optimizer_step % max(1, args.save_every) == 0 or optimizer_step == max_steps)) or (args.gate == "STAGE7" and (optimizer_step == 10 or optimizer_step == max_steps))
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
                        elif args.gate == "FORMAL":
                            report["checkpoints"] = _prune_checkpoints(args.output_dir, args.checkpoint_retention)
                    if world > 1:
                        dist.barrier()
                    if optimizer_step >= max_steps:
                        break
                if optimizer_step >= max_steps:
                    break
            if batch_in_epoch >= steps_epoch * args.gradient_accumulation_steps:
                epoch += 1
                batch_in_epoch = 0
        report.update({"status": "PASS", "start_step": start_step, "end_step": optimizer_step, "optimizer_steps": optimizer_step, "steps_per_epoch": steps_epoch, "dropped_microbatches_per_epoch": dropped_microbatches, "total_formal_steps": formal_steps, "warmup_steps": args.warmup_steps, "effective_global_batch_size": int(args.micro_batch_size * world * args.gradient_accumulation_steps), "metrics": metrics if rank == 0 else [], "ddp_broadcast_buffers": False, "router_policy": "warning_only", "model_trainable_audit": (ddp.module if hasattr(ddp, "module") else ddp).trainable_parameter_audit(), "runtime_gradient_audit": runtime_gradient_audit, "resume_position": {"epoch": epoch, "batch_in_epoch": batch_in_epoch}, "checkpoints": report.get("checkpoints", [])})
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
