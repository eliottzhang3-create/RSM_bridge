#!/usr/bin/env python3
"""Two-step, checkpoint-free CUDA memory probe for physical batch 32."""
from __future__ import annotations

import json
import os
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import train_audio_smollm2_shared_store_mellow_official_faithful_v2_135m_ddp as faithful


MICRO_BATCH = 32
GRADIENT_ACCUMULATION_STEPS = 1
PROBE_STEPS = 2
REPORT_NAME = "physical_batch32_memory_probe_report.json"


def memory_snapshot(device: torch.device) -> dict[str, int | float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    return {
        "allocated_bytes": int(allocated),
        "reserved_bytes": int(reserved),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_gib": allocated / 1024**3,
        "reserved_gib": reserved / 1024**3,
        "free_gib": free_bytes / 1024**3,
        "total_gib": total_bytes / 1024**3,
    }


def write_failure(
    output_dir: Path,
    rank: int,
    device: torch.device | None,
    exc: BaseException,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    memory: dict[str, Any] | None = None
    if device is not None and torch.cuda.is_available():
        try:
            memory = {
                **memory_snapshot(device),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        except Exception:
            memory = None
    payload = {
        "status": "FAIL",
        "probe": "mellow_official_faithful_v2_physical_batch32",
        "rank": rank,
        "exception_type": type(exc).__name__,
        "exception": repr(exc),
        "cuda_out_of_memory": bool(
            isinstance(exc, torch.cuda.OutOfMemoryError)
            or "out of memory" in str(exc).lower()
        ),
        "memory": memory,
        "traceback": traceback.format_exc(),
    }
    (output_dir / f"rank{rank}.failure.json").write_text(
        json.dumps(payload, indent=2) + chr(10),
        encoding="utf-8",
    )


def main() -> None:
    args = faithful.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
    device: torch.device | None = None
    initialized = False

    try:
        if args.mode != "formal":
            raise ValueError("probe requires --mode formal")
        if world != 8 or args.world_size != 8:
            raise ValueError("probe requires exactly eight ranks")
        if args.micro_batch_size != MICRO_BATCH:
            raise ValueError(f"probe requires microbatch={MICRO_BATCH}")
        if args.gradient_accumulation_steps != GRADIENT_ACCUMULATION_STEPS:
            raise ValueError("probe requires gradient accumulation 1")
        if args.num_workers != 0:
            raise ValueError("probe requires num_workers=0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world,
            timeout=timedelta(minutes=args.dist_timeout_minutes),
        )
        initialized = True

        output_error = [None]
        if rank == 0:
            if args.output_dir.exists():
                output_error[0] = f"refusing existing probe output: {args.output_dir}"
            else:
                args.output_dir.mkdir(parents=True)
        dist.broadcast_object_list(output_error, src=0)
        if output_error[0] is not None:
            raise FileExistsError(output_error[0])
        dist.barrier()

        faithful.baseline._seed(args.seed, rank)
        inventory = faithful.store_inventory(args)
        model, tokenizer = faithful.load_model(args, device)
        dataset = faithful.ReasonAQADataset(
            args.train_manifest,
            tokenizer,
            unique_waveform_store_dir=args.unique_waveform_store_dir,
        )
        shape = faithful.training_shape(args, len(dataset))
        if shape["global_batch_size"] != world * MICRO_BATCH:
            raise RuntimeError(f"unexpected global batch: {shape}")

        model.train()
        trainable_audit = model.trainable_parameter_audit()
        if trainable_audit.get("training_mode_contract") is not True:
            raise RuntimeError("trainable-parameter audit failed")
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        tracker = faithful.GradNormTracker(
            initial_l2_norm=faithful.GRAD_TRACKER_INITIAL_L2_NORM,
            initial_max_norm=faithful.GRAD_TRACKER_INITIAL_MAX_NORM,
            overdrive_factor=faithful.GRAD_TRACKER_OVERDRIVE_FACTOR,
            momentum=faithful.GRAD_TRACKER_MOMENTUM,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        ddp = DDP(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

        sampler = faithful.ContiguousDistributedEpochSampler(
            len(dataset),
            num_replicas=world,
            rank=rank,
            per_rank_batch_size=MICRO_BATCH,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            epoch=0,
            start_optimizer_step=0,
        )
        loader = DataLoader(
            dataset,
            batch_size=MICRO_BATCH,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(args.seed + rank),
            collate_fn=lambda rows: faithful.collate_reasonaqa(rows, tokenizer),
        )
        iterator = iter(loader)

        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        baseline_memory = memory_snapshot(device)
        step_reports: list[dict[str, Any]] = []
        first_batch_audit: dict[str, Any] | None = None
        first_gradient_audit: dict[str, Any] | None = None

        for step in range(1, PROBE_STEPS + 1):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            step_started = time.perf_counter()

            data_started = time.perf_counter()
            batch = next(iterator)
            moved = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            torch.cuda.synchronize(device)
            after_data = time.perf_counter()
            memory_after_data = memory_snapshot(device)

            forward_started = time.perf_counter()
            output = ddp(**{
                key: moved[key]
                for key in (
                    "audio1",
                    "audio2",
                    "prompt_input_ids",
                    "prompt_attention_mask",
                    "answer_input_ids",
                    "answer_attention_mask",
                    "audio2_reused_mask",
                    "single_audio_slot_mask",
                )
            })
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise RuntimeError("probe produced nonfinite loss")
            torch.cuda.synchronize(device)
            after_forward = time.perf_counter()
            memory_after_forward = memory_snapshot(device)

            owner = ddp.module
            if first_batch_audit is None:
                first_batch_audit = faithful.batch_contract_audit(owner, moved)

            backward_started = time.perf_counter()
            scaler.scale(output.loss).backward()
            torch.cuda.synchronize(device)
            after_backward = time.perf_counter()
            memory_after_backward = memory_snapshot(device)

            if first_gradient_audit is None:
                gradient = owner.runtime_gradient_audit()
                first_gradient_audit = {
                    "passed": bool(
                        gradient.get("all_decoder_layers_have_finite_gradient")
                        and gradient.get("embedding_has_finite_gradient")
                        and gradient.get("lm_head_has_finite_gradient")
                        and gradient.get("htsat_frozen_and_gradient_free")
                        and all(gradient.get("bridge_gradients", {}).values())
                        and all(gradient.get("c2l_gradients", {}).values())
                    ),
                    **gradient,
                }
                if not first_gradient_audit["passed"]:
                    raise RuntimeError("first-step gradient audit failed")

            optimizer_started = time.perf_counter()
            scaler.unscale_(optimizer)
            grad_norm, grad_scale = tracker.track_and_clip_(
                list(owner.named_parameters())
            )
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            after_optimizer = time.perf_counter()
            memory_after_optimizer = memory_snapshot(device)

            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            step_seconds = after_optimizer - step_started
            step_reports.append({
                "step": step,
                "loss": float(output.loss.detach().float().item()),
                "raw_grad_l2_norm": float(grad_norm),
                "grad_scale": float(grad_scale),
                "data_seconds": after_data - data_started,
                "forward_seconds": after_forward - forward_started,
                "backward_seconds": after_backward - backward_started,
                "optimizer_seconds": after_optimizer - optimizer_started,
                "step_seconds": step_seconds,
                "rank_samples_per_second": MICRO_BATCH / step_seconds,
                "estimated_global_samples_per_second": world * MICRO_BATCH / step_seconds,
                "memory_after_data": memory_after_data,
                "memory_after_forward": memory_after_forward,
                "memory_after_backward": memory_after_backward,
                "memory_after_optimizer": memory_after_optimizer,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "peak_allocated_gib": peak_allocated / 1024**3,
                "peak_reserved_gib": peak_reserved / 1024**3,
                "reserved_headroom_gib": (
                    memory_after_optimizer["total_bytes"] - peak_reserved
                ) / 1024**3,
            })
            # Do not retain logits or the CUDA input batch into the next
            # measured step. The recorded peak already includes the complete
            # forward, backward, GradNorm, and first Adam-state allocation.
            del output, moved, batch

        local_report = {
            "rank": rank,
            "local_rank": local_rank,
            "device_name": torch.cuda.get_device_name(device),
            "baseline_memory": baseline_memory,
            "steps": step_reports,
        }
        rank_reports = faithful.gather(local_report, world)
        if rank == 0:
            all_steps = [step for item in rank_reports for step in item["steps"]]
            report = {
                "status": "PASS",
                "probe": "mellow_official_faithful_v2_physical_batch32",
                "isolated_from_formal_training": True,
                "writes_checkpoints": False,
                "steps": PROBE_STEPS,
                "world_size": world,
                "physical_batch_size_per_rank": MICRO_BATCH,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "effective_global_batch_size": world * MICRO_BATCH,
                "precision": "float32; GradScaler disabled",
                "dataset_rows": len(dataset),
                "training_shape": shape,
                "store_inventory": inventory,
                "route_code_sha256": faithful.route_code_identity(),
                "probe_script_sha256": faithful.sha256(Path(__file__).resolve()),
                "model_trainable_audit": trainable_audit,
                "first_batch_audit": first_batch_audit,
                "first_gradient_audit": first_gradient_audit,
                "max_peak_allocated_gib": max(item["peak_allocated_gib"] for item in all_steps),
                "max_peak_reserved_gib": max(item["peak_reserved_gib"] for item in all_steps),
                "min_reserved_headroom_gib": min(item["reserved_headroom_gib"] for item in all_steps),
                "rank_reports": rank_reports,
            }
            (args.output_dir / REPORT_NAME).write_text(
                json.dumps(report, indent=2) + chr(10),
                encoding="utf-8",
            )
            print(
                "[physical-batch32-probe] PASS "
                f"peak_allocated_gib={report['max_peak_allocated_gib']:.3f} "
                f"peak_reserved_gib={report['max_peak_reserved_gib']:.3f} "
                f"reserved_headroom_gib={report['min_reserved_headroom_gib']:.3f} "
                f"report={args.output_dir / REPORT_NAME}",
                flush=True,
            )
        dist.barrier()
    except BaseException as exc:
        try:
            write_failure(args.output_dir, rank, device, exc)
        finally:
            raise
    finally:
        if initialized and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    main()
