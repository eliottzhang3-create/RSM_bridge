#!/usr/bin/env python3
"""Real depth-10 BF16/AdamW activation-memory preflight for one training GPU."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_5_10x2_5_mesh_mellow.model import _load_mellow_wrapper  # noqa: E402
from audio_5_10x2to10_5_mesh_mellow_shared_store.model import (  # noqa: E402
    AudioMeshConfig, AudioMeshModel, RecursiveLlamaForCausalLM,
)

# These must match the fixed shared-store training line.  The released Mellow
# v0.ckpt is a full-model checkpoint whose HTSAT keys live under
# audio_encoder.base.htsat.; it is not the standalone HTSAT artifact used to
# train checkpoint-011343.
DEFAULT_MELLOW = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main")
DEFAULT_HTSAT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt"
)


def _same_resolved_path(left: Path, right: Path) -> bool:
    """Treat /hpc_stor03 and /mnt/cloudstorfs aliases as the same artifact."""
    return left.expanduser().resolve(strict=True) == right.expanduser().resolve(strict=True)


def _validate_audio_artifact_provenance(root: Path, args: argparse.Namespace) -> dict[str, str]:
    """Require the exact external HTSAT/Mellow artifacts used by the source run."""
    migration = json.loads(
        (root / "variable_depth_init_report.json").read_text(encoding="utf-8")
    )
    source_checkpoint = Path(str(migration.get("source", {}).get("source", "")))
    source_config_path = source_checkpoint / "audio_mesh_config.json"
    if not source_config_path.is_file():
        raise FileNotFoundError(
            "cannot verify source audio provenance because audio_mesh_config.json is missing: "
            f"{source_config_path}"
        )
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    expected_htsat = Path(str(source_config.get("htsat_checkpoint", "")))
    expected_mellow = Path(str(source_config.get("mellow_root", "")))
    if not _same_resolved_path(args.htsat_checkpoint, expected_htsat):
        raise RuntimeError(
            "T=10 preflight HTSAT differs from checkpoint-011343 training provenance: "
            f"requested={args.htsat_checkpoint} expected={expected_htsat}"
        )
    if not _same_resolved_path(args.mellow_root, expected_mellow):
        raise RuntimeError(
            "T=10 preflight Mellow source differs from checkpoint-011343 training provenance: "
            f"requested={args.mellow_root} expected={expected_mellow}"
        )
    return {
        "htsat_checkpoint": str(args.htsat_checkpoint.resolve(strict=True)),
        "mellow_root": str(args.mellow_root.resolve(strict=True)),
        "source_audio_config": str(source_config_path.resolve(strict=True)),
    }


def _load(args: argparse.Namespace, device: torch.device) -> tuple[AudioMeshModel, Any]:
    root = args.init_artifact.resolve(strict=True)
    marker = json.loads((root / "artifact_complete.json").read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise RuntimeError("variable-depth initialization artifact is incomplete")
    provenance = _validate_audio_artifact_provenance(root, args)
    mesh = RecursiveLlamaForCausalLM.from_pretrained(
        # Match formal training residency: FP32 master parameters and AdamW
        # states, with BF16 used only inside autocast for forward/backward.
        root / "mesh_model", local_files_only=True, torch_dtype=torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper, htsat, _ = _load_mellow_wrapper(args.mellow_root, args.htsat_checkpoint, device)
    model = AudioMeshModel(
        mesh, tokenizer, wrapper, htsat,
        AudioMeshConfig(compact_single_audio_prefix=False, max_context_length=768),
    ).to(device)
    audio_state = torch.load(root / "audio_bridge.pt", map_location=device, weights_only=False)
    model.bridge.load_state_dict(audio_state["bridge"], strict=True)
    model.htsat_wrapper.c2l.load_state_dict(audio_state["c2l"], strict=True)
    model.train()
    model._preflight_audio_provenance = provenance
    return model, tokenizer


def _batch(tokenizer: Any, device: torch.device, batch_size: int) -> dict[str, torch.Tensor]:
    # Exact formal-data maximum: fixed260 + prompt129 + answer250 = 639.
    eos = int(tokenizer.eos_token_id)
    fill = int(tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos)
    text = torch.full((batch_size, 379), fill, dtype=torch.long, device=device)
    text[:, -1] = eos
    audio1 = torch.zeros((batch_size, 1, 320000), dtype=torch.float32, device=device)
    audio2 = torch.zeros_like(audio1)
    return {
        "audio1": audio1, "audio2": audio2, "text_ids": text,
        "text_attention_mask": torch.ones_like(text),
        "prompt_lengths": torch.full((batch_size,), 129, dtype=torch.long, device=device),
        "answer_lengths": torch.full((batch_size,), 250, dtype=torch.long, device=device),
        "audio2_reused_mask": torch.zeros((batch_size,), dtype=torch.bool, device=device),
        "single_audio_slot_mask": torch.zeros((batch_size,), dtype=torch.bool, device=device),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("T=10 activation preflight requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    report: dict[str, Any] = {
        "status": "FAIL", "recursive_depth": 10,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "sequence_length": 639, "prompt_tokens": 129, "answer_tokens": 250,
        "optimizer": "AdamW", "dtype": "bfloat16",
        "find_unused_parameters": True,
    }
    try:
        model, tokenizer = _load(args, device)
        report["audio_provenance"] = model._preflight_audio_provenance
        ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in ddp.parameters() if parameter.requires_grad], lr=1e-3,
        )
        batch = _batch(tokenizer, device, args.micro_batch_size)

        # Materialize AdamW moments before measuring the exact GA=4 peak.
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            warm = ddp(recursive_depth=10, **batch).loss
        warm.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)

        losses = []
        for _ in range(args.gradient_accumulation_steps):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ddp(recursive_depth=10, **batch).loss / args.gradient_accumulation_steps
            loss.backward()
            losses.append(float(loss.detach()) * args.gradient_accumulation_steps)
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        total = torch.cuda.get_device_properties(device).total_memory
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        ratio = peak_reserved / total
        report.update({
            "status": "PASS" if ratio <= args.max_reserved_ratio else "FAIL",
            "losses": losses, "device": torch.cuda.get_device_name(device),
            "total_bytes": total, "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "peak_reserved_ratio": ratio, "max_reserved_ratio": args.max_reserved_ratio,
        })
        if report["status"] != "PASS":
            report["failure"] = "T=10 completed but exceeded the reserved-memory threshold"
    except torch.cuda.OutOfMemoryError as exc:
        report["failure"] = f"CUDA OOM at exact formal microbatch/GA: {exc}"
        torch.cuda.empty_cache()
    except Exception as exc:
        report["failure"] = f"preflight execution error: {type(exc).__name__}: {exc}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-artifact", type=Path, required=True)
    parser.add_argument("--mellow-root", type=Path, default=DEFAULT_MELLOW)
    parser.add_argument("--htsat-checkpoint", type=Path, default=DEFAULT_HTSAT)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-reserved-ratio", type=float, default=0.90)
    args = parser.parse_args()
    report = run(args)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
