#!/usr/bin/env python3
"""Six-partition rank-RAM training for the original SmolLM2 audio baseline.

The data, optimizer, scheduling, compact-prefix, checkpoint, and strict
partition-release contracts mirror the current Audio MeSH partition route.
The sole experimental change is the text backbone: this route uses the
standard 30-layer SmolLM2-135M LlamaForCausalLM and has no MeSH memory/router.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
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
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import train_audio_smollm2_135m_mellow_ddp as base
from audio_smollm2_135m_mellow.data import ReasonAQADataset, collate_reasonaqa
from audio_smollm2_135m_mellow.model import (
    AUDIO_DUAL_PREFIX_TOKENS,
    AUDIO_SINGLE_PREFIX_TOKENS,
    MAPPER_CONTRACT,
    ORIGINAL_SMOLLM2_CONTRACT,
)


CONTRACT = "smollm2_component_partitions6_rank_ram_compact_audio_answer_eos_v2"
PARTITION_STORE_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/data/rsmol_reasonaqa_train_component_partitions6_32k_10s_f32_v2"
CONFIG_FILENAME = "audio_smollm2_partition_config.json"
SMOKE_SEGMENTS = ((2, 10), (0, 10), (1, 2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--partition-store-root", type=Path, default=Path(PARTITION_STORE_ROOT))
    parser.add_argument("--model-path", "--smollm2-model", dest="model_path", type=Path, default=Path(base.DEFAULT_MODEL))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--expected-resume-step", type=int)
    parser.add_argument("--smoke20-report", type=Path)
    parser.add_argument("--smoke-resume-report", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--htsat-checkpoint", type=Path, default=Path(base.DEFAULT_HTSAT))
    parser.add_argument("--mellow-root", type=Path, default=Path(base.DEFAULT_MELLOW))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--checkpoint-retention", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dist-timeout-minutes", type=int, default=30)
    parser.add_argument("--release-min-fraction", type=float, default=0.70)
    parser.add_argument("--release-timeout-seconds", type=int, default=120)
    return parser.parse_args(argv)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    if (root / "BUILDING").exists():
        raise RuntimeError("partition root is BUILDING")
    report_path = root / "materialization_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASS"
        or report.get("format") != "reasonaqa_component_partition_stores_v1"
        or report.get("duplicated_audio") != 0
        or report.get("source_payload_sha256_reverified") is not True
    ):
        raise RuntimeError("materialization root is not an audited zero-copy PASS")
    partitions = report.get("partitions", [])
    if len(partitions) != 6 or sorted(int(item["partition_id"]) for item in partitions) != list(range(6)):
        raise RuntimeError("expected exactly partitions 0..5")
    result: dict[str, Any] = {
        "root": str(root),
        "report_sha256": _sha(report_path),
        "partitions": [],
    }
    for partition_id in range(6):
        directory = root / f"partition_{partition_id}"
        if (directory / "BUILDING").exists():
            raise RuntimeError(f"partition {partition_id} is BUILDING")
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        entry = next(item for item in partitions if int(item["partition_id"]) == partition_id)
        if (
            metadata.get("status") != "PASS"
            or metadata.get("waveform_verification", {}).get("passed") is not True
            or entry.get("waveform_verification", {}).get("passed") is not True
        ):
            raise RuntimeError(f"partition {partition_id} is not verified PASS")
        if int(metadata.get("partition_id", -1)) != partition_id or int(entry["partition_id"]) != partition_id:
            raise RuntimeError("partition metadata order mismatch")
        for key in ("manifest_sha256", "index_sha256", "waveform_sha256", "num_unique_audio_files", "total_waveform_bytes"):
            if metadata.get(key) != entry.get(key):
                raise RuntimeError(f"partition {partition_id} report/metadata {key} mismatch")
        if _sha(directory / "rows.jsonl") != metadata["manifest_sha256"] or _sha(directory / "index.jsonl") != metadata["index_sha256"]:
            raise RuntimeError(f"partition {partition_id} manifest/index hash mismatch")
        if (directory / "waveforms.f32").stat().st_size != int(metadata["total_waveform_bytes"]):
            raise RuntimeError(f"partition {partition_id} payload byte mismatch")
        result["partitions"].append({
            "id": partition_id,
            "rows": int(entry["qa_rows"]),
            "audio": int(metadata["num_unique_audio_files"]),
            "bytes": int(metadata["total_waveform_bytes"]),
            "manifest_sha256": metadata["manifest_sha256"],
            "index_sha256": metadata["index_sha256"],
            "waveform_sha256": metadata["waveform_sha256"],
        })
    if sum(item["rows"] for item in result["partitions"]) != 968059:
        raise RuntimeError("partition QA total differs from the accepted materialization")
    if sum(item["bytes"] for item in result["partitions"]) != int(report["total_materialized_waveform_bytes"]):
        raise RuntimeError("partition waveform total differs from materialization")
    return result


def _quotas(counts: list[int], epochs: int, global_batch: int) -> list[list[int]]:
    if len(counts) != 6 or epochs <= 0:
        raise ValueError("six partitions and positive epochs required")
    total = sum(counts) // global_batch
    large = counts[0] // global_batch
    remaining = total - large
    weights = counts[1:]
    weight_total = sum(weights)
    cumulative = [0] * 5
    result: list[list[int]] = []
    for epoch in range(epochs):
        ideal = [remaining * count / weight_total for count in weights]
        current = [math.floor(value) for value in ideal]
        extras = remaining - sum(current)
        priority = sorted(range(5), key=lambda index: (-(epoch + 1) * ideal[index] + cumulative[index] + current[index], (index + epoch) % 5))
        for index in priority[:extras]:
            current[index] += 1
        cumulative = [before + now for before, now in zip(cumulative, current)]
        row = [large, *current]
        if sum(row) != total or row[0] * global_batch > counts[0]:
            raise AssertionError("epoch quota allocation failed")
        result.append(row)
    return result


def _order(seed: int, epoch: int) -> list[int]:
    rng = random.Random((seed + 1) * 1_000_003 + epoch * 10_007)
    order = list(range(1, 6))
    rng.shuffle(order)
    order.insert(rng.randrange(5), 0)
    if order[-1] == 0:
        raise AssertionError("partition 0 must not be the final partition")
    return order


def _schedule(args: argparse.Namespace, inventory: dict[str, Any]) -> list[dict[str, int]]:
    if args.mode == "smoke":
        return [
            {"epoch": 0, "position": position, "partition_id": partition_id, "steps": steps}
            for position, (partition_id, steps) in enumerate(SMOKE_SEGMENTS)
        ]
    counts = [entry["rows"] for entry in inventory["partitions"]]
    global_batch = args.world_size * args.micro_batch_size * args.gradient_accumulation_steps
    quotas = _quotas(counts, args.epochs, global_batch)
    return [
        {"epoch": epoch, "position": position, "partition_id": partition_id, "steps": quotas[epoch][partition_id]}
        for epoch in range(args.epochs)
        for position, partition_id in enumerate(_order(args.seed, epoch))
    ]


def _formal_smoke_gate(args: argparse.Namespace, inventory: dict[str, Any]) -> dict[str, Any]:
    if args.smoke20_report is None or args.smoke_resume_report is None:
        raise ValueError("formal training requires --smoke20-report and --smoke-resume-report")
    initial = json.loads(args.smoke20_report.read_text(encoding="utf-8"))
    resumed = json.loads(args.smoke_resume_report.read_text(encoding="utf-8"))
    expected_schedule = [
        {"epoch": 0, "position": position, "partition_id": partition_id, "steps": steps}
        for position, (partition_id, steps) in enumerate(SMOKE_SEGMENTS)
    ]
    for name, report, expected_start, expected_end in (
        ("smoke20", initial, {"segment": 0, "segment_step": 0, "global_step": 0}, {"segment": 2, "segment_step": 0, "global_step": 20}),
        ("resume2", resumed, {"segment": 2, "segment_step": 0, "global_step": 20}, {"segment": 3, "segment_step": 0, "global_step": 22}),
    ):
        if (
            report.get("status") != "PASS"
            or report.get("mode") != "smoke"
            or report.get("training_contract") != CONTRACT
            or report.get("hard_failures")
            or report.get("inventory") != inventory
            or report.get("schedule") != expected_schedule
            or report.get("start_cursor") != expected_start
            or report.get("end_cursor") != expected_end
            or report.get("seed") != args.seed
        ):
            raise RuntimeError(f"formal gate rejects {name} report")
        expected_segments = 2 if name == "smoke20" else 1
        if len(report.get("segments", [])) != expected_segments:
            raise RuntimeError(f"formal gate rejects {name} segment count")
        audit = report.get("first_step_gradient_audit", {})
        required_audit = (
            audit.get("standard_30_layer_smollm2") is True,
            audit.get("all_decoder_layers_have_finite_gradient") is True,
            audit.get("embedding_has_finite_gradient") is True,
            audit.get("lm_head_has_finite_gradient") is True,
            audit.get("all_bridge_gradients_finite") is True,
            audit.get("all_c2l_gradients_finite") is True,
            audit.get("htsat_frozen_and_gradient_free") is True,
            audit.get("has_router_parameters") is False,
            audit.get("compact_prefix_contract") is True,
            audit.get("answer_eos_contract") is True,
            audit.get("training_mode_contract") is True,
        )
        if not all(required_audit):
            raise RuntimeError(f"formal gate rejects {name} baseline gradient/sequence audit")
        if any(
            len(segment.get("release", [])) != 8
            or not all(item.get("passed") for item in segment["release"])
            or segment.get("cgroup_release", {}).get("anon_drop_bytes") is None
            or segment["cgroup_release"]["anon_drop_bytes"] < segment["cgroup_release"]["required_bytes"]
            or len(segment.get("steps", [])) != segment["segment"]["steps"]
            for segment in report["segments"]
        ):
            raise RuntimeError(f"formal gate rejects {name} release audit")
        expected_segment_rows = expected_schedule[:2] if name == "smoke20" else expected_schedule[2:]
        if [segment.get("segment") for segment in report["segments"]] != expected_segment_rows:
            raise RuntimeError(f"formal gate rejects {name} partition order/step contract")
    initial_checkpoints = initial.get("checkpoints", [])
    resumed_checkpoints = resumed.get("checkpoints", [])
    if len(initial_checkpoints) != 1 or len(resumed_checkpoints) != 1:
        raise RuntimeError("smoke reports must expose exactly checkpoint-000020 and checkpoint-000022")
    checkpoint20 = Path(initial_checkpoints[0]).resolve()
    checkpoint22 = Path(resumed_checkpoints[0]).resolve()
    if checkpoint20.name != "checkpoint-000020" or checkpoint22.name != "checkpoint-000022":
        raise RuntimeError("smoke checkpoint names do not match the 20+2 contract")
    for checkpoint_path, expected_step in ((checkpoint20, 20), (checkpoint22, 22)):
        marker = json.loads((checkpoint_path / "checkpoint_complete.json").read_text(encoding="utf-8"))
        config = json.loads((checkpoint_path / CONFIG_FILENAME).read_text(encoding="utf-8"))
        if (
            marker.get("status") != "complete"
            or marker.get("global_step") != expected_step
            or marker.get("contract") != CONTRACT
            or config.get("contract") != CONTRACT
            or config.get("mode") != "smoke"
            or config.get("architecture_contract") != ORIGINAL_SMOLLM2_CONTRACT
            or config.get("mapper_contract") != MAPPER_CONTRACT
            or config.get("compact_single_audio_prefix") is not True
            or config.get("prefix_tokens") != {"single": 130, "dual": 260}
            or config.get("answer_termination") != {"token": "<|endoftext|>", "included_in_max_answer_tokens": True, "supervised": True}
        ):
            raise RuntimeError(f"smoke checkpoint artifact failed validation: {checkpoint_path}")
    checkpoint = str(checkpoint20)
    if resumed.get("resume_checkpoint") != checkpoint or resumed.get("resume_verified_two_steps") is not True:
        raise RuntimeError("resume report does not prove continuation from smoke checkpoint-000020")
    change_audit = resumed.get("resume_parameter_change_audit", {})
    change_groups_ok = change_audit.get("all_groups_changed") is True and all(
        isinstance(change_audit.get(group), dict)
        and change_audit[group].get("changed") is True
        and change_audit[group].get("finite") is True
        and change_audit[group].get("exact_equal") is False
        and float(change_audit[group].get("max_abs_delta", 0.0)) > 0.0
        for group in ("text", "bridge", "c2l")
    )
    if not change_groups_ok:
        raise RuntimeError("resume report does not prove text/bridge/c2l parameter updates")
    return {
        "smoke20_report": str(args.smoke20_report.resolve()),
        "smoke_resume_report": str(args.smoke_resume_report.resolve()),
        "checkpoint20": checkpoint,
    }


def _plan(dataset: ReasonAQADataset, *, seed: int, epoch: int, pid: int, steps: int, global_batch: int) -> tuple[list[list[int]], dict[str, Any]]:
    rows = list(range(len(dataset)))
    random.Random((seed + 1) * 1_000_003 + epoch * 10_007 + pid * 101).shuffle(rows)
    required = steps * global_batch
    if pid == 0 and required > len(rows):
        raise RuntimeError("partition 0 repetition forbidden")
    if not rows:
        raise RuntimeError("empty partition")
    selected = [rows[index % len(rows)] for index in range(required)]
    chunks = [selected[index:index + global_batch] for index in range(0, required, global_batch)]
    if len(chunks) != steps or any(len(chunk) != global_batch for chunk in chunks):
        raise AssertionError("incomplete optimizer window")
    digest = hashlib.sha256(",".join(str(index) for chunk in chunks for index in chunk).encode("ascii")).hexdigest()
    return chunks, {
        "plan_sha256": digest,
        "source_rows": len(dataset),
        "selected_rows": required,
        "repeated_rows": max(0, required - len(rows)),
        "dropped_rows": max(0, len(rows) - required),
        "shuffle": "all partition rows together, then same-partition wraparound only",
        "global_batch": global_batch,
    }


class _RankLocalStoreWaveforms:
    """Clone every used store row into process-owned anonymous CPU RAM."""

    def __init__(self, dataset: ReasonAQADataset) -> None:
        if dataset.unique_waveform_store is None:
            raise RuntimeError("rank-local waveform residency requires a unique waveform store")
        self.dataset = dataset
        self.store = dataset.unique_waveform_store
        self.items: dict[int, torch.Tensor] = {}
        self.hits = self.misses = self.cloned_bytes = self.current_bytes = 0

    def _get_audio_id(self, audio_id: int) -> torch.Tensor:
        audio_id = int(audio_id)
        value = self.items.get(audio_id)
        if value is not None:
            self.hits += 1
            return value
        self.misses += 1
        value = self.store.load_audio_id(audio_id).clone()
        size = int(value.numel() * value.element_size())
        self.items[audio_id] = value
        self.cloned_bytes += size
        self.current_bytes += size
        return value

    def _get(self, path: str) -> torch.Tensor:
        return self._get_audio_id(self.store.locate(path))

    def materialize_from_cached_metadata(self, row: int) -> dict[str, Any]:
        row_index = int(row)
        source = self.dataset.rows[row_index]
        prompt = str(source.get("prompt") or source.get("question") or source.get("input") or "")
        answer = str(source.get("answer") or source.get("target") or source.get("output") or source.get("caption1") or "")
        if not answer:
            raise ValueError(f"manifest row {row_index} lacks answer")
        audio1, audio2 = self.dataset.audio_paths(row_index)
        single_slot, same_waveform = self.dataset.audio_structure(row_index)
        first = self._get(audio1)
        return {
            "audio1": first,
            "audio2": None if single_slot else (first if same_waveform else self._get(audio2)),
            "prompt": prompt,
            "answer": answer,
            "row_index": row_index,
            "audio2_reused": same_waveform,
            "single_audio_slot": single_slot,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "waveform_hits": self.hits,
            "waveform_misses": self.misses,
            "cloned_bytes": self.cloned_bytes,
            "resident_unique_audio": len(self.items),
            "cache_current_bytes": self.current_bytes,
            "waveform_evictions": 0,
        }


def _cgroup_memberships_and_mounts() -> tuple[list[tuple[str, str]], list[dict[str, str]], list[str]]:
    errors: list[str] = []
    memberships: list[tuple[str, str]] = []
    mounts: list[dict[str, str]] = []
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            _, controllers, path = line.split(":", 2)
            memberships.append((controllers, path))
    except (OSError, ValueError) as exc:
        errors.append(f"cgroup memberships: {exc!r}")
    try:
        for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
            left, right = line.split(" - ", 1)
            left_fields, right_fields = left.split(), right.split()
            if right_fields[0] not in {"cgroup", "cgroup2"}:
                continue
            mounts.append({
                "root": left_fields[3],
                "mountpoint": left_fields[4],
                "fstype": right_fields[0],
                "source": right_fields[1],
                "super_options": right_fields[2] if len(right_fields) > 2 else "",
            })
    except (OSError, ValueError, IndexError) as exc:
        errors.append(f"cgroup mounts: {exc!r}")
    return memberships, mounts, errors


def _cgroup_candidate_paths(controller: str) -> tuple[list[tuple[int, Path]], list[str]]:
    memberships, mounts, errors = _cgroup_memberships_and_mounts()
    candidates: list[tuple[int, Path]] = []
    for controllers, membership in memberships:
        version = 2 if controllers == "" else 1
        if version == 1 and controller not in controllers.split(","):
            continue
        for mount in mounts:
            if mount["fstype"] != ("cgroup2" if version == 2 else "cgroup"):
                continue
            if version == 1 and controller not in (mount["source"] + "," + mount["super_options"]).split(","):
                continue
            mount_root = mount["root"].rstrip("/") or "/"
            member = membership.rstrip("/") or "/"
            if mount_root == "/":
                relative = member.lstrip("/")
            elif member == mount_root:
                relative = ""
            elif member.startswith(mount_root + "/"):
                relative = member[len(mount_root):].lstrip("/")
            else:
                continue
            candidates.extend(((version, Path(mount["mountpoint"]) / relative), (version, Path(mount["mountpoint"]))))
    for controllers, membership in memberships:
        if controllers == "":
            candidates.extend(((2, Path("/sys/fs/cgroup") / membership.lstrip("/")), (2, Path("/sys/fs/cgroup"))))
        elif controller in controllers.split(","):
            root = Path("/sys/fs/cgroup") / controller
            candidates.extend(((1, root / membership.lstrip("/")), (1, root)))
    unique: list[tuple[int, Path]] = []
    seen: set[tuple[int, str]] = set()
    for version, path in candidates:
        key = (version, str(path))
        if key not in seen:
            seen.add(key)
            unique.append((version, path))
    return unique, errors


def _cgroup_memory_snapshot() -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    candidates, discovery_errors = _cgroup_candidate_paths("memory")
    for version, root in candidates:
        current_name, maximum_name = (("memory.current", "memory.max") if version == 2 else ("memory.usage_in_bytes", "memory.limit_in_bytes"))
        try:
            current = int((root / current_name).read_text(encoding="utf-8").strip())
            maximum_text = (root / maximum_name).read_text(encoding="utf-8").strip()
            raw_stats = {}
            for line in (root / "memory.stat").read_text(encoding="utf-8").splitlines():
                key, value = line.split()
                raw_stats[key] = int(value)
        except (OSError, ValueError, IndexError) as exc:
            attempts.append({"version": version, "path": str(root), "error": repr(exc)})
            continue
        maximum = None if maximum_text == "max" else int(maximum_text)
        if maximum is not None and maximum >= 1 << 60:
            maximum = None
        if version == 1:
            stats = {
                "anon": raw_stats.get("total_rss", raw_stats.get("rss", 0)),
                "file": raw_stats.get("total_cache", raw_stats.get("cache", 0)),
            }
        else:
            stats = {key: raw_stats[key] for key in ("anon", "file", "file_mapped", "file_dirty", "file_writeback", "active_file", "inactive_file") if key in raw_stats}
        return {
            "status": "PASS",
            "cgroup_version": version,
            "cgroup_path": str(root),
            "memory_current_bytes": current,
            "memory_max_bytes": maximum,
            "memory_stat_bytes": stats,
            "discovery_errors": discovery_errors,
            "failed_candidates": attempts,
        }
    return {
        "status": "UNAVAILABLE",
        "cgroup_version": None,
        "cgroup_path": None,
        "memory_current_bytes": None,
        "memory_max_bytes": None,
        "memory_stat_bytes": {},
        "discovery_errors": discovery_errors,
        "failed_candidates": attempts,
    }


def _rss() -> int:
    with Path("/proc/self/status").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/self/status lacks VmRSS")


def _memory() -> dict[str, Any]:
    return {"rss_bytes": _rss(), "cgroup": _cgroup_memory_snapshot()}


def _anon(snapshot: dict[str, Any]) -> int | None:
    cgroup = snapshot.get("cgroup") or {}
    if cgroup.get("status") != "PASS":
        return None
    value = cgroup.get("memory_stat_bytes", {}).get("anon")
    return None if value is None else int(value)


def _trim() -> None:
    if ctypes.CDLL(None).malloc_trim(0) not in (0, 1):
        raise RuntimeError("malloc_trim returned an invalid result")


def _gather(local: Any, world: int) -> list[Any]:
    gathered: list[Any] = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered[0] = local
    return gathered


def _sequence_audit(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    labels = model.last_labels
    prefix_lengths = model.last_prefix_lengths
    if labels is None or prefix_lengths is None:
        raise RuntimeError("compact baseline forward did not expose row prefix lengths/labels")
    actual = [int(value) for value in prefix_lengths.detach().cpu().tolist()]
    singles = [bool(value) for value in batch["single_audio_slot_mask"].detach().cpu().tolist()]
    expected = [AUDIO_SINGLE_PREFIX_TOKENS if single else AUDIO_DUAL_PREFIX_TOKENS for single in singles]
    labels_cpu = labels.detach().cpu()
    text_ids = batch["text_ids"].detach().cpu()
    prompt_lengths = batch["prompt_lengths"].detach().cpu()
    answer_lengths = batch["answer_lengths"].detach().cpu()
    eos = int(model.tokenizer.eos_token_id)
    valid = 0
    eos_ok = True
    alignment_ok = True
    for row, prefix in enumerate(actual):
        prompt = int(prompt_lengths[row].item())
        answer = int(answer_lengths[row].item())
        start = prefix + prompt
        end = start + answer
        expected_answer = text_ids[row, prompt:prompt + answer]
        alignment_ok = alignment_ok and torch.equal(labels_cpu[row, start:end], expected_answer)
        alignment_ok = alignment_ok and not bool((labels_cpu[row, :start] != -100).any()) and not bool((labels_cpu[row, end:] != -100).any())
        eos_ok = eos_ok and answer > 0 and int(labels_cpu[row, end - 1].item()) == eos
        valid += answer
    result = {
        "prefix_lengths": actual,
        "expected_prefix_lengths": expected,
        "compact_prefix_contract": actual == expected,
        "answer_label_alignment": alignment_ok,
        "answer_eos_contract": eos_ok,
        "supervised_tokens": int((labels_cpu != -100).sum().item()),
        "expected_supervised_tokens": valid,
    }
    if not result["compact_prefix_contract"] or not alignment_ok or not eos_ok or result["supervised_tokens"] != valid:
        raise RuntimeError(f"baseline compact-prefix/answer-EOS audit failed: {result}")
    return result


def _first_step_audit(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    gradient = model.runtime_gradient_audit()
    trainable = model.trainable_parameter_audit()
    sequence = _sequence_audit(model, batch)
    standard = trainable.get("standard_text_contract", {})
    result = {
        **gradient,
        **sequence,
        "standard_30_layer_smollm2": (
            standard.get("model_type") == "llama"
            and standard.get("num_hidden_layers") == 30
            and trainable.get("independent_decoder_layers") is True
            and trainable.get("has_router_parameters") is False
        ),
        "all_bridge_gradients_finite": bool(gradient.get("bridge_gradients")) and all(gradient["bridge_gradients"].values()),
        "all_c2l_gradients_finite": bool(gradient.get("c2l_gradients")) and all(gradient["c2l_gradients"].values()),
        "training_mode_contract": trainable.get("training_mode_contract") is True,
    }
    if not result["standard_30_layer_smollm2"] or not result["training_mode_contract"]:
        raise RuntimeError(f"baseline architecture/training audit failed: {result}")
    return result


def _checkpoint(
    path: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    args: argparse.Namespace,
    inventory: dict[str, Any],
    schedule: list[dict[str, int]],
    cursor: dict[str, int],
    rank: int,
    world: int,
    device: torch.device,
    plan_hash: str | None,
    parameter_change_audit: dict[str, Any] | None,
) -> None:
    rng_states = _gather(base._rng_state(device), world)
    if rank != 0:
        return
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent))
    published = False
    try:
        model.text_model.save_pretrained(temporary / "text_model", safe_serialization=False)
        tokenizer.save_pretrained(temporary / "tokenizer")
        torch.save(base._trainable_state(model), temporary / "audio_bridge.pt")
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scheduler_name": "cosine_lambda",
            "optimizer_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
            "training_contract": CONTRACT,
            "global_step": cursor["global_step"],
            "cursor": cursor,
            "plan_hash": plan_hash,
            "rng_states_by_rank": {str(index): state for index, state in enumerate(rng_states)},
        }, temporary / "training_state.pt")
        source_config = args.model_path.resolve() / "config.json"
        config = {
            "contract": CONTRACT,
            "architecture_contract": ORIGINAL_SMOLLM2_CONTRACT,
            "mapper_contract": MAPPER_CONTRACT,
            "compact_single_audio_prefix": True,
            "answer_termination": {"token": "<|endoftext|>", "included_in_max_answer_tokens": True, "supervised": True},
            "prefix_tokens": {"single": AUDIO_SINGLE_PREFIX_TOKENS, "dual": AUDIO_DUAL_PREFIX_TOKENS},
            "standard_text_contract": model.text_contract,
            "text_model_source_path": str(args.model_path.resolve()),
            "text_model_source_config_sha256": _sha(source_config) if source_config.is_file() else None,
            "mode": args.mode,
            "inventory": inventory,
            "schedule": schedule,
            "epochs": args.epochs,
            "world_size": world,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed,
            "max_lr": args.max_lr,
            "min_lr": args.min_lr,
            "optimizer": "AdamW",
            "optimizer_betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "gradient_clip_norm": 0.5,
            "scheduler": "cosine_lambda",
            "warmup_steps": args.warmup_steps,
            "total_steps": sum(item["steps"] for item in schedule),
            "effective_global_batch_size": world * args.micro_batch_size * args.gradient_accumulation_steps,
            "periodic_validation": False,
            "frozen_audio_encoder": True,
            "save_every": args.save_every,
            "checkpoint_retention": args.checkpoint_retention,
            "dist_timeout_minutes": args.dist_timeout_minutes,
            "release_min_fraction": args.release_min_fraction,
            "release_timeout_seconds": args.release_timeout_seconds,
            "htsat_checkpoint": str(args.htsat_checkpoint.resolve()),
            "mellow_root": str(args.mellow_root.resolve()),
            "mellow_provenance": model._audio_provenance,
            "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
            "resume_parameter_change_audit": parameter_change_audit,
        }
        (temporary / CONFIG_FILENAME).write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
        required = [
            "text_model/config.json",
            "tokenizer/tokenizer_config.json",
            "audio_bridge.pt",
            "training_state.pt",
            CONFIG_FILENAME,
            "checkpoint_complete.json",
        ]
        marker = {"status": "complete", "global_step": cursor["global_step"], "contract": CONTRACT, "required": required}
        (temporary / "checkpoint_complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        if any(not (temporary / name).is_file() for name in required):
            raise RuntimeError("partition baseline checkpoint is incomplete")
        if not base._text_model_weight_files(temporary / "text_model"):
            raise RuntimeError("partition baseline checkpoint has no text-model weights")
        temporary.replace(path)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def _resume(
    path: Path,
    args: argparse.Namespace,
    inventory: dict[str, Any],
    schedule: list[dict[str, int]],
    optimizer: Any,
    scheduler: Any,
    rank: int,
    device: torch.device,
    provenance: dict[str, Any],
    model: Any,
) -> tuple[dict[str, int], str | None]:
    config = json.loads((path / CONFIG_FILENAME).read_text(encoding="utf-8"))
    marker = json.loads((path / "checkpoint_complete.json").read_text(encoding="utf-8"))
    required = [
        "text_model/config.json",
        "tokenizer/tokenizer_config.json",
        "audio_bridge.pt",
        "training_state.pt",
        CONFIG_FILENAME,
        "checkpoint_complete.json",
    ]
    if config.get("contract") != CONTRACT or marker.get("status") != "complete" or marker.get("contract") != CONTRACT:
        raise RuntimeError("not a complete compatible SmolLM2 partition checkpoint")
    if marker.get("required") != required or any(not (path / name).is_file() for name in required):
        raise RuntimeError("SmolLM2 partition checkpoint required-file contract mismatch")
    if not base._text_model_weight_files(path / "text_model"):
        raise RuntimeError("SmolLM2 partition checkpoint has no text-model weights")
    if (
        config.get("architecture_contract") != ORIGINAL_SMOLLM2_CONTRACT
        or config.get("mapper_contract") != MAPPER_CONTRACT
        or config.get("compact_single_audio_prefix") is not True
        or config.get("prefix_tokens") != {"single": 130, "dual": 260}
        or config.get("answer_termination") != {"token": "<|endoftext|>", "included_in_max_answer_tokens": True, "supervised": True}
    ):
        raise RuntimeError("resume architecture/prefix/EOS contract mismatch")
    standard = config.get("standard_text_contract", {})
    if standard.get("model_type") != "llama" or standard.get("num_hidden_layers") != 30 or standard.get("physical_decoder_layer_count") != 30:
        raise RuntimeError("resume checkpoint is not the standard 30-layer SmolLM2 baseline")
    if config.get("mellow_provenance", {}).get("mellow_htsat_sha256") != provenance.get("mellow_htsat_sha256"):
        raise RuntimeError("resume Mellow implementation SHA256 mismatch")
    source_config = args.model_path.resolve() / "config.json"
    if Path(config["text_model_source_path"]).resolve() != args.model_path.resolve():
        raise RuntimeError("resume original SmolLM2 source path mismatch")
    if not source_config.is_file() or config.get("text_model_source_config_sha256") != _sha(source_config):
        raise RuntimeError("resume original SmolLM2 source config SHA256 mismatch")
    expected_values = {
        "mode": args.mode,
        "inventory": inventory,
        "schedule": schedule,
        "epochs": args.epochs,
        "world_size": args.world_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "max_lr": args.max_lr,
        "min_lr": args.min_lr,
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "gradient_clip_norm": 0.5,
        "scheduler": "cosine_lambda",
        "warmup_steps": args.warmup_steps,
        "total_steps": sum(item["steps"] for item in schedule),
        "effective_global_batch_size": args.world_size * args.micro_batch_size * args.gradient_accumulation_steps,
        "periodic_validation": False,
        "frozen_audio_encoder": True,
        "save_every": args.save_every,
        "checkpoint_retention": args.checkpoint_retention,
        "dist_timeout_minutes": args.dist_timeout_minutes,
        "release_min_fraction": args.release_min_fraction,
        "release_timeout_seconds": args.release_timeout_seconds,
        "htsat_checkpoint": str(args.htsat_checkpoint.resolve()),
        "mellow_root": str(args.mellow_root.resolve()),
    }
    for key, expected in expected_values.items():
        if config.get(key) != expected:
            raise RuntimeError(f"resume contract differs in {key}")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    if state.get("training_contract") != CONTRACT or state.get("scheduler_name") != "cosine_lambda":
        raise RuntimeError("resume training-state contract/scheduler mismatch")
    base._validate_optimizer_coverage(state, model)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    cursor = state["cursor"]
    if marker["global_step"] != cursor["global_step"] or cursor["global_step"] != sum(item["steps"] for item in schedule[:cursor["segment"]]) + cursor["segment_step"]:
        raise RuntimeError("checkpoint cursor/global step mismatch")
    rng_states = state["rng_states_by_rank"]
    if set(rng_states) != {str(index) for index in range(args.world_size)}:
        raise RuntimeError("checkpoint per-rank RNG coverage mismatch")
    base._restore_rng_state(rng_states[str(rank)], device)
    return cursor, state.get("plan_hash")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
    if args.resume_from is not None:
        resume_path = args.resume_from.resolve()
        output_path = args.output_dir.resolve()
        if args.tokenizer_path is not None:
            raise ValueError("--tokenizer-path cannot be combined with --resume-from; resume must use the checkpoint tokenizer")
        if output_path in {resume_path, resume_path.parent} or resume_path in output_path.parents:
            raise ValueError("resume output-dir must be separate from the source checkpoint and its parent")
    if not torch.cuda.is_available() or world != args.world_size or args.world_size != 8 or args.micro_batch_size != 8 or args.gradient_accumulation_steps != 4:
        raise RuntimeError("partition baseline training requires exactly 8 GPUs, microbatch 8, GA 4")
    if (
        args.epochs != 10
        or args.max_lr != 1e-3
        or args.min_lr != 0.0
        or args.save_every != 500
        or args.checkpoint_retention != 4
        or args.seed != 0
    ):
        raise ValueError("partition baseline requires epochs=10, max_lr=1e-3, min_lr=0, save_every=500, retention=4, seed=0")
    if args.epochs <= 0 or args.dist_timeout_minutes < 30 or args.save_every <= 0 or args.checkpoint_retention <= 0 or not 0 < args.release_min_fraction <= 1 or args.release_timeout_seconds <= 0:
        raise ValueError("invalid epochs, timeout, checkpoint or release threshold")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(minutes=args.dist_timeout_minutes))
    base._seed(args.seed, rank)
    report: dict[str, Any] = {
        "status": "FAIL",
        "mode": args.mode,
        "training_contract": CONTRACT,
        "answer_termination": {"token": "<|endoftext|>", "included_in_max_answer_tokens": True, "supervised": True},
        "rank": rank,
        "segments": [],
        "checkpoints": [],
        "hard_failures": [],
    }
    output_available = not args.output_dir.exists() or not any(args.output_dir.iterdir())
    try:
        if not output_available:
            raise FileExistsError(f"refusing nonempty output directory: {args.output_dir}")
        inventory = _inventory(args.partition_store_root)
        smoke_gate = _formal_smoke_gate(args, inventory) if args.mode == "formal" else None
        schedule = _schedule(args, inventory)
        total_steps = sum(item["steps"] for item in schedule)
        if args.mode == "formal" and total_steps != (968059 // 256) * args.epochs:
            raise RuntimeError("formal step budget differs from global epoch floor")
        args.warmup_steps = args.warmup_steps if args.warmup_steps is not None else math.ceil(total_steps * 0.05)
        if args.warmup_steps != math.ceil(total_steps * 0.05):
            raise ValueError("warmup must equal ceil(5% of total optimizer steps)")
        args.compact_single_audio_prefix = True
        model, tokenizer = base._load_model(args, device)
        model.train()
        if not model.trainable_parameter_audit()["training_mode_contract"]:
            raise RuntimeError("trainable parameter audit failed")
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.max_lr, betas=(0.9, 0.95), weight_decay=0.1)
        scheduler = base._make_scheduler(optimizer, max_lr=args.max_lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps, total_steps=total_steps)
        cursor = {"segment": 0, "segment_step": 0, "global_step": 0}
        saved_plan_hash = None
        if args.resume_from:
            cursor, saved_plan_hash = _resume(args.resume_from, args, inventory, schedule, optimizer, scheduler, rank, device, model._audio_provenance, model)
            if args.mode == "smoke" and cursor != {"segment": 2, "segment_step": 0, "global_step": 20}:
                raise RuntimeError("smoke resume requires released step-20 boundary checkpoint")
            if args.expected_resume_step is not None and cursor["global_step"] != args.expected_resume_step:
                raise RuntimeError(f"resume parent step {cursor['global_step']} != expected {args.expected_resume_step}")
        elif args.expected_resume_step is not None:
            raise ValueError("--expected-resume-step requires --resume-from")
        elif args.mode == "smoke" and total_steps != 22:
            raise AssertionError("smoke schedule must total 22 steps")
        resume_representatives = base._select_resume_representatives(model) if args.resume_from else None
        resume_snapshots = base._snapshot_resume_representatives(resume_representatives) if resume_representatives else None
        resume_parameter_change_audit = None
        ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        global_batch = world * args.micro_batch_size * args.gradient_accumulation_steps
        report.update({
            "inventory": inventory,
            "schedule": schedule,
            "total_steps": total_steps,
            "warmup_steps": args.warmup_steps,
            "optimizer": "AdamW",
            "optimizer_betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "gradient_clip_norm": 0.5,
            "scheduler": "cosine_lambda",
            "effective_global_batch_size": global_batch,
            "periodic_validation": False,
            "frozen_audio_encoder": True,
            "start_cursor": dict(cursor),
            "prefix_contract": {"single": 130, "dual": 260},
            "seed": args.seed,
            "resume_checkpoint": str(args.resume_from.resolve()) if args.resume_from else None,
            "smoke_gate": smoke_gate,
            "standard_text_contract": ddp.module.text_contract,
        })
        first_audit = None
        stop_step = 20 if args.mode == "smoke" and not args.resume_from else total_steps
        for segment_index in range(cursor["segment"], len(schedule)):
            segment = schedule[segment_index]
            if cursor["global_step"] >= stop_step:
                break
            partition_id = segment["partition_id"]
            partition = inventory["partitions"][partition_id]
            partition_path = Path(inventory["root"]) / f"partition_{partition_id}"
            memory_before = _memory()
            preload_started = time.perf_counter()
            dataset = ReasonAQADataset(partition_path / "rows.jsonl", tokenizer, unique_waveform_store_dir=partition_path)
            if len(dataset) != partition["rows"]:
                raise RuntimeError("partition QA cardinality changed")
            cache = _RankLocalStoreWaveforms(dataset)
            for audio_id in range(partition["audio"]):
                cache._get_audio_id(audio_id)
            if cache.current_bytes != partition["bytes"] or cache.misses != partition["audio"]:
                raise RuntimeError("partition preload cardinality/bytes mismatch")
            chunks, plan_audit = _plan(dataset, seed=args.seed, epoch=segment["epoch"], pid=partition_id, steps=segment["steps"], global_batch=global_batch)
            if segment_index == cursor["segment"] and cursor["segment_step"] and saved_plan_hash != plan_audit["plan_sha256"]:
                raise RuntimeError("resume optimizer-window plan hash mismatch")
            dist.barrier()
            memory_after_load = _memory()
            segment_report: dict[str, Any] = {
                "segment": segment,
                "plan": plan_audit,
                "preload_seconds": time.perf_counter() - preload_started,
                "memory_before": memory_before,
                "memory_after_load": memory_after_load,
                "steps": [],
            }
            if rank == 0:
                print(f"[smollm2-partition] loaded p{partition_id} epoch={segment['epoch']} steps={segment['steps']} bytes/rank={partition['bytes']} seconds={segment_report['preload_seconds']:.1f}", flush=True)
            initial_stats = cache.stats()
            start_at = cursor["segment_step"] if segment_index == cursor["segment"] else 0
            for local_step in range(start_at, segment["steps"]):
                if cursor["global_step"] >= stop_step:
                    break
                step_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                step_loss_sum = torch.zeros((), dtype=torch.float32, device=device)
                chunk = chunks[local_step]
                rank_rows = chunk[rank * 32:(rank + 1) * 32]
                if len(rank_rows) != 32:
                    raise AssertionError("rank optimizer window is not 32 rows")
                slot_kind = {dataset.audio_structure(index)[0] for index in chunk}
                for micro in range(4):
                    indices = rank_rows[micro * 8:(micro + 1) * 8]
                    batch = collate_reasonaqa([cache.materialize_from_cached_metadata(index) for index in indices], tokenizer)
                    device_keys = {"audio1", "audio2", "text_ids"}
                    batch = {key: (value.to(device) if torch.is_tensor(value) and key in device_keys else value) for key, value in batch.items()}
                    with (ddp.no_sync() if micro != 3 else contextlib.nullcontext()):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            output = ddp(**{key: value for key, value in batch.items() if key not in {"row_indices", "audio2_reused"}})
                        if output.loss is None or not bool(torch.isfinite(output.loss)):
                            raise RuntimeError("nonfinite SmolLM2 partition training loss")
                        step_loss_sum.add_(output.loss.detach().float())
                        (output.loss / 4).backward()
                    if first_audit is None and micro == 3:
                        first_audit = _first_step_audit(ddp.module, batch)
                    del batch, output
                if resume_representatives is not None and resume_parameter_change_audit is None:
                    base._verify_resume_representative_gradients(resume_representatives)
                torch.nn.utils.clip_grad_norm_(ddp.parameters(), 0.5, error_if_nonfinite=True)
                lr_used = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
                if resume_representatives is not None and resume_snapshots is not None and resume_parameter_change_audit is None:
                    resume_parameter_change_audit = base._compute_resume_parameter_change_audit(resume_representatives, resume_snapshots)
                    base._validate_parameter_change_audit(resume_parameter_change_audit)
                cursor = {"segment": segment_index, "segment_step": local_step + 1, "global_step": cursor["global_step"] + 1}
                torch.cuda.synchronize(device)
                step_loss = float((step_loss_sum / args.gradient_accumulation_steps).item())
                segment_report["steps"].append({
                    "global_step": cursor["global_step"],
                    "partition_step": local_step + 1,
                    "slot": "mixed" if len(slot_kind) == 2 else "single" if True in slot_kind else "dual",
                    "loss": step_loss,
                    "lr": lr_used,
                    "seconds": time.perf_counter() - step_started,
                })
                at_boundary = local_step + 1 == segment["steps"]
                if rank == 0 and (cursor["global_step"] % 10 == 0 or at_boundary):
                    print(f"[smollm2-partition] step={cursor['global_step']}/{total_steps} p{partition_id} {local_step+1}/{segment['steps']} loss={step_loss:.6f} lr={lr_used:.8g} {segment_report['steps'][-1]['seconds']:.3f}s", flush=True)
                if args.mode == "formal" and cursor["global_step"] % args.save_every == 0 and not at_boundary:
                    checkpoint = args.output_dir / f"checkpoint-{cursor['global_step']:06d}"
                    _checkpoint(checkpoint, ddp.module, tokenizer, optimizer, scheduler, args, inventory, schedule, cursor, rank, world, device, plan_audit["plan_sha256"], resume_parameter_change_audit)
                    if rank == 0:
                        report["checkpoints"].append(str(checkpoint))
                        report["retained_checkpoints"] = base._prune_checkpoints(args.output_dir, args.checkpoint_retention)
                    dist.barrier()
            after_stats = cache.stats()
            if after_stats["waveform_misses"] != initial_stats["waveform_misses"] or after_stats["cloned_bytes"] != initial_stats["cloned_bytes"] or after_stats["resident_unique_audio"] != partition["audio"]:
                raise RuntimeError("training accessed store or changed complete rank-RAM residency")
            torch.cuda.synchronize(device)
            del chunks, cache.items
            dataset.unique_waveform_store.close()
            del cache, dataset
            gc.collect()
            _trim()
            dist.barrier()
            threshold = int(partition["bytes"] * args.release_min_fraction)
            cgroup_required = int(partition["bytes"] * world * args.release_min_fraction)
            deadline = time.monotonic() + args.release_timeout_seconds
            polls = 0
            while True:
                memory_after_release = _memory()
                polls += 1
                current_anon = _anon(memory_after_release)
                loaded_anon = _anon(memory_after_load)
                rss_ok = memory_after_load["rss_bytes"] - memory_after_release["rss_bytes"] >= threshold
                cgroup_ok = current_anon is not None and loaded_anon is not None and loaded_anon - current_anon >= cgroup_required
                if (rss_ok and cgroup_ok) or time.monotonic() >= deadline:
                    break
                time.sleep(2.0)
            released = memory_after_load["rss_bytes"] - memory_after_release["rss_bytes"]
            local_release = {
                "rank": rank,
                "rss_drop_bytes": released,
                "required_drop_bytes": threshold,
                "before": memory_before,
                "after_load": memory_after_load,
                "after_release": memory_after_release,
                "polls": polls,
                "passed": released >= threshold,
            }
            releases = _gather(local_release, world)
            segment_report["release"] = releases
            loaded_anon = _anon(memory_after_load)
            released_anon = _anon(memory_after_release)
            cgroup_drop = None if loaded_anon is None or released_anon is None else loaded_anon - released_anon
            segment_report["cgroup_release"] = {
                "anon_drop_bytes": cgroup_drop,
                "required_bytes": cgroup_required,
                "available": cgroup_drop is not None,
                "timeout_seconds": args.release_timeout_seconds,
            }
            if cgroup_drop is None:
                raise RuntimeError(f"partition p{partition_id} cannot audit cgroup anon release; refusing next preload")
            if not all(item["passed"] for item in releases):
                raise RuntimeError(f"partition p{partition_id} RAM release failed")
            if cgroup_drop < cgroup_required:
                raise RuntimeError(f"partition p{partition_id} cgroup anon failed to fall: {cgroup_drop} < {cgroup_required}")
            report["segments"].append(segment_report)
            cursor = {"segment": segment_index + 1, "segment_step": 0, "global_step": cursor["global_step"]}
            at_end = cursor["global_step"] == total_steps
            save_boundary = (args.mode == "smoke" and cursor["global_step"] in (20, 22)) or (args.mode == "formal" and (cursor["global_step"] % args.save_every == 0 or at_end))
            if save_boundary:
                checkpoint = args.output_dir / f"checkpoint-{cursor['global_step']:06d}"
                _checkpoint(checkpoint, ddp.module, tokenizer, optimizer, scheduler, args, inventory, schedule, cursor, rank, world, device, None, resume_parameter_change_audit)
                if rank == 0:
                    report["checkpoints"].append(str(checkpoint))
                    report["retained_checkpoints"] = base._prune_checkpoints(args.output_dir, args.checkpoint_retention)
                dist.barrier()
            if rank == 0:
                print(f"[smollm2-partition] released p{partition_id} RSS drop/rank={[round(item['rss_drop_bytes']/1024**3, 2) for item in releases]} GiB", flush=True)
        if cursor["global_step"] != stop_step:
            raise RuntimeError(f"run ended at step {cursor['global_step']} rather than {stop_step}")
        report.update({
            "status": "PASS",
            "end_cursor": cursor,
            "first_step_gradient_audit": first_audit,
            "resume_verified_two_steps": args.mode == "smoke" and args.resume_from is not None and cursor["global_step"] == 22,
            "resume_parameter_change_audit": resume_parameter_change_audit,
        })
        return report
    except Exception as exc:
        report["hard_failures"].append({"error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        if rank == 0 and output_available:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "partition_training_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run(parse_args())
