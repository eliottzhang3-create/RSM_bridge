#!/usr/bin/env python3
"""Stage audited ReasonAQA raw audio into a job-local /dev/shm tree.

This staging-only program consumes the PASS report and JSONL mapping produced
by audit_reasonaqa_paths.py. It copies every referenced source file to its
original logical path below one run-unique /dev/shm directory, then verifies
all destination sizes and decodes a small deterministic sample with
torchaudio. It never launches a model, CUDA work, or torchrun.

The upstream Mellow dataset can use the staging root as data_path without any
path-rewriting change to the official dataset implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


CONTRACT = "reasonaqa_raw_audio_tmpfs_staging_probe_v1"
AUDIT_CONTRACT = "reasonaqa_raw_audio_path_mapping_v1"
CONTROL_DIR_NAME = ".mellow_stage"
STAGE_PREFIX = "mellow_official_reasonaqa_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--mapping-jsonl", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--free-space-margin-gib", type=float, default=10.0)
    parser.add_argument("--decode-samples-per-group", type=int, default=3)
    parser.add_argument("--copy-batch-size", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def resolved_existing_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def resolve_stage_root(path: Path) -> tuple[Path, Path]:
    shm_root = Path("/dev/shm").resolve(strict=True)
    if not shm_root.is_dir():
        raise ValueError(f"/dev/shm is not a directory: {shm_root}")
    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"stage root must be absolute: {candidate}")
    parent = candidate.parent.resolve(strict=True)
    resolved = parent / candidate.name
    if not resolved.is_relative_to(shm_root):
        raise ValueError(f"stage root must be below /dev/shm: {resolved}")
    if not resolved.name.startswith(STAGE_PREFIX):
        raise ValueError(
            f"stage root basename must start with {STAGE_PREFIX!r}: {resolved.name!r}"
        )
    if resolved.exists() or resolved.is_symlink():
        raise ValueError(f"refusing pre-existing stage root: {resolved}")
    return shm_root, resolved


def logical_parts(value: Any) -> tuple[str, ...]:
    logical = str(value or "")
    if not logical or "\\" in logical:
        raise ValueError(f"invalid logical path: {logical!r}")
    path = PurePosixPath(logical)
    parts = path.parts
    if path.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe logical path: {logical!r}")
    if parts[0] == CONTROL_DIR_NAME:
        raise ValueError(f"logical path collides with staging control directory: {logical!r}")
    return tuple(parts)


def load_mapping(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_logical: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid mapping JSON at line {line_number}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"mapping line {line_number} is not an object")
            logical = str(raw.get("logical_path", ""))
            parts = logical_parts(logical)
            if logical in seen_logical:
                raise ValueError(f"duplicate logical path in mapping: {logical!r}")
            seen_logical.add(logical)
            source = resolved_existing_file(
                Path(str(raw.get("source_path", ""))), "source audio"
            )
            expected_size = int(raw.get("source_size_bytes", -1))
            actual_size = source.stat().st_size
            if expected_size <= 0 or actual_size != expected_size:
                raise ValueError(
                    f"source size mismatch for {logical!r}: "
                    f"audit={expected_size} current={actual_size} source={source}"
                )
            entries.append(
                {
                    "logical_path": logical,
                    "logical_parts": parts,
                    "source_path": source,
                    "source_size_bytes": actual_size,
                    "group": str(raw.get("group", "unknown")),
                    "method": str(raw.get("method", "unknown")),
                }
            )
    if not entries:
        raise ValueError(f"mapping is empty: {path}")
    return entries


def validate_audit_mapping(
    audit: dict[str, Any], audit_path: Path, mapping_path: Path, entries: list[dict[str, Any]]
) -> Path:
    if audit.get("status") != "PASS":
        raise ValueError(f"path audit is not PASS: {audit_path}")
    if audit.get("contract") != AUDIT_CONTRACT:
        raise ValueError(
            f"unexpected audit contract: {audit.get('contract')!r}; expected {AUDIT_CONTRACT!r}"
        )
    recorded_mapping = Path(str(audit.get("mapping_path", ""))).expanduser().resolve(strict=True)
    same_mapping_file = recorded_mapping == mapping_path
    if not same_mapping_file:
        try:
            same_mapping_file = recorded_mapping.samefile(mapping_path)
        except OSError:
            same_mapping_file = False
    if not same_mapping_file:
        raise ValueError(
            f"mapping/report pairing mismatch: report={recorded_mapping} argument={mapping_path}"
        )
    expected_logical = int(audit.get("unique_logical_paths_resolved", -1))
    if expected_logical != len(entries):
        raise ValueError(
            f"logical file count mismatch: audit={expected_logical} mapping={len(entries)}"
        )
    unique_sources = {
        str(entry["source_path"]): int(entry["source_size_bytes"]) for entry in entries
    }
    expected_sources = int(audit.get("unique_source_files", -1))
    if expected_sources != len(unique_sources):
        raise ValueError(
            f"unique source count mismatch: audit={expected_sources} mapping={len(unique_sources)}"
        )
    expected_source_bytes = int(audit.get("total_source_bytes", -1))
    actual_source_bytes = sum(unique_sources.values())
    if expected_source_bytes != actual_source_bytes:
        raise ValueError(
            f"unique source byte mismatch: audit={expected_source_bytes} mapping={actual_source_bytes}"
        )
    return resolved_existing_file(
        Path(str(audit.get("train_json", ""))), "ReasonAQA train JSON"
    )


def copy_one(entry: dict[str, Any], stage_root: Path) -> int:
    source = entry["source_path"]
    destination = stage_root.joinpath(*entry["logical_parts"])
    temporary = destination.with_name(
        f".{destination.name}.partial-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        shutil.copyfile(source, temporary)
        copied_size = temporary.stat().st_size
        expected_size = int(entry["source_size_bytes"])
        if copied_size != expected_size:
            raise OSError(
                f"copied size mismatch for {entry['logical_path']!r}: "
                f"expected={expected_size} actual={copied_size}"
            )
        os.replace(temporary, destination)
        return copied_size
    finally:
        if temporary.exists():
            temporary.unlink()


def stage_audio(
    entries: list[dict[str, Any]],
    stage_root: Path,
    workers: int,
    batch_size: int,
    progress_every: int,
) -> tuple[int, float]:
    parent_directories = {
        stage_root.joinpath(*entry["logical_parts"][:-1]) for entry in entries
    }
    for directory in sorted(parent_directories, key=str):
        directory.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    copied_bytes = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mellow-stage") as pool:
        for batch_start in range(0, len(entries), batch_size):
            batch = entries[batch_start : batch_start + batch_size]
            futures = {pool.submit(copy_one, entry, stage_root): entry for entry in batch}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    copied_bytes += future.result()
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(
                        f"copy failed for logical={entry['logical_path']!r} "
                        f"source={entry['source_path']}"
                    ) from exc
                copied_files += 1
                if progress_every > 0 and (
                    copied_files % progress_every == 0 or copied_files == len(entries)
                ):
                    elapsed = max(time.monotonic() - started, 1e-9)
                    gib = copied_bytes / (1024**3)
                    print(
                        f"[mellow-stage] files={copied_files}/{len(entries)} "
                        f"copied_gib={gib:.3f} "
                        f"throughput_mib_s={copied_bytes / elapsed / (1024**2):.2f}",
                        flush=True,
                    )
    return copied_bytes, time.monotonic() - started


def evenly_spaced(items: list[dict[str, Any]], count: int) -> Iterable[dict[str, Any]]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return items
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in dict.fromkeys(indices)]


def decode_samples(
    entries: list[dict[str, Any]], stage_root: Path, samples_per_group: int
) -> list[dict[str, Any]]:
    import torch
    import torchaudio

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["group"]].append(entry)
    selected: list[dict[str, Any]] = []
    for group in sorted(grouped):
        ordered = sorted(grouped[group], key=lambda item: item["logical_path"])
        selected.extend(evenly_spaced(ordered, samples_per_group))

    results: list[dict[str, Any]] = []
    for entry in selected:
        staged_path = stage_root.joinpath(*entry["logical_parts"])
        waveform, sample_rate = torchaudio.load(str(staged_path), channels_first=True)
        if waveform.numel() <= 0 or waveform.ndim != 2:
            raise ValueError(
                f"invalid decoded waveform shape for {staged_path}: {tuple(waveform.shape)}"
            )
        if sample_rate <= 0:
            raise ValueError(f"invalid decoded sample rate for {staged_path}: {sample_rate}")
        if not bool(torch.isfinite(waveform).all().item()):
            raise ValueError(f"non-finite decoded samples in {staged_path}")
        result = {
            "logical_path": entry["logical_path"],
            "group": entry["group"],
            "sample_rate": int(sample_rate),
            "channels": int(waveform.shape[0]),
            "frames": int(waveform.shape[1]),
            "size_bytes": int(staged_path.stat().st_size),
        }
        results.append(result)
        print(
            f"[mellow-stage-decode] group={result['group']} rate={result['sample_rate']} "
            f"shape=({result['channels']},{result['frames']}) "
            f"logical={result['logical_path']}",
            flush=True,
        )
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.copy_batch_size < args.workers:
        raise ValueError("--copy-batch-size must be at least --workers")
    if args.decode_samples_per_group < 1:
        raise ValueError("--decode-samples-per-group must be positive")
    if args.free_space_margin_gib < 0:
        raise ValueError("--free-space-margin-gib cannot be negative")

    audit_path = resolved_existing_file(args.audit_report, "path audit report")
    mapping_path = resolved_existing_file(args.mapping_jsonl, "path mapping JSONL")
    shm_root, stage_root = resolve_stage_root(args.stage_root)
    report_path = args.report_path.expanduser().resolve(strict=False)
    if report_path == stage_root or report_path.is_relative_to(stage_root):
        raise ValueError("persistent report path must be outside the ephemeral stage root")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict):
        raise ValueError("path audit report must be a JSON object")
    entries = load_mapping(mapping_path)
    train_json = validate_audit_mapping(audit, audit_path, mapping_path, entries)

    staged_payload_bytes = sum(int(entry["source_size_bytes"]) for entry in entries)
    control_bytes = (
        audit_path.stat().st_size + mapping_path.stat().st_size + train_json.stat().st_size
    )
    margin_bytes = int(args.free_space_margin_gib * (1024**3))
    required_bytes = staged_payload_bytes + control_bytes + margin_bytes
    disk_before = shutil.disk_usage(shm_root)
    if disk_before.free < required_bytes:
        raise ValueError(
            "insufficient /dev/shm capacity: "
            f"free_gib={disk_before.free / (1024**3):.3f} "
            f"required_gib={required_bytes / (1024**3):.3f} "
            f"payload_gib={staged_payload_bytes / (1024**3):.3f} "
            f"margin_gib={args.free_space_margin_gib:.3f}"
        )

    started_wall = time.time()
    started_monotonic = time.monotonic()
    stage_root.mkdir(mode=0o700)
    control_dir = stage_root / CONTROL_DIR_NAME
    control_dir.mkdir(mode=0o700)
    building_path = control_dir / "BUILDING.json"
    write_json(
        building_path,
        {
            "status": "BUILDING",
            "contract": CONTRACT,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "started_unix": started_wall,
        },
    )

    staged_train_json = control_dir / "reasonaqa_train.json"
    shutil.copyfile(train_json, staged_train_json)
    shutil.copyfile(audit_path, control_dir / "path_audit_report.json")
    shutil.copyfile(mapping_path, control_dir / "train_audio_mapping.jsonl")

    copied_bytes, copy_seconds = stage_audio(
        entries,
        stage_root,
        workers=args.workers,
        batch_size=args.copy_batch_size,
        progress_every=args.progress_every,
    )
    if copied_bytes != staged_payload_bytes:
        raise ValueError(
            f"staged payload byte mismatch: expected={staged_payload_bytes} actual={copied_bytes}"
        )

    for entry in entries:
        staged = stage_root.joinpath(*entry["logical_parts"])
        actual_size = staged.stat().st_size
        if actual_size != entry["source_size_bytes"]:
            raise ValueError(
                f"post-copy size mismatch for {entry['logical_path']!r}: "
                f"expected={entry['source_size_bytes']} actual={actual_size}"
            )

    decode_results = decode_samples(entries, stage_root, args.decode_samples_per_group)
    disk_after = shutil.disk_usage(shm_root)
    elapsed = time.monotonic() - started_monotonic
    report = {
        "status": "PASS",
        "contract": CONTRACT,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "stage_root": str(stage_root),
        "staged_train_json": str(staged_train_json),
        "audit_report": str(audit_path),
        "mapping_jsonl": str(mapping_path),
        "source_train_json": str(train_json),
        "workers": args.workers,
        "copy_batch_size": args.copy_batch_size,
        "file_count": len(entries),
        "group_counts": dict(Counter(entry["group"] for entry in entries)),
        "method_counts": dict(Counter(entry["method"] for entry in entries)),
        "staged_payload_bytes": staged_payload_bytes,
        "staged_payload_gib": staged_payload_bytes / (1024**3),
        "control_bytes": control_bytes,
        "free_space_margin_bytes": margin_bytes,
        "shm_free_bytes_before": disk_before.free,
        "shm_free_bytes_after": disk_after.free,
        "copy_seconds": copy_seconds,
        "total_seconds": elapsed,
        "copy_throughput_mib_s": copied_bytes / max(copy_seconds, 1e-9) / (1024**2),
        "decode_samples_per_group": args.decode_samples_per_group,
        "decode_results": decode_results,
        "verification": {
            "full_file_count_checked": True,
            "full_file_size_checked": True,
            "sample_decode_checked": True,
            "sha256_used": False,
        },
        "started_unix": started_wall,
        "completed_unix": time.time(),
    }
    write_json(report_path, report)
    write_json(control_dir / "READY.json", report)
    building_path.unlink()
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return report


def main() -> int:
    args = parse_args()
    report_path = args.report_path.expanduser().resolve(strict=False)
    try:
        run(args)
        return 0
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "contract": CONTRACT,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "audit_report": str(args.audit_report),
            "mapping_jsonl": str(args.mapping_jsonl),
            "stage_root": str(args.stage_root),
            "failed_unix": time.time(),
        }
        try:
            write_json(report_path, failure)
        except Exception as report_exc:
            print(f"failed to write failure report: {report_exc}", file=sys.stderr, flush=True)
        print(json.dumps(failure, indent=2, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
