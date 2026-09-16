#!/usr/bin/env python3
"""Build a deterministic, deduplicated fixed-waveform shard cache on CPU.

The output is intentionally a simple raw-binary format: each shard contains
contiguous float32 waveforms with shape ``[rows, 320000]`` and no per-file
headers.  ``index.jsonl`` maps each canonical source path to a shard row and
byte offset.  The source list is globally shuffled once by ``--seed`` before
contiguous shard assignment; epoch-level shard/sample shuffling belongs to the
future Dataset/Sampler reader and never changes the physical shard contents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from audio_5_10x2_5_mesh_mellow.data import load_waveform  # noqa: E402


SAMPLE_RATE = 32_000
SECONDS = 10
SAMPLES_PER_AUDIO = SAMPLE_RATE * SECONDS
BYTES_PER_AUDIO = SAMPLES_PER_AUDIO * 4
AUDIO_SUFFIXES = frozenset({".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"})
DEFAULT_AUDIOCAPS_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2/train"
DEFAULT_CLOTHO_AQA_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_audio/audio_files"
DEFAULT_CLOTHO_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2_1/development"


@dataclass(frozen=True)
class AudioSource:
    source_path: str
    source_group: str
    relative_path: str
    source_size_bytes: int
    source_mtime_ns: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audiocaps-root", type=Path, default=Path(DEFAULT_AUDIOCAPS_ROOT))
    parser.add_argument("--clotho-aqa-root", type=Path, default=Path(DEFAULT_CLOTHO_AQA_ROOT))
    parser.add_argument("--clotho-root", type=Path, default=Path(DEFAULT_CLOTHO_ROOT))
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--workers", type=int, default=1, help="CPU waveform conversion workers; ordered output remains deterministic")
    parser.add_argument("--torch-threads-per-worker", type=int, default=1)
    parser.add_argument("--free-space-margin-gib", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true", help="resume a matching interrupted build, rebuilding at most the partial shard")
    parser.add_argument("--dry-run", action="store_true", help="scan, deduplicate, shuffle, and report size without reading/writing waveforms")
    return parser.parse_args(argv)


def _canonical(path: Path) -> Path:
    return path.expanduser().resolve(strict=True)


def _scan_root(root: Path, group: str) -> list[AudioSource]:
    root = _canonical(root)
    if not root.is_dir():
        raise FileNotFoundError(f"audio root is not a directory: {root}")
    rows: list[AudioSource] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            relative_path = str(path.relative_to(root))
            canonical = _canonical(path)
            stat = canonical.stat()
            rows.append(AudioSource(str(canonical), group, relative_path, int(stat.st_size), int(stat.st_mtime_ns)))
    return rows


def collect_sources(roots: list[tuple[str, Path]]) -> tuple[list[AudioSource], dict[str, Any]]:
    by_path: dict[str, AudioSource] = {}
    root_report: dict[str, Any] = {}
    duplicate_count = 0
    for group, root in roots:
        scanned = _scan_root(root, group)
        root_report[group] = {"root": str(_canonical(root)), "files_scanned": len(scanned)}
        for source in scanned:
            if source.source_path in by_path:
                duplicate_count += 1
                continue
            by_path[source.source_path] = source
    sources = list(by_path.values())
    root_report["unique_files"] = len(sources)
    root_report["duplicate_path_entries_removed"] = duplicate_count
    return sources, root_report


def _worker_init(torch_threads: int) -> None:
    torch.set_num_threads(max(1, int(torch_threads)))


def _load_bytes(source_path: str) -> tuple[str, bytes]:
    waveform = load_waveform(source_path, sample_rate=SAMPLE_RATE, seconds=SECONDS)
    if waveform.ndim != 2 or tuple(waveform.shape) != (1, SAMPLES_PER_AUDIO):
        raise RuntimeError(f"waveform contract failed for {source_path}: shape={tuple(waveform.shape)}")
    if waveform.dtype != torch.float32:
        waveform = waveform.float()
    if not bool(torch.isfinite(waveform).all()):
        raise RuntimeError(f"waveform contains non-finite values: {source_path}")
    return source_path, waveform.contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")


def _iter_loaded(sources: list[AudioSource], workers: int, torch_threads: int) -> Iterable[tuple[str, bytes]]:
    paths = [source.source_path for source in sources]
    if int(workers) <= 1:
        _worker_init(torch_threads)
        for path in paths:
            yield _load_bytes(path)
        return
    context = mp.get_context("spawn")
    with context.Pool(processes=int(workers), initializer=_worker_init, initargs=(int(torch_threads),)) as pool:
        yield from pool.imap(_load_bytes, paths, chunksize=1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.partial")
    _write_json(partial, payload)
    os.replace(partial, path)


def _source_inventory_sha256(sources: list[AudioSource]) -> str:
    digest = hashlib.sha256()
    for source in sources:
        digest.update(json.dumps({
            "source_path": source.source_path,
            "source_group": source.source_group,
            "relative_path": source.relative_path,
            "source_size_bytes": source.source_size_bytes,
            "source_mtime_ns": source.source_mtime_ns,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _shard_bounds(total: int, num_shards: int, shard_id: int) -> tuple[int, int]:
    """Balanced contiguous ranges after the one-time global shuffle."""

    return total * shard_id // num_shards, total * (shard_id + 1) // num_shards


def _source_payload(source: AudioSource, shard_id: int, row_id: int) -> dict[str, Any]:
    return {
        "source_path": source.source_path,
        "source_group": source.source_group,
        "relative_path": source.relative_path,
        "source_size_bytes": source.source_size_bytes,
        "source_mtime_ns": source.source_mtime_ns,
        "shard": f"shard-{int(shard_id):05d}.bin",
        "shard_id": int(shard_id),
        "row": int(row_id),
        "byte_offset": int(row_id * BYTES_PER_AUDIO),
        "byte_length": int(BYTES_PER_AUDIO),
        "shape": [1, SAMPLES_PER_AUDIO],
        "dtype": "float32",
        "sample_rate": SAMPLE_RATE,
        "seconds": SECONDS,
    }


def build_shards(
    sources: list[AudioSource],
    output_dir: Path,
    *,
    num_shards: int,
    seed: int,
    workers: int,
    torch_threads: int,
    free_space_margin_gib: float,
    resume: bool,
    source_report: dict[str, Any],
) -> dict[str, Any]:
    if not sources:
        raise ValueError("no audio files found")
    if int(num_shards) <= 0 or int(num_shards) > len(sources):
        raise ValueError(f"num_shards must be in [1, {len(sources)}], got {num_shards}")
    if int(workers) <= 0 or int(torch_threads) <= 0:
        raise ValueError("workers and torch-threads-per-worker must be positive")
    if float(free_space_margin_gib) < 0:
        raise ValueError("free-space-margin-gib must be non-negative")

    shuffled = list(sources)
    random.Random(int(seed)).shuffle(shuffled)
    build_config = {
        "format": "raw_fixed_waveform_shards_v1",
        "num_unique_audio_files": len(shuffled),
        "num_shards": int(num_shards),
        "seed": int(seed),
        "source_inventory_sha256_before_shuffle": _source_inventory_sha256(sources),
        "source_inventory_sha256_after_shuffle": _source_inventory_sha256(shuffled),
        "sample_rate": SAMPLE_RATE,
        "seconds": SECONDS,
        "samples_per_audio": SAMPLES_PER_AUDIO,
        "bytes_per_audio": BYTES_PER_AUDIO,
        "dtype": "float32",
        "byte_order": "little",
        "source_report": source_report,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    building_marker = output_dir / "BUILDING"
    config_path = output_dir / "build_config.json"
    index_path = output_dir / "index.jsonl"
    metadata_path = output_dir / "metadata.json"
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite existing shard output: {output_dir}; pass --resume only for a matching interrupted build")
        if metadata_path.exists() or not building_marker.is_file() or not config_path.is_file() or not index_path.is_file():
            raise RuntimeError(f"output is not a resumable interrupted build: {output_dir}")
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        existing_contract = {key: value for key, value in existing_config.items() if key != "index_sha256"}
        if existing_contract != build_config:
            raise RuntimeError("resume build configuration/source inventory mismatch")
        if _sha256_file(index_path) != existing_config.get("index_sha256"):
            raise RuntimeError("resume index SHA256 mismatch")
        build_config = existing_config
    else:
        output_dir.mkdir(parents=False, exist_ok=False)
        building_marker.write_text("incomplete waveform shard build\n", encoding="utf-8")
        index_partial = output_dir / ".index.jsonl.partial"
        with index_partial.open("w", encoding="utf-8") as index_handle:
            for shard_id in range(int(num_shards)):
                start, end = _shard_bounds(len(shuffled), int(num_shards), shard_id)
                for row_id, source in enumerate(shuffled[start:end]):
                    index_handle.write(json.dumps(_source_payload(source, shard_id, row_id), ensure_ascii=False) + "\n")
        os.replace(index_partial, index_path)
        build_config["index_sha256"] = _sha256_file(index_path)
        _write_json_atomic(config_path, build_config)

    # Existing config includes the deterministic index hash; copy it into the
    # current report after a successful resume validation.
    if "index_sha256" not in build_config:
        build_config["index_sha256"] = _sha256_file(index_path)

    complete_bytes = 0
    partial_bytes = 0
    for shard_id in range(int(num_shards)):
        start, end = _shard_bounds(len(shuffled), int(num_shards), shard_id)
        final_path = output_dir / f"shard-{shard_id:05d}.bin"
        partial_path = output_dir / f".shard-{shard_id:05d}.bin.partial"
        if final_path.exists():
            complete_bytes += int(final_path.stat().st_size)
        if partial_path.exists():
            partial_bytes += int(partial_path.stat().st_size)
    required_remaining = len(shuffled) * BYTES_PER_AUDIO - complete_bytes
    free_bytes = int(shutil.disk_usage(output_dir).free) + partial_bytes
    required_with_margin = required_remaining + int(float(free_space_margin_gib) * 1024**3)
    if free_bytes < required_with_margin:
        raise RuntimeError(
            "insufficient free space for remaining shards and safety margin: "
            f"available_gib={free_bytes / 1024**3:.3f} required_gib={required_with_margin / 1024**3:.3f}"
        )

    started = time.perf_counter()
    try:
        shard_reports = []
        completed_rows = 0
        for shard_id in range(int(num_shards)):
            start, end = _shard_bounds(len(shuffled), int(num_shards), shard_id)
            shard_sources = shuffled[start:end]
            row_count = len(shard_sources)
            path = output_dir / f"shard-{shard_id:05d}.bin"
            partial_path = output_dir / f".shard-{shard_id:05d}.bin.partial"
            sidecar_path = output_dir / f"shard-{shard_id:05d}.json"
            expected_bytes = row_count * BYTES_PER_AUDIO
            if path.exists():
                actual_bytes = path.stat().st_size
                if actual_bytes != expected_bytes:
                    raise RuntimeError(f"completed shard byte-size mismatch: {path} actual={actual_bytes} expected={expected_bytes}")
                if sidecar_path.exists():
                    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    if (
                        sidecar.get("shard") != path.name
                        or int(sidecar.get("shard_id", -1)) != shard_id
                        or int(sidecar.get("rows", -1)) != row_count
                        or int(sidecar.get("bytes", -1)) != expected_bytes
                    ):
                        raise RuntimeError(f"completed shard sidecar mismatch: {sidecar_path}")
                    if not sidecar.get("sha256"):
                        sidecar["sha256"] = _sha256_file(path)
                        _write_json_atomic(sidecar_path, sidecar)
                else:
                    sidecar = {"shard": path.name, "shard_id": shard_id, "rows": row_count, "bytes": actual_bytes, "sha256": _sha256_file(path)}
                    _write_json_atomic(sidecar_path, sidecar)
                shard_reports.append(sidecar)
                completed_rows += row_count
                print(f"[waveform-shards] resume-skip shard={shard_id + 1}/{num_shards} rows={row_count}", flush=True)
                continue

            if partial_path.exists():
                partial_path.unlink()
            digest = hashlib.sha256()
            with partial_path.open("wb") as shard_handle:
                for row_id, (source_path, payload) in enumerate(_iter_loaded(shard_sources, workers, torch_threads)):
                    expected = shard_sources[row_id]
                    if source_path != expected.source_path:
                        raise RuntimeError(f"ordered worker output mismatch in shard {shard_id} row {row_id}")
                    if len(payload) != BYTES_PER_AUDIO:
                        raise RuntimeError(f"waveform byte-size mismatch for {source_path}: {len(payload)} != {BYTES_PER_AUDIO}")
                    shard_handle.write(payload)
                    digest.update(payload)
                    completed_rows += 1
                    if completed_rows % 1000 == 0:
                        print(f"[waveform-shards] processed={completed_rows}/{len(shuffled)}", flush=True)
                shard_handle.flush()
                os.fsync(shard_handle.fileno())
            actual_bytes = partial_path.stat().st_size
            if actual_bytes != expected_bytes:
                raise RuntimeError(f"partial shard byte-size mismatch: {partial_path} actual={actual_bytes} expected={expected_bytes}")
            os.replace(partial_path, path)
            sidecar = {
                "shard": path.name,
                "shard_id": shard_id,
                "rows": row_count,
                "bytes": actual_bytes,
                "sha256": digest.hexdigest(),
            }
            _write_json_atomic(sidecar_path, sidecar)
            shard_reports.append(sidecar)
            print(f"[waveform-shards] completed shard={shard_id + 1}/{num_shards} rows={row_count}", flush=True)

        metadata = {
            **build_config,
            "status": "PASS",
            "created_unix": time.time(),
            "elapsed_seconds": time.perf_counter() - started,
            "global_shuffle": "one deterministic path-list shuffle before contiguous shard assignment",
            "epoch_read_contract": "reader may globally shuffle shard IDs each epoch and independently shuffle rows within each fixed shard",
            "total_waveform_bytes": len(shuffled) * BYTES_PER_AUDIO,
            "total_waveform_gib": len(shuffled) * BYTES_PER_AUDIO / 1024**3,
            "index": "index.jsonl",
            "shards": shard_reports,
            "workers": int(workers),
            "torch_threads_per_worker": int(torch_threads),
        }
        _write_json_atomic(metadata_path, metadata)
        building_marker.unlink()
        error_path = output_dir / "build_error.json"
        if error_path.exists():
            error_path.unlink()
        return metadata
    except Exception as exc:
        _write_json_atomic(output_dir / "build_error.json", {
            "status": "INCOMPLETE",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "resume_command_required": True,
        })
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.num_shards) <= 0:
        raise ValueError("--num-shards must be positive")
    roots = [
        ("audiocaps_train", args.audiocaps_root),
        ("clotho_aqa", args.clotho_aqa_root),
        ("clotho_development", args.clotho_root),
    ]
    sources, source_report = collect_sources(roots)
    print(f"[waveform-shards] unique_audio_files={len(sources)} num_shards={args.num_shards}", flush=True)
    print(f"[waveform-shards] estimated_float32_gib={len(sources) * BYTES_PER_AUDIO / 1024**3:.3f}", flush=True)
    if args.dry_run:
        report = {
            "status": "DRY_RUN",
            "source_report": source_report,
            "num_unique_audio_files": len(sources),
            "num_shards": int(args.num_shards),
            "seed": int(args.seed),
            "estimated_total_waveform_bytes": len(sources) * BYTES_PER_AUDIO,
            "estimated_total_waveform_gib": len(sources) * BYTES_PER_AUDIO / 1024**3,
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    metadata = build_shards(
        sources,
        args.output_dir,
        num_shards=args.num_shards,
        seed=args.seed,
        workers=args.workers,
        torch_threads=args.torch_threads_per_worker,
        free_space_margin_gib=args.free_space_margin_gib,
        resume=args.resume,
        source_report=source_report,
    )
    print(json.dumps({
        "status": metadata["status"],
        "output_dir": str(args.output_dir),
        "num_unique_audio_files": metadata["num_unique_audio_files"],
        "num_shards": metadata["num_shards"],
        "total_waveform_gib": metadata["total_waveform_gib"],
        "elapsed_seconds": metadata["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
