#!/usr/bin/env python3
"""Build one manifest-scoped, deduplicated fixed-waveform store on CPU.

The store contains exactly the unique audio files referenced by a ReasonAQA
train manifest.  Each row is the existing training pipeline's mono, 32 kHz,
ten-second float32 waveform.  Rows are packed into one fixed-stride raw file so
all local DDP ranks can mmap the same inode and share clean page-cache pages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
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

from audio_5_10x2_5_mesh_mellow.data import (  # noqa: E402
    UNIQUE_WAVEFORM_STORE_FORMAT,
    load_waveform,
)


SAMPLE_RATE = 32_000
SECONDS = 10
SAMPLES_PER_AUDIO = SAMPLE_RATE * SECONDS
BYTES_PER_AUDIO = SAMPLES_PER_AUDIO * 4
DATA_FILE = "waveforms.f32"
DEFAULT_MANIFEST = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/"
    "preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl"
)
GROUP_MARKERS = (
    ("/audiocaps_v2/train/", "audiocaps"),
    ("/clotho_aqa_audio/audio_files/", "clotho_aqa"),
    ("/clotho_v2_1/development/", "clotho"),
)


@dataclass
class _MutableSource:
    source_path: str
    source_group: str
    source_size_bytes: int
    source_mtime_ns: int
    manifest_aliases: set[str]
    slot_reference_count: int = 0
    qa_incidence_count: int = 0


@dataclass(frozen=True)
class AudioSource:
    source_path: str
    source_group: str
    source_size_bytes: int
    source_mtime_ns: int
    manifest_aliases: tuple[str, ...]
    slot_reference_count: int
    qa_incidence_count: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8, help="CPU decode workers; ordered output remains deterministic")
    parser.add_argument("--torch-threads-per-worker", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100, help="fsync/progress interval in completed audio rows")
    parser.add_argument("--verify-samples", type=int, default=128, help="deterministic rows re-decoded and compared byte-for-byte before PASS")
    parser.add_argument("--free-space-margin-gib", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true", help="resume only a matching interrupted build")
    parser.add_argument("--dry-run", action="store_true", help="audit manifest/inventory and required bytes without decoding")
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _row_path(row: dict[str, Any], first: bool) -> str:
    keys = ("audio1_path", "filepath1") if first else ("audio2_path", "filepath2")
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _normalize_alias(value: str) -> str:
    return os.path.normpath(os.path.expanduser(value)).replace("\\", "/")


def _source_group(value: str) -> str:
    normalized = _normalize_alias(value)
    for marker, group in GROUP_MARKERS:
        if marker in normalized:
            return group
    return "other"


def collect_manifest_sources(manifest: Path) -> tuple[list[AudioSource], dict[str, Any]]:
    manifest = manifest.expanduser().resolve(strict=True)
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest is not a file: {manifest}")

    alias_to_canonical: dict[str, str] = {}
    by_canonical: dict[str, _MutableSource] = {}
    rows = 0
    same_audio_rows = 0
    distinct_audio_rows = 0
    missing_audio1_rows = 0

    def resolve_source(raw_value: str) -> _MutableSource:
        alias = _normalize_alias(raw_value)
        canonical = alias_to_canonical.get(alias)
        if canonical is None:
            path = Path(raw_value).expanduser().resolve(strict=True)
            if not path.is_file():
                raise FileNotFoundError(f"manifest audio is not a file: {path}")
            canonical = _normalize_alias(str(path))
            alias_to_canonical[alias] = canonical
        source = by_canonical.get(canonical)
        if source is None:
            path = Path(canonical)
            stat = path.stat()
            source = _MutableSource(
                source_path=canonical,
                source_group=_source_group(canonical),
                source_size_bytes=int(stat.st_size),
                source_mtime_ns=int(stat.st_mtime_ns),
                manifest_aliases=set(),
            )
            by_canonical[canonical] = source
        source.manifest_aliases.add(alias)
        return source

    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at manifest line {line_number}: {exc}") from exc
            rows += 1
            audio1 = _row_path(row, True)
            audio2 = _row_path(row, False)
            if not audio1:
                missing_audio1_rows += 1
                continue
            audio2 = audio2 or audio1
            first = resolve_source(audio1)
            second = resolve_source(audio2)
            first.slot_reference_count += 1
            second.slot_reference_count += 1
            first.qa_incidence_count += 1
            if first.source_path == second.source_path:
                same_audio_rows += 1
            else:
                distinct_audio_rows += 1
                second.qa_incidence_count += 1

    if missing_audio1_rows:
        raise ValueError(f"manifest contains {missing_audio1_rows} rows without audio1")
    if not by_canonical:
        raise ValueError(f"manifest has no audio references: {manifest}")

    sources = [
        AudioSource(
            source_path=item.source_path,
            source_group=item.source_group,
            source_size_bytes=item.source_size_bytes,
            source_mtime_ns=item.source_mtime_ns,
            manifest_aliases=tuple(sorted(item.manifest_aliases)),
            slot_reference_count=item.slot_reference_count,
            qa_incidence_count=item.qa_incidence_count,
        )
        for item in sorted(by_canonical.values(), key=lambda value: value.source_path)
    ]
    source_counts: dict[str, int] = {}
    for source in sources:
        source_counts[source.source_group] = source_counts.get(source.source_group, 0) + 1
    report = {
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "rows": rows,
        "same_audio_rows": same_audio_rows,
        "distinct_audio_rows": distinct_audio_rows,
        "unique_canonical_audio_files": len(sources),
        "manifest_path_aliases": len(alias_to_canonical),
        "source_unique_audio_counts": source_counts,
        "total_slot_references": sum(item.slot_reference_count for item in sources),
        "total_qa_incidences": sum(item.qa_incidence_count for item in sources),
    }
    return sources, report


def _source_inventory_sha256(sources: list[AudioSource]) -> str:
    digest = hashlib.sha256()
    for source in sources:
        digest.update(json.dumps({
            "source_path": source.source_path,
            "source_group": source.source_group,
            "source_size_bytes": source.source_size_bytes,
            "source_mtime_ns": source.source_mtime_ns,
            "manifest_aliases": list(source.manifest_aliases),
            "slot_reference_count": source.slot_reference_count,
            "qa_incidence_count": source.qa_incidence_count,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _index_payload(audio_id: int, source: AudioSource) -> dict[str, Any]:
    return {
        "audio_id": int(audio_id),
        "source_path": source.source_path,
        "source_group": source.source_group,
        "source_size_bytes": source.source_size_bytes,
        "source_mtime_ns": source.source_mtime_ns,
        "manifest_aliases": list(source.manifest_aliases),
        "slot_reference_count": source.slot_reference_count,
        "qa_incidence_count": source.qa_incidence_count,
        "byte_offset": int(audio_id * BYTES_PER_AUDIO),
        "byte_length": BYTES_PER_AUDIO,
        "shape": [1, SAMPLES_PER_AUDIO],
        "dtype": "float32",
    }


def _worker_init(torch_threads: int) -> None:
    torch.set_num_threads(max(1, int(torch_threads)))


def _load_source(source: AudioSource) -> tuple[str, bytes]:
    path = Path(source.source_path)
    before = path.stat()
    if int(before.st_size) != source.source_size_bytes or int(before.st_mtime_ns) != source.source_mtime_ns:
        raise RuntimeError(f"source changed after inventory: {path}")
    waveform = load_waveform(path, sample_rate=SAMPLE_RATE, seconds=SECONDS)
    after = path.stat()
    if int(after.st_size) != source.source_size_bytes or int(after.st_mtime_ns) != source.source_mtime_ns:
        raise RuntimeError(f"source changed while decoding: {path}")
    if waveform.ndim != 2 or tuple(waveform.shape) != (1, SAMPLES_PER_AUDIO):
        raise RuntimeError(f"waveform contract failed for {path}: shape={tuple(waveform.shape)}")
    waveform = waveform.to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(waveform).all()):
        raise RuntimeError(f"waveform contains non-finite values: {path}")
    payload = waveform.numpy().astype("<f4", copy=False).tobytes(order="C")
    if len(payload) != BYTES_PER_AUDIO:
        raise RuntimeError(f"waveform byte-size mismatch for {path}: {len(payload)} != {BYTES_PER_AUDIO}")
    return source.source_path, payload


def _iter_loaded(sources: list[AudioSource], workers: int, torch_threads: int) -> Iterable[tuple[str, bytes]]:
    if int(workers) <= 1:
        _worker_init(torch_threads)
        for source in sources:
            yield _load_source(source)
        return
    context = mp.get_context("spawn")
    with context.Pool(
        processes=int(workers),
        initializer=_worker_init,
        initargs=(int(torch_threads),),
    ) as pool:
        yield from pool.imap(_load_source, sources, chunksize=1)


def _logical_config(sources: list[AudioSource], manifest_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": UNIQUE_WAVEFORM_STORE_FORMAT,
        "manifest": manifest_report["manifest"],
        "manifest_sha256": manifest_report["manifest_sha256"],
        "source_inventory_sha256": _source_inventory_sha256(sources),
        "num_unique_audio_files": len(sources),
        "sample_rate": SAMPLE_RATE,
        "seconds": SECONDS,
        "samples_per_audio": SAMPLES_PER_AUDIO,
        "bytes_per_audio": BYTES_PER_AUDIO,
        "dtype": "float32",
        "byte_order": "little",
        "data_file": DATA_FILE,
        "total_waveform_bytes": len(sources) * BYTES_PER_AUDIO,
        "manifest_report": manifest_report,
    }


def _verify_rows(sources: list[AudioSource], data_path: Path, sample_count: int) -> dict[str, Any]:
    if int(sample_count) <= 0:
        raise ValueError("verify-samples must be positive")
    count = min(int(sample_count), len(sources))
    if count == 1:
        audio_ids = [0]
    else:
        audio_ids = sorted({index * (len(sources) - 1) // (count - 1) for index in range(count)})
    checked_bytes = 0
    with data_path.open("rb") as handle:
        for audio_id in audio_ids:
            source_path, expected = _load_source(sources[audio_id])
            handle.seek(audio_id * BYTES_PER_AUDIO)
            actual = handle.read(BYTES_PER_AUDIO)
            if len(actual) != BYTES_PER_AUDIO:
                raise RuntimeError(f"verification row is truncated: audio_id={audio_id}")
            if actual != expected:
                raise RuntimeError(f"verification byte mismatch: audio_id={audio_id} source={source_path}")
            checked_bytes += len(actual)
    return {
        "method": "evenly spaced rows re-decoded with load_waveform and compared byte-for-byte",
        "requested_samples": int(sample_count),
        "verified_samples": len(audio_ids),
        "verified_audio_ids": audio_ids,
        "verified_bytes": checked_bytes,
        "passed": True,
    }


def _initialize_output(output_dir: Path, sources: list[AudioSource], config: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=False, exist_ok=False)
    (output_dir / "BUILDING").write_text("incomplete unique waveform store build\n", encoding="utf-8")
    index_partial = output_dir / ".index.jsonl.partial"
    with index_partial.open("w", encoding="utf-8") as handle:
        for audio_id, source in enumerate(sources):
            handle.write(json.dumps(_index_payload(audio_id, source), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(index_partial, output_dir / "index.jsonl")
    initialized = {**config, "index_sha256": _sha256_file(output_dir / "index.jsonl")}
    _write_json_atomic(output_dir / "build_config.json", initialized)
    _write_json_atomic(output_dir / "progress.json", {
        "status": "BUILDING",
        "completed_audio": 0,
        "data_bytes": 0,
        "updated_unix": time.time(),
    })
    return initialized


def _resume_output(output_dir: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    required = (output_dir / "BUILDING", output_dir / "build_config.json", output_dir / "index.jsonl", output_dir / "progress.json")
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"output is not a resumable interrupted build: {output_dir}")
    existing = json.loads((output_dir / "build_config.json").read_text(encoding="utf-8"))
    existing_logical = {key: value for key, value in existing.items() if key != "index_sha256"}
    if existing_logical != config:
        raise RuntimeError("resume build configuration/manifest/source inventory mismatch")
    if _sha256_file(output_dir / "index.jsonl") != existing.get("index_sha256"):
        raise RuntimeError("resume index SHA256 mismatch")
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    completed = int(progress.get("completed_audio", -1))
    expected_bytes = completed * BYTES_PER_AUDIO
    partial = output_dir / f".{DATA_FILE}.partial"
    final = output_dir / DATA_FILE
    if final.is_file() and completed == int(config["num_unique_audio_files"]):
        if final.stat().st_size != int(config["total_waveform_bytes"]):
            raise RuntimeError("completed data file has the wrong size during resume finalization")
        return existing, progress
    if completed < 0 or completed > int(config["num_unique_audio_files"]) or not partial.is_file():
        raise RuntimeError("resume progress/data-file contract mismatch")
    if partial.stat().st_size < expected_bytes:
        raise RuntimeError("partial data file is shorter than durable progress")
    if partial.stat().st_size != expected_bytes:
        with partial.open("r+b") as handle:
            handle.truncate(expected_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    return existing, progress


def build_store(
    sources: list[AudioSource],
    manifest_report: dict[str, Any],
    output_dir: Path,
    *,
    workers: int,
    torch_threads: int,
    checkpoint_every: int,
    verify_samples: int,
    free_space_margin_gib: float,
    resume: bool,
) -> dict[str, Any]:
    if int(workers) <= 0 or int(torch_threads) <= 0 or int(checkpoint_every) <= 0 or int(verify_samples) <= 0:
        raise ValueError("workers, torch-threads-per-worker, checkpoint-every, and verify-samples must be positive")
    if float(free_space_margin_gib) < 0:
        raise ValueError("free-space-margin-gib must be non-negative")
    config = _logical_config(sources, manifest_report)
    output_dir = output_dir.expanduser().resolve(strict=False)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite existing waveform store: {output_dir}; use --resume only after an interrupted matching build")
        config, progress = _resume_output(output_dir, config)
    else:
        config = _initialize_output(output_dir, sources, config)
        progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))

    completed = int(progress["completed_audio"])
    expected_total = int(config["total_waveform_bytes"])
    final_path = output_dir / DATA_FILE
    partial_path = output_dir / f".{DATA_FILE}.partial"
    current_bytes = final_path.stat().st_size if final_path.is_file() else (partial_path.stat().st_size if partial_path.exists() else 0)
    required_remaining = expected_total - current_bytes
    free_bytes = int(shutil.disk_usage(output_dir).free)
    required_with_margin = required_remaining + int(float(free_space_margin_gib) * 1024**3)
    if free_bytes < required_with_margin:
        raise RuntimeError(
            "insufficient free space for waveform store and safety margin: "
            f"available_gib={free_bytes / 1024**3:.3f} required_gib={required_with_margin / 1024**3:.3f}"
        )

    started = time.perf_counter()
    try:
        if completed < len(sources):
            mode = "ab" if completed else "wb"
            with partial_path.open(mode) as handle:
                for offset, (source_path, payload) in enumerate(
                    _iter_loaded(sources[completed:], workers, torch_threads),
                    start=completed,
                ):
                    expected_source = sources[offset]
                    if source_path != expected_source.source_path:
                        raise RuntimeError(f"ordered worker output mismatch at audio_id={offset}")
                    handle.write(payload)
                    durable_completed = offset + 1
                    if durable_completed % int(checkpoint_every) == 0 or durable_completed == len(sources):
                        handle.flush()
                        os.fsync(handle.fileno())
                        _write_json_atomic(output_dir / "progress.json", {
                            "status": "BUILDING",
                            "completed_audio": durable_completed,
                            "data_bytes": durable_completed * BYTES_PER_AUDIO,
                            "last_audio_id": durable_completed - 1,
                            "last_source_path": expected_source.source_path,
                            "updated_unix": time.time(),
                        })
                    if durable_completed % 1000 == 0 or durable_completed == len(sources):
                        print(f"[unique-waveform-store] processed={durable_completed}/{len(sources)}", flush=True)
            completed = len(sources)

        data_for_validation = final_path if final_path.is_file() else partial_path
        if completed != len(sources) or not data_for_validation.is_file() or data_for_validation.stat().st_size != expected_total:
            raise RuntimeError("completed waveform store has the wrong row count or byte size")
        print("[unique-waveform-store] computing final waveform SHA256", flush=True)
        waveform_sha256 = _sha256_file(data_for_validation)
        print(f"[unique-waveform-store] verifying {min(int(verify_samples), len(sources))} deterministic rows", flush=True)
        waveform_verification = _verify_rows(sources, data_for_validation, verify_samples)
        if not final_path.is_file():
            os.replace(partial_path, final_path)
        metadata = {
            **config,
            "status": "PASS",
            "created_unix": time.time(),
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "waveform_sha256": waveform_sha256,
            "waveform_verification": waveform_verification,
            "layout": "one fixed-stride row-major raw file; [num_unique_audio, 320000] little-endian float32",
            "sharing_contract": "all local ranks mmap the same immutable inode; clean pages are node-shared",
            "preprocessing_contract": "load_waveform: mono mean, 32 kHz resample, first-10-second crop or right-zero-pad",
            "workers_this_invocation": int(workers),
            "torch_threads_per_worker": int(torch_threads),
        }
        _write_json_atomic(output_dir / "metadata.json", metadata)
        _write_json_atomic(output_dir / "progress.json", {
            "status": "PASS",
            "completed_audio": len(sources),
            "data_bytes": expected_total,
            "updated_unix": time.time(),
        })
        (output_dir / "BUILDING").unlink()
        error_path = output_dir / "build_error.json"
        if error_path.exists():
            error_path.unlink()
        return metadata
    except Exception as exc:
        _write_json_atomic(output_dir / "build_error.json", {
            "status": "INCOMPLETE",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "resume_required": True,
        })
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources, manifest_report = collect_manifest_sources(args.manifest)
    estimated_bytes = len(sources) * BYTES_PER_AUDIO
    summary = {
        "manifest": manifest_report["manifest"],
        "manifest_sha256": manifest_report["manifest_sha256"],
        "rows": manifest_report["rows"],
        "num_unique_audio_files": len(sources),
        "estimated_total_waveform_bytes": estimated_bytes,
        "estimated_total_waveform_gib": estimated_bytes / 1024**3,
        "source_unique_audio_counts": manifest_report["source_unique_audio_counts"],
    }
    print("[unique-waveform-store] " + json.dumps(summary, ensure_ascii=False), flush=True)
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", **summary, "manifest_report": manifest_report}, indent=2, ensure_ascii=False))
        return 0
    metadata = build_store(
        sources,
        manifest_report,
        args.output_dir,
        workers=args.workers,
        torch_threads=args.torch_threads_per_worker,
        checkpoint_every=args.checkpoint_every,
        verify_samples=args.verify_samples,
        free_space_margin_gib=args.free_space_margin_gib,
        resume=args.resume,
    )
    print(json.dumps({
        "status": metadata["status"],
        "output_dir": str(args.output_dir),
        "num_unique_audio_files": metadata["num_unique_audio_files"],
        "total_waveform_gib": metadata["total_waveform_bytes"] / 1024**3,
        "waveform_sha256": metadata["waveform_sha256"],
        "elapsed_seconds_this_invocation": metadata["elapsed_seconds_this_invocation"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
