#!/usr/bin/env python3
"""Build a resumable variable-length waveform store for Mellow reproduction.

Run this directly on a CPU login/compute node.  Audio is decoded, converted to
mono exactly as public Mellow (average the first two channels), resampled to
32 kHz, and stored without cropping or padding.  Ten-second random cropping is
performed later by the training dataset on every epoch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from audio_smollm2_135m_mellow_shared_store_configurable_epochs.data import (
    MELLOW_REFERENCE_COMMIT,
    MELLOW_VARIABLE_STORE_FORMAT,
    SAMPLE_RATE,
    normalize_path,
    row_audio_path,
    sha256_file,
)

DATA_FILE = "waveforms.f32"
INDEX_FILE = "index.jsonl"
DEFAULT_MANIFEST = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/"
    "preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl"
)


@dataclass(frozen=True)
class AudioSource:
    source_path: str
    source_size_bytes: int
    source_mtime_ns: int
    manifest_aliases: tuple[str, ...]
    filepath1_pool_member: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-threads-per-worker", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--verify-samples", type=int, default=128)
    parser.add_argument("--free-space-margin-gib", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def write_json_atomic(path: Path, payload: Any) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def inventory_manifest(manifest: Path) -> tuple[list[AudioSource], dict[str, Any]]:
    manifest = manifest.expanduser().resolve(strict=True)
    mutable: dict[str, dict[str, Any]] = {}
    row_count = 0
    missing_audio1 = 0
    missing_audio2 = 0
    explicit_audio2 = 0

    def add(raw: str, *, pool: bool) -> None:
        path = Path(raw).expanduser().resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(path)
        canonical = normalize_path(path)
        item = mutable.get(canonical)
        if item is None:
            stat = path.stat()
            item = {
                "source_path": canonical,
                "source_size_bytes": int(stat.st_size),
                "source_mtime_ns": int(stat.st_mtime_ns),
                "manifest_aliases": set(),
                "filepath1_pool_member": False,
            }
            mutable[canonical] = item
        item["manifest_aliases"].add(normalize_path(raw))
        item["filepath1_pool_member"] = bool(item["filepath1_pool_member"] or pool)

    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
            row_count += 1
            first, second = row_audio_path(row, True), row_audio_path(row, False)
            if first:
                add(first, pool=True)
            else:
                missing_audio1 += 1
            if second:
                add(second, pool=False)
                explicit_audio2 += 1
            else:
                missing_audio2 += 1
    if not mutable:
        raise RuntimeError("manifest contains no audio paths")
    if not any(bool(item["filepath1_pool_member"]) for item in mutable.values()):
        raise RuntimeError("manifest has no non-empty filepath1 random-audio pool")
    sources = [
        AudioSource(
            source_path=item["source_path"],
            source_size_bytes=item["source_size_bytes"],
            source_mtime_ns=item["source_mtime_ns"],
            manifest_aliases=tuple(sorted(item["manifest_aliases"])),
            filepath1_pool_member=bool(item["filepath1_pool_member"]),
        )
        for item in sorted(mutable.values(), key=lambda value: value["source_path"])
    ]
    report = {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "rows": row_count,
        "missing_audio1_rows": missing_audio1,
        "missing_audio2_rows": missing_audio2,
        "explicit_audio2_rows": explicit_audio2,
        "num_unique_audio_files": len(sources),
        "random_audio_pool_size": sum(int(item.filepath1_pool_member) for item in sources),
    }
    return sources, report


def source_inventory_sha256(sources: list[AudioSource]) -> str:
    digest = hashlib.sha256()
    for source in sources:
        digest.update(json.dumps({
            "source_path": source.source_path,
            "source_size_bytes": source.source_size_bytes,
            "source_mtime_ns": source.source_mtime_ns,
            "manifest_aliases": source.manifest_aliases,
            "filepath1_pool_member": source.filepath1_pool_member,
        }, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_full_waveform(path: Path) -> torch.Tensor:
    import torchaudio

    waveform, source_rate = torchaudio.load(str(path), channels_first=True)
    waveform = waveform.float()
    if waveform.ndim != 2 or waveform.shape[0] < 1:
        raise RuntimeError(f"invalid decoded waveform {path}: {tuple(waveform.shape)}")
    if waveform.shape[0] > 1:
        waveform = ((waveform[0] + waveform[1]) / 2.0).unsqueeze(0)
    if int(source_rate) != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, int(source_rate), SAMPLE_RATE)
    waveform = waveform.to(torch.float32).contiguous()
    if waveform.ndim != 2 or waveform.shape[0] != 1 or waveform.shape[1] <= 0:
        raise RuntimeError(f"invalid decoded waveform {path}: {tuple(waveform.shape)}")
    if not bool(torch.isfinite(waveform).all()):
        raise RuntimeError(f"non-finite waveform: {path}")
    return waveform


def worker_init(threads: int) -> None:
    torch.set_num_threads(max(1, int(threads)))


def load_source(source: AudioSource) -> tuple[str, int, bytes]:
    path = Path(source.source_path)
    before = path.stat()
    if int(before.st_size) != source.source_size_bytes or int(before.st_mtime_ns) != source.source_mtime_ns:
        raise RuntimeError(f"source changed after inventory: {path}")
    waveform = load_full_waveform(path)
    after = path.stat()
    if int(after.st_size) != source.source_size_bytes or int(after.st_mtime_ns) != source.source_mtime_ns:
        raise RuntimeError(f"source changed during decoding: {path}")
    payload = waveform.numpy().astype("<f4", copy=False).tobytes(order="C")
    return source.source_path, int(waveform.shape[-1]), payload


def iter_loaded(sources: list[AudioSource], workers: int, threads: int) -> Iterable[tuple[str, int, bytes]]:
    if workers <= 1:
        worker_init(threads)
        for source in sources:
            yield load_source(source)
        return
    context = mp.get_context("spawn")
    with context.Pool(workers, initializer=worker_init, initargs=(threads,)) as pool:
        yield from pool.imap(load_source, sources, chunksize=1)


def estimate_resampled_samples(path: Path) -> int:
    """Estimate stored samples across TorchAudio 2.8 and 2.9+ I/O APIs.

    TorchAudio 2.9 removed ``torchaudio.info`` and routes ``load`` through
    TorchCodec. Prefer metadata-only probes when available and retain a full
    decode fallback so the store builder also works with older or unusual
    TorchAudio/TorchCodec combinations. This estimate is used only for the
    pre-build free-space guard; the final store records the exact decoded and
    resampled length returned by ``load_full_waveform``.
    """
    import torchaudio

    info_function = getattr(torchaudio, "info", None)
    if callable(info_function):
        try:
            info = info_function(str(path))
            frames = int(info.num_frames)
            source_rate = int(info.sample_rate)
            if frames > 0 and source_rate > 0:
                return math.ceil(frames * SAMPLE_RATE / source_rate)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    try:
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(str(path))
        metadata = decoder.metadata
        source_rate_value = getattr(metadata, "sample_rate", None)
        source_rate = int(source_rate_value) if source_rate_value is not None else 0
        frames_value = getattr(metadata, "num_frames", None)
        if frames_value is not None and source_rate > 0:
            frames = int(frames_value)
            if frames > 0:
                return math.ceil(frames * SAMPLE_RATE / source_rate)
        duration_value = getattr(metadata, "duration_seconds", None)
        if duration_value is not None:
            duration = float(duration_value)
            if math.isfinite(duration) and duration > 0:
                return max(1, math.ceil(duration * SAMPLE_RATE))
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass

    waveform, source_rate = torchaudio.load(str(path), channels_first=True)
    if waveform.ndim != 2 or waveform.shape[-1] <= 0 or int(source_rate) <= 0:
        raise RuntimeError(
            f"cannot estimate decoded length: {path}; "
            f"shape={tuple(waveform.shape)} sample_rate={source_rate}"
        )
    return math.ceil(int(waveform.shape[-1]) * SAMPLE_RATE / int(source_rate))


def logical_config(sources: list[AudioSource], report: dict[str, Any]) -> dict[str, Any]:
    estimated_samples = sum(
        estimate_resampled_samples(Path(source.source_path)) for source in sources
    )
    return {
        "format": MELLOW_VARIABLE_STORE_FORMAT,
        "mellow_reference_commit": MELLOW_REFERENCE_COMMIT,
        "manifest": report["manifest"],
        "manifest_sha256": report["manifest_sha256"],
        "source_inventory_sha256": source_inventory_sha256(sources),
        "num_unique_audio_files": len(sources),
        "filepath1_unique_pool_size": report["random_audio_pool_size"],
        "sample_rate": SAMPLE_RATE,
        "dtype": "float32",
        "byte_order": "little",
        "data_file": DATA_FILE,
        "estimated_total_waveform_bytes": estimated_samples * 4,
        "manifest_report": report,
        "preprocessing_contract": "public Mellow mono-first-two-channel mean and 32kHz resample; no crop or padding in store",
    }


def initialize(output: Path, config: dict[str, Any]) -> dict[str, int]:
    output.mkdir(parents=False, exist_ok=False)
    (output / "BUILDING").write_text("incomplete Mellow variable waveform store\n", encoding="utf-8")
    write_json_atomic(output / "build_config.json", config)
    (output / f".{DATA_FILE}.partial").touch(exist_ok=False)
    (output / f".{INDEX_FILE}.partial").touch(exist_ok=False)
    progress = {"status": "BUILDING", "completed_audio": 0, "data_bytes": 0, "updated_unix": time.time()}
    write_json_atomic(output / "progress.json", progress)
    return progress


def resume_state(output: Path, config: dict[str, Any]) -> dict[str, int]:
    for name in ("BUILDING", "build_config.json", "progress.json"):
        if not (output / name).exists():
            raise RuntimeError(f"not a resumable store: missing {name}")
    existing = json.loads((output / "build_config.json").read_text(encoding="utf-8"))
    if existing != config:
        raise RuntimeError("resume store configuration differs")
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    completed = int(progress.get("completed_audio", -1))
    durable_bytes = int(progress.get("data_bytes", -1))
    data_partial = output / f".{DATA_FILE}.partial"
    index_partial = output / f".{INDEX_FILE}.partial"
    # A crash can occur after either final atomic rename but before BUILDING is
    # removed. Move such a final file back to its partial name so the durable
    # progress cursor remains the single authority for resuming.
    for partial, final in ((data_partial, output / DATA_FILE), (index_partial, output / INDEX_FILE)):
        if partial.exists() and final.exists():
            raise RuntimeError(f"resume store has both partial and final files: {partial.name}")
        if not partial.exists() and final.is_file():
            os.replace(final, partial)
    if (
        completed < 0
        or completed > int(config["num_unique_audio_files"])
        or durable_bytes < 0
        or not data_partial.is_file()
        or not index_partial.is_file()
    ):
        raise RuntimeError("resume progress/files are invalid")
    if data_partial.stat().st_size < durable_bytes:
        raise RuntimeError("resume waveform file is shorter than durable progress")
    with data_partial.open("r+b") as handle:
        handle.truncate(durable_bytes)
    lines = index_partial.read_text(encoding="utf-8").splitlines()
    if len(lines) < completed:
        raise RuntimeError("resume index is shorter than durable progress")
    index_partial.write_text("\n".join(lines[:completed]) + ("\n" if completed else ""), encoding="utf-8")
    return progress


def verify_store(sources: list[AudioSource], index_entries: list[dict[str, Any]], data_path: Path, count: int) -> dict[str, Any]:
    count = min(max(1, count), len(sources))
    ids = sorted({i * (len(sources) - 1) // max(1, count - 1) for i in range(count)})
    with data_path.open("rb") as handle:
        for audio_id in ids:
            _, samples, expected = load_source(sources[audio_id])
            entry = index_entries[audio_id]
            if samples != int(entry["num_samples"]):
                raise RuntimeError(f"verification sample count mismatch: {audio_id}")
            handle.seek(int(entry["byte_offset"]))
            actual = handle.read(int(entry["byte_length"]))
            if actual != expected:
                raise RuntimeError(f"verification byte mismatch: {audio_id}")
    return {"passed": True, "verified_audio_ids": ids, "verified_samples": len(ids)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.workers < 1
        or args.torch_threads_per_worker < 1
        or args.checkpoint_every < 1
        or args.verify_samples < 1
        or args.free_space_margin_gib < 0
    ):
        raise ValueError("worker and checkpoint settings must be positive")
    sources, report = inventory_manifest(args.manifest)
    config = logical_config(sources, report)
    output = args.output_dir.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        return {"status": "DRY_RUN", **config}
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing existing output: {output}")
        progress = resume_state(output, config)
    else:
        progress = initialize(output, config)
    completed = int(progress["completed_audio"])
    data_bytes = int(progress["data_bytes"])
    data_partial = output / f".{DATA_FILE}.partial"
    index_partial = output / f".{INDEX_FILE}.partial"
    margin = int(args.free_space_margin_gib * 1024**3)
    estimated_remaining = max(0, int(config["estimated_total_waveform_bytes"]) - data_bytes)
    free_bytes = int(shutil.disk_usage(output).free)
    if free_bytes < estimated_remaining + margin:
        raise RuntimeError(
            "insufficient free space for estimated decoded waveforms and safety margin: "
            f"free={free_bytes} estimated_remaining={estimated_remaining} margin={margin}"
        )
    started = time.perf_counter()
    try:
        with data_partial.open("ab") as data_handle, index_partial.open("a", encoding="utf-8") as index_handle:
            for audio_id, (source_path, samples, payload) in enumerate(
                iter_loaded(sources[completed:], args.workers, args.torch_threads_per_worker), start=completed
            ):
                source = sources[audio_id]
                if source_path != source.source_path or len(payload) != samples * 4:
                    raise RuntimeError(f"ordered worker result mismatch: {audio_id}")
                if int(shutil.disk_usage(output).free) < len(payload) + margin:
                    raise RuntimeError(
                        "free space fell below the configured safety margin during construction"
                    )
                entry = {
                    "audio_id": audio_id,
                    "source_path": source.source_path,
                    "source_size_bytes": source.source_size_bytes,
                    "source_mtime_ns": source.source_mtime_ns,
                    "manifest_aliases": list(source.manifest_aliases),
                    "filepath1_pool_member": source.filepath1_pool_member,
                    "byte_offset": data_bytes,
                    "byte_length": len(payload),
                    "num_samples": samples,
                    "shape": [1, samples],
                    "dtype": "float32",
                }
                data_handle.write(payload)
                index_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                data_bytes += len(payload)
                durable = audio_id + 1
                if durable % args.checkpoint_every == 0 or durable == len(sources):
                    data_handle.flush(); os.fsync(data_handle.fileno())
                    index_handle.flush(); os.fsync(index_handle.fileno())
                    write_json_atomic(output / "progress.json", {
                        "status": "BUILDING", "completed_audio": durable,
                        "data_bytes": data_bytes, "updated_unix": time.time(),
                    })
                if durable % 1000 == 0 or durable == len(sources):
                    print(f"[mellow-store] processed={durable}/{len(sources)} bytes={data_bytes}", flush=True)
        os.replace(data_partial, output / DATA_FILE)
        os.replace(index_partial, output / INDEX_FILE)
        entries = [json.loads(line) for line in (output / INDEX_FILE).read_text(encoding="utf-8").splitlines() if line]
        if len(entries) != len(sources) or (output / DATA_FILE).stat().st_size != data_bytes:
            raise RuntimeError("final store cardinality/size mismatch")
        verification = verify_store(sources, entries, output / DATA_FILE, args.verify_samples)
        metadata = {
            **config,
            "status": "PASS",
            "total_waveform_bytes": data_bytes,
            "total_waveform_gib": data_bytes / 1024**3,
            "index_sha256": sha256_file(output / INDEX_FILE),
            "waveform_sha256": sha256_file(output / DATA_FILE),
            "waveform_verification": verification,
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "layout": "variable-length contiguous little-endian float32 with byte offsets in index.jsonl",
            "sharing_contract": "one immutable mmap inode copied once to node /dev/shm",
            "random_audio_pool_contract": "dataset rebuilds sorted unique non-empty filepath1 paths from the bound manifest",
        }
        write_json_atomic(output / "metadata.json", metadata)
        write_json_atomic(output / "progress.json", {
            "status": "PASS", "completed_audio": len(sources),
            "data_bytes": data_bytes, "updated_unix": time.time(),
        })
        (output / "BUILDING").unlink()
        error = output / "build_error.json"
        if error.exists():
            error.unlink()
        return metadata
    except Exception as exc:
        write_json_atomic(output / "build_error.json", {
            "status": "INCOMPLETE", "error": repr(exc),
            "traceback": traceback.format_exc(), "resume_required": True,
        })
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build(args)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
