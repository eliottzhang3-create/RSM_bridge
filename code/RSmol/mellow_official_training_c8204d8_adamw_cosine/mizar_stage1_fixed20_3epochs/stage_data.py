"""Resolve the filtered S1 rows using the completed audit and stage raw files.

Audio is copied as raw bytes without decoding, resampling or cropping; no encoder
feature is cached. The JSONL/index are job-local views of the unchanged source
manifest, shared by all eight ranks. Source probes run later in the trainer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from contract import (BATCH_SIZE, SEED, SOURCE_COUNTS, WORLD_SIZE, data_order,
                      digest, geometry, raw_references, write_json)
from data.indexed_jsonl import IndexedRows

sys.path.append(str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_mizar_s1_raw_audio import iter_rows  # noqa: E402


def load_resolutions(path):
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"field", "raw_path", "status", "resolved_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("resolved_paths.tsv does not have the audit schema")
        for row in reader:
            if row["status"] != "exists":
                continue
            key = (row["field"], row["raw_path"])
            resolved = row["resolved_path"]
            if not resolved or (key in result and result[key] != resolved):
                raise ValueError(f"empty/conflicting audited resolution: {key}")
            result[key] = resolved
    return result


def prepare(manifest, resolutions, metadata, *, strict_counts=True):
    """Stream all retained rows, preserving text/fields and deduplicating raw files."""
    metadata = Path(metadata)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    counts, probes, files, used = Counter(), {}, {}, {}
    row_hash = hashlib.sha256()
    rows = 0
    with metadata.open("xb") as output, Path(str(metadata) + ".idx").open("xb") as index:
        for row_index, row in enumerate(iter_rows(Path(manifest))):
            source, refs = raw_references(row)
            canonical = json.dumps(row, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")).encode("utf-8") + b"\n"
            row_hash.update(canonical)
            paths = []
            for field, raw in refs:
                # The TSV is keyed by raw field + reference, not original row ID.
                # Original audit row IDs are invalid after filtering rows.
                key = (field, raw)
                if key not in resolutions:
                    raise ValueError(f"row {row_index}, {source}: no successful audited path for {key}")
                original = used.get(key)
                if original is None:
                    original = str(Path(resolutions[key]).resolve())
                relative = "audio/" + hashlib.sha256(original.encode("utf-8")).hexdigest()[:32] + Path(original).suffix.lower()
                if relative not in files:
                    file_path = Path(original)
                    if not file_path.is_file() or file_path.stat().st_size <= 0:
                        raise ValueError(f"row {row_index}: audited raw file is missing/empty: {original}")
                    files[relative] = {"source": original, "bytes": file_path.stat().st_size}
                elif files[relative]["source"] != original:
                    raise RuntimeError("staging filename hash collision")
                paths.append(relative)
                used[key] = original
            if "s1_raw_paths" in row or "s1_source" in row:
                raise ValueError("source manifest already contains job-local staging fields")
            augmented = dict(row, s1_raw_paths=paths, s1_source=source)
            index.write(struct.pack("<Q", output.tell()))
            output.write(json.dumps(augmented, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
            counts[source] += 1
            if not probes.setdefault(source, []):
                probes[source].append(row_index)
            rows += 1
            if rows % 100000 == 0:
                print(f"prepared {rows:,} rows, {len(files):,} unique raw files", flush=True)
    if strict_counts and dict(counts) != SOURCE_COUNTS:
        raise ValueError(f"filtered manifest source counts changed: expected={SOURCE_COUNTS}, actual={dict(counts)}")
    report = {"manifest": str(Path(manifest).resolve()), "metadata": str(metadata.resolve()),
              "rows": rows, "source_rows": dict(counts), "probe_rows": probes,
              "rows_canonical_sha256": row_hash.hexdigest(),
              "mapping_sha256": digest([(field, raw, resolved) for (field, raw), resolved in sorted(used.items())]),
              "unique_raw_files": len(files), "raw_bytes": sum(x["bytes"] for x in files.values()),
              "identity_note": "canonical unchanged original rows; not a byte hash of the source JSON file"}
    return report, files


def smoke_rows(rows):
    """The actual first eight microbatches on each contiguous S1 rank shard."""
    import torch
    order = data_order(rows)
    permutation = torch.randperm(rows, generator=torch.Generator().manual_seed(SEED)).tolist()
    per_rank = rows // WORLD_SIZE
    return {str(rank): [order[i] for i in permutation[rank * per_rank:rank * per_rank + 8 * BATCH_SIZE]]
            for rank in range(WORLD_SIZE)}


def select_files(prepared, files, expected_rows, mode):
    """Stage only rows used by training or the trainer's one-row/source probes."""
    if mode == "formal":
        return files
    if mode != "smoke":
        raise ValueError(f"unknown staging mode: {mode}")
    selected = {x for indices in expected_rows.values() for x in indices}
    selected.update(indices[0] for indices in prepared["probe_rows"].values())
    indexed = IndexedRows(prepared["metadata"])
    try:
        names = {p for i in selected for p in indexed[i]["s1_raw_paths"]}
    finally:
        indexed.close()
    return {p: files[p] for p in sorted(names)}


def validate_stage_root(path):
    path = Path(path)
    if (not path.is_absolute() or path.parent != Path("/dev/shm")
            or not path.name.startswith("mizar_s1_") or path.is_symlink()
            or path.resolve() != path):
        raise ValueError("staging root must be a nonsymlink direct child /dev/shm/mizar_s1_<unique-id>")
    return path


def cleanup(path, token):
    root = validate_stage_root(path)
    if not root.exists():
        return
    marker = json.loads((root / ".owner.json").read_text(encoding="utf-8"))
    if marker != {"token": token, "root": str(root)}:
        raise ValueError("refusing cleanup: this job does not own the staging directory")
    shutil.rmtree(root)


def memory_available():
    """Cgroup v2/v1 remaining allocation, if exposed by the container runtime."""
    for limit, usage in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            value = Path(limit).read_text().strip()
            if value != "max":
                return max(0, int(value) - int(Path(usage).read_text().strip()))
        except (OSError, ValueError):
            pass
    return None


def check_capacity(root, required, reserve):
    disk_free = shutil.disk_usage(root).free
    memory_free = memory_available()
    available = min(disk_free, memory_free) if memory_free is not None else disk_free
    result = {"raw_bytes_to_copy": required, "reserve_bytes": reserve,
              "tmpfs_free_bytes": disk_free, "cgroup_remaining_bytes": memory_free}
    if required + reserve > available:
        raise RuntimeError(f"insufficient staging capacity: {result}; need {required + reserve:,} available bytes")
    return result


def copy_files(root, files):
    def copy_one(entry):
        relative, info = entry
        target = root / relative
        shutil.copyfile(info["source"], target)
        if target.stat().st_size != info["bytes"]:
            raise IOError(f"short staging copy: {info['source']}")

    entries = iter(files.items())
    done = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        while True:
            batch = []
            for _ in range(512):
                entry = next(entries, None)
                if entry is None:
                    break
                batch.append(entry)
            if not batch:
                break
            list(pool.map(copy_one, batch))
            done += len(batch)
            if done % 5120 == 0 or done == len(files):
                print(f"staged {done:,}/{len(files):,} original audio files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--resolved-paths", type=Path)
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--token", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    if args.cleanup:
        cleanup(args.stage_root, args.token)
        return
    if not all((args.manifest, args.resolved_paths, args.output)):
        parser.error("--manifest, --resolved-paths and --output are required for staging")
    import torch
    torch.set_num_threads(1)
    root = validate_stage_root(args.stage_root)
    root.mkdir(exist_ok=False)
    write_json(root / ".owner.json", {"token": args.token, "root": str(root)})
    (root / "audio").mkdir()
    reserve = 32 * 1024**3
    check_capacity(root, 0, reserve)
    start = time.monotonic()
    print("[staging] loading audited path mappings (no audio decoding)", flush=True)
    resolutions = load_resolutions(args.resolved_paths)
    mapped = time.monotonic()
    print(f"[staging] mappings loaded in {mapped - start:.1f}s; indexing manifest", flush=True)
    prepared, files = prepare(args.manifest, resolutions, root / "metadata" / "rows.jsonl")
    del resolutions
    prepared["geometry"] = geometry(prepared["rows"])
    expected_rows = smoke_rows(prepared["rows"])
    to_copy = select_files(prepared, files, expected_rows, args.mode)
    prepared.update({"mode": args.mode, "stage_root": str(root), "expected_first8_rows": expected_rows,
                     "staged_files": len(to_copy), "staged_bytes": sum(x["bytes"] for x in to_copy.values()),
                     "capacity": check_capacity(root, sum(x["bytes"] for x in to_copy.values()), reserve),
                     "audio_processing": {"decode_during_staging": False, "fixed20_during_staging": False,
                                          "encoder_feature_cache": False, "processing_phase": "training_data_loader"}})
    indexed = time.monotonic()
    print(f"[staging] index/selection ready in {indexed - mapped:.1f}s; copying "
          f"{len(to_copy):,} raw files ({prepared['staged_bytes'] / 1024**3:.2f} GiB)", flush=True)
    copy_files(root, to_copy)
    copied = time.monotonic()
    prepared["phase_seconds"] = {"load_mappings": mapped - start, "index_and_select": indexed - mapped,
                                 "copy_raw_files": copied - indexed, "total": copied - start}
    prepared["status"] = "PASS"
    write_json(args.output, prepared)
    write_json(root / "READY.json", prepared)
    print(json.dumps({"status": "PASS", "mode": args.mode, "rows": prepared["rows"],
                      "staged_files": len(to_copy), "phase_seconds": prepared["phase_seconds"],
                      "report": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
