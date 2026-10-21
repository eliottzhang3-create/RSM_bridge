"""Resolve the filtered S1 rows using the completed audit and stage raw files.

Audio is copied as raw bytes without decoding, resampling or cropping; no encoder
feature is cached. The JSONL/index are job-local views of the unchanged source
manifest, shared by all eight ranks. Source probes run later in the trainer.
"""
from __future__ import annotations

import argparse
from array import array
import csv
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from contract import (BATCH_SIZE, EPOCHS, ROUTE, SEED, SOURCE_COUNTS, WORLD_SIZE, data_order,
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


def prepare(manifest, resolutions, metadata, *, strict_counts=True, row_priorities=None):
    """Stream all retained rows, preserving text/fields and deduplicating raw files."""
    metadata = Path(metadata)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    counts, probes, files, used = Counter(), {}, {}, {}
    row_hash = hashlib.sha256()
    rows = 0
    with metadata.open("xb") as output, Path(str(metadata) + ".idx").open("xb") as index:
        for row_index, row in enumerate(iter_rows(Path(manifest))):
            source, refs = raw_references(row)
            priority = (-1 if source not in probes else row_priorities[row_index]) if row_priorities is not None else None
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
                if priority is not None:
                    files[relative]["first_batch"] = min(priority, files[relative].get("first_batch", priority))
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
    if row_priorities is not None and len(row_priorities) != rows:
        raise ValueError("staging priority row count differs from the filtered manifest")
    if strict_counts and dict(counts) != SOURCE_COUNTS:
        raise ValueError(f"filtered manifest source counts changed: expected={SOURCE_COUNTS}, actual={dict(counts)}")
    report = {"manifest": str(Path(manifest).resolve()), "metadata": str(metadata.resolve()),
              "rows": rows, "source_rows": dict(counts), "probe_rows": probes,
              "rows_canonical_sha256": row_hash.hexdigest(),
              "mapping_sha256": digest([(field, raw, resolved) for (field, raw), resolved in sorted(used.items())]),
              "unique_raw_files": len(files), "raw_bytes": sum(x["bytes"] for x in files.values()),
              "identity_note": "canonical unchanged original rows; not a byte hash of the source JSON file"}
    return report, files


def epoch_priorities(rows, epoch=0, start_batch=0, *, permutation=None):
    """Earliest batch needing each original row, following all eight real shards.

    The sentinel is steps_per_epoch: consumed/dropped rows are copied last.
    A private generator avoids consuming training/text RNG. permutation is a
    dependency-free test hook; production always uses the real torch randperm.
    """
    order = data_order(rows)
    if permutation is None:
        import torch
        permutation = torch.randperm(rows, generator=torch.Generator().manual_seed(SEED + epoch)).tolist()
    per_rank = rows // WORLD_SIZE
    steps = per_rank // BATCH_SIZE
    if not 0 <= start_batch < steps or epoch < 0 or len(permutation) != rows:
        raise ValueError("invalid staging sampler/cursor")
    priorities = array("i", [steps]) * rows
    first8 = {}
    for rank in range(WORLD_SIZE):
        offset = rank * per_rank
        first8[str(rank)] = [order[i] for i in permutation[offset:offset + 8 * BATCH_SIZE]]
        for cursor in range(start_batch * BATCH_SIZE, steps * BATCH_SIZE):
            priorities[order[permutation[offset + cursor]]] = cursor // BATCH_SIZE
    return priorities, first8


def smoke_rows(rows):
    return epoch_priorities(rows)[1]


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


def copy_one(root, entry):
    relative, info = entry
    target = root / relative
    temporary = target.with_name(target.name + ".partial")
    try:
        shutil.copyfile(info["source"], temporary)
        if temporary.stat().st_size != info["bytes"]:
            raise IOError(f"short staging copy: {info['source']}")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return info["bytes"]


def copy_files(root, files):
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
            list(pool.map(lambda entry: copy_one(root, entry), batch))
            done += len(batch)
            if done % 5120 == 0 or done == len(files):
                print(f"staged {done:,}/{len(files):,} original audio files", flush=True)


def staging_plan(files, steps, start_batch, requested_steps, *, smoke=False):
    initial_steps = min(requested_steps, steps - start_batch, 2 if smoke else steps)
    if initial_steps < 1:
        raise ValueError("staging initial steps must be positive")
    ordered = sorted(files.items(), key=lambda item: (item[1]["first_batch"], item[0]))
    budgets = []
    cursor, byte_count = 0, 0
    candidates = sorted({min(value, steps - start_batch) for value in
                         (128, 256, 512, 1024, 2048, initial_steps)})
    for count in candidates:
        while cursor < len(ordered) and ordered[cursor][1]["first_batch"] < start_batch + count:
            byte_count += ordered[cursor][1]["bytes"]
            cursor += 1
        budgets.append({"steps": count, "files": cursor, "bytes": byte_count})
    return ordered, initial_steps, budgets


def copy_background(root, prepared, ordered, output, state, *, workers=16, window_steps=128):
    """Publish a global contiguous batch watermark only after atomic copies.

    Queue depth is bounded at 512 files. The initial step budget is a strict
    boundary: trainer startup is published before enqueuing the later window.
    A single producer owns all copying; no rank independently duplicates files.
    """
    root, output = Path(root), Path(output)
    status_path = root / "STAGING.json"
    start, last_publish, cursor = time.monotonic(), 0., 0
    steps, start_batch = prepared["staging_epoch_steps"], state["start_batch"]
    initial_end = start_batch + prepared["initial_steps"]
    state.update(phase="PREFILL", copied_files=0, copied_bytes=0, ready_through_batch=start_batch)

    def publish(force=False):
        nonlocal last_publish
        now = time.monotonic()
        if force or now - last_publish >= 5.:
            state.update(heartbeat=time.time(), copy_seconds=now - start,
                         cgroup_remaining_bytes=memory_available(), tmpfs_free_bytes=shutil.disk_usage(root).free)
            write_json(status_path, state)
            write_json(output.with_name("staging_progress.json"), state)
            seconds = max(now - start, .001)
            print(f"[staging] phase={state['phase']} staged {state['copied_files']:,}/{len(ordered):,} "
                  f"files ({state['copied_bytes'] / 1024**3:.2f}/{prepared['planned_bytes'] / 1024**3:.2f} GiB) "
                  f"ready_through_batch={state['ready_through_batch']} "
                  f"MiB_per_second={state['copied_bytes'] / 1024**2 / seconds:.1f} "
                  f"cgroup_remaining_bytes={state['cgroup_remaining_bytes']}", flush=True)
            last_publish = now

    def report(status):
        prepared.update(status=status, staged_files=state["copied_files"], staged_bytes=state["copied_bytes"])
        prepared["phase_seconds"]["copy_raw_files"] = time.monotonic() - start
        prepared["phase_seconds"]["total"] = prepared["precopy_seconds"] + time.monotonic() - start
        write_json(output, prepared)
        if status == "PASS":
            write_json(root / "READY.json", prepared)

    def copy_until(end, pool):
        nonlocal cursor
        while cursor < end:
            next_cursor = min(cursor + 512, end)
            pending = {pool.submit(copy_one, root, entry) for entry in ordered[cursor:next_cursor]}
            while pending:
                done, pending = wait(pending, timeout=5., return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        copied_bytes = future.result()
                    except BaseException as exc:
                        state.update(phase="FAILED", error=f"{type(exc).__name__}: {exc}")
                        publish(True)
                        for other in pending:
                            other.cancel()
                        raise
                    state["copied_bytes"] += copied_bytes
                    state["copied_files"] += 1
                publish()
            cursor = next_cursor

    try:
        publish(True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            end_batch = initial_end
            while True:
                end = cursor
                while end < len(ordered) and ordered[end][1]["first_batch"] < end_batch:
                    end += 1
                copy_until(end, pool)
                state["ready_through_batch"] = end_batch
                if end_batch == initial_end:
                    state["phase"] = "COPYING"
                    prepared["startup_seconds"] = prepared["precopy_seconds"] + time.monotonic() - start
                    publish(True)
                    report("TRAINING_READY")
                    print(f"[staging] TRAINING_READY: first {prepared['initial_steps']} steps available; "
                          f"background copy continues; startup_seconds={prepared['startup_seconds']:.1f}", flush=True)
                else:
                    publish(True)
                if end_batch == steps:
                    break
                end_batch = min(steps, end_batch + window_steps)
            # Retained files referenced only by consumed/dropped rows are still
            # required for later epochs. Copy them before announcing COMPLETE.
            copy_until(len(ordered), pool)
        state["phase"] = "COMPLETE"
        report("PASS")
        publish(True)
    except BaseException as exc:
        state.update(phase="FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        publish(True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--resolved-paths", type=Path)
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--token", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--start-steps", type=int, default=512)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--window-steps", type=int, default=128)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    if args.cleanup:
        cleanup(args.stage_root, args.token)
        return
    if not all((args.manifest, args.resolved_paths, args.output)):
        parser.error("--manifest, --resolved-paths and --output are required for staging")
    if min(args.start_steps, args.workers, args.window_steps) < 1:
        parser.error("--start-steps, --workers and --window-steps must be positive")
    import torch
    torch.set_num_threads(1)
    root = validate_stage_root(args.stage_root)
    root.mkdir(exist_ok=False)
    write_json(root / ".owner.json", {"token": args.token, "root": str(root)})
    (root / "audio").mkdir()
    state = {"schema_version": 1, "phase": "INDEXING", "token": args.token, "producer_pid": os.getpid(),
             "epoch": 0, "start_batch": 0, "ready_through_batch": 0,
             "copied_files": 0, "copied_bytes": 0, "heartbeat": time.time()}
    write_json(root / "STAGING.json", state)
    write_json(args.output.with_name("staging_progress.json"), state)
    try:
        run_stage(args, root, state)
    except BaseException as exc:
        state.update(phase="FAILED", heartbeat=time.time(), error=f"{type(exc).__name__}: {exc}",
                     traceback=traceback.format_exc())
        write_json(root / "STAGING.json", state)
        write_json(args.output.with_name("staging_progress.json"), state)
        raise


def run_stage(args, root, state):
    reserve = 32 * 1024**3
    check_capacity(root, 0, reserve)
    start = time.monotonic()
    row_priorities = None
    if args.background:
        if args.resume_checkpoint:
            import torch
            saved = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
            if saved.get("route_variant") != ROUTE or saved.get("mode") != "formal" or args.mode != "formal":
                raise ValueError("staging resume requires this route's formal checkpoint")
            state["epoch"], state["start_batch"] = saved["epoch"], saved["next_batch"]
            if not 0 <= state["epoch"] < EPOCHS:
                raise ValueError("cannot stage a checkpoint whose training is already complete")
            del saved
        row_priorities, expected_rows = epoch_priorities(
            sum(SOURCE_COUNTS.values()), state["epoch"], state["start_batch"])
        if state["epoch"]:
            expected_rows = smoke_rows(sum(SOURCE_COUNTS.values()))
        write_json(root / "STAGING.json", dict(state, heartbeat=time.time()))
    print("[staging] loading audited path mappings (no audio decoding)", flush=True)
    resolutions = load_resolutions(args.resolved_paths)
    mapped = time.monotonic()
    print(f"[staging] mappings loaded in {mapped - start:.1f}s; indexing manifest", flush=True)
    prepared, files = prepare(args.manifest, resolutions, root / "metadata" / "rows.jsonl", row_priorities=row_priorities)
    del resolutions
    del row_priorities
    prepared["geometry"] = geometry(prepared["rows"])
    if not args.background:
        expected_rows = smoke_rows(prepared["rows"])
    to_copy = select_files(prepared, files, expected_rows, args.mode)
    prepared.update({"mode": args.mode, "stage_root": str(root), "expected_first8_rows": expected_rows,
                     "staged_files": len(to_copy), "staged_bytes": sum(x["bytes"] for x in to_copy.values()),
                     "capacity": check_capacity(root, sum(x["bytes"] for x in to_copy.values()), reserve),
                     "audio_processing": {"decode_during_staging": False, "fixed20_during_staging": False,
                                          "encoder_feature_cache": False, "processing_phase": "training_data_loader"}})
    indexed = time.monotonic()
    if args.background:
        steps = 8 if args.mode == "smoke" else prepared["geometry"]["steps_per_epoch"]
        ordered, initial_steps, budgets = staging_plan(
            to_copy, steps, state["start_batch"], args.start_steps, smoke=args.mode == "smoke")
        prepared.update(planned_files=len(to_copy), planned_bytes=prepared["staged_bytes"],
                        staging_epoch_steps=steps, initial_steps=initial_steps,
                        startup_budgets=budgets, copy_workers=args.workers,
                        precopy_seconds=indexed - start)
        prepared["async_staging"] = {key: state[key] for key in ("token", "producer_pid", "epoch", "start_batch")}
        prepared["async_staging"]["status_path"] = str(root / "STAGING.json")
        prepared["phase_seconds"] = {"load_mappings": mapped - start, "index_and_select": indexed - mapped}
        print(json.dumps({"event": "STAGING_PLAN", "epoch": state["epoch"] + 1,
                          "start_batch": state["start_batch"], "initial_steps": initial_steps,
                          "workers": args.workers, "budgets": budgets,
                          "planned_files": len(to_copy), "planned_GiB": prepared["planned_bytes"] / 1024**3}), flush=True)
        del files, to_copy
        copy_background(root, prepared, ordered, args.output, state, workers=args.workers,
                        window_steps=1 if args.mode == "smoke" else args.window_steps)
        return
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
