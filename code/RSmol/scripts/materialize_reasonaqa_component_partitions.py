#!/usr/bin/env python3
"""Materialize an audited zero-copy ReasonAQA plan into local waveform stores.

This CPU-only tool performs one sequential pass over the immutable global
waveform store and writes one self-contained ``rows.jsonl + index.jsonl +
waveforms.f32`` store per planned partition.  It does not decode audio and it
does not change the plan.  Interrupted copies are resumable at a durable global
audio-row boundary.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plan_reasonaqa_component_partitions import (  # noqa: E402
    FORMAT as PLAN_FORMAT,
    STORE_FORMAT,
    audio_path,
    iter_rows,
    normalized_path,
    sha256_file,
)


FORMAT = "reasonaqa_component_partition_stores_v1"
DATA_FILE = "waveforms.f32"
BYTES_PER_AUDIO = 1_280_000
DEFAULT_PLAN_DIR = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "reasonaqa_component_partitions6_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/"
    "rsmol_reasonaqa_train_component_partitions6_32k_10s_f32_v1"
)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _validate_source_store(store: Path, plan: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if (store / "BUILDING").exists():
        raise ValueError("source waveform store is still BUILDING")
    metadata = _read_json(store / "metadata.json")
    expected = {
        "status": "PASS",
        "format": STORE_FORMAT,
        "manifest_sha256": plan["manifest_sha256"],
        "sample_rate": 32_000,
        "seconds": 10,
        "samples_per_audio": 320_000,
        "bytes_per_audio": BYTES_PER_AUDIO,
        "dtype": "float32",
        "byte_order": "little",
        "data_file": DATA_FILE,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"source store metadata mismatch: {key}")
    if metadata.get("waveform_verification", {}).get("passed") is not True:
        raise ValueError("source store waveform verification is not PASS")
    index = store / "index.jsonl"
    index_sha = sha256_file(index)
    if index_sha != metadata.get("index_sha256") or index_sha != plan.get("index_sha256"):
        raise ValueError("source store index SHA256 mismatch")
    if metadata.get("waveform_sha256") != plan.get("waveform_sha256_from_store_metadata"):
        raise ValueError("source waveform provenance SHA256 disagrees with plan")
    entries: list[dict[str, Any]] = []
    for _, _, row, _ in iter_rows(index):
        aid = len(entries)
        if (row.get("audio_id") != aid or row.get("byte_offset") != aid * BYTES_PER_AUDIO
                or row.get("byte_length") != BYTES_PER_AUDIO or row.get("shape") != [1, 320000]
                or row.get("dtype") != "float32"):
            raise ValueError(f"source index contract mismatch: audio_id={aid}")
        entries.append(row)
    expected_bytes = len(entries) * BYTES_PER_AUDIO
    data = store / DATA_FILE
    if (not entries or metadata.get("num_unique_audio_files") != len(entries)
            or metadata.get("total_waveform_bytes") != expected_bytes
            or not data.is_file() or data.stat().st_size != expected_bytes):
        raise ValueError("source store cardinality/data-size mismatch")
    return metadata, entries


def load_contract(plan_dir: Path, source_override: Path | None = None) -> dict[str, Any]:
    plan_dir = plan_dir.expanduser().resolve(strict=True)
    if (plan_dir / "BUILDING").exists():
        raise ValueError("partition plan is still BUILDING")
    plan_path = plan_dir / "partition_plan.json"
    audit_path = plan_dir / "partition_audit.json"
    plan, audit = _read_json(plan_path), _read_json(audit_path)
    if plan.get("format") != PLAN_FORMAT or audit.get("format") != PLAN_FORMAT or audit.get("status") != "PASS":
        raise ValueError("partition plan/audit is not planning-integrity PASS")
    if sha256_file(plan_path) != audit.get("partition_plan_sha256"):
        raise ValueError("partition plan SHA256 mismatch")
    num_partitions = int(plan.get("config", {}).get("num_partitions", -1))
    if num_partitions < 2 or len(plan.get("partitions", [])) != num_partitions:
        raise ValueError("invalid partition count in plan")
    if (audit.get("duplicated_audio") != 0 or audit.get("split_components") != 0
            or audit.get("cross_partition_qa") != 0
            or audit.get("all_rows_assigned_exactly_once") is not True
            or audit.get("all_audio_assigned_exactly_once") is not True):
        raise ValueError("plan is not a zero-copy/local-reference assignment")
    artifacts = plan.get("artifacts", {})
    required = ["row_assignments.jsonl"]
    for p in range(num_partitions):
        required += [f"partition_{p}_rows.jsonl", f"partition_{p}_audio.jsonl"]
    for name in required:
        path = plan_dir / name
        expected = artifacts.get(name)
        if not path.is_file() or not isinstance(expected, dict):
            raise ValueError(f"plan artifact is missing: {name}")
        if path.stat().st_size != expected.get("size_bytes") or sha256_file(path) != expected.get("sha256"):
            raise ValueError(f"plan artifact hash/size mismatch: {name}")
    store = (source_override.expanduser().resolve(strict=True) if source_override is not None
             else Path(plan["store_dir"]).expanduser().resolve(strict=True))
    source_metadata, source_entries = _validate_source_store(store, plan)

    owner = [-1] * len(source_entries)
    local_id = [-1] * len(source_entries)
    partition_entries: list[list[dict[str, Any]]] = []
    for p in range(num_partitions):
        rows = [row for _, _, row, _ in iter_rows(plan_dir / f"partition_{p}_audio.jsonl")]
        expected_summary = plan["partitions"][p]
        if len(rows) != expected_summary.get("unique_audio"):
            raise ValueError(f"partition {p} audio cardinality mismatch")
        previous_global = -1
        for local, row in enumerate(rows):
            global_id = row.get("audio_id")
            if not isinstance(global_id, int) or not 0 <= global_id < len(source_entries):
                raise ValueError(f"partition {p} has invalid global audio_id")
            if global_id <= previous_global or owner[global_id] != -1:
                raise ValueError("partition audio ownership is duplicated or not in global order")
            previous_global = global_id
            source = source_entries[global_id]
            for key in ("source_path", "source_group", "source_size_bytes", "source_mtime_ns",
                        "manifest_aliases", "slot_reference_count", "qa_incidence_count",
                        "byte_offset", "byte_length", "shape", "dtype"):
                if row.get(key) != source.get(key):
                    raise ValueError(f"partition {p} audio entry differs from source: audio_id={global_id} key={key}")
            if row.get("partition_id") != p:
                raise ValueError("partition audio entry has incorrect partition_id")
            owner[global_id], local_id[global_id] = p, local
        partition_entries.append(rows)
    if any(p < 0 for p in owner) or len(owner) != audit.get("unique_audio"):
        raise ValueError("partition plan omits source audio")
    if sum(len(x) for x in partition_entries) != len(source_entries):
        raise ValueError("partition plan audio cardinality is not zero-copy")

    # Recheck that every planned QA resolves to two audio rows in its partition.
    aliases: dict[str, int] = {}
    for aid, source in enumerate(source_entries):
        for value in [source["source_path"], *source.get("manifest_aliases", [])]:
            key = normalized_path(str(value))
            previous = aliases.setdefault(key, aid)
            if previous != aid:
                raise ValueError(f"source store alias collision: {key}")
    qa_total = 0
    for p in range(num_partitions):
        qa_count = 0
        for _, line_number, row, _ in iter_rows(plan_dir / f"partition_{p}_rows.jsonl"):
            first = audio_path(row, True)
            second = audio_path(row, False) or first
            if not first:
                raise ValueError(f"partition {p} row lacks audio1 at line {line_number}")
            try:
                a, b = aliases[normalized_path(first)], aliases[normalized_path(second)]
            except KeyError as exc:
                raise ValueError(f"partition {p} row references absent audio: {exc}") from exc
            if owner[a] != p or owner[b] != p:
                raise ValueError(f"partition {p} row has cross-partition audio")
            qa_count += 1
        if qa_count != plan["partitions"][p].get("qa_rows"):
            raise ValueError(f"partition {p} QA cardinality mismatch")
        qa_total += qa_count
    if qa_total != audit.get("qa_rows"):
        raise ValueError("partition QA total mismatch")
    return {
        "plan_dir": plan_dir,
        "plan": plan,
        "audit": audit,
        "plan_sha256": sha256_file(plan_path),
        "source_store": store,
        "source_metadata": source_metadata,
        "source_entries": source_entries,
        "owner": owner,
        "local_id": local_id,
        "partition_entries": partition_entries,
        "num_partitions": num_partitions,
    }


def _local_index_entry(source: dict[str, Any], partition: int, local: int) -> dict[str, Any]:
    global_id = int(source["audio_id"])
    return {
        **source,
        "audio_id": local,
        "global_audio_id": global_id,
        "global_byte_offset": int(source["byte_offset"]),
        "partition_id": partition,
        "byte_offset": local * BYTES_PER_AUDIO,
    }


def _partition_source_inventory_sha256(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    keys = ("source_path", "source_group", "source_size_bytes", "source_mtime_ns",
            "manifest_aliases", "slot_reference_count", "qa_incidence_count")
    for entry in entries:
        digest.update(json.dumps({key: entry[key] for key in keys}, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _build_config(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "plan_format": PLAN_FORMAT,
        "plan_dir": str(contract["plan_dir"]),
        "partition_plan_sha256": contract["plan_sha256"],
        "balance_status_accepted": contract["audit"].get("balance_status"),
        "source_store_dir": str(contract["source_store"]),
        "source_index_sha256": contract["source_metadata"]["index_sha256"],
        "source_waveform_sha256": contract["source_metadata"]["waveform_sha256"],
        "source_total_waveform_bytes": contract["source_metadata"]["total_waveform_bytes"],
        "source_num_unique_audio": len(contract["source_entries"]),
        "num_partitions": contract["num_partitions"],
        "bytes_per_audio": BYTES_PER_AUDIO,
        "copy_order": "single sequential global audio_id pass; append to owning partition",
    }


def _initialize_output(output: Path, contract: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    (output / "BUILDING").write_text("incomplete partition materialization\n", encoding="utf-8")
    partition_config = []
    for p, entries in enumerate(contract["partition_entries"]):
        directory = output / f"partition_{p}"
        directory.mkdir()
        (directory / "BUILDING").write_text("incomplete partition waveform store\n", encoding="utf-8")
        shutil.copyfile(contract["plan_dir"] / f"partition_{p}_rows.jsonl", directory / "rows.jsonl")
        index_partial = directory / ".index.jsonl.partial"
        with index_partial.open("w", encoding="utf-8", newline="\n") as handle:
            for local, planned in enumerate(entries):
                source = contract["source_entries"][planned["audio_id"]]
                handle.write(json.dumps(_local_index_entry(source, p, local), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(index_partial, directory / "index.jsonl")
        partition_config.append({
            "partition_id": p,
            "qa_rows": contract["plan"]["partitions"][p]["qa_rows"],
            "num_unique_audio_files": len(entries),
            "total_waveform_bytes": len(entries) * BYTES_PER_AUDIO,
            "manifest_sha256": sha256_file(directory / "rows.jsonl"),
            "index_sha256": sha256_file(directory / "index.jsonl"),
        })
    config = {**_build_config(contract), "partitions": partition_config}
    write_json_atomic(output / "build_config.json", config)
    write_json_atomic(output / "progress.json", {
        "status": "BUILDING",
        "completed_global_audio": 0,
        "per_partition_completed_audio": [0] * contract["num_partitions"],
        "updated_unix": time.time(),
    })
    return config


def _validate_initialized(output: Path, contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not (output / "BUILDING").is_file():
        report = output / "materialization_report.json"
        if report.is_file() and _read_json(report).get("status") == "PASS":
            raise RuntimeError("output is already PASS; no resume is needed")
        raise RuntimeError("output is neither a resumable BUILDING directory nor PASS")
    config = _read_json(output / "build_config.json")
    expected_base = _build_config(contract)
    for key, value in expected_base.items():
        if config.get(key) != value:
            raise RuntimeError(f"resume build configuration mismatch: {key}")
    if len(config.get("partitions", [])) != contract["num_partitions"]:
        raise RuntimeError("resume partition configuration mismatch")
    for p, item in enumerate(config["partitions"]):
        directory = output / f"partition_{p}"
        if not (directory / "BUILDING").is_file():
            metadata_path = directory / "metadata.json"
            data_path = directory / DATA_FILE
            if (not metadata_path.is_file() or _read_json(metadata_path).get("status") != "PASS"
                    or not data_path.is_file() or data_path.stat().st_size != item["total_waveform_bytes"]):
                raise RuntimeError(f"partition {p} is neither resumable BUILDING nor finalized PASS")
        if sha256_file(directory / "rows.jsonl") != item["manifest_sha256"]:
            raise RuntimeError(f"partition {p} manifest changed during resume")
        if sha256_file(directory / "index.jsonl") != item["index_sha256"]:
            raise RuntimeError(f"partition {p} index changed during resume")
    progress = _read_json(output / "progress.json")
    if progress.get("status") not in {"BUILDING", "PASS"}:
        raise RuntimeError("resume progress is neither BUILDING nor interrupted-finalization PASS")
    if progress.get("status") == "PASS":
        expected_counts = [len(x) for x in contract["partition_entries"]]
        if (progress.get("completed_global_audio") != len(contract["owner"])
                or progress.get("per_partition_completed_audio") != expected_counts):
            raise RuntimeError("PASS progress does not describe a complete interrupted finalization")
    return config, progress


def _prefix_counts(owner: list[int], completed: int, num_partitions: int) -> list[int]:
    counts = [0] * num_partitions
    for p in owner[:completed]:
        counts[p] += 1
    return counts


def _reconcile_partials(output: Path, config: dict[str, Any], progress: dict[str, Any], owner: list[int]) -> int:
    completed = int(progress.get("completed_global_audio", -1))
    if not 0 <= completed <= len(owner):
        raise RuntimeError("resume global progress is out of range")
    counts = _prefix_counts(owner, completed, len(config["partitions"]))
    if progress.get("per_partition_completed_audio") != counts:
        raise RuntimeError("resume per-partition progress disagrees with global cursor")
    for p, count in enumerate(counts):
        partial = output / f"partition_{p}" / f".{DATA_FILE}.partial"
        final = output / f"partition_{p}" / DATA_FILE
        expected = count * BYTES_PER_AUDIO
        if final.exists():
            if completed != len(owner) or final.stat().st_size != config["partitions"][p]["total_waveform_bytes"]:
                raise RuntimeError(f"partition {p} has premature/invalid final data")
            continue
        if not partial.exists():
            if expected:
                raise RuntimeError(f"partition {p} partial data is missing")
            partial.touch()
        if partial.stat().st_size < expected:
            raise RuntimeError(f"partition {p} partial data is shorter than durable progress")
        if partial.stat().st_size != expected:
            with partial.open("r+b") as handle:
                handle.truncate(expected)
                handle.flush()
                os.fsync(handle.fileno())
    return completed


def _hash_source_prefix(handle, byte_count: int, digest) -> None:
    remaining = byte_count
    while remaining:
        payload = handle.read(min(8 * 1024 * 1024, remaining))
        if not payload:
            raise RuntimeError("source payload truncated while reconstructing resume checksum")
        digest.update(payload)
        remaining -= len(payload)


def _verify_partition_samples(source_data: Path, partition_data: Path, index_entries: list[dict[str, Any]],
                              requested: int) -> dict[str, Any]:
    count = min(max(1, int(requested)), len(index_entries))
    locals_to_check = ([0] if count == 1 else
                       sorted({i * (len(index_entries) - 1) // (count - 1) for i in range(count)}))
    checked = 0
    with source_data.open("rb") as source, partition_data.open("rb") as target:
        for local in locals_to_check:
            global_id = int(index_entries[local]["audio_id"])
            source.seek(global_id * BYTES_PER_AUDIO)
            target.seek(local * BYTES_PER_AUDIO)
            a, b = source.read(BYTES_PER_AUDIO), target.read(BYTES_PER_AUDIO)
            if len(a) != BYTES_PER_AUDIO or a != b:
                raise RuntimeError(f"partition sampled-byte mismatch: local={local} global={global_id}")
            checked += len(a)
    return {"method": "evenly spaced local rows compared byte-for-byte with global source rows",
            "requested_samples": int(requested), "verified_samples": len(locals_to_check),
            "verified_local_audio_ids": locals_to_check, "verified_bytes": checked, "passed": True}


def materialize(contract: dict[str, Any], output: Path, *, resume: bool, checkpoint_every: int,
                verify_samples_per_partition: int, free_space_margin_gib: float) -> dict[str, Any]:
    if checkpoint_every <= 0 or verify_samples_per_partition <= 0 or free_space_margin_gib < 0:
        raise ValueError("checkpoint/verification counts must be positive and free-space margin non-negative")
    output = output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        config, progress = _validate_initialized(output, contract)
    else:
        if resume:
            raise FileNotFoundError(f"cannot resume absent output: {output}")
        config = _initialize_output(output, contract)
        progress = _read_json(output / "progress.json")
    completed = _reconcile_partials(output, config, progress, contract["owner"])
    existing_bytes = sum(completed_count * BYTES_PER_AUDIO for completed_count in _prefix_counts(
        contract["owner"], completed, contract["num_partitions"]))
    required = int(config["source_total_waveform_bytes"]) - existing_bytes
    free = shutil.disk_usage(output).free
    required_with_margin = required + int(free_space_margin_gib * 1024**3)
    if free < required_with_margin:
        raise RuntimeError(
            f"insufficient free space: available_gib={free / 1024**3:.3f} "
            f"remaining_plus_margin_gib={required_with_margin / 1024**3:.3f}"
        )
    source_path = contract["source_store"] / DATA_FILE
    started = time.perf_counter()
    try:
        source_digest = hashlib.sha256()
        if completed < len(contract["owner"]):
            with source_path.open("rb") as source, ExitStack() as stack:
                if completed:
                    _hash_source_prefix(source, completed * BYTES_PER_AUDIO, source_digest)
                outputs = []
                for p in range(contract["num_partitions"]):
                    path = output / f"partition_{p}" / f".{DATA_FILE}.partial"
                    outputs.append(stack.enter_context(path.open("ab", buffering=8 * 1024 * 1024)))
                counts = _prefix_counts(contract["owner"], completed, contract["num_partitions"])
                for global_id in range(completed, len(contract["owner"])):
                    payload = source.read(BYTES_PER_AUDIO)
                    if len(payload) != BYTES_PER_AUDIO:
                        raise RuntimeError(f"source payload truncated: global_audio_id={global_id}")
                    source_digest.update(payload)
                    p = contract["owner"][global_id]
                    outputs[p].write(payload)
                    counts[p] += 1
                    durable = global_id + 1
                    if durable % checkpoint_every == 0 or durable == len(contract["owner"]):
                        for handle in outputs:
                            handle.flush()
                            os.fsync(handle.fileno())
                        write_json_atomic(output / "progress.json", {
                            "status": "BUILDING", "completed_global_audio": durable,
                            "per_partition_completed_audio": counts,
                            "last_global_audio_id": durable - 1, "updated_unix": time.time(),
                        })
                    if durable % 1000 == 0 or durable == len(contract["owner"]):
                        print(f"[partition-materialize] copied={durable}/{len(contract['owner'])}", flush=True)
                if source.read(1):
                    raise RuntimeError("source payload has trailing bytes")
        else:
            with source_path.open("rb") as source:
                _hash_source_prefix(source, source_path.stat().st_size, source_digest)
        source_sha = source_digest.hexdigest()
        if source_sha != config["source_waveform_sha256"]:
            raise RuntimeError("source waveform SHA256 mismatch after sequential copy")

        partition_reports = []
        for p, item in enumerate(config["partitions"]):
            directory = output / f"partition_{p}"
            partial, final = directory / f".{DATA_FILE}.partial", directory / DATA_FILE
            if final.is_file():
                data = final
            else:
                if not partial.is_file() or partial.stat().st_size != item["total_waveform_bytes"]:
                    raise RuntimeError(f"partition {p} completed payload size mismatch")
                data = partial
            print(f"[partition-materialize] hashing partition={p}", flush=True)
            waveform_sha = sha256_file(data)
            verification = _verify_partition_samples(
                source_path, data, contract["partition_entries"][p], verify_samples_per_partition)
            if not final.is_file():
                os.replace(partial, final)
            metadata = {
                "format": STORE_FORMAT, "status": "PASS",
                "manifest": str(directory / "rows.jsonl"), "manifest_sha256": item["manifest_sha256"],
                "source_inventory_sha256": _partition_source_inventory_sha256(
                    [contract["source_entries"][entry["audio_id"]]
                     for entry in contract["partition_entries"][p]]),
                "source_global_inventory_sha256": contract["source_metadata"].get("source_inventory_sha256"),
                "num_unique_audio_files": item["num_unique_audio_files"],
                "sample_rate": 32000, "seconds": 10, "samples_per_audio": 320000,
                "bytes_per_audio": BYTES_PER_AUDIO, "dtype": "float32", "byte_order": "little",
                "data_file": DATA_FILE, "total_waveform_bytes": item["total_waveform_bytes"],
                "index_sha256": item["index_sha256"], "waveform_sha256": waveform_sha,
                "waveform_verification": verification,
                "partition_materialization_format": FORMAT, "partition_id": p,
                "partition_plan_sha256": config["partition_plan_sha256"],
                "source_store_dir": config["source_store_dir"],
                "source_waveform_sha256": source_sha,
                "layout": "one fixed-stride row-major raw file; partition-local audio_id and byte_offset",
                "preprocessing_contract": contract["source_metadata"].get("preprocessing_contract"),
                "created_unix": time.time(),
            }
            write_json_atomic(directory / "metadata.json", metadata)
            partition_reports.append({
                "partition_id": p, "qa_rows": item["qa_rows"],
                "num_unique_audio_files": item["num_unique_audio_files"],
                "total_waveform_bytes": item["total_waveform_bytes"],
                "total_waveform_gib": item["total_waveform_bytes"] / 1024**3,
                "manifest_sha256": item["manifest_sha256"], "index_sha256": item["index_sha256"],
                "waveform_sha256": waveform_sha, "waveform_verification": verification,
                "directory": str(directory),
            })
        report = {
            **config, "status": "PASS",
            "scope": "partition materialization integrity; not training/memory/performance PASS",
            "source_payload_sha256_reverified": True,
            "all_source_audio_assigned_once": True,
            "duplicated_audio": 0,
            "total_materialized_waveform_bytes": sum(x["total_waveform_bytes"] for x in partition_reports),
            "partitions": partition_reports,
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "created_unix": time.time(),
        }
        if report["total_materialized_waveform_bytes"] != config["source_total_waveform_bytes"]:
            raise AssertionError("materialized bytes differ from zero-copy source total")
        write_json_atomic(output / "materialization_report.json", report)
        write_json_atomic(output / "progress.json", {
            "status": "PASS", "completed_global_audio": len(contract["owner"]),
            "per_partition_completed_audio": [len(x) for x in contract["partition_entries"]],
            "updated_unix": time.time(),
        })
        for p in range(contract["num_partitions"]):
            (output / f"partition_{p}" / "BUILDING").unlink(missing_ok=True)
        (output / "BUILDING").unlink()
        error = output / "build_error.json"
        if error.exists():
            error.unlink()
        return report
    except Exception as exc:
        write_json_atomic(output / "build_error.json", {
            "status": "INCOMPLETE", "error": repr(exc), "traceback": traceback.format_exc(),
            "resume_required": True, "updated_unix": time.time(),
        })
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--source-store-dir", type=Path, default=None,
                        help="optional relocated source store; content hashes must match the plan")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--verify-samples-per-partition", type=int, default=32)
    parser.add_argument("--free-space-margin-gib", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate plan/store and report layout without creating output")
    args = parser.parse_args(argv)
    if args.checkpoint_every <= 0 or args.verify_samples_per_partition <= 0:
        parser.error("checkpoint and verification counts must be positive")
    if args.free_space_margin_gib < 0:
        parser.error("free-space-margin-gib must be non-negative")
    if args.dry_run and args.resume:
        parser.error("--dry-run and --resume are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_contract(args.plan_dir, args.source_store_dir)
    summary = {
        "status": "DRY_RUN" if args.dry_run else "READY",
        "plan_dir": str(contract["plan_dir"]), "balance_status_accepted": contract["audit"].get("balance_status"),
        "source_store_dir": str(contract["source_store"]),
        "num_partitions": contract["num_partitions"],
        "qa_rows": contract["audit"]["qa_rows"], "unique_audio": len(contract["source_entries"]),
        "total_waveform_gib": contract["source_metadata"]["total_waveform_bytes"] / 1024**3,
        "partitions": [{"partition_id": p, "qa_rows": contract["plan"]["partitions"][p]["qa_rows"],
                        "unique_audio": len(entries), "waveform_gib": len(entries) * BYTES_PER_AUDIO / 1024**3}
                       for p, entries in enumerate(contract["partition_entries"])],
        "output_dir": str(args.output_dir.expanduser().resolve(strict=False)),
    }
    print("[partition-materialize] " + json.dumps(summary, ensure_ascii=False), flush=True)
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    report = materialize(
        contract, args.output_dir, resume=args.resume, checkpoint_every=args.checkpoint_every,
        verify_samples_per_partition=args.verify_samples_per_partition,
        free_space_margin_gib=args.free_space_margin_gib,
    )
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir),
                      "source_payload_sha256_reverified": report["source_payload_sha256_reverified"],
                      "duplicated_audio": report["duplicated_audio"],
                      "total_waveform_gib": report["total_materialized_waveform_bytes"] / 1024**3,
                      "partitions": report["partitions"]}, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
