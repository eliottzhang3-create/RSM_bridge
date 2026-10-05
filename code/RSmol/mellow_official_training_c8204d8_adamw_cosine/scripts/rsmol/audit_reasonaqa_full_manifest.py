#!/usr/bin/env python3
"""Audit a full ReasonAQA manifest and its CPU-only audio path mapping.

This is a read-only preflight.  It does not import torch, decode audio, stage
files, submit a job, or require a GPU.  The mapping/report contract is the
same one consumed by ``stage_reasonaqa_raw_audio.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


DEFAULT_MANIFEST = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/train.json")
DEFAULT_REPORT = Path("/tmp/mellow_reasonaqa_full_manifest_audit.json")
DEFAULT_MAPPING = Path("/tmp/mellow_reasonaqa_train_mapping.jsonl")
PATH_CONTRACT = "reasonaqa_raw_audio_path_mapping_v1"
CONTRACT = "reasonaqa_full_manifest_path_mapping_audit_v1"
REQUIRED_FIELDS = ("filepath1", "filepath2", "input", "answer")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{__import__('os').getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def iter_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read either the official JSON list or one-object-per-line JSONL."""
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        while first and first.isspace():
            first = handle.read(1)
        handle.seek(0)
        if first == "[":
            value = json.load(handle)
            if not isinstance(value, list):
                raise ValueError("JSON manifest must contain a list")
            for number, row in enumerate(value, start=1):
                if not isinstance(row, dict):
                    raise ValueError(f"manifest row {number} is not an object")
                yield number, row
            return
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {number} is not an object")
            yield number, row


def normalize_logical(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def safe_logical(value: str) -> bool:
    if not value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def audit_manifest(path: Path) -> dict[str, Any]:
    subtype_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    empty_counts: Counter[str] = Counter()
    invalid_logical: list[dict[str, Any]] = []
    logical_paths: set[str] = set()
    rows = 0
    for line_number, row in iter_records(path):
        rows += 1
        subtype_counts[str(row.get("subtype", ""))] += 1
        task_counts[str(row.get("taskname", ""))] += 1
        for field in REQUIRED_FIELDS:
            if field not in row:
                missing_counts[field] += 1
            elif row.get(field) in (None, ""):
                empty_counts[field] += 1
        for field in ("filepath1", "filepath2"):
            logical = normalize_logical(row.get(field, ""))
            if logical:
                if not safe_logical(logical) and len(invalid_logical) < 50:
                    invalid_logical.append({"row": line_number, "field": field, "value": logical})
                logical_paths.add(logical)
    if rows == 0:
        raise ValueError(f"manifest is empty: {path}")
    return {
        "rows": rows,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "subtype_distribution": dict(sorted(subtype_counts.items())),
        "taskname_distribution": dict(task_counts.most_common()),
        "required_field_missing_counts": dict(missing_counts),
        "selected_required_field_empty_counts": dict(empty_counts),
        "unique_referenced_logical_paths": len(logical_paths),
        "logical_paths": logical_paths,
        "invalid_logical_preview": invalid_logical,
    }


def audit_mapping(path: Path, logical_paths: set[str]) -> dict[str, Any]:
    seen: set[str] = set()
    sources: dict[str, int] = {}
    missing_source: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    groups: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            item = json.loads(line)
            if not isinstance(item, dict):
                invalid.append({"line": line_number, "error": "not an object"})
                continue
            logical = normalize_logical(item.get("logical_path", ""))
            source = Path(str(item.get("source_path", ""))).expanduser()
            expected_size = int(item.get("source_size_bytes", -1))
            if logical in seen or not safe_logical(logical) or not source.is_file() or expected_size <= 0 or source.stat().st_size != expected_size:
                if len(invalid) < 50:
                    invalid.append({"line": line_number, "logical_path": logical, "source_path": str(source), "expected_size": expected_size})
            seen.add(logical)
            if source.is_file():
                sources[str(source.resolve())] = int(source.stat().st_size)
            else:
                if len(missing_source) < 50:
                    missing_source.append({"line": line_number, "logical_path": logical, "source_path": str(source)})
            groups[str(item.get("group", "unknown"))] += 1
            methods[str(item.get("method", "unknown"))] += 1
    missing_logical = sorted(logical_paths.difference(seen))
    return {
        "rows": rows,
        "unique_logical_paths": len(seen),
        "missing_manifest_paths_count": len(missing_logical),
        "missing_manifest_paths_preview": missing_logical[:50],
        "duplicate_logical_paths": rows - len(seen),
        "unique_source_files": len(sources),
        "total_unique_source_bytes": sum(sources.values()),
        "group_distribution": dict(groups),
        "method_distribution": dict(methods),
        "invalid_preview": invalid,
        "missing_source_preview": missing_source,
    }


def audit(path: Path, report_path: Path, mapping_path: Path, *, world_size: int, per_rank_batch_size: int, gradient_accumulation_steps: int, num_epochs: int, warmup_ratio: float) -> dict[str, Any]:
    manifest = audit_manifest(path)
    mapping = audit_mapping(mapping_path, manifest.pop("logical_paths"))
    path_report = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    warnings: list[str] = []
    if path_report.get("status") != "PASS":
        failures.append("path_audit_status")
    if path_report.get("contract") != PATH_CONTRACT:
        failures.append("path_audit_contract")
    if not path_report.get("mapping_path"):
        failures.append("path_audit_mapping_path_missing")
    elif Path(str(path_report["mapping_path"])).resolve() != mapping_path.resolve():
        failures.append("path_audit_mapping_pair")
    if mapping["missing_manifest_paths_count"]:
        failures.append("manifest_paths_missing_from_mapping")
    if mapping["duplicate_logical_paths"] or mapping["invalid_preview"] or mapping["missing_source_preview"]:
        failures.append("mapping_integrity")
    if manifest["invalid_logical_preview"]:
        failures.append("unsafe_manifest_logical_paths")
    missing_required = manifest["required_field_missing_counts"]
    if missing_required:
        failures.append("missing_required_fields")
    empty_text = {
        field: manifest["selected_required_field_empty_counts"].get(field, 0)
        for field in ("input", "answer")
        if manifest["selected_required_field_empty_counts"].get(field, 0)
    }
    if empty_text:
        warnings.append(
            "empty_text_fields_are_present_but_not_missing; official Dataset appends EOS before tokenization: "
            + json.dumps(empty_text, sort_keys=True)
        )
    if path_report.get("unique_logical_paths_resolved") is not None and int(path_report["unique_logical_paths_resolved"]) != mapping["unique_logical_paths"]:
        failures.append("path_audit_mapping_logical_count")
    if path_report.get("unique_source_files") is not None and int(path_report["unique_source_files"]) != mapping["unique_source_files"]:
        failures.append("path_audit_mapping_source_count")
    if path_report.get("total_source_bytes") is not None and int(path_report["total_source_bytes"]) != mapping["total_unique_source_bytes"]:
        failures.append("path_audit_mapping_source_bytes")
    report_train_json = path_report.get("train_json")
    if report_train_json and Path(str(report_train_json)).resolve() != path.resolve():
        warnings.append("path_audit_train_json_differs_from_manifest; treating mapping as a superset and auditing manifest references")
    effective_batch = world_size * per_rank_batch_size * gradient_accumulation_steps
    steps_per_epoch = manifest["rows"] // effective_batch
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(math.ceil(total_steps * warmup_ratio))
    if steps_per_epoch <= 0:
        failures.append("zero_optimizer_steps")
    if world_size < 1 or per_rank_batch_size < 1 or gradient_accumulation_steps < 1 or num_epochs < 1:
        failures.append("invalid_batch_geometry")
    status = "PASS" if not failures else "FAIL"
    return {
        "status": status,
        "contract": CONTRACT,
        "gpu_required": False,
        "manifest": {"path": str(path.resolve()), **manifest},
        "path_audit_report": {"path": str(report_path.resolve()), "status": path_report.get("status"), "contract": path_report.get("contract"), "rows": path_report.get("rows"), "unique_logical_paths_resolved": path_report.get("unique_logical_paths_resolved"), "unique_source_files": path_report.get("unique_source_files"), "total_source_bytes": path_report.get("total_source_bytes"), "train_json": report_train_json, "missing_roots": path_report.get("missing_roots", {})},
        "mapping": {"path": str(mapping_path.resolve()), **mapping},
        "batch_geometry": {"world_size": world_size, "per_rank_batch_size": per_rank_batch_size, "gradient_accumulation_steps": gradient_accumulation_steps, "effective_global_batch": effective_batch, "num_epochs": num_epochs, "steps_per_epoch_floor": steps_per_epoch, "total_steps": total_steps, "warmup_ratio": warmup_ratio, "warmup_steps_ceil": warmup_steps},
        "failures": failures,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--path-audit-report", type=Path, required=True)
    parser.add_argument("--mapping-jsonl", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--per-rank-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    args = parser.parse_args()
    manifest = args.manifest.expanduser().resolve(strict=True)
    path_report = args.path_audit_report.expanduser().resolve(strict=True)
    mapping = args.mapping_jsonl.expanduser().resolve(strict=True)
    if not manifest.is_file() or not path_report.is_file() or not mapping.is_file():
        raise SystemExit("manifest, path-audit report, and mapping must be regular files")
    report = audit(manifest, path_report, mapping, world_size=args.world_size, per_rank_batch_size=args.per_rank_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, num_epochs=args.num_epochs, warmup_ratio=args.warmup_ratio)
    write_json(args.output_report.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
