#!/usr/bin/env python3
"""Filter original ReasonAQA metadata to native single-audio records.

This utility is intentionally CPU-only and depends only on the Python standard
library.  It does not open, copy, decode, or rewrite audio files.  A record is
eligible only when its original ``filepath2`` value is exactly ``""`` and
``filepath1`` is a non-empty string.  In particular, a dual-audio record is
never converted into a single-audio record by clearing its second path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

CONTRACT = "reasonaqa_native_single_audio_filter_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument(
        "--semantic-policy",
        choices=("exclude", "report"),
        default="exclude",
        help=(
            "How to handle an input field that explicitly asks about a second "
            "audio or a comparison. The default excludes it and records the "
            "reason; report keeps it while marking it in the report."
        ),
    )
    parser.add_argument("--preview-limit", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> tuple[Any, bytes]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"input path is not a regular file: {resolved}")
    payload = resolved.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse UTF-8 JSON: {resolved}: {exc}") from exc
    return value, payload


def normalized_text(value: Any) -> str:
    return str(value).strip().casefold() if isinstance(value, str) else ""


def semantic_flags(row: dict[str, Any]) -> list[str]:
    """Return transparent warnings for prompts that require audio 2."""

    value = normalized_text(row.get("input", ""))
    if not value:
        return ["missing_or_empty_input"]
    flags: list[str] = []
    if value in {"caption second audio", "caption both audios"}:
        flags.append("caption_requires_second_audio")
    if re.search(r"\baudio\s*2\b|\baudio2\b", value):
        flags.append("input_mentions_audio2")
    if re.search(r"\b(both|two)\s+audios?\b", value):
        flags.append("input_mentions_both_audios")
    if re.search(r"\b(compare|comparison|difference|different)\b", value):
        flags.append("input_requests_comparison")
    return list(dict.fromkeys(flags))


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in counter.most_common()}


def filter_records(
    records: Any,
    *,
    semantic_policy: str = "exclude",
    preview_limit: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise ValueError("ReasonAQA metadata must be a non-empty JSON list")
    if semantic_policy not in {"exclude", "report"}:
        raise ValueError(f"unsupported semantic policy: {semantic_policy!r}")
    if preview_limit < 0:
        raise ValueError("--preview-limit must be non-negative")

    retained: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    source_input_counts: Counter[str] = Counter()
    retained_input_counts: Counter[str] = Counter()
    taskname_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    invalid_rows: list[dict[str, Any]] = []
    excluded_preview: list[dict[str, Any]] = []

    for row_number, row in enumerate(records):
        if not isinstance(row, dict):
            exclusion_counts["row_not_object"] += 1
            invalid_rows.append({"row": row_number, "reason": "row_not_object"})
            continue

        input_value = row.get("input", "")
        source_input_counts[str(input_value)] += 1
        taskname_counts[str(row.get("taskname", ""))] += 1
        reasons: list[str] = []
        if "filepath1" not in row or "filepath2" not in row:
            reasons.append("missing_filepath_field")
        elif not isinstance(row["filepath1"], str) or not isinstance(row["filepath2"], str):
            reasons.append("filepath_field_not_string")
        else:
            # This is deliberately an exact comparison. The official loader
            # uses ``filepath2 == ''``; whitespace is not silently normalized.
            if row["filepath2"] != "":
                reasons.append(
                    "filepath2_nonempty" if row["filepath2"].strip() else "filepath2_whitespace_only"
                )
            if row["filepath1"] == "":
                reasons.append("filepath1_empty")

        flags = semantic_flags(row)
        for flag in flags:
            semantic_counts[flag] += 1
        if flags and semantic_policy == "exclude":
            reasons.extend(f"semantic:{flag}" for flag in flags)

        if reasons:
            for reason in reasons:
                exclusion_counts[reason] += 1
            if len(excluded_preview) < preview_limit:
                excluded_preview.append(
                    {
                        "row": row_number,
                        "reasons": list(dict.fromkeys(reasons)),
                        "filepath1": row.get("filepath1"),
                        "filepath2": row.get("filepath2"),
                        "input": row.get("input"),
                    }
                )
            if any(reason.startswith("missing_") or reason.endswith("not_string") or reason == "row_not_object" for reason in reasons):
                invalid_rows.append(
                    {"row": row_number, "reason": list(dict.fromkeys(reasons))}
                )
            continue

        # No mutation or normalization: retain the original dictionary values
        # and source ordering. A shallow copy prevents later bookkeeping from
        # changing the parsed source object.
        retained_row = dict(row)
        retained.append(retained_row)
        retained_input_counts[str(input_value)] += 1

    report = {
        "contract": CONTRACT,
        "selection": {
            "filepath2": "original value exactly equals empty string",
            "filepath1": "non-empty string",
            "semantic_policy": semantic_policy,
            "preserve_record_fields": True,
            "preserve_source_order": True,
        },
        "rows_source": len(records),
        "rows_retained": len(retained),
        "rows_excluded": len(records) - len(retained),
        "exclusion_counts": _counter_dict(exclusion_counts),
        "source_input_counts": _counter_dict(source_input_counts),
        "retained_input_counts": _counter_dict(retained_input_counts),
        "taskname_counts_source": _counter_dict(taskname_counts),
        "semantic_flag_counts": _counter_dict(semantic_counts),
        "invalid_row_count": len(invalid_rows),
        "invalid_row_preview": invalid_rows[:preview_limit],
        "excluded_preview": excluded_preview,
        "native_single_audio_invariant": all(
            isinstance(row.get("filepath1"), str)
            and row["filepath1"] != ""
            and row.get("filepath2") == ""
            for row in retained
        ),
    }
    if not retained:
        raise ValueError("filter retained zero native single-audio rows")
    if report["invalid_row_count"]:
        report["status"] = "FAIL_INVALID_SOURCE_ROWS"
    else:
        report["status"] = "PASS"
    return retained, report


def write_bytes_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any, *, overwrite: bool) -> bytes:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")
    write_bytes_atomic(path, payload, overwrite=overwrite)
    return payload


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args() if argv is None else parse_args_from(argv)
    input_path = args.input_json.expanduser().resolve(strict=True)
    output_path = args.output_json.expanduser().resolve(strict=False)
    report_path = args.report_path.expanduser().resolve(strict=False)
    if output_path == input_path:
        raise ValueError("output JSON must differ from input JSON")
    if output_path == report_path:
        raise ValueError("output JSON and report must be different files")

    records, source_bytes = read_json(input_path)
    filtered, report = filter_records(
        records, semantic_policy=args.semantic_policy, preview_limit=args.preview_limit
    )
    filtered_bytes = (
        json.dumps(filtered, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")
    report.update(
        {
            "source_json": str(input_path),
            "source_sha256": sha256_bytes(source_bytes),
            "output_json": str(output_path),
            "output_sha256": sha256_bytes(filtered_bytes),
            "output_bytes": len(filtered_bytes),
        }
    )
    # Write the manifest and report only after all validation has completed.
    write_bytes_atomic(output_path, filtered_bytes, overwrite=args.overwrite)
    write_json_atomic(report_path, report, overwrite=args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


def parse_args_from(argv: Iterable[str]) -> argparse.Namespace:
    # Keep the normal parser as the single source of CLI semantics while making
    # main() straightforward to exercise from dependency-free unit tests.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--semantic-policy", choices=("exclude", "report"), default="exclude")
    parser.add_argument("--preview-limit", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(list(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"[single-audio-filter] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
