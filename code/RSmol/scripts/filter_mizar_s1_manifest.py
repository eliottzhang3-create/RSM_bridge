#!/usr/bin/env python3
"""Filter excluded or unauditable rows from a Mizar Stage-1 manifest.

The source manifest is streamed and left untouched. The output remains a JSON
array of the original row objects, with only selected rows omitted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from audit_mizar_s1_raw_audio import iter_rows, raw_reference, source_key, source_label


STATUS_ISSUE_FILES = {
    "allowed_missing": "allowed_missing.jsonl",
    "retained_missing": "retained_missing.jsonl",
    "ambiguous": "ambiguous.jsonl",
    "unmapped": "unmapped.jsonl",
}
DECODE_ISSUE_FILE = "decode_failures.jsonl"
TRAIN_OTHER_500_SEGMENT = "train-other-500"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream-filter the Mizar S1 manifest using raw-audio audit row indices, "
            "and remove MusicCaps and LibriTTS-R train-other-500 rows."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--excluded-jsonl", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--keep-musiccaps",
        action="store_true",
        help="retain resolvable MusicCaps rows instead of applying the historical exclusion",
    )
    parser.add_argument(
        "--keep-train-other-500",
        action="store_true",
        help="retain resolvable SpeechCraft train-other-500 rows",
    )
    parser.add_argument(
        "--max-unique-audio-gib",
        type=float,
        default=0.0,
        help="refuse to publish if the audit's unique resolved audio exceeds this GiB limit",
    )
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args(argv)


def default_sidecar_paths(output_manifest: Path) -> tuple[Path, Path]:
    stem = output_manifest.with_suffix("")
    return (
        stem.with_name(stem.name + ".excluded_rows.jsonl"),
        stem.with_name(stem.name + ".filter_report.json"),
    )


def canonical(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def validate_paths(
    manifest: Path,
    audit_dir: Path,
    output_manifest: Path,
    excluded_jsonl: Path,
    report: Path,
) -> None:
    inputs = {canonical(manifest)}
    inputs.update(canonical(audit_dir / name) for name in (
        "summary.json",
        *STATUS_ISSUE_FILES.values(),
        DECODE_ISSUE_FILE,
    ))
    outputs = [canonical(output_manifest), canonical(excluded_jsonl), canonical(report)]
    if len(set(outputs)) != len(outputs):
        raise ValueError("output manifest, excluded JSONL, and report must be different paths")
    for path in outputs:
        if path in inputs:
            raise ValueError(f"output path collides with an input: {path}")
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")


def load_summary(audit_dir: Path, manifest: Path) -> dict[str, Any]:
    summary_path = audit_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict) or not isinstance(summary.get("counts"), dict):
        raise ValueError(f"invalid audit summary structure: {summary_path}")
    audited_manifest = summary.get("manifest")
    if not audited_manifest:
        raise ValueError(f"audit summary does not identify its input manifest: {summary_path}")
    if canonical(Path(str(audited_manifest))) != canonical(manifest):
        raise ValueError(
            "audit manifest does not match --manifest: "
            f"summary has {audited_manifest!r}, argument is {str(manifest)!r}"
        )
    for key in ("rows", "existing_rows"):
        value = summary["counts"].get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"audit summary has invalid counts.{key}: {value!r}")
    if summary["counts"]["existing_rows"] > summary["counts"]["rows"]:
        raise ValueError("audit summary existing_rows exceeds rows")
    source_rows = summary.get("source_rows")
    if not isinstance(source_rows, dict):
        raise ValueError(f"audit summary does not contain source_rows: {summary_path}")
    for source, count in source_rows.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"audit summary has invalid source_rows[{source!r}]: {count!r}")
    return summary


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def read_issue_indices(
    audit_dir: Path,
    expected_rows: int,
) -> tuple[dict[int, list[str]], dict[str, int], dict[int, dict[str, Any]]]:
    reasons_by_index: dict[int, list[str]] = defaultdict(list)
    records_by_index: dict[int, dict[str, Any]] = {}
    file_counts: dict[str, int] = {}
    status_indices: set[int] = set()

    issue_groups = [
        (f"audit_{name}", audit_dir / filename, True)
        for name, filename in STATUS_ISSUE_FILES.items()
    ]
    issue_groups.append(("audio_decode_failure", audit_dir / DECODE_ISSUE_FILE, False))

    for reason, path, is_status_issue in issue_groups:
        if not path.is_file():
            raise FileNotFoundError(f"required audit result is missing: {path}")
        count = 0
        seen_in_file: set[int] = set()
        for record in iter_jsonl(path):
            count += 1
            index = record.get("row_index")
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError(f"{path}: issue row has invalid row_index {index!r}")
            if not 0 <= index < expected_rows:
                raise ValueError(
                    f"{path}: row_index {index} is outside manifest range 0..{expected_rows - 1}"
                )
            if index in seen_in_file:
                raise ValueError(f"{path}: duplicate row_index {index}")
            seen_in_file.add(index)
            if is_status_issue:
                if index in status_indices:
                    raise ValueError(
                        f"audit status issue row_index {index} appears in multiple status files"
                    )
                status_indices.add(index)
            reasons_by_index[index].append(reason)
            records_by_index.setdefault(index, record)
        file_counts[path.name] = count

    return dict(reasons_by_index), file_counts, records_by_index


def path_has_train_other_500(value: str) -> bool:
    normalized = value.replace("\\", "/").strip("/").lower()
    return TRAIN_OTHER_500_SEGMENT in normalized.split("/")


def row_filter_reasons(
    row: dict[str, Any],
    audit_reasons: list[str],
    keep_musiccaps: bool = False,
    keep_train_other_500: bool = False,
) -> list[str]:
    reasons = list(audit_reasons)
    source = source_label(row)
    if source_key(source) == "musiccaps" and not keep_musiccaps:
        reasons.append("dataset_musiccaps")

    field, raw_path = raw_reference(row, source)
    if (
        source_key(source) == "speechcraft"
        and field
        and path_has_train_other_500(raw_path)
        and not keep_train_other_500
    ):
        reasons.append("speechcraft_train_other_500")
    return list(dict.fromkeys(reasons))


def validate_issue_record(
    index: int,
    row: dict[str, Any],
    issue: dict[str, Any],
) -> None:
    source = source_label(row)
    field, raw_path = raw_reference(row, source)
    issue_source = str(issue.get("source", ""))
    if source_key(issue_source) != source_key(source):
        raise ValueError(
            f"audit issue row_index {index} source mismatch: "
            f"manifest={source!r}, audit={issue_source!r}"
        )
    if str(issue.get("field", "")) != field or str(issue.get("raw_path", "")) != raw_path:
        raise ValueError(
            f"audit issue row_index {index} raw reference mismatch: "
            f"manifest=({field!r}, {raw_path!r}), "
            f"audit=({issue.get('field')!r}, {issue.get('raw_path')!r})"
        )


def create_temp_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=destination.name + ".partial.",
        dir=str(destination.parent),
    )
    os.close(descriptor)
    return Path(name)


def publish_without_overwrite(temporary: Path, destination: Path) -> None:
    """Atomically publish a completed file without replacing an existing path."""
    os.link(temporary, destination)
    temporary.unlink()


def write_report(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def filter_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.expanduser()
    audit_dir = args.audit_dir.expanduser()
    output_manifest = args.output_manifest.expanduser()
    default_excluded, default_report = default_sidecar_paths(output_manifest)
    excluded_jsonl = (args.excluded_jsonl or default_excluded).expanduser()
    report_path = (args.report or default_report).expanduser()
    validate_paths(manifest, audit_dir, output_manifest, excluded_jsonl, report_path)

    summary = load_summary(audit_dir, manifest)
    keep_musiccaps = bool(getattr(args, "keep_musiccaps", False))
    keep_train_other_500 = bool(getattr(args, "keep_train_other_500", False))
    max_unique_audio_gib = float(getattr(args, "max_unique_audio_gib", 0.0))
    if max_unique_audio_gib < 0:
        raise ValueError("--max-unique-audio-gib must be non-negative")
    if max_unique_audio_gib:
        unique_audio_bytes = summary.get("unique_resolved_audio_bytes")
        stat_failures = summary.get("size_stat_failures")
        if not isinstance(unique_audio_bytes, int) or isinstance(unique_audio_bytes, bool):
            raise ValueError(
                "audit summary has no unique_resolved_audio_bytes; rerun the updated "
                "CPU audio audit before filtering"
            )
        if stat_failures != 0:
            raise ValueError(
                f"audio audit has {stat_failures!r} file-size stat failures; "
                "cannot enforce the staging capacity"
            )
        limit_bytes = int(max_unique_audio_gib * (1024 ** 3))
        if unique_audio_bytes > limit_bytes:
            raise ValueError(
                "unique resolved audio exceeds the requested staging limit: "
                f"{unique_audio_bytes / (1024 ** 3):.3f} GiB > "
                f"{max_unique_audio_gib:.3f} GiB"
            )
    expected_rows = summary["counts"]["rows"]
    expected_existing = summary["counts"]["existing_rows"]
    expected_status_issues = expected_rows - expected_existing
    audit_reasons, issue_file_counts, issue_records = read_issue_indices(
        audit_dir,
        expected_rows,
    )
    status_issue_rows = sum(issue_file_counts[name] for name in STATUS_ISSUE_FILES.values())
    unique_status_rows = len({
        index for index, reasons in audit_reasons.items()
        if any(reason.startswith("audit_") for reason in reasons)
    })
    if status_issue_rows != expected_status_issues or unique_status_rows != expected_status_issues:
        raise ValueError(
            "audit issue files do not partition the manifest's non-existing rows: "
            f"file_rows={status_issue_rows}, unique_status_rows={unique_status_rows}, "
            f"summary_rows_minus_existing={expected_status_issues}"
        )
    expected_decode_failures = summary["counts"].get("decode_failures", 0)
    actual_decode_failures = issue_file_counts[DECODE_ISSUE_FILE]
    if actual_decode_failures != expected_decode_failures:
        raise ValueError(
            "decode_failures.jsonl count does not match audit summary: "
            f"file={actual_decode_failures}, summary={expected_decode_failures}"
        )

    temporary_paths: list[Path] = []
    published_paths: list[Path] = []

    rows_seen = 0
    rows_kept = 0
    rows_excluded = 0
    source_before: Counter[str] = Counter()
    source_after: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    try:
        temp_manifest = create_temp_path(output_manifest)
        temporary_paths.append(temp_manifest)
        temp_excluded = create_temp_path(excluded_jsonl)
        temporary_paths.append(temp_excluded)
        temp_report = create_temp_path(report_path)
        temporary_paths.append(temp_report)

        with (
            temp_manifest.open("w", encoding="utf-8", newline="\n") as output_handle,
            temp_excluded.open("w", encoding="utf-8", newline="\n") as excluded_handle,
        ):
            output_handle.write("[")
            first_kept = True
            for row_index, row in enumerate(iter_rows(manifest)):
                rows_seen += 1
                source = source_label(row)
                source_before[source] += 1
                issue = issue_records.get(row_index)
                if issue is not None:
                    validate_issue_record(row_index, row, issue)
                reasons = row_filter_reasons(
                    row,
                    audit_reasons.get(row_index, []),
                    keep_musiccaps=keep_musiccaps,
                    keep_train_other_500=keep_train_other_500,
                )
                if reasons:
                    rows_excluded += 1
                    reason_counts.update(reasons)
                    field, raw_path = raw_reference(row, source)
                    sample_key = row.get("sample_key") or row.get("id")
                    excluded_handle.write(json.dumps({
                        "row_index": row_index,
                        "source": source,
                        "sample_key": sample_key,
                        "raw_field": field,
                        "raw_path": raw_path,
                        "exclude_reasons": reasons,
                    }, ensure_ascii=False, separators=(",", ":")))
                    excluded_handle.write("\n")
                else:
                    if not first_kept:
                        output_handle.write(",")
                    output_handle.write(json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ))
                    first_kept = False
                    rows_kept += 1
                    source_after[source] += 1
                if args.progress_every > 0 and rows_seen % args.progress_every == 0:
                    print(f"filtered {rows_seen} rows; kept {rows_kept}", flush=True)
            output_handle.write("]\n")

        if rows_seen != expected_rows:
            raise ValueError(
                f"manifest row count changed since audit: read={rows_seen}, audit={expected_rows}"
            )
        expected_source_rows = summary["source_rows"]
        observed_source_rows = dict(source_before)
        if observed_source_rows != expected_source_rows:
            raise ValueError(
                "manifest source counts changed since audit: "
                f"read={observed_source_rows}, audit={expected_source_rows}"
            )

        report = {
            "format": "json_array_of_original_manifest_rows",
            "manifest": str(manifest),
            "audit_dir": str(audit_dir),
            "audit_summary": str(audit_dir / "summary.json"),
            "output_manifest": str(output_manifest),
            "excluded_jsonl": str(excluded_jsonl),
            "report": str(report_path),
            "counts": {
                "input_rows": rows_seen,
                "kept_rows": rows_kept,
                "excluded_rows": rows_excluded,
                "audit_status_issue_rows": expected_status_issues,
                "decode_failure_rows": actual_decode_failures,
            },
            "excluded_by_reason": dict(sorted(reason_counts.items())),
            "source_rows_before": dict(sorted(source_before.items())),
            "source_rows_after": dict(sorted(source_after.items())),
            "audit_issue_file_rows": issue_file_counts,
            "filters": [
                *([] if keep_musiccaps else ["all rows whose source is MusicCaps"]),
                *([] if keep_train_other_500 else [
                    "all SpeechCraft rows whose raw audio path contains the train-other-500 path segment"
                ]),
                "all rows listed in allowed_missing.jsonl, retained_missing.jsonl, ambiguous.jsonl, or unmapped.jsonl",
                "all rows listed in decode_failures.jsonl",
            ],
            "retention_policy": {
                "keep_musiccaps": keep_musiccaps,
                "keep_speechcraft_train_other_500": keep_train_other_500,
                "max_unique_audio_gib": max_unique_audio_gib or None,
                "audited_unique_resolved_audio_gib": summary.get(
                    "unique_resolved_audio_gib"
                ),
            },
        }
        write_report(temp_report, report)

        for temporary, destination in (
            (temp_excluded, excluded_jsonl),
            (temp_report, report_path),
            (temp_manifest, output_manifest),
        ):
            publish_without_overwrite(temporary, destination)
            published_paths.append(destination)
        return report
    except Exception:
        for path in published_paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.progress_every < 0:
            raise ValueError("--progress-every must be >= 0")
        report = filter_manifest(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
