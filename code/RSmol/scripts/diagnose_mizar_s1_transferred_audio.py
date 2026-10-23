#!/usr/bin/env python3
"""Read-only, standard-library diagnosis of transferred AudioMCQ audio.

Compare failed references in an existing S1 audit with the advisor's S2
mapping. Inventory the three destination directories and search for the exact
mapped filename. A stem match is a diagnostic hint only, never a resolution.
This script writes only to stdout; it does not decode, copy, or edit any data.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


RULE_SOURCES = {
    "audiomcq_audiocaps": "AudioCaps",
    "musiccaps": "MusicCaps",
    "speechcraft_libritts_r": "SpeechCraft",
}
AUDIO_SUFFIXES = {
    ".wav", ".wave", ".flac", ".mp3", ".m4a", ".ogg", ".opus",
    ".aac", ".aif", ".aiff", ".wma", ".webm", ".mp4",
}


def normalized_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip()


def read_failures(audit_dir: Path) -> tuple[dict, dict]:
    failures: dict[tuple[str, str], dict] = {}
    counts: dict[str, Counter] = defaultdict(Counter)
    with (audit_dir / "resolved_paths.tsv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"field", "raw_path", "references", "status", "rule"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("resolved_paths.tsv has incompatible columns")
        for row in reader:
            source = RULE_SOURCES.get(row["rule"])
            if source is None or row["field"] != "audio_path":
                continue
            references = int(row["references"])
            counts[source][f"{row['status']}_references"] += 1
            counts[source][f"{row['status']}_rows"] += references
            if row["status"] == "exists":
                continue
            key = (source, normalized_path(row["raw_path"]))
            if key in failures:
                raise ValueError(f"duplicate failed raw reference: {key}")
            failures[key] = {
                "source": source,
                "raw_path": row["raw_path"],
                "audit_status": row["status"],
                "manifest_rows": references,
            }
    return failures, {source: dict(count) for source, count in counts.items()}


def read_s2_mapping(path: Path, failures: dict) -> tuple[dict, dict]:
    """Exact source + audio_path keys; preserve conflicting mapping records."""
    records: dict[tuple[str, str], dict[tuple, dict]] = defaultdict(dict)
    source_names = {name.casefold(): name for name in RULE_SOURCES.values()}
    counts: dict[str, Counter] = defaultdict(Counter)
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            label = str(record.get("source_dataset") or "").strip()
            source = source_names.get(label.casefold(), label or "UNCLASSIFIED")
            status = str(record.get("status") or "UNSPECIFIED")
            counts[source][status] += 1
            key = (source, normalized_path(record.get("audio_path")))
            if key not in failures:
                continue
            compact = {
                "status": status,
                "local_path": normalized_path(record.get("local_path")),
                "archive_path": normalized_path(record.get("archive_path")),
                "archive_member": normalized_path(record.get("archive_member")),
            }
            signature = tuple(compact.values())
            records[key][signature] = compact
    return (
        {key: list(value.values()) for key, value in records.items()},
        {source: dict(count) for source, count in sorted(counts.items())},
    )


def inventory(root: Path) -> tuple[dict, dict, dict]:
    by_name: dict[str, list[str]] = defaultdict(list)
    by_stem: dict[str, list[str]] = defaultdict(list)
    suffix_counts: Counter[str] = Counter()
    errors: list[str] = []
    skipped_symlink_dirs: list[str] = []
    filenames: list[str] = []
    file_count = 0
    audio_count = 0
    exists = root.is_dir()
    if exists:
        for directory, dirs, files in os.walk(
            root, followlinks=False, onerror=lambda exc: errors.append(str(exc))
        ):
            for name in dirs:
                path = Path(directory) / name
                if path.is_symlink():
                    skipped_symlink_dirs.append(str(path))
            for name in files:
                path = Path(directory) / name
                file_count += 1
                suffix = path.suffix.casefold()
                suffix_counts[suffix or "<none>"] += 1
                if suffix not in AUDIO_SUFFIXES:
                    continue
                audio_count += 1
                if len(filenames) < 8:
                    filenames.append(str(path.relative_to(root)))
                by_name[name].append(str(path))
                by_stem[path.stem].append(str(path))
    return dict(by_name), dict(by_stem), {
        "root": str(root),
        "root_exists": exists,
        "file_entries": file_count,
        "audio_file_entries": audio_count,
        "extensions": dict(suffix_counts.most_common()),
        "example_audio_relative_paths": filenames,
        "walk_errors": errors[:10],
        "walk_error_count": len(errors),
        "skipped_symlink_directories_first_10": skipped_symlink_dirs[:10],
        "skipped_symlink_directory_count": len(skipped_symlink_dirs),
    }


def classify(record: dict, by_name: dict, by_stem: dict) -> tuple[str, dict]:
    mapped_path = record.get("local_path") or record.get("archive_member") or ""
    if not mapped_path:
        if str(record.get("status", "")).casefold() in {"missing", "not_found"}:
            return "advisor_also_missing", {}
        return "advisor_record_has_no_audio_filename", {}
    name = PurePosixPath(mapped_path).name
    # Keep the complete YouTube/start-time or LibriTTS utterance filename.
    # No substring, fuzzy, or stripped-ID matching is performed.
    matches = list(dict.fromkeys(by_name.get(name, [])))
    if len(matches) == 1:
        candidate = Path(matches[0])
        try:
            if not candidate.is_file():
                return "mapped_filename_is_not_a_readable_file", {"candidate": str(candidate)}
            size = candidate.stat().st_size
        except OSError as exc:
            return "mapped_filename_stat_error", {"candidate": str(candidate), "error": str(exc)}
        return "advisor_filename_found_unique", {
            "candidate": str(candidate), "size_bytes": size,
        }
    if len(matches) > 1:
        return "advisor_filename_has_multiple_candidates", {"candidates": matches[:5]}
    stem = PurePosixPath(name).stem
    hints = list(dict.fromkeys(by_stem.get(stem, [])))
    if hints:
        return "advisor_filename_absent_same_stem_only", {"hints_not_accepted": hints[:5]}
    return "advisor_filename_absent", {}


def diagnose(audit_dir: Path, mapping_path: Path, data_root: Path) -> dict:
    failures, audit_counts = read_failures(audit_dir)
    records, mapping_counts = read_s2_mapping(mapping_path, failures)
    result = {
        "read_only": True,
        "scope": "diagnosis of failed references only; no corrected mapping is published",
        "audit_dir": str(audit_dir),
        "s2_mapping": str(mapping_path),
        "current_audit": audit_counts,
        "s2_mapping_records_by_source_status": mapping_counts,
        "datasets": {},
    }
    for source in RULE_SOURCES.values():
        print(f"[diagnosis] inventory {data_root / source}", flush=True)
        by_name, by_stem, directory_report = inventory(data_root / source)
        reference_counts: Counter[str] = Counter()
        row_counts: Counter[str] = Counter()
        examples: dict[str, list[dict]] = defaultdict(list)
        for key, failure in failures.items():
            if key[0] != source:
                continue
            mapped = records.get(key, [])
            detail: dict = {}
            if not mapped:
                kind = "no_exact_source_and_raw_path_in_s2_mapping"
            elif len(mapped) > 1:
                kind = "conflicting_s2_mapping_records"
                detail["s2_records"] = mapped[:5]
            else:
                kind, detail = classify(mapped[0], by_name, by_stem)
                detail["s2_record"] = mapped[0]
            reference_counts[kind] += 1
            row_counts[kind] += failure["manifest_rows"]
            if len(examples[kind]) < 3:
                examples[kind].append({**failure, **detail})
        result["datasets"][source] = {
            "directory_inventory": directory_report,
            "failed_unique_references_by_diagnosis": dict(reference_counts),
            "failed_manifest_rows_by_diagnosis": dict(row_counts),
            "examples": dict(examples),
        }
    return result


def print_summary(report: dict[str, Any], max_examples: int = 1) -> None:
    """Print the small subset needed to decide whether mapping needs repair."""
    print("current audit:")
    for source, counts in report["current_audit"].items():
        print(f"  {source}: {counts}")
    print("S2 mapping status counts:")
    for source, counts in report["s2_mapping_records_by_source_status"].items():
        print(f"  {source}: {counts}")
    for source, details in report["datasets"].items():
        inventory = details["directory_inventory"]
        print(f"\n{source}")
        print(
            "  directory: "
            f"exists={inventory['root_exists']} "
            f"audio_files={inventory['audio_file_entries']} "
            f"all_files={inventory['file_entries']}"
        )
        print("  extensions:", inventory["extensions"])
        print("  audio path examples:", inventory["example_audio_relative_paths"][:3])
        print("  directory walk errors:", inventory["walk_error_count"])
        if inventory["walk_errors"]:
            print("    ", inventory["walk_errors"][:3])
        print("  skipped symlink directories:", inventory["skipped_symlink_directory_count"])
        if inventory["skipped_symlink_directory_count"]:
            print("    ", inventory["skipped_symlink_directories_first_10"][:3])
        print("  failed unique references:", details["failed_unique_references_by_diagnosis"])
        print("  failed manifest rows:", details["failed_manifest_rows_by_diagnosis"])
        for kind, examples in details["examples"].items():
            for example in examples[:max_examples]:
                print(
                    f"  example [{kind}] rows={example['manifest_rows']} "
                    f"raw={example['raw_path']}"
                )
                if "s2_record" in example:
                    record = example["s2_record"]
                    print(f"    s2_status={record.get('status', '')}")
                    print(f"    s2_local_path={record.get('local_path', '')}")
                    print(f"    s2_archive_path={record.get('archive_path', '')}")
                    print(f"    s2_archive_member={record.get('archive_member', '')}")
                for key in ("candidate", "hints_not_accepted", "candidates", "s2_records"):
                    if key in example:
                        print(f"    {key}={example[key]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--s2-mapping", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("/hpc_stor03/sjtu_home/jinwei.zhang/data"),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print compact counts and one example per diagnosis instead of full JSON",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=1,
        help="examples per diagnosis in --summary-only output",
    )
    args = parser.parse_args(argv)
    if args.max_examples < 0:
        parser.error("--max-examples must be non-negative")
    report = diagnose(args.audit_dir, args.s2_mapping, args.data_root)
    if args.summary_only:
        print_summary(report, args.max_examples)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
