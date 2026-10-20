#!/usr/bin/env python3
"""Audit the audio references used by a Mizar Stage 1 manifest.

The manifest keeps the original path strings, so this tool deliberately does
not rewrite paths or try to resolve them. It only reports the exact references
that must be made available on the training server.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SOURCE_TOKENS = (
    ("audiocaps", "AudioCaps"),
    ("clotho", "Clotho"),
    ("compa-r", "CompA-R"),
    ("compar", "CompA-R"),
    ("compa_r", "CompA-R"),
    ("musiccaps", "MusicCaps"),
    ("lp-musiccaps", "LP-MusicCaps-MTT"),
    ("magnat", "LP-MusicCaps-MTT"),
    ("mtt", "LP-MusicCaps-MTT"),
    ("speechcraft", "SpeechCraft"),
    ("libritts", "SpeechCraft"),
    ("tacos", "TACOS"),
)


def open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_rows(path: Path) -> Iterable[dict[str, Any]]:
    """Load either Mizar's JSON-array manifest or a JSONL manifest."""

    with open_text(path) as handle:
        try:
            value = json.load(handle)
        except json.JSONDecodeError:
            value = None

    if isinstance(value, list):
        for row in value:
            if not isinstance(row, dict):
                raise ValueError("manifest contains a non-object row")
            yield row
        return

    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            yield row


def clean_path(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def source_from_row_or_path(row: dict[str, Any], path: str) -> tuple[str, str]:
    for key in ("source_dataset", "source", "dataset", "source_name"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value), "row"

    lowered = path.replace("\\", "/").lower()
    for token, label in SOURCE_TOKENS:
        if token in lowered:
            return label, "path"
    return "UNCLASSIFIED", "path"


def path_prefix(path: str) -> str:
    parts = [part for part in re.split(r"[\\/]+", path) if part]
    if not parts:
        return ""
    # Four components are enough to expose the original dataset root without
    # making the report depend on a particular machine's absolute prefix.
    return "/".join(parts[:4])


def write_tsv(path: Path, header: list[str], rows: Iterable[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            values = [str(value).replace("\t", " ").replace("\n", " ") for value in row]
            handle.write("\t".join(values) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="path to s1.json or s1.json.gz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for the audit reports",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_rows: Counter[str] = Counter()
    source_path_refs: Counter[str] = Counter()
    source_field_rows: Counter[str] = Counter()
    row_fields: dict[str, Counter[str]] = {
        key: Counter() for key in ("source", "stageE_variant", "source_dataset")
    }
    path_refs: Counter[tuple[str, str]] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    prefix_refs: Counter[tuple[str, str]] = Counter()
    raw_path_refs: Counter[tuple[str, str, str]] = Counter()
    processed_root_rows: Counter[str] = Counter()
    row_count = 0
    missing_path1 = 0
    empty_path2 = 0

    for row in load_rows(args.manifest):
        row_count += 1
        path1 = clean_path(row.get("filepath1"))
        path2 = clean_path(row.get("filepath2"))
        if not path1:
            missing_path1 += 1
        if not path2:
            empty_path2 += 1

        pair = (path1, path2)
        pairs[pair] += 1
        for key in row_fields:
            value = clean_path(row.get(key))
            if value:
                row_fields[key][value] += 1

        row_source, row_source_kind = source_from_row_or_path(row, path1)
        source_rows[row_source] += 1
        if row_source_kind == "row":
            source_field_rows[row_source] += 1

        for raw_key in ("audio_path", "filepath1_original"):
            raw_path = clean_path(row.get(raw_key))
            if raw_path:
                raw_path_refs[(row_source, raw_key, raw_path)] += 1

        processed_marker = "/processed_clips/"
        if processed_marker in path1.replace("\\", "/"):
            tail = path1.replace("\\", "/").split(processed_marker, 1)[1]
            processed_root_rows[tail.split("/", 1)[0]] += 1

        for slot, path in (("filepath1", path1), ("filepath2", path2)):
            if not path:
                continue
            source, source_kind = source_from_row_or_path(row, path)
            path_refs[(source, path)] += 1
            source_path_refs[source] += 1
            prefix_refs[(source, path_prefix(path))] += 1

    source_summary = []
    for source in sorted(source_path_refs):
        unique_paths = sum(1 for item_source, _ in path_refs if item_source == source)
        source_summary.append(
            {
                "source": source,
                "rows_with_filepath1": source_rows.get(source, 0),
                "nonempty_path_references": source_path_refs[source],
                "unique_audio_paths": unique_paths,
                "rows_with_source_field": source_field_rows.get(source, 0),
            }
        )

    summary = {
        "manifest": str(args.manifest),
        "rows": row_count,
        "missing_filepath1_rows": missing_path1,
        "empty_filepath2_rows": empty_path2,
        "nonempty_filepath2_rows": row_count - empty_path2,
        "same_filepath_pair_rows": sum(
            count for (path1, path2), count in pairs.items() if path1 and path1 == path2
        ),
        "distinct_filepath_pair_rows": sum(
            count for (path1, path2), count in pairs.items() if path1 and path1 != path2
        ),
        "unique_filepath1": len({path for _, path in path_refs if path}),
        "unique_nonempty_audio_paths_across_both_slots": len(path_refs),
        "unique_path_pairs": len(pairs),
        "source_inference_note": (
            "row fields are preferred; otherwise source is inferred from the path. "
            "UNCLASSIFIED paths require manual inspection."
        ),
        "sources": source_summary,
        "row_field_values": {
            key: dict(counter.most_common()) for key, counter in row_fields.items()
        },
        "processed_clip_roots": dict(processed_root_rows.most_common()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    write_tsv(
        args.output_dir / "audio_paths.tsv",
        ["source", "path", "references", "prefix"],
        [
            [source, path, count, path_prefix(path)]
            for (source, path), count in sorted(path_refs.items())
        ],
    )
    write_tsv(
        args.output_dir / "audio_pairs.tsv",
        ["filepath1", "filepath2", "rows"],
        [[path1, path2, count] for (path1, path2), count in sorted(pairs.items())],
    )
    write_tsv(
        args.output_dir / "source_prefixes.tsv",
        ["source", "prefix", "references"],
        [
            [source, prefix, count]
            for (source, prefix), count in sorted(prefix_refs.items())
        ],
    )
    write_tsv(
        args.output_dir / "raw_audio_paths.tsv",
        ["source", "field", "path", "references"],
        [
            [source, field, path, count]
            for (source, field, path), count in sorted(raw_path_refs.items())
        ],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
