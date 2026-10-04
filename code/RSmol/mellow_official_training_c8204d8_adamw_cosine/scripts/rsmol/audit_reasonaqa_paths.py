#!/usr/bin/env python3
"""Audit ReasonAQA logical paths without opening audio waveforms.

The output mapping is intended for a later compute-node staging script. That
script will copy each referenced raw file into a job-local /dev/shm tree at
its original logical path, allowing the upstream Mellow Dataset to remain
unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_TRAIN_JSON = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/train.json")
DEFAULT_AUDIOCAPS_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2")
DEFAULT_CLOTHO_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2_1")
DEFAULT_CLOTHO_AQA_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_audio/audio_files")
DEFAULT_REPORT = Path("/tmp/mellow_reasonaqa_path_audit.json")
DEFAULT_MAPPING = Path("/tmp/mellow_reasonaqa_train_mapping.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--audiocaps-root", type=Path, default=DEFAULT_AUDIOCAPS_ROOT)
    parser.add_argument("--clotho-root", type=Path, default=DEFAULT_CLOTHO_ROOT)
    parser.add_argument("--clotho-aqa-root", type=Path, default=DEFAULT_CLOTHO_AQA_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--mapping-path", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--failure-preview", type=int, default=50)
    return parser.parse_args()


def normalize_logical(value: Any) -> str:
    text = str(value or "").strip().replace(chr(92), "/")
    while "//" in text:
        text = text.replace("//", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def logical_parts(logical: str) -> tuple[str, ...]:
    path = PurePosixPath(logical)
    parts = path.parts
    if path.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe logical path: {logical!r}")
    return tuple(parts)


def key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def normalized_stem(value: str) -> str:
    stem = PurePosixPath(value.replace(chr(92), "/")).stem
    decomposed = unicodedata.normalize("NFKD", stem)
    ascii_stem = decomposed.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_stem.casefold())


def build_normalized_stem_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    if not root.is_dir():
        return dict(index)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            index[normalized_stem(path.name)].append(path.resolve())
    return dict(index)


def infer_group(taskname: Any, logical: str) -> str:
    task = key(str(taskname or ""))
    prefix = key(logical.split("/", 1)[0])
    if "clothoaqa" in task or "clothoaqa" in prefix:
        return "clotho_aqa"
    if "audiocap" in task or "audiocap" in prefix:
        return "audiocaps"
    if "clotho" in task or "clotho" in prefix:
        return "clotho"
    return "unknown"


def suffix_at(parts: tuple[str, ...], markers: set[str]) -> tuple[str, ...] | None:
    wanted = {key(marker) for marker in markers}
    for index, part in enumerate(parts):
        if key(part) in wanted:
            return parts[index:]
    return None


def deduplicate_candidates(candidates: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    result = []
    seen = set()
    for method, path in candidates:
        normalized = os.path.normpath(str(path))
        if normalized not in seen:
            seen.add(normalized)
            result.append((method, Path(normalized)))
    return result


def group_candidates(
    logical: str,
    group: str,
    audiocaps_root: Path,
    clotho_root: Path,
    clotho_aqa_root: Path,
) -> list[tuple[str, Path]]:
    parts = logical_parts(logical)
    tail = parts[1:] if len(parts) > 1 else parts
    result: list[tuple[str, Path]] = []
    if group == "audiocaps":
        split = suffix_at(parts, {"train", "val", "test"})
        if split:
            result.append(("audiocaps_split_suffix", audiocaps_root.joinpath(*split)))
        result.append(("audiocaps_drop_prefix", audiocaps_root.joinpath(*tail)))
    elif group == "clotho":
        split = suffix_at(parts, {"development", "validation", "evaluation"})
        if split:
            result.append(("clotho_split_suffix", clotho_root.joinpath(*split)))
        result.append(("clotho_drop_prefix", clotho_root.joinpath(*tail)))
    elif group == "clotho_aqa":
        result.append(("clotho_aqa_basename", clotho_aqa_root / parts[-1]))
        result.append(("clotho_aqa_drop_prefix", clotho_aqa_root.joinpath(*tail)))
    else:
        for candidate_group in ("audiocaps", "clotho", "clotho_aqa"):
            result.extend(
                group_candidates(
                    logical,
                    candidate_group,
                    audiocaps_root,
                    clotho_root,
                    clotho_aqa_root,
                )
            )
    return deduplicate_candidates(result)


def resolve(
    logical: str,
    group: str,
    audiocaps_root: Path,
    clotho_root: Path,
    clotho_aqa_root: Path,
    clotho_aqa_name_index: dict[str, list[Path]],
) -> dict[str, Any]:
    try:
        candidates = group_candidates(
            logical,
            group,
            audiocaps_root,
            clotho_root,
            clotho_aqa_root,
        )
    except ValueError as exc:
        return {"status": "unsafe", "logical_path": logical, "group": group, "error": str(exc)}

    matches: dict[str, list[str]] = defaultdict(list)
    for method, candidate in candidates:
        if candidate.is_file():
            matches[str(candidate.resolve())].append(method)
    if len(matches) == 1:
        source, methods = next(iter(matches.items()))
        size = Path(source).stat().st_size
        return {
            "status": "resolved",
            "logical_path": logical,
            "group": group,
            "source_path": source,
            "source_size_bytes": int(size),
            "method": methods[0],
        }
    if len(matches) > 1:
        return {
            "status": "ambiguous",
            "logical_path": logical,
            "group": group,
            "matches": sorted(matches),
        }
    if group == "clotho_aqa":
        normalized_matches = clotho_aqa_name_index.get(normalized_stem(logical), [])
        if len(normalized_matches) == 1:
            source = normalized_matches[0]
            return {
                "status": "resolved",
                "logical_path": logical,
                "group": group,
                "source_path": str(source),
                "source_size_bytes": int(source.stat().st_size),
                "method": "clotho_aqa_normalized_stem",
            }
        if len(normalized_matches) > 1:
            return {
                "status": "ambiguous",
                "logical_path": logical,
                "group": group,
                "method": "clotho_aqa_normalized_stem",
                "matches": [str(path) for path in normalized_matches],
            }
    return {
        "status": "missing",
        "logical_path": logical,
        "group": group,
        "candidates": [str(path) for _, path in candidates],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + chr(10)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    train_json = args.train_json.expanduser().resolve(strict=True)
    roots = {
        # Resolve without strict=True so a missing root becomes a reportable
        # audit failure instead of aborting before prefix statistics are written.
        "audiocaps": args.audiocaps_root.expanduser().resolve(strict=False),
        "clotho": args.clotho_root.expanduser().resolve(strict=False),
        "clotho_aqa": args.clotho_aqa_root.expanduser().resolve(strict=False),
    }
    missing_roots = {
        group: str(root) for group, root in roots.items() if not root.is_dir()
    }
    clotho_aqa_name_index = build_normalized_stem_index(roots["clotho_aqa"])
    clotho_aqa_name_collisions = {
        stem: [str(path) for path in paths]
        for stem, paths in clotho_aqa_name_index.items()
        if len(paths) > 1
    }

    records = json.loads(train_json.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise SystemExit("ReasonAQA training metadata must be a non-empty JSON list")

    task_counts: Counter[str] = Counter()
    prefix_counts: Counter[str] = Counter()
    empty_slots: Counter[str] = Counter()
    reference_counts: Counter[tuple[str, str]] = Counter()
    prefix_examples: dict[str, list[str]] = defaultdict(list)

    for row_number, row in enumerate(records):
        if not isinstance(row, dict):
            raise SystemExit(f"row {row_number} is not a JSON object")
        taskname = str(row.get("taskname", ""))
        task_counts[taskname] += 1
        for slot in ("filepath1", "filepath2"):
            logical = normalize_logical(row.get(slot, ""))
            if not logical:
                empty_slots[slot] += 1
                continue
            prefix = logical.split("/", 1)[0]
            prefix_counts[prefix] += 1
            if len(prefix_examples[prefix]) < 5 and logical not in prefix_examples[prefix]:
                prefix_examples[prefix].append(logical)
            reference_counts[(logical, infer_group(taskname, logical))] += 1

    resolutions = {}
    for logical, group in sorted(reference_counts):
        resolutions[(logical, group)] = resolve(
            logical,
            group,
            roots["audiocaps"],
            roots["clotho"],
            roots["clotho_aqa"],
            clotho_aqa_name_index,
        )

    failures = [item for item in resolutions.values() if item["status"] != "resolved"]
    by_logical: dict[str, dict[str, Any]] = {}
    inconsistent = []
    for pair, item in resolutions.items():
        logical, group = pair
        if item["status"] != "resolved":
            continue
        existing = by_logical.get(logical)
        if existing and existing["source_path"] != item["source_path"]:
            inconsistent.append(
                {
                    "logical_path": logical,
                    "first_source": existing["source_path"],
                    "second_source": item["source_path"],
                    "second_group": group,
                }
            )
            continue
        if existing is None:
            by_logical[logical] = dict(item)
            by_logical[logical]["groups"] = []
            by_logical[logical]["reference_count"] = 0
        by_logical[logical]["reference_count"] += reference_counts[pair]
        if group not in by_logical[logical]["groups"]:
            by_logical[logical]["groups"].append(group)

    args.mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with args.mapping_path.open("w", encoding="utf-8") as handle:
        for logical in sorted(by_logical):
            item = by_logical[logical]
            entry = {
                "logical_path": logical,
                "source_path": item["source_path"],
                "source_size_bytes": item["source_size_bytes"],
                "group": item["group"],
                "groups": sorted(item["groups"]),
                "method": item["method"],
                "reference_count": item["reference_count"],
            }
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + chr(10))

    unique_sources = {item["source_path"]: item["source_size_bytes"] for item in by_logical.values()}
    resolved_items = [item for item in resolutions.values() if item["status"] == "resolved"]
    status = "PASS" if not failures and not inconsistent and not missing_roots else "FAIL"
    report = {
        "status": status,
        "contract": "reasonaqa_raw_audio_path_mapping_v1",
        "train_json": str(train_json),
        "roots": {name: str(path) for name, path in roots.items()},
        "missing_roots": missing_roots,
        "rows": len(records),
        "slot_references": sum(reference_counts.values()),
        "empty_slots": dict(empty_slots),
        "unique_logical_group_pairs": len(resolutions),
        "unique_logical_paths_resolved": len(by_logical),
        "unique_source_files": len(unique_sources),
        "total_source_bytes": sum(unique_sources.values()),
        "total_source_gib": sum(unique_sources.values()) / (1024 ** 3),
        "prefix_counts": dict(prefix_counts.most_common()),
        "prefix_examples": {name: values for name, values in sorted(prefix_examples.items())},
        "taskname_counts": dict(task_counts.most_common()),
        "resolution_method_counts": dict(Counter(item["method"] for item in resolved_items)),
        "resolved_group_counts": dict(Counter(item["group"] for item in resolved_items)),
        "clotho_aqa_normalized_stem_collisions": clotho_aqa_name_collisions,
        "failure_counts": dict(Counter(item["status"] for item in failures)),
        "failure_preview": failures[: max(0, args.failure_preview)],
        "inconsistent_logical_paths": inconsistent[: max(0, args.failure_preview)],
        "mapping_path": str(args.mapping_path),
    }
    write_json(args.report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
