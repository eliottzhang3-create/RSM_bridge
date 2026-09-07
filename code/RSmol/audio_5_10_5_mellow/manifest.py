"""Deterministic, waveform-free ReasonAQA manifest preparation.

The remote audio trees are intentionally treated as opaque path namespaces.
This module indexes filenames only; it never opens an audio file.  A later
stage may optionally inspect waveform metadata, but Stage 1 must remain safe
on a login node and must not silently choose another sample.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AUDIO_EXTENSIONS = frozenset(
    {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".webm", ".wma"}
)


class ManifestAuditError(ValueError):
    """An input or path resolution error that must be visible in the report."""


def canonical_json_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible values."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _json_records(payload: Any, *, source: Path) -> list[dict[str, Any]]:
    """Accept list JSON and common mapping wrappers without losing metadata."""

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("data", "items", "examples", "records", "annotations", "questions"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                records = candidate
                break
        if records is None and payload and all(isinstance(value, dict) for value in payload.values()):
            # Some exports use an ID -> annotation mapping.  Keep the ID in
            # the original record rather than dropping it.
            records = []
            for record_id, value in payload.items():
                item = dict(value)
                item.setdefault("record_id", record_id)
                records.append(item)
        if records is None:
            raise ManifestAuditError(f"{source}: expected a list or a mapping containing a list")
    else:
        raise ManifestAuditError(f"{source}: top-level JSON must be a list or object")
    if not all(isinstance(record, dict) for record in records):
        bad = next(type(record).__name__ for record in records if not isinstance(record, dict))
        raise ManifestAuditError(f"{source}: every record must be an object, found {bad}")
    return [dict(record) for record in records]


def load_reasonaqa_json(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Read one split, returning records and its source SHA-256."""

    path = Path(path)
    if not path.is_file():
        raise ManifestAuditError(f"ReasonAQA split does not exist or is not a file: {path}")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - report exact source failure
        raise ManifestAuditError(f"could not parse JSON {path}: {type(exc).__name__}: {exc}") from exc
    return _json_records(payload, source=path), hashlib.sha256(raw).hexdigest()


def _iter_audio_files(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return ()
    return (
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def _path_key(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def build_audio_index(roots: Sequence[Path]) -> dict[str, Any]:
    """Build exact, basename, and stem indexes with deterministic ordering."""

    files: list[Path] = []
    root_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        root = Path(root).expanduser()
        discovered = sorted(_iter_audio_files(root), key=lambda item: _path_key(item))
        unique: list[Path] = []
        for path in discovered:
            key = _path_key(path.resolve())
            if key not in seen:
                seen.add(key)
                files.append(path.resolve())
                unique.append(path.resolve())
        root_records.append({"root": str(root), "exists": root.is_dir(), "audio_file_count": len(unique)})
    files = sorted(files, key=_path_key)
    by_basename: dict[str, list[str]] = {}
    by_stem: dict[str, list[str]] = {}
    by_name: dict[str, list[str]] = {}
    path_roots: dict[str, str] = {}
    for path in files:
        path_string = str(path)
        by_basename.setdefault(path.name.lower(), []).append(path_string)
        by_stem.setdefault(path.stem.lower(), []).append(path_string)
        by_name.setdefault(path.name.lower(), []).append(path_string)
        owners = [str(root) for root in roots if _path_key(path).startswith(_path_key(Path(root).resolve()) + "/")]
        path_roots[path_string] = owners[0] if owners else "<unknown>"
    duplicates = {
        key: values
        for key, values in by_basename.items()
        if len(values) > 1
    }
    return {
        "roots": root_records,
        "files": [str(path) for path in files],
        "by_basename": by_basename,
        "by_stem": by_stem,
        "by_name": by_name,
        "path_roots": path_roots,
        "duplicate_basenames": duplicates,
        "cross_root_duplicate_basenames": {
            key: sorted({path_roots.get(path, "<unknown>") for path in values})
            for key, values in duplicates.items()
            if len({path_roots.get(path, "<unknown>") for path in values}) > 1
        },
        "audio_file_count": len(files),
    }


def _find_value(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    wanted = {_normalise_key(name) for name in names}
    def visit(value: Any) -> Any:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if _normalise_key(key) in wanted:
                    return item
            for item in value.values():
                found = visit(item)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = visit(item)
                if found is not None:
                    return found
        return None
    return visit(record)


def _text_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value)


def _path_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _suffix_match(requested: str, candidate: Path) -> bool:
    request = requested.replace("\\", "/").lower().lstrip("/")
    candidate_text = _path_key(candidate).lstrip("/")
    return candidate_text == request or candidate_text.endswith("/" + request)


def resolve_audio_path(raw_value: Any, index: Mapping[str, Any], *, preferred_tokens: Sequence[str] = ()) -> dict[str, Any]:
    """Resolve one filepath deterministically, retaining all ambiguity detail."""

    values = _path_values(raw_value)
    if not values:
        return {"status": "empty", "raw": raw_value, "candidates": []}
    if len(values) > 1:
        return {"status": "invalid", "raw": raw_value, "candidates": values, "reason": "multiple paths"}
    requested = values[0]
    direct = Path(requested).expanduser()
    if direct.is_file():
        return {"status": "resolved", "raw": requested, "path": str(direct.resolve()), "method": "direct"}
    preferred = [str(token).lower() for token in preferred_tokens if token]
    def prefer(items: list[Path]) -> list[Path]:
        if not preferred:
            return items
        selected = [item for item in items if any(token in index.get("path_roots", {}).get(str(item), "").lower() for token in preferred)]
        return selected or items
    basename = Path(requested).name.lower()
    stem = Path(requested).stem.lower()
    basename_candidates = prefer([Path(item) for item in index.get("by_basename", {}).get(basename, [])])
    exact_suffix = [item for item in basename_candidates if _suffix_match(requested, item)]
    if len(exact_suffix) == 1:
        return {"status": "resolved", "raw": requested, "path": str(exact_suffix[0]), "method": "relative_suffix"}
    if len(exact_suffix) > 1:
        return {"status": "ambiguous", "raw": requested, "candidates": [str(item) for item in exact_suffix], "method": "relative_suffix"}
    if len(basename_candidates) == 1:
        return {"status": "resolved", "raw": requested, "path": str(basename_candidates[0]), "method": "basename"}
    if len(basename_candidates) > 1:
        stem_candidates = prefer([Path(item) for item in index.get("by_stem", {}).get(stem, [])])
        if len(stem_candidates) == 1:
            return {"status": "resolved", "raw": requested, "path": str(stem_candidates[0]), "method": "unique_stem_after_duplicate_basename"}
        return {"status": "ambiguous", "raw": requested, "candidates": [str(item) for item in basename_candidates], "method": "basename"}
    stem_candidates = prefer([Path(item) for item in index.get("by_stem", {}).get(stem, [])])
    if len(stem_candidates) == 1:
        return {"status": "resolved", "raw": requested, "path": str(stem_candidates[0]), "method": "stem"}
    if len(stem_candidates) > 1:
        return {"status": "ambiguous", "raw": requested, "candidates": [str(item) for item in stem_candidates], "method": "stem"}
    return {"status": "missing", "raw": requested, "candidates": [], "method": "index"}


def _split_name(path: Path) -> str:
    name = path.stem.lower()
    return name if name in {"train", "val", "valid", "validation", "test"} else path.parent.name.lower()


def _manifest_record(record: Mapping[str, Any], *, split: str, row_index: int, index: Mapping[str, Any], allow_missing: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_one = _find_value(record, ("filepath1", "audio1path", "audio1", "audio_path", "audiopath", "file_path"))
    raw_two = _find_value(record, ("filepath2", "audio2path", "audio2"))
    task_value = _find_value(record, ("taskname", "task_name", "task"))
    task_text = str(task_value or "").lower()
    preferred_tokens = ("audiocaps",) if "audiocap" in task_text else ("clotho",) if "clotho" in task_text else ()
    one = resolve_audio_path(raw_one, index, preferred_tokens=preferred_tokens)
    two_empty = not _path_values(raw_two)
    two = one if two_empty else resolve_audio_path(raw_two, index, preferred_tokens=preferred_tokens)
    failures: list[dict[str, Any]] = []
    for slot, result in (("filepath1", one), ("filepath2", two)):
        if result["status"] in {"missing", "ambiguous", "invalid", "empty"}:
            failures.append({"slot": slot, **result})
    if failures and not allow_missing:
        pass
    output = {
        "split": split,
        "row_index": row_index,
        "taskname": task_value,
        "subtype": _find_value(record, ("subtype", "sub_type", "type")),
        "input": _find_value(record, ("input", "question", "prompt")),
        "answer": _find_value(record, ("answer", "answers", "target", "label")),
        "captions": _find_value(record, ("captions", "caption", "text")),
        "caption1": _find_value(record, ("caption1", "caption_1")),
        "caption2": _find_value(record, ("caption2", "caption_2")),
        "filepath1_raw": raw_one,
        "filepath2_raw": raw_two,
        "audio1_path": one.get("path"),
        "audio2_path": two.get("path"),
        "audio1_resolution": one,
        "audio2_resolution": two,
        "audio2_source": "filepath1_duplicate" if two_empty else "filepath2",
        "audio2_reused": bool(two_empty),
        "is_duplicate": bool(two_empty or (one.get("path") and one.get("path") == two.get("path"))),
        "metadata": dict(record),
    }
    return output, {"failures": failures, "duplicate": output["is_duplicate"], "two_empty": two_empty}


def build_reasonaqa_manifests(
    split_paths: Mapping[str, Path],
    audio_roots: Sequence[Path],
    *,
    allow_missing: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Prepare all splits and return rows plus a detailed audit report."""

    index = build_audio_index(audio_roots)
    manifests: dict[str, list[dict[str, Any]]] = {}
    report: dict[str, Any] = {
        "stage": "stage1_reasonaqa_manifest_5_10_5_mellow",
        "status": "PASS",
        "allow_missing": bool(allow_missing),
        "index": {key: value for key, value in index.items() if key not in {"files", "by_basename", "by_stem", "by_name"}},
        "splits": {},
        "counts": {"records": 0, "resolved": 0, "missing": 0, "ambiguous": 0, "invalid": 0, "duplicate_audio2": 0, "audio2_reused": 0},
        "task_distribution": {},
        "subtype_distribution": {},
        "hard_failures": [],
        "warnings": [],
        "input_hashes": {},
        "input_paths": {split: str(Path(path)) for split, path in split_paths.items()},
    }
    for split, path in split_paths.items():
        records, source_hash = load_reasonaqa_json(Path(path))
        report["input_hashes"][split] = source_hash
        rows: list[dict[str, Any]] = []
        split_stats: Counter[str] = Counter()
        for row_index, record in enumerate(records):
            row, details = _manifest_record(record, split=split, row_index=row_index, index=index, allow_missing=allow_missing)
            rows.append(row)
            report["counts"]["records"] += 1
            split_stats["records"] += 1
            if row["audio2_reused"]:
                report["counts"]["audio2_reused"] += 1
                split_stats["audio2_reused"] += 1
            if row["is_duplicate"]:
                report["counts"]["duplicate_audio2"] += 1
                split_stats["duplicate_audio2"] += 1
            task = str(row["taskname"] if row["taskname"] is not None else "<missing>")
            subtype = str(row["subtype"] if row["subtype"] is not None else "<missing>")
            report["task_distribution"][task] = report["task_distribution"].get(task, 0) + 1
            report["subtype_distribution"][subtype] = report["subtype_distribution"].get(subtype, 0) + 1
            for failure in details["failures"]:
                kind = failure["status"]
                report["counts"][kind] += 1
                split_stats[kind] += 1
                item = {"split": split, "row_index": row_index, **failure}
                if allow_missing:
                    report["warnings"].append(item)
                else:
                    report["hard_failures"].append(item)
            if not details["failures"]:
                report["counts"]["resolved"] += 1
                split_stats["resolved"] += 1
        manifests[split] = rows
        report["splits"][split] = dict(sorted(split_stats.items()))
    if report["hard_failures"]:
        report["status"] = "FAIL"
    elif report["warnings"]:
        report["status"] = "PASS_WITH_WARNINGS"
    report["summary"] = {
        "records": report["counts"]["records"],
        "resolved": report["counts"]["resolved"],
        "missing": report["counts"]["missing"],
        "ambiguous": report["counts"]["ambiguous"],
        "invalid": report["counts"]["invalid"],
        "audio2_reused": report["counts"]["audio2_reused"],
        "duplicate_audio2": report["counts"]["duplicate_audio2"],
        "hard_failures": len(report["hard_failures"]),
        "warnings": len(report["warnings"]),
    }
    report["manifest_hash"] = canonical_json_hash(manifests)
    report["index_hash"] = canonical_json_hash(index)
    return manifests, report


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str) for row in rows]
    payload = ("\n".join(lines) + "\n") if lines else ""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)
