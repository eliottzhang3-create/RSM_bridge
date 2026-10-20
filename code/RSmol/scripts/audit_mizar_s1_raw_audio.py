#!/usr/bin/env python3
"""Audit raw audio references in a Mizar Stage-1 manifest.

The released Mizar manifest contains historical processed ``filepath1`` and
``filepath2`` values.  This tool intentionally audits the raw source fields
(``filepath1_original`` for ReasonAQA and ``audio_path`` for AudioMCQ) instead
of checking those processed cache paths.  It is designed for a login node:
only the Python standard library is required for path auditing.  Optional
CPU-only decoding can be enabled with ``--decode`` when ``ffprobe`` or
``soundfile`` is available.

Mapping JSON format::

    {
      "rules": [
        {
          "name": "reasonaqa_audiocaps",
          "field": "filepath1_original",
          "prefix": "AudioCapsLarger/",
          "roots": ["/datasets/audiocaps_v2"]
        }
      ]
    }

``prefix`` is removed before joining each candidate root.  The first existing
candidate wins.  Keeping several roots in one rule is useful when a dataset
has separate train/validation directories.  A rule may set ``match`` to
``basename`` or ``normalized_stem`` to search recursively below its roots
after the exact candidates fail; only a unique match is accepted.  Original
manifest text is never rewritten.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import unicodedata
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


CHUNK_SIZE = 1024 * 1024
ALLOWED_MUSICCAPS = "musiccaps"
ALLOWED_OTHER500 = "speechcraft_train_other_500"


def _open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_json_array(handle, first_chunk: str) -> Iterator[Any]:
    """Stream a JSON array without materialising the 1.5M-row manifest."""

    decoder = json.JSONDecoder()
    buffer = first_chunk
    position = 0
    ended = False

    def refill() -> bool:
        nonlocal buffer, position
        extra = handle.read(CHUNK_SIZE)
        if extra:
            buffer += extra
            return True
        return False

    while True:
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            if not refill():
                raise ValueError("unexpected end of manifest JSON array")
        if not ended:
            if buffer[position] != "[":
                raise ValueError("manifest is not a JSON array")
            position += 1
            ended = True
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if not refill():
                    raise ValueError("unexpected end of manifest JSON array")
            if buffer[position] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if not refill():
                    raise ValueError("invalid or truncated JSON array")
                continue
            position = end
            yield value
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if not refill():
                    raise ValueError("unexpected end after JSON row")
            if buffer[position] == ",":
                position += 1
            elif buffer[position] == "]":
                return
            else:
                raise ValueError("expected ',' or ']' after JSON row")
            # Compact consumed data so a long manifest does not grow in RAM.
            if position > CHUNK_SIZE:
                buffer = buffer[position:]
                position = 0


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Read a JSON array, JSONL file, or their gzip-compressed equivalent."""

    with _open_text(path) as handle:
        first = handle.read(1)
        while first and first.isspace():
            first = handle.read(1)
        if not first:
            return
        if first == "[":
            for row in _iter_json_array(handle, first):
                if not isinstance(row, dict):
                    raise ValueError("manifest contains a non-object row")
                yield row
            return
        first_line = first + handle.readline()
        for line_number, line in enumerate(first_line.splitlines(), start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                yield row
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            yield row


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def slash(value: str) -> str:
    return value.replace("\\", "/")


def source_label(row: dict[str, Any]) -> str:
    for key in ("source_dataset", "source", "dataset", "source_name"):
        value = text(row.get(key))
        if value:
            return value
    return "UNCLASSIFIED"


def source_key(value: str) -> str:
    return "_".join(value.lower().replace("-", "_").split())


def is_allowed_row(row: dict[str, Any], source: str, raw_path: str) -> tuple[bool, str]:
    lowered_source = source_key(source)
    lowered_path = slash(raw_path).lower()
    if lowered_source == ALLOWED_MUSICCAPS:
        return True, ALLOWED_MUSICCAPS
    if lowered_source == "speechcraft" and "/train-other-500/" in f"/{lowered_path.strip('/')}/":
        return True, ALLOWED_OTHER500
    # Some manifests only expose a path-level SpeechCraft label.
    if "libritts_r_audios/train-other-500/" in lowered_path:
        return True, ALLOWED_OTHER500
    return False, "retained"


def raw_reference(row: dict[str, Any], source: str) -> tuple[str, str]:
    lowered = source_key(source)
    if lowered in {"reasonaqa_oneaudio", "reasonaqa", "mellow"}:
        value = text(row.get("filepath1_original"))
        if value:
            return "filepath1_original", value
    value = text(row.get("audio_path"))
    if value:
        return "audio_path", value
    value = text(row.get("filepath1_original"))
    if value:
        return "filepath1_original", value
    return "", ""


def load_rules(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        payload = {"rules": payload}
    if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
        raise ValueError("mapping JSON must contain a top-level 'rules' list")
    rules: list[dict[str, Any]] = []
    for index, raw in enumerate(payload["rules"]):
        if not isinstance(raw, dict):
            raise ValueError(f"mapping rule {index} is not an object")
        name = text(raw.get("name")) or f"rule_{index}"
        field = text(raw.get("field"))
        prefix = slash(text(raw.get("prefix")))
        roots = raw.get("roots", raw.get("root"))
        if isinstance(roots, (str, Path)):
            roots = [str(roots)]
        match = text(raw.get("match") or "exact").lower()
        if match not in {"exact", "basename", "normalized_stem"}:
            raise ValueError(
                f"mapping rule {name!r} has unsupported match {match!r}; "
                "use exact, basename, or normalized_stem"
            )
        if not field or not isinstance(roots, list) or not roots:
            raise ValueError(f"mapping rule {name!r} requires field and roots")
        normalised_roots = [Path(str(root)).expanduser() for root in roots if text(root)]
        if not normalised_roots:
            raise ValueError(f"mapping rule {name!r} has no non-empty roots")
        rules.append({
            "name": name,
            "field": field,
            "prefix": prefix,
            "roots": normalised_roots,
            "match": match,
        })
    return rules


def normalized_stem_key(value: str) -> str:
    stem = Path(slash(value)).stem
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_stem.casefold())


def build_recursive_indexes(rules: list[dict[str, Any]]) -> None:
    """Build small basename/stem indexes only for rules that request fallback matching."""

    for rule in rules:
        if rule["match"] == "exact":
            continue
        basename: defaultdict[str, list[str]] = defaultdict(list)
        normalized_stem: defaultdict[str, list[str]] = defaultdict(list)
        seen: set[str] = set()
        for root in rule["roots"]:
            if not root.is_dir():
                continue
            for candidate in root.rglob("*"):
                if not candidate.is_file():
                    continue
                resolved = str(candidate.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                basename[candidate.name.casefold()].append(resolved)
                normalized_stem[normalized_stem_key(candidate.name)].append(resolved)
        rule["_basename_index"] = dict(basename)
        rule["_normalized_stem_index"] = dict(normalized_stem)


def resolve(raw_path: str, field: str, rules: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_rules = []
    normalized = slash(raw_path)
    for rule in rules:
        if rule["field"] != field:
            continue
        prefix = rule["prefix"]
        if prefix and not normalized.startswith(prefix):
            continue
        candidate_rules.append(rule)
    # Longest prefix wins, matching Mizar's path-map convention.
    candidate_rules.sort(key=lambda item: len(item["prefix"]), reverse=True)
    first_missing: dict[str, Any] | None = None
    for rule in candidate_rules:
        prefix = rule["prefix"]
        tail = normalized[len(prefix):] if prefix else normalized
        candidate_paths = [str(root / Path(tail)) for root in rule["roots"]]
        for root in rule["roots"]:
            candidate = root / Path(tail)
            if candidate.is_file():
                return {
                    "status": "exists",
                    "rule": rule["name"],
                    "path": str(candidate),
                    "candidate_paths": candidate_paths,
                    "method": "exact",
                }
        if rule["match"] != "exact":
            if rule["match"] == "basename":
                key = Path(tail).name.casefold()
                matches = list(rule.get("_basename_index", {}).get(key, []))
            else:
                key = normalized_stem_key(tail)
                matches = list(rule.get("_normalized_stem_index", {}).get(key, []))
            if len(matches) == 1:
                return {
                    "status": "exists",
                    "rule": rule["name"],
                    "path": matches[0],
                    "candidate_paths": candidate_paths + matches,
                    "method": rule["match"],
                }
            if len(matches) > 1:
                return {
                    "status": "ambiguous",
                    "rule": rule["name"],
                    "path": "",
                    "candidate_paths": candidate_paths + sorted(matches),
                    "method": rule["match"],
                }
        if first_missing is None:
            first_missing = {
                "status": "missing",
                "rule": rule["name"],
                "path": candidate_paths[0],
                "candidate_paths": candidate_paths,
                "method": "exact",
            }
    if first_missing is not None:
        return first_missing
    # Permit an absolute raw path as a final explicit fallback.
    absolute = Path(normalized)
    if absolute.is_absolute() and absolute.is_file():
        return {"status": "exists", "rule": "absolute", "path": str(absolute), "candidate_paths": [str(absolute)]}
    return {"status": "unmapped", "rule": "", "path": "", "candidate_paths": []}


def tsv_value(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def decode_with_wave(path: Path) -> dict[str, Any] | None:
    if path.suffix.lower() not in {".wav", ".wave"}:
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            channels = handle.getnchannels()
            if frames <= 0 or rate <= 0 or channels <= 0:
                raise ValueError("empty or invalid WAV")
            return {"sample_rate": rate, "channels": channels, "duration": frames / rate, "decoder": "wave"}
    except Exception as exc:
        return {"error": f"wave: {exc}", "decoder": "wave"}


def decode_with_soundfile(path: Path) -> dict[str, Any] | None:
    try:
        import soundfile as sf  # type: ignore
    except Exception:
        return None
    try:
        info = sf.info(str(path))
        if int(info.frames) <= 0 or int(info.samplerate) <= 0 or int(info.channels) <= 0:
            raise ValueError("empty or invalid audio")
        return {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration": float(info.frames) / float(info.samplerate),
            "decoder": "soundfile",
        }
    except Exception as exc:
        return {"error": f"soundfile: {exc}", "decoder": "soundfile"}


def decode_with_ffprobe(path: Path, executable: str) -> dict[str, Any]:
    command = [
        executable,
        "-v", "error",
        "-show_entries", "stream=codec_type,sample_rate,channels,duration",
        "-of", "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except Exception as exc:
        return {"error": f"ffprobe: {exc}", "decoder": "ffprobe"}
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().replace("\n", " ")
        return {"error": f"ffprobe exit {completed.returncode}: {detail[:300]}", "decoder": "ffprobe"}
    try:
        payload = json.loads(completed.stdout)
        streams = [item for item in payload.get("streams", []) if item.get("codec_type") == "audio"]
        if not streams:
            raise ValueError("no audio stream")
        stream = streams[0]
        rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        duration = float(stream.get("duration") or 0.0)
        if rate <= 0 or channels <= 0 or not math.isfinite(duration) or duration <= 0:
            raise ValueError("empty or invalid audio stream")
        return {"sample_rate": rate, "channels": channels, "duration": duration, "decoder": "ffprobe"}
    except Exception as exc:
        return {"error": f"ffprobe JSON: {exc}", "decoder": "ffprobe"}


def inspect_decode(path: Path, ffprobe: str | None) -> dict[str, Any]:
    result = decode_with_wave(path)
    if result is not None:
        return result
    result = decode_with_soundfile(path)
    if result is not None and "error" not in result:
        return result
    if ffprobe:
        return decode_with_ffprobe(path, ffprobe)
    if result is not None:
        return result
    return {"error": "no CPU decoder available (install soundfile or ffprobe)", "decoder": "unavailable"}


def write_jsonl(handle, item: dict[str, Any]) -> None:
    handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path, help="s1.json, s1.json.gz, or JSONL")
    parser.add_argument("--mapping-json", required=True, type=Path, help="dataset-specific raw-audio mapping JSON")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--decode", action="store_true", help="CPU decode each unique existing file")
    parser.add_argument("--decode-limit", type=int, default=0, help="maximum unique files to decode; 0 means all")
    parser.add_argument("--ffprobe", default=None, help="ffprobe executable; default searches PATH")
    parser.add_argument("--progress-every", type=int, default=100000)
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    if not args.mapping_json.is_file():
        parser.error(f"mapping JSON does not exist: {args.mapping_json}")
    if args.decode_limit < 0:
        parser.error("--decode-limit must be non-negative")
    rules = load_rules(args.mapping_json)
    build_recursive_indexes(rules)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ffprobe = args.ffprobe or shutil.which("ffprobe")

    allowed_missing = (args.output_dir / "allowed_missing.jsonl").open("w", encoding="utf-8")
    retained_missing = (args.output_dir / "retained_missing.jsonl").open("w", encoding="utf-8")
    unmapped = (args.output_dir / "unmapped.jsonl").open("w", encoding="utf-8")
    decode_failures = (args.output_dir / "decode_failures.jsonl").open("w", encoding="utf-8")
    issue_handles = (allowed_missing, retained_missing, unmapped, decode_failures)
    counters: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_status: Counter[tuple[str, str]] = Counter()
    rule_counts: Counter[tuple[str, str]] = Counter()
    reference_counts: Counter[str] = Counter()
    resolutions: dict[tuple[str, str], dict[str, Any]] = {}
    decode_results: dict[str, dict[str, Any]] = {}
    decode_attempts = 0

    try:
        for row_index, row in enumerate(iter_rows(args.manifest)):
            counters["rows"] += 1
            source = source_label(row)
            source_counts[source] += 1
            field, raw_path = raw_reference(row, source)
            allowed, allowed_kind = is_allowed_row(row, source, raw_path)
            base = {
                "row_index": row_index,
                "source": source,
                "source_dataset": text(row.get("source_dataset")),
                "stageE_variant": text(row.get("stageE_variant")),
                "field": field,
                "raw_path": raw_path,
                "sample_key": text(row.get("sample_key") or row.get("id")),
                "allowed_filter": allowed_kind if allowed else "",
            }
            if not field or not raw_path:
                status = "missing_raw_field"
                result = {"status": status, "rule": "", "path": "", "candidate_paths": []}
            else:
                reference_counts[f"{field}\t{raw_path}"] += 1
                cache_key = (field, raw_path)
                result = resolutions.get(cache_key)
                if result is None:
                    result = resolve(raw_path, field, rules)
                    resolutions[cache_key] = result
            status = str(result["status"])
            counters[status] += 1
            source_status[(source, status)] += 1
            rule_counts[(str(result.get("rule") or ""), status)] += 1
            base.update({"status": status, "rule": result.get("rule", ""), "resolved_path": result.get("path", "")})

            if status == "exists":
                counters["existing_rows"] += 1
                if args.decode:
                    resolved = str(result["path"])
                    if resolved not in decode_results and (args.decode_limit == 0 or decode_attempts < args.decode_limit):
                        decode_results[resolved] = inspect_decode(Path(resolved), ffprobe)
                        decode_attempts += 1
                    decoded = decode_results.get(resolved)
                    if decoded and "error" in decoded:
                        counters["decode_failures"] += 1
                        if allowed:
                            counters["allowed_decode_failure_rows"] += 1
                        else:
                            counters["retained_decode_failure_rows"] += 1
                        failure = dict(base)
                        failure.update(decoded)
                        write_jsonl(decode_failures, failure)
                        # Keep a failure attached to the source class for the summary.
                        source_status[(source, "decode_failure")] += 1
                continue

            counters[f"{status}_rows"] += 1
            issue = dict(base)
            if result.get("candidate_paths"):
                issue["candidate_paths"] = result["candidate_paths"]
            if allowed:
                counters["allowed_missing_rows"] += 1
                counters[f"allowed_{allowed_kind}_rows"] += 1
                write_jsonl(allowed_missing, issue)
            elif status == "unmapped":
                counters["retained_unmapped_rows"] += 1
                write_jsonl(unmapped, issue)
            else:
                counters["retained_missing_rows"] += 1
                write_jsonl(retained_missing, issue)
            if args.progress_every > 0 and (row_index + 1) % args.progress_every == 0:
                print(f"audited {row_index + 1} rows", flush=True)
    finally:
        for handle in issue_handles:
            handle.close()

    with (args.output_dir / "resolved_paths.tsv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("field\traw_path\treferences\tstatus\trule\tresolved_path\tcandidate_paths\n")
        for key, result in sorted(resolutions.items()):
            field, raw_path = key
            handle.write("\t".join([
                tsv_value(field), tsv_value(raw_path), str(reference_counts.get(f"{field}\t{raw_path}", 0)),
                tsv_value(result.get("status", "")), tsv_value(result.get("rule", "")),
                tsv_value(result.get("path", "")), tsv_value(" | ".join(result.get("candidate_paths", []))),
            ]) + "\n")

    summary = {
        "manifest": str(args.manifest),
        "mapping_json": str(args.mapping_json),
        "output_dir": str(args.output_dir),
        "rules": [
            {
                "name": rule["name"],
                "field": rule["field"],
                "prefix": rule["prefix"],
                "match": rule["match"],
                "roots": [str(root) for root in rule["roots"]],
            }
            for rule in rules
        ],
        "decoder": {"requested": bool(args.decode), "ffprobe": ffprobe, "unique_attempts": decode_attempts},
        "counts": dict(counters),
        "source_rows": dict(source_counts),
        "source_status": {f"{source}\t{status}": count for (source, status), count in sorted(source_status.items())},
        "rule_status": {f"{rule}\t{status}": count for (rule, status), count in sorted(rule_counts.items())},
        "unique_raw_references": len(resolutions),
        "unique_existing_files": sum(1 for item in resolutions.values() if item.get("status") == "exists"),
        "unique_missing_or_unmapped_references": sum(1 for item in resolutions.values() if item.get("status") != "exists"),
        "decode_results": decode_results if args.decode else {},
        "exit_policy": "allowed MusicCaps/train-other-500 issues are recorded but do not fail; retained missing/unmapped or decode failures return 2",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    hard_failures = (
        counters["retained_missing_rows"]
        + counters["retained_unmapped_rows"]
        + counters["retained_decode_failure_rows"]
    )
    return 2 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
