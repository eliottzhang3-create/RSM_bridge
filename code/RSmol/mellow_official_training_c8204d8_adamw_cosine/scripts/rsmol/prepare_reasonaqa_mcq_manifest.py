#!/usr/bin/env python3
"""Create and audit a CPU-only ReasonAQA MCQ manifest.

The source split is never modified.  Only records whose ``subtype`` is one
of the two explicit MCQ exports are copied to a new JSON manifest.  The
script also validates the fields consumed by the official Mellow dataset and
records source/output SHA-256 identities for later remote training audits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_SUBTYPES = ("AudioCaps-MCQ.json", "Clotho-MCQ.json")
REQUIRED_FIELDS = ("filepath1", "filepath2", "input", "answer")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError("source JSON must be a list of object records")
    return payload, raw


def build_manifest(source: Path, output: Path) -> dict[str, Any]:
    records, raw = load_records(source)
    subtype_counts = Counter(str(row.get("subtype", "<missing>")) for row in records)
    selected = [row for row in records if row.get("subtype") in ALLOWED_SUBTYPES]
    if not selected:
        raise ValueError("no MCQ records matched the allowed subtypes")

    missing_counts: Counter[str] = Counter()
    empty_counts: Counter[str] = Counter()
    for index, row in enumerate(selected):
        for field in REQUIRED_FIELDS:
            if field not in row:
                missing_counts[field] += 1
            elif row[field] is None or (isinstance(row[field], str) and not row[field].strip()):
                empty_counts[field] += 1
        if "filepath1" not in row or "filepath2" not in row:
            raise ValueError(f"selected row {index} is missing official dataset audio fields")
        if "input" not in row or "answer" not in row:
            raise ValueError(f"selected row {index} is missing official dataset text fields")
    if missing_counts:
        raise ValueError(f"selected records missing required fields: {dict(missing_counts)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_sha = sha256_file(output)
    report = {
        "status": "PASS",
        "contract": "reasonaqa_mcq_manifest_v1",
        "source_json": str(source.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "source_rows": len(records),
        "source_subtype_distribution": dict(sorted(subtype_counts.items())),
        "allowed_subtypes": list(ALLOWED_SUBTYPES),
        "selected_rows": len(selected),
        "selected_subtype_distribution": dict(sorted(Counter(str(row["subtype"]) for row in selected).items())),
        "selected_required_field_empty_counts": dict(sorted(empty_counts.items())),
        "output_manifest": str(output.resolve()),
        "output_sha256": output_sha,
        "output_bytes": output.stat().st_size,
        "output_format": "json_list_of_original_records",
        "gpu_required": False,
    }
    report_path = output.with_suffix(output.suffix + ".audit.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_json.expanduser().resolve(strict=True)
    if not source.is_file():
        raise SystemExit(f"source is not a regular file: {source}")
    report = build_manifest(source, args.output_json.expanduser().resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
