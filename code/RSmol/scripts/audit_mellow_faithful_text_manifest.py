#!/usr/bin/env python3
"""CPU-only full text-contract audit for the isolated Mellow-faithful route."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


RSMOL_ROOT = Path(__file__).resolve().parents[1]
if str(RSMOL_ROOT) not in sys.path:
    sys.path.insert(0, str(RSMOL_ROOT))

from audio_smollm2_135m_mellow_shared_store_configurable_epochs.data import (  # noqa: E402
    prepare_mellow_text_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan every manifest row on CPU without starting training or loading model weights."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=968_059)
    parser.add_argument("--max-error-examples", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=50_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve(strict=True)
    report_path = args.report.expanduser().resolve(strict=False)
    if args.expected_rows <= 0 or args.max_error_examples < 0 or args.progress_every <= 0:
        raise ValueError("expected-rows/progress-every must be positive and max-error-examples nonnegative")

    started = time.perf_counter()
    digest = hashlib.sha256()
    rows = valid_rows = invalid_rows = 0
    recovered = Counter({"prompt": 0, "answer": 0, "caption1": 0, "caption2": 0})
    template_rows: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    invalid_examples: list[dict[str, Any]] = []

    with manifest.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            manifest_index = rows
            rows += 1
            try:
                row = json.loads(raw_line)
                if not isinstance(row, dict):
                    raise TypeError(f"expected JSON object, got {type(row).__name__}")
                audit = prepare_mellow_text_row(row)
            except Exception as exc:
                audit = {
                    "valid": False,
                    "group": "parse_error",
                    "missing": ["json_row"],
                    "recovered_from_metadata": {},
                    "error": repr(exc),
                }

            template_rows[str(audit.get("group", "unknown"))] += 1
            recovered.update(audit.get("recovered_from_metadata", {}))
            if audit.get("valid") is True:
                valid_rows += 1
            else:
                invalid_rows += 1
                missing_fields.update(str(value) for value in audit.get("missing", []))
                if len(invalid_examples) < args.max_error_examples:
                    invalid_examples.append({
                        "manifest_index": manifest_index,
                        "line_number": line_number,
                        **{key: value for key, value in audit.items() if key != "recovered_from_metadata"},
                    })

            if rows % args.progress_every == 0:
                print(
                    f"[mellow-text-audit] rows={rows} valid={valid_rows} invalid={invalid_rows}",
                    file=sys.stderr,
                    flush=True,
                )

    count_matches = rows == args.expected_rows
    status = "PASS" if invalid_rows == 0 and count_matches else "FAIL"
    report = {
        "status": status,
        "manifest": str(manifest),
        "manifest_sha256": digest.hexdigest(),
        "expected_rows": args.expected_rows,
        "rows": rows,
        "row_count_matches": count_matches,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "recovered_from_metadata": dict(recovered),
        "template_rows": dict(sorted(template_rows.items())),
        "invalid_missing_fields": dict(sorted(missing_fields.items())),
        "invalid_examples": invalid_examples,
        "validation_rng_calls": 0,
        "cpu_only": True,
        "training_startup_full_manifest_scan": False,
        "training_invalid_row_policy": "lazy deterministic forward replacement",
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "rows": rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "report": str(report_path),
    }, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
