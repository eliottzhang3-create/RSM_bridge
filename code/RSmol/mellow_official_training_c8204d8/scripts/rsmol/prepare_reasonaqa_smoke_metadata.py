#!/usr/bin/env python3
"""Create a deterministic compact ReasonAQA subset for an 8-GPU smoke run."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rows < 256:
        raise ValueError("--rows must be at least 256 for global batch 256")
    if not args.input_json.is_file():
        raise FileNotFoundError(args.input_json)

    records = json.loads(args.input_json.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("ReasonAQA metadata must be a JSON list")
    if len(records) < args.rows:
        raise ValueError(f"requested {args.rows} rows, but dataset has {len(records)}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("every ReasonAQA record must be an object")

    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(range(len(records)), args.rows))
    selected = [records[index] for index in selected_indices]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_name(
        f".{args.output_json.name}.tmp-{__import__('os').getpid()}"
    )
    temporary.write_text(
        json.dumps(selected, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output_json)

    report = {
        "status": "PASS",
        "input_json": str(args.input_json.resolve()),
        "output_json": str(args.output_json.resolve()),
        "input_rows": len(records),
        "output_rows": len(selected),
        "seed": args.seed,
        "first_selected_index": selected_indices[0],
        "last_selected_index": selected_indices[-1],
        "created_unix": time.time(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
