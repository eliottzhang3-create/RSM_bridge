#!/usr/bin/env python3
"""CPU-only inventory and exact comparison audit for the x6 text parquet store."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--compare-report", type=Path)
    parser.add_argument("--skip-content-hash", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, *, hash_content: bool) -> dict[str, Any]:
    import pyarrow.parquet as pq

    root = root.resolve(strict=True)
    paths = sorted(root.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards under {root}")
    files: list[dict[str, Any]] = []
    total_rows = 0
    total_bytes = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        if "text" not in parquet.schema_arrow.names:
            raise ValueError(f"missing text column: {path}")
        rows = int(parquet.metadata.num_rows)
        size = int(path.stat().st_size)
        if rows <= 0 or size <= 0:
            raise ValueError(f"empty parquet shard: {path}")
        item: dict[str, Any] = {"name": path.name, "bytes": size, "rows": rows}
        if hash_content:
            item["sha256"] = sha256(path)
        files.append(item)
        total_rows += rows
        total_bytes += size
    identity = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "root": str(root),
        "shards": len(files),
        "rows": total_rows,
        "bytes": total_bytes,
        "content_hashes": hash_content,
        "inventory_sha256": identity,
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    current = inventory(args.data_dir, hash_content=not args.skip_content_hash)
    comparison_matches = None
    compared_to = None
    if args.compare_report:
        reference = json.loads(
            args.compare_report.resolve(strict=True).read_text(encoding="utf-8")
        )
        reference_inventory = reference.get("inventory", reference)
        compared_to = str(args.compare_report.resolve())
        comparison_matches = current["files"] == reference_inventory.get("files")
    status = "PASS" if comparison_matches is not False else "FAIL"
    report = {
        "status": status,
        "comparison_matches": comparison_matches,
        "compared_to": compared_to,
        "inventory": current,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "shards": current["shards"],
                "rows": current["rows"],
                "bytes": current["bytes"],
                "comparison_matches": comparison_matches,
                "report": str(args.report_path),
            }
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
