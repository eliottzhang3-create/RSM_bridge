#!/usr/bin/env python3
"""Rescore existing MMAU predictions with Mellow issue #5's label metric.

This command performs no model loading or inference.  It isolates the effect
of the scorer only; it cannot retrofit the author-reply prompt, audio, or
generation protocol onto predictions produced by an older run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as mmau  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-rows", type=int, choices=(5, 1000))
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir or args.predictions.parent
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.predictions.is_file():
        raise FileNotFoundError(f"predictions JSON not found: {args.predictions}")
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("predictions JSON must contain a list")
    if args.expected_rows is not None and len(predictions) != args.expected_rows:
        raise ValueError(
            f"prediction denominator mismatch: expected={args.expected_rows} actual={len(predictions)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score = mmau.write_mellow_author_reply_evaluation(args.output_dir, predictions)
    report = {
        "status": "PASS",
        "operation": "scorer_only_no_inference",
        "source_predictions": str(args.predictions.resolve()),
        "source_predictions_sha256": _sha256(args.predictions),
        "prediction_count": len(predictions),
        "scorer": mmau.MELLOW_AUTHOR_REPLY_SCORER,
        "score": score["total"],
        "warning": (
            "This result changes only the scoring rule. It is not a matched-protocol "
            "Mellow comparison unless the source run already used the author-reply prompt, "
            "audio preprocessing, max_len=300, and decoding protocol."
        ),
    }
    mmau._write_json(args.output_dir / "mellow_author_reply_rescore_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
