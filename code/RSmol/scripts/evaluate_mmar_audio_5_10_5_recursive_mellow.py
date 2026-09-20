#!/usr/bin/env python3
"""Evaluate the fixed-recursive partition-v2 audio checkpoint on official MMAR."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import evaluate_mmar_5_10x2_5_mesh_mellow as official  # noqa: E402
import evaluate_mmau_test_mini_audio_5_10_5_recursive_mellow as recursive  # noqa: E402


DEFAULT_CHECKPOINT = recursive.DEFAULT_CHECKPOINT


def parse_args(argv: Sequence[str] | None = None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if not any(item == "--mode" or item.startswith("--mode=") for item in raw):
        raw = ["--mode", "full", *raw]
    return official.parse_args(
        raw,
        default_checkpoint=DEFAULT_CHECKPOINT,
        description=__doc__,
    )


def run(args):
    report = official.run(
        args,
        load_runtime_model=recursive._load_runtime_model,
        run_model_generation=recursive._run_model_generation,
        stage="mmar_audio_5_10_5_recursive_official_accuracy",
    )
    report.setdefault("protocol", {})["logical_trace"] = (
        "exact physical trace 0..14,5..14,15..19 verified on every generation step"
    )
    inference_failures = int(report.get("records", {}).get("skip_reasons", {}).get("sample_exception", 0))
    if inference_failures:
        report["status"] = "FAILED"
        report["fatal_error"] = {
            "error": f"{inference_failures} recursive generation failures were recorded as skipped rows",
            "detail": "Inspect skipped.jsonl; do not interpret official accuracy as a valid model score.",
        }
    recursive.official._write_json(args.output_dir / "evaluation_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps({
        "stage": report.get("stage"),
        "status": report.get("status"),
        "mode": report.get("mode"),
        "records": report.get("records", {}),
        "official_evaluation": report.get("official_evaluation", {}),
        "report": str(args.output_dir / "evaluation_report.json"),
    }, ensure_ascii=False, default=recursive._json_default))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
