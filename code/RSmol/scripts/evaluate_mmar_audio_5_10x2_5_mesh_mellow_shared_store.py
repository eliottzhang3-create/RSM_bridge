#!/usr/bin/env python3
"""Evaluate the completed 3-epoch shared-store Audio MeSH checkpoint on MMAR."""
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
import evaluate_mmau_test_mini_audio_5_10x2_5_mesh_mellow_shared_store as shared  # noqa: E402


DEFAULT_CHECKPOINT = shared.DEFAULT_CHECKPOINT
FIXED260_MAX_PROMPT_TOKENS = (
    shared.DEFAULT_MAX_CONTEXT_LENGTH - 260 - shared.DEFAULT_MAX_NEW_TOKENS
)


def parse_args(argv: Sequence[str] | None = None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if not any(item == "--mode" or item.startswith("--mode=") for item in raw):
        raw = ["--mode", "full", *raw]
    return official.parse_args(
        raw,
        default_checkpoint=DEFAULT_CHECKPOINT,
        max_prompt_tokens=FIXED260_MAX_PROMPT_TOKENS,
        description=__doc__,
    )


def run(args):
    report = official.run(
        args,
        load_runtime_model=shared._load_runtime_model,
        run_model_generation=shared._run_model_generation,
        prepare_prediction=shared.prepare_model_output_for_official_scorer,
        prediction_format=shared.PREDICTION_FORMAT,
        audio_prefix_tokens=260,
        protocol_description=(
            "official order; ReasonAQA lowercase labels; fixed260 two-slot single-audio "
            "prefix; decoded prediction passed verbatim; single cuda:0; bf16; greedy"
        ),
        stage="mmar_audio_5_10x2_5_mesh_mellow_shared_store_fixed260_official_accuracy",
    )
    report.setdefault("protocol", {})["logical_trace"] = (
        "exact MeSH 5-10-10-5 trace verified on every generation step"
    )
    inference_failures = int(
        report.get("records", {}).get("skip_reasons", {}).get("sample_exception", 0)
    )
    if inference_failures:
        report["status"] = "FAILED"
        report["fatal_error"] = {
            "error": f"{inference_failures} shared-store generation failures were recorded as skipped rows",
            "detail": "Inspect skipped.jsonl; do not interpret official accuracy as a valid model score.",
        }
    shared.official._write_json(args.output_dir / "evaluation_report.json", report)
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
    }, ensure_ascii=False, default=shared._json_default))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
