#!/usr/bin/env python3
"""Evaluate the partition-v2 audio SmolLM2-135M baseline on official MMAR."""
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
import evaluate_mmau_test_mini_audio_smollm2 as smollm2  # noqa: E402


DEFAULT_CHECKPOINT = smollm2.DEFAULT_CHECKPOINT


def parse_args(argv: Sequence[str] | None = None):
    return official.parse_args(
        argv,
        default_checkpoint=DEFAULT_CHECKPOINT,
        description=__doc__,
    )


def run(args):
    return official.run(
        args,
        load_runtime_model=smollm2._load_runtime_model,
        run_model_generation=smollm2._run_model_generation,
        stage="mmar_audio_smollm2_official_accuracy",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps({
        "stage": report.get("stage"),
        "status": report.get("status"),
        "mode": report.get("mode"),
        "records": report.get("records"),
        "official_evaluation": report.get("official_evaluation"),
        "report": str(args.output_dir / "evaluation_report.json"),
    }, ensure_ascii=False, default=smollm2._json_default))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
