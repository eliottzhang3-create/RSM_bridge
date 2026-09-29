#!/usr/bin/env python3
"""Audit the local Qwen2-Audio-Instruct Hugging Face artifact."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_qwen2_audio_instruct_common as qwen  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=qwen.DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = qwen.audit_model_artifact(args.model_path, load_processor=True)
    except Exception as exc:
        report = {
            "status": "FAILED",
            "model_path": str(args.model_path),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
