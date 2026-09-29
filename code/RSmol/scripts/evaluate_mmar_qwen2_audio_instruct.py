#!/usr/bin/env python3
"""Evaluate local Qwen2-Audio-Instruct on MMAR with dual scoring."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import evaluate_mmar_5_10x2_5_mesh_mellow as official  # noqa: E402
import evaluate_qwen2_audio_instruct_common as qwen  # noqa: E402


MAX_PROMPT_TOKENS = 476
PREDICTION_FORMAT = "qwen2_audio_instruct_raw_generation_dual_scoring_v1"


def prepare_model_output_for_official_scorer(value: Any) -> str:
    """Preserve the decoded Qwen answer for both scorers."""

    return str(value)


def _load_runtime_model(
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    return qwen.load_runtime_model(args)


def _run_model_generation(
    model: Any,
    processor: Any,
    device: Any,
    sample: Mapping[str, Any],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    return qwen.generate_greedy(
        model,
        processor,
        device,
        str(sample["prompt"]),
        sample["waveform"],
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not any(item == "--mode" or item.startswith("--mode=") for item in raw):
        raw = ["--mode", "smoke", *raw]
    args = official.parse_args(
        raw,
        default_checkpoint=qwen.DEFAULT_MODEL_PATH,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        description=__doc__,
    )
    if args.dtype != "bf16":
        raise ValueError("Qwen2-Audio MMAR evaluation is fixed to --dtype bf16")
    return qwen.apply_qwen_compatibility_args(args)


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = official.run(
        args,
        load_runtime_model=_load_runtime_model,
        run_model_generation=_run_model_generation,
        prepare_prediction=prepare_model_output_for_official_scorer,
        prediction_format=PREDICTION_FORMAT,
        audio_prefix_tokens=0,
        protocol_description=(
            "official MMAR order and fixed choices; native Qwen2-Audio ChatML; "
            "decoded prediction passed verbatim; single cuda:0; bf16; "
            "manual greedy argmax full recompute; use_cache=False"
        ),
        stage="mmar_qwen2_audio_instruct_dual_scoring",
        record_static_fields={"qwen_native_single_audio_input": True},
    )
    report["qwen2_audio_contract"] = {
        "model_contract": qwen.MODEL_CONTRACT,
        "chat_contract": qwen.CHAT_CONTRACT,
        "dtype": "bfloat16",
        "system_prompt": None,
        "prediction_preparse": False,
    }
    inference_failures = int(
        report.get("records", {}).get("skip_reasons", {}).get("sample_exception", 0)
    )
    if inference_failures:
        report["status"] = "FAILED"
        report["comparable_official_score"] = False
        report["fatal_error"] = {
            "error": f"{inference_failures} Qwen2-Audio generation failures were recorded",
            "detail": "Inspect skipped.jsonl; do not interpret either score as valid.",
        }
    predictions_path = args.output_dir / "predictions_official.json"
    if (
        predictions_path.is_file()
        and report.get("inference_coverage", {}).get("status") == "PASS"
    ):
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        prediction_key = str(
            report.get("protocol", {}).get("prediction_key", "answer_prediction")
        )
        prefix_score = official.write_choice_label_prefix_evaluation(
            args.output_dir,
            predictions,
            output_key=prediction_key,
        )
        report["choice_label_prefix_evaluation"] = prefix_score
        report["dual_scoring"] = {
            "choice_label_prefix": prefix_score.get("status"),
            "official_mmar": report.get("official_evaluation", {}).get("status"),
            "prediction_text_shared_without_preparse": True,
        }
    official.common._write_json(args.output_dir / "evaluation_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(
        json.dumps(
            {
                "stage": report.get("stage"),
                "status": report.get("status"),
                "mode": report.get("mode"),
                "records": report.get("records", {}),
                "choice_label_prefix_evaluation": report.get(
                    "choice_label_prefix_evaluation", {}
                ),
                "official_evaluation": report.get("official_evaluation", {}),
                "report": str(args.output_dir / "evaluation_report.json"),
            },
            ensure_ascii=False,
            default=official.common._json_default,
        )
    )
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
