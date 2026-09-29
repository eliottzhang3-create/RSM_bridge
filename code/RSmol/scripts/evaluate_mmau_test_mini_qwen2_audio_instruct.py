#!/usr/bin/env python3
"""Evaluate local Qwen2-Audio-Instruct on MMAU test-mini with dual scoring."""
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

import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as official  # noqa: E402
import evaluate_qwen2_audio_instruct_common as qwen  # noqa: E402


DEFAULT_MAX_NEW_TOKENS = 300
PREDICTION_FORMAT = "qwen2_audio_instruct_raw_generation_dual_scoring_v1"
PROTOCOL_CONTRACT = "qwen2_audio_instruct_mmau_author_reply_and_v051525_v1"


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
    waveform, segment = official.mellow_author_reply_audio_segment(sample["waveform"])
    generation = qwen.generate_top_p_argmax(
        model,
        processor,
        device,
        str(sample["prompt"]),
        waveform,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        top_p=0.8,
        temperature=1.0,
    )
    generation["audio_segment"] = segment
    generation["audio_segment_policy"] = official.MELLOW_AUTHOR_REPLY_AUDIO_FORMAT
    return generation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not any(item == "--mode" or item.startswith("--mode=") for item in raw):
        raw = ["--mode", "smoke", *raw]
    args = official.parse_args(
        raw,
        default_checkpoint=qwen.DEFAULT_MODEL_PATH,
        default_max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        add_audio_root=True,
        description=__doc__,
    )
    if args.dtype != "bf16":
        raise ValueError("Qwen2-Audio MMAU evaluation is fixed to --dtype bf16")
    return qwen.apply_qwen_compatibility_args(args)


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = official.run(
        args,
        load_runtime_model=_load_runtime_model,
        run_model_generation=_run_model_generation,
        prepare_prediction=prepare_model_output_for_official_scorer,
        prediction_format=PREDICTION_FORMAT,
        prompt_builder=official.build_mellow_author_reply_prompt,
        audio_decoder=official.decode_mellow_author_reply_audio,
        audio_root=args.audio_root,
        prefer_official_audio_file=True,
        prompt_format=official.MELLOW_AUTHOR_REPLY_PROMPT_FORMAT,
        audio_format=official.MELLOW_AUTHOR_REPLY_AUDIO_FORMAT,
        protocol_contract=PROTOCOL_CONTRACT,
        generation_protocol={
            "model_family": "Qwen2-Audio-Instruct",
            "model_runtime": qwen.MODEL_CONTRACT,
            "chat_contract": qwen.CHAT_CONTRACT,
            "decoder": "qwen2_audio_top_p_filter_then_argmax_full_recompute",
            "top_p": 0.8,
            "temperature": 1.0,
            "do_sample": False,
            "use_cache": False,
            "inference_dtype": "bfloat16",
            "audio_tokenization": "Qwen2AudioProcessor dynamic audio tokens",
        },
        audio_prefix_tokens=0,
        stage="mmau_test_mini_qwen2_audio_instruct_dual_scoring",
        logical_trace="not applicable; native Qwen2-Audio transformer execution",
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
    predictions_path = args.output_dir / "predictions_fixed_order.json"
    if (
        predictions_path.is_file()
        and report.get("inference_coverage", {}).get("status") == "PASS"
    ):
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        author_score = official.write_mellow_author_reply_evaluation(
            args.output_dir, predictions
        )
        payload_sources = (
            report.get("records", {}).get("audio", {}).get("payload_sources", {})
        )
        fallback_audio_rows = sum(
            int(count)
            for source, count in payload_sources.items()
            if source != "official_id_wav"
        )
        report["mellow_author_reply_evaluation"] = author_score
        report["mellow_author_reply_context"] = official.MELLOW_AUTHOR_REPLY_CONTEXT
        report["primary_comparison_score"] = {
            "scorer": official.MELLOW_AUTHOR_REPLY_SCORER,
            "comparable": bool(
                args.mode == "full"
                and inference_failures == 0
                and int(author_score["total"]["total"]) == official.EXPECTED_FULL_ROWS
                and fallback_audio_rows == 0
            ),
            "record_errors_counted_incorrect": int(
                author_score.get("record_errors", {}).get("total", 0)
            ),
            **author_score["total"],
        }
        report["mmau_v051525_evaluation"] = report.get("official_evaluation", {})
        report["dual_scoring"] = {
            "choice_label_prefix": author_score.get("status"),
            "official_mmau_v051525": report.get("official_evaluation", {}).get("status"),
            "prediction_text_shared_without_preparse": True,
        }
        report["mellow_author_reply_protocol_audit"] = {
            "official_id_wav_rows": int(payload_sources.get("official_id_wav", 0)),
            "fallback_audio_rows": fallback_audio_rows,
            "payload_sources": payload_sources,
            "status": "PASS" if fallback_audio_rows == 0 else "NONCOMPARABLE_FALLBACK",
        }
    official._write_json(args.output_dir / "evaluation_report.json", report)
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
                "primary_comparison_score": report.get("primary_comparison_score", {}),
                "official_evaluation": report.get("official_evaluation", {}),
                "report": str(args.output_dir / "evaluation_report.json"),
            },
            ensure_ascii=False,
            default=official._json_default,
        )
    )
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
