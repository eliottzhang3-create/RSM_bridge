#!/usr/bin/env python3
"""Evaluate released native Mellow-v0 on MMAR with matched dual scoring."""
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

import audit_mellow_v0_artifact as preflight  # noqa: E402
import evaluate_mmar_5_10x2_5_mesh_mellow as official  # noqa: E402
import evaluate_mmau_test_mini_mellow_v0 as mellow  # noqa: E402


DEFAULT_DATASET_DIR = Path(official.DEFAULT_DATASET_DIR)
DEFAULT_OUTPUT_DIR = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0/"
    "mmar_mellow_v0_dual_scoring_matched_smollm2_113430_v1"
)
MMAR_MAX_NEW_TOKENS = 32
MELLOW_MAX_PROMPT_TOKENS = mellow.MELLOW_PROMPT_TOKENS
MODEL_CONTRACT_FILENAME = "mellow_v0_mmar_model_contract.json"
PROTOCOL_CONTRACT = "mellow_v0_mmar_official_dual_scoring_matched_smollm2_113430_v1"
PROTOCOL_DESCRIPTION = (
    "official MMAR order and scorer; ReasonAQA lowercase fixed-order labels; "
    "first-10-second audio; native Mellow two-slot prefix with fixed 129-token "
    "prompt; decoded prediction passed verbatim; 32-token deterministic argmax; "
    "official MMAR plus choice-label-prefix scoring"
)
MMAR_COMPARISON_REFERENCE = {
    "model_route": "audio_smollm2_135m_mellow_shared_store_configurable_epochs",
    "checkpoint": (
        "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
        "audio_smollm2_135m_mellow_shared_store_configurable_epochs/"
        "formal_30epochs_20260923_v1/checkpoint-113430"
    ),
    "evaluation_output": (
        "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
        "audio_smollm2_135m_mellow_shared_store_configurable_epochs/"
        "formal_30epochs_20260923_v1/mmar_checkpoint_113430_dual_scoring_v1"
    ),
    "matched_components": [
        "official MMAR metadata order and complete denominator",
        "official MMAR prompt and fixed choice order",
        "32kHz audio with first-10-second long-audio policy",
        "32 generated tokens",
        "verbatim decoded prediction with no choice pre-parser",
        "official MMAR scorer with dynamically detected prediction key",
        "choice-label-prefix diagnostic score over the same predictions",
    ],
    "model_inherent_difference": (
        "released Mellow-v0 fixes the prompt portion of its multimodal prefix to "
        "129 tokens and runs the released model in FP32; the comparison SmolLM2 "
        "checkpoint permits 476 prompt tokens and was evaluated in BF16"
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--mellow-source-root", type=Path, default=preflight.DEFAULT_MELLOW_SOURCE_ROOT
    )
    parser.add_argument(
        "--mellow-snapshot", type=Path, default=preflight.DEFAULT_MELLOW_SNAPSHOT
    )
    parser.add_argument(
        "--mellow-checkpoint", type=Path, default=preflight.DEFAULT_MELLOW_CHECKPOINT
    )
    parser.add_argument("--base-smollm2", type=Path, default=preflight.DEFAULT_BASE_SMOLLM2)
    parser.add_argument("--preflight-report", type=Path, default=preflight.DEFAULT_REPORT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--evaluation-script", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-prompt-tokens", type=int, default=MELLOW_MAX_PROMPT_TOKENS
    )
    parser.add_argument("--max-new-tokens", type=int, default=MMAR_MAX_NEW_TOKENS)
    parser.add_argument("--dtype", choices=("fp32",), default="fp32")
    parser.add_argument("--run-official-evaluation", action="store_true")
    args = parser.parse_args(argv)
    args.metadata_json = args.metadata_json or args.dataset_dir / "MMAR-meta.json"
    args.audio_root = args.audio_root or args.dataset_dir / "mmar-audio"
    args.evaluation_script = (
        args.evaluation_script or args.dataset_dir / "code" / "evaluation.py"
    )
    if args.max_prompt_tokens != MELLOW_MAX_PROMPT_TOKENS:
        parser.error(
            f"--max-prompt-tokens is fixed at {MELLOW_MAX_PROMPT_TOKENS} "
            "by released Mellow-v0"
        )
    if args.max_new_tokens != MMAR_MAX_NEW_TOKENS:
        parser.error(f"--max-new-tokens is fixed at {MMAR_MAX_NEW_TOKENS}")
    args.checkpoint = args.mellow_snapshot
    args.htsat_checkpoint = args.mellow_checkpoint
    args.mellow_root = args.mellow_source_root
    return args


def _model_contract(
    args: argparse.Namespace,
    preflight_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_contract": PROTOCOL_CONTRACT,
        "artifact_contract": preflight.ARTIFACT_CONTRACT,
        "path_identity_policy": "known_shared_storage_alias_v1",
        "preflight_report": mellow._shared_storage_identity(args.preflight_report),
        "preflight_report_sha256": preflight._sha256(args.preflight_report.resolve()),
        "mellow_source_root": mellow._shared_storage_identity(args.mellow_source_root),
        "mellow_snapshot": mellow._shared_storage_identity(args.mellow_snapshot),
        "mellow_checkpoint": mellow._shared_storage_identity(args.mellow_checkpoint),
        "mellow_checkpoint_sha256": preflight_report["mellow_checkpoint_sha256"],
        "base_smollm2": mellow._shared_storage_identity(args.base_smollm2),
        "single_audio_policy": (
            "MMAR first-10-second waveform independently encoded in two native slots"
        ),
        "prompt_tokens": MELLOW_MAX_PROMPT_TOKENS,
        "audio_prefix_tokens": mellow.MELLOW_AUDIO_PREFIX_TOKENS,
        "total_prefix_tokens_before_generation": mellow.MELLOW_PREFIX_TOKENS,
        "generation": {
            "decoder": "mellow_top_p_filter_then_argmax_greedy_equivalent",
            "do_sample": False,
            "top_p": 0.8,
            "temperature": 1.0,
            "max_new_tokens": MMAR_MAX_NEW_TOKENS,
            "inference_dtype": "float32",
        },
        "scoring": {
            "official_mmar": True,
            "choice_label_prefix": True,
            "prediction_text_shared_without_preparse": True,
        },
        "comparison_reference": MMAR_COMPARISON_REFERENCE,
    }


def _ensure_output_contract(
    args: argparse.Namespace,
    preflight_report: Mapping[str, Any],
) -> None:
    official._ensure_output_dir(
        args,
        prediction_format=mellow.PREDICTION_FORMAT,
        protocol_description=PROTOCOL_DESCRIPTION,
    )
    path = args.output_dir / MODEL_CONTRACT_FILENAME
    requested = _model_contract(args, preflight_report)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != requested:
            raise RuntimeError(
                "existing MMAR output belongs to another native Mellow model/protocol: "
                f"existing={existing} requested={requested}"
            )
    else:
        mellow.official._write_json(path, requested)


def _load_runtime_model(args: argparse.Namespace):
    model, tokenizer, device, runtime = mellow._load_runtime_model(args)
    preflight_report = getattr(args, "_validated_preflight_report", None)
    if preflight_report is None:
        raise RuntimeError("validated native Mellow preflight report is unavailable")
    runtime["runtime_model_contract"] = _model_contract(args, preflight_report)
    return model, tokenizer, device, runtime


def run(args: argparse.Namespace) -> dict[str, Any]:
    preflight_report = mellow._load_and_validate_preflight(args)
    args._validated_preflight_report = preflight_report
    _ensure_output_contract(args, preflight_report)
    report = official.run(
        args,
        load_runtime_model=_load_runtime_model,
        run_model_generation=mellow._run_model_generation,
        prepare_prediction=mellow.prepare_model_output_for_official_scorer,
        prediction_format=mellow.PREDICTION_FORMAT,
        audio_prefix_tokens=mellow.MELLOW_AUDIO_PREFIX_TOKENS,
        protocol_description=PROTOCOL_DESCRIPTION,
        stage="mmar_native_mellow_v0_dual_scoring_matched_smollm2_113430",
    )
    report.setdefault("protocol", {})["logical_trace"] = (
        "released native Mellow-v0 30-layer SmolLM2; MMAR waveform encoded "
        "independently in two native audio slots"
    )
    report["protocol"].update({
        "decoder": "mellow_top_p_filter_then_argmax_greedy_equivalent",
        "top_p": 0.8,
        "temperature": 1.0,
        "do_sample": False,
        "use_cache": "language_model_default_exactly_as_released_wrapper",
        "inference_dtype": "float32",
        "native_prompt_tokens": MELLOW_MAX_PROMPT_TOKENS,
    })
    report["comparison_reference"] = MMAR_COMPARISON_REFERENCE
    report["native_mellow_protocol_audit"] = {
        "status": "PASS",
        "official_metadata_and_scorer_shared_with_comparison": True,
        "prediction_text_shared_between_both_scorers": True,
        "prediction_preparser": None,
        "max_new_tokens": MMAR_MAX_NEW_TOKENS,
        "fixed_native_prompt_tokens": MELLOW_MAX_PROMPT_TOKENS,
        "top_p_argmax_is_greedy_equivalent": True,
        "native_audio_encoder_invocations": 2,
    }
    inference_failures = int(
        report.get("records", {}).get("skip_reasons", {}).get("sample_exception", 0)
    )
    if inference_failures:
        report["status"] = "FAILED"
        report["comparable_official_score"] = False
        report["fatal_error"] = {
            "error": f"{inference_failures} native Mellow MMAR generation failures",
            "detail": (
                "Inspect skipped.jsonl; neither MMAR score is a valid comparison "
                "when native model inference failed."
            ),
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
    mellow.official._write_json(args.output_dir / "evaluation_report.json", report)
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
                "records": report.get("records"),
                "choice_label_prefix_evaluation": report.get(
                    "choice_label_prefix_evaluation"
                ),
                "official_evaluation": report.get("official_evaluation"),
                "report": str(args.output_dir / "evaluation_report.json"),
            },
            ensure_ascii=False,
            default=mellow.official._json_default,
        )
    )
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
