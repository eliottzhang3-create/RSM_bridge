#!/usr/bin/env python3
"""Evaluate an official Mellow training checkpoint on MMAU test-mini."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
BASE_SCRIPTS = ROOT.parent / "scripts"
for import_root in (SCRIPT_DIR, BASE_SCRIPTS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as official  # noqa: E402
import evaluate_mellow_official_training_common as adapter  # noqa: E402
import mesh_eval_common as mesh  # noqa: E402

PROTOCOL_CONTRACT = "mellow_adamw_cosine_mesh_mmau_author_reply_eos_pad_native_dual_v1"
PREDICTION_FORMAT = "mellow_adamw_cosine_mesh_author_reply_raw_choice_label_scoring_eos_pad_v1"
DEFAULT_DATASET_DIR = Path(official.DEFAULT_DATASET_DIR)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full",), default="full")
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--route-root", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--evaluation-script", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parquet-batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=adapter.PROMPT_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--dtype", choices=("fp32",), default="fp32")
    parser.add_argument("--run-official-evaluation", action="store_true")
    args = parser.parse_args(argv)
    args = mesh.configure_args(
        args,
        parser=parser,
        output_name="mmau_test_mini_mesh_official_training_v1",
        dataset_dir=DEFAULT_DATASET_DIR,
    )
    if args.max_prompt_tokens != adapter.PROMPT_TOKENS:
        parser.error(f"--max-prompt-tokens is fixed at {adapter.PROMPT_TOKENS}")
    if args.max_new_tokens != 300:
        parser.error("--max-new-tokens is fixed at 300 for MMAU")
    if args.parquet_batch_size <= 0:
        parser.error("--parquet-batch-size must be positive")
    return args


def prepare_prediction(value: Any) -> str:
    return official.prepare_model_output_for_official_scorer(value)


def _raw_generation_by_id(output_dir: Path) -> dict[str, str]:
    path = output_dir / "raw_generations.jsonl"
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if str(item.get("status", "")) == "generated":
            result[str(item.get("id", ""))] = str(item.get("generated_text", ""))
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_contract = mesh.validate_checkpoint(args)
    loaded_runtime: dict[str, Any] = {}

    def load_model(current_args: argparse.Namespace):
        result = adapter.load_runtime_model(current_args)
        loaded_runtime.clear()
        loaded_runtime.update(result[3])
        return result

    def run_generation(model: Any, tokenizer: Any, device: Any, sample: Any, *, max_prompt_tokens: int, max_new_tokens: int):
        return adapter.run_generation(
            model, tokenizer, device, sample,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            common=official,
        )

    report = official.run(
        args,
        load_runtime_model=load_model,
        run_model_generation=run_generation,
        prepare_prediction=prepare_prediction,
        prediction_format=PREDICTION_FORMAT,
        prompt_builder=official.build_mellow_author_reply_prompt,
        audio_decoder=official.decode_mellow_author_reply_audio,
        audio_root=args.audio_root,
        prefer_official_audio_file=True,
        prompt_format=official.MELLOW_AUTHOR_REPLY_PROMPT_FORMAT,
        audio_format=official.MELLOW_AUTHOR_REPLY_AUDIO_FORMAT,
        protocol_contract=PROTOCOL_CONTRACT,
        generation_protocol={
            "decoder": "official_mellow_top_p_filter_then_argmax_full_recompute",
            "top_p": 0.8,
            "temperature": 1.0,
            "do_sample": False,
            "use_cache": False,
            "inference_dtype": "float32",
            "prompt_contract": "training_eos_then_right_pad_bang_v1",
            "audio2_policy": "same_waveform_native_dual_encoding",
        },
        audio_prefix_tokens=adapter.AUDIO_PREFIX_TOKENS,
        stage="mmau_test_mini_mellow_adamw_cosine_mesh_native_dual",
        logical_trace="5-10x2-5 MeSH Mellow; same 10-second waveform in two native audio slots; recursive 30-logical-layer text decoder",
        record_static_fields={
            "model_family": "mellow_adamw_cosine_5_10x2_5_mesh",
            "training_branch": args.training_branch,
            "checkpoint_file": str(args.training_checkpoint),
            "standalone_smollm2_checkpoint": False,
            "legacy_mellow_v0_checkpoint": False,
        },
    )
    adapter.attach_model_identity(report, args, loaded_runtime)
    report["checkpoint_contract"] = checkpoint_contract
    report.setdefault("model_identity", {})["model_family"] = "mellow_adamw_cosine_5_10x2_5_mesh"
    report["model_identity"]["training_branch"] = mesh.ROUTE_NAME
    inference_failures = int(report.get("records", {}).get("skip_reasons", {}).get("sample_exception", 0))
    if inference_failures:
        report["status"] = "FAILED"
        report["comparable_official_score"] = False
        report["fatal_error"] = {
            "error": f"{inference_failures} official Mellow generation failures",
            "detail": "Inspect skipped.jsonl; neither MMAU score is valid for comparison.",
        }
    predictions_path = args.output_dir / "predictions_fixed_order.json"
    if predictions_path.is_file() and report.get("inference_coverage", {}).get("status") == "PASS":
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        raw_by_id = _raw_generation_by_id(args.output_dir)
        author_predictions = []
        for prediction in predictions:
            item = copy.deepcopy(prediction)
            item["model_output"] = raw_by_id.get(str(item.get("id", "")), "")
            author_predictions.append(item)
        author_score = official.write_mellow_author_reply_evaluation(args.output_dir, author_predictions)
        payload_sources = report.get("records", {}).get("audio", {}).get("payload_sources", {})
        fallback_rows = sum(int(count) for source, count in payload_sources.items() if source != "official_id_wav")
        report["mellow_author_reply_evaluation"] = author_score
        report["mellow_author_reply_context"] = official.MELLOW_AUTHOR_REPLY_CONTEXT
        report["primary_comparison_score"] = {
            "scorer": official.MELLOW_AUTHOR_REPLY_SCORER,
            "comparable": bool(
                args.mode == "full"
                and inference_failures == 0
                and int(author_score["total"]["total"]) == official.EXPECTED_FULL_ROWS
                and fallback_rows == 0
            ),
            "record_errors_counted_incorrect": int(author_score.get("record_errors", {}).get("total", 0)),
            **author_score["total"],
        }
        report["mmau_v051525_evaluation"] = report.get("official_evaluation", {})
        report["dual_scoring"] = {
            "official_mmau": report.get("official_evaluation", {}).get("status"),
            "mellow_author_reply": author_score.get("status"),
            "prediction_text_shared_without_preparse": False,
            "official_scorer_prediction_source": "predictions_fixed_order.json:model_output_stripped_leading_label",
            "choice_label_prefix_prediction_source": "raw_generations.jsonl:generated_text",
        }
        report["mellow_author_reply_protocol_audit"] = {
            "official_id_wav_rows": int(payload_sources.get("official_id_wav", 0)),
            "fallback_audio_rows": fallback_rows,
            "payload_sources": payload_sources,
            "status": "PASS" if fallback_rows == 0 else "NONCOMPARABLE_FALLBACK",
        }
    report["checkpoint_file"] = str(args.training_checkpoint)
    adapter.write_json(args.output_dir / "evaluation_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps({
        "stage": report.get("stage"),
        "status": report.get("status"),
        "mode": report.get("mode"),
        "records": report.get("records", {}),
        "primary_comparison_score": report.get("primary_comparison_score", {}),
        "official_evaluation": report.get("official_evaluation", {}),
        "report": str(args.output_dir / "evaluation_report.json"),
    }, ensure_ascii=False, default=official._json_default))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
