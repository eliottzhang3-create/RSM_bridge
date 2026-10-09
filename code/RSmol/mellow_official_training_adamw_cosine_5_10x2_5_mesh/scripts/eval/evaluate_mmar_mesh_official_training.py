#!/usr/bin/env python3
"""Evaluate an official Mellow training checkpoint on MMAR."""
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

import evaluate_mmar_5_10x2_5_mesh_mellow as official  # noqa: E402
import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as mmau_common  # noqa: E402
import evaluate_mellow_official_training_common as adapter  # noqa: E402
import mesh_eval_common as mesh  # noqa: E402

PROTOCOL_CONTRACT = "mellow_adamw_cosine_mesh_mmar_dual_scoring_eos_pad_native_dual_v1"
PREDICTION_FORMAT = "mellow_adamw_cosine_mesh_mmar_raw_choice_label_scoring_eos_pad_v1"
MMAR_MAX_NEW_TOKENS = 32
DEFAULT_DATASET_DIR = Path(official.DEFAULT_DATASET_DIR)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full",), default="full")
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--route-root", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--evaluation-script", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-prompt-tokens", type=int, default=adapter.PROMPT_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=MMAR_MAX_NEW_TOKENS)
    parser.add_argument("--dtype", choices=("fp32",), default="fp32")
    parser.add_argument("--run-official-evaluation", action="store_true")
    args = parser.parse_args(argv)
    args = mesh.configure_args(
        args,
        parser=parser,
        output_name="mmar_mesh_official_training_v1",
        dataset_dir=DEFAULT_DATASET_DIR,
        metadata_name="MMAR-meta.json",
        audio_subdir="mmar-audio",
        evaluation_relative="code/evaluation.py",
    )
    if args.max_prompt_tokens != adapter.PROMPT_TOKENS:
        parser.error(f"--max-prompt-tokens is fixed at {adapter.PROMPT_TOKENS}")
    if args.max_new_tokens != MMAR_MAX_NEW_TOKENS:
        parser.error(f"--max-new-tokens is fixed at {MMAR_MAX_NEW_TOKENS} for MMAR")
    return args


def prepare_prediction(value: Any) -> str:
    return mmau_common.prepare_model_output_for_official_scorer(value)


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


def _mmar_native_segment(waveform: Any):
    import torch

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if tuple(waveform.shape) != (1, adapter.SAMPLE_RATE * adapter.AUDIO_SECONDS):
        raise RuntimeError(
            "MMAR wrapper waveform must already be [1, 320000] after first-10-second/right-pad normalization: "
            f"got {tuple(waveform.shape)}"
        )
    if not bool(torch.isfinite(waveform).all()):
        raise RuntimeError("MMAR normalized waveform contains non-finite values")
    return waveform.contiguous(), {
        "source_samples": int(waveform.shape[-1]),
        "target_samples": int(waveform.shape[-1]),
        "policy": "mmar_wrapper_first10s_right_zero_pad",
        "repeat_factor": 1,
        "crop_start": 0,
        "crop_end": int(waveform.shape[-1]),
    }


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
            common=mmau_common,
            segmenter=_mmar_native_segment,
        )

    report = official.run(
        args,
        load_runtime_model=load_model,
        run_model_generation=run_generation,
        prepare_prediction=prepare_prediction,
        prediction_format=PREDICTION_FORMAT,
        audio_prefix_tokens=adapter.AUDIO_PREFIX_TOKENS,
        protocol_description=(
            "official MMAR order; ReasonAQA lowercase fixed-order labels; first 10 seconds; "
            "native official Mellow two-slot same-waveform prefix; 129-token EOS+bang prompt; "
            "32-token FP32 deterministic argmax; official MMAR plus choice-label-prefix scoring"
        ),
        stage="mmar_mellow_adamw_cosine_mesh_native_dual_dual_scoring",
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
    report.setdefault("protocol", {})["prediction_key"] = report.get("protocol", {}).get(
        "prediction_key", "detected_from_official_evaluation.py"
    )
    report["protocol"]["max_new_tokens"] = MMAR_MAX_NEW_TOKENS
    inference_failures = int(report.get("records", {}).get("skip_reasons", {}).get("sample_exception", 0))
    if inference_failures:
        report["status"] = "FAILED"
        report["comparable_official_score"] = False
        report["fatal_error"] = {
            "error": f"{inference_failures} official Mellow MMAR generation failures",
            "detail": "Inspect skipped.jsonl; neither MMAR score is valid for comparison.",
        }
    predictions_path = args.output_dir / "predictions_official.json"
    if predictions_path.is_file() and report.get("inference_coverage", {}).get("status") == "PASS":
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        prediction_key = str(report.get("protocol", {}).get("prediction_key", "answer_prediction"))
        raw_by_id = _raw_generation_by_id(args.output_dir)
        prefix_predictions = []
        for prediction in predictions:
            item = copy.deepcopy(prediction)
            item[prediction_key] = raw_by_id.get(str(item.get("id", "")), "")
            prefix_predictions.append(item)
        prefix_score = official.write_choice_label_prefix_evaluation(
            args.output_dir, prefix_predictions, output_key=prediction_key
        )
        report["choice_label_prefix_evaluation"] = prefix_score
        report["dual_scoring"] = {
            "choice_label_prefix": prefix_score.get("status"),
            "official_mmar": report.get("official_evaluation", {}).get("status"),
            "prediction_text_shared_without_preparse": False,
            "official_scorer_prediction_source": "predictions_official.json:official_prediction_stripped_leading_label",
            "choice_label_prefix_prediction_source": "raw_generations.jsonl:generated_text",
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
        "choice_label_prefix_evaluation": report.get("choice_label_prefix_evaluation", {}),
        "official_evaluation": report.get("official_evaluation", {}),
        "report": str(args.output_dir / "evaluation_report.json"),
    }, ensure_ascii=False, default=official.common._json_default))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
