#!/usr/bin/env python3
"""Evaluate a current two-stage recursive MeSH checkpoint on MMAU test-mini."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_ROOT = SCRIPT_DIR.parents[2]
RSMOL_ROOT = ROUTE_ROOT.parent
BASE_SCRIPTS = RSMOL_ROOT / "scripts"
for import_root in (SCRIPT_DIR, BASE_SCRIPTS, ROUTE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as official  # noqa: E402
import evaluate_mellow_official_training_common as common  # noqa: E402


ROUTE_CONTRACT = "mellow_v0_official_adamw_cosine_two_stage_5_10x2_5_mesh_v1"
TEXT_CONTRACT = "logical_30_physical_20_5_10x2_5"
PROTOCOL_CONTRACT = "mellow_v0_two_stage_mmau_author_reply_eos_pad_recursive_mesh_v1"
PREDICTION_FORMAT = "mellow_v0_two_stage_author_reply_raw_choice_label_scoring_eos_pad_v1"
DEFAULT_CHECKPOINT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "mellow_v0_two_stage_reasonaqa/stage1_formal_5epochs_epoch17_routers/checkpoints/"
    "mellow_v0_two_stage_stage1_formal_20_20261007_045627285353623/"
    "model--epo-5.ckpt"
)
DEFAULT_RUNTIME = DEFAULT_CHECKPOINT.parents[2] / "runtime_stage1_formal.yaml"
DEFAULT_OUTPUT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "mellow_v0_two_stage_reasonaqa/eval_stage1_formal_epoch5_mmau_test_mini"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--checkpoint-file", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--route-root", type=Path, default=ROUTE_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=Path(official.DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--evaluation-script", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--parquet-batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=common.PROMPT_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--dtype", choices=("fp32",), default="fp32")
    parser.add_argument("--run-official-evaluation", action="store_true")
    args = parser.parse_args(argv)
    args.checkpoint_file = args.checkpoint_file.expanduser().resolve(strict=True)
    args.runtime_config = args.runtime_config.expanduser().resolve(strict=True)
    args.route_root = args.route_root.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.parquet = (args.parquet or args.dataset_dir / "test_mini.parquet").resolve()
    args.metadata_json = (args.metadata_json or args.dataset_dir / "mmau-test-mini.json").resolve()
    args.evaluation_script = (args.evaluation_script or args.dataset_dir / "evaluation.py").resolve()
    args.audio_root = (args.audio_root or args.dataset_dir / "test-mini-audios").resolve()
    args.training_checkpoint = args.checkpoint_file
    args.checkpoint = args.checkpoint_file.parent
    args.mellow_root = args.route_root
    args.htsat_checkpoint = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT")
    args.training_branch = "mellow_official_training_mellow_v0_two_stage_adamw_cosine_5_10x2_5_mesh"
    if args.max_prompt_tokens != common.PROMPT_TOKENS:
        parser.error(f"--max-prompt-tokens is fixed at {common.PROMPT_TOKENS}")
    if args.max_new_tokens != 300:
        parser.error("--max-new-tokens is fixed at 300 for MMAU")
    if args.parquet_batch_size <= 0:
        parser.error("--parquet-batch-size must be positive")
    return args


def _validate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import yaml

    checkpoint = torch.load(args.checkpoint_file, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise RuntimeError("checkpoint must be a schema-v2 full training checkpoint")
    required = {
        "schema_version", "route_contract", "text_model_contract",
        "training_stage", "epoch_completed", "num_epochs", "state_dict",
    }
    missing = sorted(required.difference(checkpoint))
    if checkpoint.get("schema_version") != 2 or missing:
        raise RuntimeError(f"invalid checkpoint schema or fields: missing={missing}")
    if checkpoint["route_contract"] != ROUTE_CONTRACT:
        raise RuntimeError("checkpoint belongs to another training route")
    if checkpoint["text_model_contract"] != TEXT_CONTRACT:
        raise RuntimeError("checkpoint belongs to another text model contract")
    if (
        checkpoint["training_stage"] != "stage1"
        or int(checkpoint["epoch_completed"]) != 5
        or int(checkpoint["num_epochs"]) != 5
    ):
        raise RuntimeError("MMAU target must be a completed Stage 1 five-epoch checkpoint")
    config = yaml.safe_load(args.runtime_config.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise RuntimeError("runtime config must be a mapping")
    if config.get("route_contract") != ROUTE_CONTRACT:
        raise RuntimeError("runtime config route contract mismatch")
    if config.get("text_model_contract") != TEXT_CONTRACT:
        raise RuntimeError("runtime config text contract mismatch")
    if config.get("training_stage") != "stage1":
        raise RuntimeError("runtime config is not Stage 1")
    if int((config.get("train") or {}).get("num_epochs", -1)) != 5:
        raise RuntimeError("runtime config does not declare num_epochs=5")
    state = checkpoint["state_dict"]
    nonfinite = [
        key for key, value in state.items()
        if hasattr(value, "is_floating_point")
        and value.is_floating_point()
        and not torch.isfinite(value).all().item()
    ]
    if nonfinite:
        raise RuntimeError(f"checkpoint contains non-finite tensors: {nonfinite[:8]}")
    return {
        "schema_version": checkpoint["schema_version"],
        "epoch_completed": checkpoint["epoch_completed"],
        "num_epochs": checkpoint["num_epochs"],
        "state_tensor_count": len(state),
        "route_contract": checkpoint["route_contract"],
        "text_model_contract": checkpoint["text_model_contract"],
    }


def _raw_generation_by_id(output_dir: Path) -> dict[str, str]:
    path = output_dir / "raw_generations.jsonl"
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("status") == "generated":
            result[str(item.get("id", ""))] = str(item.get("generated_text", ""))
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_contract = _validate_checkpoint(args)
    loaded_runtime: dict[str, Any] = {}

    def load_model(current_args: argparse.Namespace):
        result = common.load_runtime_model(current_args)
        loaded_runtime.clear()
        loaded_runtime.update(result[3])
        return result

    def run_generation(
        model: Any,
        tokenizer: Any,
        device: Any,
        sample: Any,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
    ):
        return common.run_generation(
            model,
            tokenizer,
            device,
            sample,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            common=official,
        )

    report = official.run(
        args,
        load_runtime_model=load_model,
        run_model_generation=run_generation,
        prepare_prediction=official.prepare_model_output_for_official_scorer,
        prediction_format=PREDICTION_FORMAT,
        prompt_builder=official.build_mellow_author_reply_prompt,
        audio_decoder=official.decode_mellow_author_reply_audio,
        audio_root=args.audio_root,
        prefer_official_audio_file=True,
        prompt_format=official.MELLOW_AUTHOR_REPLY_PROMPT_FORMAT,
        audio_format=official.MELLOW_AUTHOR_REPLY_AUDIO_FORMAT,
        protocol_contract=PROTOCOL_CONTRACT,
        generation_protocol={
            "decoder": "recursive_mesh_top_p_filter_then_argmax_full_recompute",
            "top_p": 0.8,
            "temperature": 1.0,
            "do_sample": False,
            "use_cache": False,
            "inference_dtype": "float32",
            "prompt_contract": "training_eos_then_right_pad_bang_v1",
            "audio2_policy": "same_waveform_native_dual_encoding",
        },
        audio_prefix_tokens=common.AUDIO_PREFIX_TOKENS,
        stage="mmau_test_mini_mellow_v0_two_stage_recursive_mesh",
        logical_trace=(
            "Mellow-v0 two-stage recursive 5-10x2-5 MeSH; "
            "same 10-second waveform in two native audio slots"
        ),
        record_static_fields={
            "model_family": "mellow_v0_two_stage_recursive_mesh",
            "training_branch": args.training_branch,
            "checkpoint_file": str(args.checkpoint_file),
            "checkpoint_contract": checkpoint_contract,
            "standalone_smollm2_checkpoint": False,
            "legacy_mellow_v0_checkpoint": False,
        },
    )
    common.attach_model_identity(report, args, loaded_runtime)
    report["checkpoint_contract"] = checkpoint_contract
    inference_failures = int(
        report.get("records", {}).get("skip_reasons", {}).get("sample_exception", 0)
    )
    if inference_failures:
        report["status"] = "FAILED"
        report["comparable_official_score"] = False
        report["fatal_error"] = {
            "error": f"{inference_failures} recursive Mellow generation failures",
            "detail": "Inspect skipped.jsonl; neither MMAU score is valid for comparison.",
        }
    predictions_path = args.output_dir / "predictions_fixed_order.json"
    if (
        predictions_path.is_file()
        and report.get("inference_coverage", {}).get("status") == "PASS"
    ):
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        raw_by_id = _raw_generation_by_id(args.output_dir)
        author_predictions = []
        for prediction in predictions:
            item = copy.deepcopy(prediction)
            item["model_output"] = raw_by_id.get(str(item.get("id", "")), "")
            author_predictions.append(item)
        author_score = official.write_mellow_author_reply_evaluation(
            args.output_dir, author_predictions
        )
        payload_sources = report.get("records", {}).get("audio", {}).get("payload_sources", {})
        fallback_rows = sum(
            int(count) for source, count in payload_sources.items() if source != "official_id_wav"
        )
        report["mellow_author_reply_evaluation"] = author_score
        report["mellow_author_reply_context"] = official.MELLOW_AUTHOR_REPLY_CONTEXT
        report["mmau_v051525_evaluation"] = report.get("official_evaluation", {})
        report["primary_comparison_score"] = {
            "scorer": official.MELLOW_AUTHOR_REPLY_SCORER,
            "comparable": bool(
                args.mode == "full"
                and inference_failures == 0
                and int(author_score["total"]["total"]) == official.EXPECTED_FULL_ROWS
                and fallback_rows == 0
            ),
            "record_errors_counted_incorrect": int(
                author_score.get("record_errors", {}).get("total", 0)
            ),
            **author_score["total"],
        }
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
    common.write_json(args.output_dir / "evaluation_report.json", report)
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
    status = report.get("status")
    coverage_ok = report.get("inference_coverage", {}).get("status") == "PASS"
    return 0 if status == "PASS" or (status == "INFERENCE_ONLY" and coverage_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
