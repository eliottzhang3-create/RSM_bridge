#!/usr/bin/env python3
"""Evaluate the original September 2026 fixed-260 Audio MeSH checkpoint on MMAU.

This adapter uses the current audited MMAU pipeline and fixed-260 audio1 reuse
runtime, but validates the checkpoint against the historical pre-contract
schema written by the trainer revision used for the 2026-09-10 formal run.
"""
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

import evaluate_mmau_test_mini_audio_5_10x2_5_mesh_mellow_shared_store as fixed260  # noqa: E402


DEFAULT_CHECKPOINT = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "audio_5_10x2_5_mesh_mellow/"
    "formal_restart_save500_20260910_105248/checkpoint-011343"
)
HISTORICAL_TRAINER_COMMIT = "6f516fffb7ac2c5a1c538ef92680571f85f211ea"
EXPECTED_FINAL_STEP = 11_343
EXPECTED_EPOCHS = 3
EXPECTED_STEPS_PER_EPOCH = 3_781
EXPECTED_FINAL_EPOCH_INDEX = 2
EXPECTED_FINAL_BATCH_IN_EPOCH = 15_124
EXPECTED_WARMUP_STEPS = 568
EXPECTED_PREFIX_TOKENS = {"single": 260, "dual": 260}


def _audit_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the exact old fixed-260 checkpoint without relabeling its schema."""
    import torch

    from audio_5_10x2_5_mesh_mellow.model import (
        ARCHITECTURE_CONTRACT,
        AUDIO_PREFIX_TOKENS,
        AUDIO_TOKENS_PER_CLIP,
        MAPPER_CONTRACT,
        MESH_HIDDEN_SIZE,
    )
    from train_audio_smollm2_135m_mellow_ddp import _text_model_weight_files

    checkpoint = args.checkpoint
    required = [
        "mesh_model/config.json",
        "tokenizer/tokenizer_config.json",
        "audio_bridge.pt",
        "training_state.pt",
        "audio_mesh_config.json",
        "checkpoint_complete.json",
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"legacy fixed260 checkpoint missing required files: {missing}")
    empty = [name for name in required if (checkpoint / name).stat().st_size <= 0]
    if empty:
        raise RuntimeError(f"legacy fixed260 checkpoint contains empty required files: {empty}")
    weights = _text_model_weight_files(checkpoint / "mesh_model")
    if not weights:
        raise RuntimeError("legacy fixed260 checkpoint has no MeSH model weights")

    config = json.loads((checkpoint / "audio_mesh_config.json").read_text(encoding="utf-8"))
    marker = json.loads((checkpoint / "checkpoint_complete.json").read_text(encoding="utf-8"))
    expected = {
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "mapper_initialization": "random_c2l_and_xavier_projection",
        "mesh_hidden_size": MESH_HIDDEN_SIZE,
        "audio_tokens_per_clip": AUDIO_TOKENS_PER_CLIP,
        "audio_prefix_tokens_with_separators": AUDIO_PREFIX_TOKENS,
        "epochs": EXPECTED_EPOCHS,
        "max_lr": 1e-3,
        "min_lr": 0.0,
        "warmup_steps": EXPECTED_WARMUP_STEPS,
        "total_steps": EXPECTED_FINAL_STEP,
        "global_step": EXPECTED_FINAL_STEP,
        "epoch": EXPECTED_FINAL_EPOCH_INDEX,
        "batch_in_epoch": EXPECTED_FINAL_BATCH_IN_EPOCH,
        "world_size": 8,
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "save_every": 500,
        "checkpoint_retention": 4,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    # These fields did not exist in the historical checkpoint schema. Their
    # absence is material: this adapter must not silently relabel a current
    # compact, answer-EOS, partition, or shared-store artifact as the old run.
    unexpected_modern_fields = sorted(
        key
        for key in (
            "contract",
            "compact_single_audio_prefix",
            "prefix_tokens",
            "answer_termination",
            "data_pipeline",
            "initialization",
        )
        if key in config
    )
    if unexpected_modern_fields:
        mismatches["historical_schema"] = {
            "expected": "pre-contract fixed260 schema",
            "unexpected_modern_fields": unexpected_modern_fields,
        }
    marker_required = [
        "mesh_model",
        "tokenizer",
        "audio_bridge.pt",
        "training_state.pt",
        "audio_mesh_config.json",
    ]
    if (
        marker.get("status") != "complete"
        or int(marker.get("global_step", -1)) != EXPECTED_FINAL_STEP
        or marker.get("required") not in (None, marker_required)
        or "contract" in marker
    ):
        mismatches["completion_marker"] = {
            "expected": {
                "status": "complete",
                "global_step": EXPECTED_FINAL_STEP,
                "required": marker_required,
                "contract": "absent in historical schema",
            },
            "actual": marker,
        }
    suffix = checkpoint.name.removeprefix("checkpoint-")
    directory_step = int(suffix) if checkpoint.name.startswith("checkpoint-") and suffix.isdigit() else -1
    if directory_step != EXPECTED_FINAL_STEP:
        mismatches["checkpoint_directory_step"] = {
            "expected": EXPECTED_FINAL_STEP,
            "actual": directory_step,
        }
    manifest_sha256 = str(config.get("manifest_sha256", ""))
    if len(manifest_sha256) != 64 or any(char not in "0123456789abcdef" for char in manifest_sha256.lower()):
        mismatches["manifest_sha256"] = {
            "expected": "64-character lowercase/uppercase hexadecimal SHA256",
            "actual": manifest_sha256,
        }
    if AUDIO_TOKENS_PER_CLIP != 129 or AUDIO_PREFIX_TOKENS != 260:
        mismatches["runtime_audio_constants"] = {
            "expected": {"per_clip": 129, "fixed_prefix": 260},
            "actual": {"per_clip": AUDIO_TOKENS_PER_CLIP, "fixed_prefix": AUDIO_PREFIX_TOKENS},
        }
    for key, requested in (
        ("htsat_checkpoint", args.htsat_checkpoint),
        ("mellow_root", args.mellow_root),
    ):
        saved = str(config.get(key, ""))
        if not saved or Path(saved).resolve() != requested.resolve():
            mismatches[key] = {"expected": str(requested.resolve()), "actual": saved}
    if mismatches:
        raise RuntimeError(f"legacy fixed260 checkpoint contract mismatch: {mismatches}")

    state = fixed260._load_training_state_metadata(checkpoint / "training_state.pt")
    required_state = {
        "optimizer",
        "scheduler",
        "global_step",
        "epoch",
        "batch_in_epoch",
        "rng_states_by_rank",
    }
    missing_state = sorted(required_state.difference(state))
    if missing_state:
        raise RuntimeError(f"legacy fixed260 training state missing keys: {missing_state}")
    expected_state = {
        "global_step": EXPECTED_FINAL_STEP,
        "epoch": EXPECTED_FINAL_EPOCH_INDEX,
        "batch_in_epoch": EXPECTED_FINAL_BATCH_IN_EPOCH,
    }
    state_mismatches = {
        key: {"expected": value, "actual": state.get(key)}
        for key, value in expected_state.items()
        if int(state.get(key, -1)) != value
    }
    if state_mismatches:
        raise RuntimeError(f"legacy fixed260 final training cursor mismatch: {state_mismatches}")
    rng_ranks = {str(key) for key in state.get("rng_states_by_rank", {})}
    if rng_ranks != {str(index) for index in range(8)}:
        raise RuntimeError(f"legacy fixed260 RNG rank coverage mismatch: {sorted(rng_ranks)}")
    del state
    try:
        audio_state = torch.load(
            checkpoint / "audio_bridge.pt", map_location="cpu", weights_only=True
        )
    except TypeError:
        audio_state = torch.load(checkpoint / "audio_bridge.pt", map_location="cpu")
    if not isinstance(audio_state, Mapping) or set(audio_state) != {"bridge", "c2l"}:
        raise RuntimeError("legacy audio_bridge.pt must contain exactly bridge and c2l states")
    if not audio_state["bridge"] or not audio_state["c2l"]:
        raise RuntimeError("legacy audio_bridge.pt contains an empty bridge or c2l state")

    return {
        "status": "PASS",
        "artifact_kind": "legacy_pre_contract_fixed260_audio_mesh_checkpoint",
        "path": str(checkpoint),
        "historical_trainer_commit": HISTORICAL_TRAINER_COMMIT,
        "config_path": str(checkpoint / "audio_mesh_config.json"),
        "config_sha256": fixed260._sha256(checkpoint / "audio_mesh_config.json"),
        "global_step": EXPECTED_FINAL_STEP,
        "epochs": EXPECTED_EPOCHS,
        "final_epoch_index": EXPECTED_FINAL_EPOCH_INDEX,
        "final_batch_in_epoch": EXPECTED_FINAL_BATCH_IN_EPOCH,
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "required_files": required,
        "text_model_weight_files": [str(path) for path in weights],
        "compact_single_audio_prefix": False,
        "prefix_tokens": EXPECTED_PREFIX_TOKENS,
        "single_audio_slot_semantics": (
            "reuse audio1 HTSAT embedding for slot2; invoke bridge separately for both slots"
        ),
        "answer_termination_contract": "historical pre-answer-EOS checkpoint",
    }


def _load_runtime_model(args: argparse.Namespace):
    return fixed260._load_runtime_model(
        args,
        audit_checkpoint=_audit_checkpoint,
        route_label="legacy pre-contract fixed260",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not any(item == "--mode" or item.startswith("--mode=") for item in raw):
        raw = ["--mode", "full", *raw]
    return fixed260.official.parse_args(
        raw,
        default_checkpoint=DEFAULT_CHECKPOINT,
        description=__doc__,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = fixed260.official.run(
        args,
        load_runtime_model=_load_runtime_model,
        run_model_generation=fixed260._run_model_generation,
        prepare_prediction=fixed260.prepare_model_output_for_official_scorer,
        prediction_format=fixed260.PREDICTION_FORMAT,
        audio_prefix_tokens=260,
        stage="mmau_test_mini_audio_5_10x2_5_mesh_mellow_legacy_fixed260",
        logical_trace="exact MeSH 5-10-10-5 trace verified by shared greedy decoder",
    )
    inference_failures = int(
        report.get("records", {}).get("skip_reasons", {}).get("sample_exception", 0)
    )
    if inference_failures:
        report["status"] = "FAILED"
        report["comparable_official_score"] = False
        report["fatal_error"] = {
            "error": f"{inference_failures} legacy fixed260 generation failures were recorded as skipped rows",
            "detail": "Inspect skipped.jsonl; do not interpret official accuracy as a valid model score.",
        }
        fixed260.official._write_json(args.output_dir / "evaluation_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps({
        "stage": report.get("stage"),
        "status": report.get("status"),
        "mode": report.get("mode"),
        "records": report.get("records", {}),
        "prompt_length_audit": report.get("prompt_length_audit", {}),
        "official_evaluation": report.get("official_evaluation", {}),
        "report": str(args.output_dir / "evaluation_report.json"),
    }, ensure_ascii=False, default=fixed260._json_default))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
