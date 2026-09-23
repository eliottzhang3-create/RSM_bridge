#!/usr/bin/env python3
"""Evaluate the completed 10-epoch shared-store Audio MeSH checkpoint on MMAU.

The canonical evaluator owns dataset traversal, fixed-order prompts, resumable
outputs, official-artifact audits, and official scoring.  This adapter owns the
shared-store checkpoint contract, fixed-260 runtime, and the deliberate rule
that decoded model text is passed to the official scorer verbatim.
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

import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as official  # noqa: E402


DEFAULT_CHECKPOINT = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "audio_5_10x2_5_mesh_mellow_shared_store_configurable_epochs/"
    "formal_10epochs_20260923/checkpoint-037810"
)
DEFAULT_DATASET_DIR = official.DEFAULT_DATASET_DIR
DEFAULT_HTSAT = official.DEFAULT_HTSAT
DEFAULT_MELLOW = official.DEFAULT_MELLOW
DEFAULT_MAX_PROMPT_TOKENS = official.DEFAULT_MAX_PROMPT_TOKENS
DEFAULT_MAX_NEW_TOKENS = 300
DEFAULT_MAX_CONTEXT_LENGTH = official.DEFAULT_MAX_CONTEXT_LENGTH
CONFIG_FILENAME = "audio_mesh_config.json"
CONTRACT = "node_shared_unique_store_fullshuffle_fixed260_audio_reuse_answer_eos_v2"
EXPECTED_FINAL_STEP = 37_810
EXPECTED_EPOCHS = 10
EXPECTED_STEPS_PER_EPOCH = 3_781
EXPECTED_WARMUP_STEPS = 1_891
EXPECTED_PREFIX_TOKENS = {"single": 260, "dual": 260}
ANSWER_TERMINATION = {
    "token": "<|endoftext|>",
    "included_in_max_answer_tokens": True,
    "supervised": True,
}
SINGLE_AUDIO_SLOT_SEMANTICS = (
    "fixed260_second_slot_reuses_audio1_htsat_embedding_then_runs_bridge_separately"
)
PREDICTION_FORMAT = "mellow_author_reply_raw_generation_choice_label_scoring_v1"
MMAU_PROTOCOL_CONTRACT = "shared_store_mmau_github_issue5_author_reply_reproduction_v1"

RowSkip = official.RowSkip
ProgressStore = official.ProgressStore
build_fixed_order_prompt = official.build_fixed_order_prompt
decode_and_normalize_audio = official.decode_and_normalize_audio
row_key = official.row_key
_json_default = official._json_default
_sha256 = official._sha256
_tokenize_without_truncation = official._tokenize_without_truncation


def prepare_model_output_for_official_scorer(value: Any) -> str:
    """Return decoded text unchanged; do not remove leading ``a)``-``d)``."""

    return str(value)


def _load_training_state_metadata(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _audit_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Fail closed unless the checkpoint is the completed fixed-260 10-epoch run."""
    import torch

    from audio_5_10x2_5_mesh_mellow_shared_store import TRAINING_CONTRACT
    from audio_5_10x2_5_mesh_mellow_shared_store.model import (
        ARCHITECTURE_CONTRACT,
        AUDIO_DUAL_PREFIX_TOKENS,
        AUDIO_PREFIX_TOKENS,
        AUDIO_TOKENS_PER_CLIP,
        MAPPER_CONTRACT,
        MESH_HIDDEN_SIZE,
    )
    from train_audio_smollm2_135m_mellow_ddp import _text_model_weight_files

    if TRAINING_CONTRACT != CONTRACT:
        raise RuntimeError(
            f"shared-store package contract changed: {TRAINING_CONTRACT!r} != {CONTRACT!r}"
        )
    checkpoint = args.checkpoint
    required = [
        "mesh_model/config.json",
        "tokenizer/tokenizer_config.json",
        "audio_bridge.pt",
        "training_state.pt",
        CONFIG_FILENAME,
        "checkpoint_complete.json",
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"shared-store checkpoint missing required files: {missing}")
    empty = [name for name in required if (checkpoint / name).stat().st_size <= 0]
    if empty:
        raise RuntimeError(f"shared-store checkpoint contains empty required files: {empty}")
    weights = _text_model_weight_files(checkpoint / "mesh_model")
    if not weights:
        raise RuntimeError("shared-store checkpoint has no MeSH model weights")

    config = json.loads((checkpoint / CONFIG_FILENAME).read_text(encoding="utf-8"))
    marker = json.loads((checkpoint / "checkpoint_complete.json").read_text(encoding="utf-8"))
    expected = {
        "contract": CONTRACT,
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "mapper_initialization": "random_c2l_and_xavier_projection",
        "compact_single_audio_prefix": False,
        "single_audio_slot_semantics": SINGLE_AUDIO_SLOT_SEMANTICS,
        "answer_termination": ANSWER_TERMINATION,
        "prefix_tokens": EXPECTED_PREFIX_TOKENS,
        "mesh_hidden_size": MESH_HIDDEN_SIZE,
        "audio_tokens_per_clip": AUDIO_TOKENS_PER_CLIP,
        "audio_prefix_tokens_with_separators": AUDIO_PREFIX_TOKENS,
        "mode": "formal",
        "epochs": EXPECTED_EPOCHS,
        "world_size": 8,
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "num_workers": 0,
        "max_lr": 1e-3,
        "min_lr": 1e-4,
        "warmup_steps": EXPECTED_WARMUP_STEPS,
        "total_steps": EXPECTED_FINAL_STEP,
        "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    expected_marker_required = [
        "mesh_model",
        "tokenizer",
        "audio_bridge.pt",
        "training_state.pt",
        CONFIG_FILENAME,
    ]
    if (
        marker.get("status") != "complete"
        or marker.get("contract") != CONTRACT
        or int(marker.get("global_step", -1)) != EXPECTED_FINAL_STEP
    ):
        mismatches["completion_marker"] = {
            "expected": {"status": "complete", "contract": CONTRACT, "global_step": EXPECTED_FINAL_STEP},
            "actual": marker,
        }
    if marker.get("required") not in (None, expected_marker_required):
        mismatches["required_files"] = {
            "expected": expected_marker_required,
            "actual": marker.get("required"),
        }
    suffix = checkpoint.name.removeprefix("checkpoint-")
    directory_step = int(suffix) if checkpoint.name.startswith("checkpoint-") and suffix.isdigit() else -1
    if directory_step != EXPECTED_FINAL_STEP:
        mismatches["checkpoint_directory_step"] = {
            "expected": EXPECTED_FINAL_STEP,
            "actual": directory_step,
        }
    expected_cursor = {
        "epoch": EXPECTED_EPOCHS,
        "batch_in_epoch": 0,
        "global_step": EXPECTED_FINAL_STEP,
    }
    for key, expected_value in expected_cursor.items():
        if int(config.get(key, -1)) != expected_value:
            mismatches[f"config_{key}"] = {
                "expected": expected_value,
                "actual": config.get(key),
            }
    if AUDIO_TOKENS_PER_CLIP != 129 or AUDIO_PREFIX_TOKENS != 260 or AUDIO_DUAL_PREFIX_TOKENS != 260:
        mismatches["runtime_audio_constants"] = {
            "expected": {"per_clip": 129, "fixed_prefix": 260},
            "actual": {
                "per_clip": AUDIO_TOKENS_PER_CLIP,
                "prefix": AUDIO_PREFIX_TOKENS,
                "dual_prefix": AUDIO_DUAL_PREFIX_TOKENS,
            },
        }
    for key, requested in (
        ("htsat_checkpoint", args.htsat_checkpoint),
        ("mellow_root", args.mellow_root),
    ):
        saved = str(config.get(key, ""))
        if not saved or Path(saved).resolve() != requested.resolve():
            mismatches[key] = {"expected": str(requested.resolve()), "actual": saved}
    if mismatches:
        raise RuntimeError(f"shared-store fixed260 checkpoint contract mismatch: {mismatches}")

    state = _load_training_state_metadata(checkpoint / "training_state.pt")
    required_state = {"optimizer", "scheduler", "global_step", "cursor", "rng_states_by_rank"}
    missing_state = sorted(required_state.difference(state))
    if missing_state:
        raise RuntimeError(f"shared-store training state missing keys: {missing_state}")
    cursor = state.get("cursor")
    if not isinstance(cursor, Mapping) or {
        key: int(cursor.get(key, -1)) for key in expected_cursor
    } != expected_cursor:
        raise RuntimeError(f"shared-store final cursor mismatch: {cursor!r}")
    if int(state.get("global_step", -1)) != EXPECTED_FINAL_STEP:
        raise RuntimeError("shared-store training state is not at optimizer step 37,810")
    rng_ranks = {str(key) for key in state.get("rng_states_by_rank", {})}
    if rng_ranks != {str(index) for index in range(8)}:
        raise RuntimeError(f"shared-store RNG rank coverage mismatch: {sorted(rng_ranks)}")
    del state
    try:
        audio_state = torch.load(
            checkpoint / "audio_bridge.pt", map_location="cpu", weights_only=True
        )
    except TypeError:
        audio_state = torch.load(checkpoint / "audio_bridge.pt", map_location="cpu")
    if not isinstance(audio_state, Mapping) or set(audio_state) != {"bridge", "c2l"}:
        raise RuntimeError("audio_bridge.pt must contain exactly non-empty bridge and c2l states")
    if not audio_state["bridge"] or not audio_state["c2l"]:
        raise RuntimeError("audio_bridge.pt contains an empty bridge or c2l state")

    return {
        "status": "PASS",
        "artifact_kind": "shared_store_fixed260_formal_checkpoint",
        "path": str(checkpoint),
        "config_path": str(checkpoint / CONFIG_FILENAME),
        "config_sha256": _sha256(checkpoint / CONFIG_FILENAME),
        "global_step": EXPECTED_FINAL_STEP,
        "epochs": EXPECTED_EPOCHS,
        "training_contract": CONTRACT,
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "required_files": required,
        "text_model_weight_files": [str(path) for path in weights],
        "compact_single_audio_prefix": False,
        "prefix_tokens": EXPECTED_PREFIX_TOKENS,
        "single_audio_slot_semantics": SINGLE_AUDIO_SLOT_SEMANTICS,
    }


def _load_runtime_model(
    args: argparse.Namespace,
    *,
    audit_checkpoint: Any | None = None,
    route_label: str = "shared-store fixed260",
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"{route_label} evaluation requires one CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    from train_audio_5_10x2_5_mesh_mellow_ddp import _load_model

    checkpoint_audit = (audit_checkpoint or _audit_checkpoint)(args)
    config = json.loads((args.checkpoint / CONFIG_FILENAME).read_text(encoding="utf-8"))
    load_args = argparse.Namespace(
        resume_from=args.checkpoint,
        init_from_audio_checkpoint=None,
        model_path=args.checkpoint / "mesh_model",
        tokenizer_path=None,
        htsat_checkpoint=args.htsat_checkpoint,
        mellow_root=args.mellow_root,
        compact_single_audio_prefix=False,
    )
    model, tokenizer = _load_model(load_args, device)
    saved_provenance = config.get("mellow_provenance") or {}
    loaded_provenance = getattr(model, "_audio_provenance", {})
    for key in ("module", "mellow_htsat_source", "mellow_htsat_sha256"):
        if saved_provenance.get(key) != loaded_provenance.get(key):
            raise RuntimeError(
                f"Mellow provenance mismatch for {key}: "
                f"saved={saved_provenance.get(key)!r} loaded={loaded_provenance.get(key)!r}"
            )
    model.eval()
    if bool(model.config_audio.compact_single_audio_prefix):
        raise RuntimeError(f"{route_label} evaluator must use a fixed 260-token two-slot prefix")
    actual_context_length = int(getattr(model.config_audio, "max_context_length", 0))
    if actual_context_length != DEFAULT_MAX_CONTEXT_LENGTH:
        raise RuntimeError(
            f"inference context mismatch: expected={DEFAULT_MAX_CONTEXT_LENGTH} actual={actual_context_length}"
        )
    owner = model.mesh_model.model
    owner.audit_mode = False
    owner.gradient_audit_mode = False
    owner.routing_stats_mode = False
    modes = {
        "composite": bool(model.training),
        "mesh": bool(model.mesh_model.training),
        "bridge": bool(model.bridge.training),
        "wrapper": bool(model.htsat_wrapper.training),
        "htsat": bool(model.htsat_backbone.training),
        "c2l": bool(model.htsat_wrapper.c2l.training),
    }
    if any(modes.values()):
        raise RuntimeError(f"inference requires every module in eval mode: {modes}")
    parameter_devices = {
        str(parameter.device) for parameter in model.parameters() if parameter.requires_grad
    }
    if parameter_devices != {str(device)}:
        raise RuntimeError(
            f"{route_label} trainable parameters are not colocated on cuda:0: "
            f"{sorted(parameter_devices)}"
        )
    config = dict(config)
    config["checkpoint_artifact_audit"] = checkpoint_audit
    config["runtime_max_context_length"] = actual_context_length
    return model, tokenizer, device, config


def _build_fixed260_reused_audio1_prefix(
    model: Any,
    waveform: Any,
    device: Any,
    *,
    autocast_enabled: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Build both slots from one audio1 HTSAT embedding, then bridge twice."""
    import torch

    from audio_5_10x2_5_mesh_mellow_shared_store.model import (
        AUDIO_TOKENS_PER_CLIP,
        MESH_HIDDEN_SIZE,
    )
    from audio_5_10x2_5_mesh_mellow.model import _find_embedding

    audio1 = waveform.unsqueeze(0).to(device, non_blocking=True)
    reused_mask = torch.ones((1,), dtype=torch.bool, device=device)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        first, second = model.encode_audio(
            audio1,
            None,
            reused_mask,
            skip_second_prefix=False,
        )
        separator_ids = torch.full(
            (1, 1), int(model.separator_token_id), dtype=torch.long, device=device
        )
        separator = _find_embedding(model.mesh_model, separator_ids)
        prefix = torch.cat((first, separator, second, separator), dim=1)
    expected_slot = (AUDIO_TOKENS_PER_CLIP, MESH_HIDDEN_SIZE)
    if tuple(first.shape[1:]) != expected_slot or tuple(second.shape[1:]) != expected_slot:
        raise RuntimeError(
            f"shared-store reused-audio slot shape mismatch: first={tuple(first.shape)} "
            f"second={tuple(second.shape)}"
        )
    if tuple(prefix.shape[1:]) != (260, MESH_HIDDEN_SIZE):
        raise RuntimeError(f"shared-store fixed260 prefix shape mismatch: {tuple(prefix.shape)}")
    if not bool(torch.isfinite(prefix).all()):
        raise RuntimeError("shared-store fixed260 prefix contains non-finite values")
    # eval() disables bridge dropout, so the two separate bridge calls over
    # the same reused HTSAT embedding must agree exactly at inference time.
    if not bool(torch.equal(first, second)):
        raise RuntimeError(
            "shared-store single-audio slots differ in eval mode; audio1 embedding "
            "was not reused consistently across two bridge calls"
        )
    return prefix, {
        "audio1_prefix_shape": list(first.shape),
        "audio2_prefix_shape": list(second.shape),
        "combined_prefix_shape": list(prefix.shape),
        "prefix_token_count": 260,
        "prefix_layout": "audio1_bridge_pass1 + separator1 + reused_audio1_bridge_pass2 + separator2",
        "separator_token_id": int(model.separator_token_id),
        "separator_token": model.tokenizer.decode(
            [int(model.separator_token_id)], skip_special_tokens=False
        ),
        "checkpoint_compact_single_audio_prefix": False,
        "single_audio_slot": True,
        "compact_single_audio_prefix_used": False,
        "audio2_prefix_materialized": True,
        "audio2_reused": True,
        "htsat_audio1_embedding_reused_for_slot2": True,
        "bridge_invocations_for_reused_embedding": 2,
        "bridge_dropout_active": False,
    }


def _run_model_generation(
    model: Any,
    tokenizer: Any,
    device: Any,
    sample: Mapping[str, Any],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run the pre-existing shared-store decoder used by non-MMAU adapters."""
    import torch
    from generate_audio_checkpoint_reasonaqa import _greedy_decode

    prompt_ids_cpu, prompt_token_count = _tokenize_without_truncation(
        tokenizer,
        str(sample["prompt"]),
        max_prompt_tokens=max_prompt_tokens,
    )
    prompt_ids = prompt_ids_cpu.to(device)
    with torch.inference_mode():
        prefix, prefix_audit = _build_fixed260_reused_audio1_prefix(
            model,
            sample["waveform"],
            device,
            autocast_enabled=True,
        )
        generation = _greedy_decode(
            model,
            tokenizer,
            prefix,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            autocast_enabled=True,
        )
    generation["prompt_token_count"] = prompt_token_count
    generation.update(prefix_audit)
    _validate_fixed260_generation(generation)
    return generation


def _validate_fixed260_generation(generation: Mapping[str, Any]) -> None:
    if (
        int(generation.get("prefix_token_count", -1)) != 260
        or generation.get("combined_prefix_shape", [None, None])[1] != 260
        or generation.get("compact_single_audio_prefix_used") is not False
        or generation.get("audio2_prefix_materialized") is not True
        or generation.get("audio2_reused") is not True
        or generation.get("htsat_audio1_embedding_reused_for_slot2") is not True
        or int(generation.get("bridge_invocations_for_reused_embedding", -1)) != 2
    ):
        raise RuntimeError(f"shared-store fixed260 inference prefix audit failed: {generation}")


def _run_mmau_author_reply_generation(
    model: Any,
    tokenizer: Any,
    device: Any,
    sample: Mapping[str, Any],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run only MMAU with the protocol from Mellow issue #5's author reply."""
    import torch
    from generate_audio_checkpoint_reasonaqa import _greedy_decode

    prompt_ids_cpu, prompt_original_token_count, prompt_truncated = (
        official.tokenize_mellow_author_reply_prompt(
            tokenizer,
            str(sample["prompt"]),
            max_prompt_tokens=max_prompt_tokens,
        )
    )
    prompt_ids = prompt_ids_cpu.to(device)
    with torch.inference_mode():
        waveform, audio_segment = official.mellow_author_reply_audio_segment(
            sample["waveform"]
        )
        prefix, prefix_audit = _build_fixed260_reused_audio1_prefix(
            model,
            waveform,
            device,
            autocast_enabled=False,
        )
        generation = _greedy_decode(
            model,
            tokenizer,
            prefix,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            autocast_enabled=False,
            top_p=0.8,
            temperature=1.0,
        )
    # MellowWrapper decodes with special tokens present and then splits on its
    # stop-token string; do not reuse the generic RSmol skip-special-tokens
    # rendering for the author-protocol score.
    generated_text_raw = tokenizer.decode(
        generation["generated_token_ids"],
        skip_special_tokens=False,
    )
    generation["generated_text_raw"] = generated_text_raw
    generation["generated_text"] = generated_text_raw.split(
        tokenizer.eos_token or "<|endoftext|>"
    )[0]
    generation["decode_policy"] = "mellow_wrapper_decode_then_split_stop_token"
    generation["prompt_token_count"] = int(prompt_ids.shape[1])
    generation["prompt_original_token_count"] = prompt_original_token_count
    generation["prompt_truncated"] = prompt_truncated
    generation["audio1_segment"] = audio_segment
    generation["audio2_segment"] = {
        **audio_segment,
        "policy": "reuse_audio1_segment_and_htsat_embedding",
    }
    generation.update(prefix_audit)
    _validate_fixed260_generation(generation)
    return generation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not any(item == "--mode" or item.startswith("--mode=") for item in raw):
        raw = ["--mode", "full", *raw]
    return official.parse_args(
        raw,
        default_checkpoint=DEFAULT_CHECKPOINT,
        default_max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        add_audio_root=True,
        description=__doc__,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = official.run(
        args,
        load_runtime_model=_load_runtime_model,
        run_model_generation=_run_mmau_author_reply_generation,
        prepare_prediction=prepare_model_output_for_official_scorer,
        prediction_format=PREDICTION_FORMAT,
        prompt_builder=official.build_mellow_author_reply_prompt,
        audio_decoder=official.decode_mellow_author_reply_audio,
        audio_root=args.audio_root,
        prefer_official_audio_file=True,
        prompt_format=official.MELLOW_AUTHOR_REPLY_PROMPT_FORMAT,
        audio_format=(
            official.MELLOW_AUTHOR_REPLY_AUDIO_FORMAT
            + "__single_segment_reused_to_match_shared_store_training_contract"
        ),
        protocol_contract=MMAU_PROTOCOL_CONTRACT,
        generation_protocol={
            "decoder": "mellow_wrapper_top_p_filter_then_argmax_full_recompute",
            "top_p": 0.8,
            "temperature": 1.0,
            "do_sample": False,
            "use_cache": False,
            "inference_dtype": "float32",
            "author_reply_difference": (
                "audio1 segment and HTSAT embedding reuse preserve this checkpoint's training contract"
            ),
        },
        audio_prefix_tokens=260,
        stage="mmau_test_mini_audio_5_10x2_5_mesh_mellow_shared_store_fixed260",
        logical_trace="exact MeSH 5-10-10-5 trace verified by shared greedy decoder",
    )
    inference_failures = int(
        report.get("records", {}).get("skip_reasons", {}).get("sample_exception", 0)
    )
    if inference_failures:
        report["status"] = "FAILED"
        report["comparable_official_score"] = False
        report["fatal_error"] = {
            "error": f"{inference_failures} shared-store generation failures were recorded as skipped rows",
            "detail": "Inspect skipped.jsonl; do not interpret official accuracy as a valid model score.",
        }
        official._write_json(args.output_dir / "evaluation_report.json", report)
    predictions_path = args.output_dir / "predictions_fixed_order.json"
    if predictions_path.is_file() and report.get("inference_coverage", {}).get("status") == "PASS":
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        author_score = official.write_mellow_author_reply_evaluation(args.output_dir, predictions)
        payload_sources = report.get("records", {}).get("audio", {}).get("payload_sources", {})
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
            **author_score["total"],
        }
        report["mmau_v051525_evaluation"] = report.get("official_evaluation", {})
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
    print(json.dumps({
        "stage": report.get("stage"),
        "status": report.get("status"),
        "mode": report.get("mode"),
        "records": report.get("records", {}),
        "primary_comparison_score": report.get("primary_comparison_score", {}),
        "official_evaluation": report.get("official_evaluation", {}),
        "report": str(args.output_dir / "evaluation_report.json"),
    }, ensure_ascii=False, default=_json_default))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
