#!/usr/bin/env python3
"""Evaluate the partition-v2 audio SmolLM2-135M baseline on MMAU test-mini."""
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
    "audio_smollm2_135m_mellow/partition_formal_eos_v2_10epochs_20260918/"
    "checkpoint-037810"
)
DEFAULT_DATASET_DIR = official.DEFAULT_DATASET_DIR
DEFAULT_HTSAT = official.DEFAULT_HTSAT
DEFAULT_MELLOW = official.DEFAULT_MELLOW
DEFAULT_SAMPLE_RATE = official.DEFAULT_SAMPLE_RATE
DEFAULT_AUDIO_SECONDS = official.DEFAULT_AUDIO_SECONDS
DEFAULT_MAX_PROMPT_TOKENS = official.DEFAULT_MAX_PROMPT_TOKENS
DEFAULT_MAX_NEW_TOKENS = official.DEFAULT_MAX_NEW_TOKENS
DEFAULT_AUDIO_PREFIX_TOKENS = 130
DEFAULT_MAX_CONTEXT_LENGTH = official.DEFAULT_MAX_CONTEXT_LENGTH
PROMPT_FORMAT = official.PROMPT_FORMAT
PREDICTION_FORMAT = official.PREDICTION_FORMAT
PARTITION_CONFIG_FILENAME = "audio_smollm2_partition_config.json"
PARTITION_CONTRACT = "smollm2_component_partitions6_rank_ram_compact_audio_answer_eos_v2"
EXPECTED_FINAL_STEP = 37_810

# Re-export the dependency-light official data/scoring helpers. MMAU and MMAR
# intentionally share these exact contracts; only model loading and generation
# differ between MeSH and the standard SmolLM2 baseline.
RowSkip = official.RowSkip
ProgressStore = official.ProgressStore
build_fixed_order_prompt = official.build_fixed_order_prompt
decode_and_normalize_audio = official.decode_and_normalize_audio
row_key = official.row_key
_json_default = official._json_default
_write_json = official._write_json
_sha256 = official._sha256
_canonical_sha256 = official._canonical_sha256
_tokenize_without_truncation = official._tokenize_without_truncation
_run_official_evaluation = official._run_official_evaluation


def _audit_partition_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the completed compact-prefix partition-v2 baseline artifact."""

    import torch

    from audio_smollm2_135m_mellow.model import (
        AUDIO_DUAL_PREFIX_TOKENS,
        AUDIO_SINGLE_PREFIX_TOKENS,
        AUDIO_TOKENS_PER_CLIP,
        MAPPER_CONTRACT,
        ORIGINAL_SMOLLM2_CONTRACT,
        SMOLLM2_HIDDEN_SIZE,
    )
    from train_audio_smollm2_135m_mellow_ddp import _text_model_weight_files

    checkpoint = args.checkpoint
    config_path = checkpoint / PARTITION_CONFIG_FILENAME
    marker_path = checkpoint / "checkpoint_complete.json"
    state_path = checkpoint / "training_state.pt"
    required = [
        "text_model/config.json",
        "tokenizer/tokenizer_config.json",
        "audio_bridge.pt",
        "training_state.pt",
        PARTITION_CONFIG_FILENAME,
        "checkpoint_complete.json",
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"partition-v2 SmolLM2 checkpoint missing required files: {missing}")
    weight_files = _text_model_weight_files(checkpoint / "text_model")
    if not weight_files:
        raise RuntimeError("partition-v2 SmolLM2 checkpoint has no text-model weights")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete" or marker.get("contract") != PARTITION_CONTRACT:
        raise RuntimeError(f"invalid partition-v2 completion marker: {marker}")
    if marker.get("required") != required:
        raise RuntimeError("partition-v2 completion marker required-file contract differs")
    expected = {
        "contract": PARTITION_CONTRACT,
        "architecture_contract": ORIGINAL_SMOLLM2_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "compact_single_audio_prefix": True,
        "prefix_tokens": {
            "single": AUDIO_SINGLE_PREFIX_TOKENS,
            "dual": AUDIO_DUAL_PREFIX_TOKENS,
        },
        "answer_termination": {
            "token": "<|endoftext|>",
            "included_in_max_answer_tokens": True,
            "supervised": True,
        },
        "mode": "formal",
        "epochs": 10,
        "world_size": 8,
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "effective_global_batch_size": 256,
        "frozen_audio_encoder": True,
        "periodic_validation": False,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"partition-v2 SmolLM2 checkpoint contract mismatch: {mismatches}")
    standard = config.get("standard_text_contract") or {}
    expected_standard = {
        "model_type": "llama",
        "hidden_size": SMOLLM2_HIDDEN_SIZE,
        "num_hidden_layers": 30,
        "physical_decoder_layer_count": 30,
        "independent_decoder_layers": True,
        "forbidden_custom_parameter_names": [],
    }
    standard_mismatches = {
        key: {"expected": value, "actual": standard.get(key)}
        for key, value in expected_standard.items()
        if standard.get(key) != value
    }
    if standard_mismatches:
        raise RuntimeError(f"standard SmolLM2 architecture mismatch: {standard_mismatches}")
    if int(config.get("total_steps", -1)) != EXPECTED_FINAL_STEP:
        raise RuntimeError(
            f"formal partition schedule must contain {EXPECTED_FINAL_STEP} steps: "
            f"{config.get('total_steps')!r}"
        )
    marker_step = int(marker.get("global_step", -1))
    if marker_step != EXPECTED_FINAL_STEP:
        raise RuntimeError(
            f"evaluation requires the completed checkpoint-{EXPECTED_FINAL_STEP:06d}: "
            f"marker_step={marker_step}"
        )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    expected_cursor = {
        "segment": len(config.get("schedule", [])),
        "segment_step": 0,
        "global_step": EXPECTED_FINAL_STEP,
    }
    if (
        state.get("training_contract") != PARTITION_CONTRACT
        or int(state.get("global_step", -1)) != EXPECTED_FINAL_STEP
        or state.get("cursor") != expected_cursor
    ):
        raise RuntimeError("partition-v2 training state does not prove completed formal training")
    if not state.get("optimizer", {}).get("state") or not state.get("scheduler"):
        raise RuntimeError("partition-v2 checkpoint lacks optimizer/scheduler restoration evidence")
    del state
    if not isinstance(config.get("mellow_provenance"), Mapping):
        raise RuntimeError("partition-v2 checkpoint lacks Mellow provenance")
    for key, requested in (
        ("htsat_checkpoint", args.htsat_checkpoint),
        ("mellow_root", args.mellow_root),
    ):
        saved = str(config.get(key, ""))
        if not saved or Path(saved).resolve() != requested.resolve():
            raise RuntimeError(f"checkpoint {key} mismatch: saved={saved!r} requested={requested}")
    if AUDIO_TOKENS_PER_CLIP != 129 or AUDIO_SINGLE_PREFIX_TOKENS != DEFAULT_AUDIO_PREFIX_TOKENS:
        raise RuntimeError("runtime compact audio-prefix constants changed")
    return {
        "status": "PASS",
        "path": str(checkpoint),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "global_step": marker_step,
        "required_files": required,
        "text_model_weight_files": [str(path) for path in weight_files],
        "training_contract": PARTITION_CONTRACT,
        "compact_single_audio_prefix": True,
        "single_audio_prefix_tokens": AUDIO_SINGLE_PREFIX_TOKENS,
        "dual_audio_prefix_tokens": AUDIO_DUAL_PREFIX_TOKENS,
        "standard_text_contract": standard,
    }


def _load_runtime_model(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("MMAU/MMAR audio SmolLM2 inference requires one CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    from audio_smollm2_135m_mellow.model import SMOLLM2_HIDDEN_SIZE
    from train_audio_smollm2_135m_mellow_ddp import _load_model

    checkpoint_audit = _audit_partition_checkpoint(args)
    config = json.loads(
        (args.checkpoint / PARTITION_CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    load_args = argparse.Namespace(
        resume_from=args.checkpoint,
        model_path=args.checkpoint / "text_model",
        tokenizer_path=None,
        htsat_checkpoint=args.htsat_checkpoint,
        mellow_root=args.mellow_root,
        compact_single_audio_prefix=True,
    )
    model, tokenizer = _load_model(load_args, device)
    runtime_text_contract = dict(getattr(model, "text_contract", {}))
    if (
        runtime_text_contract.get("model_type") != "llama"
        or int(runtime_text_contract.get("hidden_size", -1)) != SMOLLM2_HIDDEN_SIZE
        or int(runtime_text_contract.get("num_hidden_layers", -1)) != 30
        or runtime_text_contract.get("physical_decoder_layer_count") != 30
        or runtime_text_contract.get("independent_decoder_layers") is not True
        or runtime_text_contract.get("forbidden_custom_parameter_names") != []
    ):
        raise RuntimeError(f"loaded model is not the standard SmolLM2 contract: {runtime_text_contract}")
    saved_provenance = config.get("mellow_provenance") or {}
    loaded_provenance = getattr(model, "_audio_provenance", {})
    for key in ("module", "mellow_htsat_source", "mellow_htsat_sha256"):
        if saved_provenance.get(key) != loaded_provenance.get(key):
            raise RuntimeError(
                f"Mellow provenance mismatch for {key}: "
                f"saved={saved_provenance.get(key)!r} loaded={loaded_provenance.get(key)!r}"
            )
    model.eval()
    if not bool(model.config_audio.compact_single_audio_prefix):
        raise RuntimeError("partition-v2 evaluation requires compact single-audio prefix")
    actual_context_length = int(getattr(model.config_audio, "max_context_length", 0))
    if actual_context_length != DEFAULT_MAX_CONTEXT_LENGTH:
        raise RuntimeError(
            f"inference context mismatch: expected={DEFAULT_MAX_CONTEXT_LENGTH} "
            f"actual={actual_context_length}"
        )
    modes = {
        "composite": bool(model.training),
        "text": bool(model.text_model.training),
        "bridge": bool(model.bridge.training),
        "wrapper": bool(model.htsat_wrapper.training),
        "htsat": bool(model.htsat_backbone.training),
        "c2l": bool(model.htsat_wrapper.c2l.training),
    }
    if any(modes.values()):
        raise RuntimeError(f"inference requires all modules in eval mode: {modes}")
    config["checkpoint_artifact_audit"] = checkpoint_audit
    config["runtime_max_context_length"] = actual_context_length
    config["runtime_text_contract"] = runtime_text_contract
    return model, tokenizer, device, config


def _build_compact_audio_prefix(
    model: Any,
    waveform: Any,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch

    from audio_smollm2_135m_mellow.model import AUDIO_TOKENS_PER_CLIP, SMOLLM2_HIDDEN_SIZE
    from generate_audio_smollm2_checkpoint_reasonaqa import _find_embedding

    audio = waveform.unsqueeze(0).to(device, non_blocking=True)
    reused_mask = torch.ones((1,), dtype=torch.bool, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        first, _ = model.encode_audio(
            audio,
            None,
            reused_mask,
            skip_second_prefix=True,
        )
        separator_ids = torch.full(
            (1, 1),
            int(model.separator_token_id),
            dtype=torch.long,
            device=device,
        )
        separator = _find_embedding(model.text_model, separator_ids)
        prefix = torch.cat((first, separator), dim=1)
    expected_first = (AUDIO_TOKENS_PER_CLIP, SMOLLM2_HIDDEN_SIZE)
    expected_prefix = (DEFAULT_AUDIO_PREFIX_TOKENS, SMOLLM2_HIDDEN_SIZE)
    if tuple(first.shape[1:]) != expected_first or tuple(prefix.shape[1:]) != expected_prefix:
        raise RuntimeError(
            f"compact prefix shape mismatch: first={tuple(first.shape)} combined={tuple(prefix.shape)}"
        )
    if not bool(torch.isfinite(prefix).all()):
        raise RuntimeError("compact audio prefix contains non-finite values")
    return prefix, {
        "audio1_prefix_shape": list(first.shape),
        "combined_prefix_shape": list(prefix.shape),
        "separator_token_id": int(model.separator_token_id),
        "separator_token": model.tokenizer.decode(
            [int(model.separator_token_id)],
            skip_special_tokens=False,
        ),
        "audio2_reused": True,
        "single_audio_slot": True,
        "compact_single_audio_prefix": True,
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
    from generate_audio_smollm2_checkpoint_reasonaqa import _greedy_decode

    prompt_ids_cpu, prompt_token_count = _tokenize_without_truncation(
        tokenizer,
        str(sample["prompt"]),
        max_prompt_tokens=max_prompt_tokens,
    )
    prompt_ids = prompt_ids_cpu.to(device)
    with __import__("torch").inference_mode():
        audio_prefix, prefix_audit = _build_compact_audio_prefix(
            model,
            sample["waveform"],
            device,
        )
        generated = _greedy_decode(
            model,
            tokenizer,
            audio_prefix,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            autocast_enabled=True,
        )
    generated["prompt_token_count"] = prompt_token_count
    generated.update(prefix_audit)
    return generated


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return official.parse_args(
        argv,
        default_checkpoint=DEFAULT_CHECKPOINT,
        description=__doc__,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    return official.run(
        args,
        load_runtime_model=_load_runtime_model,
        run_model_generation=_run_model_generation,
        stage="mmau_test_mini_audio_smollm2_fixed_order",
        logical_trace="standard 30-layer SmolLM2 per generation step",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps({
        "stage": report.get("stage"),
        "status": report.get("status"),
        "mode": report.get("mode"),
        "records": report.get("records", {}),
        "official_evaluation": report.get("official_evaluation", {}),
        "report": str(args.output_dir / "evaluation_report.json"),
    }, ensure_ascii=False, default=_json_default))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
