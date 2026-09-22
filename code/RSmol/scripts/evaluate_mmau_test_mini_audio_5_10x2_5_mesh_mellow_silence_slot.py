#!/usr/bin/env python3
"""Evaluate the fixed-260 runtime-silence MeSH checkpoint on MMAU.

The benchmark traversal, prompt construction, resumable JSONL state, official
artifact audit, and scorer are imported from the canonical MeSH evaluator.
This route-local adapter owns only the silence-slot checkpoint contract and
the always-two-slot audio prefix construction.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
    "audio_5_10x2_5_mesh_mellow_silence_slot/"
    "partition_formal_answer_eos_v2_10epochs_20260921/checkpoint-037810"
)
DEFAULT_DATASET_DIR = official.DEFAULT_DATASET_DIR
DEFAULT_HTSAT = official.DEFAULT_HTSAT
DEFAULT_MELLOW = official.DEFAULT_MELLOW
DEFAULT_MAX_PROMPT_TOKENS = official.DEFAULT_MAX_PROMPT_TOKENS
DEFAULT_MAX_NEW_TOKENS = official.DEFAULT_MAX_NEW_TOKENS
DEFAULT_MAX_CONTEXT_LENGTH = official.DEFAULT_MAX_CONTEXT_LENGTH
CONFIG_FILENAME = "audio_mesh_fixed260_silence_slot_config.json"
CONTRACT = "component_partitions6_rank_ram_fixed260_runtime_silence_second_slot_answer_eos_v2"
ARCHITECTURE_CONTRACT = (
    "logical_30_physical_20_5_10x2_5_mesh_audio_mellow_"
    "fixed260_runtime_silence_second_slot"
)
EXPECTED_FINAL_STEP = 37_810
EXPECTED_PREFIX_TOKENS = {"single": 260, "dual": 260}
ANSWER_TERMINATION = {
    "token": "<|endoftext|>",
    "included_in_max_answer_tokens": True,
    "supervised": True,
}
SILENCE_SLOT_CONTRACT = {
    "single_audio_second_slot": "one exact-zero waveform created on GPU per containing microbatch",
    "silence_persisted_or_loaded": False,
    "silence_h2d_transfer": False,
    "silence_shape": [1, 1, 320000],
    "explicit_identical_dual_audio": "reuse audio1 encoder embedding; retain second slot",
    "distinct_dual_audio": "encode real audio2",
    "fixed_prefix_tokens_per_row": 260,
}

# Keep one implementation of the benchmark pipeline and its dependency-light
# helpers.  No model/checkpoint code is imported from the compact route.
RowSkip = official.RowSkip
ProgressStore = official.ProgressStore
build_fixed_order_prompt = official.build_fixed_order_prompt
decode_and_normalize_audio = official.decode_and_normalize_audio
prepare_model_output_for_official_scorer = official.prepare_model_output_for_official_scorer
row_key = official.row_key
_json_default = official._json_default
_sha256 = official._sha256
_tokenize_without_truncation = official._tokenize_without_truncation


def _load_training_state_metadata(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _audit_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Fail closed unless this is the completed fixed-260 silence checkpoint."""
    import torch

    from audio_5_10x2_5_mesh_mellow_silence_slot.model import (
        AUDIO_PREFIX_TOKENS,
        AUDIO_TOKENS_PER_CLIP,
        FIXED_PREFIX_TOKEN_CONTRACT,
        MAPPER_CONTRACT,
        SILENCE_SLOT_ARCHITECTURE_CONTRACT,
    )
    from train_audio_smollm2_135m_mellow_ddp import _text_model_weight_files

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
        raise RuntimeError(f"silence-slot checkpoint missing required files: {missing}")
    weights = _text_model_weight_files(checkpoint / "mesh_model")
    if not weights:
        raise RuntimeError("silence-slot checkpoint has no mesh-model weights")
    config = json.loads((checkpoint / CONFIG_FILENAME).read_text(encoding="utf-8"))
    marker = json.loads((checkpoint / "checkpoint_complete.json").read_text(encoding="utf-8"))
    expected = {
        "contract": CONTRACT,
        "architecture_contract": SILENCE_SLOT_ARCHITECTURE_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "compact_single_audio_prefix": False,
        "prefix_tokens": FIXED_PREFIX_TOKEN_CONTRACT,
        "answer_termination": ANSWER_TERMINATION,
        "silence_slot": SILENCE_SLOT_CONTRACT,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if marker.get("status") != "complete" or marker.get("contract") != CONTRACT:
        mismatches["completion_marker"] = {"expected": "complete + route contract", "actual": marker}
    if marker.get("required") not in (None, required):
        mismatches["required_files"] = {"expected": required, "actual": marker.get("required")}
    if mismatches:
        raise RuntimeError(f"fixed260 silence-slot checkpoint contract mismatch: {mismatches}")
    if int(config.get("total_steps", -1)) != EXPECTED_FINAL_STEP:
        raise RuntimeError("evaluation requires the completed 37,810-step formal checkpoint")
    marker_step = int(marker.get("global_step", -1))
    suffix = checkpoint.name.removeprefix("checkpoint-")
    directory_step = int(suffix) if checkpoint.name.startswith("checkpoint-") and suffix.isdigit() else -1
    if marker_step != EXPECTED_FINAL_STEP or directory_step != EXPECTED_FINAL_STEP:
        raise RuntimeError(
            "evaluation requires checkpoint-037810: "
            f"directory_step={directory_step} marker_step={marker_step}"
        )
    for key, requested in (("htsat_checkpoint", args.htsat_checkpoint), ("mellow_root", args.mellow_root)):
        saved = str(config.get(key, ""))
        if not saved or Path(saved).resolve() != requested.resolve():
            raise RuntimeError(f"checkpoint {key} mismatch: saved={saved!r} requested={requested}")
    state = _load_training_state_metadata(checkpoint / "training_state.pt")
    cursor = state.get("cursor")
    if (
        state.get("training_contract") not in (None, CONTRACT)
        or int(state.get("global_step", -1)) != EXPECTED_FINAL_STEP
        or (cursor is not None and int(cursor.get("global_step", -1)) != EXPECTED_FINAL_STEP)
    ):
        raise RuntimeError("training state does not prove completed silence-slot formal training")
    del state
    try:
        audio_state = torch.load(checkpoint / "audio_bridge.pt", map_location="cpu", weights_only=True)
    except TypeError:
        audio_state = torch.load(checkpoint / "audio_bridge.pt", map_location="cpu")
    if not isinstance(audio_state, Mapping) or set(audio_state) != {"bridge", "c2l"}:
        raise RuntimeError("audio_bridge.pt must contain exactly bridge and c2l")
    del audio_state
    if AUDIO_TOKENS_PER_CLIP != 129 or AUDIO_PREFIX_TOKENS != 260:
        raise RuntimeError("runtime silence-slot audio token constants changed")
    return {
        "status": "PASS",
        "artifact_kind": "fixed260_runtime_silence_slot_partition_checkpoint",
        "path": str(checkpoint),
        "config_path": str(checkpoint / CONFIG_FILENAME),
        "config_sha256": _sha256(checkpoint / CONFIG_FILENAME),
        "global_step": marker_step,
        "training_contract": CONTRACT,
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "required_files": required,
        "text_model_weight_files": [str(path) for path in weights],
        "compact_single_audio_prefix": False,
        "prefix_tokens": EXPECTED_PREFIX_TOKENS,
        "runtime_silence_second_slot": True,
    }


def _load_runtime_model(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("fixed260 silence-slot MMAU evaluation requires one CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    from recursive_model_5_10x2_5_mesh import RecursiveLlamaForCausalLM, register_auto_class
    from transformers import AutoTokenizer
    from audio_5_10x2_5_mesh_mellow_silence_slot.model import (
        AudioMeshSilenceSlotConfig,
        AudioMeshSilenceSlotModel,
        _load_mellow_wrapper,
    )

    register_auto_class()
    checkpoint_audit = _audit_checkpoint(args)
    config = json.loads((args.checkpoint / CONFIG_FILENAME).read_text(encoding="utf-8"))
    mesh = RecursiveLlamaForCausalLM.from_pretrained(
        args.checkpoint / "mesh_model", local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint / "tokenizer", local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper, htsat, provenance = _load_mellow_wrapper(
        args.mellow_root, args.htsat_checkpoint, device
    )
    model = AudioMeshSilenceSlotModel(
        mesh.to(device), tokenizer, wrapper, htsat, config=AudioMeshSilenceSlotConfig()
    )
    if not hasattr(model, "mesh_model") or hasattr(model, "text_model"):
        raise RuntimeError(
            "silence-slot evaluator requires the MeSH backend attribute mesh_model "
            "and must not use the SmolLM2 text_model contract"
        )
    audio_state = torch.load(args.checkpoint / "audio_bridge.pt", map_location=device, weights_only=False)
    model.bridge.load_state_dict(audio_state["bridge"], strict=True)
    model.htsat_wrapper.c2l.load_state_dict(audio_state["c2l"], strict=True)
    model._audio_provenance = provenance
    saved_provenance = config.get("mellow_provenance") or {}
    for key in ("module", "mellow_htsat_source", "mellow_htsat_sha256"):
        if saved_provenance.get(key) != provenance.get(key):
            raise RuntimeError(f"Mellow provenance mismatch for {key}")
    # The bridge is constructed after the text model and external audio
    # modules, so explicitly move the composite once after restoring its
    # trainable state.  Without this, bridge weights remain on CPU while
    # HTSAT embeddings are on cuda:0 and every row fails at linear1.
    model = model.to(device)
    model.eval()
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
        raise RuntimeError(f"inference requires all silence-slot modules in eval mode: {modes}")
    if bool(model.config_audio.compact_single_audio_prefix):
        raise RuntimeError("silence-slot evaluator must never use compact single-audio prefixes")
    if int(model.config_audio.max_context_length) != DEFAULT_MAX_CONTEXT_LENGTH:
        raise RuntimeError("silence-slot inference context contract differs from 768")
    parameter_devices = {
        str(parameter.device)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if parameter_devices != {str(device)}:
        raise RuntimeError(
            "silence-slot trainable parameters are not colocated with inference device: "
            f"expected={device} actual={sorted(parameter_devices)}"
        )
    config = dict(config)
    config["checkpoint_artifact_audit"] = checkpoint_audit
    config["runtime_max_context_length"] = int(model.config_audio.max_context_length)
    return model, tokenizer, device, config


def _build_fixed260_silence_prefix(model: Any, waveform: Any, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    from audio_5_10x2_5_mesh_mellow_silence_slot.model import AUDIO_TOKENS_PER_CLIP, MESH_HIDDEN_SIZE
    from audio_5_10x2_5_mesh_mellow.model import _find_embedding

    audio1 = waveform.unsqueeze(0).to(device, non_blocking=True)
    silence_mask = torch.ones((1,), dtype=torch.bool, device=device)
    same_real_mask = torch.zeros((1,), dtype=torch.bool, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        first, second = model.encode_audio(audio1, None, silence_mask, same_real_mask)
        separator_ids = torch.full((1, 1), int(model.separator_token_id), dtype=torch.long, device=device)
        separator = _find_embedding(model.mesh_model, separator_ids)
        prefix = torch.cat((first, separator, second, separator), dim=1)
    if tuple(first.shape[1:]) != (AUDIO_TOKENS_PER_CLIP, MESH_HIDDEN_SIZE):
        raise RuntimeError(f"silence-slot audio1 prefix shape mismatch: {tuple(first.shape)}")
    if tuple(second.shape[1:]) != (AUDIO_TOKENS_PER_CLIP, MESH_HIDDEN_SIZE):
        raise RuntimeError(f"silence-slot runtime-zero prefix shape mismatch: {tuple(second.shape)}")
    if tuple(prefix.shape[1:]) != (260, MESH_HIDDEN_SIZE) or not bool(torch.isfinite(prefix).all()):
        raise RuntimeError(f"fixed260 silence-slot prefix invariant failed: {tuple(prefix.shape)}")
    return prefix, {
        "audio1_prefix_shape": list(first.shape),
        "audio2_prefix_shape": list(second.shape),
        "combined_prefix_shape": list(prefix.shape),
        "prefix_token_count": 260,
        "prefix_layout": "audio1 + separator1 + runtime_zero_wave + separator2",
        "separator_token_id": int(model.separator_token_id),
        "separator_token": model.tokenizer.decode([int(model.separator_token_id)], skip_special_tokens=False),
        "audio2_reused": False,
        "single_audio_slot": True,
        "compact_single_audio_prefix_used": False,
        "audio2_prefix_materialized": True,
        "runtime_silence_second_slot": True,
        "runtime_silence_shape": list(model.last_audio_slot_audit.get("runtime_silence_shape") or []),
        "audio_slot_audit": dict(model.last_audio_slot_audit),
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
    import torch
    # This is the MeSH decoder: it reads model.mesh_model and verifies the
    # 5-10-10-5 trace.  The similarly named SmolLM2 helper reads text_model
    # and is intentionally incompatible with this route.
    from generate_audio_checkpoint_reasonaqa import _greedy_decode

    prompt_ids_cpu, prompt_token_count = _tokenize_without_truncation(
        tokenizer, str(sample["prompt"]), max_prompt_tokens=max_prompt_tokens
    )
    prompt_ids = prompt_ids_cpu.to(device)
    with torch.inference_mode():
        prefix, prefix_audit = _build_fixed260_silence_prefix(model, sample["waveform"], device)
        generated = _greedy_decode(
            model,
            tokenizer,
            prefix,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            autocast_enabled=True,
        )
    generated["prompt_token_count"] = prompt_token_count
    generated.update(prefix_audit)
    return generated


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not any(item == "--mode" or item.startswith("--mode=") for item in raw):
        raw = ["--mode", "full", *raw]
    return official.parse_args(raw, default_checkpoint=DEFAULT_CHECKPOINT, description=__doc__)


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = official.run(
        args,
        load_runtime_model=_load_runtime_model,
        run_model_generation=_run_model_generation,
        stage="mmau_test_mini_audio_5_10x2_5_mesh_mellow_silence_slot_fixed260",
        logical_trace="exact MeSH 5-10-10-5 trace verified by shared greedy decoder",
    )
    inference_failures = int(report.get("records", {}).get("skip_reasons", {}).get("sample_exception", 0))
    if inference_failures:
        report["status"] = "FAILED"
        report["fatal_error"] = {
            "error": f"{inference_failures} silence-slot generation failures were recorded as skipped rows",
            "detail": "Inspect skipped.jsonl; do not interpret official accuracy as a valid model score.",
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
        "official_evaluation": report.get("official_evaluation", {}),
        "report": str(args.output_dir / "evaluation_report.json"),
    }, ensure_ascii=False, default=_json_default))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
