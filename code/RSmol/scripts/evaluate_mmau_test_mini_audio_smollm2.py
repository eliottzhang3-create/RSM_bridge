#!/usr/bin/env python3
"""Evaluate supported audio SmolLM2-135M artifacts on MMAU test-mini."""
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
    "audio_smollm2_135m_mellow_shared_store_configurable_epochs/"
    "formal_30epochs_20260923_v1/checkpoint-113430"
)
DEFAULT_DATASET_DIR = official.DEFAULT_DATASET_DIR
DEFAULT_HTSAT = official.DEFAULT_HTSAT
DEFAULT_MELLOW = official.DEFAULT_MELLOW
DEFAULT_SAMPLE_RATE = official.DEFAULT_SAMPLE_RATE
DEFAULT_AUDIO_SECONDS = official.DEFAULT_AUDIO_SECONDS
DEFAULT_MAX_PROMPT_TOKENS = official.DEFAULT_MAX_PROMPT_TOKENS
DEFAULT_MAX_NEW_TOKENS = 300
DEFAULT_AUDIO_PREFIX_TOKENS = 130
SHARED_STORE_AUDIO_PREFIX_TOKENS = 260
DEFAULT_MAX_CONTEXT_LENGTH = official.DEFAULT_MAX_CONTEXT_LENGTH
PROMPT_FORMAT = official.PROMPT_FORMAT
PARTITION_CONFIG_FILENAME = "audio_smollm2_partition_config.json"
PARTITION_CONTRACT = "smollm2_component_partitions6_rank_ram_compact_audio_answer_eos_v2"
EXPECTED_FINAL_STEP = 37_810
EVAL_ONLY_CONFIG_FILENAME = "mellow_audio_smollm2_eval_config.json"
EVAL_ONLY_MARKER_FILENAME = "artifact_complete.json"
EVAL_ONLY_CONTRACT = "mellow_v0_to_audio_smollm2_compact_eval_v1"
SHARED_STORE_CONFIG_FILENAME = "audio_smollm2_shared_store_config.json"
SHARED_STORE_CONTRACT = (
    "smollm2_node_shared_unique_store_fullshuffle_fixed260_audio_reuse_answer_eos_v2"
)
SHARED_STORE_PREFIX_TOKENS = {"single": 260, "dual": 260}
PREDICTION_FORMAT = "mellow_author_reply_raw_generation_choice_label_scoring_v1"
MMAU_PROTOCOL_CONTRACT = "smollm2_shared_store_mmau_github_issue5_author_reply_reproduction_v1"

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


def _load_training_state_metadata(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


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
        "artifact_kind": "training_checkpoint",
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


def _audit_shared_store_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Audit only the portable completed-checkpoint invariants needed for eval."""

    import torch

    from audio_smollm2_135m_mellow.model import (
        MAPPER_CONTRACT,
        ORIGINAL_SMOLLM2_CONTRACT,
        SMOLLM2_HIDDEN_SIZE,
    )
    from audio_smollm2_135m_mellow_shared_store import TRAINING_CONTRACT
    from train_audio_smollm2_135m_mellow_ddp import _text_model_weight_files

    if TRAINING_CONTRACT != SHARED_STORE_CONTRACT:
        raise RuntimeError(
            f"shared-store package contract changed: {TRAINING_CONTRACT!r} "
            f"!= {SHARED_STORE_CONTRACT!r}"
        )
    checkpoint = args.checkpoint
    config_path = checkpoint / SHARED_STORE_CONFIG_FILENAME
    required_files = [
        "text_model/config.json",
        "tokenizer/tokenizer_config.json",
        "audio_bridge.pt",
        "training_state.pt",
        SHARED_STORE_CONFIG_FILENAME,
        "checkpoint_complete.json",
    ]
    missing = [name for name in required_files if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"SmolLM2 shared-store checkpoint is incomplete: {missing}")
    empty = [name for name in required_files if (checkpoint / name).stat().st_size <= 0]
    if empty:
        raise RuntimeError(f"SmolLM2 shared-store checkpoint contains empty files: {empty}")
    weights = _text_model_weight_files(checkpoint / "text_model")
    if not weights:
        raise RuntimeError("SmolLM2 shared-store checkpoint has no text-model weights")

    suffix = checkpoint.name.removeprefix("checkpoint-")
    directory_step = (
        int(suffix)
        if checkpoint.name.startswith("checkpoint-") and suffix.isdigit()
        else -1
    )
    if directory_step <= 0:
        raise RuntimeError(
            f"checkpoint directory must be checkpoint-<positive optimizer step>: {checkpoint.name}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    marker = json.loads(
        (checkpoint / "checkpoint_complete.json").read_text(encoding="utf-8")
    )
    expected_marker_required = [
        "text_model",
        "tokenizer",
        "audio_bridge.pt",
        "training_state.pt",
        SHARED_STORE_CONFIG_FILENAME,
    ]
    if (
        marker.get("status") != "complete"
        or marker.get("contract") != SHARED_STORE_CONTRACT
        or int(marker.get("global_step", -1)) != directory_step
        or marker.get("required") not in (None, expected_marker_required)
    ):
        raise RuntimeError(f"invalid SmolLM2 shared-store completion marker: {marker}")

    expected_config = {
        "contract": SHARED_STORE_CONTRACT,
        "architecture_contract": ORIGINAL_SMOLLM2_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "compact_single_audio_prefix": False,
        "prefix_tokens": SHARED_STORE_PREFIX_TOKENS,
        "text_hidden_size": SMOLLM2_HIDDEN_SIZE,
        "audio_tokens_per_clip": 129,
        "audio_prefix_tokens_with_separators": 260,
        "mode": "formal",
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
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
        key: {"expected": expected, "actual": standard.get(key)}
        for key, expected in expected_standard.items()
        if standard.get(key) != expected
    }
    if mismatches or standard_mismatches:
        raise RuntimeError(
            "SmolLM2 shared-store architecture/contract mismatch: "
            f"config={mismatches} standard={standard_mismatches}"
        )

    expected_cursor = {
        "epoch": int(config.get("epoch", -1)),
        "batch_in_epoch": int(config.get("batch_in_epoch", -1)),
        "global_step": int(config.get("global_step", -1)),
    }
    if (
        expected_cursor["epoch"] <= 0
        or expected_cursor["batch_in_epoch"] != 0
        or expected_cursor["global_step"] != directory_step
    ):
        raise RuntimeError(f"invalid completed SmolLM2 config cursor: {expected_cursor}")

    state = _load_training_state_metadata(checkpoint / "training_state.pt")
    cursor = state.get("cursor")
    if (
        state.get("training_contract") != SHARED_STORE_CONTRACT
        or int(state.get("global_step", -1)) != directory_step
        or not isinstance(cursor, Mapping)
        or {key: int(cursor.get(key, -1)) for key in expected_cursor} != expected_cursor
    ):
        raise RuntimeError(
            f"SmolLM2 shared-store training-state cursor mismatch: {cursor!r}"
        )
    rng_ranks = {str(key) for key in state.get("rng_states_by_rank", {})}
    if rng_ranks != {str(index) for index in range(8)}:
        raise RuntimeError(f"SmolLM2 shared-store RNG rank coverage mismatch: {sorted(rng_ranks)}")
    del state

    try:
        audio_state = torch.load(
            checkpoint / "audio_bridge.pt", map_location="cpu", weights_only=True
        )
    except TypeError:
        audio_state = torch.load(checkpoint / "audio_bridge.pt", map_location="cpu")
    if not isinstance(audio_state, Mapping) or set(audio_state) != {"bridge", "c2l"}:
        raise RuntimeError("audio_bridge.pt must contain exactly bridge and c2l states")
    if not audio_state["bridge"] or not audio_state["c2l"]:
        raise RuntimeError("audio_bridge.pt contains an empty bridge or c2l state")

    return {
        "status": "PASS",
        "artifact_kind": "shared_store_fixed260_training_checkpoint",
        "path": str(checkpoint),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "global_step": directory_step,
        "epochs": expected_cursor["epoch"],
        "training_contract": SHARED_STORE_CONTRACT,
        "architecture_contract": ORIGINAL_SMOLLM2_CONTRACT,
        "compact_single_audio_prefix": False,
        "single_audio_prefix_tokens": 260,
        "dual_audio_prefix_tokens": 260,
        "required_files": required_files,
        "text_model_weight_files": [str(path) for path in weights],
        "standard_text_contract": standard,
    }


def _artifact_config_path(checkpoint: Path) -> tuple[str, Path]:
    """Select exactly one supported AudioSmolLM2 artifact contract."""

    partition = checkpoint / PARTITION_CONFIG_FILENAME
    eval_only = checkpoint / EVAL_ONLY_CONFIG_FILENAME
    shared_store = checkpoint / SHARED_STORE_CONFIG_FILENAME
    present = [
        ("training_checkpoint", partition),
        ("eval_only_model", eval_only),
        ("shared_store_fixed260_training_checkpoint", shared_store),
    ]
    found = [(kind, path) for kind, path in present if path.is_file()]
    if len(found) != 1:
        raise RuntimeError(
            "AudioSmolLM2 evaluation requires exactly one supported artifact config; "
            f"found={[str(path) for _, path in found]}"
        )
    return found[0]


def _audit_eval_only_artifact(args: argparse.Namespace) -> dict[str, Any]:
    """Validate a CPU-converted Mellow-v0 compact evaluation artifact."""

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
    config_path = checkpoint / EVAL_ONLY_CONFIG_FILENAME
    marker_path = checkpoint / EVAL_ONLY_MARKER_FILENAME
    required = [
        "text_model/config.json",
        "tokenizer/tokenizer_config.json",
        "audio_bridge.pt",
        EVAL_ONLY_CONFIG_FILENAME,
        EVAL_ONLY_MARKER_FILENAME,
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"Mellow eval-only artifact is incomplete: {missing}")
    forbidden_training_files = [
        name for name in ("training_state.pt", PARTITION_CONFIG_FILENAME, "checkpoint_complete.json")
        if (checkpoint / name).exists()
    ]
    if forbidden_training_files:
        raise RuntimeError(
            "Mellow eval-only artifact must not be mixed with a training checkpoint: "
            f"{forbidden_training_files}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        config.get("artifact_contract") != EVAL_ONLY_CONTRACT
        or config.get("artifact_kind") != "eval_only_model"
    ):
        raise RuntimeError(f"invalid Mellow eval-only config contract: {config.get('artifact_contract')!r}")
    if (
        marker.get("status") != "complete"
        or marker.get("artifact_contract") != EVAL_ONLY_CONTRACT
        or marker.get("artifact_kind") != "eval_only_model"
        or marker.get("required") != required
        or marker.get("config_sha256") != _sha256(config_path)
        or marker.get("htsat_included") is not False
    ):
        raise RuntimeError(f"invalid Mellow eval-only completion marker: {marker}")

    expected = {
        "architecture_contract": ORIGINAL_SMOLLM2_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "compact_single_audio_prefix": True,
        "audio_tokens_per_clip": AUDIO_TOKENS_PER_CLIP,
        "prefix_tokens": {
            "single": AUDIO_SINGLE_PREFIX_TOKENS,
            "dual": AUDIO_DUAL_PREFIX_TOKENS,
        },
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "audio_seconds": DEFAULT_AUDIO_SECONDS,
        "max_prompt_tokens": DEFAULT_MAX_PROMPT_TOKENS,
        "max_context_length": DEFAULT_MAX_CONTEXT_LENGTH,
        "training_state_included": False,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Mellow eval-only artifact contract mismatch: {mismatches}")
    external_htsat = config.get("external_htsat") or {}
    if external_htsat.get("included") is not False or external_htsat.get("required_at_runtime") is not True:
        raise RuntimeError("Mellow eval-only artifact must require the evaluator's external HTSAT")

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
        raise RuntimeError(f"converted Mellow SmolLM2 architecture mismatch: {standard_mismatches}")

    text_config = json.loads((checkpoint / "text_model/config.json").read_text(encoding="utf-8"))
    if (
        text_config.get("model_type") != "llama"
        or int(text_config.get("hidden_size", -1)) != SMOLLM2_HIDDEN_SIZE
        or int(text_config.get("num_hidden_layers", -1)) != 30
    ):
        raise RuntimeError("converted text_model/config.json is not SmolLM2-135M")
    weight_files = _text_model_weight_files(checkpoint / "text_model")
    if not weight_files:
        raise RuntimeError("Mellow eval-only artifact has no text-model weights")
    if marker.get("text_model_weight_files") != [path.name for path in weight_files]:
        raise RuntimeError("Mellow eval-only marker text-model weight inventory differs")

    try:
        audio_state = torch.load(
            checkpoint / "audio_bridge.pt", map_location="cpu", weights_only=True
        )
    except TypeError:
        audio_state = torch.load(checkpoint / "audio_bridge.pt", map_location="cpu")
    if not isinstance(audio_state, Mapping) or set(audio_state) != {"bridge", "c2l"}:
        raise RuntimeError("Mellow eval-only audio_bridge.pt must contain exactly bridge and c2l")
    expected_shapes = {
        "bridge": {
            "linear1.weight": (576, 768),
            "linear2.weight": (576, 576),
            "norm.weight": (576,),
            "norm.bias": (576,),
        },
        "c2l": {
            "weight": (768, 527),
            "bias": (768,),
        },
    }
    for group_name, shapes in expected_shapes.items():
        group = audio_state.get(group_name)
        if not isinstance(group, Mapping) or set(group) != set(shapes):
            raise RuntimeError(f"Mellow eval-only {group_name} state keys differ")
        actual_shapes = {key: tuple(value.shape) for key, value in group.items()}
        if actual_shapes != shapes:
            raise RuntimeError(
                f"Mellow eval-only {group_name} tensor shapes differ: "
                f"expected={shapes} actual={actual_shapes}"
            )
    return {
        "status": "PASS",
        "artifact_kind": "eval_only_model",
        "artifact_contract": EVAL_ONLY_CONTRACT,
        "path": str(checkpoint),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "required_files": required,
        "text_model_weight_files": [str(path) for path in weight_files],
        "source_checkpoint": config.get("source_checkpoint"),
        "source_checkpoint_sha256": config.get("source_checkpoint_sha256"),
        "external_htsat_checkpoint": str(args.htsat_checkpoint.resolve()),
        "compact_single_audio_prefix": True,
        "single_audio_prefix_tokens": AUDIO_SINGLE_PREFIX_TOKENS,
        "dual_audio_prefix_tokens": AUDIO_DUAL_PREFIX_TOKENS,
        "standard_text_contract": standard,
    }


def _audit_checkpoint_artifact(args: argparse.Namespace) -> dict[str, Any]:
    kind, _ = _artifact_config_path(args.checkpoint)
    if kind == "eval_only_model":
        return _audit_eval_only_artifact(args)
    if kind == "shared_store_fixed260_training_checkpoint":
        return _audit_shared_store_checkpoint(args)
    return _audit_partition_checkpoint(args)


def _load_runtime_model(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("MMAU/MMAR audio SmolLM2 inference requires one CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    from audio_smollm2_135m_mellow.model import SMOLLM2_HIDDEN_SIZE
    from train_audio_smollm2_135m_mellow_ddp import _load_model

    checkpoint_audit = _audit_checkpoint_artifact(args)
    config = json.loads(Path(checkpoint_audit["config_path"]).read_text(encoding="utf-8"))
    load_args = argparse.Namespace(
        resume_from=args.checkpoint,
        model_path=args.checkpoint / "text_model",
        tokenizer_path=None,
        htsat_checkpoint=args.htsat_checkpoint,
        mellow_root=args.mellow_root,
        compact_single_audio_prefix=bool(
            checkpoint_audit.get("compact_single_audio_prefix", True)
        ),
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
    loaded_provenance = getattr(model, "_audio_provenance", {})
    if checkpoint_audit["artifact_kind"] == "training_checkpoint":
        saved_provenance = config.get("mellow_provenance") or {}
        for key in ("module", "mellow_htsat_source", "mellow_htsat_sha256"):
            if saved_provenance.get(key) != loaded_provenance.get(key):
                raise RuntimeError(
                    f"Mellow provenance mismatch for {key}: "
                    f"saved={saved_provenance.get(key)!r} loaded={loaded_provenance.get(key)!r}"
                )
    elif checkpoint_audit["artifact_kind"] == "eval_only_model":
        saved_separator = int((config.get("tokenizer") or {}).get("separator_token_id", -1))
        if saved_separator != int(model.separator_token_id):
            raise RuntimeError(
                f"converted Mellow separator mismatch: saved={saved_separator} "
                f"runtime={model.separator_token_id}"
            )
        config["runtime_external_htsat_provenance"] = loaded_provenance
    else:
        # Shared-store evaluation deliberately avoids path/provenance pinning;
        # its portable gate is the saved contract plus bridge/c2l state.
        config["runtime_audio_provenance"] = loaded_provenance
    model.eval()
    expected_compact = bool(checkpoint_audit.get("compact_single_audio_prefix", True))
    if bool(model.config_audio.compact_single_audio_prefix) != expected_compact:
        raise RuntimeError(
            "AudioSmolLM2 runtime compact-prefix setting differs from checkpoint contract"
        )
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
        "compact_single_audio_prefix_used": True,
        "prefix_token_count": DEFAULT_AUDIO_PREFIX_TOKENS,
    }


def _build_fixed260_audio_prefix(
    model: Any,
    waveform: Any,
    device: Any,
    *,
    autocast_enabled: bool,
) -> tuple[Any, dict[str, Any]]:
    """Reuse audio1's HTSAT embedding and run the trainable bridge twice."""

    import torch

    from audio_smollm2_135m_mellow.model import AUDIO_TOKENS_PER_CLIP, SMOLLM2_HIDDEN_SIZE
    from generate_audio_smollm2_checkpoint_reasonaqa import _find_embedding

    audio = waveform.unsqueeze(0).to(device, non_blocking=True)
    reused_mask = torch.ones((1,), dtype=torch.bool, device=device)
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
    ):
        first, second = model.encode_audio(
            audio, None, reused_mask, skip_second_prefix=False
        )
        separator_ids = torch.full(
            (1, 1), int(model.separator_token_id), dtype=torch.long, device=device
        )
        separator = _find_embedding(model.text_model, separator_ids)
        prefix = torch.cat((first, separator, second, separator), dim=1)
    expected_slot = (AUDIO_TOKENS_PER_CLIP, SMOLLM2_HIDDEN_SIZE)
    if tuple(first.shape[1:]) != expected_slot or tuple(second.shape[1:]) != expected_slot:
        raise RuntimeError(
            f"fixed260 slot shape mismatch: first={tuple(first.shape)} second={tuple(second.shape)}"
        )
    if tuple(prefix.shape[1:]) != (SHARED_STORE_AUDIO_PREFIX_TOKENS, SMOLLM2_HIDDEN_SIZE):
        raise RuntimeError(f"fixed260 prefix shape mismatch: {tuple(prefix.shape)}")
    if not bool(torch.isfinite(prefix).all()):
        raise RuntimeError("fixed260 audio prefix contains non-finite values")
    if not bool(torch.equal(first, second)):
        raise RuntimeError(
            "single-audio slots differ in eval mode; audio1 HTSAT embedding reuse failed"
        )
    return prefix, {
        "audio1_prefix_shape": list(first.shape),
        "audio2_prefix_shape": list(second.shape),
        "combined_prefix_shape": list(prefix.shape),
        "prefix_token_count": SHARED_STORE_AUDIO_PREFIX_TOKENS,
        "separator_token_id": int(model.separator_token_id),
        "separator_token": model.tokenizer.decode(
            [int(model.separator_token_id)], skip_special_tokens=False
        ),
        "audio2_reused": True,
        "single_audio_slot": True,
        "compact_single_audio_prefix": False,
        "compact_single_audio_prefix_used": False,
        "audio2_prefix_materialized": True,
        "htsat_audio1_embedding_reused_for_slot2": True,
        "bridge_invocations_for_reused_embedding": 2,
    }


def _build_audio_prefix(
    model: Any,
    waveform: Any,
    device: Any,
    *,
    autocast_enabled: bool,
) -> tuple[Any, dict[str, Any]]:
    if bool(model.config_audio.compact_single_audio_prefix):
        return _build_compact_audio_prefix(model, waveform, device)
    return _build_fixed260_audio_prefix(
        model, waveform, device, autocast_enabled=autocast_enabled
    )


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
        audio_prefix, prefix_audit = _build_audio_prefix(
            model,
            sample["waveform"],
            device,
            autocast_enabled=True,
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


def prepare_model_output_for_official_scorer(value: Any) -> str:
    """Pass decoded text unchanged to both scorer implementations."""

    return str(value)


def _run_mmau_author_reply_generation(
    model: Any,
    tokenizer: Any,
    device: Any,
    sample: Mapping[str, Any],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch
    from generate_audio_smollm2_checkpoint_reasonaqa import _greedy_decode

    prompt_ids_cpu, prompt_original_token_count, prompt_truncated = (
        official.tokenize_mellow_author_reply_prompt(
            tokenizer, str(sample["prompt"]), max_prompt_tokens=max_prompt_tokens
        )
    )
    prompt_ids = prompt_ids_cpu.to(device)
    with torch.inference_mode():
        waveform, audio_segment = official.mellow_author_reply_audio_segment(
            sample["waveform"]
        )
        audio_prefix, prefix_audit = _build_audio_prefix(
            model, waveform, device, autocast_enabled=False
        )
        generated = _greedy_decode(
            model,
            tokenizer,
            audio_prefix,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            autocast_enabled=False,
            top_p=0.8,
            temperature=1.0,
        )
    generated_text_raw = tokenizer.decode(
        generated["generated_token_ids"], skip_special_tokens=False
    )
    generated["generated_text_raw"] = generated_text_raw
    generated["generated_text"] = generated_text_raw.split(
        tokenizer.eos_token or "<|endoftext|>"
    )[0]
    generated["decode_policy"] = "mellow_wrapper_decode_then_split_stop_token"
    generated["prompt_token_count"] = int(prompt_ids.shape[1])
    generated["prompt_original_token_count"] = prompt_original_token_count
    generated["prompt_truncated"] = prompt_truncated
    generated["audio1_segment"] = audio_segment
    generated["audio2_segment"] = {
        **audio_segment,
        "policy": "reuse_audio1_segment_and_htsat_embedding",
    }
    generated.update(prefix_audit)
    return generated


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
        },
        audio_prefix_tokens=SHARED_STORE_AUDIO_PREFIX_TOKENS,
        stage="mmau_test_mini_audio_smollm2_shared_store_fixed260",
        logical_trace="standard 30-layer SmolLM2 per generation step",
    )
    predictions_path = args.output_dir / "predictions_fixed_order.json"
    if predictions_path.is_file() and report.get("inference_coverage", {}).get("status") == "PASS":
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        author_score = official.write_mellow_author_reply_evaluation(
            args.output_dir, predictions
        )
        report["mellow_author_reply_evaluation"] = author_score
        report["mellow_author_reply_context"] = official.MELLOW_AUTHOR_REPLY_CONTEXT
        report["primary_comparison_score"] = {
            "scorer": official.MELLOW_AUTHOR_REPLY_SCORER,
            "record_errors_counted_incorrect": int(
                author_score.get("record_errors", {}).get("total", 0)
            ),
            **author_score["total"],
        }
        report["mmau_v051525_evaluation"] = report.get("official_evaluation", {})
        _write_json(args.output_dir / "evaluation_report.json", report)
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
