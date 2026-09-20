#!/usr/bin/env python3
"""Evaluate the partition-v2 fixed 5-10-5 recursive audio model on MMAU.

The official MMAU data/order/scorer pipeline is shared with the established
MeSH evaluator.  Checkpoint validation, model loading, compact-prefix
construction, and generation are route-local and fail closed on the exact
20-physical/30-logical fixed-recursion contract.
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
    "audio_5_10_5_recursive_mellow/partition_formal_eos_v2_10epochs_20260919/"
    "checkpoint-037810"
)
DEFAULT_DATASET_DIR = official.DEFAULT_DATASET_DIR
DEFAULT_HTSAT = official.DEFAULT_HTSAT
DEFAULT_MELLOW = official.DEFAULT_MELLOW
DEFAULT_MAX_PROMPT_TOKENS = official.DEFAULT_MAX_PROMPT_TOKENS
DEFAULT_MAX_NEW_TOKENS = official.DEFAULT_MAX_NEW_TOKENS
DEFAULT_MAX_CONTEXT_LENGTH = official.DEFAULT_MAX_CONTEXT_LENGTH
DEFAULT_AUDIO_PREFIX_TOKENS = official.DEFAULT_AUDIO_PREFIX_TOKENS
PARTITION_CONFIG_FILENAME = "audio_recursive_5_10_5_partition_config.json"
PARTITION_CONTRACT = "recursive_5_10_5_component_partitions6_rank_ram_compact_audio_answer_eos_v2"
ARCHITECTURE_CONTRACT = "logical_30_physical_20_5_10_5_loops_2_no_mesh_audio_mellow"
EXPECTED_FINAL_STEP = 37_810
EXPECTED_PREFIX_TOKENS = {"single": 130, "dual": 260}
EXPECTED_ANSWER_TERMINATION = {
    "token": "<|endoftext|>",
    "included_in_max_answer_tokens": True,
    "supervised": True,
}

# Re-export the dependency-light official helpers used by static audits and by
# the MMAR adapter.  There is one data/scorer implementation, but no shared
# model backend or checkpoint contract across experiment routes.
RowSkip = official.RowSkip
ProgressStore = official.ProgressStore
build_fixed_order_prompt = official.build_fixed_order_prompt
decode_and_normalize_audio = official.decode_and_normalize_audio
prepare_model_output_for_official_scorer = official.prepare_model_output_for_official_scorer
row_key = official.row_key
_json_default = official._json_default
_sha256 = official._sha256
_tokenize_without_truncation = official._tokenize_without_truncation


def _expected_recursive_metadata() -> dict[str, Any]:
    from recursive_model_5_10_5 import (
        LOGICAL_LAYER_COUNT,
        LOGICAL_TO_PHYSICAL,
        MIDDLE_LAYER_COUNT,
        PHYSICAL_LAYER_COUNT,
        PREFIX_LAYER_COUNT,
        RECURSIVE_LOOPS,
        SOURCE_LAYER_INDICES_0BASED,
        SUFFIX_LAYER_COUNT,
    )

    return {
        "logical_layer_count": LOGICAL_LAYER_COUNT,
        "physical_layer_count": PHYSICAL_LAYER_COUNT,
        "recursive_loops": RECURSIVE_LOOPS,
        "recursive_loops_scope": "middle_only",
        "prefix_layer_count": PREFIX_LAYER_COUNT,
        "middle_layer_count": MIDDLE_LAYER_COUNT,
        "suffix_layer_count": SUFFIX_LAYER_COUNT,
        "logical_to_physical": list(LOGICAL_TO_PHYSICAL),
        "source_layer_indices_0based": list(SOURCE_LAYER_INDICES_0BASED),
        "has_mesh_router_or_memory": False,
    }


def _load_training_state_metadata(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        # Kept only for environments predating torch.load(mmap=...).  The
        # production rsmol environment supports mmap and does not materialize
        # the large optimizer tensors for this metadata-only proof.
        return torch.load(path, map_location="cpu", weights_only=False)


def _audit_partition_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the exact completed fixed-recursive formal checkpoint."""
    import torch

    from audio_5_10_5_recursive_mellow.model import (
        AUDIO_DUAL_PREFIX_TOKENS,
        AUDIO_SINGLE_PREFIX_TOKENS,
        AUDIO_TOKENS_PER_CLIP,
        MAPPER_CONTRACT,
        RECURSIVE_AUDIO_CONTRACT,
        RECURSIVE_HIDDEN_SIZE,
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
        raise RuntimeError(f"fixed-recursive checkpoint missing required files: {missing}")
    weight_files = _text_model_weight_files(checkpoint / "text_model")
    if not weight_files:
        raise RuntimeError("fixed-recursive checkpoint has no text-model weights")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete" or marker.get("contract") != PARTITION_CONTRACT:
        raise RuntimeError(f"invalid fixed-recursive completion marker: {marker}")
    if marker.get("required") != required:
        raise RuntimeError("fixed-recursive completion marker required-file contract differs")

    expected = {
        "contract": PARTITION_CONTRACT,
        "architecture_contract": RECURSIVE_AUDIO_CONTRACT,
        "compact_single_audio_prefix": True,
        "prefix_tokens": EXPECTED_PREFIX_TOKENS,
        "answer_termination": EXPECTED_ANSWER_TERMINATION,
        "recursive_text_contract": _expected_recursive_metadata(),
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
        raise RuntimeError(f"fixed-recursive checkpoint contract mismatch: {mismatches}")
    if config.get("mapper_contract") != MAPPER_CONTRACT:
        raise RuntimeError("fixed-recursive checkpoint mapper contract mismatch")
    if ARCHITECTURE_CONTRACT != RECURSIVE_AUDIO_CONTRACT:
        raise RuntimeError("runtime recursive architecture constant changed")
    if (
        AUDIO_TOKENS_PER_CLIP != 129
        or AUDIO_SINGLE_PREFIX_TOKENS != DEFAULT_AUDIO_PREFIX_TOKENS
        or AUDIO_DUAL_PREFIX_TOKENS != 260
        or RECURSIVE_HIDDEN_SIZE != 576
    ):
        raise RuntimeError("runtime recursive audio shape constants changed")

    runtime_contract = config.get("text_model_runtime_contract") or {}
    runtime_expected = {
        "model_type": "llama",
        "hidden_size": 576,
        "logical_layer_count": 30,
        "physical_decoder_layer_count": 20,
        "recursive_layer_count": 20,
        "recursive_loops": 2,
        "recursive_loops_scope": "middle_only",
        "prefix_layer_count": 5,
        "middle_layer_count": 10,
        "suffix_layer_count": 5,
        "logical_to_physical": _expected_recursive_metadata()["logical_to_physical"],
        "source_layer_indices_0based": _expected_recursive_metadata()["source_layer_indices_0based"],
        "distinct_physical_decoder_layers": True,
        "forbidden_custom_parameter_names": [],
    }
    runtime_mismatches = {
        key: {"expected": value, "actual": runtime_contract.get(key)}
        for key, value in runtime_expected.items()
        if runtime_contract.get(key) != value
    }
    if runtime_mismatches:
        raise RuntimeError(f"fixed-recursive saved runtime contract mismatch: {runtime_mismatches}")
    if int(config.get("total_steps", -1)) != EXPECTED_FINAL_STEP:
        raise RuntimeError("fixed-recursive formal schedule must contain 37,810 steps")

    marker_step = int(marker.get("global_step", -1))
    suffix = checkpoint.name.removeprefix("checkpoint-")
    directory_step = int(suffix) if checkpoint.name.startswith("checkpoint-") and suffix.isdigit() else -1
    if marker_step != EXPECTED_FINAL_STEP or directory_step != EXPECTED_FINAL_STEP:
        raise RuntimeError(
            "evaluation requires completed checkpoint-037810: "
            f"directory_step={directory_step} marker_step={marker_step}"
        )

    state = _load_training_state_metadata(state_path)
    expected_cursor = {
        "segment": len(config.get("schedule", [])),
        "segment_step": 0,
        "global_step": EXPECTED_FINAL_STEP,
    }
    if (
        state.get("training_contract") != PARTITION_CONTRACT
        or int(state.get("global_step", -1)) != EXPECTED_FINAL_STEP
        or state.get("cursor") != expected_cursor
        or state.get("scheduler_name") != "cosine_lambda"
    ):
        raise RuntimeError("training state does not prove completed recursive formal training")
    if not state.get("optimizer", {}).get("state") or not state.get("scheduler"):
        raise RuntimeError("fixed-recursive checkpoint lacks optimizer/scheduler completion evidence")
    del state

    try:
        audio_state = torch.load(checkpoint / "audio_bridge.pt", map_location="cpu", weights_only=True)
    except TypeError:
        audio_state = torch.load(checkpoint / "audio_bridge.pt", map_location="cpu")
    if not isinstance(audio_state, Mapping) or set(audio_state) != {"bridge", "c2l"}:
        raise RuntimeError("audio_bridge.pt must contain exactly bridge and c2l")
    expected_shapes = {
        "bridge": {
            "linear1.weight": (576, 768),
            "linear2.weight": (576, 576),
            "norm.weight": (576,),
            "norm.bias": (576,),
        },
        "c2l": {"weight": (768, 527), "bias": (768,)},
    }
    for group_name, shapes in expected_shapes.items():
        group = audio_state.get(group_name)
        actual = {} if not isinstance(group, Mapping) else {
            name: tuple(value.shape) for name, value in group.items()
        }
        if actual != shapes:
            raise RuntimeError(
                f"fixed-recursive {group_name} state shapes differ: expected={shapes} actual={actual}"
            )
    del audio_state

    hashes = config.get("checkpoint_audio_state_hashes")
    if not isinstance(hashes, Mapping) or set(hashes) != {"bridge_sha256", "c2l_sha256"}:
        raise RuntimeError("checkpoint lacks exact bridge/c2l state hashes")
    if not isinstance(config.get("mellow_provenance"), Mapping):
        raise RuntimeError("checkpoint lacks Mellow provenance")
    for key, requested in (
        ("htsat_checkpoint", args.htsat_checkpoint),
        ("mellow_root", args.mellow_root),
    ):
        saved = str(config.get(key, ""))
        if not saved or Path(saved).resolve() != requested.resolve():
            raise RuntimeError(f"checkpoint {key} mismatch: saved={saved!r} requested={requested}")

    return {
        "status": "PASS",
        "artifact_kind": "fixed_recursive_partition_training_checkpoint",
        "path": str(checkpoint),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "global_step": marker_step,
        "training_contract": PARTITION_CONTRACT,
        "architecture_contract": RECURSIVE_AUDIO_CONTRACT,
        "required_files": required,
        "text_model_weight_files": [str(path) for path in weight_files],
        "compact_single_audio_prefix": True,
        "single_audio_prefix_tokens": AUDIO_SINGLE_PREFIX_TOKENS,
        "dual_audio_prefix_tokens": AUDIO_DUAL_PREFIX_TOKENS,
        "recursive_text_contract": runtime_contract,
    }


def _load_runtime_model(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("fixed-recursive MMAU/MMAR evaluation requires one CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    from audio_5_10_5_recursive_mellow.model import validate_recursive_5_10_5
    from train_audio_partitioned_5_10_5_recursive_mellow_ddp import (
        _audio_state_hashes,
        _load_model,
    )

    checkpoint_audit = _audit_partition_checkpoint(args)
    config = json.loads(Path(checkpoint_audit["config_path"]).read_text(encoding="utf-8"))
    source_value = str(config.get("text_model_source_path", ""))
    if not source_value:
        raise RuntimeError("fixed-recursive checkpoint has no recorded text-model source")
    source_path = Path(source_value)
    load_args = argparse.Namespace(
        resume_from=args.checkpoint,
        model_path=source_path,
        htsat_checkpoint=args.htsat_checkpoint,
        mellow_root=args.mellow_root,
    )
    model, tokenizer = _load_model(load_args, device)
    runtime_contract = validate_recursive_5_10_5(model.text_model)
    if runtime_contract != config.get("text_model_runtime_contract"):
        raise RuntimeError("loaded recursive text model differs from the saved runtime contract")
    if _audio_state_hashes(model) != config.get("checkpoint_audio_state_hashes"):
        raise RuntimeError("loaded bridge/c2l tensors differ from the checkpoint hashes")

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
        raise RuntimeError("fixed-recursive evaluation requires compact single-audio prefix")
    if int(model.config_audio.max_context_length) != DEFAULT_MAX_CONTEXT_LENGTH:
        raise RuntimeError("fixed-recursive inference context length differs from 768")
    modes = {
        "composite": bool(model.training),
        "text": bool(model.text_model.training),
        "bridge": bool(model.bridge.training),
        "wrapper": bool(model.htsat_wrapper.training),
        "htsat": bool(model.htsat_backbone.training),
        "c2l": bool(model.htsat_wrapper.c2l.training),
    }
    if any(modes.values()):
        raise RuntimeError(f"inference requires all recursive modules in eval mode: {modes}")
    parameter_names = [name.lower() for name, _ in model.named_parameters(remove_duplicate=False)]
    forbidden = [name for name in parameter_names if "router" in name or "memory" in name]
    if forbidden:
        raise RuntimeError(f"fixed-recursive evaluator loaded MeSH parameters: {forbidden[:8]}")

    config = dict(config)
    config["checkpoint_artifact_audit"] = checkpoint_audit
    config["runtime_recursive_text_contract"] = runtime_contract
    config["runtime_max_context_length"] = int(model.config_audio.max_context_length)
    return model, tokenizer, device, config


def _build_compact_audio_prefix(model: Any, waveform: Any, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    from audio_5_10_5_recursive_mellow.model import (
        AUDIO_SINGLE_PREFIX_TOKENS,
        AUDIO_TOKENS_PER_CLIP,
        RECURSIVE_HIDDEN_SIZE,
    )
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
            (1, 1), int(model.separator_token_id), dtype=torch.long, device=device
        )
        separator = _find_embedding(model.text_model, separator_ids)
        prefix = torch.cat((first, separator), dim=1)
    if tuple(first.shape[1:]) != (AUDIO_TOKENS_PER_CLIP, RECURSIVE_HIDDEN_SIZE):
        raise RuntimeError(f"recursive audio1 prefix shape mismatch: {tuple(first.shape)}")
    if tuple(prefix.shape[1:]) != (AUDIO_SINGLE_PREFIX_TOKENS, RECURSIVE_HIDDEN_SIZE):
        raise RuntimeError(f"recursive compact prefix shape mismatch: {tuple(prefix.shape)}")
    if not bool(torch.isfinite(prefix).all()):
        raise RuntimeError("recursive compact audio prefix contains non-finite values")
    return prefix, {
        "audio1_prefix_shape": list(first.shape),
        "combined_prefix_shape": list(prefix.shape),
        "prefix_token_count": AUDIO_SINGLE_PREFIX_TOKENS,
        "prefix_layout": "audio1 + separator1",
        "separator_token_id": int(model.separator_token_id),
        "separator_token": model.tokenizer.decode(
            [int(model.separator_token_id)], skip_special_tokens=False
        ),
        "audio2_reused": True,
        "single_audio_slot": True,
        "compact_single_audio_prefix_used": True,
        "audio2_prefix_materialized": False,
    }


def _greedy_decode_recursive(
    model: Any,
    tokenizer: Any,
    audio_prefix: Any,
    prompt_ids: Any,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Greedy full-recompute decoding with a trace proof on every token."""
    import torch

    from generate_audio_smollm2_checkpoint_reasonaqa import _eos_ids, _find_embedding
    from recursive_model_5_10_5 import LOGICAL_TO_PHYSICAL, PHYSICAL_LAYER_COUNT

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    expected_trace = list(LOGICAL_TO_PHYSICAL)
    layers = list(model.text_model.model.layers)
    if len(layers) != PHYSICAL_LAYER_COUNT or len({id(layer) for layer in layers}) != PHYSICAL_LAYER_COUNT:
        raise RuntimeError("recursive generation requires 20 distinct physical decoder modules")
    max_context = int(model.config_audio.max_context_length)
    available = max_context - int(audio_prefix.shape[1]) - int(prompt_ids.shape[1])
    if available <= 0:
        raise RuntimeError(
            f"prompt leaves no generation room: prefix={audio_prefix.shape[1]} prompt={prompt_ids.shape[1]}"
        )
    token_budget = min(int(max_new_tokens), available)
    generated: list[int] = []
    text_ids = prompt_ids
    eos_ids = _eos_ids(model.text_model, tokenizer)
    stop_reason = "max_new_tokens" if token_budget == max_new_tokens else "max_context_length"
    verified_steps = 0
    started = time.perf_counter()
    trace: list[int] = []
    handles = [
        layer.register_forward_hook(
            lambda _module, _inputs, _output, index=index: trace.append(index)
        )
        for index, layer in enumerate(layers)
    ]
    try:
        for generation_step in range(token_budget):
            trace.clear()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                text_embeds = _find_embedding(model.text_model, text_ids)
                inputs_embeds = torch.cat((audio_prefix, text_embeds), dim=1)
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
                )
                output = model.text_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
            if trace != expected_trace:
                raise RuntimeError(
                    "recursive generation logical trace mismatch at token step "
                    f"{generation_step}: expected={expected_trace} actual={trace}"
                )
            verified_steps += 1
            if tuple(output.logits.shape[:2]) != (1, 1):
                raise RuntimeError(f"last-token logits shape mismatch: {tuple(output.logits.shape)}")
            if not bool(torch.isfinite(output.logits).all()):
                raise RuntimeError("recursive generation logits contain non-finite values")
            next_token = int(torch.argmax(output.logits[:, -1, :], dim=-1).item())
            generated.append(next_token)
            text_ids = torch.cat(
                (text_ids, torch.tensor([[next_token]], dtype=torch.long, device=text_ids.device)),
                dim=1,
            )
            if next_token in eos_ids:
                stop_reason = "eos_token"
                break
    finally:
        for handle in handles:
            handle.remove()
    elapsed = time.perf_counter() - started
    return {
        "generated_token_ids": generated,
        "generated_token_count": len(generated),
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        "generated_text_raw": tokenizer.decode(generated, skip_special_tokens=False),
        "stop_reason": stop_reason,
        "eos_token_ids": sorted(eos_ids),
        "requested_max_new_tokens": int(max_new_tokens),
        "effective_token_budget": token_budget,
        "generation_seconds": elapsed,
        "tokens_per_second": len(generated) / max(elapsed, 1e-9),
        "decoder": "greedy_full_recompute_use_cache_false",
        "logical_trace_verified": verified_steps == len(generated),
        "logical_trace_verified_steps": verified_steps,
        "logical_trace": expected_trace,
        "physical_decoder_layer_count": PHYSICAL_LAYER_COUNT,
        "logical_layer_count": len(expected_trace),
        "has_mesh_router_or_memory": False,
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

    prompt_ids_cpu, prompt_token_count = _tokenize_without_truncation(
        tokenizer, str(sample["prompt"]), max_prompt_tokens=max_prompt_tokens
    )
    prompt_ids = prompt_ids_cpu.to(device)
    with torch.inference_mode():
        audio_prefix, prefix_audit = _build_compact_audio_prefix(
            model, sample["waveform"], device
        )
        generated = _greedy_decode_recursive(
            model,
            tokenizer,
            audio_prefix,
            prompt_ids,
            max_new_tokens=max_new_tokens,
        )
    generated["prompt_token_count"] = prompt_token_count
    generated.update(prefix_audit)
    return generated


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not any(item == "--mode" or item.startswith("--mode=") for item in raw):
        raw = ["--mode", "full", *raw]
    return official.parse_args(
        raw,
        default_checkpoint=DEFAULT_CHECKPOINT,
        description=__doc__,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = official.run(
        args,
        load_runtime_model=_load_runtime_model,
        run_model_generation=_run_model_generation,
        stage="mmau_test_mini_audio_5_10_5_recursive_fixed_order",
        logical_trace="exact physical trace 0..14,5..14,15..19 verified on every generation step",
    )
    # The shared official pipeline counts skipped rows as incorrect, which is
    # appropriate for malformed benchmark rows.  A recursive model inference
    # exception, however, is a broken backend rather than an accuracy score.
    inference_failures = int(report.get("records", {}).get("skip_reasons", {}).get("sample_exception", 0))
    if inference_failures:
        report["status"] = "FAILED"
        report["fatal_error"] = {
            "error": f"{inference_failures} recursive generation failures were recorded as skipped rows",
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
