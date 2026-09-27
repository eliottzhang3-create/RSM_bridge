#!/usr/bin/env python3
"""Evaluate variable-depth shared-store Audio MeSH at one fixed recursive depth."""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import evaluate_mmau_test_mini_audio_5_10x2_5_mesh_mellow_shared_store as fixed  # noqa: E402

official = fixed.official
DEFAULT_CHECKPOINT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "audio_5_10x2to10_5_mesh_mellow_shared_store/"
    "formal_7epochs_20260925_v1/checkpoint-026467"
)
EXPECTED_GLOBAL_STEP = 26467
EXPECTED_EPOCHS = 7
EXPECTED_STEPS_PER_EPOCH = 3781
EXPECTED_WORLD_SIZE = 8
EXPECTED_MICRO_BATCH_SIZE = 8
EXPECTED_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_MAX_NEW_TOKENS = 300
CONFIG_FILENAME = "audio_mesh_config.json"
PREDICTION_FORMAT = fixed.PREDICTION_FORMAT
MMAU_PROTOCOL_CONTRACT = (
    "variable_depth_shared_store_fixed_r_mmau_author_reply_protocol_v1"
)

_FIRST_RUNTIME_AUDIT: dict[str, Any] | None = None


def _expected_router_calls(depth: int) -> dict[str, int]:
    return {
        "pre_write": 1,
        "pre_read": 1,
        "loop1_write": 1,
        "loop1_read": 1,
        "refine_write": depth - 1,
        "refine_read": max(0, depth - 2),
        "out_read": 1,
    }


def _normalize_router_calls(actual: Mapping[str, Any], depth: int) -> dict[str, int]:
    expected = _expected_router_calls(depth)
    unexpected = sorted(set(actual) - set(expected))
    if unexpected:
        raise RuntimeError(f"unexpected variable-depth router calls: {unexpected}")
    return {name: int(actual.get(name, 0)) for name in expected}


def _audit_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from audio_5_10x2to10_5_mesh_mellow_shared_store import TRAINING_CONTRACT
    from audio_5_10x2to10_5_mesh_mellow_shared_store.model import (
        ARCHITECTURE_CONTRACT,
        AUDIO_DUAL_PREFIX_TOKENS,
        AUDIO_PREFIX_TOKENS,
        AUDIO_TOKENS_PER_CLIP,
        MAPPER_CONTRACT,
        MESH_HIDDEN_SIZE,
    )
    from recursive_model_5_10x2to10_5_mesh import (
        MAX_RECURSIVE_DEPTH,
        MIN_RECURSIVE_DEPTH,
        MODEL_ARCHITECTURE_CONTRACT,
        PHYSICAL_LAYER_COUNT,
    )
    from train_audio_smollm2_135m_mellow_ddp import _text_model_weight_files

    checkpoint = args.checkpoint.resolve(strict=True)
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
        raise RuntimeError(f"variable-depth checkpoint missing required files: {missing}")
    empty = [name for name in required if (checkpoint / name).stat().st_size <= 0]
    if empty:
        raise RuntimeError(f"variable-depth checkpoint contains empty required files: {empty}")
    if checkpoint.name != f"checkpoint-{EXPECTED_GLOBAL_STEP:06d}":
        raise RuntimeError(
            f"expected checkpoint-{EXPECTED_GLOBAL_STEP:06d}, got {checkpoint.name}"
        )

    weights = _text_model_weight_files(checkpoint / "mesh_model")
    if not weights:
        raise RuntimeError("variable-depth checkpoint has no MeSH model weights")

    config = json.loads((checkpoint / CONFIG_FILENAME).read_text(encoding="utf-8"))
    marker = json.loads(
        (checkpoint / "checkpoint_complete.json").read_text(encoding="utf-8")
    )
    mesh_config = json.loads(
        (checkpoint / "mesh_model" / "config.json").read_text(encoding="utf-8")
    )
    depth_sampling = config.get("recursive_depth_sampling") or {}
    expected_config = {
        "contract": TRAINING_CONTRACT,
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "compact_single_audio_prefix": False,
        "prefix_tokens": {"single": 260, "dual": 260},
        "mesh_hidden_size": MESH_HIDDEN_SIZE,
        "audio_tokens_per_clip": AUDIO_TOKENS_PER_CLIP,
        "audio_prefix_tokens_with_separators": AUDIO_PREFIX_TOKENS,
        "mode": "formal",
        "epochs": EXPECTED_EPOCHS,
        "world_size": EXPECTED_WORLD_SIZE,
        "micro_batch_size": EXPECTED_MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION_STEPS,
        "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
        "total_steps": EXPECTED_GLOBAL_STEP,
        "global_step": EXPECTED_GLOBAL_STEP,
        "epoch": EXPECTED_EPOCHS,
        "batch_in_epoch": 0,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    expected_sampling = {
        "distribution": "discrete_uniform",
        "values": list(range(MIN_RECURSIVE_DEPTH, MAX_RECURSIVE_DEPTH + 1)),
        "sampling_unit": "micro_step",
        "rank0_sample_then_broadcast": True,
        "full_backpropagation": True,
        "ddp_sync_each_micro_step": True,
    }
    for key, expected in expected_sampling.items():
        if depth_sampling.get(key) != expected:
            mismatches[f"recursive_depth_sampling.{key}"] = {
                "expected": expected,
                "actual": depth_sampling.get(key),
            }
    if (
        marker.get("status") != "complete"
        or marker.get("contract") != TRAINING_CONTRACT
        or int(marker.get("global_step", -1)) != EXPECTED_GLOBAL_STEP
    ):
        mismatches["checkpoint_complete"] = {"actual": marker}

    architectures = mesh_config.get("architectures") or []
    if "RecursiveLlama5_10x2to10_5MeshForCausalLM" not in architectures:
        mismatches["mesh_model.architectures"] = {
            "expected": ["RecursiveLlama5_10x2to10_5MeshForCausalLM"],
            "actual": architectures,
        }
    mesh_expectations = {
        "recursive_layer_count": PHYSICAL_LAYER_COUNT,
        "recursive_min_depth": MIN_RECURSIVE_DEPTH,
        "recursive_max_depth": MAX_RECURSIVE_DEPTH,
        "model_architecture_contract": MODEL_ARCHITECTURE_CONTRACT,
    }
    for key, expected in mesh_expectations.items():
        if mesh_config.get(key) != expected:
            mismatches[f"mesh_model.{key}"] = {
                "expected": expected,
                "actual": mesh_config.get(key),
            }
    if (
        AUDIO_TOKENS_PER_CLIP != 129
        or AUDIO_PREFIX_TOKENS != 260
        or AUDIO_DUAL_PREFIX_TOKENS != 260
    ):
        mismatches["runtime_audio_constants"] = {
            "expected": {"per_clip": 129, "fixed_prefix": 260},
            "actual": {
                "per_clip": AUDIO_TOKENS_PER_CLIP,
                "prefix": AUDIO_PREFIX_TOKENS,
                "dual_prefix": AUDIO_DUAL_PREFIX_TOKENS,
            },
        }
    if mismatches:
        raise RuntimeError(f"variable-depth checkpoint contract mismatch: {mismatches}")

    state = fixed._load_training_state_metadata(checkpoint / "training_state.pt")
    required_state = {
        "optimizer",
        "scheduler",
        "global_step",
        "cursor",
        "rng_states_by_rank",
        "depth_sampler",
        "depth_sampler_rank_summaries",
    }
    missing_state = sorted(required_state.difference(state))
    if missing_state:
        raise RuntimeError(f"variable-depth training state missing keys: {missing_state}")
    expected_cursor = {
        "epoch": EXPECTED_EPOCHS,
        "batch_in_epoch": 0,
        "global_step": EXPECTED_GLOBAL_STEP,
    }
    cursor = state.get("cursor")
    if not isinstance(cursor, Mapping) or {
        key: int(cursor.get(key, -1)) for key in expected_cursor
    } != expected_cursor:
        raise RuntimeError(f"variable-depth final cursor mismatch: {cursor!r}")
    if int(state.get("global_step", -1)) != EXPECTED_GLOBAL_STEP:
        raise RuntimeError("training_state global_step differs from checkpoint-026467")
    rng_ranks = {str(key) for key in state.get("rng_states_by_rank", {})}
    if rng_ranks != {str(index) for index in range(EXPECTED_WORLD_SIZE)}:
        raise RuntimeError(f"variable-depth RNG rank coverage mismatch: {sorted(rng_ranks)}")
    expected_draws = EXPECTED_GLOBAL_STEP * EXPECTED_GRADIENT_ACCUMULATION_STEPS
    depth_state = state.get("depth_sampler") or {}
    if int(depth_state.get("draws", -1)) != expected_draws:
        raise RuntimeError(
            f"depth sampler draw cursor mismatch: expected={expected_draws} "
            f"actual={depth_state.get('draws')}"
        )
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
        "artifact_kind": "variable_depth_shared_store_formal_checkpoint",
        "path": str(checkpoint),
        "global_step": EXPECTED_GLOBAL_STEP,
        "epochs": EXPECTED_EPOCHS,
        "training_contract": TRAINING_CONTRACT,
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "model_architecture_contract": MODEL_ARCHITECTURE_CONTRACT,
        "recursive_depth_training_range": [MIN_RECURSIVE_DEPTH, MAX_RECURSIVE_DEPTH],
        "recursive_depth_training_policy": depth_sampling,
        "physical_layer_count": PHYSICAL_LAYER_COUNT,
        "required_files": required,
        "text_model_weight_files": [str(path) for path in weights],
        "config_sha256": fixed._sha256(checkpoint / CONFIG_FILENAME),
        "mesh_config_sha256": fixed._sha256(checkpoint / "mesh_model" / "config.json"),
        "compact_single_audio_prefix": False,
        "prefix_tokens": {"single": 260, "dual": 260},
        "single_audio_slot_semantics": fixed.SINGLE_AUDIO_SLOT_SEMANTICS,
    }


def _load_runtime_model(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer

    from audio_5_10x2to10_5_mesh_mellow_shared_store.model import (
        AudioMeshConfig,
        AudioMeshModel,
        RecursiveLlamaForCausalLM,
    )
    import train_audio_5_10x2_5_mesh_mellow_ddp as base_trainer

    if args.dtype != "fp32":
        raise ValueError("variable-depth Mellow-faithful MMAU evaluation requires --dtype fp32")
    if not torch.cuda.is_available():
        raise RuntimeError("variable-depth MMAU evaluation requires one CUDA GPU")
    checkpoint_audit = _audit_checkpoint(args)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    mesh = RecursiveLlamaForCausalLM.from_pretrained(
        args.checkpoint / "mesh_model",
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint / "tokenizer", local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper, htsat, provenance = base_trainer._load_mellow_wrapper(
        args.mellow_root, args.htsat_checkpoint, device
    )
    model = AudioMeshModel(
        mesh,
        tokenizer,
        wrapper,
        htsat,
        AudioMeshConfig(compact_single_audio_prefix=False),
    ).to(device)
    audio_state = torch.load(
        args.checkpoint / "audio_bridge.pt", map_location=device, weights_only=False
    )
    model.bridge.load_state_dict(audio_state["bridge"], strict=True)
    model.htsat_wrapper.c2l.load_state_dict(audio_state["c2l"], strict=True)
    model._audio_provenance = provenance
    model.eval()
    model.mesh_model.set_recursive_depth(args.recursive_depth)

    if bool(model.config_audio.compact_single_audio_prefix):
        raise RuntimeError("variable-depth evaluator requires the fixed 260-token prefix")
    context_length = int(getattr(model.config_audio, "max_context_length", 0))
    if context_length != fixed.DEFAULT_MAX_CONTEXT_LENGTH:
        raise RuntimeError(
            f"inference context mismatch: expected={fixed.DEFAULT_MAX_CONTEXT_LENGTH} "
            f"actual={context_length}"
        )
    owner = model.mesh_model.model
    owner.audit_mode = False
    owner.gradient_audit_mode = False
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
    devices = {str(parameter.device) for parameter in model.parameters() if parameter.requires_grad}
    if devices != {str(device)}:
        raise RuntimeError(f"trainable parameters are not all on cuda:0: {sorted(devices)}")

    config = json.loads((args.checkpoint / CONFIG_FILENAME).read_text(encoding="utf-8"))
    saved_provenance = config.get("mellow_provenance") or {}
    for key in ("module", "mellow_htsat_source", "mellow_htsat_sha256"):
        if saved_provenance.get(key) != provenance.get(key):
            raise RuntimeError(
                f"Mellow provenance mismatch for {key}: "
                f"saved={saved_provenance.get(key)!r} loaded={provenance.get(key)!r}"
            )
    config["checkpoint_artifact_audit"] = checkpoint_audit
    config["runtime_audio_provenance"] = provenance
    config["runtime_max_context_length"] = context_length
    config["evaluation_recursive_depth"] = args.recursive_depth
    return model, tokenizer, device, config


def _fixed_depth_decode(
    model: Any,
    tokenizer: Any,
    audio_prefix: Any,
    prompt_ids: Any,
    *,
    recursive_depth: int,
    max_new_tokens: int,
    top_p: float,
    temperature: float,
) -> dict[str, Any]:
    import torch
    import generate_audio_checkpoint_reasonaqa as generation_common
    from audio_5_10x2_5_mesh_mellow.model import _find_embedding
    from recursive_model_5_10x2to10_5_mesh import build_mesh_schedule, logical_layer_count

    max_context = int(model.config_audio.max_context_length)
    available = max_context - int(audio_prefix.shape[1]) - int(prompt_ids.shape[1])
    if available <= 0:
        raise RuntimeError(
            f"prompt leaves no generation room: prefix={audio_prefix.shape[1]} "
            f"prompt={prompt_ids.shape[1]} max_context={max_context}"
        )
    token_budget = min(int(max_new_tokens), available)
    eos_ids = generation_common._eos_ids(model.mesh_model, tokenizer)
    generated: list[int] = []
    text_ids = prompt_ids
    expected_physical = list(build_mesh_schedule(recursive_depth))
    expected_trace = [
        {"logical_index": index, "physical_index": physical}
        for index, physical in enumerate(expected_physical)
    ]
    expected_calls = _expected_router_calls(recursive_depth)
    started = time.perf_counter()
    stop_reason = "max_new_tokens" if token_budget == max_new_tokens else "max_context_length"
    first_audit: dict[str, Any] | None = None

    for _generation_step in range(token_budget):
        text_embeds = _find_embedding(model.mesh_model, text_ids)
        inputs_embeds = torch.cat((audio_prefix, text_embeds), dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
        )
        output = model.mesh_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
            recursive_depth=recursive_depth,
        )
        if tuple(output.logits.shape[:2]) != (1, 1):
            raise RuntimeError(f"last-token logits shape mismatch: {tuple(output.logits.shape)}")
        if not bool(torch.isfinite(output.logits).all()):
            raise RuntimeError("generation logits contain non-finite values")

        owner = model.mesh_model.model
        trace = list(owner.last_forward_trace)
        calls = _normalize_router_calls(owner.last_router_call_counts, recursive_depth)
        if int(owner.last_recursive_depth or -1) != recursive_depth:
            raise RuntimeError(
                f"runtime recursive depth mismatch: expected={recursive_depth} "
                f"actual={owner.last_recursive_depth}"
            )
        if trace != expected_trace:
            raise RuntimeError(
                f"generation trace mismatch at R={recursive_depth}: "
                f"expected={expected_trace} actual={trace}"
            )
        if calls != expected_calls:
            raise RuntimeError(
                f"router calls mismatch at R={recursive_depth}: "
                f"expected={expected_calls} actual={calls}"
            )
        if first_audit is None:
            first_audit = {
                "status": "PASS",
                "recursive_depth": recursive_depth,
                "logical_layer_count": logical_layer_count(recursive_depth),
                "physical_trace": expected_physical,
                "forward_trace": trace,
                "router_call_counts": calls,
            }

        logits = output.logits[:, -1, :] / (float(temperature) if temperature > 0 else 1.0)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative_probs > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        for batch_index in range(remove.shape[0]):
            indices = sorted_indices[batch_index][remove[batch_index]]
            logits[batch_index, indices] = -float("inf")
        next_token = int(torch.argmax(logits, dim=-1).item())
        generated.append(next_token)
        text_ids = torch.cat(
            (text_ids, torch.tensor([[next_token]], dtype=torch.long, device=text_ids.device)),
            dim=1,
        )
        if next_token in eos_ids:
            stop_reason = "eos_token"
            break

    if first_audit is None:
        raise RuntimeError("fixed-depth decoder performed no forward pass")
    global _FIRST_RUNTIME_AUDIT
    if _FIRST_RUNTIME_AUDIT is None:
        _FIRST_RUNTIME_AUDIT = first_audit
    elif _FIRST_RUNTIME_AUDIT != first_audit:
        raise RuntimeError("fixed-depth runtime audit changed across MMAU samples")

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
        "decoder": "mellow_wrapper_top_p_filter_then_argmax_full_recompute",
        "do_sample": False,
        "top_p": float(top_p),
        "temperature": float(temperature),
        "logical_trace_verified": True,
        "recursive_depth": recursive_depth,
        "logical_layer_count": len(expected_physical),
        "logical_trace": expected_trace,
        "router_call_counts": expected_calls,
    }


def _run_mmau_author_reply_generation(
    model: Any,
    tokenizer: Any,
    device: Any,
    sample: Mapping[str, Any],
    *,
    recursive_depth: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    prompt_ids_cpu, original_count, truncated = official.tokenize_mellow_author_reply_prompt(
        tokenizer, str(sample["prompt"]), max_prompt_tokens=max_prompt_tokens
    )
    prompt_ids = prompt_ids_cpu.to(device)
    with torch.inference_mode():
        waveform, audio_segment = official.mellow_author_reply_audio_segment(sample["waveform"])
        prefix, prefix_audit = fixed._build_fixed260_reused_audio1_prefix(
            model, waveform, device, autocast_enabled=False
        )
        generation = _fixed_depth_decode(
            model,
            tokenizer,
            prefix,
            prompt_ids,
            recursive_depth=recursive_depth,
            max_new_tokens=max_new_tokens,
            top_p=0.8,
            temperature=1.0,
        )
    raw_text = tokenizer.decode(generation["generated_token_ids"], skip_special_tokens=False)
    generation["generated_text_raw"] = raw_text
    generation["generated_text"] = raw_text.split(
        tokenizer.eos_token or "<|endoftext|>"
    )[0]
    generation["decode_policy"] = "mellow_wrapper_decode_then_split_stop_token"
    generation["prompt_token_count"] = int(prompt_ids.shape[1])
    generation["prompt_original_token_count"] = original_count
    generation["prompt_truncated"] = truncated
    generation["audio1_segment"] = audio_segment
    generation["audio2_segment"] = {
        **audio_segment,
        "policy": "reuse_audio1_segment_and_htsat_embedding",
    }
    generation.update(prefix_audit)
    fixed._validate_fixed260_generation(generation)
    return generation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    from recursive_model_5_10x2to10_5_mesh import validate_recursive_depth

    raw = list(sys.argv[1:] if argv is None else argv)
    depth_parser = argparse.ArgumentParser(add_help=False)
    depth_parser.add_argument("--recursive-depth", type=int, required=True)
    depth_args, remaining = depth_parser.parse_known_args(raw)
    if not any(item == "--mode" or item.startswith("--mode=") for item in remaining):
        remaining = ["--mode", "full", *remaining]
    args = official.parse_args(
        remaining,
        default_checkpoint=DEFAULT_CHECKPOINT,
        default_max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        add_audio_root=True,
        description=__doc__,
    )
    args.recursive_depth = validate_recursive_depth(depth_args.recursive_depth)
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    from recursive_model_5_10x2to10_5_mesh import build_mesh_schedule, logical_layer_count

    global _FIRST_RUNTIME_AUDIT
    _FIRST_RUNTIME_AUDIT = None
    depth = int(args.recursive_depth)
    report = official.run(
        args,
        load_runtime_model=_load_runtime_model,
        run_model_generation=partial(
            _run_mmau_author_reply_generation, recursive_depth=depth
        ),
        prepare_prediction=fixed.prepare_model_output_for_official_scorer,
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
        protocol_contract=f"{MMAU_PROTOCOL_CONTRACT}:R={depth}",
        generation_protocol={
            "decoder": "mellow_wrapper_top_p_filter_then_argmax_full_recompute",
            "top_p": 0.8,
            "temperature": 1.0,
            "do_sample": False,
            "use_cache": False,
            "inference_dtype": "float32",
            "fixed_recursive_depth": depth,
            "logical_layer_count": logical_layer_count(depth),
        },
        audio_prefix_tokens=260,
        stage="mmau_test_mini_audio_5_10x2to10_5_mesh_shared_store_fixed_r",
        logical_trace=f"exact variable-depth MeSH 5-10x{depth}-5 trace",
    )
    inference_failures = int(
        report.get("records", {}).get("skip_reasons", {}).get("sample_exception", 0)
    )
    if inference_failures:
        report["status"] = "FAILED"
        report["comparable_official_score"] = False
        report["fatal_error"] = {
            "error": f"{inference_failures} variable-depth generation failures were recorded",
            "detail": "Inspect skipped.jsonl; the resulting accuracy is not comparable.",
        }

    predictions_path = args.output_dir / "predictions_fixed_order.json"
    if predictions_path.is_file() and report.get("inference_coverage", {}).get("status") == "PASS":
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        author_score = official.write_mellow_author_reply_evaluation(args.output_dir, predictions)
        payload_sources = report.get("records", {}).get("audio", {}).get("payload_sources", {})
        fallback_rows = sum(
            int(count) for source, count in payload_sources.items() if source != "official_id_wav"
        )
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
            "record_errors_counted_incorrect": int(
                author_score.get("record_errors", {}).get("total", 0)
            ),
            **author_score["total"],
        }
        report["mmau_v051525_evaluation"] = report.get("official_evaluation", {})
        report["mellow_author_reply_protocol_audit"] = {
            "official_id_wav_rows": int(payload_sources.get("official_id_wav", 0)),
            "fallback_audio_rows": fallback_rows,
            "payload_sources": payload_sources,
            "status": "PASS" if fallback_rows == 0 else "NONCOMPARABLE_FALLBACK",
        }

    report["fixed_recursive_depth_evaluation"] = {
        "status": "PASS" if _FIRST_RUNTIME_AUDIT is not None else "NOT_OBSERVED",
        "requested_recursive_depth": depth,
        "trained_recursive_depth_range": [2, 10],
        "training_depth_policy": "discrete uniform R=2..10 sampled once per micro-step",
        "logical_layer_count": logical_layer_count(depth),
        "physical_layer_count": 20,
        "expected_physical_trace": list(build_mesh_schedule(depth)),
        "expected_router_call_counts": _expected_router_calls(depth),
        "first_observed_runtime_audit": _FIRST_RUNTIME_AUDIT,
    }
    official._write_json(args.output_dir / "evaluation_report.json", report)
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
                "recursive_depth": args.recursive_depth,
                "records": report.get("records", {}),
                "primary_comparison_score": report.get("primary_comparison_score", {}),
                "official_evaluation": report.get("official_evaluation", {}),
                "report": str(args.output_dir / "evaluation_report.json"),
            },
            ensure_ascii=False,
            default=fixed._json_default,
        )
    )
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
