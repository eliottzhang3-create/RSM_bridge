#!/usr/bin/env python3
"""Evaluate the released native Mellow-v0 model on MMAU test-mini.

The benchmark traversal, fixed-order prompt, append-only resumption, complete
denominator, and official scorer are shared with the established MMAU path.
This adapter strictly loads the full released Mellow checkpoint (including
HTSAT) and reproduces the protocol posted by Mellow's authors in GitHub issue
#5.  The same source audio is independently preprocessed and encoded in the
two native slots.  Generation applies top-p filtering and then argmax exactly
as MellowWrapper does; it is therefore deterministic rather than sampling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import audit_mellow_v0_artifact as preflight  # noqa: E402
import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as official  # noqa: E402


DEFAULT_DATASET_DIR = Path(official.DEFAULT_DATASET_DIR)
DEFAULT_OUTPUT_DIR = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mellow_v0/"
    "mmau_test_mini_mellow_author_reply_protocol_v1"
)
DEFAULT_MAX_PROMPT_TOKENS = 129
DEFAULT_MAX_NEW_TOKENS = 300
MELLOW_AUDIO_TOKENS_PER_SLOT = 129
MELLOW_PROMPT_TOKENS = 129
MELLOW_AUDIO_PREFIX_TOKENS = 260
MELLOW_PREFIX_TOKENS = 389
MELLOW_HIDDEN_SIZE = 576
PREDICTION_FORMAT = "mellow_author_reply_raw_generation_choice_label_scoring_v1"
PROTOCOL_CONTRACT = "mellow_v0_mmau_github_issue5_author_reply_reproduction_v1"
MODEL_CONTRACT_FILENAME = "mellow_v0_model_contract.json"
SMOKE_GATE_FILENAME = "mellow_v0_smoke_gate.json"
SHARED_STORAGE_PREFIXES = ("/hpc_stor03", "/mnt/cloudstorfs")


def prepare_model_output_for_official_scorer(value: Any) -> str:
    """Pass decoded Mellow output verbatim to MMAU's official scorer."""

    return str(value)


def _shared_storage_identity(value: str | Path) -> str:
    """Normalize the two cluster mount names without weakening artifact identity."""

    text = Path(value).as_posix().rstrip("/") or "/"
    for prefix in SHARED_STORAGE_PREFIXES:
        if text == prefix:
            return "shared-storage:/"
        if text.startswith(prefix + "/"):
            return "shared-storage:/" + text[len(prefix) + 1:]
    return text


def _same_artifact_path(expected: Path, reported: Any) -> bool:
    if not isinstance(reported, str) or not reported:
        return False
    actual = Path(reported)
    try:
        if os.path.samefile(expected, actual):
            return True
    except (FileNotFoundError, OSError, ValueError):
        pass
    candidates_expected = {
        _shared_storage_identity(expected),
        _shared_storage_identity(expected.resolve()),
    }
    candidates_actual = {
        _shared_storage_identity(actual),
        _shared_storage_identity(actual.resolve()),
    }
    return bool(candidates_expected & candidates_actual)


def _load_and_validate_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if not args.preflight_report.is_file():
        raise FileNotFoundError(
            f"Mellow artifact preflight report not found: {args.preflight_report}"
        )
    report = json.loads(args.preflight_report.read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASS"
        or report.get("artifact_contract") != preflight.ARTIFACT_CONTRACT
        or report.get("strict_state_dict_load") is not True
        or report.get("missing_keys") != []
        or report.get("unexpected_keys") != []
    ):
        raise RuntimeError(f"Mellow artifact preflight did not PASS strictly: {report}")
    expected_paths = {
        "mellow_source_root": args.mellow_source_root.resolve(),
        "mellow_snapshot": args.mellow_snapshot.resolve(),
        "mellow_checkpoint": args.mellow_checkpoint.resolve(),
        "base_smollm2": args.base_smollm2.resolve(),
    }
    mismatches = {
        key: {
            "expected": str(value),
            "actual": report.get(key),
            "expected_storage_identity": _shared_storage_identity(value),
            "actual_storage_identity": _shared_storage_identity(
                str(report.get(key, ""))
            ),
        }
        for key, value in expected_paths.items()
        if not _same_artifact_path(value, report.get(key))
    }
    checkpoint_hash = preflight._sha256(args.mellow_checkpoint.resolve())
    if report.get("mellow_checkpoint_sha256") != checkpoint_hash:
        mismatches["mellow_checkpoint_sha256"] = {
            "expected": checkpoint_hash,
            "actual": report.get("mellow_checkpoint_sha256"),
        }
    snapshot_config_hash = preflight._sha256(
        args.mellow_snapshot.resolve() / "config.json"
    )
    if report.get("snapshot_config_sha256") != snapshot_config_hash:
        mismatches["snapshot_config_sha256"] = {
            "expected": snapshot_config_hash,
            "actual": report.get("snapshot_config_sha256"),
        }
    current_source_hashes = preflight._source_inventory(args.mellow_source_root.resolve())
    if report.get("mellow_source_sha256") != current_source_hashes:
        mismatches["mellow_source_sha256"] = {
            "expected": current_source_hashes,
            "actual": report.get("mellow_source_sha256"),
        }
    current_smollm2_inventory = preflight._smollm2_inventory(
        args.base_smollm2.resolve()
    )
    if report.get("base_smollm2_inventory") != current_smollm2_inventory:
        mismatches["base_smollm2_inventory"] = {
            "expected": current_smollm2_inventory,
            "actual": report.get("base_smollm2_inventory"),
        }
    if mismatches:
        raise RuntimeError(f"Mellow preflight report is stale or belongs to other artifacts: {mismatches}")
    report["runtime_path_validation"] = {
        "status": "PASS",
        "policy": "samefile_or_known_shared_storage_alias_plus_content_hashes",
        "accepted_aliases": list(SHARED_STORAGE_PREFIXES),
        "paths": {
            key: {
                "runtime": str(value),
                "reported": report.get(key),
                "storage_identity": _shared_storage_identity(value),
            }
            for key, value in expected_paths.items()
        },
    }
    return report


def _model_contract(args: argparse.Namespace, preflight_report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_contract": PROTOCOL_CONTRACT,
        "artifact_contract": preflight.ARTIFACT_CONTRACT,
        "path_identity_policy": "known_shared_storage_alias_v1",
        "preflight_report": _shared_storage_identity(args.preflight_report),
        "preflight_report_sha256": preflight._sha256(args.preflight_report.resolve()),
        "mellow_source_root": _shared_storage_identity(args.mellow_source_root),
        "mellow_snapshot": _shared_storage_identity(args.mellow_snapshot),
        "mellow_checkpoint": _shared_storage_identity(args.mellow_checkpoint),
        "mellow_checkpoint_sha256": preflight_report["mellow_checkpoint_sha256"],
        "base_smollm2": _shared_storage_identity(args.base_smollm2),
        "single_audio_policy": "same_source_independently_preprocessed_and_encoded_in_two_native_slots",
        "audio_preprocessing": official.MELLOW_AUTHOR_REPLY_AUDIO_FORMAT,
        "prompt_tokens": MELLOW_PROMPT_TOKENS,
        "prefix_tokens": MELLOW_PREFIX_TOKENS,
        "generation": {
            "decoder": "mellow_wrapper_top_p_filter_then_argmax_default_lm_cache",
            "do_sample": False,
            "top_p": 0.8,
            "temperature": 1.0,
            "use_cache": "language_model_default_exactly_as_wrapper",
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "inference_dtype": "float32",
        },
        "prediction_format": PREDICTION_FORMAT,
    }


def _ensure_mellow_output_contract(
    args: argparse.Namespace,
    preflight_report: Mapping[str, Any],
) -> None:
    # Let the common evaluator claim/check the directory first, then add the
    # native-Mellow identity that its generic run_config.json does not know.
    official._ensure_output_dir(
        args,
        prediction_format=PREDICTION_FORMAT,
        prompt_format=official.MELLOW_AUTHOR_REPLY_PROMPT_FORMAT,
        protocol=PROTOCOL_CONTRACT,
    )
    path = args.output_dir / MODEL_CONTRACT_FILENAME
    requested = _model_contract(args, preflight_report)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != requested:
            raise RuntimeError(
                f"existing MMAU output belongs to another Mellow model/protocol: "
                f"existing={existing} requested={requested}"
            )
    else:
        official._write_json(path, requested)


def _smoke_gate_payload(
    args: argparse.Namespace,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": "PASS",
        "protocol_contract": PROTOCOL_CONTRACT,
        "model_contract_sha256": preflight._sha256(
            args.output_dir / MODEL_CONTRACT_FILENAME
        ),
        "preflight_report_sha256": preflight._sha256(args.preflight_report.resolve()),
        "inference_status": report.get("inference_coverage", {}).get("status"),
        "expected_rows": report.get("inference_coverage", {}).get("expected_rows"),
        "terminal_records": report.get("inference_coverage", {}).get("terminal_records"),
        "official_evaluation_status": report.get("official_evaluation", {}).get("status"),
    }


def _require_completed_smoke(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output_dir / SMOKE_GATE_FILENAME
    if not path.is_file():
        raise RuntimeError(
            "full Mellow-v0 MMAU evaluation requires the completed first-five smoke gate: "
            f"{path}"
        )
    gate = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "PASS",
        "protocol_contract": PROTOCOL_CONTRACT,
        "model_contract_sha256": preflight._sha256(
            args.output_dir / MODEL_CONTRACT_FILENAME
        ),
        "preflight_report_sha256": preflight._sha256(args.preflight_report.resolve()),
        "inference_status": "PASS",
        "expected_rows": official.SMOKE_ROWS,
        "terminal_records": official.SMOKE_ROWS,
        "official_evaluation_status": "NOT_REQUESTED",
    }
    mismatches = {
        key: {"expected": value, "actual": gate.get(key)}
        for key, value in expected.items()
        if gate.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Mellow-v0 MMAU smoke gate mismatch: {mismatches}")
    return gate


def _load_runtime_model(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("native Mellow-v0 MMAU evaluation requires one CUDA GPU")
    preflight_report = getattr(args, "_validated_preflight_report", None)
    if preflight_report is None:
        preflight_report = _load_and_validate_preflight(args)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model, tokenizer = preflight._instantiate_native_model(
        args.mellow_source_root.resolve(),
        args.base_smollm2.resolve(),
    )
    state = preflight._load_state(args.mellow_checkpoint.resolve())
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"runtime strict state load differs: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del state
    model.to(device)
    model.eval()
    modes = {
        "model": bool(model.training),
        "audio_encoder": bool(model.audio_encoder.training),
        "htsat": bool(model.audio_encoder.base.htsat.training),
        "c2l": bool(model.audio_encoder.base.c2l.training),
        "projection": bool(model.audio_encoder.projection.training),
        "caption_decoder": bool(model.caption_decoder.training),
        "language_model": bool(model.caption_decoder.lm.training),
    }
    if any(modes.values()):
        raise RuntimeError(f"native Mellow inference requires every module in eval mode: {modes}")
    parameter_devices = {str(parameter.device) for parameter in model.parameters()}
    if parameter_devices != {str(device)}:
        raise RuntimeError(f"native Mellow parameters are not all on cuda:0: {parameter_devices}")
    buffer_devices = {str(buffer.device) for buffer in model.buffers()}
    if buffer_devices and buffer_devices != {str(device)}:
        raise RuntimeError(f"native Mellow buffers are not all on cuda:0: {buffer_devices}")
    return model, tokenizer, device, {
        "artifact_preflight": preflight_report,
        "runtime_model_contract": _model_contract(args, preflight_report),
        "runtime_eval_modes": modes,
        "runtime_parameter_devices": sorted(parameter_devices),
        "runtime_buffer_devices": sorted(buffer_devices),
    }


def _padded_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    max_prompt_tokens: int,
    device: Any,
) -> tuple[dict[str, Any], int]:
    import torch

    encoded = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_prompt_tokens,
        add_special_tokens=True,
        return_tensors="pt",
    )
    untruncated = tokenizer(
        prompt,
        truncation=False,
        padding=False,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_ids_cpu = encoded["input_ids"]
    prompt_token_count = len(official._token_rows(untruncated["input_ids"]))
    if tokenizer.pad_token_id is None:
        raise RuntimeError("native Mellow tokenizer has no pad token")
    prompt_ids = prompt_ids_cpu.to(device)
    attention_mask = encoded["attention_mask"].to(device)
    if tuple(prompt_ids.shape) != (1, MELLOW_PROMPT_TOKENS):
        raise RuntimeError(f"native Mellow padded prompt shape mismatch: {tuple(prompt_ids.shape)}")
    return {
        "input_ids": prompt_ids,
        "attention_mask": attention_mask,
    }, prompt_token_count


def _build_native_prefix(
    model: Any,
    text_input: Mapping[str, Any],
    waveform: Any,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch

    audio1_cpu, audio1_segment = official.mellow_author_reply_audio_segment(waveform)
    audio2_cpu, audio2_segment = official.mellow_author_reply_audio_segment(waveform)
    audio1 = audio1_cpu.to(device, non_blocking=True)
    audio2 = audio2_cpu.to(device, non_blocking=True)
    prefix, _, _ = model.generate_prefix_inference({
        "audio1": audio1,
        "audio2": audio2,
        "input": dict(text_input),
    })
    if tuple(prefix.shape) != (1, MELLOW_PREFIX_TOKENS, MELLOW_HIDDEN_SIZE):
        raise RuntimeError(f"native Mellow prefix shape mismatch: {tuple(prefix.shape)}")
    if not bool(torch.isfinite(prefix).all()):
        raise RuntimeError("native Mellow prefix contains non-finite values")
    first = prefix[:, :MELLOW_AUDIO_TOKENS_PER_SLOT, :]
    second_start = MELLOW_AUDIO_TOKENS_PER_SLOT + 1
    second = prefix[:, second_start:second_start + MELLOW_AUDIO_TOKENS_PER_SLOT, :]
    return prefix, {
        "audio1_prefix_shape": list(first.shape),
        "audio2_prefix_shape": list(second.shape),
        "combined_prefix_shape": list(prefix.shape),
        "prefix_token_count": MELLOW_PREFIX_TOKENS,
        "prefix_layout": "audio1_independent_mellow_preprocess_129 + separator + audio2_independent_mellow_preprocess_129 + separator + truncated_padded_prompt_129",
        "single_audio_slot": True,
        "audio2_same_source": True,
        "audio2_reused": False,
        "audio1_segment": audio1_segment,
        "audio2_segment": audio2_segment,
        "audio2_encoded_separately": True,
        "native_audio_encoder_invocations": 2,
        "compact_single_audio_prefix_used": False,
    }


def _greedy_decode_native(
    model: Any,
    tokenizer: Any,
    prefix: Any,
    *,
    max_new_tokens: int,
    top_p: float = 0.8,
    temperature: float = 1.0,
) -> dict[str, Any]:
    import torch

    eos_id = int(tokenizer.eos_token_id)
    generated_ids: list[int] = []
    generated = prefix
    started = time.perf_counter()
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        # Keep the LM call identical to MellowWrapper._generate_batch.  The
        # wrapper does not pass attention_mask/use_cache/logits_to_keep and
        # recomputes from the full growing embedding sequence on every step.
        output = model.caption_decoder.lm(inputs_embeds=generated)
        if output.logits.ndim != 3 or output.logits.shape[0] != 1:
            raise RuntimeError(f"native Mellow logits shape mismatch: {tuple(output.logits.shape)}")
        if not bool(torch.isfinite(output.logits).all()):
            raise RuntimeError("native Mellow generation logits contain non-finite values")
        logits = output.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        for batch_index in range(sorted_indices_to_remove.shape[0]):
            indices_to_remove = sorted_indices[batch_index][sorted_indices_to_remove[batch_index]]
            logits[batch_index, indices_to_remove] = -float("inf")
        next_token = int(torch.argmax(logits, dim=-1).item())
        generated_ids.append(next_token)
        token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=generated.device)
        token_embed = model.caption_decoder.lm.model.embed_tokens(token_tensor)
        generated = torch.cat((generated, token_embed), dim=1)
        if next_token == eos_id:
            stop_reason = "eos_token"
            break
    elapsed = time.perf_counter() - started
    return {
        "generated_token_ids": generated_ids,
        "generated_text": tokenizer.decode(
            generated_ids, skip_special_tokens=False
        ).split(tokenizer.eos_token or "<|endoftext|>")[0],
        "generated_text_raw": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "stop_reason": stop_reason,
        "eos_token_ids": [eos_id],
        "requested_max_new_tokens": int(max_new_tokens),
        "effective_token_budget": int(max_new_tokens),
        "generation_seconds": elapsed,
        "tokens_per_second": len(generated_ids) / max(elapsed, 1e-9),
        "decoder": "mellow_wrapper_top_p_filter_then_argmax_default_lm_cache",
        "do_sample": False,
        "top_p": float(top_p),
        "temperature": float(temperature),
        "use_cache": "language_model_default_exactly_as_wrapper",
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

    text_input, prompt_token_count = _padded_prompt(
        tokenizer,
        str(sample["prompt"]),
        max_prompt_tokens=max_prompt_tokens,
        device=device,
    )
    with torch.inference_mode():
        prefix, prefix_audit = _build_native_prefix(
            model,
            text_input,
            sample["waveform"],
            device,
        )
        generated = _greedy_decode_native(
            model,
            tokenizer,
            prefix,
            max_new_tokens=max_new_tokens,
        )
    generated["prompt_token_count"] = prompt_token_count
    generated["padded_prompt_token_count"] = MELLOW_PROMPT_TOKENS
    generated["prompt_truncated"] = prompt_token_count > MELLOW_PROMPT_TOKENS
    generated.update(prefix_audit)
    return generated


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--mellow-source-root", type=Path, default=preflight.DEFAULT_MELLOW_SOURCE_ROOT)
    parser.add_argument("--mellow-snapshot", type=Path, default=preflight.DEFAULT_MELLOW_SNAPSHOT)
    parser.add_argument("--mellow-checkpoint", type=Path, default=preflight.DEFAULT_MELLOW_CHECKPOINT)
    parser.add_argument("--base-smollm2", type=Path, default=preflight.DEFAULT_BASE_SMOLLM2)
    parser.add_argument("--preflight-report", type=Path, default=preflight.DEFAULT_REPORT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--evaluation-script", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parquet-batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--dtype", choices=("fp32",), default="fp32")
    parser.add_argument("--run-official-evaluation", action="store_true")
    args = parser.parse_args(argv)
    args.parquet = args.parquet or args.dataset_dir / "test_mini.parquet"
    args.metadata_json = args.metadata_json or args.dataset_dir / "mmau-test-mini.json"
    args.evaluation_script = args.evaluation_script or args.dataset_dir / "evaluation.py"
    args.audio_root = args.audio_root or args.dataset_dir / "test-mini-audios"
    if args.max_prompt_tokens != DEFAULT_MAX_PROMPT_TOKENS:
        parser.error(f"--max-prompt-tokens is fixed at {DEFAULT_MAX_PROMPT_TOKENS}")
    if args.max_new_tokens != DEFAULT_MAX_NEW_TOKENS:
        parser.error(f"--max-new-tokens is fixed at {DEFAULT_MAX_NEW_TOKENS}")
    if args.parquet_batch_size <= 0:
        parser.error("--parquet-batch-size must be positive")
    # Generic common-pipeline names.  They deliberately point at the native
    # Mellow artifacts rather than an RSmol training checkpoint.
    args.checkpoint = args.mellow_snapshot
    args.htsat_checkpoint = args.mellow_checkpoint
    args.mellow_root = args.mellow_source_root
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    preflight_report = _load_and_validate_preflight(args)
    args._validated_preflight_report = preflight_report
    if args.mode == "full":
        _require_completed_smoke(args)
    _ensure_mellow_output_contract(args, preflight_report)
    report = official.run(
        args,
        load_runtime_model=_load_runtime_model,
        run_model_generation=_run_model_generation,
        prepare_prediction=prepare_model_output_for_official_scorer,
        prediction_format=PREDICTION_FORMAT,
        prompt_builder=official.build_mellow_author_reply_prompt,
        audio_decoder=official.decode_mellow_author_reply_audio,
        audio_root=args.audio_root,
        prefer_official_audio_file=True,
        prompt_format=official.MELLOW_AUTHOR_REPLY_PROMPT_FORMAT,
        audio_format=official.MELLOW_AUTHOR_REPLY_AUDIO_FORMAT,
        protocol_contract=PROTOCOL_CONTRACT,
        generation_protocol={
            "decoder": "mellow_wrapper_top_p_filter_then_argmax_default_lm_cache",
            "top_p": 0.8,
            "temperature": 1.0,
            "do_sample": False,
            "use_cache": "language_model_default_exactly_as_wrapper",
            "inference_dtype": "float32",
        },
        audio_prefix_tokens=MELLOW_AUDIO_PREFIX_TOKENS,
        stage="mmau_test_mini_native_mellow_v0_matched_protocol",
        logical_trace="native Mellow-v0 30-layer SmolLM2; two separately encoded same-waveform audio slots",
    )
    inference_failures = int(
        report.get("records", {}).get("skip_reasons", {}).get("sample_exception", 0)
    )
    if inference_failures:
        report["status"] = "FAILED"
        report["comparable_official_score"] = False
        report["fatal_error"] = {
            "error": f"{inference_failures} native Mellow generation failures were recorded as skips",
            "detail": "Inspect skipped.jsonl; the official score is not a valid model comparison.",
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
    if (
        args.mode == "smoke"
        and report.get("status") == official.INFERENCE_ONLY_STATUS
        and report.get("inference_coverage", {}).get("status") == "PASS"
        and report.get("official_evaluation", {}).get("status") == "NOT_REQUESTED"
    ):
        official._write_json(
            args.output_dir / SMOKE_GATE_FILENAME,
            _smoke_gate_payload(args, report),
        )
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
    if report.get("status") == "PASS":
        return 0
    if (
        report.get("status") == official.INFERENCE_ONLY_STATUS
        and report.get("inference_coverage", {}).get("status") == "PASS"
    ):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
