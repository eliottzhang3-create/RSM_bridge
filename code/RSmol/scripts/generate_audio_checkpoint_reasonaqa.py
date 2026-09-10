#!/usr/bin/env python3
"""Generate a few deterministic ReasonAQA test samples from an audio checkpoint."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_5_10x2_5_mesh_mellow.data import ReasonAQADataset  # noqa: E402
from audio_5_10x2_5_mesh_mellow.model import (  # noqa: E402
    ARCHITECTURE_CONTRACT,
    AUDIO_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    MAPPER_CONTRACT,
    MESH_HIDDEN_SIZE,
    _find_embedding,
)
from recursive_model_5_10x2_5_mesh import LOGICAL_TO_PHYSICAL  # noqa: E402
from train_audio_5_10x2_5_mesh_mellow_ddp import (  # noqa: E402
    _audit_saved_checkpoint,
    _load_model,
)


DEFAULT_CHECKPOINT = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10x2_5_mesh_mellow/formal_restart_save500_20260910_105248/checkpoint-001500"
DEFAULT_TEST_MANIFEST = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_test.jsonl"
DEFAULT_HTSAT = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt"
DEFAULT_MELLOW = "/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--test-manifest", type=Path, default=Path(DEFAULT_TEST_MANIFEST))
    parser.add_argument("--htsat-checkpoint", type=Path, default=Path(DEFAULT_HTSAT))
    parser.add_argument("--mellow-root", type=Path, default=Path(DEFAULT_MELLOW))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        help="Explicit zero-based manifest rows; overrides --num-samples selection.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=129)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args(argv)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_path(row: dict[str, Any], *, first: bool) -> str:
    keys = ("audio1_path", "filepath1") if first else ("audio2_path", "filepath2")
    return next((str(row[key]) for key in keys if row.get(key)), "")


def _token_rows(value: Any) -> list[int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def _eos_ids(model: Any, tokenizer: Any) -> set[int]:
    values: list[Any] = [
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ]
    resolved: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            resolved.update(int(item) for item in value if item is not None)
        else:
            resolved.add(int(value))
    return resolved


def _select_indices(length: int, args: argparse.Namespace) -> list[int]:
    if length <= 0:
        raise ValueError("ReasonAQA test manifest is empty")
    if args.sample_indices:
        indices = list(args.sample_indices)
        if len(set(indices)) != len(indices):
            raise ValueError(f"sample indices must be unique: {indices}")
    else:
        if args.num_samples <= 0:
            raise ValueError("num-samples must be positive")
        if args.num_samples > length:
            raise ValueError(f"requested {args.num_samples} samples from a {length}-row manifest")
        indices = random.Random(args.seed).sample(range(length), args.num_samples)
    invalid = [index for index in indices if index < 0 or index >= length]
    if invalid:
        raise IndexError(f"sample indices outside [0, {length}): {invalid}")
    return indices


def _validate_checkpoint_contract(args: argparse.Namespace) -> dict[str, Any]:
    artifact = _audit_saved_checkpoint(args.checkpoint)
    config = _json(args.checkpoint / "audio_mesh_config.json")
    expected = {
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "mesh_hidden_size": MESH_HIDDEN_SIZE,
        "audio_tokens_per_clip": AUDIO_TOKENS_PER_CLIP,
        "audio_prefix_tokens_with_separators": AUDIO_PREFIX_TOKENS,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"checkpoint generation contract mismatch: {mismatches}")
    saved_htsat_value = str(config.get("htsat_checkpoint", ""))
    saved_mellow_value = str(config.get("mellow_root", ""))
    if not saved_htsat_value:
        raise RuntimeError("checkpoint does not record its external HTSAT checkpoint")
    if not saved_mellow_value:
        raise RuntimeError("checkpoint does not record its external Mellow root")
    saved_htsat = Path(saved_htsat_value)
    saved_mellow = Path(saved_mellow_value)
    if saved_htsat.resolve() != args.htsat_checkpoint.resolve():
        raise RuntimeError(
            f"external HTSAT mismatch: saved={saved_htsat} requested={args.htsat_checkpoint}"
        )
    if saved_mellow.resolve() != args.mellow_root.resolve():
        raise RuntimeError(
            f"external Mellow root mismatch: saved={saved_mellow} requested={args.mellow_root}"
        )
    return {
        **artifact,
        "audio_mesh_config": config,
        "external_htsat_verified": str(args.htsat_checkpoint.resolve()),
        "mellow_root_verified": str(args.mellow_root.resolve()),
    }


def _expected_trace() -> list[dict[str, int]]:
    return [
        {"logical_index": logical, "physical_index": int(physical)}
        for logical, physical in enumerate(LOGICAL_TO_PHYSICAL)
    ]


def _build_audio_prefix(
    model: Any,
    item: dict[str, Any],
    device: torch.device,
    *,
    autocast_enabled: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    audio1 = item["audio1"].unsqueeze(0).to(device, non_blocking=True)
    audio2 = item["audio2"]
    if audio2 is not None:
        audio2 = audio2.unsqueeze(0).to(device, non_blocking=True)
    reused_mask = torch.tensor([bool(item["audio2_reused"])], dtype=torch.bool, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        first, second = model.encode_audio(audio1, audio2, reused_mask)
        separator_ids = torch.full(
            (1, 1),
            int(model.separator_token_id),
            dtype=torch.long,
            device=device,
        )
        separator = _find_embedding(model.mesh_model, separator_ids)
        prefix = torch.cat((first, separator, second, separator), dim=1)
    if tuple(first.shape[1:]) != (AUDIO_TOKENS_PER_CLIP, MESH_HIDDEN_SIZE):
        raise RuntimeError(f"audio1 prefix shape mismatch: {tuple(first.shape)}")
    if tuple(second.shape[1:]) != (AUDIO_TOKENS_PER_CLIP, MESH_HIDDEN_SIZE):
        raise RuntimeError(f"audio2 prefix shape mismatch: {tuple(second.shape)}")
    if tuple(prefix.shape[1:]) != (AUDIO_PREFIX_TOKENS, MESH_HIDDEN_SIZE):
        raise RuntimeError(f"combined audio prefix shape mismatch: {tuple(prefix.shape)}")
    if not bool(torch.isfinite(prefix).all()):
        raise RuntimeError("combined audio prefix contains non-finite values")
    return prefix, {
        "audio1_prefix_shape": list(first.shape),
        "audio2_prefix_shape": list(second.shape),
        "combined_prefix_shape": list(prefix.shape),
        "separator_token_id": int(model.separator_token_id),
        "separator_token": model.tokenizer.decode(
            [int(model.separator_token_id)], skip_special_tokens=False
        ),
        "audio2_reused": bool(item["audio2_reused"]),
    }


def _greedy_decode(
    model: Any,
    tokenizer: Any,
    audio_prefix: torch.Tensor,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    autocast_enabled: bool,
) -> dict[str, Any]:
    if max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    max_context = int(model.config_audio.max_context_length)
    available = max_context - int(audio_prefix.shape[1]) - int(prompt_ids.shape[1])
    if available <= 0:
        raise RuntimeError(
            f"prompt leaves no generation room: prefix={audio_prefix.shape[1]} "
            f"prompt={prompt_ids.shape[1]} max_context={max_context}"
        )
    token_budget = min(int(max_new_tokens), available)
    eos_ids = _eos_ids(model.mesh_model, tokenizer)
    generated: list[int] = []
    text_ids = prompt_ids
    expected_trace = _expected_trace()
    started = time.perf_counter()
    stop_reason = "max_new_tokens" if token_budget == max_new_tokens else "max_context_length"
    for _ in range(token_budget):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
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
            )
        if tuple(output.logits.shape[:2]) != (1, 1):
            raise RuntimeError(f"last-token logits shape mismatch: {tuple(output.logits.shape)}")
        if not bool(torch.isfinite(output.logits).all()):
            raise RuntimeError("generation logits contain non-finite values")
        trace = list(model.mesh_model.model.last_forward_trace)
        if trace != expected_trace:
            raise RuntimeError(
                f"MeSH generation trace mismatch: expected={expected_trace} actual={trace}"
            )
        next_token = int(torch.argmax(output.logits[:, -1, :], dim=-1).item())
        generated.append(next_token)
        text_ids = torch.cat(
            (text_ids, torch.tensor([[next_token]], dtype=torch.long, device=text_ids.device)),
            dim=1,
        )
        if next_token in eos_ids:
            stop_reason = "eos_token"
            break
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
        "logical_trace_verified": True,
        "logical_trace": expected_trace,
    }


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ReasonAQA checkpoint sample generation",
        "",
        f"- Status: `{report['status']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Test manifest: `{report['test_manifest']}`",
        f"- Selected rows: `{report.get('selected_indices', [])}`",
        f"- Decoder: greedy, `use_cache=False`, max new tokens `{report['generation']['max_new_tokens']}`",
        "",
    ]
    for number, sample in enumerate(report.get("samples", []), start=1):
        lines.extend(
            [
                f"## Sample {number} -- manifest row {sample['row_index']}",
                "",
                "### Question / prompt",
                "",
                sample["prompt"],
                "",
                "### Model generation",
                "",
                sample["generated_text"] or "*(empty after removing special tokens)*",
                "",
                "### Reference answer",
                "",
                sample["reference_answer"],
                "",
                f"- Audio 1: `{sample['audio1_path']}`",
                f"- Audio 2: `{sample['audio2_path']}`",
                f"- Audio 2 reused: `{sample['audio2_reused']}`",
                f"- Stop reason: `{sample['stop_reason']}`",
                f"- Generated token IDs: `{sample['generated_token_ids']}`",
                f"- Raw decode: `{sample['generated_text_raw']}`",
                f"- Normalized exact match: `{sample['normalized_exact_match']}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("ReasonAQA audio checkpoint generation requires one CUDA GPU")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {args.checkpoint}")
    if not args.test_manifest.is_file():
        raise FileNotFoundError(f"ReasonAQA test manifest not found: {args.test_manifest}")
    if not args.htsat_checkpoint.is_file():
        raise FileNotFoundError(f"HTSAT checkpoint not found: {args.htsat_checkpoint}")
    if not args.mellow_root.is_dir():
        raise FileNotFoundError(f"Mellow root not found: {args.mellow_root}")
    if args.max_prompt_tokens != 129:
        raise ValueError("this checkpoint was trained with max_prompt_tokens=129")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    checkpoint_audit = _validate_checkpoint_contract(args)
    # The artifact audit loads optimizer state on CPU. Release it before the
    # model and HTSAT wrapper are materialized for inference.
    gc.collect()
    load_args = argparse.Namespace(
        resume_from=args.checkpoint,
        model_path=args.checkpoint / "mesh_model",
        tokenizer_path=None,
        htsat_checkpoint=args.htsat_checkpoint,
        mellow_root=args.mellow_root,
    )
    model, tokenizer = _load_model(load_args, device)
    saved_provenance = checkpoint_audit["audio_mesh_config"].get("mellow_provenance") or {}
    loaded_provenance = getattr(model, "_audio_provenance", {})
    for key in ("module", "mellow_htsat_source", "mellow_htsat_sha256"):
        if saved_provenance.get(key) != loaded_provenance.get(key):
            raise RuntimeError(
                f"Mellow provenance mismatch for {key}: "
                f"saved={saved_provenance.get(key)!r} loaded={loaded_provenance.get(key)!r}"
            )
    model.eval()
    model.mesh_model.model.audit_mode = False
    model.mesh_model.model.gradient_audit_mode = False
    model.mesh_model.model.routing_stats_mode = False
    modes = {
        "composite_training": bool(model.training),
        "mesh_training": bool(model.mesh_model.training),
        "bridge_training": bool(model.bridge.training),
        "wrapper_training": bool(model.htsat_wrapper.training),
        "htsat_training": bool(model.htsat_backbone.training),
        "c2l_training": bool(model.htsat_wrapper.c2l.training),
    }
    if any(modes.values()):
        raise RuntimeError(f"generation requires every module in eval mode: {modes}")

    dataset = ReasonAQADataset(
        args.test_manifest,
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    selected = _select_indices(len(dataset), args)
    autocast_enabled = args.dtype == "bf16"
    samples: list[dict[str, Any]] = []
    with torch.inference_mode():
        for ordinal, row_index in enumerate(selected, start=1):
            item = dataset[row_index]
            row = dataset.rows[row_index]
            prompt = str(item["prompt"])
            reference = str(item["answer"])
            untruncated = tokenizer(
                prompt,
                truncation=False,
                padding=False,
                add_special_tokens=True,
                return_tensors=None,
            )
            encoded = tokenizer(
                prompt,
                max_length=args.max_prompt_tokens,
                truncation=True,
                padding=False,
                add_special_tokens=True,
                return_tensors="pt",
            )
            prompt_token_ids = _token_rows(encoded["input_ids"])
            full_prompt_ids = _token_rows(untruncated["input_ids"])
            if not prompt_token_ids:
                raise RuntimeError(f"manifest row {row_index} produced an empty prompt token sequence")
            prompt_ids = torch.tensor(
                [prompt_token_ids], dtype=torch.long, device=device
            )
            audio_prefix, prefix_audit = _build_audio_prefix(
                model,
                item,
                device,
                autocast_enabled=autocast_enabled,
            )
            generated = _greedy_decode(
                model,
                tokenizer,
                audio_prefix,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                autocast_enabled=autocast_enabled,
            )
            audio1_path = _manifest_path(row, first=True)
            audio2_path = _manifest_path(row, first=False) or audio1_path
            record = {
                "sample_ordinal": ordinal,
                "row_index": row_index,
                "prompt": prompt,
                "reference_answer": reference,
                "audio1_path": audio1_path,
                "audio2_path": audio2_path,
                "prompt_token_ids": prompt_token_ids,
                "prompt_token_count": len(prompt_token_ids),
                "prompt_original_token_count": len(full_prompt_ids),
                "prompt_truncated": len(prompt_token_ids) < len(full_prompt_ids),
                "normalized_exact_match": _normalize(generated["generated_text"])
                == _normalize(reference),
                **prefix_audit,
                **generated,
            }
            samples.append(record)
            print(
                json.dumps(
                    {
                        "sample": ordinal,
                        "row_index": row_index,
                        "question": prompt,
                        "generated": record["generated_text"],
                        "reference": reference,
                        "stop_reason": record["stop_reason"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    return {
        "stage": "reasonaqa_test_sample_generation_5_10x2_5_mesh_mellow",
        "status": "PASS",
        "checkpoint": str(args.checkpoint.resolve()),
        "test_manifest": str(args.test_manifest.resolve()),
        "test_manifest_sha256": hashlib.sha256(args.test_manifest.read_bytes()).hexdigest(),
        "test_manifest_rows": len(dataset),
        "selected_indices": selected,
        "checkpoint_audit": checkpoint_audit,
        "loaded_mellow_provenance": loaded_provenance,
        "model_eval_modes": modes,
        "device": {
            "torch_device": str(device),
            "cuda_name": torch.cuda.get_device_name(device),
            "dtype": args.dtype,
        },
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "use_cache": False,
            "max_new_tokens": args.max_new_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "seed": args.seed,
            "sample_selection": "explicit_indices" if args.sample_indices else "seeded_without_replacement",
            "audio_encoded_once_per_sample": True,
            "multimodal_prefix_order": "audio1 + separator + audio2 + separator + prompt + generated_tokens",
        },
        "samples": samples,
        "summary": {
            "sample_count": len(samples),
            "nonempty_generations": sum(bool(sample["generated_text"].strip()) for sample in samples),
            "normalized_exact_matches": sum(bool(sample["normalized_exact_match"]) for sample in samples),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_path or args.output_dir / "reasonaqa_samples.json"
    markdown_path = args.output_dir / "reasonaqa_samples.md"
    if report_path.exists() or markdown_path.exists():
        raise FileExistsError(
            f"refusing to overwrite generation report: {report_path} or {markdown_path}"
        )
    report: dict[str, Any] = {
        "stage": "reasonaqa_test_sample_generation_5_10x2_5_mesh_mellow",
        "status": "FAIL",
        "checkpoint": str(args.checkpoint),
        "test_manifest": str(args.test_manifest),
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "hard_failures": [],
    }
    try:
        report = run(args)
    except Exception as exc:
        report["hard_failures"].append(
            {"error": repr(exc), "traceback": traceback.format_exc()}
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    if report["status"] == "PASS":
        markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": report["stage"],
                "status": report["status"],
                "report": str(report_path),
                "comparison_report": str(markdown_path) if markdown_path.exists() else None,
                "summary": report.get("summary"),
                "hard_failures": len(report.get("hard_failures", [])),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
