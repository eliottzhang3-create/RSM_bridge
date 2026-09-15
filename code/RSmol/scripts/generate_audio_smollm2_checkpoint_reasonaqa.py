#!/usr/bin/env python3
"""Generate deterministic ReasonAQA samples from the audio SmolLM2 baseline."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_smollm2_135m_mellow.data import ReasonAQADataset  # noqa: E402
from audio_smollm2_135m_mellow.model import (  # noqa: E402
    AUDIO_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    MAPPER_CONTRACT,
    ORIGINAL_SMOLLM2_CONTRACT,
    SMOLLM2_HIDDEN_SIZE,
)
from train_audio_smollm2_135m_mellow_ddp import (  # noqa: E402
    _audit_saved_checkpoint,
    _load_model,
)


DEFAULT_CHECKPOINT = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_smollm2_135m_mellow/formal_20260911_v1/checkpoint-011343"
DEFAULT_TEST_MANIFEST = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_test.jsonl"
DEFAULT_HTSAT = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt"
DEFAULT_MELLOW = "/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main"
DEFAULT_MAX_PROMPT_TOKENS = 129
DEFAULT_MAX_NEW_TOKENS = 16


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--test-manifest", type=Path, default=Path(DEFAULT_TEST_MANIFEST))
    parser.add_argument("--htsat-checkpoint", type=Path, default=Path(DEFAULT_HTSAT))
    parser.add_argument("--mellow-root", type=Path, default=Path(DEFAULT_MELLOW))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--sample-indices", type=int, nargs="+", help="Zero-based manifest rows.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args(argv)
    if args.max_new_tokens != DEFAULT_MAX_NEW_TOKENS:
        parser.error(f"--max-new-tokens is fixed at {DEFAULT_MAX_NEW_TOKENS} for the established comparison protocol")
    if args.max_prompt_tokens != DEFAULT_MAX_PROMPT_TOKENS:
        parser.error(f"--max-prompt-tokens is fixed at {DEFAULT_MAX_PROMPT_TOKENS} for this checkpoint")
    return args


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _token_rows(value: Any) -> list[int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def _eos_ids(model: Any, tokenizer: Any) -> set[int]:
    values = [
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ]
    result: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            result.update(int(item) for item in value if item is not None)
        else:
            result.add(int(value))
    return result


def _select_indices(length: int, args: argparse.Namespace) -> list[int]:
    if length <= 0:
        raise ValueError("ReasonAQA test manifest is empty")
    indices = list(args.sample_indices) if args.sample_indices else list(range(args.num_samples))
    if not indices or len(set(indices)) != len(indices):
        raise ValueError(f"sample indices must be non-empty and unique: {indices}")
    if any(index < 0 or index >= length for index in indices):
        raise IndexError(f"sample indices outside [0, {length}): {indices}")
    return indices


def _find_embedding(model: Any, ids: torch.Tensor) -> torch.Tensor:
    return model.get_input_embeddings()(ids)


def _manifest_path(row: dict[str, Any], *, first: bool) -> str:
    keys = ("audio1_path", "filepath1") if first else ("audio2_path", "filepath2")
    return next((str(row[key]) for key in keys if row.get(key)), "")


def _validate_checkpoint_contract(args: argparse.Namespace) -> dict[str, Any]:
    artifact = _audit_saved_checkpoint(args.checkpoint)
    config_path = args.checkpoint / "audio_smollm2_config.json"
    config = _json(config_path)
    expected = {
        "architecture_contract": ORIGINAL_SMOLLM2_CONTRACT,
        "mapper_contract": MAPPER_CONTRACT,
        "text_model_type": "llama",
        "text_model_hidden_size": SMOLLM2_HIDDEN_SIZE,
        "text_model_num_hidden_layers": 30,
        "audio_tokens_per_clip": AUDIO_TOKENS_PER_CLIP,
        "audio_prefix_tokens_with_separators": AUDIO_PREFIX_TOKENS,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"standard SmolLM2 checkpoint contract mismatch: {mismatches}")
    forbidden = [
        key for key in config.get("trainable_parameter_names", [])
        if any(marker in str(key).lower() for marker in ("router", "memory", "recursive", "shared_loop"))
    ]
    if forbidden:
        raise RuntimeError(f"checkpoint contains forbidden custom parameters: {forbidden[:8]}")
    for key, requested in (("htsat_checkpoint", args.htsat_checkpoint), ("mellow_root", args.mellow_root)):
        saved = config.get(key)
        if not saved:
            raise RuntimeError(f"checkpoint does not record {key}")
        if Path(str(saved)).resolve() != requested.resolve():
            raise RuntimeError(f"checkpoint {key} mismatch: saved={saved} requested={requested}")
    return {
        **artifact,
        "audio_smollm2_config": config,
        "architecture_audit": {
            "architecture_contract": ORIGINAL_SMOLLM2_CONTRACT,
            "independent_decoder_layers": True,
            "decoder_layer_count": 30,
            "has_router_parameters": False,
            "has_memory_parameters": False,
        },
        "external_htsat_verified": str(args.htsat_checkpoint.resolve()),
        "mellow_root_verified": str(args.mellow_root.resolve()),
        "config_path": str(config_path.resolve()),
    }


def _build_audio_prefix(
    model: Any,
    item: dict[str, Any],
    device: torch.device,
    *,
    autocast_enabled: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    audio1 = item["audio1"].unsqueeze(0).to(device, non_blocking=True)
    audio2 = item.get("audio2")
    if audio2 is not None:
        audio2 = audio2.unsqueeze(0).to(device, non_blocking=True)
    reused_mask = torch.tensor([bool(item.get("audio2_reused", audio2 is None))], dtype=torch.bool, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        first, second = model.encode_audio(audio1, audio2, reused_mask)
        separator_ids = torch.full((1, 1), int(model.separator_token_id), dtype=torch.long, device=device)
        separator = _find_embedding(model.text_model, separator_ids)
        prefix = torch.cat((first, separator, second, separator), dim=1)
    if tuple(first.shape[1:]) != (AUDIO_TOKENS_PER_CLIP, SMOLLM2_HIDDEN_SIZE):
        raise RuntimeError(f"audio1 prefix shape mismatch: {tuple(first.shape)}")
    if tuple(second.shape[1:]) != (AUDIO_TOKENS_PER_CLIP, SMOLLM2_HIDDEN_SIZE):
        raise RuntimeError(f"audio2 prefix shape mismatch: {tuple(second.shape)}")
    if tuple(prefix.shape[1:]) != (AUDIO_PREFIX_TOKENS, SMOLLM2_HIDDEN_SIZE):
        raise RuntimeError(f"combined audio prefix shape mismatch: {tuple(prefix.shape)}")
    if not bool(torch.isfinite(prefix).all()):
        raise RuntimeError("combined audio prefix contains non-finite values")
    return prefix, {
        "audio1_prefix_shape": list(first.shape),
        "audio2_prefix_shape": list(second.shape),
        "combined_prefix_shape": list(prefix.shape),
        "separator_token_id": int(model.separator_token_id),
        "separator_token": model.tokenizer.decode([int(model.separator_token_id)], skip_special_tokens=False),
        "audio2_reused": bool(item.get("audio2_reused", audio2 is None)),
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
    if max_new_tokens != DEFAULT_MAX_NEW_TOKENS:
        raise ValueError(f"the established comparison protocol requires max_new_tokens={DEFAULT_MAX_NEW_TOKENS}")
    max_context = int(model.config_audio.max_context_length)
    available = max_context - int(audio_prefix.shape[1]) - int(prompt_ids.shape[1])
    if available <= 0:
        raise RuntimeError(f"prompt leaves no generation room: prefix={audio_prefix.shape[1]} prompt={prompt_ids.shape[1]}")
    token_budget = min(int(max_new_tokens), available)
    generated: list[int] = []
    text_ids = prompt_ids
    eos_ids = _eos_ids(model.text_model, tokenizer)
    started = time.perf_counter()
    stop_reason = "max_new_tokens" if token_budget == max_new_tokens else "max_context_length"
    for _generation_step in range(token_budget):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            text_embeds = _find_embedding(model.text_model, text_ids)
            inputs_embeds = torch.cat((audio_prefix, text_embeds), dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
            output = model.text_model(
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
        next_token = int(torch.argmax(output.logits[:, -1, :], dim=-1).item())
        generated.append(next_token)
        text_ids = torch.cat((text_ids, torch.tensor([[next_token]], dtype=torch.long, device=text_ids.device)), dim=1)
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
    }


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ReasonAQA SmolLM2 baseline sample generation", "",
        f"- Status: `{report['status']}`", f"- Checkpoint: `{report['checkpoint']}`",
        f"- Selected rows: `{report.get('selected_indices', [])}`",
        f"- Decoder: greedy, `use_cache=False`, max new tokens `{report['generation']['max_new_tokens']}`", "",
    ]
    for number, sample in enumerate(report.get("samples", []), start=1):
        lines.extend([
            f"## Sample {number} -- manifest row {sample['row_index']}", "",
            "### Question / prompt", "", sample["prompt"], "",
            "### Model generation", "", sample["generated_text"] or "*(empty)*", "",
            "### Reference answer", "", sample["reference_answer"], "",
            f"- Audio 1: `{sample['audio1_path']}`", f"- Audio 2: `{sample['audio2_path']}`",
            f"- Stop reason: `{sample['stop_reason']}`", f"- Raw decode: `{sample['generated_text_raw']}`", "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def _ensure_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"output path is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"refusing to use non-empty generation output directory: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("ReasonAQA SmolLM2 baseline generation requires one CUDA GPU")
    for path, label in ((args.checkpoint, "checkpoint"), (args.test_manifest, "ReasonAQA test manifest"), (args.htsat_checkpoint, "HTSAT checkpoint"), (args.mellow_root, "Mellow root")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    checkpoint_audit = _validate_checkpoint_contract(args)
    gc.collect()
    load_args = argparse.Namespace(
        resume_from=args.checkpoint,
        model_path=args.checkpoint / "text_model",
        tokenizer_path=None,
        htsat_checkpoint=args.htsat_checkpoint,
        mellow_root=args.mellow_root,
    )
    model, tokenizer = _load_model(load_args, device)
    runtime_text_contract = dict(getattr(model, "text_contract", {}))
    if (
        runtime_text_contract.get("model_type") != "llama"
        or int(runtime_text_contract.get("hidden_size", -1)) != SMOLLM2_HIDDEN_SIZE
        or int(runtime_text_contract.get("num_hidden_layers", -1)) != 30
        or runtime_text_contract.get("independent_decoder_layers") is not True
    ):
        raise RuntimeError(f"loaded model is not the standard SmolLM2 contract: {runtime_text_contract}")
    config = checkpoint_audit["audio_smollm2_config"]
    saved_provenance = config.get("mellow_provenance") or {}
    loaded_provenance = getattr(model, "_audio_provenance", {})
    for key in ("module", "mellow_htsat_source", "mellow_htsat_sha256"):
        if saved_provenance.get(key) != loaded_provenance.get(key):
            raise RuntimeError(f"Mellow provenance mismatch for {key}")
    model.eval()
    modes = {"composite": bool(model.training), "text": bool(model.text_model.training), "bridge": bool(model.bridge.training), "wrapper": bool(model.htsat_wrapper.training), "htsat": bool(model.htsat_backbone.training), "c2l": bool(model.htsat_wrapper.c2l.training)}
    if any(modes.values()):
        raise RuntimeError(f"generation requires every module in eval mode: {modes}")
    dataset = ReasonAQADataset(args.test_manifest, tokenizer, max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS)
    selected = _select_indices(len(dataset), args)
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
            encoded = tokenizer(prompt, truncation=True, max_length=DEFAULT_MAX_PROMPT_TOKENS, padding=False, add_special_tokens=True, return_tensors="pt")
            prompt_token_ids = _token_rows(encoded["input_ids"])
            full_prompt_token_ids = _token_rows(untruncated["input_ids"])
            prompt_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
            if not prompt_ids.shape[1]:
                raise RuntimeError(f"manifest row {row_index} produced empty prompt tokens")
            audio_prefix, prefix_audit = _build_audio_prefix(model, item, device, autocast_enabled=args.dtype == "bf16")
            generated = _greedy_decode(model, tokenizer, audio_prefix, prompt_ids, max_new_tokens=args.max_new_tokens, autocast_enabled=args.dtype == "bf16")
            samples.append({
                "sample_ordinal": ordinal, "row_index": row_index, "prompt": prompt,
                "reference_answer": reference, "audio1_path": _manifest_path(row, first=True),
                "audio2_path": _manifest_path(row, first=False) or _manifest_path(row, first=True),
                "prompt_token_ids": prompt_token_ids,
                "prompt_token_count": len(prompt_token_ids),
                "prompt_original_token_count": len(full_prompt_token_ids),
                "prompt_truncated": len(prompt_token_ids) < len(full_prompt_token_ids),
                "normalized_exact_match": _normalize(generated["generated_text"]) == _normalize(reference),
                **prefix_audit, **generated,
            })
            print(json.dumps({"sample": ordinal, "row_index": row_index, "question": prompt, "generated": generated["generated_text"], "reference": reference, "stop_reason": generated["stop_reason"]}, ensure_ascii=False), flush=True)
    return {
        "stage": "reasonaqa_test_sample_generation_audio_smollm2_135m_mellow",
        "status": "PASS", "checkpoint": str(args.checkpoint.resolve()),
        "test_manifest": str(args.test_manifest.resolve()),
        "test_manifest_sha256": hashlib.sha256(args.test_manifest.read_bytes()).hexdigest(),
        "test_manifest_rows": len(dataset), "selected_indices": selected,
        "checkpoint_audit": checkpoint_audit, "loaded_mellow_provenance": loaded_provenance,
        "runtime_text_contract": runtime_text_contract,
        "model_eval_modes": modes, "generation": {
            "do_sample": False, "temperature": 0.0, "use_cache": False,
            "max_new_tokens": args.max_new_tokens, "max_prompt_tokens": args.max_prompt_tokens,
            "seed": args.seed, "audio_encoded_once_per_sample": True,
            "multimodal_prefix_order": "audio1 + separator + audio2 + separator + prompt + generated_tokens",
        }, "samples": samples,
        "summary": {"sample_count": len(samples), "nonempty_generations": sum(bool(s["generated_text"].strip()) for s in samples), "normalized_exact_matches": sum(bool(s["normalized_exact_match"]) for s in samples)},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _ensure_output_dir(args.output_dir)
    report_path = args.report_path or args.output_dir / "reasonaqa_samples.json"
    markdown_path = args.output_dir / "reasonaqa_samples.md"
    if report_path.exists() or markdown_path.exists():
        raise FileExistsError(f"refusing to overwrite generation outputs in {args.output_dir}")
    report: dict[str, Any] = {"stage": "reasonaqa_test_sample_generation_audio_smollm2_135m_mellow", "status": "FAIL", "checkpoint": str(args.checkpoint), "test_manifest": str(args.test_manifest), "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "hard_failures": []}
    try:
        report = run(args)
    except Exception as exc:
        report["hard_failures"].append({"error": repr(exc), "traceback": traceback.format_exc()})
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    if report["status"] == "PASS":
        markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"stage": report["stage"], "status": report["status"], "report": str(report_path), "comparison_report": str(markdown_path) if markdown_path.exists() else None, "summary": report.get("summary"), "hard_failures": len(report.get("hard_failures", []))}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
