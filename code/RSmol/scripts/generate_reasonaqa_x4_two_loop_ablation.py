#!/usr/bin/env python3
"""Generate ReasonAQA test samples from the x4 audio checkpoint with two loops."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for directory in (SCRIPT_DIR, ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import generate_reasonaqa_mellow_official_training as samples  # noqa: E402
import evaluate_mmau_test_mini_audio_mesh_shared_store as mesh_eval  # noqa: E402
from audio_5_10x2_5_mesh_mellow.data import load_waveform  # noqa: E402
from audio_5_10x2_5_mesh_mellow.model import _find_embedding  # noqa: E402
from audio_5_10x4_5_mesh_mellow_shared_store_configurable_epochs.model import MESH_HIDDEN_SIZE  # noqa: E402
from generate_audio_checkpoint_reasonaqa import _greedy_decode  # noqa: E402

DEFAULT_CHECKPOINT = Path(mesh_eval.DEFAULT_CHECKPOINTS["x4"])
DEFAULT_TEST_JSON = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/test.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--sample-indices", type=int, nargs="+")
    parser.add_argument("--max-prompt-tokens", type=int, default=129)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--htsat-checkpoint", type=Path, default=Path(mesh_eval.DEFAULT_HTSAT))
    parser.add_argument("--mellow-root", type=Path, default=Path(mesh_eval.DEFAULT_MELLOW))
    parser.add_argument("--data-root", type=Path, default=samples.DEFAULT_DATA_ROOT)
    parser.add_argument("--audiocaps-root", type=Path, default=samples.DEFAULT_AUDIOCAPS_ROOT)
    parser.add_argument("--clotho-root", type=Path, default=samples.DEFAULT_CLOTHO_ROOT)
    parser.add_argument("--clotho-aqa-root", type=Path, default=samples.DEFAULT_CLOTHO_AQA_ROOT)
    parser.add_argument("--compare-full", action="store_true", help="Generate the full x4 output for the same inputs too.")
    args = parser.parse_args(argv)
    if args.max_prompt_tokens != 129 or args.max_new_tokens <= 0:
        parser.error("requires max-prompt-tokens=129 and positive max-new-tokens")
    return args


def expected_trace() -> list[dict[str, int]]:
    physical = (*range(5), *range(5, 15), *range(5, 15), *range(15, 20))
    return [{"logical_index": index, "physical_index": layer} for index, layer in enumerate(physical)]


def select_rows(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = samples.load_rows(path)
    indices = samples._select_indices(len(rows), args)
    selected = []
    for index in indices:
        row = rows[index]
        first = samples._path_field(row, first=True)
        second = samples._path_field(row, first=False)
        if not first:
            raise ValueError(f"ReasonAQA test row {index} has no first audio")
        selected.append({
            "index": index, "row": row,
            "audio1": str(samples.resolve_audio(first, row, args)),
            "audio2": str(samples.resolve_audio(second, row, args)) if second else None,
            "prompt": str(row.get("prompt") or row.get("question") or row.get("input") or ""),
            "reference": str(row.get("answer") or row.get("target") or row.get("output") or row.get("caption1") or ""),
        })
    return selected


def build_prefix(model: Any, item: dict[str, Any], device: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    first_wave = load_waveform(item["audio1"]).unsqueeze(0).to(device)
    if item["audio2"] is None:
        prefix, audit = mesh_eval._build_fixed260_zero_prefix(
            model, first_wave, device, mesh_eval.ROUTES["x4"]
        )
        return prefix, {**audit, "audio2_policy": "runtime_zero_waveform"}
    second_wave = load_waveform(item["audio2"]).unsqueeze(0).to(device)
    same_real = item["audio1"] == item["audio2"]
    silence_mask = torch.zeros((1,), dtype=torch.bool, device=device)
    same_mask = torch.tensor([same_real], dtype=torch.bool, device=device)
    first, second = model.encode_audio(first_wave, second_wave, silence_mask, same_mask)
    separator_ids = torch.full((1, 1), int(model.separator_token_id), dtype=torch.long, device=device)
    separator = _find_embedding(model.mesh_model, separator_ids)
    prefix = torch.cat((first, separator, second, separator), dim=1)
    if tuple(prefix.shape[1:]) != (260, MESH_HIDDEN_SIZE) or not bool(torch.isfinite(prefix).all()):
        raise RuntimeError(f"invalid x4 audio prefix: {tuple(prefix.shape)}")
    return prefix, {"prefix_token_count": 260, "audio2_policy": "same_real_audio" if same_real else "distinct_real_audio"}


def tokenize_prompt(tokenizer: Any, prompt: str, device: Any, limit: int) -> tuple[Any, bool]:
    import torch
    encoded = tokenizer(prompt, max_length=limit, truncation=True, padding=False, add_special_tokens=True, return_tensors="pt")
    ids = encoded["input_ids"].to(device)
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise RuntimeError("invalid x4 prompt tokens")
    full = tokenizer(prompt, truncation=False, padding=False, add_special_tokens=True, return_tensors="pt")["input_ids"]
    return ids, int(full.shape[1]) > int(ids.shape[1])


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    if not args.test_json.is_file():
        raise FileNotFoundError(args.test_json)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {args.output_dir}")
    selected = select_rows(args.test_json, args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, tokenizer, device, config = mesh_eval._load_runtime_model(args, mesh_eval.ROUTES["x4"])
    backbone = model.mesh_model.model
    if len(backbone.write_routers) != 5 or len(backbone.read_routers) != 5:
        raise RuntimeError("x4 checkpoint must have five write/read router groups")
    records = []
    watched = {
        "write_1": backbone.write_routers[2],
        "write_2": backbone.write_routers[3],
        "write_3": backbone.write_routers[4],
        "read_1": backbone.read_routers[2],
        "read_2": backbone.read_routers[3],
        "read_3": backbone.read_routers[4],
    }
    with torch.inference_mode():
        for ordinal, item in enumerate(selected, start=1):
            if not item["prompt"]:
                raise ValueError(f"empty prompt at test row {item['index']}")
            prefix, prefix_audit = build_prefix(model, item, device)
            prompt_ids, truncated = tokenize_prompt(tokenizer, item["prompt"], device, args.max_prompt_tokens)
            backbone.ablate_after_two_loops = True
            router_calls = {name: 0 for name in watched}
            def count_call(name: str):
                def hook(_module: Any, _inputs: Any, _output: Any) -> None:
                    router_calls[name] += 1
                return hook
            hooks = [module.register_forward_hook(count_call(name)) for name, module in watched.items()]
            try:
                ablated = _greedy_decode(
                    model, tokenizer, prefix, prompt_ids, max_new_tokens=args.max_new_tokens,
                    autocast_enabled=False, top_p=0.8, temperature=1.0, expected_trace=expected_trace(),
                )
            finally:
                for hook in hooks:
                    hook.remove()
            forwards = ablated["generated_token_count"]
            expected_calls = {"write_1": forwards, "write_2": 0, "write_3": 0, "read_1": 0, "read_2": 0, "read_3": forwards}
            if router_calls != expected_calls:
                raise RuntimeError(f"two-loop router call mismatch: expected={expected_calls} actual={router_calls}")
            full = None
            if args.compare_full:
                backbone.ablate_after_two_loops = False
                full = _greedy_decode(
                    model, tokenizer, prefix, prompt_ids, max_new_tokens=args.max_new_tokens,
                    autocast_enabled=False, top_p=0.8, temperature=1.0,
                    expected_trace=mesh_eval._expected_trace(__import__("recursive_model_5_10x4_5_mesh"), mesh_eval.ROUTES["x4"]),
                )
            record = {
                "sample_ordinal": ordinal, "row_index_zero_based": item["index"],
                "row_number_one_based": item["index"] + 1,
                "prompt": item["prompt"], "reference_answer": item["reference"],
                "audio1_path": item["audio1"], "audio2_path": item["audio2"],
                "prompt_truncated": truncated, "prompt_token_count": int(prompt_ids.shape[1]),
                "prefix_audit": prefix_audit, "router_calls": router_calls, "two_loop_ablation": ablated, "full_x4": full,
            }
            records.append(record)
            print(json.dumps({"row": item["index"], "ablated": ablated["generated_text"], "full": None if full is None else full["generated_text"]}, ensure_ascii=False), flush=True)
    backbone.ablate_after_two_loops = False
    return {
        "status": "PASS", "checkpoint": str(args.checkpoint.resolve()),
        "test_json": str(args.test_json.resolve()),
        "test_json_sha256": hashlib.sha256(args.test_json.read_bytes()).hexdigest(),
        "selected_indices_zero_based": [item["index"] for item in selected],
        "model_route": "x4_7slot_audio_mesh", "checkpoint_config": config,
        "ablation": {"middle_loops": 2, "write_after_loop_2": "write_1 / write_routers[2]", "final_read": "read_3 / read_routers[4]", "use_cache": False, "inference_dtype": "fp32"},
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {args.output_dir}")
    try:
        report = run(args)
    except Exception as exc:
        report = {"status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "generation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    if report["status"] == "PASS":
        with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as stream:
            for record in report["records"]:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output_dir), "samples": len(report.get("records", [])), "error": report.get("error")}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
