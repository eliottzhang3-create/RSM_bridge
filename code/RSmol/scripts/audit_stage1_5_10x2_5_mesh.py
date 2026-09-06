#!/usr/bin/env python3
"""CUDA-only fail-closed Stage 1 audit for the isolated MeSH model."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import torch

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from recursive_model_5_10x2_5_mesh import (  # noqa: E402
    LOGICAL_TO_PHYSICAL,
    MEMORY_SLOT_COUNT,
    ROUTER_PARAMETER_COUNT,
    RecursiveLlamaForCausalLM,
    register_auto_class,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=8)
    return parser.parse_args(argv)


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _check(report: dict[str, Any], name: str, condition: bool, detail: Any = None) -> None:
    item = {"name": name, "passed": bool(condition)}
    if detail is not None:
        item["detail"] = detail
    (report["checks"] if condition else report["hard_failures"]).append(item)


def _finite_nonzero(parameter: torch.nn.Parameter) -> bool:
    return parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) and bool(torch.any(parameter.grad != 0))


def _load(model_path: Path, tokenizer_path: Path | None):
    from transformers import AutoTokenizer
    register_auto_class()
    model = RecursiveLlamaForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32)
    tok_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "FAIL", "stage": "stage1_5_10x2_5_mesh", "checks": [], "warnings": [], "hard_failures": [], "cuda_required": True}
    if not torch.cuda.is_available():
        report["hard_failures"].append({"name": "cuda_available", "passed": False, "detail": "Stage 1 is a remote CUDA audit"})
        return report
    device = torch.device("cuda")
    report["device"] = str(device)
    try:
        model, tokenizer = _load(args.model_path, args.tokenizer_path)
        model.to(device).eval()
        model.model.audit_mode = True
        vocab = int(model.config.vocab_size)
        length = max(3, int(args.sequence_length))
        input_ids = torch.arange(length, device=device, dtype=torch.long).unsqueeze(0).remainder(vocab)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        _check(report, "output_shape", tuple(outputs.logits.shape) == (1, length, vocab), tuple(outputs.logits.shape))
        _check(report, "output_finite", bool(torch.isfinite(outputs.logits).all()))
        _check(report, "memory_shape", model.model.last_memory_shape == (1, MEMORY_SLOT_COUNT, length, int(model.config.hidden_size)), model.model.last_memory_shape)
        initial = model.model.last_initial_memory
        _check(report, "memory_initial_slot0_embedding", initial is not None and torch.equal(initial[:, 0], model.model.embed_tokens(input_ids).detach().cpu()))
        _check(report, "memory_initial_slots_zero", initial is not None and bool(torch.equal(initial[:, 1:], torch.zeros_like(initial[:, 1:]))))
        _check(report, "physical_trace_5_10_10_5", [x["physical_index"] for x in model.model.last_forward_trace] == list(LOGICAL_TO_PHYSICAL), model.model.last_forward_trace)
        _check(report, "logical_cache_slots_0_29", [x["logical_index"] for x in model.model.last_forward_trace] == list(range(30)), [x["logical_index"] for x in model.model.last_forward_trace])
        expected_router_names = {"write_pre", "read_pre", "write_0", "read_0", "write_1", "read_1"}
        _check(report, "six_router_outputs", set(model.model.last_router_weights) == expected_router_names, sorted(model.model.last_router_weights))
        for name, weights in model.model.last_router_weights.items():
            _check(report, f"{name}_shape", tuple(weights.shape) == (1, length, MEMORY_SLOT_COUNT), tuple(weights.shape))
            _check(report, f"{name}_finite_nonnegative", bool(torch.isfinite(weights).all()) and bool((weights >= 0).all()))
            _check(report, f"{name}_slot_sum_one", bool(torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-5, rtol=1e-5)))
        prefix = model.model.last_prefix_output
        for name in ("write_pre", "read_pre"):
            _check(report, f"{name}_query_prefix_output", prefix is not None and torch.allclose(model.model.last_router_queries[name], prefix, atol=1e-6, rtol=1e-5))
        for loop in range(2):
            for kind in ("write", "read"):
                _check(report, f"{kind}_{loop}_query_h{loop}", len(model.model.last_core_inputs) == 2 and torch.allclose(model.model.last_router_queries[f"{kind}_{loop}"], model.model.last_core_inputs[loop], atol=1e-6, rtol=1e-5))
        _check(report, "three_write_history_entries", len(model.model.last_memory_write_history) == 3, len(model.model.last_memory_write_history))
        if len(model.model.last_memory_write_history) == 3:
            _check(report, "write_history_has_nonzero_updates", all(bool(torch.any(item[:, 1:] != 0)) for item in model.model.last_memory_write_history))
            # Independent tensor reference for write-then-read accumulation.
            reference_memory = model.model.last_initial_memory.clone()
            reference_memory = reference_memory + model.model.last_prefix_output.unsqueeze(1) * model.model.last_router_weights["write_pre"].transpose(1, 2).unsqueeze(-1)
            _check(report, "reference_transition_write_matches", bool(torch.allclose(reference_memory, model.model.last_memory_write_history[0], atol=1e-5, rtol=1e-5)))
            reference_hidden = (reference_memory * model.model.last_router_weights["read_pre"].transpose(1, 2).unsqueeze(-1)).sum(dim=1)
            for loop in range(2):
                reference_memory = reference_memory + model.model.last_core_outputs[loop].unsqueeze(1) * model.model.last_router_weights[f"write_{loop}"].transpose(1, 2).unsqueeze(-1)
                _check(report, f"reference_core_{loop}_write_matches", bool(torch.allclose(reference_memory, model.model.last_memory_write_history[loop + 1], atol=1e-5, rtol=1e-5)))
                reference_hidden = (reference_memory * model.model.last_router_weights[f"read_{loop}"].transpose(1, 2).unsqueeze(-1)).sum(dim=1)
            _check(report, "reference_read_path_finite", bool(torch.isfinite(reference_hidden).all()))

        # Cache contract: one full call and prefill + one-token incremental call.
        with torch.no_grad():
            full = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            split = max(1, length - 1)
            prefill = model(input_ids=input_ids[:, :split], attention_mask=torch.ones((1, split), device=device, dtype=torch.long), use_cache=True)
            incremental = model(input_ids=input_ids[:, split:], attention_mask=attention_mask, past_key_values=prefill.past_key_values, cache_position=torch.tensor([split], device=device), use_cache=True)
        _check(report, "cache_present", full.past_key_values is not None and prefill.past_key_values is not None)
        _check(report, "prefill_incremental_shape", tuple(incremental.logits.shape) == (1, 1, vocab), tuple(incremental.logits.shape))
        _check(report, "prefill_incremental_logits_close", bool(torch.allclose(full.logits[:, -1:].float(), incremental.logits.float(), atol=3e-3, rtol=3e-3)), float((full.logits[:, -1:].float() - incremental.logits.float()).abs().max().item()))
        _check(report, "generation_explicit_mask_pad", tokenizer.pad_token_id is not None)
        try:
            with torch.no_grad():
                generated = model.generate(input_ids=input_ids[:, :2], attention_mask=torch.ones((1, 2), device=device, dtype=torch.long), pad_token_id=int(tokenizer.pad_token_id), max_new_tokens=1, do_sample=False)
            _check(report, "generation", generated.shape[1] == 3, tuple(generated.shape))
        except Exception as exc:
            report["hard_failures"].append({"name": "generation", "passed": False, "detail": repr(exc), "traceback": traceback.format_exc()})

        # Backward audit with all six routers and both calls to shared middle.
        model.train()
        model.model.audit_mode = False
        model.zero_grad(set_to_none=True)
        train_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        train_out.logits.float().square().mean().backward()
        _check(report, "prefix_gradients", all(_finite_nonzero(p) for n, p in model.named_parameters() if n.startswith("model.layers.0.")), "layer0")
        _check(report, "middle_gradients", all(_finite_nonzero(p) for n, p in model.named_parameters() if n.startswith("model.layers.5.")), "shared middle")
        _check(report, "suffix_gradients", all(_finite_nonzero(p) for n, p in model.named_parameters() if n.startswith("model.layers.15.")), "layer15")
        router_params = [p for n, p in model.named_parameters() if ".write_routers." in n or ".read_routers." in n]
        _check(report, "six_router_gradients", len(router_params) == ROUTER_PARAMETER_COUNT * 2 and all(_finite_nonzero(p) for p in router_params), len(router_params))
        _check(report, "recurrence_graph_not_detached", all(p.grad_fn is not None for p in [train_out.logits]), str(train_out.logits.grad_fn))

        # Save/reload is a strict structural contract; use a temporary external directory.
        model.eval()
        with tempfile.TemporaryDirectory(prefix="mesh-stage1-") as tmp:
            tmp_path = Path(tmp)
            model.save_pretrained(tmp_path, safe_serialization=True)
            reloaded = RecursiveLlamaForCausalLM.from_pretrained(tmp_path, local_files_only=True, torch_dtype=torch.float32).to(device).eval()
            with torch.no_grad():
                reloaded_out = reloaded(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            _check(report, "save_reload_shape", tuple(reloaded_out.logits.shape) == tuple(outputs.logits.shape))
            _check(report, "save_reload_logits_close", bool(torch.allclose(reloaded_out.logits.float(), outputs.logits.float(), atol=1e-5, rtol=1e-5)))
            _check(report, "memory_not_serialized", not any("memory" in key.lower() for key in reloaded.state_dict()))
        report["status"] = "PASS" if not report["hard_failures"] else "FAIL"
    except Exception as exc:
        report["hard_failures"].append({"name": "audit_exception", "passed": False, "detail": repr(exc), "traceback": traceback.format_exc()})
        report["status"] = "FAIL"
    report["summary"] = {"checks_passed": len(report["checks"]), "hard_failures": len(report["hard_failures"]), "warnings": len(report["warnings"])}
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    _write(args.report_path, report)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
