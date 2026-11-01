#!/usr/bin/env python3
"""Generate ten text continuations with the original MeSH checkpoint and bound K.

Only run_generation imports torch/transformers. Template creation, input checks,
and recalibration of the recorded CSV use the Python standard library only.
No seven-slot modules are installed and no training modules are imported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import struct
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

RSMOL = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x2_5_mesh/"
                     "formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244")
DEFAULT_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mesh_text_router_k_5090")
MAX_NEW_TOKENS = 100
PROMPT_COUNT = 10
CONTEXT_LIMIT = 1024
SCHEDULE = tuple(range(15)) + tuple(range(5, 15)) + tuple(range(15, 20))
ROUTERS = ("write_pre", "read_pre", "write_0", "read_0", "write_1", "read_1")
TEXT_CONTRACT = "logical_30_physical_20_5_10x2_5"
TEXT_ARCHITECTURE = "logical_30_physical_20_5_10x2_5_mesh"
CSV_FIELDS = ("sample_id", "generation_step", "forward_kind", "sequence_position",
              "token_region", "token_id", "token_text", "router",
              *(f"logit_{i}" for i in range(5)), *(f"prob_{i}" for i in range(5)),
              "logsumexp")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_prompts(path):
    prompts, ids = [], set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line, object_pairs_hook=unique_keys)
            except ValueError as exc:
                raise ValueError(f"{path}: line {line_number}: {exc}") from exc
            if not isinstance(item, dict) or set(item) != {"id", "prompt"}:
                raise ValueError(f"line {line_number}: each object needs exactly id and prompt")
            if not isinstance(item["id"], str) or not item["id"].strip() or item["id"] in ids:
                raise ValueError(f"line {line_number}: id must be a unique nonempty string")
            if not isinstance(item["prompt"], str) or not item["prompt"].strip():
                raise ValueError(f"line {line_number}: prompt must be a nonempty string; fill the template first")
            ids.add(item["id"])
            prompts.append(item)
    if len(prompts) != PROMPT_COUNT:
        raise ValueError(f"expected exactly {PROMPT_COUNT} prompts, got {len(prompts)}")
    return prompts


def write_template(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for index in range(1, PROMPT_COUNT + 1):
            stream.write(json.dumps({"id": f"p{index:02d}", "prompt": ""}, ensure_ascii=False) + "\n")


def validate_config(config):
    expected = {"model_type": "llama", "hidden_size": 576, "num_hidden_layers": 30,
                "recursive_layer_count": 20, "recursive_loops": 2}
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"text checkpoint {name}: expected {value!r}, got {config.get(name)!r}")
    for canonical, alias, expected_value in (("mesh_memory_slots", "memory_slots", 5),
                                            ("mesh_router_count", "router_count", 6)):
        if config.get(canonical, config.get(alias)) != expected_value:
            raise ValueError(f"text checkpoint requires {canonical}={expected_value}")
    if config.get("mesh_architecture_contract") != TEXT_CONTRACT:
        raise ValueError("text checkpoint MeSH architecture contract mismatch")
    if config.get("model_architecture_contract") not in (None, "", TEXT_CONTRACT, TEXT_ARCHITECTURE):
        raise ValueError("text checkpoint model architecture contract mismatch")
    if tuple(config.get("logical_to_physical", config.get("logical_to_physical_schedule", ()))) != SCHEDULE:
        raise ValueError("text checkpoint logical schedule mismatch")


def check_inputs(args):
    prompts = load_prompts(args.prompts_jsonl)
    config_path = args.text_model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if not any(args.text_model_dir.glob("*.safetensors")) and not any(args.text_model_dir.glob("pytorch_model*.bin")):
        raise ValueError("text checkpoint has no local model weights")
    for name in ("tokenizer_config.json", "tokenizer.json"):
        if not (args.text_model_dir / name).is_file():
            raise ValueError(f"text checkpoint tokenizer missing {name}")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError(f"output requires a new or empty directory: {args.output_dir}")
    return prompts


def logsumexp(values):
    if len(values) != 5 or not all(math.isfinite(x) for x in values):
        raise ValueError("expected five finite original router logits")
    largest = max(values)
    return largest + math.log(sum(math.exp(x - largest) for x in values))


def new_mass(k, lse):
    """Stable combined probability of two zero-weight rows with bias=-K."""
    x = math.log(2.0) - k - lse
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def k_boundary(lse, cap):
    # At equality q=cap. A STRICT probability cap needs K above this boundary.
    return math.log(2.0) + math.log1p(-cap) - math.log(cap) - lse


def quantile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def distribution(values):
    if not values:
        raise ValueError("empty distribution")
    return {"count": len(values), "min": min(values), "mean": math.fsum(values) / len(values),
            "median": quantile(values, .5), "p95": quantile(values, .95), "max": max(values)}


def load_records(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("router CSV schema mismatch")
        records = list(reader)
    for row in records:
        for key in ("generation_step", "sequence_position", "token_id"):
            row[key] = int(row[key])
        for key in (*[f"logit_{i}" for i in range(5)], *[f"prob_{i}" for i in range(5)], "logsumexp"):
            row[key] = float(row[key])
    validate_records(records)
    return records


def validate_records(records):
    positions, sample_positions = {}, {}
    for row in records:
        if row["router"] not in ROUTERS or row["token_region"] not in ("prompt", "generated"):
            raise ValueError("invalid router or token region in records")
        if row["sequence_position"] < 0 or not row["sample_id"]:
            raise ValueError("invalid sample id or position")
        logits = [row[f"logit_{i}"] for i in range(5)]
        lse = logsumexp(logits)
        if not math.isfinite(row["logsumexp"]) or not math.isclose(lse, row["logsumexp"], abs_tol=1e-10):
            raise ValueError("recorded logsumexp disagrees with raw logits")
        probs = [row[f"prob_{i}"] for i in range(5)]
        if (not all(math.isfinite(p) and 0 <= p <= 1 for p in probs)
                or not math.isclose(math.fsum(probs), 1., abs_tol=2e-6)
                or any(abs(p - math.exp(z - lse)) > 2e-6 for p, z in zip(probs, logits))):
            raise ValueError("recorded five-slot probabilities disagree with logits")
        key = row["sample_id"], row["sequence_position"]
        seen = positions.setdefault(key, set())
        if row["router"] in seen:
            raise ValueError("duplicate router record for the same unique token position")
        seen.add(row["router"])
        identity = row["token_region"], row["token_id"]
        sample = sample_positions.setdefault(row["sample_id"], {})
        if row["sequence_position"] in sample and sample[row["sequence_position"]] != identity:
            raise ValueError("router token metadata disagrees")
        sample[row["sequence_position"]] = identity
    if len(sample_positions) != PROMPT_COUNT or any(seen != set(ROUTERS) for seen in positions.values()):
        raise ValueError("records must contain ten samples and six routers at every position")
    for sample in sample_positions.values():
        if sorted(sample) != list(range(len(sample))):
            raise ValueError("missing token positions in router records")
        regions = [sample[i][0] for i in range(len(sample))]
        if (regions[0] != "prompt" or "generated" not in regions
                or regions != sorted(regions, key=lambda x: x == "generated")
                or regions.count("generated") > MAX_NEW_TOKENS):
            raise ValueError("records need a prompt followed by 1..100 generated positions")


def calibrate(records, cap=.005, margin=1.):
    validate_records(records)
    if not 0 < cap < 1 or not math.isfinite(margin) or margin <= 0:
        raise ValueError("require 0<cap<1 and positive finite K margin")
    grouped = {name: [row for row in records if row["router"] == name] for name in ROUTERS}
    global_min = min(row["logsumexp"] for row in records)
    boundary = k_boundary(global_min, cap)
    # Keep K positive; the +margin gives strict inequality, then round UP to 0.1.
    k = max(.1, math.ceil((boundary + margin) * 10.) / 10.)
    summaries = {}
    for name, rows in grouped.items():
        worst = min(rows, key=lambda row: row["logsumexp"])
        per_router_boundary = k_boundary(worst["logsumexp"], cap)
        mass = [new_mass(k, row["logsumexp"]) for row in rows]
        if not max(mass) < cap:
            raise RuntimeError("recommended K failed the strict recorded probability cap")
        region_reports = {}
        for region in ("prompt", "generated"):
            region_rows = [row for row in rows if row["token_region"] == region]
            # A ten-sample run has both regions for every router.
            region_reports[region] = distribution([new_mass(k, row["logsumexp"]) for row in region_rows])
        summaries[name] = {
            "logsumexp": distribution([row["logsumexp"] for row in rows]),
            "old_slot_mean_probabilities": [math.fsum(row[f"prob_{i}"] for row in rows) / len(rows) for i in range(5)],
            "k_boundary_at_requested_cap": per_router_boundary,
            "k_with_margin_rounded_up": max(.1, math.ceil((per_router_boundary + margin) * 10.) / 10.),
            "combined_new_slot_probability_at_common_k": distribution(mass),
            "by_region": region_reports,
            "worst_position": {key: worst[key] for key in ("sample_id", "sequence_position", "token_region", "token_id", "logsumexp")},
        }
    boundaries = {str(limit): {
        "common_k_boundary": k_boundary(global_min, limit),
        "router_k_boundaries": {name: k_boundary(min(row["logsumexp"] for row in rows), limit) for name, rows in grouped.items()}
    } for limit in (.01, .005)}
    return {"status": "COMPLETE", "scope": "recorded_unique_text_positions_only",
            "target": "combined_probability_of_two_new_slots", "combined_probability_cap": cap,
            "comparison": "strictly_less_than", "k_logit_margin": margin, "k_rounding": "up_to_0.1",
            "recommended_common_k": k, "new_weight_rows": 0., "new_bias_each": -k,
            "new_slots_equal_each_at_initialization": True,
            "worst_combined_probability_at_common_k": max(new_mass(k, row["logsumexp"]) for row in records),
            "minimum_observed_logsumexp": global_min, "common_k_boundary_at_cap": boundary,
            "reference_boundaries": boundaries, "routers": summaries,
            "record_count": len(records), "sample_count": PROMPT_COUNT,
            "limitation": "Text calibration is an initialization reference, not a bound for future audio inputs or training."}


class RouterRecorder:
    """Read-only hooks; record each causal token position once, without audit-mode copies."""

    def __init__(self, model):
        self.model = model
        self.handles, self.captured, self.physical_trace = [], {}, []

    def __enter__(self):
        try:
            for index, name in enumerate(ROUTERS):
                routers = self.model.model.write_routers if name.startswith("write") else self.model.model.read_routers
                self.handles.append(routers[index // 2].register_forward_hook(self._hook(name)))
            for index, layer in enumerate(self.model.model.layers):
                self.handles.append(layer.register_forward_hook(self._layer_hook(index)))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.captured.clear()

    def _layer_hook(self, index):
        def capture(_module, _inputs, _output):
            self.physical_trace.append(index)
        return capture

    def _hook(self, name):
        def capture(_module, inputs, output):
            import torch
            if name in self.captured or tuple(output.shape) != (1, self.length, 5):
                raise RuntimeError(f"unexpected {name} router call/shape")
            if not inputs or inputs[0].dtype != torch.float32 or output.dtype != torch.float32:
                raise RuntimeError("router capture requires FP32")
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError(f"nonfinite router logits: {name}")
            selected = output.detach()[:, self.start:, :]
            # This is the existing _route softmax, evaluated separately without replacing its output.
            self.captured[name] = (selected[0].cpu().tolist(), torch.softmax(selected.float(), -1)[0].cpu().tolist())
        return capture

    def begin(self, length, start):
        if self.captured or not 0 <= start < length:
            raise RuntimeError("unconsumed router capture or invalid unique-position range")
        self.length, self.start, self.physical_trace = length, start, []

    def finish(self, ids, prompt_length, sample_id, step, kind, tokenizer):
        expected_trace = [{"logical_index": i, "physical_index": p} for i, p in enumerate(SCHEDULE)]
        if (self.physical_trace != list(SCHEDULE) or self.model.model.last_forward_trace != expected_trace
                or set(self.captured) != set(ROUTERS)):
            raise RuntimeError("actual physical schedule, logical schedule or router calls mismatch")
        records = []
        for name in ROUTERS:
            logits_rows, probability_rows = self.captured[name]
            for offset, (logits, probabilities) in enumerate(zip(logits_rows, probability_rows)):
                position = self.start + offset
                token_id = ids[position]
                row = {"sample_id": sample_id, "generation_step": step, "forward_kind": kind,
                       "sequence_position": position, "token_region": "prompt" if position < prompt_length else "generated",
                       "token_id": token_id, "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                       "router": name, "logsumexp": logsumexp(logits)}
                row.update({f"logit_{i}": value for i, value in enumerate(logits)})
                row.update({f"prob_{i}": value for i, value in enumerate(probabilities)})
                records.append(row)
        self.captured.clear()
        return records


def generate_one(model, tokenizer, prompt, recorder, writer, records):
    import torch
    encoded = tokenizer(prompt["prompt"], add_special_tokens=True, truncation=False, return_tensors="pt")
    text = encoded["input_ids"].to(next(model.parameters()).device)
    prompt_ids = text[0].tolist()
    prompt_length = len(prompt_ids)
    context = min(CONTEXT_LIMIT, int(model.config.max_position_embeddings))
    if not prompt_length or prompt_length + MAX_NEW_TOKENS > context:
        raise ValueError(f"{prompt['id']}: prompt has {prompt_length} tokens; prompt+100 must fit context {context}; no truncation")
    # All non-null checkpoint EOS declarations must agree. No silent stop-token override.
    eos_sets = []
    for value in (tokenizer.eos_token_id, model.config.eos_token_id, model.generation_config.eos_token_id):
        if value is not None:
            eos_sets.append(set(value if isinstance(value, (list, tuple)) else [value]))
    if not eos_sets or any(value != eos_sets[0] for value in eos_sets):
        raise ValueError("checkpoint tokenizer/model/generation-config EOS mismatch or missing EOS")
    eos_ids = eos_sets[0]
    generated, stop, started = [], "max_new_tokens", time.perf_counter()

    def observe(step, kind, start):
        recorder.begin(text.shape[1], start)
        output = model(input_ids=text, attention_mask=None, use_cache=False, return_dict=True, logits_to_keep=1)
        if (tuple(output.logits.shape[:2]) != (1, 1) or output.past_key_values is not None
                or not bool(torch.isfinite(output.logits).all())):
            raise RuntimeError("invalid last-token logits or unexpected KV cache")
        batch = recorder.finish(text[0].tolist(), prompt_length, prompt["id"], step, kind, tokenizer)
        records.extend(batch)
        writer.writerows(batch)
        return output

    for step in range(MAX_NEW_TOKENS):
        output = observe(step, "generation", 0 if step == 0 else text.shape[1] - 1)
        token = int(output.logits[0, -1].argmax().item())
        generated.append(token)
        text = torch.cat((text, text.new_tensor([[token]])), dim=1)
        if token in eos_ids:
            stop = "eos_token"
            break
    # The last generated token (including EOS) has never been a model input yet.
    # One observation-only forward records it; no 101st token is selected.
    observe(len(generated), "post_generation_observation", text.shape[1] - 1)
    return {"id": prompt["id"], "prompt": prompt["prompt"], "prompt_token_ids": prompt_ids,
            "prompt_token_count": prompt_length, "generated_token_ids": generated,
            "generated_token_count": len(generated), "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
            "generated_text_raw": tokenizer.decode(generated, skip_special_tokens=False),
            "stop_reason": stop, "eos_token_ids": sorted(eos_ids), "max_new_tokens": MAX_NEW_TOKENS,
            "forward_count": len(generated) + 1, "seconds": time.perf_counter() - started}


def router_digest(model):
    digest = hashlib.sha256()
    for router in list(model.model.write_routers) + list(model.model.read_routers):
        for parameter in router.parameters():
            values = parameter.detach().float().cpu().reshape(-1).tolist()
            digest.update(struct.pack(f"<{len(values)}f", *values))
    return digest.hexdigest()


def run_generation(args, prompts):
    import torch
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError("run generation on a submitted single-GPU compute job")
    sys.path.insert(0, str(RSMOL))
    from recursive_model_5_10x2_5_mesh import RecursiveLlamaForCausalLM, parameter_audit

    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model, loading = RecursiveLlamaForCausalLM.from_pretrained(
        args.text_model_dir, local_files_only=True, torch_dtype=torch.float32, output_loading_info=True)
    failures = {key: loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
    if failures:
        raise ValueError(f"incomplete text checkpoint load: {failures}")
    validate_config(model.config.to_dict())
    core = model.model
    routers = list(core.write_routers) + list(core.read_routers)
    if (len(core.layers) != 20 or len({id(layer) for layer in core.layers}) != 20
            or len(routers) != 6 or len({id(router) for router in routers}) != 6
            or any(tuple(router.weight.shape) != (5, 576) or tuple(router.bias.shape) != (5,) for router in routers)):
        raise ValueError("loaded model must have 20 physical layers and six distinct 576-to-5 routers")
    model = model.float().to("cuda:0").eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("all model parameters must be FP32")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.text_model_dir, local_files_only=True, use_fast=True)
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        raise ValueError("checkpoint tokenizer vocabulary and embeddings disagree")
    initial_digest = router_digest(model)
    runtime = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
               "transformers": transformers.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0), "parameter_audit": parameter_audit(model),
               "attention_implementation": getattr(model.config, "_attn_implementation", None),
               "strict_loading": loading, "router_parameters_sha256_before": initial_digest}
    write_json(args.output_dir / "runtime_audit.json", runtime)
    records, generations = [], []
    with (args.output_dir / "router_records.csv").open("x", encoding="utf-8", newline="") as stream, \
            (args.output_dir / "generations.jsonl").open("x", encoding="utf-8") as generated_stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        with torch.inference_mode(), RouterRecorder(model) as recorder:
            for index, prompt in enumerate(prompts, 1):
                print(f"[mesh-router-k] sample {index}/10 id={prompt['id']!r}", flush=True)
                generation = generate_one(model, tokenizer, prompt, recorder, writer, records)
                generations.append(generation)
                generated_stream.write(json.dumps(generation, ensure_ascii=False, allow_nan=False) + "\n")
                generated_stream.flush()
                stream.flush()
                print(f"[mesh-router-k] generated={generation['generated_token_count']} stop={generation['stop_reason']}", flush=True)
    runtime["router_parameters_sha256_after"] = router_digest(model)
    if runtime["router_parameters_sha256_after"] != initial_digest:
        raise RuntimeError("read-only capture changed original router parameters")
    runtime["checked_forwards"] = sum(g["forward_count"] for g in generations)
    runtime["physical_schedule_checked_every_forward"] = list(SCHEDULE)
    write_json(args.output_dir / "runtime_audit.json", runtime)
    return records, generations


def artifact_inventory(path):
    files = {p for pattern in ("*.json", "*.safetensors", "pytorch_model*.bin", "tokenizer.model", "merges.txt", "vocab.txt")
             for p in path.glob(pattern) if p.is_file()}
    return {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(files)}


def fenced(text):
    fence = "`" * max(3, max((len(x) + 1 for x in re.findall(r"`+", text)), default=3))
    return f"{fence}\n{text}\n{fence}"


def write_reports(output, calibration, generations):
    write_json(output / "calibration_report.json", calibration)
    k = calibration["recommended_common_k"]
    lines = ["# MeSH text router K calibration", "", f"Recommended common **K = {k:.1f}**, each new bias = **{-k:.1f}**.", "",
             f"Two new slots COMBINED must be below {calibration['combined_probability_cap']:.3%} on recorded text positions.",
             "Original model generates with five slots. Seven-slot probabilities below are offline counterfactual calculations.", "",
             "| Router | Mean combined new probability | P95 | Maximum |", "|---|---:|---:|---:|"]
    for name, summary in calibration["routers"].items():
        p = summary["combined_new_slot_probability_at_common_k"]
        lines.append(f"| {name} | {p['mean']:.6%} | {p['p95']:.6%} | {p['max']:.6%} |")
    lines += ["", "K uses the maximum required boundary over all recorded tokens and six routers,",
              f"adds a {calibration['k_logit_margin']:g}-logit margin, then rounds upward to 0.1.",
              "Each new slot initially receives half the reported combined probability.",
              "This text-only result does not bound future audio inputs or training.", ""]
    (output / "calibration_report.md").write_text("\n".join(lines), encoding="utf-8")
    if generations:
        text = ["# Original five-slot MeSH text generations", ""]
        for item in generations:
            text += [f"## {item['id']}", "", "Prompt:", fenced(item["prompt"]), "", "Raw continuation (special tokens retained):",
                     fenced(item["generated_text_raw"]), "", f"Tokens: {item['generated_token_count']}; stop: {item['stop_reason']}", ""]
        (output / "generations.md").write_text("\n".join(text), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompts-jsonl", type=Path)
    inputs.add_argument("--records-csv", type=Path, help="Offline recalibration only; never imports torch.")
    inputs.add_argument("--write-prompts-template", type=Path, help="Create ten blank prompts; refuse overwrite.")
    parser.add_argument("--text-model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fUTC")))
    parser.add_argument("--max-new-tokens", type=int, choices=(MAX_NEW_TOKENS,), default=MAX_NEW_TOKENS)
    parser.add_argument("--new-slot-mass-cap", type=float, default=.005, help="Combined mass of BOTH new slots, default 0.005=0.5%%.")
    parser.add_argument("--k-margin", type=float, default=1., help="Positive extra logit margin above the worst-case boundary.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--check-inputs", action="store_true", help="Standard-library input/config checks only; no output writes or GPU.")
    args = parser.parse_args(argv)
    if not 0 < args.new_slot_mass_cap < 1 or not math.isfinite(args.k_margin) or args.k_margin <= 0:
        parser.error("require 0<new-slot-mass-cap<1 and finite positive k-margin")
    if args.check_inputs and args.prompts_jsonl is None:
        parser.error("--check-inputs requires --prompts-jsonl")
    return args


def main(argv=None):
    args = parse_args(argv)
    owned_output = False
    try:
        if args.write_prompts_template:
            write_template(args.write_prompts_template)
            print(f"Created ten blank prompts; fill every prompt: {args.write_prompts_template}")
            return 0
        prompts = check_inputs(args) if args.prompts_jsonl else None
        if args.check_inputs:
            print("INPUT_CHECK_OK: ten prompts, checkpoint config/assets, fixed max-new-tokens=100; tokenizer length/weights are checked on GPU job startup")
            return 0
        if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
            raise ValueError(f"output requires a new or empty directory: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive ownership: concurrent runs cannot both claim this output.
        with (args.output_dir / "run_config.json").open("x", encoding="utf-8") as stream:
            config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
            config.update({"status": "RUNNING", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                           "generation": "original_five_slot_greedy_fp32_no_cache", "add_special_tokens": True,
                           "chat_template": False, "prompt_padding": False, "prompt_truncation": False})
            json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
        owned_output = True
        if args.records_csv:
            records, generations = load_records(args.records_csv), []
            config["source_records_sha256"] = sha256(args.records_csv)
        else:
            print("[mesh-router-k] hashing local checkpoint/tokenizer assets (no downloads)", flush=True)
            config["checkpoint_inventory"] = artifact_inventory(args.text_model_dir)
            config["source_prompts_sha256"] = sha256(args.prompts_jsonl)
            config["source_sha256"] = {str(path.relative_to(RSMOL)): sha256(path) for path in
                (Path(__file__).resolve(), RSMOL / "recursive_model_5_10x2_5_mesh.py",
                 RSMOL / "scripts/calibrate_mesh_text_router_k.sh", RSMOL / "run_mesh_text_router_k_5090.sh")}
            try:
                commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RSMOL, capture_output=True, text=True)
                config["git_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
            except OSError:
                config["git_commit"] = None
            with (args.output_dir / "prompts.jsonl").open("x", encoding="utf-8") as stream:
                for item in prompts:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            write_json(args.output_dir / "run_config.json", config)
            records, generations = run_generation(args, prompts)
        calibration = calibrate(records, args.new_slot_mass_cap, args.k_margin)
        write_reports(args.output_dir, calibration, generations)
        report = {"status": "COMPLETE", "mode": "offline_recalibration" if args.records_csv else "text_generation_and_calibration",
                  "sample_count": PROMPT_COUNT, "record_count": len(records), "max_new_tokens": MAX_NEW_TOKENS,
                  "recommended_common_k": calibration["recommended_common_k"], "audio_inputs_verified": False}
        write_json(args.output_dir / "generation_report.json", report)
        config["status"] = "COMPLETE"
        write_json(args.output_dir / "run_config.json", config)
        print(json.dumps({"output_dir": str(args.output_dir), **report}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        if owned_output:
            config["status"] = "FAIL"
            write_json(args.output_dir / "run_config.json", config)
            write_json(args.output_dir / "generation_report.json", {"status": "FAIL", "error": str(exc), "traceback": traceback.format_exc()})
        print(f"[mesh-router-k] FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
