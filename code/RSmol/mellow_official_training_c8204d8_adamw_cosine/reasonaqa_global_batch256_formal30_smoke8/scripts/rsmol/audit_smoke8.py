#!/usr/bin/env python3
"""Audit the eight-step trace and its complete training checkpoint."""

import argparse
import json
import math
from pathlib import Path


def audit_records(output_dir):
    diagnostics = output_dir / "diagnostics"
    total = 0
    for rank in range(8):
        path = diagnostics / f"rank_{rank}.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(records) != 8:
            raise ValueError(f"{path}: expected 8 records, got {len(records)}")
        for step, record in enumerate(records, 1):
            if (record["contract"] != "mellow_formal30_smoke8_trace_v1"
                    or record["rank"] != rank or record["optimizer_step_completed"] != step
                    or record["selected_microbatch"] != 0 or record["selected_row"] != 0):
                raise ValueError(f"rank {rank}, step {step}: wrong trace identity")
            rows = record["text_tokens"]
            mask = record["lm_input"]["attention_mask_row"]
            if len(mask) != 260 + len(rows) or mask[:260] != [1] * 260:
                raise ValueError(f"rank {rank}, step {step}: bad prefix mask")
            if [row["attention_mask"] for row in rows] != mask[260:]:
                raise ValueError(f"rank {rank}, step {step}: token mask mismatch")
            if not rows or any(row["text_position"] != i or row["lm_position"] != 260 + i
                               for i, row in enumerate(rows)):
                raise ValueError(f"rank {rank}, step {step}: token positions are not contiguous")
            prompt_count = sum(row["role"] == "prompt" for row in rows)
            answer_count = sum(row["role"] == "answer" for row in rows)
            pad_count = sum(row["role"] == "padding" for row in rows)
            if (prompt_count < 1 or answer_count < 1
                    or [row["role"] for row in rows] !=
                    ["prompt"] * prompt_count + ["answer"] * answer_count + ["padding"] * pad_count
                    or mask[260:] != [1] * (prompt_count + answer_count) + [0] * pad_count):
                raise ValueError(f"rank {rank}, step {step}: text mask or right padding differs")
            eos_id = record["tokenizer_ids"]["eos_token_id"]
            pad_id = record["tokenizer_ids"]["pad_token_id"]
            if (rows[prompt_count - 1]["input_id"] != eos_id
                    or rows[prompt_count + answer_count - 1]["input_id"] != eos_id
                    or any(row["input_id"] != pad_id for row in rows[prompt_count + answer_count:])):
                raise ValueError(f"rank {rank}, step {step}: EOS or padding token differs")
            expected_sources = list(range(260 + prompt_count - 1,
                                          260 + prompt_count + answer_count - 1))
            if ([row["loss_source"] for row in rows] !=
                    [False] * (prompt_count - 1) + [True] * answer_count +
                    [False] * (pad_count + 1)):
                raise ValueError(f"rank {rank}, step {step}: loss source positions differ")
            sources = [row["lm_position"] for row in rows if row["loss_source"]]
            targets = [row["answer_target_id"] for row in rows if row["loss_source"]]
            answer_ids = [row["input_id"] for row in rows[prompt_count:prompt_count + answer_count]]
            if (sources != expected_sources or sources != record["loss_source_positions"]
                    or targets != answer_ids or targets != record["loss_target_ids"]
                    or [row["loss_target_index"] for row in rows if row["loss_source"]] != list(range(answer_count))):
                raise ValueError(f"rank {rank}, step {step}: loss alignment mismatch")
            if not sources or any(b != a + 1 for a, b in zip(sources, sources[1:])):
                raise ValueError(f"rank {rank}, step {step}: noncontiguous loss positions")
            if record["attention_ranges"]["audio_prefix"] != [0, 259]:
                raise ValueError(f"rank {rank}, step {step}: wrong audio prefix")
            if (record["attention_ranges"]["prompt"] != [260, 260 + prompt_count - 1]
                    or record["attention_ranges"]["answer"] !=
                    [260 + prompt_count, 260 + prompt_count + answer_count - 1]
                    or record["attention_ranges"]["padding"] !=
                    ([260 + prompt_count + answer_count, 259 + len(rows)] if pad_count else None)):
                raise ValueError(f"rank {rank}, step {step}: attention ranges differ")
            if any(row["causal_unmasked_key_positions"] != [0, row["lm_position"]]
                   for row in rows if row["attention_mask"]):
                raise ValueError(f"rank {rank}, step {step}: causal coverage differs")
            if any(row["causal_unmasked_key_positions"] is not None
                   for row in rows if not row["attention_mask"]):
                raise ValueError(f"rank {rank}, step {step}: padded query coverage differs")
            audio = record["audio_waveforms_received"]
            if audio["audio1"]["samples"] != 320000 or audio["audio2"]["samples"] != 320000:
                raise ValueError(f"rank {rank}, step {step}: wrong audio waveform length")
            if audio["audio2_source_missing"] and audio["audio2"]["nonzero_samples"] != 0:
                raise ValueError(f"rank {rank}, step {step}: missing second audio is not silent")
            if (not math.isfinite(record["global_mean_loss"]) or record["lr_used"] <= 0
                    or not math.isfinite(record["gradient_l2_norm_before_global_clip"])
                    or record["gradient_l2_norm_before_global_clip"] <= 0
                    or record["global_gradient_clip_max_norm"] != 0.5):
                raise ValueError(f"rank {rank}, step {step}: invalid optimizer evidence")
            total += 1
    return total


def audit_checkpoint(path):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_geometry = {"per_rank_batch_size": 8, "world_size": 8, "gradient_accumulation_steps": 4}
    if checkpoint.get("schema_version") != 2 or checkpoint.get("total_step") != 8:
        raise ValueError("checkpoint is not a completed step-8 V2 training checkpoint")
    if checkpoint.get("num_epochs") != 30 or checkpoint.get("batch_geometry") != expected_geometry:
        raise ValueError("checkpoint training horizon or batch geometry differs")
    if checkpoint.get("route_variant") != "formal_30epochs_compact":
        raise ValueError("checkpoint route differs")
    if checkpoint.get("loss_reduction") != "global_token_mean":
        raise ValueError("checkpoint loss reduction differs")
    if checkpoint.get("attention_mask_policy") != "compact_text_right_pad_after_answer_eos":
        raise ValueError("checkpoint mask policy differs")
    if checkpoint.get("htsat_freeze_policy") != "eval_no_grad_base_only" or checkpoint.get("audio_slot_policy") != "zero_if_missing_audio2":
        raise ValueError("checkpoint audio policy differs")
    if checkpoint.get("gradient_clip_policy") != "global_l2_hard_clip" or checkpoint.get("max_grad_norm") != 0.5:
        raise ValueError("checkpoint clip policy differs")
    scheduler = checkpoint["scheduler"]
    if scheduler["last_step"] != 8 or scheduler["max_lr"] != 2e-4 or scheduler["min_lr"] != 1e-5:
        raise ValueError("checkpoint scheduler differs")
    if scheduler["total_steps"] <= 8 or scheduler["warmup_steps"] != math.ceil(scheduler["total_steps"] * 0.05):
        raise ValueError("checkpoint does not retain the 30-epoch schedule")
    if len(checkpoint.get("random_state_by_rank", [])) != 8:
        raise ValueError("checkpoint is missing rank states")
    optimizer = checkpoint.get("optimizer", {})
    if not optimizer.get("state"):
        raise ValueError("checkpoint optimizer did not acquire updated state")
    for state in optimizer["state"].values():
        if "step" in state and int(state["step"]) != 8:
            raise ValueError("AdamW state does not record eight parameter updates")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    count = audit_records(args.output_dir)
    audit_checkpoint(args.checkpoint)
    report = {"status": "PASS", "contract": "mellow_formal30_smoke8_trace_v1",
              "rank_count": 8, "optimizer_steps": 8, "sample_records": count,
              "checkpoint": str(args.checkpoint)}
    target = args.output_dir / "diagnostics" / "smoke8_audit.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
