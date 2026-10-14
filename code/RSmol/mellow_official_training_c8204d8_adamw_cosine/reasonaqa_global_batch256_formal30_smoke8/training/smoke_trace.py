"""Record one actual training row per rank and optimizer step."""

import json
import math
from pathlib import Path

import torch


def _token(tokenizer, token_id):
    return tokenizer.convert_ids_to_tokens(int(token_id))


def _waveform(waveform):
    wave = waveform[0]
    return {
        "samples": int(wave.numel()),
        "nonzero_samples": int(torch.count_nonzero(wave).item()),
        "max_abs": float(wave.abs().max().item()),
        "mean_abs": float(wave.abs().mean().item()),
    }


class LmInputProbe:
    def __init__(self, expected_ids, expected_mask, audio_prefix_length):
        self.expected_ids = expected_ids[0]
        self.expected_mask = expected_mask[0]
        self.audio_prefix_length = audio_prefix_length
        self.calls = 0
        self.summary = None

    def __call__(self, lm, args, kwargs):
        self.calls += 1
        if self.calls != 1 or args or "inputs_embeds" not in kwargs:
            raise ValueError("smoke probe expected exactly one LM call with inputs_embeds")
        embeds = kwargs["inputs_embeds"]
        mask = kwargs["attention_mask"]
        prefix_len = self.audio_prefix_length
        if embeds.ndim != 3 or embeds.shape[1] != prefix_len + self.expected_ids.shape[0]:
            raise ValueError("LM input sequence differs from the collated training row")
        if mask.shape != embeds.shape[:2] or not torch.equal(mask[0, prefix_len:], self.expected_mask):
            raise ValueError("actual LM text attention mask differs from the collated mask")
        if not torch.all(mask[0, :prefix_len] == 1):
            raise ValueError("audio prefix was masked in the LM call")
        if not torch.isfinite(embeds[0]).all():
            raise ValueError("LM received non-finite input embeddings")
        # Decoder supplies embeddings, not token IDs. Check its text embeddings
        # against the actual collated token IDs without altering the forward.
        with torch.no_grad():
            expected_text = lm.model.embed_tokens(self.expected_ids)
            text_diff = float((embeds[0, prefix_len:] - expected_text).abs().max().item())
            separator = lm.model.embed_tokens(torch.zeros(1, dtype=torch.long, device=embeds.device))[0]
            sep_diff = max(
                float((embeds[0, position] - separator).abs().max().item())
                for position in (129, 259)
            )
        if not math.isclose(text_diff, 0.0, abs_tol=1e-6):
            raise ValueError(f"LM text embeddings differ from token IDs: {text_diff}")
        if not math.isclose(sep_diff, 0.0, abs_tol=1e-6):
            raise ValueError(f"LM separator embeddings differ from token 0: {sep_diff}")
        with torch.no_grad():
            self.summary = {
                "shape": list(embeds.shape),
                "attention_mask_row": mask[0].detach().cpu().tolist(),
                "text_embedding_max_abs_diff": text_diff,
                "separator_embedding_max_abs_diff": sep_diff,
                "audio_prefix": [
                    {"name": name, "positions": [lo, hi - 1],
                     "nonzero_values": int(torch.count_nonzero(embeds[0, lo:hi]).item()),
                     "l2_norm": float(torch.linalg.vector_norm(embeds[0, lo:hi].float()).item())}
                    for name, lo, hi in (
                        ("audio1", 0, 129), ("separator1", 129, 130),
                        ("audio2", 130, 259), ("separator2", 259, 260)
                    )
                ],
            }


def make_record(*, rank, step, epoch, batch, logits, probe, tokenizer,
                audio_prefix_length, global_token_count):
    if probe.calls != 1 or probe.summary is None:
        raise ValueError("LM input probe did not observe exactly one forward")
    ids = batch["input"]["input_ids"][0].tolist()
    mask = batch["input"]["attention_mask"][0].tolist()
    answer_start = int(batch["input"]["answer_start"][0])
    target_ids = batch["answer"]["input_ids"][0].tolist()
    target_mask = batch["answer"]["attention_mask"][0].tolist()
    answer_count = sum(target_mask)
    valid_length = sum(mask)
    if audio_prefix_length != 260 or answer_start < 1 or answer_count < 1:
        raise ValueError("invalid compact-text or audio-prefix geometry")
    if answer_start > 129 or answer_count > 250:
        raise ValueError("prompt or answer exceeds its token cap")
    if target_mask != [1] * answer_count + [0] * (len(target_mask) - answer_count):
        raise ValueError("answer target mask is not a valid prefix")
    if valid_length != answer_start + answer_count or mask != [1] * valid_length + [0] * (len(mask) - valid_length):
        raise ValueError("text padding does not start after answer EOS")
    if any(token_id != tokenizer.pad_token_id for token_id in ids[valid_length:]):
        raise ValueError("right-padding token ID differs from tokenizer pad ID")
    if ids[answer_start:valid_length] != target_ids[:answer_count]:
        raise ValueError("LM text does not contain the supervised answer IDs")
    if ids[answer_start - 1] != tokenizer.eos_token_id or ids[valid_length - 1] != tokenizer.eos_token_id:
        raise ValueError("prompt/answer EOS missing from actual LM input")
    if target_ids[answer_count - 1] != tokenizer.eos_token_id:
        raise ValueError("answer target EOS missing")
    if logits.shape[1] != audio_prefix_length + len(ids):
        raise ValueError("LM logits do not cover the full padded input")
    if probe.summary["attention_mask_row"] != [1] * audio_prefix_length + mask:
        raise ValueError("LM attention mask differs from trace")
    pred_ids = logits[0, audio_prefix_length:].detach().argmax(dim=-1).cpu().tolist()
    rows = []
    for i, (input_id, predicted_id) in enumerate(zip(ids, pred_ids)):
        answer_offset = i - answer_start + 1
        supervised = 0 <= answer_offset < answer_count
        rows.append({
            "text_position": i,
            "lm_position": audio_prefix_length + i,
            "role": "padding" if not mask[i] else ("prompt" if i < answer_start else "answer"),
            "input_id": input_id,
            "input_token": _token(tokenizer, input_id),
            "attention_mask": mask[i],
            "predicted_next_id": predicted_id,
            "predicted_next_token": _token(tokenizer, predicted_id),
            "loss_source": supervised,
            "loss_target_index": answer_offset if supervised else None,
            "answer_target_id": target_ids[answer_offset] if supervised else None,
            "answer_target_token": _token(tokenizer, target_ids[answer_offset]) if supervised else None,
            "prediction_matches_target": predicted_id == target_ids[answer_offset] if supervised else None,
            "causal_unmasked_key_positions": [0, audio_prefix_length + i] if mask[i] else None,
        })
    supervised_sources = [audio_prefix_length + answer_start + j - 1 for j in range(answer_count)]
    return {
        "contract": "mellow_formal30_smoke8_trace_v1",
        "rank": rank, "optimizer_step_completed": step, "epoch": epoch,
        "selected_microbatch": 0, "selected_row": 0,
        "mean_answer_token_count_per_rank_in_update": float(global_token_count),
        "audio_waveforms_received": {
            "audio1": _waveform(batch["waveform1"]),
            "audio2": _waveform(batch["waveform2"]),
            "audio1_source_path": batch["file_path1"][0],
            "audio2_source_path": batch["file_path2"][0],
            "audio2_source_missing": batch["file_path2"][0] == "",
        },
        "tokenizer_ids": {
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "audio_separator_token_id": 0,
        },
        "mask_interpretation": "A token is excluded only where attention_mask=0; an ordinary literal with the same ID as pad_token_id remains active when its mask is 1.",
        "lm_input": probe.summary,
        "attention_rule": "For valid query positions, causal key positions <= query position with attention_mask=1; padded query outputs are not used for loss",
        "attention_ranges": {"audio_prefix": [0, 259],
                             "prompt": [260, 260 + answer_start - 1],
                             "answer": [260 + answer_start, 260 + valid_length - 1],
                             "padding": [260 + valid_length, 260 + len(ids) - 1] if valid_length < len(ids) else None},
        "loss_source_positions": supervised_sources,
        "loss_target_ids": target_ids[:answer_count],
        "text_tokens": rows,
    }


def append_record(output_dir, record):
    trace_dir = Path(output_dir) / "diagnostics"
    trace_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    with (trace_dir / f"rank_{record['rank']}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
    print("[mellow-smoke8] " + payload, flush=True)
