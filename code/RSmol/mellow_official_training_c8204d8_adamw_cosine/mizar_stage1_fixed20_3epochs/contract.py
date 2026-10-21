"""Shared, dependency-free contract for the isolated HTSAT / Mizar S1 route."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

ROUTE = "htsat_mizar_s1_fixed20_v1"
WORLD_SIZE, BATCH_SIZE, EPOCHS, SEED = 8, 4, 3, 1234
PROMPT_TOKENS, ANSWER_TOKENS, AUDIO_TOKENS = 256, 300, 258
PREFIX_TOKENS = AUDIO_TOKENS + 1 + PROMPT_TOKENS
SOURCE_COUNTS = {
    "reasonaqa_oneaudio": 756968, "AudioCaps": 113844, "Clotho": 4798,
    "CompA-R": 249161, "LP-MusicCaps-MTT": 19632,
    "SpeechCraft": 124790, "TACOS": 42141,
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def code_digest():
    root = Path(__file__).resolve().parent
    result = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.suffix in {".py", ".sh"} and "__pycache__" not in path.parts:
            result.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
            # The source identity is independent of Git's CRLF checkout setting.
            result.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return result.hexdigest()


def geometry(rows):
    per_rank = rows // WORLD_SIZE
    steps = per_rank // BATCH_SIZE
    total = steps * EPOCHS
    if steps < 8:
        raise ValueError("manifest is too small for the requested smoke")
    return {"rows": rows, "world_size": WORLD_SIZE, "microbatch": BATCH_SIZE,
            "gradient_accumulation_steps": 1, "global_batch_size": 32,
            "samples_per_rank": per_rank, "steps_per_epoch": steps,
            "total_steps": total, "warmup_steps": round(total * 0.05),
            "dropped_rows_per_epoch": rows - steps * 32}


def lr_factor(step, total_steps):
    """Exact stage1 trainer.py warmup_cosine LambdaLR function (step zero included)."""
    warmup = max(0, min(round(total_steps * 0.05), total_steps - 1))
    if warmup > 0 and step < warmup:
        return max(0.1, (step + 1) / warmup)
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def data_order(rows):
    order = list(range(rows))
    random.Random(SEED).shuffle(order)
    return order


def rank_seed(rank):
    """Reference trainer: stochastic model/text RNG is seeded per rank."""
    return SEED + rank


def raw_references(row):
    """Recover original recordings, never the historical processed chunk paths.

    All released S1 rows have one raw recording. Nonempty historical filepath2
    is a processed second half, NOT evidence of a second original recording.
    An explicit original second recording is supported for future audited pairs.
    """
    source = next((str(row[k]).strip() for k in
                   ("source_dataset", "source", "dataset", "source_name")
                   if row.get(k)), "UNCLASSIFIED")
    key = source.lower().replace("-", "_")
    field = "filepath1_original" if key in {"reasonaqa_oneaudio", "reasonaqa", "mellow"} else "audio_path"
    raw = str(row.get(field) or "").strip()
    if not raw:
        raise ValueError(f"missing original audio field {field!r} for {source}")
    if source == "MusicCaps" or "train-other-500" in raw.replace("\\", "/"):
        raise ValueError("unfiltered MusicCaps / train-other-500 row in fixed manifest")
    refs = [(field, raw)]
    second = str(row.get("filepath2_original") or row.get("audio_path2") or "").strip()
    if second:
        second_field = "filepath2_original" if row.get("filepath2_original") else "audio_path2"
        refs.append((second_field, second))
    return source, refs


def semantic_contract(prepared, text_model_dir, htsat_root):
    """Resume/smoke identity deliberately excludes job-local staging paths."""
    return {
        "route": ROUTE, "code_sha256": code_digest(),
        "rows_canonical_sha256": prepared["rows_canonical_sha256"],
        "mapping_sha256": prepared["mapping_sha256"],
        "source_rows": prepared["source_rows"], "geometry": geometry(prepared["rows"]),
        "text_model_dir": str(text_model_dir), "htsat_root": str(htsat_root),
        "sample_rate": 32000, "audio_policy": "mono_resample_center_crop_right_pad_fixed20_split10",
        "prefix": [129, 129, 1, 256], "answer_tokens": 300,
        "attention_mask": None, "pad_token": "!", "eos_suffix": " <|endoftext|>",
        "loss": "rank_local_token_mean_then_DDP_mean",
        "adamw": {"decoder_lr": 3e-4, "map_lr": 6e-4, "betas": [0.9, 0.999],
                  "eps": 1e-8, "weight_decay": 0.01, "exclude_bias_ndim_le1": True},
        "scheduler": "Mizar_LambdaLR_warmup_cosine_5pct_floor_0.1_step_after_optimizer",
        "gradient_clip": {"type": "GradNormTracker", "initial_l2": 1., "initial_max": 10.,
                          "overdrive": 2.5, "momentum": 0.995},
        "forward_dtype": "float32", "autocast": False, "grad_scaler": True,
        "workers": 0, "sync_batchnorm": False, "seed": SEED,
        "rank_rng": "1234_plus_rank",
        "order": "python_shuffle_1234_then_torch_randperm_1234_plus_epoch_contiguous_rank_shards",
    }
