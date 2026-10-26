"""Shared, dependency-free contract for the isolated HTSAT / Mizar S1 route."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

ROUTE = "htsat_mizar_s1_gbs256_recursive_5_10_5_core_sft_2epochs_v1"
SOURCE_ROUTE = "htsat_mizar_s1_fixed20_gbs256_recursive_5_10_5_v1"
SOURCE_STEP = 17313
DEFAULT_INIT_CHECKPOINT = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "mizar_s1_htsat_gbs256_recursive_5_10_5_5090/"
    "formal_20261025_160348996763174/formal/checkpoints/model--step-17313.ckpt"
)
MAX_LR, MIN_LR = 5e-5, 5e-6
TEXT_MODEL_CONTRACT = "logical_30_physical_20_5_10_5_loops_2_no_mesh"
TEXT_MODEL_ARCHITECTURE = TEXT_MODEL_CONTRACT
TEXT_MODEL_SOURCE_MAPPING = (
    0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,
    25, 26, 27, 28, 29,
)
DEFAULT_TEXT_MODEL_DIR = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/"
    "formal-epoch2-continue-20260902_184936/checkpoint-step-009244"
)
TEXT_MODEL_SCHEDULE = tuple(
    list(range(0, 15)) + list(range(5, 15)) + list(range(15, 20))
)
WORLD_SIZE, BATCH_SIZE, ACCUMULATION_STEPS, EPOCHS, SEED = 8, 8, 4, 2, 1234
RANK_UPDATE_SAMPLES = BATCH_SIZE * ACCUMULATION_STEPS
GLOBAL_BATCH_SIZE = WORLD_SIZE * RANK_UPDATE_SAMPLES
PROMPT_TOKENS, ANSWER_TOKENS, AUDIO_TOKENS = 256, 300, 258
PREFIX_TOKENS = AUDIO_TOKENS + 1 + PROMPT_TOKENS
PAD_TOKEN, SEPARATOR_TOKEN_ID, EOS_TOKEN_ID = "!", 0, 0
SOURCE_COUNTS = {
    "reasonaqa_oneaudio": 756968, "AudioCaps": 113844, "Clotho": 4798,
    "MusicCaps": 3221, "CompA-R": 249161, "LP-MusicCaps-MTT": 19632,
    "SpeechCraft": 287727, "TACOS": 42141,
}


def validate_tokenizer_contract(tokenizer, pad_id, embedding_rows):
    """Validate the Mizar text-token contract without assuming PAD aliases EOS.

    Mizar registers the existing token ``!`` as padding.  In the checked
    SmolLM2 vocabulary that token is ID 17; ID 0 remains both EOS and the
    single audio/text separator.  Derive the padding ID from the tokenizer so
    this remains valid for an unchanged compatible tokenizer, while requiring
    it to be distinct from the separator/EOS token.
    """
    tokenizer_rows = len(tokenizer)
    tokenizer_pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    # This intentionally mirrors AudioTextDataset and the released Mizar
    # trainer, which derive ignore_index with encode(tokenizer.pad_token)[0].
    encoded_pad = tokenizer.encode(tokenizer.pad_token)
    audit = {
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_id": int(pad_id),
        "tokenizer_pad_id": tokenizer_pad_id,
        "sep_id": SEPARATOR_TOKEN_ID,
        "eos_id": eos_id,
        "encoded_pad_ids": encoded_pad,
        "tokenizer_vocab_rows": tokenizer_rows,
        "embedding_rows": int(embedding_rows),
    }
    problems = []
    if audit["pad_token"] != PAD_TOKEN:
        problems.append(f"pad_token={audit['pad_token']!r}, expected {PAD_TOKEN!r}")
    if audit["pad_id"] != tokenizer_pad_id:
        problems.append(f"dataset pad_id={audit['pad_id']}, tokenizer.pad_token_id={tokenizer_pad_id}")
    if not encoded_pad or encoded_pad[0] != audit["pad_id"]:
        problems.append(f"encode({PAD_TOKEN!r})={encoded_pad}, expected first ID {audit['pad_id']}")
    if audit["pad_id"] in {SEPARATOR_TOKEN_ID, EOS_TOKEN_ID}:
        problems.append(f"pad_id={audit['pad_id']} aliases SEP/EOS={SEPARATOR_TOKEN_ID}")
    if eos_id != EOS_TOKEN_ID:
        problems.append(f"eos_token_id={eos_id}, expected {EOS_TOKEN_ID}")
    if tokenizer_rows != embedding_rows:
        problems.append(f"tokenizer rows={tokenizer_rows}, embedding rows={embedding_rows}")
    if not 0 <= audit["pad_id"] < embedding_rows:
        problems.append(f"pad_id={audit['pad_id']} is outside embedding rows={embedding_rows}")
    if problems:
        details = ", ".join(f"{key}={value!r}" for key, value in audit.items())
        raise ValueError("invalid Mizar tokenizer contract: " + "; ".join(problems) + "; " + details)
    return audit


def validate_checkpoint_tokenizer(tokenizer, model_vocab_size):
    """Apply and audit Mizar's exact tokenizer padding operation.

    Recursive checkpoints commonly serialize EOS as their source padding
    token.  Mizar unconditionally calls ``add_special_tokens`` with the
    existing ``!`` vocabulary token, so the source PAD identity is
    informational.  The important constraints are the post-override
    ``!``/PAD contract, EOS=0, and no vocabulary growth.
    """
    try:
        model_vocab_size = int(model_vocab_size)
    except (TypeError, ValueError):
        model_vocab_size = -1
    source_rows = len(tokenizer)
    source_pad_id = getattr(tokenizer, "pad_token_id", None)
    source_pad_token = getattr(tokenizer, "pad_token", None)
    source_encoded_pad = tokenizer.encode(PAD_TOKEN, add_special_tokens=False)
    added_tokens = tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer_rows = len(tokenizer)
    # Fast tokenizers can return 1 when registering an existing vocabulary
    # token as special, while keeping both its ID and vocabulary size intact.
    # The API return value is diagnostic, not a vocabulary-size delta.
    vocabulary_growth = tokenizer_rows - source_rows
    # Keep this call identical to AudioTextDataset/Mizar rather than relying
    # on a separately derived token ID.  The no-special-token encoding below
    # separately proves that ``!`` is one existing vocabulary token.
    pad_encoding = tokenizer.encode(tokenizer.pad_token)
    if not pad_encoding:
        raise ValueError("Mizar pad token encoding is empty")
    pad_id = pad_encoding[0]
    encoded_pad = tokenizer.encode(PAD_TOKEN, add_special_tokens=False)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    audit = {
        "source": "recursive_checkpoint",
        "source_pad_token": source_pad_token,
        "source_pad_id": source_pad_id,
        "source_mizar_pad_ids": source_encoded_pad,
        "mizar_pad_token": PAD_TOKEN,
        "mizar_pad_id": pad_id,
        "mizar_pad_ids": encoded_pad,
        "mizar_pad_encoding": pad_encoding,
        "added_tokens": int(added_tokens),
        "vocabulary_growth": vocabulary_growth,
        "eos_id": eos_id,
        "source_vocab_rows": source_rows,
        "tokenizer_vocab_rows": tokenizer_rows,
        "model_vocab_size": model_vocab_size,
        "existing_pad_contract": (
            vocabulary_growth == 0 and source_encoded_pad == encoded_pad == [pad_id]
        ),
    }
    problems = []
    if eos_id != EOS_TOKEN_ID:
        problems.append(f"eos_token_id={eos_id!r}, expected {EOS_TOKEN_ID}")
    if model_vocab_size <= 0 or tokenizer_rows != model_vocab_size:
        problems.append(
            f"tokenizer rows={tokenizer_rows}, model vocab_size={model_vocab_size}"
        )
    if vocabulary_growth != 0:
        problems.append(
            f"Mizar pad assignment changed vocabulary rows from {source_rows} to {tokenizer_rows}; "
            f"expected existing {PAD_TOKEN!r} (add_special_tokens returned {added_tokens})"
        )
    if source_encoded_pad != encoded_pad:
        problems.append(
            f"Mizar pad assignment changed {PAD_TOKEN!r} token IDs from "
            f"{source_encoded_pad!r} to {encoded_pad!r}"
        )
    if tokenizer.pad_token != PAD_TOKEN:
        problems.append(f"pad_token={tokenizer.pad_token!r}, expected {PAD_TOKEN!r}")
    if len(encoded_pad) != 1 or encoded_pad != [pad_id]:
        problems.append(
            f"encode({PAD_TOKEN!r})={encoded_pad!r}, expected [{pad_id}]"
        )
    elif not 0 <= int(pad_id) < min(tokenizer_rows, model_vocab_size):
        problems.append(f"{PAD_TOKEN!r} token ID {pad_id} is outside the model vocabulary")
    elif int(pad_id) == EOS_TOKEN_ID:
        problems.append(f"{PAD_TOKEN!r} token ID aliases EOS/SEP={EOS_TOKEN_ID}")
    if source_pad_id is not None and not 0 <= int(source_pad_id) < tokenizer_rows:
        problems.append(f"source pad_token_id={source_pad_id!r} is outside the tokenizer vocabulary")
    if problems:
        raise ValueError("invalid recursive checkpoint tokenizer: " + "; ".join(problems))
    return audit


def validate_recursive_config_values(config):
    """Validate the converted fixed 5+10x2+5 model without importing Torch."""
    def integer(name, default=-1):
        try:
            return int(config.get(name, default))
        except (TypeError, ValueError):
            return default

    geometry = {
        "hidden_size": integer("hidden_size"),
        "logical_layers": integer("num_hidden_layers"),
        "physical_layers": integer("recursive_layer_count"),
        "recursive_loops": integer("recursive_loops"),
        "prefix_layers": integer("recursive_prefix_layer_count"),
        "middle_layers": integer("recursive_middle_layer_count"),
        "suffix_layers": integer("recursive_suffix_layer_count"),
        "memory_slots": 0,
        "routers": 0,
    }
    expected = {"hidden_size": 576, "logical_layers": 30, "physical_layers": 20,
                "recursive_loops": 2, "prefix_layers": 5, "middle_layers": 10,
                "suffix_layers": 5}
    problems = [f"{name}={geometry[name]} expected {value}"
                for name, value in expected.items() if geometry[name] != value]
    if config.get("model_type") != "llama":
        problems.append("model_type must be llama")
    if config.get("recursive_loops_scope") != "middle_only":
        problems.append("recursive_loops_scope must be middle_only")
    schedule = config.get("logical_to_physical") or config.get("logical_to_physical_schedule")
    if list(schedule or ()) != list(TEXT_MODEL_SCHEDULE):
        problems.append("logical_to_physical schedule mismatch")
    for name in ("logical_to_physical_schedule", "recursive_logical_to_physical",
                 "recursive_logical_to_physical_schedule"):
        if name in config and list(config[name] or ()) != list(TEXT_MODEL_SCHEDULE):
            problems.append(f"{name} contradicts the fixed schedule")
    mapping = config.get("recursive_source_layer_indices_0based")
    if list(mapping or ()) != list(TEXT_MODEL_SOURCE_MAPPING):
        problems.append("recursive source-layer mapping mismatch")
    for name in ("memory_slots", "mesh_memory_slots", "router_count", "mesh_router_count"):
        if integer(name, 0) != 0:
            problems.append(f"non-MeSH checkpoint must not enable {name}")
    if config.get("mesh_architecture_contract") or config.get("mesh_enabled"):
        problems.append("MeSH checkpoint is not a fixed non-MeSH initialization")
    architectures = config.get("architectures") or []
    if any(name not in {"RecursiveLlamaForCausalLM", "RecursiveLlama5_10_5ForCausalLM"}
           for name in architectures):
        problems.append(f"unexpected non-MeSH architectures={architectures!r}")
    if integer("vocab_size") <= 0:
        problems.append("vocab_size must be positive")
    eos = config.get("eos_token_id")
    if isinstance(eos, list):
        eos = eos[0] if len(eos) == 1 else None
    if eos != EOS_TOKEN_ID:
        problems.append(f"eos_token_id={eos!r} expected {EOS_TOKEN_ID}")
    if problems:
        raise ValueError("invalid non-MeSH 5-10-5 config: " + "; ".join(problems))
    return geometry


def recursive_checkpoint_contract(checkpoint):
    """Audit the CPU-visible Hugging Face contract before model construction."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    config_path = checkpoint / "config.json"
    if not checkpoint.is_dir() or not config_path.is_file():
        raise FileNotFoundError(f"recursive text checkpoint/config.json missing: {checkpoint}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid recursive text config: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError("recursive text config.json must contain an object")

    geometry = validate_recursive_config_values(config)
    problems = []

    tokenizer_names = {
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "tokenizer.model", "spiece.model", "vocab.json", "merges.txt",
    }
    tokenizer_files = sorted(
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_file() and path.name in tokenizer_names
    )
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
    weight_files = sorted(
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_file() and path.suffix in weight_suffixes
        and ("model" in path.name or "pytorch" in path.name or path.suffix == ".safetensors")
    )
    if not weight_files:
        problems.append("no recursive model weight files")
    if not tokenizer_files:
        problems.append("no tokenizer files in recursive checkpoint")
    if problems:
        raise ValueError("invalid recursive text checkpoint contract: " + "; ".join(problems))
    return {
        "checkpoint": str(checkpoint),
        "config": {
            "model_type": config.get("model_type"),
            "model_architecture_contract": TEXT_MODEL_ARCHITECTURE,
            "hidden_size": geometry["hidden_size"],
            "logical_layers": geometry["logical_layers"],
            "physical_layers": geometry["physical_layers"],
            "recursive_loops": geometry["recursive_loops"],
            "recursive_loops_scope": config["recursive_loops_scope"],
            "source_layer_indices_0based": list(TEXT_MODEL_SOURCE_MAPPING),
            "memory_slots": geometry["memory_slots"],
            "routers": geometry["routers"],
            "logical_to_physical": list(TEXT_MODEL_SCHEDULE),
            "vocab_size": config.get("vocab_size"),
            "eos_token_id": config.get("eos_token_id"),
            "pad_token_id": config.get("pad_token_id"),
        },
        "tokenizer_files": tokenizer_files,
        "weight_files": weight_files,
        "tokenizer_contract": {
            "pad_token": PAD_TOKEN,
            "separator_token_id": SEPARATOR_TOKEN_ID,
            "eos_token_id": EOS_TOKEN_ID,
            "comparison": "runtime AutoTokenizer audit against recursive checkpoint files",
        },
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            # Windows readers briefly deny replacing an open file. Linux
            # renames remain atomic immediately; never hide a real Unix error.
            if os.name != "nt" or attempt == 19:
                raise
            time.sleep(.01)


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


def recursive_model_source_sha256():
    """Include the shared recursive implementation in resume identity."""
    path = Path(__file__).resolve().parents[2] / "recursive_model_5_10_5.py"
    if not path.is_file():
        raise FileNotFoundError(f"recursive model implementation is missing: {path}")
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def geometry(rows):
    per_rank = rows // WORLD_SIZE
    steps = per_rank // RANK_UPDATE_SAMPLES
    total = steps * EPOCHS
    if steps < 1:
        raise ValueError("manifest is too small for one optimizer update per epoch")
    return {"rows": rows, "world_size": WORLD_SIZE, "microbatch": BATCH_SIZE,
            "gradient_accumulation_steps": ACCUMULATION_STEPS, "global_batch_size": GLOBAL_BATCH_SIZE,
            "rank_samples_per_optimizer_step": RANK_UPDATE_SAMPLES,
            "samples_per_rank": per_rank, "steps_per_epoch": steps,
            "total_steps": total, "warmup_steps": round(total * 0.05),
            "dropped_rows_per_epoch": rows - steps * GLOBAL_BATCH_SIZE}


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
    refs = [(field, raw)]
    second = str(row.get("filepath2_original") or row.get("audio_path2") or "").strip()
    if second:
        second_field = "filepath2_original" if row.get("filepath2_original") else "audio_path2"
        refs.append((second_field, second))
    return source, refs


def semantic_contract(prepared, text_model_dir, htsat_root):
    """Resume identity deliberately excludes job-local staging paths."""
    return {
        "route": ROUTE, "code_sha256": code_digest(),
        "recursive_model_source_sha256": recursive_model_source_sha256(),
        "rows_canonical_sha256": prepared["rows_canonical_sha256"],
        "mapping_sha256": prepared["mapping_sha256"],
        "source_rows": prepared["source_rows"], "geometry": geometry(prepared["rows"]),
        "text_model_dir": str(text_model_dir), "text_model_contract": TEXT_MODEL_CONTRACT,
        "text_model_architecture": TEXT_MODEL_ARCHITECTURE,
        "text_model_schedule": list(TEXT_MODEL_SCHEDULE), "htsat_root": str(htsat_root),
        "sample_rate": 32000, "audio_policy": "mono_resample_center_crop_right_pad_fixed20_split10",
        "audio_read_policy": "shared_until_complete_collective_optimizer_boundary_switch",
        "prefix": [129, 129, 1, 256], "answer_tokens": 300,
        "attention_mask": None, "pad_token": "!", "eos_suffix": " <|endoftext|>",
        "loss": "rank_local_token_mean_then_DDP_mean",
        "adamw": {"core_max_lr": MAX_LR, "core_min_lr": MIN_LR, "betas": [0.9, 0.999],
                  "eps": 1e-8, "weight_decay": 0.01, "exclude_bias_ndim_le1": True},
        "scheduler": "Mizar_LambdaLR_warmup_cosine_5pct_floor_0.1_step_after_optimizer",
        "gradient_clip": {"type": "GradNormTracker", "initial_l2": 1., "initial_max": 10.,
                          "overdrive": 2.5, "momentum": 0.995},
        "forward_dtype": "float32", "autocast": False, "grad_scaler": True,
        "workers": 0, "sync_batchnorm": False, "seed": SEED,
        "rank_rng": "1234_plus_rank",
        "order": "python_shuffle_1234_then_torch_randperm_1234_plus_epoch_contiguous_rank_shards",
        "gradient_accumulation": {"steps": ACCUMULATION_STEPS,
                                  "loss": "mean_of_rank_local_microbatch_token_means"},
        "initialization": {"source_route": SOURCE_ROUTE, "source_step": SOURCE_STEP,
                           "model_tensors_only": True,
                           "optimizer_scheduler_scaler_tracker_rng_restarted": True},
        "trainable": ["physical_layers_5_to_14"],
        "frozen": ["HTSAT", "c2l", "projection", "embed_tokens", "physical_layers_0_to_4",
                   "physical_layers_15_to_19", "final_norm", "lm_head"],
        "frozen_module_mode": "eval_including_dropout",
        "architecture": "original_non_mesh_5_10_5_middle_only_two_loops",
    }
