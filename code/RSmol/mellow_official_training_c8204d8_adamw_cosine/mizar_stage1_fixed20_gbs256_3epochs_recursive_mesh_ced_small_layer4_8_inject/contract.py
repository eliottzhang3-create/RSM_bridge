"""Shared, dependency-free contract for the isolated CED-Small / MeSH Mizar S1 route."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

ROUTE = "ced_small_mizar_s1_fixed20_gbs256_recursive_mesh_layer4_8_inject_v1"
INJECTION_CONTRACT = {
    "ced_layers_1based": [4, 8], "capture": "complete_block_output_before_final_norm",
    "logical_indices_0based": [5, 15], "read_router_indices_0based": [0, 1],
    "placement": "after_memory_read_before_middle_loop", "audio_positions": [0, 126],
    "grid": [126, 4, 384], "mlp": [1536, 576, 576],
    "mlp_structure": "LayerNorm_Linear_GELU_Linear_LayerNorm",
    "mlp_dropout": 0.0, "independent_mlps": True, "independent_gates": True,
    "gate": "tanh_Linear_576_1_per_token", "gate_weight_init": 0.0, "gate_bias_init": 0.0,
    "max_lr": 6e-4, "temporal_interpolation": False,
    "extra_position_parameters": False, "encoder_forwards_per_batch": 1,
}
TEXT_MODEL_CONTRACT = "logical_30_physical_20_5_10x2_5"
TEXT_MODEL_ARCHITECTURE = "logical_30_physical_20_5_10x2_5_mesh"
TEXT_MODEL_SCHEDULE = tuple(
    list(range(0, 15)) + list(range(5, 15)) + list(range(15, 20))
)
WORLD_SIZE, BATCH_SIZE, ACCUMULATION_STEPS, EPOCHS, SEED = 8, 8, 4, 3, 1234
RANK_UPDATE_SAMPLES = BATCH_SIZE * ACCUMULATION_STEPS
GLOBAL_BATCH_SIZE = WORLD_SIZE * RANK_UPDATE_SAMPLES
PROMPT_TOKENS, ANSWER_TOKENS, AUDIO_TOKENS = 256, 300, 126
PREFIX_TOKENS = AUDIO_TOKENS + 1 + PROMPT_TOKENS
PAD_TOKEN, SEPARATOR_TOKEN_ID, EOS_TOKEN_ID = "!", 0, 0
CED_ASSET_FILES = ("config.json", "configuration_ced.py", "modeling_ced.py",
                   "feature_extraction_ced.py", "preprocessor_config.json", "model.safetensors")
CED_EXPECTED_CONFIG = {
    "embed_dim": 384, "depth": 12, "num_heads": 6, "n_mels": 64,
    "patch_size": 16, "patch_stride": 16, "target_length": 1012,
    "n_fft": 512, "win_size": 512, "hop_size": 160,
    "center": True, "f_min": 0, "f_max": 8000,
    "pooling": "mean", "pad_last": True, "outputdim": 527,
}
SOURCE_COUNTS = {
    "reasonaqa_oneaudio": 756968, "AudioCaps": 113844, "Clotho": 4798,
    "MusicCaps": 3221, "CompA-R": 249161, "LP-MusicCaps-MTT": 19632,
    "SpeechCraft": 287727, "TACOS": 42141,
}


def ced_label_metadata(config):
    """Read the real local labels, retaining duplicate display names by index."""
    raw = config.get("id2label")
    error = "incompatible CED-Small assets: config.json needs complete local id2label metadata (indices 0..526)"
    if not isinstance(raw, dict) or len(raw) != 527:
        raise ValueError(error)
    labels = {}
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            raise ValueError(error) from None
        if str(index) != str(key) or not isinstance(value, str) or not value.strip():
            raise ValueError(error)
        labels[index] = value
    if set(labels) != set(range(527)):
        raise ValueError(error)
    labels = dict(sorted(labels.items()))
    reverse = config.get("label2id")
    if reverse is None:
        reverse = {label: index for index, label in labels.items()}
    if (not isinstance(reverse, dict) or set(reverse) != set(labels.values())
            or any(type(index) is not int or index not in labels or labels[index] != label
                   for label, index in reverse.items())):
        raise ValueError("incompatible CED-Small assets: inconsistent local label2id metadata")
    return labels, dict(reverse)


def validate_ced_config(config):
    """Reject incompatible feature geometry instead of reshaping/truncating it."""
    if not isinstance(config, dict):
        raise ValueError("incompatible CED-Small assets: config.json must contain an object")
    problems = [f"{name}={config.get(name)!r}, expected {expected!r}"
                for name, expected in CED_EXPECTED_CONFIG.items()
                if config.get(name) != expected]
    if config.get("model_type") != "ced":
        problems.append(f"model_type={config.get('model_type')!r}, expected 'ced'")
    auto_map = config.get("auto_map", {})
    if not isinstance(auto_map, dict):
        raise ValueError("incompatible CED-Small assets: auto_map must contain an object")
    for name, expected in (("AutoConfig", "configuration_ced.CedConfig"),
                           ("AutoModelForAudioClassification", "modeling_ced.CedForAudioClassification")):
        if auto_map.get(name) != expected:
            problems.append(f"auto_map.{name}={auto_map.get(name)!r}, expected {expected!r}")
    if problems:
        raise ValueError("incompatible CED-Small assets: " + "; ".join(problems))
    labels, _ = ced_label_metadata(config)
    return {**CED_EXPECTED_CONFIG, "input_sampling_rate": 32000, "ced_sampling_rate": 16000,
            "mel_frames": 2001, "segments": 2, "hidden_tokens_per_segment": 252,
            "hidden_tokens": 504, "temporal_tokens": AUDIO_TOKENS, "output_dim": 576,
            "mapper": "freq_merge_126", "mapper_dropout": 0.05, "mapper_resample": False,
            "label_metadata": "local_config_json", "label_count": len(labels)}


def validate_ced_assets(root):
    root = Path(root).resolve()
    for name in CED_ASSET_FILES:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing/empty local CED asset: {path}")
    return validate_ced_config(json.loads((root / "config.json").read_text(encoding="utf-8")))


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

    def integer(name, default=-1):
        value = config.get(name, config.get(f"mesh_{name}", default))
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    geometry = {
        "hidden_size": integer("hidden_size"),
        "logical_layers": integer("num_hidden_layers"),
        "physical_layers": integer("recursive_layer_count"),
        "recursive_loops": integer("recursive_loops"),
        "memory_slots": integer("memory_slots"),
        "routers": integer("router_count"),
    }
    # The converter has used both direct and mesh-prefixed names for the two
    # latter fields.  Accept only the same values, never a missing alias.
    geometry["memory_slots"] = integer("memory_slots")
    geometry["routers"] = integer("router_count")
    expected_geometry = {
        "hidden_size": 576,
        "logical_layers": 30,
        "physical_layers": 20,
        "recursive_loops": 2,
        "memory_slots": 5,
        "routers": 6,
    }
    problems = [f"{key}={geometry[key]} expected {value}"
                for key, value in expected_geometry.items() if geometry[key] != value]
    schedule = config.get("logical_to_physical") or config.get("logical_to_physical_schedule")
    if list(schedule or ()) != list(TEXT_MODEL_SCHEDULE):
        problems.append("logical_to_physical schedule mismatch")
    mesh_contract = config.get("mesh_architecture_contract")
    if mesh_contract != TEXT_MODEL_CONTRACT:
        problems.append(f"mesh_architecture_contract={mesh_contract!r}")
    architecture = config.get("model_architecture_contract", TEXT_MODEL_ARCHITECTURE)
    if architecture not in {TEXT_MODEL_ARCHITECTURE, TEXT_MODEL_CONTRACT}:
        problems.append(f"model_architecture_contract={architecture!r}")
    config_eos = config.get("eos_token_id")
    if isinstance(config_eos, list):
        config_eos = config_eos[0] if len(config_eos) == 1 else None
    if config_eos is not None and config_eos != EOS_TOKEN_ID:
        problems.append(f"eos_token_id={config_eos!r} expected {EOS_TOKEN_ID}")

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
            "mesh_architecture_contract": mesh_contract,
            "model_architecture_contract": architecture,
            "hidden_size": geometry["hidden_size"],
            "logical_layers": geometry["logical_layers"],
            "physical_layers": geometry["physical_layers"],
            "recursive_loops": geometry["recursive_loops"],
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
    """Bind resume to this route's isolated MeSH residual implementation."""
    path = Path(__file__).resolve().parent / "models/recursive_layer_inject.py"
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


def semantic_contract(prepared, text_model_dir, ced_model_dir):
    """Resume identity deliberately excludes job-local staging paths."""
    return {
        "route": ROUTE, "code_sha256": code_digest(),
        "recursive_model_source_sha256": recursive_model_source_sha256(),
        "rows_canonical_sha256": prepared["rows_canonical_sha256"],
        "mapping_sha256": prepared["mapping_sha256"],
        "source_rows": prepared["source_rows"], "geometry": geometry(prepared["rows"]),
        "text_model_dir": str(text_model_dir), "text_model_contract": TEXT_MODEL_CONTRACT,
        "text_model_architecture": TEXT_MODEL_ARCHITECTURE,
        "text_model_schedule": list(TEXT_MODEL_SCHEDULE), "ced_model_dir": str(ced_model_dir),
        "sample_rate": 32000, "audio_policy": "mono_resample_center_crop_right_pad_fixed20_split10",
        "audio_read_policy": "shared_until_complete_collective_optimizer_boundary_switch",
        "prefix": [126, 1, 256], "answer_tokens": 300,
        "audio_encoder": {"name": "CEDSmall", "frozen": True, "mode": "eval",
                          "input_sample_rate": 32000, "ced_sample_rate": 16000,
                          "hidden_cache": False, "hidden_dtype": "float32",
                          "preprocessing": "whole20s_mel_then_init_bn_then_1012_frame_splits_pad_last",
                          "amplitude_to_db": "power_top_db120_per_sample"},
        "mapper": {"name": "freq_merge_126", "resample": False, "dropout": 0.05,
                   "hidden_dim": 384, "frequency_bins": 4, "output_dim": 576,
                   "max_temporal_tokens": 512, "external_projection": "Identity"},
        "layer_injection": dict(INJECTION_CONTRACT),
        "attention_mask": None, "pad_token": "!", "eos_suffix": " <|endoftext|>",
        "loss": "rank_local_token_mean_then_DDP_mean",
        "adamw": {"decoder_lr": 3e-4, "map_lr": 6e-4, "new_parameters_lr": 6e-4, "betas": [0.9, 0.999],
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
    }
