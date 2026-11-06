"""Independent S1 training contract for the complete converted official Mizar final."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Import only dependency-free metadata here. Never invoke the CPU conversion CLI
# or tensor_runtime.dependencies(): it deliberately hides CUDA devices.
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))
from mizar_official_recursive_5_20x2_5.contract import (
    architecture, implementation_identity, parameter_role as model_parameter_role)

ROUTE = "official_final_mizar_s1_fixed20_gbs128_5_20x2_5_serial84_looplora_r8_v1"
INJECTION_CONTRACT = {
    "ced_layers_1based": [8, 4], "capture": "complete_block_output_before_final_norm",
    "logical_indices_0based": [5, 25], "audio_positions": [0, 126],
    "grid": [126, 4, 384], "mlp": [1536, 576, 576],
    "mlp_structure": "LayerNorm_Linear_GELU_Linear_LayerNorm",
    "mlp_dropout": 0.0, "independent_mlps": True, "independent_gates": True,
    "gate": "tanh_Linear_576_1_per_token", "gate_weight_init": 0.0, "gate_bias_init": 0.0,
    "max_lr": 6e-4, "temporal_interpolation": False,
    "extra_position_parameters": False, "encoder_forwards_per_batch": 1,
}
AUDIO_REINJECTION = {
    "contract": "mapped_audio_shared_linear_tanh_token_gate_before_each_middle_loop_v1",
    "audio_tokens": 126, "hidden_size": 576, "logical_indices_0based": [5, 25],
    "linear": [576, 576], "gate": [576, 1], "activation": "tanh",
    "linear_weight_init": "torch_nn_Linear_default_uniform", "linear_bias_init": 0,
    "gate_weight_init": 0, "gate_bias_init": 0, "initialization_rng": "isolated_cpu",
    "shared_linear": True, "shared_gate": True, "compute_A_once_per_forward": True,
    "source": "original_mapped_audio_input_embeddings", "detach_A": False,
    "direct_residual_positions": [0, 126], "gate_condition": "current_audio_hidden_per_token",
    "lr": 6e-4, "new_parameters": 332929, "use_cache": False,
}
SERIAL_INJECTION_CONTRACT = {
    "contract": "mapped_input_residual_then_ced_layer8_4_residual_v1",
    "order": ["audio_input_embeddings", "ced_intermediate"],
    "logical_indices_0based": [5, 25], "ced_layers_1based": [8, 4],
    "input_gate_condition": "audio_hidden_before_input_residual",
    "intermediate_gate_condition": "audio_hidden_after_input_residual",
    "trace": [[5, "input"], [5, "layer8"], [25, "input"], [25, "layer4"]],
    "shared_input_projection_and_gate": True, "independent_intermediate_mlps_and_gates": True,
    "audio_positions": [0, 126], "additional_normalization": False,
    "additional_scale": False, "detach_residuals": False, "new_parameters": 2777859,
}
LOOP_LORA_CONTRACT = {
    "contract": "two_independent_loop_lora_on_shared_trainable_middle_v1",
    "physical_layers_0based": list(range(5, 25)), "logical_ranges_0based_inclusive": [[5, 24], [25, 44]],
    "targets": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "rank": 8, "alpha": 16, "scale": 2.0, "dropout": 0.0, "loops": 2,
    "A_init": "kaiming_uniform_a_sqrt5", "B_init": "zeros", "initialization_rng": "isolated_cpu",
    "base_trainable": True, "shared_base_between_loops": True, "independent_per_layer_projection_loop": True,
    "prefix_suffix_lora": False, "positions": "full_sequence", "max_lr": 6e-4,
    "weight_decay": 0.01, "new_parameters": 3256320, "use_cache": False,
    "gradient_checkpointing": False, "merge_adapters": False,
}
TEXT_MODEL_CONTRACT = "official_final_physical30_logical50_5_20x2_5_serial84_looplora_r8"
TEXT_MODEL_SCHEDULE = tuple(list(range(25)) + list(range(5, 25)) + list(range(25, 30)))
DEFAULT_MODEL_DIR = "/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M-5-20x2-5-serial84-looplora-r8"
WORLD_SIZE, BATCH_SIZE, ACCUMULATION_STEPS, EPOCHS, SEED = 8, 4, 4, 3, 1234
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
    for name in ("scripts/audit_mizar_s1_raw_audio.py", "run_mizar_official_final_s1_gbs128_5_20x2_5_5090.sh"):
        path = root.parents[1] / name
        result.update(name.encode("utf-8") + b"\0")
        result.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return result.hexdigest()


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


def recursive_model_source_sha256():
    return digest(implementation_identity())


def checkpoint_steps():
    return [5771, 11542, 17313, 23084, 28855, 34626]


def artifact_identity(directory):
    """Audit every artifact byte before executing its local CED constructor code."""
    from mizar_official_recursive_5_20x2_5.audit import audit_artifact
    result = audit_artifact(directory)
    manifest = result["manifest"]
    return {"manifest_sha256": digest(manifest), "files": manifest["files"],
            "implementation": manifest["implementation"], "architecture": manifest["architecture"]}


def semantic_contract(prepared, model_dir, artifact):
    """Exclude allocation-local paths; bind resume to full assets, data and recipe."""
    if (prepared["rows"] != sum(SOURCE_COUNTS.values()) or prepared["source_rows"] != SOURCE_COUNTS
            or prepared.get("read_policy") != "shared_until_complete"):
        raise ValueError("requires the unchanged S1 hybrid mixture and shared-until-complete policy")
    return {
        "route": ROUTE, "code_sha256": code_digest(),
        "recursive_model_source_sha256": recursive_model_source_sha256(),
        "rows_canonical_sha256": prepared["rows_canonical_sha256"],
        "mapping_sha256": prepared["mapping_sha256"],
        "source_rows": prepared["source_rows"], "geometry": geometry(prepared["rows"]),
        "model_dir": str(Path(model_dir).resolve()), "initialization_artifact": artifact,
        "text_model_contract": TEXT_MODEL_CONTRACT, "architecture": architecture(),
        "sample_rate": 32000, "audio_policy": "mono_resample_center_crop_right_pad_fixed20_split10",
        "audio_read_policy": "shared_until_complete_collective_optimizer_boundary_switch",
        "prefix": [126, 1, 256], "answer_tokens": ANSWER_TOKENS,
        "audio_encoder": {"name": "CEDSmall", "frozen": True, "mode": "eval",
                          "input_sample_rate": 32000, "ced_sample_rate": 16000,
                          "hidden_cache": False, "hidden_dtype": "float32",
                          "preprocessing": "whole20s_mel_then_init_bn_then_1012_frame_splits_pad_last",
                          "amplitude_to_db": "power_top_db120_per_sample"},
        "mapper": {"name": "freq_merge_126", "resample": False, "dropout": .05,
                   "hidden_dim": 384, "frequency_bins": 4, "output_dim": 576,
                   "max_temporal_tokens": 512, "external_projection": "Identity",
                   "initialization": "inherited_official_final"},
        "layer_injection": INJECTION_CONTRACT, "audio_reinjection": AUDIO_REINJECTION,
        "serial_audio_injection": SERIAL_INJECTION_CONTRACT, "loop_lora": LOOP_LORA_CONTRACT,
        "attention_mask": None, "pad_token": "!", "pad_id": 17, "eos_id": 0, "sep_id": 0,
        "eos_suffix": " <|endoftext|>",
        "loss": "mean_of_rank_local_microbatch_token_means_then_DDP_mean",
        "adamw": {"recurrent_base_lr": 6e-4, "new_parameters_lr": 6e-4,
                  "mapper_lr": 3e-4, "outer_lm_lr": 3e-4, "other_inherited_lm_lr": 3e-4,
                  "betas": [.9, .999], "eps": 1e-8, "weight_decay": .01,
                  "exclude_bias_ndim_le1": True},
        "scheduler": "Mizar_LambdaLR_warmup_cosine_5pct_floor_0.1_step_after_optimizer",
        "gradient_clip": {"type": "GradNormTracker", "initial_l2": 1., "initial_max": 10.,
                          "overdrive": 2.5, "momentum": .995},
        "forward_dtype": "float32", "autocast": False, "tf32": False, "grad_scaler": True,
        "workers": 0, "sync_batchnorm": False, "seed": SEED, "rank_rng": "1234_plus_rank",
        "order": "python_shuffle_1234_then_torch_randperm_1234_plus_epoch_contiguous_rank_shards",
        "gradient_accumulation": {"steps": ACCUMULATION_STEPS,
                                  "loss": "mean_of_rank_local_microbatch_token_means"},
        "checkpoint_steps": checkpoint_steps(), "epochs": EPOCHS,
    }


def resolved_contract(prepared, model_dir, artifact):
    import importlib.metadata
    import platform
    contract = semantic_contract(prepared, model_dir, artifact)
    contract["packages"] = {name: importlib.metadata.version(name) for name in
                            ("torch", "torchaudio", "transformers", "numpy", "soundfile", "safetensors")}
    contract["python"] = platform.python_version()
    if contract["packages"]["transformers"] != "4.54.1":
        raise ValueError("requires transformers4.54.1 in mellow_c8204d8")
    return contract


def validate_resume_payload(saved, config):
    """Check full-state/cursor invariants before any state is applied."""
    required = {"state_dict", "optimizer", "scheduler", "grad_scaler", "grad_norm_tracker",
                "random_state_by_rank", "total_step", "epoch", "next_optimizer_step", "contract",
                "contract_sha256", "schema_version", "route_variant", "mode", "num_epochs",
                "loss_reduction", "batch_geometry"}
    if (not isinstance(saved, dict) or not required.issubset(saved)
            or saved["schema_version"] != 1 or saved["route_variant"] != ROUTE
            or saved["mode"] != "formal" or saved["num_epochs"] != EPOCHS
            or saved["contract"] != config["contract"]
            or saved["contract_sha256"] != config["contract_sha256"]
            or digest(saved["contract"]) != saved["contract_sha256"]):
        raise ValueError("resume requires this route's complete formal state with identical code/data/assets/recipe")
    g = config["contract"]["geometry"]
    step, epoch, batch = saved["total_step"], saved["epoch"], saved["next_optimizer_step"]
    if (any(type(x) is not int for x in (step, epoch, batch))
            or not 0 < step < g["total_steps"] or not 0 <= epoch < EPOCHS
            or not 0 <= batch < g["steps_per_epoch"]
            or step != epoch * g["steps_per_epoch"] + batch
            or saved["scheduler"].get("last_epoch") != step
            or saved["loss_reduction"] != "mean_of_rank_local_microbatch_token_means_then_DDP_mean"
            or saved["batch_geometry"] != {"world_size": WORLD_SIZE, "microbatch_size": BATCH_SIZE,
                "gradient_accumulation_steps": ACCUMULATION_STEPS, "global_batch_size": GLOBAL_BATCH_SIZE}):
        raise ValueError("invalid resume sampler/scheduler/batch/loss cursor or already complete")
    states = saved["random_state_by_rank"]
    if (not isinstance(states, list) or len(states) != WORLD_SIZE
            or any(not isinstance(s, dict) or set(s) != {"python", "numpy", "torch", "cuda"} for s in states)
            or not saved["grad_scaler"] or not saved["grad_norm_tracker"]
            or not saved["optimizer"].get("state")
            or any(int(s["step"]) != step for s in saved["optimizer"]["state"].values())):
        raise ValueError("incomplete resume optimizer/scaler/GradNorm/eight-rank RNG state")
    return step, epoch, batch




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
