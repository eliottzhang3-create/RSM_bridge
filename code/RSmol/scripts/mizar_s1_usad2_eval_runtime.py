"""Inference adapter for the isolated online USAD2-Small Mizar S1 route."""
from __future__ import annotations

import importlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE_ROOT = ROOT / "mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_usad2_small"
DEFAULT_RUN = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "mizar_s1_usad2_small_gbs256_5090/formal_20261027_115510113068168/formal"
)
DEFAULT_CHECKPOINT = DEFAULT_RUN / "checkpoints/model--step-17313.ckpt"
DEFAULT_RUNTIME = DEFAULT_RUN / "runtime.json"
PROMPT_TOKENS, MAX_NEW_TOKENS, AUDIO_TOKENS = 256, 300, 251
PREFIX_TOKENS, HIDDEN_SIZE = AUDIO_TOKENS + 1 + PROMPT_TOKENS, 576
ROUTE = "usad2_small_mizar_s1_fixed20_gbs256_v1"
TRAINING_LOSS_REDUCTION = "mean_of_rank_local_microbatch_token_means_then_DDP_mean"
USAD2_ASSET_FILES = (
    "__init__.py", "config.json", "configuration_usad2.py", "modeling_usad2.py",
    "usad_model.py", "usad_modules.py", "model.safetensors",
)

PROTOCOL = {
    "version": "usad2_small_mizar_s1_fixed20_raw_greedy_v1",
    "audio_policy": "training_load_fixed20_single_center_crop_or_right_pad20_whole_waveform",
    "audio_sample_rate": 32000,
    "audio_seconds": 20,
    "prefix_layout": [251, 1, 256],
    "prefix_tokens": PREFIX_TOKENS,
    "audio_tokens": AUDIO_TOKENS,
    "separator_id": 0,
    "prompt_tokens": PROMPT_TOKENS,
    "prompt_policy": "fixed_official_choice_order_lowercase_then_training_tokenize",
    "prompt_eos_suffix": " <|endoftext|>",
    "prompt_padding": "right_bang_visible",
    "attention_mask": None,
    "answer_start_position": PREFIX_TOKENS,
    "first_answer_logit_position": PREFIX_TOKENS - 1,
    "max_new_tokens": MAX_NEW_TOKENS,
    "decoder": "pure_argmax_full_recompute",
    "do_sample": False,
    "use_cache": False,
    "dtype": "float32",
    "tf32": False,
    "stop": "first_generated_eos_or_300_new_tokens",
    "decode": {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
    "prediction_preprocessing": "none_including_eos_no_strip_no_label_extraction",
    "scoring": "benchmark_official_evaluation.py_only",
}


def sha256_file(path):
    import hashlib

    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def import_training_route(route_root=ROUTE_ROOT):
    """Import the USAD2 route in isolation from all other training routes."""
    route_root = Path(route_root).resolve()
    if not (route_root / "models/mellow.py").is_file():
        raise FileNotFoundError(route_root / "models/mellow.py")
    for name in list(sys.modules):
        if any(name == root or name.startswith(root + ".")
               for root in ("models", "data", "utils", "contract")):
            del sys.modules[name]
    sys.path.insert(0, str(route_root))
    importlib.invalidate_caches()
    names = ("contract", "models.mellow", "data.fixed20", "data.audiotext_dataset")
    modules = {name: importlib.import_module(name) for name in names}
    for name, module in modules.items():
        if not Path(module.__file__).resolve().is_relative_to(route_root):
            raise RuntimeError(f"training import escaped USAD2 route: {name}: {module.__file__}")
    return modules


def parse_training_geometry(value):
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = list(value)
    if len(parts) != 3:
        raise ValueError(f"expected geometry world_size,microbatch,gradient_accumulation, got {value!r}")
    try:
        result = tuple(int(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected integer geometry, got {value!r}") from exc
    if any(part < 1 for part in result):
        raise ValueError(f"expected positive geometry, got {result!r}")
    return result



def _asset_entries(contract):
    assets = contract.get("assets", {})
    usad2 = assets.get("usad2_small", {})
    if not assets.get("text") or not isinstance(usad2, dict) or not usad2.get("files"):
        raise ValueError("USAD2 checkpoint has no text/USAD2 asset identities")
    return [(Path(contract["text_model_dir"]), item) for item in assets["text"]] + [
        (Path(contract["usad2_root"]), item) for item in usad2["files"]
    ]



def audit_assets(contract):
    entries = _asset_entries(contract)
    for root, item in entries:
        path = root / item["name"]
        if (not path.is_file() or path.stat().st_size != item["bytes"]
                or sha256_file(path) != item["sha256"]):
            raise ValueError(f"pretrained component changed since USAD2 training: {path}")
    files = contract["assets"]["usad2_small"]["files"]
    names = [item["name"] for item in files]
    if len(names) != len(USAD2_ASSET_FILES) or set(USAD2_ASSET_FILES) != set(names):
        raise ValueError(f"USAD2 asset identity is incomplete or duplicated: {sorted(names)}")
    # The released package has an empty __init__.py, which is valid.
    return {"status": "PASS", "verified_files": len(entries),
            "usad2_files": list(USAD2_ASSET_FILES)}


def validate_checkpoint(saved, runtime, route_contract, expected_step,
                        expected_geometry=(8, 8, 4)):
    if route_contract.ROUTE != ROUTE:
        raise ValueError(f"unsupported USAD2 Mizar S1 training route: {route_contract.ROUTE}")
    contract = saved.get("contract", {})
    if (saved.get("schema_version") != 1 or saved.get("route_variant") != ROUTE
            or saved.get("mode") != "formal" or not saved.get("state_dict")):
        raise ValueError("expected a complete formal USAD2-Small Mizar S1 schema-1 checkpoint")
    identity = route_contract.digest(contract)
    if saved.get("contract_sha256") != identity:
        raise ValueError("checkpoint embedded USAD2 contract SHA256 mismatch")
    if (runtime.get("contract_sha256") != identity or runtime.get("contract") != contract
            or runtime.get("mode") != "formal" or runtime.get("route") != ROUTE):
        raise ValueError("runtime.json does not belong to this formal USAD2 checkpoint")
    if contract.get("code_sha256") != route_contract.code_digest():
        raise ValueError("USAD2 training source differs from the source recorded in the checkpoint")
    if (saved.get("total_step") != expected_step or saved.get("num_epochs") != 3
            or saved.get("loss_reduction") != TRAINING_LOSS_REDUCTION):
        raise ValueError("USAD2 checkpoint step / epoch horizon / training loss contract mismatch")
    expected = {
        "route": ROUTE,
        "sample_rate": 32000,
        "audio_policy": "mono_resample_center_crop_right_pad_fixed20_whole_waveform_then_resample_to_16k",
        "prefix": [251, 1, 256],
        "answer_tokens": 300,
        "attention_mask": None,
        "pad_token": "!",
        "eos_suffix": " <|endoftext|>",
        "forward_dtype": "float32",
        "autocast": False,
    }
    for key, value in expected.items():
        if key not in contract or contract[key] != value:
            raise ValueError(f"USAD2 training contract mismatch: {key}: {contract.get(key)!r}")
    geometry = contract.get("geometry", {})
    expected_geometry = parse_training_geometry(expected_geometry)
    actual_geometry = (geometry.get("world_size"), geometry.get("microbatch"),
                       geometry.get("gradient_accumulation_steps"))
    if actual_geometry != expected_geometry:
        raise ValueError(
            f"expected training geometry {expected_geometry[0]} x {expected_geometry[1]} x "
            f"{expected_geometry[2]}, got {actual_geometry[0]} x {actual_geometry[1]} x "
            f"{actual_geometry[2]}"
        )
    expected_usad2 = {
        "sample_rate": 16000, "encoder_dim": 384, "frames_20s": 999,
        "output_tokens": 251, "temporal_average_group_size": 4,
        "partial_group_policy": "mean_only_valid_frames", "frozen": True,
    }
    if contract.get("usad2") != expected_usad2:
        raise ValueError(f"unsupported USAD2 encoder/mapper contract: {contract.get('usad2')}")
    if (route_contract.USAD2_FRAMES, route_contract.USAD2_DIM, route_contract.AUDIO_GROUP_SIZE,
            route_contract.AUDIO_TOKENS, route_contract.PREFIX_TOKENS) != (999, 384, 4, 251, 508):
        raise ValueError("selected USAD2 route has incompatible encoder/mapper geometry")
    if not 0 < expected_step <= geometry.get("total_steps", 0):
        raise ValueError("USAD2 checkpoint step is outside the recorded training horizon")
    return contract



def audit_text_model(model):
    lm = model.caption_decoder.lm
    config, core = lm.config, lm.model
    if (config.hidden_size != HIDDEN_SIZE or config.num_hidden_layers != 30
            or getattr(config, "recursive_layer_count", None) is not None
            or len(core.layers) != 30 or len({id(layer) for layer in core.layers}) != 30
            or lm.get_input_embeddings().weight.shape[1] != HIDDEN_SIZE
            or getattr(model.caption_decoder, "is_recursive_mesh", False)):
        raise ValueError("ordinary USAD2 evaluation requires the full 30-layer, 576-d SmolLM2")
    return {"hidden_size": HIDDEN_SIZE, "physical_layers": 30, "recursive": False}


def load_runtime(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("USAD2 Mizar S1 evaluation requires one CUDA GPU")
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    modules = import_training_route(args.route_root)
    route_contract = modules["contract"]
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    contract = validate_checkpoint(saved, runtime, route_contract, args.expected_step,
                                   args.expected_geometry)
    assets = audit_assets(contract)
    usad2_assets = route_contract.validate_usad2_assets(contract["usad2_root"])
    recorded_usad2 = contract["assets"]["usad2_small"]
    if any(usad2_assets[key] != recorded_usad2.get(key) for key in ("root", "config")):
        raise ValueError("USAD2 asset configuration differs from the training identity")
    model = modules["models.mellow"].Mellow(
        audioenc_name="USAD2-Small", d_in=384, d_out=HIDDEN_SIZE,
        text_decoder=contract["text_model_dir"], prefix_length=PREFIX_TOKENS,
        freeze_text_decoder_weights=False, use_pretrained_audioencoder=True,
        freeze_audio_encoder_weights=True, pretrained_audioencoder_path=contract["usad2_root"])
    model.float()
    state_dict = saved["state_dict"]
    keys = list(state_dict)
    if keys and all(str(key).startswith("module.") for key in keys):
        state_dict = {str(key)[len("module."):]: value for key, value in state_dict.items()}
    elif keys and any(str(key).startswith("module.") for key in keys):
        raise ValueError("checkpoint has a mixed DDP state_dict namespace")
    model.load_state_dict(state_dict, strict=True)
    text_audit = audit_text_model(model)
    model.to(device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(contract["text_model_dir"], local_files_only=True)
    tokenizer.add_special_tokens({"pad_token": "!"})
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    pad_id = tokenizer.encode(tokenizer.pad_token)[0]
    token_audit = route_contract.validate_tokenizer_contract(
        tokenizer, pad_id, model.caption_decoder.lm.get_input_embeddings().num_embeddings)
    if any(module.training for module in model.modules()):
        raise RuntimeError("all USAD2 inference modules must be in eval mode")
    if {p.dtype for p in model.parameters()} != {torch.float32}:
        raise RuntimeError("all USAD2 inference parameters must be float32")
    text_encoder = modules["data.audiotext_dataset"].AudioTextDataset.__new__(
        modules["data.audiotext_dataset"].AudioTextDataset)
    text_encoder.tokenizer = tokenizer
    audit = {
        "status": "PASS", "checkpoint_step": saved["total_step"],
        "checkpoint_epoch": saved.get("epoch"), "contract_sha256": saved["contract_sha256"],
        "training_code_sha256": contract["code_sha256"], "strict_state_dict": True,
        "tokenizer": token_audit, "assets": assets, "all_modules_eval": True,
        "text_model": text_audit,
        "usad2_all_eval": all(not module.training for module in model.audio_encoder.base.usad2.modules()),
        "usad2_parameters_frozen": all(not p.requires_grad for p in model.audio_encoder.base.usad2.parameters()),
        "dtype": "float32", "deterministic_greedy": True, "tf32": False,
        "ddp_state_dict_prefix_normalized": bool(keys and all(str(key).startswith("module.") for key in keys)),
        "state_dict_source": "training_checkpoint_state_dict", "training_route": saved["route_variant"],
        "training_geometry": contract["geometry"], "training_loss_reduction": saved["loss_reduction"],
        "expected_geometry": list(parse_training_geometry(args.expected_geometry)),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }
    del saved, state_dict
    return model, tokenizer, device, text_encoder, modules["data.fixed20"], audit, modules


def greedy_decode(model, tokenizer, prefix, max_new_tokens=MAX_NEW_TOKENS):
    import torch

    if max_new_tokens != MAX_NEW_TOKENS:
        raise ValueError("both benchmarks use exactly max_new_tokens=300")
    if tuple(prefix.shape) != (1, PREFIX_TOKENS, HIDDEN_SIZE):
        raise ValueError(f"USAD2 prefix shape differs from training: {tuple(prefix.shape)}")
    generated_ids, generated = [], prefix
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        output = model.caption_decoder.lm(
            inputs_embeds=generated, attention_mask=None, labels=None, use_cache=False)
        logits = output.logits[:, -1, :]
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite USAD2 generation logits")
        next_id = int(torch.argmax(logits, dim=-1).item())
        generated_ids.append(next_id)
        del output, logits
        if next_id == int(tokenizer.eos_token_id):
            stop_reason = "eos_token"
            break
        ids = torch.tensor([[next_id]], device=prefix.device, dtype=torch.long)
        generated = torch.cat((generated, model.caption_decoder.lm.get_input_embeddings()(ids)), dim=1)
    text = tokenizer.decode(generated_ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    return {"generated_token_ids": generated_ids, "generated_text": text,
            "generated_text_raw": text, "generated_token_count": len(generated_ids),
            "stop_reason": stop_reason, "max_new_tokens": max_new_tokens}
