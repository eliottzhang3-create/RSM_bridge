"""Inference adapter for the unchanged HTSAT / Mizar Stage 1 training route."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE_ROOT = ROOT / "mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_3epochs"
DEFAULT_RUN = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_5090/formal_async_v3/formal")
DEFAULT_CHECKPOINT = DEFAULT_RUN / "checkpoints/model--step-122937.ckpt"
DEFAULT_RUNTIME = DEFAULT_RUN / "runtime.json"
PROMPT_TOKENS, MAX_NEW_TOKENS, PREFIX_TOKENS, HIDDEN_SIZE = 256, 300, 515, 576
TRAINING_LOSS_REDUCTIONS = {
    "htsat_mizar_s1_fixed20_v1": "rank_local_token_mean_then_DDP_mean",
    "htsat_mizar_s1_fixed20_gbs256_v1": "mean_of_rank_local_microbatch_token_means_then_DDP_mean",
    "htsat_mizar_s1_fixed20_gbs256_recursive_mesh_v1": "mean_of_rank_local_microbatch_token_means_then_DDP_mean",
}
PROTOCOL = {
    "version": "htsat_mizar_s1_fixed20_raw_greedy_v1",
    "audio_policy": "training_load_fixed20_single_center_crop_or_right_pad20_then_split10",
    "audio_sample_rate": 32000, "audio_seconds": 20,
    "prefix_layout": [129, 129, 1, 256], "prefix_tokens": 515,
    "separator_id": 0, "prompt_tokens": 256,
    "prompt_policy": "fixed_official_choice_order_lowercase_then_training_tokenize",
    "prompt_eos_suffix": " <|endoftext|>", "prompt_padding": "right_bang_visible",
    "attention_mask": None, "answer_start_position": 515,
    "first_answer_logit_position": 514, "max_new_tokens": 300,
    "decoder": "pure_argmax_full_recompute", "do_sample": False,
    "use_cache": False, "dtype": "float32", "tf32": False,
    "stop": "first_generated_eos_or_300_new_tokens",
    "decode": {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
    "prediction_preprocessing": "none_including_eos_no_strip_no_label_extraction",
    "scoring": "benchmark_official_evaluation.py_only",
}


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def import_training_route(route_root=ROUTE_ROOT):
    """Keep the training route's absolute imports isolated from other models."""
    route_root = Path(route_root).resolve()
    if not (route_root / "models/mellow.py").is_file():
        raise FileNotFoundError(route_root / "models/mellow.py")
    for name in list(sys.modules):
        if any(name == root or name.startswith(root + ".")
               for root in ("models", "data", "utils", "contract")):
            del sys.modules[name]
    sys.path.insert(0, str(route_root))
    importlib.invalidate_caches()
    modules = {name: importlib.import_module(name) for name in
               ("contract", "models.mellow", "data.fixed20", "data.audiotext_dataset")}
    for name, module in modules.items():
        if not Path(module.__file__).resolve().is_relative_to(route_root):
            raise RuntimeError(f"training import escaped route: {name}: {module.__file__}")
    return modules


def parse_training_geometry(value):
    """Normalize an evaluator geometry override to (world, microbatch, accumulation)."""
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


def validate_checkpoint(saved, runtime, route_contract, expected_step,
                        expected_geometry=(8, 4, 1)):
    if route_contract.ROUTE not in TRAINING_LOSS_REDUCTIONS:
        raise ValueError(f"unsupported HTSAT / Mizar S1 training route: {route_contract.ROUTE}")
    contract = saved.get("contract", {})
    if (saved.get("schema_version") != 1 or saved.get("route_variant") != route_contract.ROUTE
            or saved.get("mode") != "formal" or not saved.get("state_dict")):
        raise ValueError("expected a complete formal HTSAT / Mizar S1 schema-1 checkpoint")
    identity = route_contract.digest(contract)
    if saved.get("contract_sha256") != identity:
        raise ValueError("checkpoint embedded contract SHA256 mismatch")
    if (runtime.get("contract_sha256") != identity or runtime.get("contract") != contract
            or runtime.get("mode") != "formal" or runtime.get("route") != route_contract.ROUTE):
        raise ValueError("runtime.json does not belong to this formal checkpoint")
    if contract.get("code_sha256") != route_contract.code_digest():
        raise ValueError("training source differs from the source recorded in the checkpoint")
    if (saved.get("total_step") != expected_step or saved.get("num_epochs") != 3
            or saved.get("loss_reduction") != TRAINING_LOSS_REDUCTIONS[route_contract.ROUTE]):
        raise ValueError("checkpoint step / epoch horizon / training loss contract mismatch")
    expected = {"route": route_contract.ROUTE, "sample_rate": 32000,
                "audio_policy": "mono_resample_center_crop_right_pad_fixed20_split10",
                "prefix": [129, 129, 1, 256], "answer_tokens": 300,
                "attention_mask": None, "pad_token": "!", "eos_suffix": " <|endoftext|>",
                "forward_dtype": "float32", "autocast": False}
    for key, value in expected.items():
        if key not in contract or contract[key] != value:
            raise ValueError(f"checkpoint training contract mismatch: {key}: {contract.get(key)!r}")
    if hasattr(route_contract, "TEXT_MODEL_CONTRACT"):
        recursive_expected = {
            "text_model_contract": route_contract.TEXT_MODEL_CONTRACT,
            "text_model_architecture": route_contract.TEXT_MODEL_ARCHITECTURE,
            "text_model_schedule": list(route_contract.TEXT_MODEL_SCHEDULE),
        }
        for key, value in recursive_expected.items():
            if contract.get(key) != value:
                raise ValueError(
                    f"recursive text training contract mismatch: {key}: {contract.get(key)!r}"
                )
        expected_source_sha = route_contract.recursive_model_source_sha256()
        if contract.get("recursive_model_source_sha256") != expected_source_sha:
            raise ValueError("recursive model source differs from the source recorded in the checkpoint")
    geometry = contract.get("geometry", {})
    expected_geometry = parse_training_geometry(expected_geometry)
    actual_geometry = (geometry.get("world_size"), geometry.get("microbatch"),
                       geometry.get("gradient_accumulation_steps"))
    if actual_geometry != expected_geometry:
        raise ValueError(
            "expected training geometry "
            f"{expected_geometry[0]} x {expected_geometry[1]} x {expected_geometry[2]}, "
            f"got {actual_geometry[0]} x {actual_geometry[1]} x {actual_geometry[2]}"
        )
    if not 0 < expected_step <= geometry.get("total_steps", 0):
        raise ValueError("checkpoint step is outside the recorded training horizon")
    return contract


def audit_assets(contract):
    """Verify the original tokenizer/model components before strict loading."""
    assets = contract.get("assets", {})
    if not assets.get("text") or not assets.get("htsat"):
        raise ValueError("checkpoint has no original component asset identities")
    entries = [(Path(contract["text_model_dir"]), item) for item in assets["text"]]
    entries.append((Path(contract["htsat_root"]), assets["htsat"]))
    for root, item in entries:
        path = root / item["name"]
        if (not path.is_file() or path.stat().st_size != item["bytes"]
                or sha256_file(path) != item["sha256"]):
            raise ValueError(f"pretrained component changed since training: {path}")
    return {"status": "PASS", "verified_files": len(entries)}


def audit_recursive_route(route_contract, contract, model):
    """Recheck the recursive text artifact and loaded parameter geometry."""
    if not hasattr(route_contract, "recursive_checkpoint_contract"):
        return None
    text_root = Path(contract["text_model_dir"])
    expected = contract.get("recursive_text_checkpoint")
    actual = route_contract.recursive_checkpoint_contract(text_root)
    if expected is not None and actual != expected:
        raise ValueError("recursive text checkpoint changed since the training runtime was written")
    # The config validator belongs to the route's recursive text loader, not
    # to contract.py.  Import it after the route has been isolated so a stale
    # contract module from another training route cannot be used.
    recursive_text = importlib.import_module("models.recursive_text")
    recursive_text.validate_recursive_config(model.caption_decoder.lm.config)
    recursive_impl = importlib.import_module("recursive_model_5_10x2_5_mesh")
    parameter_audit = recursive_impl.parameter_audit(model.caption_decoder.lm)
    required = {
        "logical_layer_count": 30,
        "physical_layer_count": 20,
        "logical_cache_slot_count": 30,
        "recursive_loops": 2,
        "memory_slots": 5,
        "router_count": 6,
    }
    if any(parameter_audit.get(key) != value for key, value in required.items()):
        raise ValueError(f"loaded recursive parameter audit mismatch: {parameter_audit}")
    if parameter_audit.get("schedule") != list(route_contract.TEXT_MODEL_SCHEDULE):
        raise ValueError("loaded recursive parameter schedule mismatch")
    return {"checkpoint": actual, "parameters": parameter_audit}


def load_runtime(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Mizar S1 evaluation requires one CUDA GPU")
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
    model = modules["models.mellow"].Mellow(
        audioenc_name="HTSAT", d_in=768, d_out=HIDDEN_SIZE,
        text_decoder=contract["text_model_dir"], prefix_length=PREFIX_TOKENS,
        freeze_text_decoder_weights=False, use_pretrained_audioencoder=True,
        freeze_audio_encoder_weights=True, pretrained_audioencoder_path=contract["htsat_root"])
    state_dict = saved["state_dict"]
    # train.py currently saves the unwrapped Mellow state. Normalize only an
    # exact uniform DDP `module.` namespace if a checkpoint was wrapped by a
    # compatible torch version; mixed or unexpected schemas fail strictly.
    keys = list(state_dict)
    if keys and all(str(key).startswith("module.") for key in keys):
        state_dict = {str(key)[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    recursive_audit = audit_recursive_route(route_contract, contract, model)
    model.float().to(device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(contract["text_model_dir"], local_files_only=True)
    tokenizer.add_special_tokens({"pad_token": "!"})
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    pad_id = tokenizer.encode(tokenizer.pad_token)[0]
    token_audit = route_contract.validate_tokenizer_contract(
        tokenizer, pad_id, model.caption_decoder.lm.get_input_embeddings().num_embeddings)
    if any(module.training for module in model.modules()):
        raise RuntimeError("all model modules must be in eval mode")
    if {p.dtype for p in model.parameters()} != {torch.float32}:
        raise RuntimeError("all inference parameters must be float32")
    # Reuse the actual training tokenize() implementation without opening its
    # million-row training manifest or the old node-local staging directory.
    text_encoder = modules["data.audiotext_dataset"].AudioTextDataset.__new__(
        modules["data.audiotext_dataset"].AudioTextDataset)
    text_encoder.tokenizer = tokenizer
    audit = {"status": "PASS", "checkpoint_step": saved["total_step"],
             "checkpoint_epoch": saved.get("epoch"), "contract_sha256": saved["contract_sha256"],
             "training_code_sha256": contract["code_sha256"], "strict_state_dict": True,
             "tokenizer": token_audit, "assets": assets, "all_modules_eval": True,
             "dtype": "float32", "deterministic_greedy": True, "tf32": False,
             "ddp_state_dict_prefix_normalized": bool(keys and all(str(key).startswith("module.") for key in keys)),
             "state_dict_source": "training_checkpoint_state_dict",
             "training_route": saved["route_variant"],
             "training_geometry": contract["geometry"],
             "training_loss_reduction": saved["loss_reduction"],
             "recursive_audit": recursive_audit,
             "expected_geometry": list(parse_training_geometry(args.expected_geometry)),
             "parameter_count": sum(p.numel() for p in model.parameters()),
             "torch": torch.__version__, "cuda": torch.version.cuda,
             "gpu": torch.cuda.get_device_name(device)}
    del saved
    return model, tokenizer, device, text_encoder, modules["data.fixed20"], audit, modules


def greedy_decode(model, tokenizer, prefix, max_new_tokens=MAX_NEW_TOKENS):
    import torch

    if max_new_tokens != MAX_NEW_TOKENS:
        raise ValueError("both benchmarks use exactly max_new_tokens=300")
    if tuple(prefix.shape) != (1, PREFIX_TOKENS, HIDDEN_SIZE):
        raise ValueError(f"prefix shape differs from training: {tuple(prefix.shape)}")
    generated_ids, generated = [], prefix
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        output = model.caption_decoder.lm(
            inputs_embeds=generated, attention_mask=None, labels=None, use_cache=False)
        logits = output.logits[:, -1, :]
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite generation logits")
        next_id = int(torch.argmax(logits, dim=-1).item())
        generated_ids.append(next_id)
        del output, logits
        if next_id == int(tokenizer.eos_token_id):
            stop_reason = "eos_token"
            break
        ids = torch.tensor([[next_id]], device=prefix.device, dtype=torch.long)
        generated = torch.cat((generated, model.caption_decoder.lm.get_input_embeddings()(ids)), dim=1)
    # Keep every continuation token, including EOS; no split/strip/rewrite.
    text = tokenizer.decode(generated_ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    return {"generated_token_ids": generated_ids, "generated_text": text,
            "generated_text_raw": text, "generated_token_count": len(generated_ids),
            "stop_reason": stop_reason, "max_new_tokens": max_new_tokens}


