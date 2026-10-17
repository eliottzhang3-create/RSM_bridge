"""One FP32 prefix/decoding implementation for all three formal30 backbones."""
from __future__ import annotations

import gc
import hashlib
import importlib
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import contracts as C


def import_route(architecture: str):
    route = C.ROUTE_PARENT / C.ARCHITECTURES[architecture]["route"]
    # Same overlay order as the formal30 trainers; no global AutoModel registration.
    for name in list(sys.modules):
        if name.split(".")[0] in {"models", "data", "utils", "route_contract", "comparison"}:
            del sys.modules[name]
    for root in reversed((route, C.SHARED_ROUTE, C.RSMOL)):
        while str(root) in sys.path:
            sys.path.remove(str(root))
        sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    return importlib.import_module("models.model").get_model_class("Mellow")


def model_source_files(architecture: str) -> list[Path]:
    route = C.ROUTE_PARENT / C.ARCHITECTURES[architecture]["route"]
    paths = [route / "models/mellow.py", route / "models/decoder.py",
             route / "data/audiotext_dataset.py", C.SHARED_ROUTE / "data/template.py",
             C.SHARED_ROUTE / "utils/audio_io.py"]
    paths.extend(C.SHARED_ROUTE / "models" / name
                 for name in ("model.py", "audio.py", "htsat.py", "config.py"))
    if architecture != "smollm2":
        paths.extend([route / "models/recursive_text.py", route / "route_contract.py",
                      C.RSMOL / f"recursive_model_5_10x{C.ARCHITECTURES[architecture]['loops']}_5_mesh.py"])
    return paths


def seed_runtime() -> None:
    import numpy as np
    import torch

    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    torch.cuda.manual_seed_all(C.SEED)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(8)


def sample_crop_seed(sample_id: Any) -> int:
    return int(C.fingerprint([C.SEED, str(sample_id)])[:16], 16)


def load_model(args: Any):
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("formal30 evaluation requires one CUDA GPU")
    seed_runtime()
    cls = import_route(args.architecture)
    config = args.runtime_config_data
    encoder = config["model"]["encoder"]
    decoder = config["model"]["decoder"]
    print(f"[formal30-eval] constructing {args.architecture} in FP32", flush=True)
    model = cls(
        audioenc_name=encoder["audioenc_name"], d_in=encoder["out_emb"],
        text_decoder=decoder["text_decoder"], prefix_length=decoder["prefix_length"],
        freeze_text_decoder_weights=decoder["freeze_gpt_weights"], d_out=encoder["d_proj"],
        use_pretrained_audioencoder=encoder["use_pretrained_audioencoder"],
        freeze_audio_encoder_weights=encoder["freeze_audio_encoder_weights"],
        pretrained_audioencoder_path=encoder["pretrained_audioencoder_path"],
    )
    checkpoint = args.loaded_checkpoint
    for name, tensor in checkpoint["state_dict"].items():
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f"checkpoint tensor is not FP32: {name}: {tensor.dtype}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    # Never restore optimizer/scheduler/RNG. Release their deserialized CPU tensors.
    args.loaded_checkpoint = None
    del checkpoint
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(config["data"]["tokenizer_type"], local_files_only=True)
    tokenizer.add_special_tokens({"pad_token": "!"})
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.eos_token != "<|endoftext|>" or tokenizer.eos_token_id is None:
        raise ValueError("canonical SmolLM2 EOS contract differs")
    if tokenizer.pad_token_id != tokenizer.convert_tokens_to_ids("!"):
        raise ValueError("canonical SmolLM2 pad token must be !")
    lm = model.caption_decoder.lm
    if max(tokenizer.get_vocab().values()) >= lm.get_input_embeddings().weight.shape[0]:
        raise ValueError("tokenizer token IDs exceed the trained embedding vocabulary")
    if lm.config.hidden_size != C.HIDDEN_SIZE:
        raise ValueError("formal30 hidden size must be 576")
    spec = C.ARCHITECTURES[args.architecture]
    if len(lm.model.layers) != spec["physical_layers"]:
        raise ValueError("loaded physical layer count differs from checkpoint contract")
    architecture_audit = {"physical_layer_count": len(lm.model.layers)}
    if args.architecture != "smollm2":
        route_contract = importlib.import_module("route_contract")
        route_contract.validate_mesh_config(lm.config)
        module = importlib.import_module(f"recursive_model_5_10x{spec['loops']}_5_mesh")
        architecture_audit = module.parameter_audit(lm)
        for field, expected in {"physical_layer_count": 20, "logical_layer_count": spec["logical_layers"],
                                "memory_slots": spec["memory_slots"], "router_count": spec["routers"],
                                "recursive_loops": spec["loops"], "memory_persistent": False,
                                "router_objects_independent": True}.items():
            if architecture_audit.get(field) != expected:
                raise ValueError(f"loaded architecture audit mismatch: {field}")
        if tuple(architecture_audit["schedule"]) != route_contract.LOGICAL_TO_PHYSICAL:
            raise ValueError("loaded recursive execution schedule differs")
        if getattr(lm.model, "ablate_after_two_loops", False):
            raise ValueError("x4 ablation must be disabled")
    elif type(lm).__name__ != "LlamaForCausalLM":
        raise ValueError("baseline must remain the ordinary LlamaForCausalLM")
    # Reuse the actual formal30 tokenizer function; no dataset/staging is opened.
    args.tokenize_text = importlib.import_module("data.audiotext_dataset").AudioTextDataset._tokenize_text
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    for name, module in model.named_modules():
        if module.training:
            raise ValueError(f"module is not in eval mode: {name}")
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.device != device or (value.is_floating_point() and value.dtype != torch.float32):
            raise ValueError(f"non-FP32/CUDA parameter or buffer: {name}: {value.dtype} {value.device}")
    if getattr(lm.config, "max_position_embeddings", 0) < C.AUDIO_PREFIX_TOKENS + C.MAX_PROMPT_TOKENS + C.MAX_NEW_TOKENS:
        raise ValueError("LM context cannot accommodate the common evaluation protocol")
    args.loaded_model_audit = {
        "architecture": args.architecture, "architecture_audit": architecture_audit,
        "class": f"{type(lm).__module__}.{type(lm).__name__}",
        "attention_implementation": lm.config._attn_implementation,
        "dtype": "float32", "device": str(device), "all_modules_eval": True,
        "strict_state_load": True, "optimizer_scheduler_rng_restored": False,
        "tokenizer_vocab_sha256": C.fingerprint(tokenizer.get_vocab()),
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
        "tf32": False, "torch_version": torch.__version__,
        "transformers_version": importlib.import_module("transformers").__version__,
        "cuda_version": torch.version.cuda,
        "shared_lm_config": {field: getattr(lm.config, field, None) for field in (
            "hidden_size", "vocab_size", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "rms_norm_eps", "rope_theta", "rope_scaling",
            "max_position_embeddings", "attention_dropout", "hidden_act",
            "attention_bias", "mlp_bias", "tie_word_embeddings", "pad_token_id",
        )},
    }
    print(f"[formal30-eval] strict model load PASS: {args.loaded_model_audit}", flush=True)
    return model, tokenizer, device, args.loaded_model_audit


def tokenize_prompt(args: Any, tokenizer: Any, prompt: str, device: Any):
    original = tokenizer.encode_plus(text=prompt + " <|endoftext|>", add_special_tokens=True,
                                     truncation=False, padding=False)["input_ids"]
    encoded = args.tokenize_text(SimpleNamespace(tokenizer=tokenizer),
                                 prompt, C.MAX_PROMPT_TOKENS, padding=False)
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    if not (1 <= ids.shape[-1] <= C.MAX_PROMPT_TOKENS) or not bool(mask.eq(1).all()):
        raise ValueError("prompt must contain only its actual tokens, with no padding")
    if int(ids[0, -1]) != tokenizer.eos_token_id:
        raise ValueError("generation must begin immediately after prompt EOS")
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device)}, {
        "prompt_token_count": ids.shape[-1], "prompt_original_token_count": len(original),
        "prompt_truncated": len(original) > C.MAX_PROMPT_TOKENS,
        "prompt_token_ids": ids[0].tolist(), "prompt_final_token_id": int(ids[0, -1]),
        "prompt_padding_tokens": 0, "prompt_eos_appended": True,
    }


def make_prefix(model: Any, audio1: Any, text_input: dict, device: Any):
    import torch

    audio1 = audio1.to(device=device, dtype=torch.float32).contiguous()
    if tuple(audio1.shape) != (1, 320000) or not bool(torch.isfinite(audio1).all()):
        raise ValueError("audio1 must be a finite FP32 [1,320000] waveform")
    # The MMAU single-audio contract has two decoder slots, but the second
    # slot reuses the first slot's *encoded* output.  This avoids both a
    # silent waveform and a second, numerically independent HTSAT pass.
    audio_embed1, _, _ = model.audio_encoder(audio1)
    prefix = model.caption_decoder.generate_prefix_inference(
        audio_embed1, audio_embed1, text_input
    )
    expected = (1, C.AUDIO_PREFIX_TOKENS + text_input["input_ids"].shape[1], C.HIDDEN_SIZE)
    if tuple(prefix.shape) != expected or prefix.dtype != torch.float32 or not bool(torch.isfinite(prefix).all()):
        raise ValueError(f"invalid formal30 multimodal prefix: {tuple(prefix.shape)}, expected {expected}")
    return prefix, {
        "audio2_policy": "reuse_audio1_encoded_output", "audio2_reused": True,
        "audio2_waveform_materialized": False, "audio_encoder_invocations": 1,
        "single_audio_slot": False, "audio_prefix_tokens": 260,
        "prefix_token_count": prefix.shape[1], "combined_prefix_shape": list(prefix.shape),
        "prefix_layout": "audio1_129_sep_audio2_129_sep_variable_prompt_eos",
        "audio1_waveform_sha256": hashlib.sha256(audio1.cpu().numpy().tobytes()).hexdigest(),
    }


def greedy_decode(model: Any, tokenizer: Any, prefix: Any) -> dict:
    import torch

    generated, ids = prefix, []
    mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device)
    started = time.perf_counter()
    stop_reason = "max_new_tokens"
    for _ in range(C.MAX_NEW_TOKENS):
        output = model.caption_decoder.lm(inputs_embeds=generated, attention_mask=mask,
                                          use_cache=False, return_dict=True)
        logits = output.logits[:, -1, :].clone()
        if logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
            raise ValueError("generation logits must be finite FP32")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        remove = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > 0.8
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        logits.scatter_(1, sorted_indices, sorted_logits.masked_fill(remove, -float("inf")))
        next_token = int(torch.argmax(logits, dim=-1).item())
        ids.append(next_token)
        del output
        if next_token == tokenizer.eos_token_id:
            stop_reason = "eos_token"
            break
        embed = model.caption_decoder.lm.get_input_embeddings()(
            torch.tensor([[next_token]], dtype=torch.long, device=prefix.device))
        generated = torch.cat((generated, embed), dim=1)
        mask = torch.cat((mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)), dim=1)
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    return {
        "generated_token_ids": ids, "generated_text_raw": raw,
        "generated_text": raw.split(tokenizer.eos_token)[0], "stop_reason": stop_reason,
        "generated_token_count": len(ids), "requested_max_new_tokens": C.MAX_NEW_TOKENS,
        "generation_seconds": time.perf_counter() - started, "inference_dtype": "float32",
        "generation_do_sample": False, "generation_use_cache": False,
        "generation_top_p": 0.8, "generation_temperature": 1.0,
    }


def run_generation(args: Any, common: Any, model: Any, tokenizer: Any, device: Any,
                   sample: dict, *, max_prompt_tokens: int, max_new_tokens: int):
    import torch

    if (max_prompt_tokens, max_new_tokens) != (C.MAX_PROMPT_TOKENS, C.MAX_NEW_TOKENS):
        raise ValueError("formal30 generation limits cannot vary between models/tasks")
    # Rebuild here too: MMAR's legacy traversal uses its own prompt helper.
    prompt = C.build_prompt(sample["question"], sample["choices"])
    if prompt != sample["prompt"]:
        raise ValueError("dataset traversal prompt differs from the shared prompt builder")
    text_input, prompt_audit = tokenize_prompt(args, tokenizer, prompt, device)
    waveform = sample["waveform"]
    segment_audit = {"policy": "mono_first10s_right_zero_pad"}
    if args.task == "mmau_test_mini":
        # A crop is a property of sample ID, independent of model construction,
        # earlier output length, node RNG state, skipped rows or resume position.
        seed = sample_crop_seed(sample["id"])
        waveform, segment_audit = common.mellow_author_reply_audio_segment(waveform, rng=random.Random(seed))
        segment_audit["sample_seed"] = seed
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        prefix, prefix_audit = make_prefix(model, waveform, text_input, device)
        result = greedy_decode(model, tokenizer, prefix)
    return {**result, **prompt_audit, **prefix_audit, "audio1_segment": segment_audit}
