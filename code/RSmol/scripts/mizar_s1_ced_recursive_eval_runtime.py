"""Inference for four isolated non-MeSH CED 5+10x2+5 training routes.

The standard CED adapter owns the common generation protocol and read-only
asset/import helpers. Training files are never changed or patched for inference.
"""
from __future__ import annotations

import copy
import importlib
import json
import os
import random
from pathlib import Path

import mizar_s1_ced_eval_runtime as reference

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "mellow_official_training_c8204d8_adamw_cosine"
REMOTE_OUTPUT_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol")
PROMPT_TOKENS, MAX_NEW_TOKENS, AUDIO_TOKENS = 256, 300, 126
PREFIX_TOKENS, HIDDEN_SIZE = 383, 576
TEXT_MODEL_CONTRACT = "logical_30_physical_20_5_10_5_loops_2_no_mesh"
TEXT_MODEL_SCHEDULE = tuple(list(range(15)) + list(range(5, 15)) + list(range(15, 20)))
TRAINING_LOSS_REDUCTION = reference.TRAINING_LOSS_REDUCTION
PROTOCOL = copy.deepcopy(reference.PROTOCOL)
sha256_file = reference.sha256_file
parse_training_geometry = reference.parse_training_geometry
audit_assets = reference.audit_assets
import_training_route = reference.import_training_route

ROUTE_SPECS = {}
for _suffix, _timestamp, _implementation, _layers in (
    ("", "20261028_160823559722647", "recursive_model_5_10_5", ()),
    ("_layer8_4_inject", "20261030_000807991327585", "models.recursive_layer_inject", (8, 4)),
    ("_layer4_8_inject", "20261029_210116592758429", "models.recursive_layer_inject", (4, 8)),
    ("_audio_reinject", "20261029_110929631468450", "models.recursive_reinject", ()),
):
    _variant = "recursive_5_10_5" + _suffix
    ROUTE_SPECS[_variant] = {
        "route": "ced_small_mizar_s1_fixed20_gbs256_" + _variant + "_v1",
        "route_root": TRAINING_ROOT / ("mizar_stage1_fixed20_gbs256_3epochs_"
                                       "recursive_5_10_5_ced_small" + _suffix),
        "run_root": REMOTE_OUTPUT_ROOT / ("mizar_s1_ced_small_gbs256_recursive_5_10_5"
                                          + _suffix + "_5090") / ("formal_" + _timestamp),
        "implementation": _implementation,
        "layer_order": _layers,
        "reinjection": _suffix == "_audio_reinject",
    }


def variant_spec(eval_variant):
    if eval_variant not in ROUTE_SPECS:
        raise ValueError(f"unsupported non-MeSH CED evaluation variant: {eval_variant!r}")
    return ROUTE_SPECS[eval_variant]


def validate_checkpoint(saved, runtime, route_contract, expected_step,
                        expected_geometry=(8, 8, 4), *, eval_variant):
    spec = variant_spec(eval_variant)
    route = spec["route"]
    if route_contract.ROUTE != route:
        raise ValueError(f"selected training route does not match variant {eval_variant}: {route_contract.ROUTE}")
    contract = saved.get("contract", {})
    if (saved.get("schema_version") != 1 or saved.get("route_variant") != route
            or saved.get("mode") != "formal" or not saved.get("state_dict")):
        raise ValueError("expected a complete formal checkpoint for the selected non-MeSH CED route")
    identity = route_contract.digest(contract)
    if saved.get("contract_sha256") != identity:
        raise ValueError("checkpoint embedded CED contract SHA256 mismatch")
    if (runtime.get("contract_sha256") != identity or runtime.get("contract") != contract
            or runtime.get("mode") != "formal" or runtime.get("route") != route):
        raise ValueError("runtime.json does not belong to this formal CED checkpoint")
    if contract.get("code_sha256") != route_contract.code_digest():
        raise ValueError("CED training source differs from the source recorded in the checkpoint")
    if contract.get("recursive_model_source_sha256") != route_contract.recursive_model_source_sha256():
        raise ValueError("recursive model source differs from the source recorded in the checkpoint")
    expected = {
        "route": route, "text_model_contract": TEXT_MODEL_CONTRACT,
        "text_model_architecture": TEXT_MODEL_CONTRACT,
        "text_model_schedule": list(TEXT_MODEL_SCHEDULE), "sample_rate": 32000,
        "audio_policy": "mono_resample_center_crop_right_pad_fixed20_split10",
        "prefix": [126, 1, 256], "answer_tokens": 300, "attention_mask": None,
        "pad_token": "!", "eos_suffix": " <|endoftext|>",
        "forward_dtype": "float32", "autocast": False,
    }
    for key, value in expected.items():
        if key not in contract or contract[key] != value:
            raise ValueError(f"CED training contract mismatch: {key}: {contract.get(key)!r}")
    if (route_contract.TEXT_MODEL_CONTRACT != TEXT_MODEL_CONTRACT
            or route_contract.TEXT_MODEL_ARCHITECTURE != TEXT_MODEL_CONTRACT
            or tuple(route_contract.TEXT_MODEL_SCHEDULE) != TEXT_MODEL_SCHEDULE):
        raise ValueError("selected route does not implement the required non-MeSH architecture")
    if not isinstance(contract.get("recursive_text_checkpoint"), dict) or not contract["recursive_text_checkpoint"]:
        raise ValueError("CED checkpoint has no recursive text checkpoint identity")
    if (saved.get("total_step") != expected_step or saved.get("num_epochs") != 3
            or saved.get("loss_reduction") != TRAINING_LOSS_REDUCTION):
        raise ValueError("CED checkpoint step / epoch horizon / training loss contract mismatch")
    geometry = contract.get("geometry", {})
    actual_geometry = tuple(geometry.get(key) for key in
                            ("world_size", "microbatch", "gradient_accumulation_steps"))
    if actual_geometry != parse_training_geometry(expected_geometry):
        raise ValueError(f"expected training geometry {expected_geometry}, got {actual_geometry}")
    if not 0 < expected_step <= geometry.get("total_steps", 0):
        raise ValueError("CED checkpoint step is outside the recorded training horizon")
    encoder = contract.get("audio_encoder", {})
    if encoder.get("name") != "CEDSmall" or encoder.get("hidden_cache") is not False:
        raise ValueError("CED evaluation requires the online CEDSmall hidden-feature contract")
    if any(contract.get("mapper", {}).get(key) != value for key, value in {
            "name": "freq_merge_126", "hidden_dim": 384, "frequency_bins": 4,
            "output_dim": 576, "resample": False, "dropout": 0.05,
            "max_temporal_tokens": 512, "external_projection": "Identity"}.items()):
        raise ValueError("unsupported CED mapper contract")
    if spec["layer_order"]:
        injection = contract.get("layer_injection")
        if (injection != route_contract.INJECTION_CONTRACT
                or injection.get("ced_layers_1based") != list(spec["layer_order"])
                or injection.get("logical_indices_0based") != [5, 15]
                or injection.get("audio_positions") != [0, 126]):
            raise ValueError("CED layer injection contract/order mismatch")
    elif "layer_injection" in contract:
        raise ValueError("unexpected CED layer injection contract")
    if spec["reinjection"]:
        injection = contract.get("audio_reinjection")
        if (injection != route_contract.AUDIO_REINJECTION
                or injection.get("logical_indices_0based") != [5, 15]
                or injection.get("source") != "original_mapped_audio_input_embeddings"
                or injection.get("shared_linear") is not True
                or injection.get("shared_gate") is not True):
            raise ValueError("mapped audio reinjection contract mismatch")
    elif "audio_reinjection" in contract:
        raise ValueError("unexpected mapped audio reinjection contract")
    return contract


def audit_recursive_route(route_contract, contract, model, *, eval_variant):
    """Audit trained tensors and the explicit implementation after strict load."""
    spec = variant_spec(eval_variant)
    route_root = Path(route_contract.__file__).resolve().parent
    actual = route_contract.recursive_checkpoint_contract(Path(contract["text_model_dir"]))
    if actual != contract["recursive_text_checkpoint"]:
        raise ValueError("recursive text checkpoint changed since training")
    recursive_text = importlib.import_module("models.recursive_text")
    implementation = importlib.import_module(spec["implementation"])
    implementation_path = (ROOT / "recursive_model_5_10_5.py" if spec["implementation"] == "recursive_model_5_10_5"
                           else route_root / (spec["implementation"].replace(".", "/") + ".py"))
    if (Path(recursive_text.__file__).resolve() != route_root / "models/recursive_text.py"
            or Path(implementation.__file__).resolve() != implementation_path):
        raise RuntimeError("recursive loader or implementation escaped the selected training route")
    decoder, lm = model.caption_decoder, model.caption_decoder.lm
    if (not isinstance(lm, implementation.RecursiveLlamaForCausalLM)
            or not getattr(decoder, "is_recursive_5_10_5", False)
            or getattr(decoder, "text_model_contract", None) != TEXT_MODEL_CONTRACT
            or lm.config.hidden_size != HIDDEN_SIZE):
        raise ValueError("loaded decoder is not the required hidden576 non-MeSH language model")
    parameters = implementation.parameter_audit(lm)
    recursive_text.validate_loaded_model(lm, parameters)
    core = lm.model
    new_parameters = 0
    if spec["layer_order"]:
        injection = recursive_text.validate_layer_injection(lm)
        mlps = model.audio_encoder.base.layer_mlps
        if (injection["ced_layers"] != list(spec["layer_order"])
                or set(mlps) != {"4", "8"} or mlps["4"] is mlps["8"]
                or getattr(core, "collect_injection_stats", True)):
            raise ValueError("independent CED layer MLP/gate geometry or injection order mismatch")
        new_parameters = sum(p.numel() for module in (mlps, core.audio_gates) for p in module.parameters())
        if new_parameters != 2444930:
            raise ValueError(f"expected 2444930 layer injection parameters, got {new_parameters}")
    elif hasattr(core, "audio_gates") or hasattr(model.audio_encoder.base, "layer_mlps"):
        raise ValueError("unexpected layer injection modules")
    if spec["reinjection"]:
        injection = recursive_text.validate_audio_reinjection(lm)
        new_parameters = injection["parameter_count"]
        if new_parameters != 332929 or getattr(core, "collect_reinject_stats", True):
            raise ValueError("shared audio reinjection geometry/statistics mode mismatch")
    elif hasattr(core, "audio_linear") or hasattr(core, "audio_gate"):
        raise ValueError("unexpected mapped audio reinjection modules")
    return {"checkpoint": actual, "parameters": parameters,
            "implementation": spec["implementation"],
            "implementation_sha256": contract["recursive_model_source_sha256"],
            "layer_order": list(spec["layer_order"]), "reinjection": spec["reinjection"],
            "new_parameter_count": new_parameters, "memory_slots": 0, "routers": 0}


def install_recursive_trace_audit(lm, *, eval_variant):
    """Observe actual physical calls; all hooks return None and preserve outputs."""
    spec = variant_spec(eval_variant)
    trace = []
    expected_injection = ([[index, layer] for index, layer in zip((5, 15), spec["layer_order"])]
                          if spec["layer_order"] else [5, 15] if spec["reinjection"] else [])
    audit = {"status": "PENDING", "checked_forwards": 0, "use_cache": False,
             "expected_schedule": list(TEXT_MODEL_SCHEDULE),
             "expected_injection_trace": expected_injection}

    def begin(_module, _inputs, kwargs):
        trace.clear()
        if kwargs.get("use_cache") is not False or kwargs.get("past_key_values") is not None:
            raise RuntimeError("recursive CED evaluation requires full recomputation without KV cache")

    def record(index):
        def hook(_module, _inputs, _output):
            trace.append(index)
        return hook

    def finish(module, _inputs, output):
        if trace != list(TEXT_MODEL_SCHEDULE):
            raise RuntimeError(f"non-MeSH CED actual physical schedule mismatch: {trace}")
        if output.past_key_values is not None:
            raise RuntimeError("recursive CED evaluation unexpectedly returned a KV cache")
        actual = []
        if spec["layer_order"]:
            actual = [list(item) for item in module.model.last_injection_trace]
        elif spec["reinjection"]:
            actual = list(module.model.last_reinject_trace)
        if actual != expected_injection:
            raise RuntimeError(f"recursive CED injection trace mismatch: {actual}")
        audit.update(status="PASS", checked_forwards=audit["checked_forwards"] + 1,
                     last_sequence_tokens=int(output.logits.shape[1]),
                     last_physical_schedule=list(trace), last_injection_trace=actual)

    lm.register_forward_pre_hook(begin, with_kwargs=True)
    for index, layer in enumerate(lm.model.layers):
        layer.register_forward_hook(record(index))
    lm.register_forward_hook(finish)
    return audit


def load_runtime(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    variant_spec(args.eval_variant)
    if not torch.cuda.is_available():
        raise RuntimeError("CED Mizar S1 evaluation requires one CUDA GPU")
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
                                   args.expected_geometry, eval_variant=args.eval_variant)
    assets = audit_assets(contract)
    model = modules["models.mellow"].Mellow(
        audioenc_name="CEDSmall", d_in=384, d_out=HIDDEN_SIZE,
        text_decoder=contract["text_model_dir"], prefix_length=PREFIX_TOKENS,
        freeze_text_decoder_weights=False, use_pretrained_audioencoder=True,
        freeze_audio_encoder_weights=True, pretrained_audioencoder_path=contract["ced_model_dir"])
    model.float()
    state_dict = saved["state_dict"]
    keys = list(state_dict)
    ddp = bool(keys and all(str(key).startswith("module.") for key in keys))
    if ddp:
        state_dict = {str(key)[len("module."):]: value for key, value in state_dict.items()}
    elif any(str(key).startswith("module.") for key in keys):
        raise ValueError("checkpoint has a mixed DDP state_dict namespace")
    model.load_state_dict(state_dict, strict=True)
    recursive_audit = audit_recursive_route(route_contract, contract, model, eval_variant=args.eval_variant)
    model.to(device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(contract["text_model_dir"], local_files_only=True)
    checkpoint_token_audit = route_contract.validate_checkpoint_tokenizer(tokenizer, model.caption_decoder.lm.config.vocab_size)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    pad_id = tokenizer.encode(tokenizer.pad_token)[0]
    token_audit = route_contract.validate_tokenizer_contract(
        tokenizer, pad_id, model.caption_decoder.lm.get_input_embeddings().num_embeddings)
    token_audit["checkpoint_source"] = checkpoint_token_audit
    if any(module.training for module in model.modules()):
        raise RuntimeError("all CED inference modules must be in eval mode")
    if {p.dtype for p in model.parameters()} != {torch.float32}:
        raise RuntimeError("all CED inference parameters must be float32")
    text_encoder = modules["data.audiotext_dataset"].AudioTextDataset.__new__(modules["data.audiotext_dataset"].AudioTextDataset)
    text_encoder.tokenizer = tokenizer
    trace_audit = install_recursive_trace_audit(model.caption_decoder.lm, eval_variant=args.eval_variant)
    audit = {
        "status": "PASS", "eval_variant": args.eval_variant,
        "checkpoint_step": saved["total_step"], "checkpoint_epoch": saved.get("epoch"),
        "contract_sha256": saved["contract_sha256"], "training_code_sha256": contract["code_sha256"],
        "strict_state_dict": True, "ddp_state_dict_prefix_normalized": ddp,
        "state_dict_source": "training_checkpoint_state_dict", "training_route": saved["route_variant"],
        "training_geometry": contract["geometry"], "training_loss_reduction": saved["loss_reduction"],
        "expected_geometry": list(parse_training_geometry(args.expected_geometry)),
        "tokenizer": token_audit, "assets": assets, "all_modules_eval": True,
        "ced_all_eval": all(not module.training for module in model.audio_encoder.base.ced.modules()),
        "ced_parameters_frozen": all(not p.requires_grad for p in model.audio_encoder.base.ced.parameters()),
        "dtype": "float32", "deterministic_greedy": True, "tf32": False,
        "recursive_audit": recursive_audit, "recursive_forward_audit": trace_audit,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
    }
    del saved, state_dict
    return model, tokenizer, device, text_encoder, modules["data.fixed20"], audit, modules


def greedy_decode(model, tokenizer, prefix, max_new_tokens=MAX_NEW_TOKENS, *,
                  eval_variant, audio_information=None):
    import torch

    spec = variant_spec(eval_variant)
    if max_new_tokens != MAX_NEW_TOKENS:
        raise ValueError("both benchmarks use exactly max_new_tokens=300")
    if tuple(prefix.shape) != (1, PREFIX_TOKENS, HIDDEN_SIZE):
        raise ValueError(f"CED prefix shape differs from training: {tuple(prefix.shape)}")
    forward_kwargs = {}
    if spec["layer_order"]:
        if not isinstance(audio_information, (tuple, list)) or len(audio_information) != 2:
            raise ValueError("layer injection evaluation requires explicit (A4,A8) for this sample")
        for value in audio_information:
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (1, AUDIO_TOKENS, HIDDEN_SIZE)
                    or value.device != prefix.device or value.dtype != prefix.dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("A4/A8 shape, dtype, device or finite-value mismatch")
        forward_kwargs["audio_information"] = audio_information
    elif audio_information is not None:
        raise ValueError("this CED variant does not accept external layer audio_information")
    generated_ids, generated = [], prefix
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        output = model.caption_decoder.lm(
            inputs_embeds=generated, attention_mask=None, labels=None, use_cache=False,
            **forward_kwargs)
        logits = output.logits[:, -1, :]
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite CED generation logits")
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
