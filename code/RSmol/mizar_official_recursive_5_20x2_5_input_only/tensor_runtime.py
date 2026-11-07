"""CPU tensor validation, strict model construction and artifact reload API."""
from __future__ import annotations

import copy
import os
from collections import Counter
from collections.abc import Mapping

from .audit import audit_artifact, read_json, release_manifest
from .contract import (AUDIO_PARAMETERS, BASE_PARAMETERS, NEW_TENSORS,
                       SCHEDULE, INPUT_TRACE, TOTAL_PARAMETERS, build_text_config)


def dependencies(threads=8):
    # Set before importing the tensor/Hub libraries. Constructor assets are local
    # pinned code, so their trust_remote_code=True never fetches remote code.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import torchaudio
    import transformers
    if transformers.__version__ != "4.54.1":
        raise RuntimeError("conversion/reload requires transformers==4.54.1 (mellow_c8204d8)")
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("conversion requires the FP32 default dtype")
    torch.set_num_threads(threads)
    return torch, {"torch": torch.__version__, "torchaudio": torchaudio.__version__,
                   "transformers": transformers.__version__, "device": "cpu",
                   "threads": threads, "attention": "eager", "precision": "float32"}


def config_from_dict(value):
    from transformers import LlamaConfig
    # Transformers from_dict can insert runtime keys into its input dictionary;
    # constructors can also retain/mutate nested values. Never pass the canonical
    # source/persisted config itself across that mutable boundary.
    config = LlamaConfig.from_dict(copy.deepcopy(value))
    config._attn_implementation = "eager"
    return config


def construct(config, ced_dir):
    from .model import MizarInputOnlyRecursive
    return MizarInputOnlyRecursive(config_from_dict(config), ced_dir).float().cpu()


def validate_source_state(state):
    import torch
    release = release_manifest()
    if (not isinstance(state, Mapping) or len(state) != release["tensor_count"]
            or any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in state.items())
            or any(k.startswith("module.") for k in state)
            or sum(v.numel() for v in state.values()) != release["tensor_elements"]):
        raise ValueError("requires official plain 454-tensor state_dict; wrappers/DDP are rejected")
    for key, value in state.items():
        if value.device.type != "cpu" or (value.is_floating_point() and value.dtype != torch.float32):
            raise ValueError(f"source tensor must be CPU / FP32: {key}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"nonfinite source tensor: {key}")
    return {"tensor_count": len(state), "tensor_elements": sum(v.numel() for v in state.values()),
            "plain_state_dict": True, "weights_only": True, "floating_dtype": "float32"}


def equal_inherited(model, state):
    import torch
    current = model.state_dict()
    for name, value in state.items():
        if (name not in current or current[name].shape != value.shape
                or current[name].dtype != value.dtype or not torch.equal(current[name], value)):
            raise ValueError(f"inherited tensor changed: {name}")
    return {"inherited_tensors_equal": len(state), "comparison": "torch.equal every tensor"}


def assert_input_only(model):
    """Reject hidden or frozen remnants as well as unexpected state keys."""
    forbidden = ("lora_", "audio_gates", "layer_mlps", "router", "memory")
    if any(any(token in name for token in forbidden) for name in model.state_dict()):
        raise ValueError("input-only model contains removed LoRA/intermediate/memory parameters")
    for module in model.modules():
        if any(hasattr(module, name) for name in ("lora_A", "lora_B", "audio_gates", "layer_mlps", "loop_lora_enabled")):
            raise ValueError("input-only model contains a removed module, even without state tensors")


def structure_audit(model, *, initialized, inherited=None):
    import torch
    from .model import trainability_audit
    core, lm = model.caption_decoder.lm.model, model.caption_decoder.lm
    assert_input_only(model)
    count = sum(p.numel() for p in model.parameters())
    expected = TOTAL_PARAMETERS if initialized else BASE_PARAMETERS
    if count != expected or len(core.layers) != 30 or len({id(x) for x in core.layers}) != 30:
        raise ValueError(f"physical layers or unique parameter count mismatch: {count} != {expected}")
    if (lm.get_input_embeddings().weight is not lm.get_output_embeddings().weight
            or tuple(lm.get_input_embeddings().weight.shape) != (49152, 576)):
        raise ValueError("requires tied 49152 x 576 input embeddings / lm_head")
    if any("router" in name or "memory" in name for name in model.state_dict()):
        raise ValueError("this route has no memory or routers")
    if {p.dtype for p in model.parameters()} != {torch.float32}:
        raise ValueError("all parameters must remain FP32")
    encoder = model.audio_encoder.base
    if (encoder.ced.training or any(p.requires_grad for p in encoder.ced.parameters())
            or encoder.mapper.resample or encoder.mapper.output_tokens != 126):
        raise ValueError("CED must remain frozen/eval with the inherited freq_merge_126 mapper")
    result = {"physical_layers": 30, "logical_layers": 50, "parameter_count": count,
              "tied_embeddings": True, "trainability": trainability_audit(model)}
    if not initialized:
        if len(model.state_dict()) != 454:
            raise ValueError("base constructor must have exactly the 454 official keys")
        return result
    roles = Counter()
    for group in result["trainability"]:
        roles[group["role"]] += group["parameters"]
    if roles["new_audio"] != AUDIO_PARAMETERS:
        raise ValueError(f"new parameter roles differ from the agreed counts: {roles}")
    if len(model.state_dict()) != 454 + NEW_TENSORS:
        raise ValueError("expected exactly 458 state tensors")
    if inherited is not None:
        result.update(equal_inherited(model, inherited))
        new = {k: v for k, v in model.state_dict().items() if k not in inherited}
        if len(new) != NEW_TENSORS or sum(v.numel() for v in new.values()) != AUDIO_PARAMETERS:
            raise ValueError("unexpected newly registered tensors")
        for name, value in new.items():
            if ".audio_gate." in name:
                if torch.count_nonzero(value):
                    raise ValueError(f"new residual gate must start at zero: {name}")
        if torch.count_nonzero(core.audio_linear.bias):
            raise ValueError("input projection bias must start at zero")
    return result


def load_source(source):
    """Construct the base only, strict-load 454 keys, THEN add fresh modules."""
    import torch
    source_config = read_json(source / "assets/smollm2-135m/config.json")
    config = build_text_config(source_config)
    state = torch.load(source / release_manifest()["checkpoint"]["path"],
                       map_location="cpu", weights_only=True)
    state_report = validate_source_state(state)
    model = construct(config, source / "assets/ced-small")
    base = structure_audit(model, initialized=False)
    if set(model.state_dict()) != set(state):
        raise ValueError("official source namespaces do not exactly match the unextended model")
    model.load_state_dict(state, strict=True)
    equal_inherited(model, state)
    model.enable_new_modules()
    initialized = structure_audit(model, initialized=True, inherited=state)
    return model, state, source_config, config, {"source_state": state_report,
                                               "base": base, "initialized": initialized}


def tokenizer_audit(directory):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    original_size = len(tokenizer)
    tokenizer.add_special_tokens({"pad_token": "!"})
    tokenizer.padding_side = tokenizer.truncation_side = "right"
    if (len(tokenizer) != original_size or len(tokenizer) != 49152
            or tokenizer.pad_token_id != 17 or tokenizer.eos_token_id != 0
            or tokenizer.encode("!", add_special_tokens=False) != [17]):
        raise ValueError("requires unexpanded vocab49152 / PAD17 / EOS=SEP0 tokenizer")
    return {"vocab_size": len(tokenizer), "pad_token_id": 17, "eos_token_id": 0,
            "sep_token_id": 0, "padding_side": "right", "truncation_side": "right"}


def source_alignment(model, inherited, source_config):
    """Compare the same 30-layer computation, with disabled new residuals."""
    import torch
    from transformers import LlamaForCausalLM
    reference = LlamaForCausalLM(config_from_dict(source_config)).float().cpu().eval()
    prefix = "caption_decoder.lm."
    reference.load_state_dict({k[len(prefix):]: v for k, v in inherited.items()
                               if k.startswith(prefix)}, strict=True)
    model.eval()
    lm = model.caption_decoder.lm
    with torch.no_grad():
        ids = torch.arange(130).reshape(1, -1) % 49152
        embeddings = lm.get_input_embeddings()(ids)
        expected = reference(inputs_embeds=embeddings, attention_mask=None, use_cache=False).logits
        actual = lm.lm_head(lm.model._execute(embeddings, single_pass_audit=True).last_hidden_state)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        error = (actual - expected).abs().max().item()
    del reference
    return {"status": "PASS", "scope": "same_30_layer_single_pass_only", "tokens": 130,
            "max_abs_error": error, "rtol": 2e-5, "atol": 2e-5,
            "recursive_50_equals_original_30": False}


def probe_inputs():
    """Deterministic synthetic fixed20 waveform; no dataset or task quality claim."""
    import torch
    time = torch.arange(640000, dtype=torch.float32) / 32000.
    waveform = (.02 * torch.sin(time * (2 * 3.141592653589793 * 440))).unsqueeze(0)
    return {"audio1": waveform[:, :320000], "audio2": waveform[:, 320000:],
            "input": {"input_ids": torch.arange(256).reshape(1, -1) % 49152},
            "answer": {"input_ids": torch.arange(300).reshape(1, -1) % 49152}}


def forward_audit(model):
    import torch
    model.eval()
    core = model.caption_decoder.lm.model
    trace, audio_calls, projection_calls, gate_inputs, loop_inputs = [], [], [], [], []
    handles = []
    for i, layer in enumerate(core.layers):
        handles.append(layer.register_forward_pre_hook(lambda module, args, i=i: trace.append(i)))
    # CED calls forward_features directly, so observe block1 once per batch.
    handles.append(model.audio_encoder.base.ced.blocks[0].register_forward_pre_hook(
        lambda module, args: audio_calls.append("features")))
    handles.append(core.audio_linear.register_forward_hook(
        lambda module, args, result: projection_calls.append((args[0].detach().clone(), result.detach().clone()))))
    handles.append(core.audio_gate.register_forward_hook(
        lambda module, args, result: gate_inputs.append((args[0].detach().clone(), result.detach().clone()))))
    handles.append(core.layers[5].register_forward_pre_hook(
        lambda module, args: loop_inputs.append(args[0].detach().clone())))
    try:
        with torch.no_grad():
            logits = model(probe_inputs()).logits
        if (tuple(logits.shape) != (1, 683, 49152) or not torch.isfinite(logits).all()
                or tuple(trace) != SCHEDULE or tuple(core.last_layer_trace) != SCHEDULE
                or tuple(core.last_input_injection_trace) != INPUT_TRACE
                or core.last_input_projection_calls != 1 or len(projection_calls) != 1
                or len(gate_inputs) != 2 or len(loop_inputs) != 2 or audio_calls != ["features"]):
            raise ValueError("fixed20 / 683-token / 50-call / input-only forward audit failed")
        projected = projection_calls[0][1]
        for (before, gate), after in zip(gate_inputs, loop_inputs):
            torch.testing.assert_close(after[:, :126], before + torch.tanh(gate) * projected, rtol=0, atol=0)
        layout = model.audio_encoder.base.last_audio_layout
        if layout.get("intermediate_capture") is not False:
            raise ValueError("input-only CED must not capture intermediate features")
    finally:
        for handle in handles:
            handle.remove()
    return {"status": "PASS", "input": "synthetic_fixed20_440Hz", "logits_shape": list(logits.shape),
            "layer_trace": trace, "input_injection_trace": list(core.last_input_injection_trace),
            "input_projection_calls": len(projection_calls), "ced_feature_calls": len(audio_calls),
            "input_residual_formula_verified_by_hooks": True, "audio_layout": layout}, logits


def reload_verified(directory, *, threads=8, _pending_reload=False):
    """Strict public loader. Returns trainable FP32 CPU model and file/tensor audit.

    All artifact bytes, architecture, pinned constructor assets and implementation
    are verified before executing CED code or deserializing weights. No optimizer,
    scheduler or cursor is restored. Call model.eval() for inference.
    """
    from pathlib import Path
    dependencies(threads)
    import torch
    audit = audit_artifact(directory, _pending_reload=_pending_reload)
    root = Path(directory).resolve()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = construct(read_json(root / "config.json"), root / "ced-small")
        model.enable_new_modules()
    state = torch.load(root / "model.ckpt", map_location="cpu", weights_only=True)
    if (not isinstance(state, Mapping) or len(state) != 458
            or any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in state.items())):
        raise ValueError("artifact requires a plain complete 458-tensor state_dict")
    for name, value in state.items():
        if value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError(f"invalid saved floating tensor: {name}")
    model.load_state_dict(state, strict=True)
    equal_inherited(model, state)
    model.train()
    audit["tensor_audit"] = structure_audit(model, initialized=True)
    audit["tokenizer"] = tokenizer_audit(root / "tokenizer")
    return model, audit
