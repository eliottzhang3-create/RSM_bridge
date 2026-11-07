"""GPU-safe strict loading and execution audits; conversion artifacts stay immutable."""
from collections.abc import Mapping
from pathlib import Path

import torch
import transformers

from contract import (TEXT_MODEL_SCHEDULE,
                      model_parameter_role)
from mizar_official_recursive_5_20x2_5_input_only.audit import read_json
from mizar_official_recursive_5_20x2_5_input_only.contract import TOTAL_PARAMETERS
from mizar_official_recursive_5_20x2_5_input_only.tensor_runtime import (
    construct, equal_inherited, structure_audit)


def load_model(directory):
    """Caller must first audit all artifact files collectively (write_config + rank0).

    Constructors use isolated CPU RNG, then all 458 tensors overwrite their
    temporary initialization. No CUDA environment or model RNG is reset here.
    """
    if transformers.__version__ != "4.54.1" or torch.get_default_dtype() != torch.float32:
        raise RuntimeError("requires mellow_c8204d8 / transformers4.54.1 / FP32 default dtype")
    root = Path(directory).resolve()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = construct(read_json(root / "config.json"), root / "ced-small")
        model.enable_new_modules()
    state = torch.load(root / "model.ckpt", map_location="cpu", weights_only=True)
    validate_state(state, model)
    model.load_state_dict(state, strict=True)
    equal_inherited(model, state)
    del state
    model.train()
    structure_audit(model, initialized=True)
    return model


def validate_state(state, model):
    current = model.state_dict()
    if (not isinstance(state, Mapping) or len(state) != 458 or set(state) != set(current)
            or any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in state.items())):
        raise ValueError("requires the complete plain 458-tensor model state")
    for name, value in state.items():
        if (value.shape != current[name].shape or value.dtype != current[name].dtype
                or value.device.type != "cpu"
                or (value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all()))):
            raise ValueError(f"invalid saved tensor shape/dtype/device/value: {name}")
    # Both keys must agree before strict loading; tied assignment must not hide
    # a contradictory lm_head or embedding in a malformed full checkpoint.
    if not torch.equal(state["caption_decoder.lm.model.embed_tokens.weight"],
                       state["caption_decoder.lm.lm_head.weight"]):
        raise ValueError("saved tied embedding/lm_head values disagree")


def model_audit(model):
    report = structure_audit(model, initialized=True)
    encoder, core = model.audio_encoder.base, model.caption_decoder.lm.model
    if (sum(p.numel() for p in model.parameters()) != TOTAL_PARAMETERS
            or any(m.training for m in encoder.ced.modules())
            or not encoder.mapper.training):
        raise ValueError("requires frozen/eval CED and trainable inherited final mapper")
    for name, p in model.named_parameters():
        if p.requires_grad != (model_parameter_role(name)[1] is not None):
            raise ValueError(f"unexpected trainability: {name}")
    if core.gradient_checkpointing or model.caption_decoder.lm.config.use_cache:
        raise ValueError("gradient checkpointing and KV cache must be disabled")
    return report


def execution_audit(model, batch_size):
    """Use actual traces and CED shape evidence, without modifying model files."""
    core = model.caption_decoder.lm.model
    ced = model.audio_encoder.base.last_audio_layout
    if (tuple(core.last_layer_trace) != TEXT_MODEL_SCHEDULE
            or tuple(core.last_input_injection_trace) != (5, 25)
            or core.last_input_projection_calls != 1
            or not isinstance(ced, dict) or ced.get("embedding") != [batch_size, 126, 576]
            or ced.get("grid") != [batch_size, 126, 4, 384]
            or ced.get("hidden") != [batch_size, 504, 384] or not ced.get("encoder_eval")
            or ced.get("intermediate_capture") is not False or ced.get("encoder_feature_calls") != 1):
        raise ValueError("actual 50-call / two-loop / input-only / single-projection / CED layout mismatch")
    return {"audio_tokens": 126, "sep_tokens": 1, "prompt_tokens": 256, "answer_tokens": 300,
            "prefix_tokens": 383, "total_tokens": 683, "batch_size": batch_size,
            "physical_layers": 30, "logical_layers": 50,
            "layer_trace": list(core.last_layer_trace),
            "input_injection_trace": list(core.last_input_injection_trace),
            "input_projection_calls": core.last_input_projection_calls, "ced_layout": ced}
