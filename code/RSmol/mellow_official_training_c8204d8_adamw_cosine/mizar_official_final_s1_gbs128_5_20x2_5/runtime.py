"""GPU-safe strict loading and execution audits; conversion artifacts stay immutable."""
from collections.abc import Mapping
from pathlib import Path

import torch
import transformers

from contract import (TEXT_MODEL_SCHEDULE, SERIAL_INJECTION_CONTRACT,
                      model_parameter_role)
from mizar_official_recursive_5_20x2_5.audit import read_json
from mizar_official_recursive_5_20x2_5.contract import TOTAL_PARAMETERS
from mizar_official_recursive_5_20x2_5.loop_lora import validate_loop_lora
from mizar_official_recursive_5_20x2_5.tensor_runtime import (
    construct, equal_inherited, structure_audit)


def load_model(directory):
    """Caller must first audit all artifact files collectively (write_config + rank0).

    Constructors use isolated CPU RNG, then all 1038 tensors overwrite their
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
    if (not isinstance(state, Mapping) or len(state) != 1038 or set(state) != set(current)
            or any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in state.items())):
        raise ValueError("requires the complete plain 1038-tensor model state")
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
            or not encoder.mapper.training or not encoder.layer_mlps.training
            or set(encoder.layer_mlps) != {"4", "8"}
            or {id(p) for p in encoder.layer_mlps["4"].parameters()}
               & {id(p) for p in encoder.layer_mlps["8"].parameters()}):
        raise ValueError("requires frozen/eval CED and independently trainable final mapper/MLPs")
    for name, p in model.named_parameters():
        if p.requires_grad != (model_parameter_role(name)[1] is not None):
            raise ValueError(f"unexpected trainability: {name}")
    if core.gradient_checkpointing or model.caption_decoder.lm.config.use_cache:
        raise ValueError("gradient checkpointing and KV cache must be disabled")
    report["loop_lora"] = validate_loop_lora(core)
    return report


def execution_audit(model, batch_size):
    """Use actual traces and CED shape evidence, without modifying model files."""
    core = model.caption_decoder.lm.model
    expected_lora = [(logical, physical, 0 if logical < 25 else 1)
                     for logical, physical in enumerate(TEXT_MODEL_SCHEDULE) if 5 <= physical < 25]
    ced = model.audio_encoder.base.last_audio_layout
    if (tuple(core.last_layer_trace) != TEXT_MODEL_SCHEDULE
            or list(core.last_loop_lora_trace) != expected_lora
            or [list(x) for x in core.last_serial_injection_trace] != SERIAL_INJECTION_CONTRACT["trace"]
            or core.last_input_projection_calls != 1
            or not isinstance(ced, dict) or ced.get("embedding") != [batch_size, 126, 576]
            or ced.get("grid") != [batch_size, 126, 4, 384]
            or ced.get("hidden") != [batch_size, 504, 384] or not ced.get("encoder_eval")
            or ced.get("layer_audio") != {"4": [batch_size, 126, 576], "8": [batch_size, 126, 576]}):
        raise ValueError("actual 50-call / two-loop / serial84 / single-projection / CED layout mismatch")
    return {"audio_tokens": 126, "sep_tokens": 1, "prompt_tokens": 256, "answer_tokens": 300,
            "prefix_tokens": 383, "total_tokens": 683, "batch_size": batch_size,
            "physical_layers": 30, "logical_layers": 50,
            "layer_trace": list(core.last_layer_trace),
            "loop_lora_trace": [list(x) for x in core.last_loop_lora_trace],
            "serial_injection_trace": [list(x) for x in core.last_serial_injection_trace],
            "input_projection_calls": core.last_input_projection_calls, "ced_layout": ced}
