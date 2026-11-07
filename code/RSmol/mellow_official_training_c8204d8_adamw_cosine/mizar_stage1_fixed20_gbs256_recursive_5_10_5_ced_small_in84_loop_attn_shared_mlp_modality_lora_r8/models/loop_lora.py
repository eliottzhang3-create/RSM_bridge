"""Loop-specific rank8 LoRA: shared Attention, modality-specific Transformer MLP.

Every layer/projection/round owns its A/B tensors. Only Attention shares its
adapter between audio and text positions. Base weights remain shared/trainable.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

ATTENTION_TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
MLP_TARGETS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
TARGETS = ATTENTION_TARGETS + MLP_TARGETS
MIDDLE_LAYERS = tuple(range(5, 15))
RANK, ALPHA, LOOPS, AUDIO_TOKENS = 8, 16, 2, 126


class LoopLoRALinear(nn.Linear):
    def __init__(self, base: nn.Linear, mode="shared"):
        if mode not in {"shared", "modality"}:
            raise ValueError("LoRA mode must be shared Attention or modality-specific MLP")
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.rank, self.alpha, self.scaling = RANK, ALPHA, ALPHA / RANK
        self.loop_index, self.mode, self.audio_tokens = None, mode, AUDIO_TOKENS
        branches = ("shared",) if mode == "shared" else ("audio", "text")
        self.lora_A = nn.ModuleDict({branch: nn.ParameterList([
            nn.Parameter(torch.empty(RANK, self.in_features, dtype=base.weight.dtype, device="cpu"))
            for _ in range(LOOPS)]) for branch in branches})
        self.lora_B = nn.ModuleDict({branch: nn.ParameterList([
            nn.Parameter(torch.zeros(self.out_features, RANK, dtype=base.weight.dtype, device="cpu"))
            for _ in range(LOOPS)]) for branch in branches})
        for branch in branches:
            for matrix in self.lora_A[branch]:
                nn.init.kaiming_uniform_(matrix, a=math.sqrt(5))
        self.lora_A.to(base.weight.device)
        self.lora_B.to(base.weight.device)
        self.forward_counts = [0, 0]
        self.last_token_shapes = [None, None]

    def forward(self, inputs):
        loop = self.loop_index
        if loop not in (0, 1):
            raise RuntimeError("LoRA projection requires an explicit middle-loop index")
        if inputs.ndim != 3 or inputs.shape[1] <= self.audio_tokens:
            raise ValueError("LoRA requires [B,L,D] with audio126 and a nonempty text region")
        if self.mode == "shared":
            delta = F.linear(F.linear(inputs, self.lora_A["shared"][loop]), self.lora_B["shared"][loop])
        elif self.mode == "modality":
            audio, text = inputs[:, :self.audio_tokens], inputs[:, self.audio_tokens:]
            audio_delta = F.linear(F.linear(audio, self.lora_A["audio"][loop]), self.lora_B["audio"][loop])
            text_delta = F.linear(F.linear(text, self.lora_A["text"][loop]), self.lora_B["text"][loop])
            delta = torch.cat((audio_delta, text_delta), dim=1)
        else:
            raise RuntimeError("invalid LoRA projection mode")
        self.forward_counts[loop] += 1
        self.last_token_shapes[loop] = (inputs.shape[0], self.audio_tokens, inputs.shape[1] - self.audio_tokens)
        return F.linear(inputs, self.weight, self.bias) + self.scaling * delta


def projections(layer):
    return tuple(layer.get_submodule(name) for name in TARGETS)


def install_loop_lora(core):
    if len(core.layers) != 20 or getattr(core, "loop_lora_enabled", False):
        raise ValueError("LoRA requires 20 physical layers and must be installed exactly once")
    if any(isinstance(module, LoopLoRALinear) for module in core.modules()):
        raise ValueError("refusing to replace/reinitialize existing LoRA")
    # CPU-only initialization preserves inherited weights and the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        for index in MIDDLE_LAYERS:
            layer = core.layers[index]
            for target in TARGETS:
                parent_name, attr = target.rsplit(".", 1)
                parent = layer.get_submodule(parent_name)
                base = getattr(parent, attr)
                if type(base) is not nn.Linear:
                    raise TypeError(f"expected an unwrapped Linear: layers.{index}.{target}")
                mode = "shared" if target in ATTENTION_TARGETS else "modality"
                setattr(parent, attr, LoopLoRALinear(base, mode))
    core.loop_lora_enabled = True
    core.last_loop_lora_trace = ()
    return validate_loop_lora(core)


def call_with_loop_lora(layer, loop_index, callback, *args, **kwargs):
    """Set both branches trainable permanently; reset selection after each call.

    Each backward uses the A/B tensors selected in its own forward. Reentrant
    decoder checkpointing would replay without this scope and is unsupported.
    """
    modules = projections(layer)
    if loop_index not in (0, 1) or any(module.loop_index is not None for module in modules):
        raise RuntimeError("invalid or nested LoRA loop selection")
    try:
        for module in modules:
            module.loop_index = loop_index
        return callback(layer, *args, **kwargs)
    finally:
        for module in modules:
            module.loop_index = None


def validate_loop_lora(core):
    if not getattr(core, "loop_lora_enabled", False) or len(core.layers) != 20:
        raise ValueError("middle-loop LoRA has not been installed")
    seen_parameters, seen_storage = set(), set()
    counts = {"attention_shared": 0, "mlp_audio": 0, "mlp_text": 0}
    for index, layer in enumerate(core.layers):
        for target, module in zip(TARGETS, projections(layer)):
            if index not in MIDDLE_LAYERS:
                if not isinstance(module, nn.Linear) or isinstance(module, LoopLoRALinear):
                    raise ValueError("prefix/suffix must remain plain Linear without LoRA")
                continue
            mode = "shared" if target in ATTENTION_TARGETS else "modality"
            branches = ("shared",) if mode == "shared" else ("audio", "text")
            if (not isinstance(module, LoopLoRALinear) or module.mode != mode
                    or module.rank != RANK or module.alpha != ALPHA or module.scaling != 2.
                    or tuple(module.lora_A) != branches or tuple(module.lora_B) != branches
                    or module.audio_tokens != AUDIO_TOKENS or not module.weight.requires_grad
                    or module.loop_index is not None):
                raise ValueError(f"invalid shared trainable projection: {index}.{target}")
            for branch in branches:
                if len(module.lora_A[branch]) != LOOPS or len(module.lora_B[branch]) != LOOPS:
                    raise ValueError("each branch requires two independent round adapters")
                for loop in range(LOOPS):
                    a, b = module.lora_A[branch][loop], module.lora_B[branch][loop]
                    if tuple(a.shape) != (RANK, module.in_features) or tuple(b.shape) != (module.out_features, RANK):
                        raise ValueError("LoRA matrix geometry mismatch")
                    for parameter in (a, b):
                        storage = (parameter.device, parameter.untyped_storage().data_ptr())
                        if (not parameter.requires_grad or id(parameter) in seen_parameters
                                or storage in seen_storage or parameter.device != module.weight.device
                                or parameter.dtype != module.weight.dtype):
                            raise ValueError("LoRA matrices must be independent, trainable and colocated")
                        seen_parameters.add(id(parameter))
                        seen_storage.add(storage)
                        counts["attention_shared" if mode == "shared" else f"mlp_{branch}"] += parameter.numel()
    all_lora = [(name, parameter) for name, parameter in core.named_parameters()
                if ".lora_A." in name or ".lora_B." in name]
    if len(all_lora) != 400 or {id(p) for _, p in all_lora} != seen_parameters:
        raise ValueError("expected 400 A/B tensors across ten layers and two rounds")
    return {"physical_layers_0based": list(MIDDLE_LAYERS), "targets": list(TARGETS),
            "rank": RANK, "alpha": ALPHA, "scale": ALPHA / RANK, "dropout": 0.,
            "loops": LOOPS, "attention": "shared_audio_text", "mlp": "modality_specific",
            "audio_positions": [0, AUDIO_TOKENS], "text_start": AUDIO_TOKENS,
            "matrix_count": len(all_lora), "parameter_count": sum(counts.values()),
            "parameter_count_by_family": counts, "base_trainable": True, "max_lr": 6e-4}


def reset_loop_lora_trace(core):
    for index in MIDDLE_LAYERS:
        for module in projections(core.layers[index]):
            if module.loop_index is not None:
                raise RuntimeError("stale LoRA round scope at forward entry")
            module.forward_counts = [0, 0]
            module.last_token_shapes = [None, None]


def validate_loop_lora_execution(core):
    if core.last_sequence_length <= AUDIO_TOKENS or core.last_batch_size < 1:
        raise ValueError("missing complete-forward shape metadata")
    expected = (core.last_batch_size, AUDIO_TOKENS, core.last_sequence_length - AUDIO_TOKENS)
    for index in MIDDLE_LAYERS:
        for target, module in zip(TARGETS, projections(core.layers[index])):
            mode = "shared" if target in ATTENTION_TARGETS else "modality"
            if (module.mode != mode or module.loop_index is not None or module.forward_counts != [1, 1]
                    or module.last_token_shapes != [expected, expected]):
                raise ValueError("missing, duplicated, wrong-mode/length or stale LoRA invocation")
    return {"projection_calls": 140, "adapter_calls": 200,
            "calls_by_round": {f"loop{loop+1}": {"attention_shared": 40, "mlp_audio": 30, "mlp_text": 30}
                               for loop in range(LOOPS)},
            "audio_tokens": expected[1], "text_tokens": expected[2], "scope_reset": True}
