"""Four independent audio/text x loop LoRA sets on shared base projections.

The original weight/bias objects and state-dict names survive replacement.
Branch selection only chooses tensors for this invocation; autograd captures
those tensors, so a later invocation cannot change an earlier one's backward.
There is no adapter merging, parameter freezing or weight mutation.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
MIDDLE_LAYERS = tuple(range(5, 15))
RANK, ALPHA, LOOPS = 8, 16, 2
MODALITIES = ("audio", "text")
AUDIO_TOKENS = 126


class LoopLoRALinear(nn.Linear):
    def __init__(self, base: nn.Linear):
        # Do not allocate/reinitialize inherited weights or consume base RNG.
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.rank, self.alpha, self.scaling = RANK, ALPHA, ALPHA / RANK
        self.loop_index = None
        self.audio_tokens = AUDIO_TOKENS
        self.lora_A = nn.ModuleDict({modality: nn.ParameterList([
            nn.Parameter(torch.empty(RANK, self.in_features, dtype=base.weight.dtype, device="cpu"))
            for _ in range(LOOPS)]) for modality in MODALITIES})
        self.lora_B = nn.ModuleDict({modality: nn.ParameterList([
            nn.Parameter(torch.zeros(self.out_features, RANK, dtype=base.weight.dtype, device="cpu"))
            for _ in range(LOOPS)]) for modality in MODALITIES})
        for modality in MODALITIES:
            for matrix in self.lora_A[modality]:
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
            raise ValueError("modality LoRA requires [B,L,D] with audio126 and a nonempty text region")
        audio, text = inputs[:, :self.audio_tokens], inputs[:, self.audio_tokens:]
        audio_delta = F.linear(F.linear(audio, self.lora_A["audio"][loop]), self.lora_B["audio"][loop])
        text_delta = F.linear(F.linear(text, self.lora_A["text"][loop]), self.lora_B["text"][loop])
        delta = torch.cat((audio_delta, text_delta), dim=1)
        self.forward_counts[loop] += 1
        self.last_token_shapes[loop] = (inputs.shape[0], audio.shape[1], text.shape[1])
        return F.linear(inputs, self.weight, self.bias) + self.scaling * delta


def projections(layer):
    return tuple(layer.get_submodule(name) for name in TARGETS)


def install_loop_lora(core):
    if len(core.layers) != 20 or getattr(core, "loop_lora_enabled", False):
        raise ValueError("LoRA requires 20 physical layers and must be installed exactly once")
    if any(isinstance(module, LoopLoRALinear) for module in core.modules()):
        raise ValueError("refusing to replace/reinitialize existing LoRA")
    # One CPU-only fork for all 280 branches. Layers, projections, modalities
    # and rounds draw independently; inherited base objects remain unchanged.
    with torch.random.fork_rng(devices=[]):
        for index in MIDDLE_LAYERS:
            layer = core.layers[index]
            for target in TARGETS:
                parent_name, attr = target.rsplit(".", 1)
                parent = layer.get_submodule(parent_name)
                base = getattr(parent, attr)
                if type(base) is not nn.Linear:
                    raise TypeError(f"expected an unwrapped Linear: layers.{index}.{target}")
                setattr(parent, attr, LoopLoRALinear(base))
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
    seen_parameters, seen_storage, count = set(), set(), 0
    for index, layer in enumerate(core.layers):
        for target, module in zip(TARGETS, projections(layer)):
            if index not in MIDDLE_LAYERS:
                if not isinstance(module, nn.Linear) or isinstance(module, LoopLoRALinear):
                    raise ValueError("prefix/suffix must remain plain Linear without LoRA")
                continue
            if (not isinstance(module, LoopLoRALinear) or module.rank != RANK
                    or module.alpha != ALPHA or module.scaling != 2.
                    or tuple(module.lora_A) != MODALITIES or tuple(module.lora_B) != MODALITIES
                    or module.audio_tokens != AUDIO_TOKENS
                    or not module.weight.requires_grad or module.loop_index is not None):
                raise ValueError(f"invalid shared trainable projection: {index}.{target}")
            for modality in MODALITIES:
                if len(module.lora_A[modality]) != LOOPS or len(module.lora_B[modality]) != LOOPS:
                    raise ValueError("each modality requires exactly two loop adapters")
                for loop in range(LOOPS):
                    a, b = module.lora_A[modality][loop], module.lora_B[modality][loop]
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
                        count += parameter.numel()
    all_lora = [(name, parameter) for name, parameter in core.named_parameters()
                if ".lora_A." in name or ".lora_B." in name]
    if len(all_lora) != 560 or {id(p) for _, p in all_lora} != seen_parameters:
        raise ValueError("LoRA must cover seven projections in ten layers, two modalities and two loops")
    return {"physical_layers_0based": list(MIDDLE_LAYERS), "targets": list(TARGETS),
            "rank": RANK, "alpha": ALPHA, "scale": ALPHA / RANK, "dropout": 0.,
            "loops": LOOPS, "modalities": list(MODALITIES), "adapter_sets": 4,
            "audio_positions": [0, AUDIO_TOKENS], "text_start": AUDIO_TOKENS,
            "matrix_count": len(all_lora), "parameter_count": count,
            "parameter_count_per_set": count // 4,
            "base_trainable": True, "max_lr": 6e-4}


def reset_loop_lora_trace(core):
    """Reset only small Python metadata once per complete LM forward."""
    for index in MIDDLE_LAYERS:
        for module in projections(core.layers[index]):
            if module.loop_index is not None:
                raise RuntimeError("stale LoRA loop scope at forward entry")
            module.forward_counts = [0, 0]
            module.last_token_shapes = [None, None]


def validate_loop_lora_execution(core):
    """Require each projection to apply both modalities once in each round."""
    shape = core.last_memory_shape
    if shape is None or core.last_sequence_length <= AUDIO_TOKENS:
        raise ValueError("missing complete-forward shape metadata")
    expected = (shape[0], AUDIO_TOKENS, core.last_sequence_length - AUDIO_TOKENS)
    for index in MIDDLE_LAYERS:
        for module in projections(core.layers[index]):
            if (module.loop_index is not None or module.forward_counts != [1, 1]
                    or module.last_token_shapes != [expected, expected]):
                raise ValueError("missing, duplicated, wrong-length or stale modality LoRA invocation")
    return {"projection_calls": 140, "adapter_calls": 280,
            "calls_per_set": {f"loop{loop+1}_{m}": 70 for loop in range(LOOPS) for m in MODALITIES},
            "audio_tokens": expected[1], "text_tokens": expected[2], "scope_reset": True}
