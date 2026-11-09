"""Two trainable LoRA branches on each shared middle-layer projection.

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


class LoopLoRALinear(nn.Linear):
    def __init__(self, base: nn.Linear):
        # Do not allocate/reinitialize inherited weights or consume base RNG.
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.rank, self.alpha, self.scaling = RANK, ALPHA, ALPHA / RANK
        self.loop_index = None
        self._is_hf_initialized = True
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.empty(RANK, self.in_features, dtype=base.weight.dtype, device="cpu"))
            for _ in range(LOOPS)])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(self.out_features, RANK, dtype=base.weight.dtype, device="cpu"))
            for _ in range(LOOPS)])
        for matrix in self.lora_A:
            nn.init.kaiming_uniform_(matrix, a=math.sqrt(5))
        self.lora_A.to(base.weight.device)
        self.lora_B.to(base.weight.device)

    def forward(self, inputs):
        loop = self.loop_index
        if loop not in (0, 1):
            raise RuntimeError("LoRA projection requires an explicit middle-loop index")
        delta = F.linear(F.linear(inputs, self.lora_A[loop]), self.lora_B[loop])
        return F.linear(inputs, self.weight, self.bias) + self.scaling * delta


def projections(layer):
    return tuple(layer.get_submodule(name) for name in TARGETS)


def install_loop_lora(core):
    if len(core.layers) != 20 or getattr(core, "loop_lora_enabled", False):
        raise ValueError("LoRA requires 20 physical layers and must be installed exactly once")
    if any(isinstance(module, LoopLoRALinear) for module in core.modules()):
        raise ValueError("refusing to replace/reinitialize existing LoRA")
    # One CPU-only fork for all 140 branches. Loops/layers draw independently
    # while the surrounding model/training RNG state remains unchanged.
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
                    or len(module.lora_A) != LOOPS or len(module.lora_B) != LOOPS
                    or not module.weight.requires_grad or module.loop_index is not None):
                raise ValueError(f"invalid shared trainable projection: {index}.{target}")
            for loop in range(LOOPS):
                a, b = module.lora_A[loop], module.lora_B[loop]
                if tuple(a.shape) != (RANK, module.in_features) or tuple(b.shape) != (module.out_features, RANK):
                    raise ValueError("LoRA matrix geometry mismatch")
                for parameter in (a, b):
                    # HF may construct the architecture on meta before loading.
                    # Object independence is checked there; storage is checked
                    # again on the fully materialized CPU/CUDA runtime model.
                    storage = ((parameter.device, parameter.untyped_storage().data_ptr())
                               if parameter.device.type != "meta" else id(parameter))
                    if (not parameter.requires_grad or id(parameter) in seen_parameters
                            or storage in seen_storage or parameter.device != module.weight.device
                            or parameter.dtype != module.weight.dtype):
                        raise ValueError("LoRA matrices must be independent, trainable and colocated")
                    seen_parameters.add(id(parameter))
                    seen_storage.add(storage)
                    count += parameter.numel()
    all_lora = [(name, parameter) for name, parameter in core.named_parameters()
                if ".lora_A." in name or ".lora_B." in name]
    if len(all_lora) != 280 or {id(p) for _, p in all_lora} != seen_parameters:
        raise ValueError("LoRA must cover exactly seven projections in ten layers, twice")
    return {"physical_layers_0based": list(MIDDLE_LAYERS), "targets": list(TARGETS),
            "rank": RANK, "alpha": ALPHA, "scale": ALPHA / RANK, "dropout": 0.,
            "loops": LOOPS, "matrix_count": len(all_lora), "parameter_count": count,
            "base_trainable": True, "lr_policy": "same_as_base"}
