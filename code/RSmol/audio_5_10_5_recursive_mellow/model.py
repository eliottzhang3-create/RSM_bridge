"""Mellow audio adapter for the fixed two-pass SmolLM2 5-10-5 model.

The audio path is deliberately inherited from the original SmolLM2 baseline:
frozen HTSAT, trainable Mellow c2l, the exact 768->576->576 bridge, two
129-token audio prefixes, and answer-only labels.  Only the text backbone and
its architecture/runtime audits differ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from audio_smollm2_135m_mellow.model import (
    AUDIO_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    MAPPER_CONTRACT,
    AudioSmolLM2Config,
    AudioSmolLM2Model,
    _load_mellow_wrapper,
)
from recursive_model_5_10_5 import (
    LOGICAL_LAYER_COUNT,
    LOGICAL_TO_PHYSICAL,
    MIDDLE_LAYER_COUNT,
    PHYSICAL_LAYER_COUNT,
    PREFIX_LAYER_COUNT,
    RECURSIVE_LOOPS,
    SOURCE_LAYER_INDICES_0BASED,
    SUFFIX_LAYER_COUNT,
    RecursiveLlamaForCausalLM,
)


RECURSIVE_HIDDEN_SIZE = 576
RECURSIVE_AUDIO_CONTRACT = (
    "logical_30_physical_20_5_10_5_loops_2_no_mesh_audio_mellow"
)


@dataclass
class AudioRecursive5_10_5Config(AudioSmolLM2Config):
    """Audio settings shared with the established ReasonAQA routes."""

    architecture_contract: str = RECURSIVE_AUDIO_CONTRACT


def validate_recursive_5_10_5(model: nn.Module) -> dict[str, Any]:
    """Fail closed unless ``model`` is the exact fixed 5-10-5 checkpoint."""

    if not isinstance(model, RecursiveLlamaForCausalLM):
        raise TypeError(
            "fixed 5-10-5 audio training requires RecursiveLlamaForCausalLM, "
            f"got {type(model).__module__}.{type(model).__name__}"
        )
    config = model.config
    model_type = str(getattr(config, "model_type", ""))
    hidden_size = int(getattr(config, "hidden_size", -1))
    logical_layers = int(getattr(config, "num_hidden_layers", -1))
    physical_layers = int(getattr(config, "recursive_layer_count", -1))
    loops = int(getattr(config, "recursive_loops", -1))
    loops_scope = str(getattr(config, "recursive_loops_scope", ""))
    prefix_layers = int(getattr(config, "recursive_prefix_layer_count", -1))
    middle_layers = int(getattr(config, "recursive_middle_layer_count", -1))
    suffix_layers = int(getattr(config, "recursive_suffix_layer_count", -1))
    schedule = tuple(
        int(index)
        for index in getattr(
            config,
            "logical_to_physical",
            getattr(config, "logical_to_physical_schedule", ()),
        )
    )
    source_mapping = tuple(
        int(index)
        for index in getattr(config, "recursive_source_layer_indices_0based", ())
    )
    decoder = getattr(getattr(model, "model", None), "layers", None)
    if model_type != "llama":
        raise ValueError(f"fixed 5-10-5 requires model_type='llama', got {model_type!r}")
    if hidden_size != RECURSIVE_HIDDEN_SIZE:
        raise ValueError(
            f"fixed 5-10-5 hidden_size must be {RECURSIVE_HIDDEN_SIZE}, got {hidden_size}"
        )
    if (logical_layers, physical_layers, loops) != (
        LOGICAL_LAYER_COUNT,
        PHYSICAL_LAYER_COUNT,
        RECURSIVE_LOOPS,
    ):
        raise ValueError(
            "fixed 5-10-5 requires logical=30, physical=20, loops=2; "
            f"got logical={logical_layers} physical={physical_layers} loops={loops}"
        )
    if loops_scope != "middle_only":
        raise ValueError(f"fixed 5-10-5 requires loops_scope='middle_only', got {loops_scope!r}")
    if (prefix_layers, middle_layers, suffix_layers) != (
        PREFIX_LAYER_COUNT,
        MIDDLE_LAYER_COUNT,
        SUFFIX_LAYER_COUNT,
    ):
        raise ValueError(
            "fixed 5-10-5 prefix/middle/suffix metadata mismatch: "
            f"{prefix_layers}/{middle_layers}/{suffix_layers}"
        )
    if schedule != tuple(LOGICAL_TO_PHYSICAL):
        raise ValueError(f"fixed 5-10-5 logical schedule mismatch: {schedule}")
    if source_mapping != tuple(SOURCE_LAYER_INDICES_0BASED):
        raise ValueError(f"fixed 5-10-5 source-layer mapping mismatch: {source_mapping}")
    if decoder is None or len(decoder) != PHYSICAL_LAYER_COUNT:
        raise ValueError("fixed 5-10-5 must expose exactly 20 physical decoder modules")
    if len({id(layer) for layer in decoder}) != PHYSICAL_LAYER_COUNT:
        raise ValueError("fixed 5-10-5 physical decoder modules must be distinct objects")
    parameter_names = [name.lower() for name, _ in model.named_parameters(remove_duplicate=False)]
    forbidden = [name for name in parameter_names if "router" in name or "memory" in name]
    if forbidden:
        raise ValueError(f"fixed 5-10-5 contains forbidden MeSH parameters: {forbidden[:8]}")
    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    if input_embedding is None or output_embedding is None:
        raise ValueError("fixed 5-10-5 must expose input and output embeddings")
    input_weight = getattr(input_embedding, "weight", None)
    output_weight = getattr(output_embedding, "weight", None)
    if input_weight is None or output_weight is None:
        raise ValueError("fixed 5-10-5 embeddings must expose weights")
    architecture_names = [
        str(value) for value in (getattr(config, "architectures", None) or [])
    ]
    return {
        "model_type": model_type,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "architectures": architecture_names,
        "hidden_size": hidden_size,
        "num_hidden_layers": logical_layers,
        "logical_layer_count": logical_layers,
        "physical_decoder_layer_count": len(decoder),
        "recursive_layer_count": physical_layers,
        "recursive_loops": loops,
        "recursive_loops_scope": loops_scope,
        "prefix_layer_count": prefix_layers,
        "middle_layer_count": middle_layers,
        "suffix_layer_count": suffix_layers,
        "logical_to_physical": list(schedule),
        "source_layer_indices_0based": list(source_mapping),
        "shared_middle_physical_indices": list(range(5, 15)),
        "distinct_physical_decoder_layers": True,
        "forbidden_custom_parameter_names": [],
        "embedding_lm_head_tied": bool(input_weight.data_ptr() == output_weight.data_ptr()),
        "vocab_size": int(getattr(config, "vocab_size", 0)),
    }


class AudioRecursive5_10_5Model(AudioSmolLM2Model):
    """Exact Mellow composite with a shared-middle fixed recursive LM."""

    validate_text_model = staticmethod(validate_recursive_5_10_5)

    def __init__(
        self,
        text_model: nn.Module,
        tokenizer: Any,
        htsat_wrapper: nn.Module,
        htsat_backbone: nn.Module,
        config: AudioRecursive5_10_5Config | None = None,
    ) -> None:
        super().__init__(
            text_model,
            tokenizer,
            htsat_wrapper,
            htsat_backbone,
            config or AudioRecursive5_10_5Config(),
        )
        self.last_forward_trace: list[int] = []
        self._capture_schedule_once = True

    def forward(self, **kwargs: Any) -> Any:
        if not self._capture_schedule_once:
            return super().forward(**kwargs)
        trace: list[int] = []
        handles = []
        for physical_index, layer in enumerate(self.text_model.model.layers):
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, _output, index=physical_index: trace.append(index)
                )
            )
        succeeded = False
        try:
            output = super().forward(**kwargs)
            succeeded = True
            return output
        finally:
            for handle in handles:
                handle.remove()
            if succeeded:
                self.last_forward_trace = trace
                self._capture_schedule_once = False

    def trainable_parameter_audit(self) -> dict[str, Any]:
        result = super().trainable_parameter_audit()
        layers = list(self.text_model.model.layers)
        result.update(
            {
                "recursive_text_contract": self.text_contract,
                "decoder_layer_count": len(layers),
                "independent_decoder_layers": (
                    len(layers) == PHYSICAL_LAYER_COUNT
                    and len({id(layer) for layer in layers}) == PHYSICAL_LAYER_COUNT
                ),
                "distinct_physical_decoder_layers": (
                    len(layers) == PHYSICAL_LAYER_COUNT
                    and len({id(layer) for layer in layers}) == PHYSICAL_LAYER_COUNT
                ),
                "logical_layer_count": LOGICAL_LAYER_COUNT,
                "recursive_loops": RECURSIVE_LOOPS,
                "has_router_parameters": any(
                    "router" in name.lower() or "memory" in name.lower()
                    for name, _ in self.named_parameters(remove_duplicate=False)
                ),
            }
        )
        if result["has_router_parameters"]:
            result["training_mode_contract"] = False
        return result

    def runtime_gradient_audit(self) -> dict[str, Any]:
        result = super().runtime_gradient_audit()
        expected = list(LOGICAL_TO_PHYSICAL)
        trace = list(self.last_forward_trace)
        counts = {index: trace.count(index) for index in range(PHYSICAL_LAYER_COUNT)}
        invocation_counts_ok = (
            all(counts[index] == 1 for index in range(0, 5))
            and all(counts[index] == 2 for index in range(5, 15))
            and all(counts[index] == 1 for index in range(15, 20))
        )
        no_router_parameters = not any(
            "router" in name.lower() or "memory" in name.lower()
            for name, _ in self.named_parameters(remove_duplicate=False)
        )
        result.update(
            {
                "expected_forward_trace": expected,
                "forward_trace": trace,
                "forward_trace_matches_exact_5_10x2_5": trace == expected,
                "physical_layer_invocation_counts": counts,
                "prefix_middle_suffix_invocation_counts_valid": invocation_counts_ok,
                "no_mesh_router_or_memory_parameters": no_router_parameters,
                "recursive_text_model": self.text_contract,
            }
        )
        if not all(
            (
                result["forward_trace_matches_exact_5_10x2_5"],
                result["prefix_middle_suffix_invocation_counts_valid"],
                result["no_mesh_router_or_memory_parameters"],
            )
        ):
            raise RuntimeError(f"fixed 5-10-5 runtime architecture audit failed: {result}")
        return result


__all__ = [
    "AUDIO_PREFIX_TOKENS",
    "AUDIO_TOKENS_PER_CLIP",
    "MAPPER_CONTRACT",
    "RECURSIVE_AUDIO_CONTRACT",
    "RECURSIVE_HIDDEN_SIZE",
    "AudioRecursive5_10_5Config",
    "AudioRecursive5_10_5Model",
    "_load_mellow_wrapper",
    "validate_recursive_5_10_5",
]
