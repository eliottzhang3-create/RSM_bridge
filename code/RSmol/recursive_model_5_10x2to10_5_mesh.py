"""Isolated variable-depth 5-10xT-5 MeSH model (T in [2, 10]).

The twenty physical decoder layers are unchanged from the fixed 5-10x2-5
route.  Only the ten middle layers recur.  Recursive depth is selected once
per forward/micro-step and applies to the complete batch.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaForCausalLM, LlamaPreTrainedModel,
)

from recursive_model_5_10x2_5_mesh import (
    EMBEDDING_SCALE, MEMORY_SLOT_COUNT, PHYSICAL_LAYER_COUNT,
    SOURCE_LAYER_INDICES_0BASED, TRANSITION_ROUTER_QUERY,
    _assert_llama_api, _call_layer, _causal_mask, _cfg, _init_router,
)

PREFIX_LAYER_COUNT = 5
MIDDLE_LAYER_COUNT = 10
SUFFIX_LAYER_COUNT = 5
MIN_RECURSIVE_DEPTH = 2
MAX_RECURSIVE_DEPTH = 10
DEFAULT_RECURSIVE_DEPTH = 2
MIN_LOGICAL_LAYER_COUNT = 30
MAX_LOGICAL_LAYER_COUNT = 110
ROUTER_MODULE_COUNT = 7
ROUTER_PARAMETER_COUNT = 14
MODEL_ARCHITECTURE_CONTRACT = "logical_30_to_110_physical_20_5_10x2to10_5_mesh"


def validate_recursive_depth(depth: int) -> int:
    depth = int(depth)
    if not MIN_RECURSIVE_DEPTH <= depth <= MAX_RECURSIVE_DEPTH:
        raise ValueError(
            f"recursive_depth must be in [{MIN_RECURSIVE_DEPTH}, {MAX_RECURSIVE_DEPTH}], got {depth}"
        )
    return depth


def logical_layer_count(depth: int) -> int:
    return PREFIX_LAYER_COUNT + MIDDLE_LAYER_COUNT * validate_recursive_depth(depth) + SUFFIX_LAYER_COUNT


def build_mesh_schedule(depth: int) -> tuple[int, ...]:
    depth = validate_recursive_depth(depth)
    return (
        tuple(range(0, 5))
        + tuple(range(5, 15)) * depth
        + tuple(range(15, 20))
    )


class VariableDepthMeshLlamaModel(LlamaPreTrainedModel):
    """Twenty physical layers with a sampled 2--10 pass shared middle."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        physical = int(getattr(config, "recursive_layer_count", PHYSICAL_LAYER_COUNT))
        minimum = int(getattr(config, "recursive_min_depth", MIN_RECURSIVE_DEPTH))
        maximum = int(getattr(config, "recursive_max_depth", MAX_RECURSIVE_DEPTH))
        if physical != PHYSICAL_LAYER_COUNT or (minimum, maximum) != (MIN_RECURSIVE_DEPTH, MAX_RECURSIVE_DEPTH):
            raise ValueError(
                "variable MeSH requires physical=20 and recursive depth range [2,10]"
            )
        self.logical_layer_count = MAX_LOGICAL_LAYER_COUNT
        self.recursive_layer_count = PHYSICAL_LAYER_COUNT
        self.recursive_min_depth = MIN_RECURSIVE_DEPTH
        self.recursive_max_depth = MAX_RECURSIVE_DEPTH
        self.default_recursive_depth = validate_recursive_depth(
            int(getattr(config, "recursive_default_depth", DEFAULT_RECURSIVE_DEPTH))
        )
        self.memory_slots = MEMORY_SLOT_COUNT
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx=i) for i in range(PHYSICAL_LAYER_COUNT)]
        )
        _assert_llama_api(self.layers[0])
        from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # Explicit names make the migration contract reviewable and stable.
        self.pre_write = nn.Linear(config.hidden_size, MEMORY_SLOT_COUNT, bias=True)
        self.pre_read = nn.Linear(config.hidden_size, MEMORY_SLOT_COUNT, bias=True)
        self.loop1_write = nn.Linear(config.hidden_size, MEMORY_SLOT_COUNT, bias=True)
        self.loop1_read = nn.Linear(config.hidden_size, MEMORY_SLOT_COUNT, bias=True)
        self.refine_write = nn.Linear(config.hidden_size, MEMORY_SLOT_COUNT, bias=True)
        self.refine_read = nn.Linear(config.hidden_size, MEMORY_SLOT_COUNT, bias=True)
        self.out_read = nn.Linear(config.hidden_size, MEMORY_SLOT_COUNT, bias=True)
        self.gradient_checkpointing = False
        self.audit_mode = False
        self.gradient_audit_mode = False
        self.last_recursive_depth: int | None = None
        self.last_forward_trace: list[dict[str, int]] = []
        self.last_router_call_counts: dict[str, int] = {}
        self.last_core_input_refs: list[torch.Tensor] = []
        self.last_core_output_refs: list[torch.Tensor] = []
        self.last_memory_shape: tuple[int, ...] | None = None
        self.post_init()
        for router in self.router_modules():
            _init_router(router)

    def router_modules(self) -> tuple[nn.Linear, ...]:
        return (
            self.pre_write, self.pre_read, self.loop1_write, self.loop1_read,
            self.refine_write, self.refine_read, self.out_read,
        )

    def set_recursive_depth(self, depth: int) -> None:
        self.default_recursive_depth = validate_recursive_depth(depth)

    def _route(self, router: nn.Linear, query: torch.Tensor, name: str) -> torch.Tensor:
        self.last_router_call_counts[name] = self.last_router_call_counts.get(name, 0) + 1
        with record_function(f"mesh/router_{name}"):
            return F.softmax(router(query).float(), dim=-1).to(dtype=query.dtype)

    @staticmethod
    def _write(memory: torch.Tensor, value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return memory + value.unsqueeze(1) * weights.transpose(1, 2).unsqueeze(-1)

    @staticmethod
    def _read(memory: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (memory * weights.transpose(1, 2).unsqueeze(-1)).sum(dim=1)

    def _run_stack(
        self, hidden: torch.Tensor, layer_indices: Sequence[int], logical_start: int, *,
        attention_mask: torch.Tensor, position_ids: torch.Tensor, cache_position: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        output_attentions: bool, all_attentions: list[torch.Tensor],
    ) -> torch.Tensor:
        for offset, physical_index in enumerate(layer_indices):
            logical_index = logical_start + offset
            self.last_forward_trace.append(
                {"logical_index": logical_index, "physical_index": int(physical_index)}
            )
            outputs = _call_layer(
                self.layers[physical_index], hidden, attention_mask=attention_mask,
                position_ids=position_ids, cache=None, use_cache=False,
                cache_position=cache_position, position_embeddings=position_embeddings,
                output_attentions=output_attentions,
            )
            hidden = outputs[0]
            if output_attentions:
                all_attentions.append(outputs[1])
        return hidden

    def forward(
        self, input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        recursive_depth: int | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast | tuple[Any, ...]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if kwargs:
            raise TypeError(f"unsupported variable-depth MeSH arguments: {sorted(kwargs)}")
        if past_key_values is not None or bool(use_cache):
            raise ValueError(
                "variable-depth MeSH phase-1 training path requires use_cache=False and no past_key_values"
            )
        depth = validate_recursive_depth(
            self.default_recursive_depth if recursive_depth is None else recursive_depth
        )
        output_attentions = bool(
            _cfg(self.config, "output_attentions", False) if output_attentions is None else output_attentions
        )
        output_hidden_states = bool(
            _cfg(self.config, "output_hidden_states", False) if output_hidden_states is None else output_hidden_states
        )
        return_dict = bool(
            _cfg(self.config, "use_return_dict", True) if return_dict is None else return_dict
        )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))
        hidden = inputs_embeds
        batch_size, query_length = hidden.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(query_length, device=hidden.device).unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = position_ids.to(hidden.device)
        if cache_position is None:
            cache_position = position_ids[0]
        mask = _causal_mask(
            attention_mask, batch_size=batch_size, query_length=query_length, past_length=0,
            dtype=hidden.dtype, device=hidden.device,
        )
        position_embeddings = self.rotary_emb(hidden, position_ids=position_ids)
        self.last_recursive_depth = depth
        self.last_forward_trace = []
        self.last_router_call_counts = {}
        self.last_core_input_refs = []
        self.last_core_output_refs = []
        hidden_states: list[torch.Tensor] = [hidden] if output_hidden_states else []
        attentions: list[torch.Tensor] = []

        memory = torch.zeros(
            (batch_size, MEMORY_SLOT_COUNT, query_length, hidden.shape[-1]),
            dtype=hidden.dtype, device=hidden.device,
        )
        memory[:, 0] = hidden
        with record_function("mesh/prefix_5"):
            prefix_output = self._run_stack(
                hidden, range(0, 5), 0, attention_mask=mask, position_ids=position_ids,
                cache_position=cache_position, position_embeddings=position_embeddings,
                output_attentions=output_attentions, all_attentions=attentions,
            )
        memory = self._write(memory, prefix_output, self._route(self.pre_write, prefix_output, "pre_write"))
        hidden = self._read(memory, self._route(self.pre_read, prefix_output, "pre_read"))
        if output_hidden_states:
            hidden_states.append(hidden)

        for loop_index in range(depth):
            if self.gradient_audit_mode and hidden.requires_grad:
                hidden.retain_grad()
                self.last_core_input_refs.append(hidden)
            core = self._run_stack(
                hidden, range(5, 15), PREFIX_LAYER_COUNT + loop_index * MIDDLE_LAYER_COUNT,
                attention_mask=mask, position_ids=position_ids, cache_position=cache_position,
                position_embeddings=position_embeddings, output_attentions=output_attentions,
                all_attentions=attentions,
            )
            if self.gradient_audit_mode and core.requires_grad:
                core.retain_grad()
                self.last_core_output_refs.append(core)
            if loop_index == 0:
                write_router, read_router = self.loop1_write, self.loop1_read
                write_name, read_name = "loop1_write", "loop1_read"
            else:
                write_router, write_name = self.refine_write, "refine_write"
                if loop_index == depth - 1:
                    read_router, read_name = self.out_read, "out_read"
                else:
                    read_router, read_name = self.refine_read, "refine_read"
            memory = self._write(memory, core, self._route(write_router, hidden, write_name))
            hidden = self._read(memory, self._route(read_router, hidden, read_name))
            if output_hidden_states:
                hidden_states.append(hidden)

        suffix_start = PREFIX_LAYER_COUNT + depth * MIDDLE_LAYER_COUNT
        with record_function("mesh/suffix_5"):
            hidden = self._run_stack(
                hidden, range(15, 20), suffix_start, attention_mask=mask,
                position_ids=position_ids, cache_position=cache_position,
                position_embeddings=position_embeddings, output_attentions=output_attentions,
                all_attentions=attentions,
            )
        hidden = self.norm(hidden)
        self.last_memory_shape = tuple(memory.shape)
        if output_hidden_states:
            hidden_states.append(hidden)
        result = BaseModelOutputWithPast(
            last_hidden_state=hidden, past_key_values=None,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            attentions=tuple(attentions) if output_attentions else None,
        )
        return result if return_dict else result.to_tuple()


class RecursiveLlama5_10x2to10_5MeshForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig) -> None:
        LlamaPreTrainedModel.__init__(self, config)
        self.model = VariableDepthMeshLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        for router in self.model.router_modules():
            _init_router(router)

    def set_recursive_depth(self, depth: int) -> None:
        self.model.set_recursive_depth(depth)

    def forward(
        self, input_ids: torch.LongTensor | None = None, attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None, past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None, labels: torch.LongTensor | None = None,
        use_cache: bool | None = None, output_attentions: bool | None = None,
        output_hidden_states: bool | None = None, return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None, logits_to_keep: int | torch.Tensor = 0,
        recursive_depth: int | None = None, **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[Any, ...]:
        loss_kwargs = {}
        if "num_items_in_batch" in kwargs:
            loss_kwargs["num_items_in_batch"] = kwargs.pop("num_items_in_batch")
        if kwargs:
            raise TypeError(f"unsupported variable-depth CausalLM arguments: {sorted(kwargs)}")
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=True, cache_position=cache_position, recursive_depth=recursive_depth,
        )
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, indices, :])
        loss = None
        if labels is not None:
            with record_function("loss"):
                loss = self.loss_function(
                    logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs
                )
        result = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=None,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )
        if return_dict is None:
            return_dict = bool(_cfg(self.config, "use_return_dict", True))
        return result if return_dict else result.to_tuple()


RecursiveLlamaForCausalLM = RecursiveLlama5_10x2to10_5MeshForCausalLM


def parameter_audit(model: nn.Module) -> dict[str, Any]:
    names = list(model.named_parameters(remove_duplicate=False))
    by_id: dict[int, list[str]] = defaultdict(list)
    values: dict[int, nn.Parameter] = {}
    for name, parameter in names:
        by_id[id(parameter)].append(name)
        values[id(parameter)] = parameter
    owner = getattr(model, "model", model)
    router_names = [
        name for name, _ in names
        if any(token in name for token in (
            "pre_write", "pre_read", "loop1_write", "loop1_read",
            "refine_write", "refine_read", "out_read",
        ))
    ]
    return {
        "parameter_count_unique": int(sum(p.numel() for p in values.values())),
        "physical_layer_count": PHYSICAL_LAYER_COUNT,
        "recursive_depth_range": [MIN_RECURSIVE_DEPTH, MAX_RECURSIVE_DEPTH],
        "logical_layer_count_range": [MIN_LOGICAL_LAYER_COUNT, MAX_LOGICAL_LAYER_COUNT],
        "memory_slots": MEMORY_SLOT_COUNT,
        "router_module_count": ROUTER_MODULE_COUNT,
        "router_parameter_tensor_count": len(router_names),
        "router_objects_independent": len({id(r) for r in owner.router_modules()}) == ROUTER_MODULE_COUNT,
        "source_mapping_0based": list(SOURCE_LAYER_INDICES_0BASED),
        "embedding_scale": EMBEDDING_SCALE,
        "transition_query": TRANSITION_ROUTER_QUERY,
        "architecture_contract": MODEL_ARCHITECTURE_CONTRACT,
    }


__all__ = [
    "DEFAULT_RECURSIVE_DEPTH", "MAX_LOGICAL_LAYER_COUNT", "MAX_RECURSIVE_DEPTH",
    "MIN_LOGICAL_LAYER_COUNT", "MIN_RECURSIVE_DEPTH", "MODEL_ARCHITECTURE_CONTRACT",
    "PHYSICAL_LAYER_COUNT", "RecursiveLlama5_10x2to10_5MeshForCausalLM",
    "RecursiveLlamaForCausalLM", "VariableDepthMeshLlamaModel", "build_mesh_schedule",
    "logical_layer_count", "parameter_audit", "validate_recursive_depth",
]
