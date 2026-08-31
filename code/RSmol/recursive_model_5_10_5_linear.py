"""SmolLM2 5-10-5 linear Llama model with an explicit 20-layer schedule.

The linear 5-10-5 model owns twenty physical decoder modules.  Physical modules
0--4 are the prefix, 5--14 are the sampled middle, and 15--19 are the suffix.
Every physical module executes exactly once, in this schedule::

    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

This file is deliberately separate from both the 15R and recursive 5-10-5
implementations.  The cache view is retained for API compatibility, but each
physical layer has one logical cache slot and is called once.
"""

from __future__ import annotations

import inspect
import warnings
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
from torch import nn

try:
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaDecoderLayer,
        LlamaForCausalLM,
        LlamaPreTrainedModel,
    )
except ImportError as exc:  # pragma: no cover - dependency-less static checkout
    raise ImportError(
        "code.RSmol.recursive_model_5_10_5_linear requires torch and transformers"
    ) from exc


LOGICAL_LAYER_COUNT = 20
PHYSICAL_LAYER_COUNT = 20
PREFIX_LAYER_COUNT = 5
MIDDLE_LAYER_COUNT = 10
SUFFIX_LAYER_COUNT = 5
RECURSIVE_LOOPS = 1
MAPPING_POLICY = "explicit_5_10_5_linear_source_layers"
SUPPORTED_TRANSFORMERS_VERSION = "4.54.1"
SOURCE_LAYER_INDICES_0BASED = (
    0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,
    25, 26, 27, 28, 29,
)
SOURCE_LAYER_INDICES_1BASED = tuple(index + 1 for index in SOURCE_LAYER_INDICES_0BASED)
SOURCE_MAPPING_0BASED = SOURCE_LAYER_INDICES_0BASED
SOURCE_MAPPING_1BASED = SOURCE_LAYER_INDICES_1BASED
LOGICAL_TO_PHYSICAL = tuple(range(PHYSICAL_LAYER_COUNT))
LOGICAL_TO_PHYSICAL_SCHEDULE = LOGICAL_TO_PHYSICAL


def build_5_10_5_linear_schedule(
    *, logical_layer_count: int = LOGICAL_LAYER_COUNT,
    physical_layer_count: int = PHYSICAL_LAYER_COUNT,
) -> tuple[int, ...]:
    """Validate and return the one true 20-entry linear schedule."""

    if int(logical_layer_count) != LOGICAL_LAYER_COUNT:
        raise ValueError(f"5-10-5 linear requires logical depth 20, got {logical_layer_count}")
    if int(physical_layer_count) != PHYSICAL_LAYER_COUNT:
        raise ValueError(f"5-10-5 requires 20 physical layers, got {physical_layer_count}")
    schedule = tuple(LOGICAL_TO_PHYSICAL)
    if len(schedule) != LOGICAL_LAYER_COUNT:
        raise AssertionError("5-10-5 linear schedule must have exactly 20 logical entries")
    if any(index < 0 or index >= physical_layer_count for index in schedule):
        raise AssertionError("5-10-5 linear schedule contains an invalid physical index")
    if schedule[:PREFIX_LAYER_COUNT] != tuple(range(5)):
        raise AssertionError("5-10-5 linear prefix schedule mismatch")
    if schedule[PREFIX_LAYER_COUNT:PREFIX_LAYER_COUNT + MIDDLE_LAYER_COUNT] != tuple(range(5, 15)):
        raise AssertionError("5-10-5 linear middle schedule mismatch")
    if schedule[PREFIX_LAYER_COUNT + MIDDLE_LAYER_COUNT:] != tuple(range(15, 20)):
        raise AssertionError("5-10-5 linear suffix schedule mismatch")
    return schedule


def build_5_10_5_linear_source_mapping(num_hidden_layers: int) -> tuple[int, ...]:
    """Return the fixed source mapping after checking source depth is 30."""

    if int(num_hidden_layers) != 30:
        raise ValueError(
            "SmolLM2-5-10-5 linear conversion requires source num_hidden_layers=30; "
            f"got {num_hidden_layers}"
        )
    return SOURCE_LAYER_INDICES_0BASED


def logical_slot_for_execution(logical_index: int, physical_index: int) -> int:
    """Map one schedule entry to its explicit logical KV-cache slot."""

    logical_index = int(logical_index)
    physical_index = int(physical_index)
    schedule = build_5_10_5_linear_schedule()
    if not 0 <= logical_index < len(schedule) or schedule[logical_index] != physical_index:
        raise ValueError(
            f"schedule entry mismatch: logical={logical_index} physical={physical_index}"
        )
    return logical_index


def _config_value(config: Any, name: str, default: Any) -> Any:
    value = getattr(config, name, default)
    return default if value is None else value


def _assert_supported_llama_api(layer: nn.Module) -> None:
    """Validate the singular-cache Llama API used by Transformers 4.54.1."""

    import transformers

    installed = str(getattr(transformers, "__version__", "unknown"))
    if installed != SUPPORTED_TRANSFORMERS_VERSION:
        warnings.warn(
            "SmolLM2 5-10-5 linear was designed for "
            f"transformers=={SUPPORTED_TRANSFORMERS_VERSION}; installed={installed}",
            RuntimeWarning,
            stacklevel=2,
        )
    layer_parameters = inspect.signature(layer.forward).parameters
    if not ({"past_key_value", "past_key_values"} & set(layer_parameters)):
        raise RuntimeError("Unsupported LlamaDecoderLayer cache signature")
    attention_parameters = inspect.signature(layer.self_attn.forward).parameters
    if not ({"past_key_value", "past_key_values"} & set(attention_parameters)):
        raise RuntimeError("Unsupported LlamaAttention cache signature")
    if "layer_idx" not in inspect.signature(DynamicCache.update).parameters:
        raise RuntimeError("Unsupported DynamicCache.update API: missing layer_idx")


class LogicalSlotCacheView:
    """Per-layer cache view translating a physical index to one logical slot.

    A separate view is passed for every schedule entry.  This is intentionally
    explicit instead of looping a 20-layer ModuleList twice: middle physical
    layer ``5`` first updates slot ``5`` and then updates slot ``15``.
    """

    def __init__(self, cache: Any, *, physical_index: int, logical_slot: int) -> None:
        self._cache = cache
        self._physical_index = int(physical_index)
        self._logical_slot = int(logical_slot)

    def _slot(self, layer_idx: int | None = None) -> int:
        if layer_idx is not None and int(layer_idx) != self._physical_index:
            raise ValueError(
                "physical decoder layer index does not match its logical cache view: "
                f"expected={self._physical_index} got={layer_idx}"
            )
        return self._logical_slot

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot = self._slot(layer_idx)
        parameters = inspect.signature(self._cache.update).parameters
        if "cache_kwargs" in parameters:
            return self._cache.update(
                key_states, value_states, slot, cache_kwargs=cache_kwargs, **kwargs
            )
        return self._cache.update(key_states, value_states, slot, **kwargs)

    def get_seq_length(
        self, layer_idx: int = 0, cache_position: torch.LongTensor | None = None
    ) -> int:
        # Llama attention occasionally asks a cache view for its default
        # sequence length without passing ``layer_idx``.  The view itself is
        # already bound to one physical/logical pair, so default 0 must not be
        # interpreted as physical layer zero here.
        slot = self._logical_slot
        method = self._cache.get_seq_length
        parameters = inspect.signature(method).parameters
        if "cache_position" in parameters:
            return int(method(slot, cache_position=cache_position))
        return int(method(slot))

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        method = getattr(self._cache, "get_usable_length", None)
        if method is None:
            return self.get_seq_length(layer_idx)
        slot = self._logical_slot
        names = [item.name for item in inspect.signature(method).parameters.values()]
        if names and names[0] in {"layer_idx", "layer_index"}:
            return int(method(slot, new_seq_length))
        return int(method(new_seq_length, slot))

    def __len__(self) -> int:
        try:
            return len(self._cache)
        except TypeError:
            return 0

    def __iter__(self):
        return iter(self._cache)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cache, name)


def make_dynamic_cache() -> DynamicCache:
    """Use the 4.54.1-compatible lazy cache constructor."""

    return DynamicCache()


def _make_cache(config: LlamaConfig) -> DynamicCache:
    """Compatibility factory that deliberately does not pass config kwargs."""

    del config
    return make_dynamic_cache()


def _cache_seq_length(cache: Any) -> int:
    if cache is None:
        return 0
    method = getattr(cache, "get_seq_length", None)
    if method is None:
        raise TypeError("past_key_values must be a Transformers Cache object")
    try:
        return int(method())
    except (IndexError, KeyError):
        return 0


def _validate_cache_capacity(cache: Any, logical_layer_count: int) -> None:
    if cache is None:
        return
    if not hasattr(cache, "get_seq_length") or not hasattr(cache, "update"):
        raise TypeError("past_key_values must be a Transformers Cache object")
    if isinstance(cache, DynamicCache):
        return  # DynamicCache() grows lazily to all 20 logical slots.
    try:
        capacity = len(cache)
    except TypeError:
        capacity = None
    if capacity is not None and capacity < int(logical_layer_count):
        raise ValueError(
            "past_key_values does not cover 20 logical cache slots: "
            f"capacity={capacity} required={logical_layer_count}"
        )


def _causal_mask(
    *, attention_mask: torch.Tensor | None, batch_size: int, query_length: int,
    past_length: int, dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    total_length = past_length + query_length
    if attention_mask is not None and attention_mask.ndim == 4:
        if attention_mask.shape[-2:] != (query_length, total_length):
            raise ValueError("4-D attention_mask has an invalid query/key shape")
        return attention_mask.to(device=device, dtype=dtype)
    if attention_mask is None:
        key_valid = torch.ones((batch_size, total_length), device=device, dtype=torch.bool)
    elif attention_mask.ndim == 2:
        provided = attention_mask.to(device=device).bool()
        if provided.shape[0] != batch_size:
            raise ValueError("attention_mask batch dimension mismatch")
        if provided.shape[1] == total_length:
            key_valid = provided
        elif provided.shape[1] == query_length:
            key_valid = torch.cat(
                (torch.ones((batch_size, past_length), device=device, dtype=torch.bool), provided), dim=1
            )
        else:
            raise ValueError("2-D attention_mask must cover query or total sequence length")
    else:
        raise ValueError("attention_mask must be rank 2 or rank 4")
    q = torch.arange(past_length, total_length, device=device).view(1, 1, query_length, 1)
    k = torch.arange(total_length, device=device).view(1, 1, 1, total_length)
    allowed = (k <= q) & key_valid.view(batch_size, 1, 1, total_length)
    return torch.zeros((batch_size, 1, query_length, total_length), device=device, dtype=dtype).masked_fill(
        ~allowed, torch.finfo(dtype).min
    )


def _call_decoder_layer(
    layer: nn.Module, hidden_states: torch.Tensor, *, attention_mask: torch.Tensor,
    position_ids: torch.LongTensor, cache: Any, use_cache: bool,
    cache_position: torch.LongTensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    output_attentions: bool,
) -> tuple[Any, ...]:
    parameters = inspect.signature(layer.forward).parameters
    kwargs: dict[str, Any] = {
        "attention_mask": attention_mask, "position_ids": position_ids,
        "use_cache": use_cache,
    }
    accepts_var_kwargs = any(
        item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()
    )
    if "output_attentions" in parameters or accepts_var_kwargs:
        kwargs["output_attentions"] = output_attentions
    if "past_key_value" in parameters:
        kwargs["past_key_value"] = cache
    elif "past_key_values" in parameters:
        kwargs["past_key_values"] = cache
    else:
        raise TypeError("Unsupported LlamaDecoderLayer cache signature")
    if "cache_position" in parameters:
        kwargs["cache_position"] = cache_position
    if "position_embeddings" in parameters:
        kwargs["position_embeddings"] = position_embeddings
    result = layer(hidden_states, **kwargs)
    return tuple(result) if isinstance(result, tuple) else (result,)


class SmolLM2_5_10_5LinearModel(LlamaPreTrainedModel):
    """Llama decoder that executes each of the 20 selected layers once."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        logical = int(getattr(config, "num_hidden_layers", 0))
        physical = int(getattr(config, "recursive_layer_count", 0))
        loops = int(getattr(config, "recursive_loops", 0))
        schedule = tuple(
            getattr(
                config,
                "logical_to_physical",
                getattr(config, "logical_to_physical_schedule", LOGICAL_TO_PHYSICAL),
            )
        )
        if (logical, physical, loops) != (20, 20, 1):
            raise ValueError(
                "SmolLM2-5-10-5 linear requires logical=20, physical=20, loops=1; "
                f"got logical={logical} physical={physical} loops={loops}"
            )
        if str(getattr(config, "recursive_loops_scope", "none")) != "none":
            raise ValueError(
                "5-10-5 linear requires recursive_loops_scope='none'; "
                f"got {getattr(config, 'recursive_loops_scope', None)!r}"
            )
        if schedule != build_5_10_5_linear_schedule():
            raise ValueError(f"Invalid 5-10-5 linear logical_to_physical schedule: {schedule}")
        source_mapping = getattr(config, "recursive_source_layer_indices_0based", None)
        source_layer_count = int(getattr(config, "recursive_source_layer_count", 30))
        if source_layer_count != 30:
            raise ValueError(
                "5-10-5 linear requires source_layer_count=30; "
                f"got {source_layer_count}"
            )
        if source_mapping is None or tuple(int(index) for index in source_mapping) != SOURCE_LAYER_INDICES_0BASED:
            raise ValueError("Invalid 5-10-5 linear source-layer mapping metadata")
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.logical_layer_count = logical
        self.recursive_layer_count = physical
        self.recursive_loops = loops
        self.recursive_loops_scope = "none"
        self.logical_to_physical = schedule
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx=index) for index in range(physical)]
        )
        _assert_supported_llama_api(self.layers[0])
        from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self._supports_output_attentions = "output_attentions" in inspect.signature(
            self.layers[0].forward
        ).parameters
        self.post_init()

    def forward(
        self, input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None, output_attentions: bool | None = None,
        output_hidden_states: bool | None = None, return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None, **kwargs: Any,
    ) -> BaseModelOutputWithPast | tuple[Any, ...]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if kwargs:
            raise TypeError(f"Unsupported SmolLM2_5_10_5LinearModel arguments: {sorted(kwargs)}")
        use_cache = bool(_config_value(self.config, "use_cache", True) if use_cache is None else use_cache)
        output_attentions = bool(_config_value(self.config, "output_attentions", False) if output_attentions is None else output_attentions)
        output_hidden_states = bool(_config_value(self.config, "output_hidden_states", False) if output_hidden_states is None else output_hidden_states)
        return_dict = bool(_config_value(self.config, "use_return_dict", True) if return_dict is None else return_dict)
        if output_attentions and not self._supports_output_attentions:
            raise RuntimeError("Installed LlamaDecoderLayer does not support output_attentions")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))
        hidden_states = inputs_embeds
        batch_size, query_length = hidden_states.shape[:2]
        cache = past_key_values
        if use_cache and cache is None:
            cache = _make_cache(self.config)
        if use_cache:
            _validate_cache_capacity(cache, self.logical_layer_count)
        past_length = _cache_seq_length(cache)
        if cache_position is not None:
            cache_position = cache_position.to(device=hidden_states.device)
            if cache_position.ndim != 1 or cache_position.numel() != query_length:
                raise ValueError("cache_position must match the query length")
        if position_ids is None:
            positions = cache_position if cache_position is not None else torch.arange(
                past_length, past_length + query_length, device=hidden_states.device
            )
            position_ids = positions.unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = position_ids.to(device=hidden_states.device)
            if tuple(position_ids.shape) != (batch_size, query_length):
                raise ValueError(
                    f"position_ids must have shape {(batch_size, query_length)}, got {tuple(position_ids.shape)}"
                )
        if cache_position is None:
            cache_position = position_ids[0]
        causal_mask = _causal_mask(
            attention_mask=attention_mask, batch_size=batch_size, query_length=query_length,
            past_length=past_length, dtype=hidden_states.dtype, device=hidden_states.device,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        all_hidden_states: tuple[torch.Tensor, ...] = ()
        all_attentions: tuple[torch.Tensor, ...] = ()
        # One invocation per logical schedule entry. There is deliberately no
        # second middle-layer pass in the linear comparison architecture.
        for logical_index, physical_index in enumerate(self.logical_to_physical):
            layer = self.layers[physical_index]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_cache = (
                LogicalSlotCacheView(
                    cache, physical_index=physical_index, logical_slot=logical_index
                ) if cache is not None else None
            )
            outputs = _call_decoder_layer(
                layer, hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                cache=layer_cache, use_cache=use_cache, cache_position=cache_position,
                position_embeddings=position_embeddings, output_attentions=output_attentions,
            )
            hidden_states = outputs[0]
            if output_attentions:
                if len(outputs) < 2 or outputs[1] is None:
                    raise RuntimeError("Decoder did not return attentions")
                all_attentions += (outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        present = cache if use_cache else None
        if not return_dict:
            values: tuple[Any, ...] = (hidden_states,)
            if present is not None:
                values += (present,)
            if output_hidden_states:
                values += (all_hidden_states,)
            if output_attentions:
                values += (all_attentions,)
            return values
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=present,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=all_attentions if output_attentions else None,
        )


class SmolLM2_5_10_5LinearForCausalLM(LlamaForCausalLM):
    """Causal-LM head for :class:`SmolLM2_5_10_5LinearModel`."""

    def __init__(self, config: LlamaConfig) -> None:
        LlamaPreTrainedModel.__init__(self, config)
        self.model = SmolLM2_5_10_5LinearModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _prepare_cache_for_generation(self, generation_config: Any, model_kwargs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        implementation = getattr(generation_config, "cache_implementation", None)
        if implementation == "dynamic":
            raise ValueError("cache_implementation='dynamic' is unsupported; use lazy DynamicCache()")
        if model_kwargs.get("past_key_values") is None and bool(getattr(generation_config, "use_cache", True)) and implementation is None:
            model_kwargs["past_key_values"] = make_dynamic_cache()
        return super()._prepare_cache_for_generation(generation_config, model_kwargs, *args, **kwargs)

    def forward(
        self, input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None, use_cache: bool | None = None,
        output_attentions: bool | None = None, output_hidden_states: bool | None = None,
        return_dict: bool | None = None, cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0, **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[Any, ...]:
        loss_kwargs = {}
        if "num_items_in_batch" in kwargs:
            loss_kwargs["num_items_in_batch"] = kwargs.pop("num_items_in_batch")
        if kwargs:
            raise TypeError(f"Unsupported SmolLM2_5_10_5LinearForCausalLM arguments: {sorted(kwargs)}")
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=True, cache_position=cache_position,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)
        result = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )
        if return_dict is None:
            return_dict = bool(_config_value(self.config, "use_return_dict", True))
        return result if return_dict else result.to_tuple()


# Descriptive alias for callers that want the architecture encoded in the
# Python symbol while the checkpoint keeps the stable HF architecture name.
LinearLlamaForCausalLM = SmolLM2_5_10_5LinearForCausalLM


def register_auto_class() -> None:
    """Opt-in process-local registration for linear 5-10-5 checkpoints."""

    try:
        AutoModelForCausalLM.register(LlamaConfig, SmolLM2_5_10_5LinearForCausalLM, exist_ok=True)
    except TypeError:
        AutoModelForCausalLM.register(LlamaConfig, SmolLM2_5_10_5LinearForCausalLM)


def parameter_audit(model: nn.Module) -> dict[str, Any]:
    """Report physical storage and the 20 logical schedule uses."""

    recursive_model = getattr(model, "model", model)
    names = list(model.named_parameters(remove_duplicate=False))
    by_id: dict[int, list[str]] = defaultdict(list)
    for name, parameter in names:
        by_id[id(parameter)].append(name)
    unique_numel = sum(next(parameter for _, parameter in names if id(parameter) == key).numel() for key in by_id)
    return {
        "parameter_count_unique": int(unique_numel),
        "parameter_count_references": int(sum(parameter.numel() for _, parameter in names)),
        "logical_layer_count": LOGICAL_LAYER_COUNT,
        "logical_cache_slot_count": LOGICAL_LAYER_COUNT,
        "physical_layer_count": PHYSICAL_LAYER_COUNT,
        "recursive_layer_count": PHYSICAL_LAYER_COUNT,
        "recursive_loops": RECURSIVE_LOOPS,
        "middle_recurrent_count": 0,
        "schedule": list(LOGICAL_TO_PHYSICAL),
        "schedule_length": len(LOGICAL_TO_PHYSICAL),
        "parameter_storage_unique": len(by_id) == len(list(model.parameters())),
        "middle_physical_indices": list(range(5, 15)),
        "all_physical_layers_independent": True,
        "decoder_layer_parameter_storage_unique": len({id(parameter) for layer in getattr(recursive_model, "layers", ()) for parameter in layer.parameters()}) == sum(1 for layer in getattr(recursive_model, "layers", ()) for _ in layer.parameters()),
        "depth_consistent": len(LOGICAL_TO_PHYSICAL) == LOGICAL_LAYER_COUNT,
        "model_physical_module_count": len(getattr(recursive_model, "layers", ())),
    }


__all__ = [
    "LOGICAL_LAYER_COUNT", "PHYSICAL_LAYER_COUNT", "PREFIX_LAYER_COUNT",
    "MIDDLE_LAYER_COUNT", "SUFFIX_LAYER_COUNT", "RECURSIVE_LOOPS",
    "MAPPING_POLICY", "SUPPORTED_TRANSFORMERS_VERSION", "SOURCE_LAYER_INDICES_0BASED", "SOURCE_LAYER_INDICES_1BASED", "SOURCE_MAPPING_0BASED", "SOURCE_MAPPING_1BASED",
    "LOGICAL_TO_PHYSICAL", "LOGICAL_TO_PHYSICAL_SCHEDULE", "LogicalSlotCacheView", "SmolLM2_5_10_5LinearModel",
    "SmolLM2_5_10_5LinearForCausalLM", "LinearLlamaForCausalLM", "build_5_10_5_linear_schedule",
    "build_5_10_5_linear_source_mapping", "logical_slot_for_execution", "make_dynamic_cache",
    "parameter_audit", "register_auto_class",
]
