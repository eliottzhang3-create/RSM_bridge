"""Strict parameter-sharing recursive Llama causal language model.

The target config keeps ``num_hidden_layers=L`` as the logical decoder depth,
while ``recursive_layer_count=K=L/2`` selects the unique physical decoder
layers.  The model executes that same ``ModuleList`` ``recursive_loops`` times.
The cache adapter is important:
Transformers' Llama attention uses a layer index as the cache slot, so each
loop receives a short-lived view that translates physical indices to logical
indices ``loop * K + physical``.  No module or parameter is duplicated.

This module intentionally depends only on the public Hugging Face model classes
and the small amount of cache protocol used by Llama 4.54.x.  It does not use
ms-swift to implement the model.
"""

from __future__ import annotations

import inspect
import copy
import warnings
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
from torch import nn

try:  # Keep import errors useful when the source checkout is inspected offline.
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaDecoderLayer,
        LlamaForCausalLM,
        LlamaPreTrainedModel,
    )
except ImportError as exc:  # pragma: no cover - exercised on dependency-less workstations
    raise ImportError(
        "code.RSmol.recursive_model requires torch and transformers; "
        "install the project's runtime environment before importing the model"
    ) from exc


MAPPING_POLICY = "explicit_1based_odd_plus_last"
DEFAULT_LOOPS = 2
PROJECT_SOURCE_LAYERS = 30
SUPPORTED_TRANSFORMERS_VERSION = "4.54.1"


def build_stepwise_mapping(
    num_hidden_layers: int,
    *,
    source_layer_indices_0based: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return the source-layer indices used to initialise physical layers.

    Production Stepwise conversion is deliberately explicit for the currently
    supported 30-layer SmolLM2 checkpoint: source 1-based layers
    ``[1,3,...,27,30]`` become 0-based ``[0,2,...,26,29]``.  A caller may pass
    an explicit mapping for a tiny local fixture; no mapping is guessed for a
    different model depth.
    """

    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers must be a positive integer, got {num_hidden_layers!r}")
    if num_hidden_layers % 2:
        raise ValueError(
            f"Recursive conversion requires an even source layer count, got L={num_hidden_layers}"
        )
    expected_count = num_hidden_layers // 2
    if source_layer_indices_0based is None:
        if num_hidden_layers != PROJECT_SOURCE_LAYERS:
            raise ValueError(
                "No implicit Stepwise mapping exists for this model depth: "
                f"expected L={PROJECT_SOURCE_LAYERS}, got L={num_hidden_layers}. "
                "Pass an explicit source_layer_indices_0based mapping for a fixture."
            )
        mapping = tuple(range(0, 27, 2)) + (29,)
    else:
        mapping = tuple(int(index) for index in source_layer_indices_0based)
        if len(mapping) != expected_count:
            raise ValueError(
                f"Explicit mapping must contain K=L/2={expected_count} indices, got {len(mapping)}"
            )
        if len(set(mapping)) != len(mapping):
            raise ValueError(f"Explicit mapping contains duplicate indices: {mapping}")
        if any(index < 0 or index >= num_hidden_layers for index in mapping):
            raise ValueError(
                f"Explicit mapping indices must be in [0,{num_hidden_layers}), got {mapping}"
            )
    if len(mapping) != expected_count or len(set(mapping)) != expected_count:
        raise ValueError(f"Invalid Stepwise mapping for L={num_hidden_layers}: {mapping}")
    return mapping


def _config_value(config: Any, name: str, default: Any) -> Any:
    value = getattr(config, name, default)
    return default if value is None else value


class _LogicalCacheView:
    """A per-loop view translating physical Llama layer indices.

    The view is created inside ``forward`` and is never stored on the model or
    the attention module.  Thus a layer has no mutable loop state and each
    invocation has an unambiguous cache namespace.
    """

    def __init__(self, cache: Any, slot_offset: int) -> None:
        self._cache = cache
        self._slot_offset = int(slot_offset)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logical_idx = self._slot_offset + int(layer_idx)
        # 4.54.x takes cache_kwargs; newer releases made it optional.  Inspect
        # the bound method rather than catching arbitrary TypeErrors raised by
        # a failed tensor update (which would otherwise execute the update a
        # second time).
        update_parameters = inspect.signature(self._cache.update).parameters
        if "cache_kwargs" in update_parameters:
            return self._cache.update(
                key_states,
                value_states,
                logical_idx,
                cache_kwargs=cache_kwargs,
                **kwargs,
            )
        return self._cache.update(key_states, value_states, logical_idx, **kwargs)

    def get_seq_length(self, layer_idx: int = 0, cache_position: torch.LongTensor | None = None) -> int:
        logical_idx = self._slot_offset + int(layer_idx)
        method = self._cache.get_seq_length
        parameters = inspect.signature(method).parameters
        if "cache_position" in parameters:
            return int(method(logical_idx, cache_position=cache_position))
        return int(method(logical_idx))

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        method = getattr(self._cache, "get_usable_length", None)
        if method is None:
            return self.get_seq_length(layer_idx)
        logical_idx = self._slot_offset + int(layer_idx)
        parameters = list(inspect.signature(method).parameters.values())
        parameter_names = [parameter.name for parameter in parameters]
        if parameter_names and parameter_names[0] in {"layer_idx", "layer_index"}:
            return int(method(logical_idx, new_seq_length))
        return int(method(new_seq_length, logical_idx))

    def __len__(self) -> int:
        try:
            return len(self._cache)
        except TypeError:
            return 0

    def __iter__(self):
        return iter(self._cache)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cache, name)


def _cache_seq_length(cache: Any) -> int:
    if cache is None:
        return 0
    method = getattr(cache, "get_seq_length", None)
    if method is None:
        raise TypeError(
            "past_key_values must be a Transformers Cache object; legacy tuples "
            "are not accepted because recursive logical slots would be ambiguous"
        )
    try:
        return int(method())
    except (IndexError, KeyError):
        # A newly-created DynamicCache is empty before the first physical
        # layer initializes slot zero.
        return 0


def _validate_cache_capacity(cache: Any, logical_layer_count: int) -> None:
    """Validate an externally supplied cache without rejecting lazy caches.

    ``DynamicCache()`` may start with zero materialized layers and grow on the
    first update.  A non-empty cache, however, must already expose the complete
    logical namespace because a recursive forward addresses slots in both
    loops (``0..L-1``).  This catches a cache pre-created from the old K-depth
    config before an opaque IndexError occurs in the second loop.
    """

    if cache is None:
        return
    if not hasattr(cache, "get_seq_length") or not hasattr(cache, "update"):
        raise TypeError(
            "past_key_values must be a Transformers Cache object; legacy tuples "
            "are not accepted because recursive logical slots would be ambiguous"
        )
    try:
        capacity = len(cache)
    except TypeError:
        capacity = None
    if capacity is not None and capacity not in (0,) and capacity < logical_layer_count:
        raise ValueError(
            "past_key_values does not cover the recursive logical cache namespace: "
            f"capacity={capacity}, required_slots={logical_layer_count}. "
            "Create it with DynamicCache(config=model.config) or pass an empty lazy cache."
        )


def _make_cache(config: LlamaConfig) -> Any:
    # DynamicCache is the native generation cache in the supported release.
    # In 4.54.1, DynamicCache.__init__ exposes ``**kwargs`` and forwards the
    # public ``config=`` argument to Cache.__init__; later releases expose
    # ``config`` explicitly.  Detect both exact signatures without catching a
    # constructor TypeError and retrying with a different call.
    constructor_parameters = inspect.signature(DynamicCache.__init__).parameters
    accepts_config = "config" in constructor_parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in constructor_parameters.values()
    )
    if accepts_config:
        # ``config.num_hidden_layers`` is the logical depth (L), not the
        # number of unique physical modules (K).  The converted config already
        # advertises all 2K logical cache slots, so do not multiply it here.
        # This also keeps a cache pre-created by GenerationMixin compatible
        # with a cache created lazily by this model.
        cache_config = copy.deepcopy(config)
        cache_config.num_hidden_layers = int(config.num_hidden_layers)
        layer_types = getattr(cache_config, "layer_types", None)
        if isinstance(layer_types, (list, tuple)) and layer_types:
            logical_layers = int(config.num_hidden_layers)
            if len(layer_types) not in {logical_layers, 1}:
                raise ValueError(
                    "Target config layer_types must have one entry or one entry per "
                    f"logical layer ({logical_layers}), got {len(layer_types)}"
                )
            cache_config.layer_types = (
                list(layer_types) * logical_layers if len(layer_types) == 1 else list(layer_types)
            )
        return DynamicCache(config=cache_config)
    # Older cache implementations use lazy layer lists and have no config
    # argument.  This branch is explicit and inspectable, rather than a broad
    # TypeError retry that could hide a constructor bug.
    return DynamicCache()


def _assert_supported_llama_api(layer: nn.Module) -> None:
    """Fail early if the installed Llama/cache protocol is not understood.

    The production target is Transformers 4.54.1.  We permit later releases
    when they retain the same named protocol, but emit a warning so a remote
    smoke report cannot be mistaken for validation against the pinned runtime.
    """

    import transformers

    installed = str(getattr(transformers, "__version__", "unknown"))
    if installed != SUPPORTED_TRANSFORMERS_VERSION:
        warnings.warn(
            "RecursiveLlamaModel was designed and validated for "
            f"transformers=={SUPPORTED_TRANSFORMERS_VERSION}; installed={installed}. "
            "Run the remote smoke before relying on cache behavior.",
            RuntimeWarning,
            stacklevel=2,
        )
    layer_parameters = inspect.signature(layer.forward).parameters
    if not ({"past_key_value", "past_key_values"} & set(layer_parameters)):
        raise RuntimeError(
            "Unsupported LlamaDecoderLayer API: expected past_key_value or "
            f"past_key_values, got {tuple(layer_parameters)}"
        )
    attention_parameters = inspect.signature(layer.self_attn.forward).parameters
    if not ({"past_key_value", "past_key_values"} & set(attention_parameters)):
        raise RuntimeError(
            "Unsupported LlamaAttention API: expected past_key_value or "
            f"past_key_values, got {tuple(attention_parameters)}"
        )
    cache_parameters = inspect.signature(DynamicCache.update).parameters
    if "layer_idx" not in cache_parameters:
        raise RuntimeError(
            "Unsupported DynamicCache.update API: missing layer_idx; "
            f"got {tuple(cache_parameters)}"
        )
    seq_parameters = inspect.signature(DynamicCache.get_seq_length).parameters
    seq_argument_names = [name for name in seq_parameters if name != "self"]
    if not seq_argument_names or seq_argument_names[0] not in {"layer_idx", "layer_index"}:
        raise RuntimeError(
            "Unsupported DynamicCache.get_seq_length API: expected the layer index "
            f"as the first argument, got {tuple(seq_parameters)}"
        )


def _causal_mask(
    *,
    attention_mask: torch.Tensor | None,
    batch_size: int,
    query_length: int,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a 4-D additive mask for a cached decoder-only forward."""

    total_length = past_length + query_length
    if attention_mask is not None and attention_mask.ndim == 4:
        if attention_mask.shape[-2:] == (query_length, total_length):
            return attention_mask.to(device=device, dtype=dtype)
        raise ValueError(
            "4-D attention_mask must have shape "
            f"(batch, heads, {query_length}, {total_length}), got {tuple(attention_mask.shape)}"
        )

    if attention_mask is None:
        key_valid = torch.ones((batch_size, total_length), device=device, dtype=torch.bool)
    elif attention_mask.ndim == 2:
        provided = attention_mask.to(device=device).bool()
        if provided.shape[0] != batch_size:
            raise ValueError(
                f"attention_mask batch mismatch: expected {batch_size}, got {provided.shape[0]}"
            )
        if provided.shape[1] == total_length:
            key_valid = provided
        elif provided.shape[1] == query_length:
            prefix = torch.ones((batch_size, past_length), device=device, dtype=torch.bool)
            key_valid = torch.cat((prefix, provided), dim=1)
        else:
            raise ValueError(
                f"2-D attention_mask length must be {query_length} or {total_length}, "
                f"got {provided.shape[1]}"
            )
    else:
        raise ValueError(f"Unsupported attention_mask rank {attention_mask.ndim}; expected 2 or 4")

    query_positions = torch.arange(
        past_length, total_length, device=device, dtype=torch.long
    ).view(1, 1, query_length, 1)
    key_positions = torch.arange(total_length, device=device, dtype=torch.long).view(1, 1, 1, total_length)
    allowed = key_positions <= query_positions
    allowed = allowed & key_valid.view(batch_size, 1, 1, total_length)
    min_value = torch.finfo(dtype).min
    return torch.zeros((batch_size, 1, query_length, total_length), device=device, dtype=dtype).masked_fill(
        ~allowed, min_value
    )


def _call_decoder_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    cache: Any,
    use_cache: bool,
    cache_position: torch.LongTensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    output_attentions: bool,
) -> tuple[Any, ...]:
    """Call LlamaDecoderLayer across the 4.54.x singular-cache API variants."""

    parameters = inspect.signature(layer.forward).parameters
    kwargs: dict[str, Any] = {
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": use_cache,
    }
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if "output_attentions" in parameters or accepts_var_kwargs:
        kwargs["output_attentions"] = output_attentions
    if "past_key_value" in parameters:
        kwargs["past_key_value"] = cache
    elif "past_key_values" in parameters:
        kwargs["past_key_values"] = cache
    else:  # pragma: no cover - protects against an incompatible transformers release
        raise TypeError(f"Unsupported LlamaDecoderLayer cache signature: {parameters}")
    if "cache_position" in parameters:
        kwargs["cache_position"] = cache_position
    if "position_embeddings" in parameters:
        kwargs["position_embeddings"] = position_embeddings
    result = layer(hidden_states, **kwargs)
    return tuple(result) if isinstance(result, tuple) else (result,)


class RecursiveLlamaModel(LlamaPreTrainedModel):
    """Llama decoder with one physical layer stack executed repeatedly."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        loops = int(_config_value(config, "recursive_loops", DEFAULT_LOOPS))
        if loops != DEFAULT_LOOPS:
            raise ValueError(f"Stage 1 requires recursive_loops={DEFAULT_LOOPS}, got {loops}")
        logical_layer_count = int(getattr(config, "num_hidden_layers", 0))
        if logical_layer_count <= 0:
            raise ValueError("Target config num_hidden_layers (logical layer count) must be positive")
        if not hasattr(config, "recursive_layer_count"):
            raise ValueError(
                "Target config is missing recursive_layer_count, the number of unique physical layers"
            )
        physical_layer_count = int(getattr(config, "recursive_layer_count"))
        if physical_layer_count <= 0:
            raise ValueError(
                "Target config recursive_layer_count (unique physical layer count) must be positive"
            )
        expected_logical = physical_layer_count * loops
        if expected_logical != logical_layer_count:
            raise ValueError(
                "Recursive config depth mismatch: logical num_hidden_layers must equal "
                f"recursive_layer_count * recursive_loops ({physical_layer_count} * {loops} = "
                f"{expected_logical}), got {logical_layer_count}"
            )
        self.recursive_loops = loops
        self.logical_layer_count = logical_layer_count
        self.recursive_layer_count = physical_layer_count
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx=index) for index in range(self.recursive_layer_count)]
        )
        _assert_supported_llama_api(self.layers[0])
        decoder_parameters = inspect.signature(self.layers[0].forward).parameters
        self._supports_output_attentions = "output_attentions" in decoder_parameters
        from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast | tuple[Any, ...]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(
                "RecursiveLlamaModel received unsupported forward arguments: "
                f"{unsupported}. Supported arguments are input_ids, attention_mask, "
                "position_ids, inputs_embeds, labels (on the LM head), use_cache, "
                "past_key_values, cache_position, output_attentions, "
                "output_hidden_states, and return_dict."
            )
        use_cache = bool(_config_value(self.config, "use_cache", True) if use_cache is None else use_cache)
        output_attentions = bool(
            _config_value(self.config, "output_attentions", False)
            if output_attentions is None
            else output_attentions
        )
        output_hidden_states = bool(
            _config_value(self.config, "output_hidden_states", False)
            if output_hidden_states is None
            else output_hidden_states
        )
        return_dict = bool(
            _config_value(self.config, "use_return_dict", True) if return_dict is None else return_dict
        )
        if output_attentions and not self._supports_output_attentions:
            raise RuntimeError(
                "This Transformers LlamaDecoderLayer does not expose output_attentions; "
                "use a runtime with the Stage 1-supported 4.54.1 decoder API or set "
                "output_attentions=False."
            )
        if inputs_embeds is None:
            assert input_ids is not None
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
                raise ValueError("cache_position must be a 1-D tensor matching the query length")
        if position_ids is None:
            if cache_position is not None:
                positions = cache_position
            else:
                positions = torch.arange(
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
            attention_mask=attention_mask,
            batch_size=batch_size,
            query_length=query_length,
            past_length=past_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        all_hidden_states: tuple[torch.Tensor, ...] = ()
        all_attentions: tuple[torch.Tensor, ...] = ()

        for loop_index in range(self.recursive_loops):
            loop_cache = _LogicalCacheView(cache, loop_index * self.recursive_layer_count) if cache is not None else None
            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                outputs = _call_decoder_layer(
                    layer,
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    cache=loop_cache,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    output_attentions=output_attentions,
                )
                hidden_states = outputs[0]
                if output_attentions:
                    if len(outputs) < 2 or outputs[1] is None:
                        raise RuntimeError(
                            "LlamaDecoderLayer accepted output_attentions=True but did not "
                            "return attention weights; use an eager attention implementation."
                        )
                    all_attentions += (outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        present = cache if use_cache else None
        if not return_dict:
            # Match BaseModelOutputWithPast.to_tuple(): optional fields are
            # omitted rather than represented by a positional None.
            values: tuple[Any, ...] = (hidden_states,)
            if present is not None:
                values += (present,)
            if output_hidden_states:
                values += (all_hidden_states,)
            if output_attentions:
                values += (all_attentions,)
            return values
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=present,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=all_attentions if output_attentions else None,
        )


class RecursiveLlamaForCausalLM(LlamaForCausalLM):
    """Hugging Face causal-LM head over :class:`RecursiveLlamaModel`."""

    def __init__(self, config: LlamaConfig) -> None:
        # Reproduce LlamaForCausalLM.__init__ while replacing only the decoder.
        LlamaPreTrainedModel.__init__(self, config)
        self.model = RecursiveLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[Any, ...]:
        """Run the recursive decoder and apply the standard causal-LM head."""

        # Keep the model/LM contract explicit.  ``num_items_in_batch`` is a
        # Trainer loss-only kwarg; all other unknown kwargs are rejected by the
        # decoder instead of being silently discarded.
        loss_kwargs = {}
        if "num_items_in_batch" in kwargs:
            loss_kwargs["num_items_in_batch"] = kwargs.pop("num_items_in_batch")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported RecursiveLlamaForCausalLM forward arguments: {unsupported}")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **loss_kwargs,
            )
        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        if return_dict is None:
            return_dict = bool(_config_value(self.config, "use_return_dict", True))
        return result if return_dict else result.to_tuple()


def register_auto_class() -> None:
    """Explicitly register this implementation for ``AutoModelForCausalLM``.

    Registration is opt-in so importing this module never changes how ordinary
    Llama checkpoints load in a caller's process.  Runtime wrappers call this
    before loading a converted checkpoint through the Auto API.
    """

    try:
        AutoModelForCausalLM.register(LlamaConfig, RecursiveLlamaForCausalLM, exist_ok=True)
    except TypeError:  # pragma: no cover - older Auto API
        try:
            AutoModelForCausalLM.register(LlamaConfig, RecursiveLlamaForCausalLM)
        except ValueError as exc:
            raise RuntimeError(
                "Could not register RecursiveLlamaForCausalLM with AutoModelForCausalLM; "
                "the installed Auto API does not permit replacing LlamaConfig"
            ) from exc
    except ValueError as exc:
        raise RuntimeError(
            "Could not register RecursiveLlamaForCausalLM with AutoModelForCausalLM; "
            "verify the installed transformers AutoModel API"
        ) from exc


def parameter_audit(model: nn.Module) -> dict[str, Any]:
    """Report unique parameters and detect duplicate recursive layer storage."""

    try:
        named_parameters = list(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - old torch fallback
        named_parameters = list(model.named_parameters())
    unique_by_id: dict[int, list[str]] = defaultdict(list)
    parameter_by_id: dict[int, nn.Parameter] = {}
    for name, parameter in named_parameters:
        unique_by_id[id(parameter)].append(name)
        parameter_by_id[id(parameter)] = parameter
    unique_numel = sum(parameter.numel() for parameter in parameter_by_id.values())
    shared_unique_numel = sum(
        parameter_by_id[parameter_id].numel()
        for parameter_id, names in unique_by_id.items()
        if len(names) > 1
    )
    shared_reference_numel = sum(
        parameter_by_id[parameter_id].numel() * len(names)
        for parameter_id, names in unique_by_id.items()
        if len(names) > 1
    )
    # ``model.parameters()`` already de-duplicates shared Parameter objects.
    layer_prefix = "model.layers." if hasattr(model, "model") else "layers."
    layer_state_keys = [
        name
        for name in model.state_dict().keys()
        if name.startswith(layer_prefix)
    ]
    layer_param_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(layer_prefix)
    }
    physical_layer_parameter_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(layer_prefix)
    )
    recursive_model = getattr(model, "model", model)
    recursive_loops = int(getattr(recursive_model, "recursive_loops", DEFAULT_LOOPS))
    physical_layer_count = len(getattr(recursive_model, "layers", ()))
    logical_layer_count = int(
        getattr(
            recursive_model,
            "logical_layer_count",
            getattr(getattr(recursive_model, "config", None), "num_hidden_layers", 0),
        )
    )
    recursive_layer_count = int(
        getattr(recursive_model, "recursive_layer_count", physical_layer_count)
    )
    return {
        "parameter_count_unique": int(unique_numel),
        "parameter_count_references": int(sum(parameter.numel() for _, parameter in named_parameters)),
        "shared_parameter_count_unique": int(shared_unique_numel),
        "shared_parameter_count_references": int(shared_reference_numel),
        "shared_parameter_groups": [names for names in unique_by_id.values() if len(names) > 1],
        "unique_layer_parameter_count": int(
            physical_layer_parameter_count
        ),
        # The loops deliberately have no second Parameter object.  This pair
        # distinguishes storage uniqueness from the number of logical layer
        # uses represented by the recursive computation.
        "recursive_shared_parameter_count_unique": int(physical_layer_parameter_count),
        "recursive_shared_parameter_count_logical_references": int(
            physical_layer_parameter_count * recursive_loops
        ),
        "recursive_loops": recursive_loops,
        "logical_layer_count": logical_layer_count,
        "logical_cache_slot_count": logical_layer_count,
        "recursive_layer_count": recursive_layer_count,
        "unique_layer_parameter_objects": len(layer_param_ids),
        "layer_state_key_count": len(layer_state_keys),
        "layer_state_keys": layer_state_keys,
        "physical_layer_count": physical_layer_count,
        "depth_consistent": logical_layer_count == recursive_layer_count * recursive_loops,
    }


__all__ = [
    "MAPPING_POLICY",
    "PROJECT_SOURCE_LAYERS",
    "SUPPORTED_TRANSFORMERS_VERSION",
    "RecursiveLlamaModel",
    "RecursiveLlamaForCausalLM",
    "build_stepwise_mapping",
    "parameter_audit",
    "register_auto_class",
]
