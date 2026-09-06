"""Isolated MeSH-style SmolLM2 5-10x2-5 causal language model.

The implementation deliberately owns its runtime contract.  The backbone has
twenty physical Llama decoder layers, while the attention cache has thirty
logical slots (the ten middle layers execute twice).  MeSH memory is a
per-forward activation and is never put in the cache or checkpoint.
"""

from __future__ import annotations

import inspect
import math
import warnings
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

try:
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaPreTrainedModel
except ImportError as exc:  # pragma: no cover - static checkout without ML dependencies
    raise ImportError("recursive_model_5_10x2_5_mesh requires torch and transformers") from exc


LOGICAL_LAYER_COUNT = 30
PHYSICAL_LAYER_COUNT = 20
PREFIX_LAYER_COUNT = 5
MIDDLE_LAYER_COUNT = 10
SUFFIX_LAYER_COUNT = 5
RECURSIVE_LOOPS = 2
MEMORY_SLOT_COUNT = 5
ROUTER_COUNT = 3
ROUTER_PARAMETER_COUNT = 6
MAPPING_POLICY = "explicit_5_10_5_source_layers_mesh"
MODEL_ARCHITECTURE_CONTRACT = "logical_30_physical_20_5_10x2_5_mesh"
TRANSITION_ROUTER_QUERY = "prefix_output"
EMBEDDING_SCALE = "disabled"
SUPPORTED_TRANSFORMERS_VERSION = "4.54.1"
SOURCE_LAYER_INDICES_0BASED = (0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29)
SOURCE_LAYER_INDICES_1BASED = tuple(i + 1 for i in SOURCE_LAYER_INDICES_0BASED)
SOURCE_MAPPING_0BASED = SOURCE_LAYER_INDICES_0BASED
SOURCE_MAPPING_1BASED = SOURCE_LAYER_INDICES_1BASED
LOGICAL_TO_PHYSICAL = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
LOGICAL_TO_PHYSICAL_SCHEDULE = LOGICAL_TO_PHYSICAL


def build_mesh_schedule(*, logical_layer_count: int = LOGICAL_LAYER_COUNT, physical_layer_count: int = PHYSICAL_LAYER_COUNT) -> tuple[int, ...]:
    if int(logical_layer_count) != 30 or int(physical_layer_count) != 20:
        raise ValueError("MeSH 5-10x2-5 requires logical_layer_count=30 and physical_layer_count=20")
    if LOGICAL_TO_PHYSICAL[:5] != tuple(range(5)) or LOGICAL_TO_PHYSICAL[5:15] != tuple(range(5, 15)):
        raise AssertionError("invalid prefix/core schedule")
    if LOGICAL_TO_PHYSICAL[15:25] != tuple(range(5, 15)) or LOGICAL_TO_PHYSICAL[25:] != tuple(range(15, 20)):
        raise AssertionError("invalid second-core/suffix schedule")
    return LOGICAL_TO_PHYSICAL


def build_5_10x2_5_mesh_schedule(**kwargs: Any) -> tuple[int, ...]:
    return build_mesh_schedule(**kwargs)


def build_source_mapping(num_hidden_layers: int = 30) -> tuple[int, ...]:
    if int(num_hidden_layers) != 30:
        raise ValueError(f"source mapping requires 30 source layers, got {num_hidden_layers}")
    return SOURCE_LAYER_INDICES_0BASED


def build_5_10x2_5_mesh_source_mapping(num_hidden_layers: int = 30) -> tuple[int, ...]:
    return build_source_mapping(num_hidden_layers)


def _cfg(config: Any, name: str, default: Any) -> Any:
    value = getattr(config, name, default)
    return default if value is None else value


def _assert_llama_api(layer: nn.Module) -> None:
    try:
        import transformers
        installed = str(getattr(transformers, "__version__", "unknown"))
        if installed != SUPPORTED_TRANSFORMERS_VERSION:
            warnings.warn(f"MeSH model targets transformers=={SUPPORTED_TRANSFORMERS_VERSION}; installed={installed}", RuntimeWarning)
    except Exception:
        pass
    if not ({"past_key_value", "past_key_values"} & set(inspect.signature(layer.forward).parameters)):
        raise RuntimeError("unsupported LlamaDecoderLayer cache API")
    if "layer_idx" not in inspect.signature(DynamicCache.update).parameters:
        raise RuntimeError("unsupported DynamicCache.update API")


class LogicalSlotCacheView:
    """Translate a physical layer's cache calls to one of 30 logical slots."""

    def __init__(self, cache: Any, *, physical_index: int, logical_slot: int) -> None:
        self._cache = cache
        self._physical_index = int(physical_index)
        self._logical_slot = int(logical_slot)

    def _slot(self, layer_idx: int | None = None) -> int:
        if layer_idx is not None and int(layer_idx) != self._physical_index:
            raise ValueError(f"cache view physical index mismatch: expected {self._physical_index}, got {layer_idx}")
        return self._logical_slot

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: Mapping[str, Any] | None = None, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        slot = self._slot(layer_idx)
        if "cache_kwargs" in inspect.signature(self._cache.update).parameters:
            return self._cache.update(key_states, value_states, slot, cache_kwargs=cache_kwargs, **kwargs)
        return self._cache.update(key_states, value_states, slot, **kwargs)

    def get_seq_length(self, layer_idx: int = 0, cache_position: torch.LongTensor | None = None) -> int:
        method = self._cache.get_seq_length
        if "cache_position" in inspect.signature(method).parameters:
            return int(method(self._logical_slot, cache_position=cache_position))
        return int(method(self._logical_slot))

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        method = getattr(self._cache, "get_usable_length", None)
        if method is None:
            return self.get_seq_length(layer_idx)
        names = [p.name for p in inspect.signature(method).parameters.values()]
        if names and names[0] in {"layer_idx", "layer_index"}:
            return int(method(self._logical_slot, new_seq_length))
        return int(method(new_seq_length, self._logical_slot))

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
    return DynamicCache()


def _cache_seq_length(cache: Any) -> int:
    if cache is None:
        return 0
    try:
        return int(cache.get_seq_length())
    except (IndexError, KeyError):
        return 0


def _validate_cache(cache: Any) -> None:
    if cache is None:
        return
    if not hasattr(cache, "update") or not hasattr(cache, "get_seq_length"):
        raise TypeError("past_key_values must be a Transformers Cache object")
    if not isinstance(cache, DynamicCache):
        try:
            if len(cache) < LOGICAL_LAYER_COUNT:
                raise ValueError(f"cache must expose at least {LOGICAL_LAYER_COUNT} logical slots")
        except TypeError:
            pass


def _causal_mask(attention_mask: torch.Tensor | None, *, batch_size: int, query_length: int, past_length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    total = past_length + query_length
    if attention_mask is not None and attention_mask.ndim == 4:
        if tuple(attention_mask.shape[-2:]) != (query_length, total):
            raise ValueError("4-D attention_mask does not match query/cache lengths")
        return attention_mask.to(device=device, dtype=dtype)
    if attention_mask is None:
        valid = torch.ones((batch_size, total), dtype=torch.bool, device=device)
    elif attention_mask.ndim == 2:
        given = attention_mask.to(device=device).bool()
        if given.shape[0] != batch_size:
            raise ValueError("attention_mask batch dimension mismatch")
        if given.shape[1] == total:
            valid = given
        elif given.shape[1] == query_length:
            valid = torch.cat((torch.ones((batch_size, past_length), dtype=torch.bool, device=device), given), dim=1)
        else:
            raise ValueError("2-D attention_mask must cover query or total sequence")
    else:
        raise ValueError("attention_mask must be rank 2 or rank 4")
    q = torch.arange(past_length, total, device=device).view(1, 1, query_length, 1)
    k = torch.arange(total, device=device).view(1, 1, 1, total)
    allowed = (k <= q) & valid.view(batch_size, 1, 1, total)
    return torch.zeros((batch_size, 1, query_length, total), dtype=dtype, device=device).masked_fill(~allowed, torch.finfo(dtype).min)


def _call_layer(layer: nn.Module, hidden: torch.Tensor, *, attention_mask: torch.Tensor, position_ids: torch.Tensor, cache: Any, use_cache: bool, cache_position: torch.Tensor | None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None, output_attentions: bool) -> tuple[Any, ...]:
    params = inspect.signature(layer.forward).parameters
    kwargs: dict[str, Any] = {"attention_mask": attention_mask, "position_ids": position_ids, "use_cache": use_cache}
    if "output_attentions" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs["output_attentions"] = output_attentions
    if "past_key_value" in params:
        kwargs["past_key_value"] = cache
    elif "past_key_values" in params:
        kwargs["past_key_values"] = cache
    else:
        raise RuntimeError("unsupported decoder cache argument")
    if "cache_position" in params:
        kwargs["cache_position"] = cache_position
    if "position_embeddings" in params:
        kwargs["position_embeddings"] = position_embeddings
    result = layer(hidden, **kwargs)
    return tuple(result) if isinstance(result, tuple) else (result,)


def _init_router(router: nn.Linear) -> None:
    # ``from_pretrained`` constructs modules under an empty/meta-weight
    # context before materializing checkpoint tensors.  Data-dependent
    # initialization (notably the finite check below) is invalid there; the
    # real router values are loaded from the checkpoint immediately after.
    if router.weight.is_meta or (router.bias is not None and router.bias.is_meta):
        return
    std = math.sqrt(2.0 / (MEMORY_SLOT_COUNT * router.in_features))
    with torch.no_grad():
        values = torch.empty_like(router.weight)
        while True:
            values.normal_(0.0, std)
            values.clamp_(-3.0 * std, 3.0 * std)
            if torch.isfinite(values).all():
                break
        router.weight.copy_(values)
        router.bias.zero_()


class MeshLlamaModel(LlamaPreTrainedModel):
    """Backbone with 5 prefix, 10 shared middle, and 5 suffix modules."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        if (int(getattr(config, "num_hidden_layers", 0)), int(getattr(config, "recursive_layer_count", 0)), int(getattr(config, "recursive_loops", 0))) != (30, 20, 2):
            raise ValueError("MeSH requires config logical=30, physical=20, loops=2")
        if tuple(getattr(config, "logical_to_physical", LOGICAL_TO_PHYSICAL)) != LOGICAL_TO_PHYSICAL:
            raise ValueError("invalid MeSH logical schedule")
        self.logical_layer_count = 30
        self.recursive_layer_count = 20
        self.recursive_loops = 2
        self.memory_slots = MEMORY_SLOT_COUNT
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx=i) for i in range(PHYSICAL_LAYER_COUNT)])
        _assert_llama_api(self.layers[0])
        from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.write_routers = nn.ModuleList([nn.Linear(config.hidden_size, MEMORY_SLOT_COUNT, bias=True) for _ in range(ROUTER_COUNT)])
        self.read_routers = nn.ModuleList([nn.Linear(config.hidden_size, MEMORY_SLOT_COUNT, bias=True) for _ in range(ROUTER_COUNT)])
        for router in list(self.write_routers) + list(self.read_routers):
            _init_router(router)
        self.gradient_checkpointing = False
        self.audit_mode = False
        self.routing_stats_mode = False
        self.last_forward_trace: list[dict[str, int]] = []
        self.last_memory_shape: tuple[int, ...] | None = None
        self.last_router_weights: dict[str, torch.Tensor] = {}
        self.last_router_queries: dict[str, torch.Tensor] = {}
        self.last_prefix_output: torch.Tensor | None = None
        self.last_core_inputs: list[torch.Tensor] = []
        self.last_core_outputs: list[torch.Tensor] = []
        self.last_initial_memory: torch.Tensor | None = None
        self.last_memory_write_history: list[torch.Tensor] = []
        self.last_routing_stats: dict[str, dict[str, float]] = {}
        self.post_init()
        # post_init initializes all modules.  Restore the explicit router policy.
        for router in list(self.write_routers) + list(self.read_routers):
            _init_router(router)

    def _route(self, router: nn.Linear, query: torch.Tensor, name: str) -> torch.Tensor:
        weights = F.softmax(router(query).float(), dim=-1).to(dtype=query.dtype)
        if self.audit_mode:
            self.last_router_weights[name] = weights.detach().cpu()
            self.last_router_queries[name] = query.detach().cpu()
        if self.routing_stats_mode:
            detached = weights.float().detach()
            entropy = -(detached.clamp_min(1e-12) * detached.clamp_min(1e-12).log()).sum(dim=-1).mean()
            self.last_routing_stats[name] = {
                "mean_entropy": float(entropy.cpu()),
                "mean_slot_usage": float(detached.mean(dim=(0, 1)).mean().cpu()),
                "max_slot_usage": float(detached.mean(dim=(0, 1)).max().cpu()),
                "slot_probabilities": [float(value) for value in detached.mean(dim=(0, 1)).cpu().tolist()],
            }
        return weights

    @staticmethod
    def _write(memory: torch.Tensor, value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return memory + value.unsqueeze(1) * weights.transpose(1, 2).unsqueeze(-1)

    @staticmethod
    def _read(memory: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (memory * weights.transpose(1, 2).unsqueeze(-1)).sum(dim=1)

    def _run_stack(self, hidden: torch.Tensor, layer_indices: Sequence[int], logical_start: int, *, attention_mask: torch.Tensor, position_ids: torch.Tensor, cache: Any, use_cache: bool, cache_position: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None, output_attentions: bool, all_attentions: list[torch.Tensor]) -> torch.Tensor:
        for offset, physical_index in enumerate(layer_indices):
            logical_index = logical_start + offset
            self.last_forward_trace.append({"logical_index": logical_index, "physical_index": int(physical_index)})
            layer_cache = LogicalSlotCacheView(cache, physical_index=physical_index, logical_slot=logical_index) if cache is not None else None
            outputs = _call_layer(self.layers[physical_index], hidden, attention_mask=attention_mask, position_ids=position_ids, cache=layer_cache, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, output_attentions=output_attentions)
            hidden = outputs[0]
            if output_attentions:
                if len(outputs) < 2 or outputs[1] is None:
                    raise RuntimeError("decoder did not return attentions")
                all_attentions.append(outputs[1])
        return hidden

    def forward(self, input_ids: torch.LongTensor | None = None, attention_mask: torch.Tensor | None = None, position_ids: torch.LongTensor | None = None, past_key_values: Any | None = None, inputs_embeds: torch.FloatTensor | None = None, use_cache: bool | None = None, output_attentions: bool | None = None, output_hidden_states: bool | None = None, return_dict: bool | None = None, cache_position: torch.LongTensor | None = None, **kwargs: Any) -> BaseModelOutputWithPast | tuple[Any, ...]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if kwargs:
            raise TypeError(f"unsupported MeshLlamaModel arguments: {sorted(kwargs)}")
        use_cache = bool(_cfg(self.config, "use_cache", True) if use_cache is None else use_cache)
        output_attentions = bool(_cfg(self.config, "output_attentions", False) if output_attentions is None else output_attentions)
        output_hidden_states = bool(_cfg(self.config, "output_hidden_states", False) if output_hidden_states is None else output_hidden_states)
        return_dict = bool(_cfg(self.config, "use_return_dict", True) if return_dict is None else return_dict)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))
        hidden = inputs_embeds
        batch_size, query_length = hidden.shape[:2]
        cache = past_key_values
        if use_cache and cache is None:
            cache = make_dynamic_cache()
        if use_cache:
            _validate_cache(cache)
        past_length = _cache_seq_length(cache)
        if cache_position is not None:
            cache_position = cache_position.to(hidden.device)
            if cache_position.ndim != 1 or cache_position.numel() != query_length:
                raise ValueError("cache_position must match query length")
        if position_ids is None:
            positions = cache_position if cache_position is not None else torch.arange(past_length, past_length + query_length, device=hidden.device)
            position_ids = positions.unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = position_ids.to(hidden.device)
            if tuple(position_ids.shape) != (batch_size, query_length):
                raise ValueError("position_ids shape mismatch")
        if cache_position is None:
            cache_position = position_ids[0]
        mask = _causal_mask(attention_mask, batch_size=batch_size, query_length=query_length, past_length=past_length, dtype=hidden.dtype, device=hidden.device)
        position_embeddings = self.rotary_emb(hidden, position_ids=position_ids)
        self.last_forward_trace = []
        self.last_router_weights = {}
        self.last_router_queries = {}
        self.last_prefix_output = None
        self.last_core_inputs = []
        self.last_core_outputs = []
        self.last_initial_memory = None
        self.last_memory_write_history = []
        self.last_routing_stats = {}
        self.last_memory_shape = None
        hidden_states: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        if output_hidden_states:
            hidden_states.append(hidden)
        # The memory starts from raw embedding, before prefix execution.
        memory = torch.zeros((batch_size, MEMORY_SLOT_COUNT, query_length, hidden.shape[-1]), dtype=hidden.dtype, device=hidden.device)
        memory[:, 1:] = 0
        memory[:, 0] = hidden
        if self.audit_mode:
            self.last_initial_memory = memory.detach().cpu()
        prefix_output = self._run_stack(hidden, range(0, 5), 0, attention_mask=mask, position_ids=position_ids, cache=cache, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, output_attentions=output_attentions, all_attentions=attentions)
        if self.audit_mode:
            self.last_prefix_output = prefix_output.detach().cpu()
        write_pre = self._route(self.write_routers[0], prefix_output, "write_pre")
        read_pre = self._route(self.read_routers[0], prefix_output, "read_pre")
        memory = self._write(memory, prefix_output, write_pre)
        if self.audit_mode:
            self.last_memory_write_history.append(memory.detach().cpu())
        hidden = self._read(memory, read_pre)
        if output_hidden_states:
            hidden_states.append(hidden)
        for loop in range(2):
            if self.audit_mode:
                self.last_core_inputs.append(hidden.detach().cpu())
            core = self._run_stack(hidden, range(5, 15), 5 + loop * 10, attention_mask=mask, position_ids=position_ids, cache=cache, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, output_attentions=output_attentions, all_attentions=attentions)
            if self.audit_mode:
                self.last_core_outputs.append(core.detach().cpu())
            write = self._route(self.write_routers[loop + 1], hidden, f"write_{loop}")
            read = self._route(self.read_routers[loop + 1], hidden, f"read_{loop}")
            memory = self._write(memory, core, write)
            if self.audit_mode:
                self.last_memory_write_history.append(memory.detach().cpu())
            hidden = self._read(memory, read)
            if output_hidden_states:
                hidden_states.append(hidden)
        hidden = self._run_stack(hidden, range(15, 20), 25, attention_mask=mask, position_ids=position_ids, cache=cache, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, output_attentions=output_attentions, all_attentions=attentions)
        hidden = self.norm(hidden)
        if self.audit_mode:
            self.last_memory_shape = tuple(memory.shape)
            self.last_memory = memory.detach().cpu()
        if output_hidden_states:
            hidden_states.append(hidden)
        present = cache if use_cache else None
        if not return_dict:
            values: tuple[Any, ...] = (hidden,)
            if present is not None:
                values += (present,)
            if output_hidden_states:
                values += (tuple(hidden_states),)
            if output_attentions:
                values += (tuple(attentions),)
            return values
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=present, hidden_states=tuple(hidden_states) if output_hidden_states else None, attentions=tuple(attentions) if output_attentions else None)


class RecursiveLlama5_10x2_5MeshForCausalLM(LlamaForCausalLM):
    """Causal-LM wrapper for the isolated MeSH backbone."""

    def __init__(self, config: LlamaConfig) -> None:
        LlamaPreTrainedModel.__init__(self, config)
        self.model = MeshLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        # ``post_init`` walks the complete module tree, so restore the explicit
        # policy once more after the LM wrapper has initialized its head.
        for router in list(self.model.write_routers) + list(self.model.read_routers):
            _init_router(router)

    def _prepare_cache_for_generation(self, generation_config: Any, model_kwargs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        implementation = getattr(generation_config, "cache_implementation", None)
        if implementation == "dynamic":
            raise ValueError("cache_implementation='dynamic' is unsupported; use lazy DynamicCache()")
        if model_kwargs.get("past_key_values") is None and bool(getattr(generation_config, "use_cache", True)) and implementation is None:
            model_kwargs["past_key_values"] = make_dynamic_cache()
        return super()._prepare_cache_for_generation(generation_config, model_kwargs, *args, **kwargs)

    def forward(self, input_ids: torch.LongTensor | None = None, attention_mask: torch.Tensor | None = None, position_ids: torch.LongTensor | None = None, past_key_values: Any | None = None, inputs_embeds: torch.FloatTensor | None = None, labels: torch.LongTensor | None = None, use_cache: bool | None = None, output_attentions: bool | None = None, output_hidden_states: bool | None = None, return_dict: bool | None = None, cache_position: torch.LongTensor | None = None, logits_to_keep: int | torch.Tensor = 0, **kwargs: Any) -> CausalLMOutputWithPast | tuple[Any, ...]:
        loss_kwargs = {}
        if "num_items_in_batch" in kwargs:
            loss_kwargs["num_items_in_batch"] = kwargs.pop("num_items_in_batch")
        if kwargs:
            raise TypeError(f"unsupported MeshForCausalLM arguments: {sorted(kwargs)}")
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True, cache_position=cache_position)
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)
        result = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        if return_dict is None:
            return_dict = bool(_cfg(self.config, "use_return_dict", True))
        return result if return_dict else result.to_tuple()


RecursiveLlamaForCausalLM = RecursiveLlama5_10x2_5MeshForCausalLM


def register_auto_class() -> None:
    try:
        AutoModelForCausalLM.register(LlamaConfig, RecursiveLlama5_10x2_5MeshForCausalLM, exist_ok=True)
    except TypeError:
        AutoModelForCausalLM.register(LlamaConfig, RecursiveLlama5_10x2_5MeshForCausalLM)


def parameter_audit(model: nn.Module) -> dict[str, Any]:
    names = list(model.named_parameters(remove_duplicate=False))
    by_id: dict[int, list[str]] = defaultdict(list)
    values: dict[int, nn.Parameter] = {}
    for name, parameter in names:
        by_id[id(parameter)].append(name)
        values[id(parameter)] = parameter
    router_names = [name for name, _ in names if ".write_routers." in name or ".read_routers." in name]
    return {
        "parameter_count_unique": int(sum(p.numel() for p in values.values())),
        "parameter_count_references": int(sum(p.numel() for _, p in names)),
        "logical_layer_count": 30, "physical_layer_count": 20, "logical_cache_slot_count": 30,
        "recursive_loops": 2, "memory_slots": 5, "router_count": 6,
        "schedule": list(LOGICAL_TO_PHYSICAL), "source_mapping_0based": list(SOURCE_LAYER_INDICES_0BASED),
        "router_parameter_names": router_names, "router_parameter_count": ROUTER_PARAMETER_COUNT, "router_parameter_tensor_count": len(router_names),
        "router_objects_independent": len({id(r) for r in list(getattr(getattr(model, "model", model), "write_routers", [])) + list(getattr(getattr(model, "model", model), "read_routers", []))}) == 6,
        "embedding_scale": "disabled", "transition_query": "prefix_output", "memory_persistent": False,
    }


__all__ = [
    "LOGICAL_LAYER_COUNT", "PHYSICAL_LAYER_COUNT", "PREFIX_LAYER_COUNT", "MIDDLE_LAYER_COUNT", "SUFFIX_LAYER_COUNT", "RECURSIVE_LOOPS", "MEMORY_SLOT_COUNT", "ROUTER_COUNT", "ROUTER_PARAMETER_COUNT", "MODEL_ARCHITECTURE_CONTRACT", "TRANSITION_ROUTER_QUERY", "EMBEDDING_SCALE", "SOURCE_LAYER_INDICES_0BASED", "SOURCE_LAYER_INDICES_1BASED", "SOURCE_MAPPING_0BASED", "SOURCE_MAPPING_1BASED", "LOGICAL_TO_PHYSICAL", "LOGICAL_TO_PHYSICAL_SCHEDULE", "LogicalSlotCacheView", "MeshLlamaModel", "RecursiveLlama5_10x2_5MeshForCausalLM", "RecursiveLlamaForCausalLM", "build_mesh_schedule", "build_5_10x2_5_mesh_schedule", "build_source_mapping", "build_5_10x2_5_mesh_source_mapping", "make_dynamic_cache", "parameter_audit", "register_auto_class",
]
