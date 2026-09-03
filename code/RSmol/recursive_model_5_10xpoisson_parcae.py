"""Independent 5-10xpoisson-parcae recursive Llama implementation.

This module intentionally has no imports from the older recursive variants.
It keeps twenty physical decoder modules (5 prefix, 10 shared middle and 5
suffix), while a training forward accepts one middle-loop depth per local
sequence.  The maximum depth is local to the rank; it is never broadcast.

The recurrent core follows the Parcae diagonal-injection form::

    PN(e) = PreludeNorm(prefix(x))
    u_t   = Abar(h_t) + Bbar(PN(e))
    h_{t+1} = MiddleBlockStack(u_t)

The first ``Tmax - T_i`` aligned calls are no-op/copy calls for sample ``i``;
equivalently ``tau_i = Tmax - T_i``.
Only the final four aligned calls use live recurrent parameters.  Earlier
calls use ``torch.func.functional_call`` with detached recurrent parameters;
this retains the hidden-input autograd path to the prefix.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import warnings
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

try:
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaDecoderLayer,
        LlamaForCausalLM,
        LlamaPreTrainedModel,
        LlamaRMSNorm,
    )
except ImportError as exc:  # pragma: no cover - dependency-light checkout
    raise ImportError(
        "code.RSmol.recursive_model_5_10xpoisson_parcae requires torch and transformers"
    ) from exc


MODEL_LABEL = "5_10xpoisson_parcae"
ARCHITECTURE_CONTRACT = "logical_50_110_physical_20_5_10xpoisson_parcae_tail4"
MAPPING_POLICY = "explicit_5_10xpoisson_parcae_source_layers"
BACKWARD_POLICY = "hidden_path_all_calls_parameter_gradients_final_four_aligned_calls_v1"
PHYSICAL_LAYER_COUNT = 20
PREFIX_LAYER_COUNT = 5
MIDDLE_LAYER_COUNT = 10
SUFFIX_LAYER_COUNT = 5
MIN_MIDDLE_LOOPS = 4
MAX_MIDDLE_LOOPS = 10
DEFAULT_INFERENCE_MIDDLE_LOOPS = 7
DEFAULT_INFERENCE_R = DEFAULT_INFERENCE_MIDDLE_LOOPS
PARAMETER_GRADIENT_TAIL_LOOPS = 4
DEFAULT_SSM_DECAY = math.sqrt(1.0 / 5.0)
DEFAULT_TARGET_PRODUCT = -math.log(DEFAULT_SSM_DECAY)
# Keep the bounds explicit in metadata: 50 <= logical depth <= 110.
MIN_LOGICAL_LAYER_COUNT = 50
MAX_LOGICAL_LAYER_COUNT = 110
LOGICAL_LAYER_COUNT = MAX_LOGICAL_LAYER_COUNT
RECURSIVE_LOOPS = MAX_MIDDLE_LOOPS
LOOPS_SCOPE = "middle_only"

# Exact standard-Poisson(lambda=7) truncation contract, support 4..10:
# P(k)=exp(-7)*7**k/k!, followed by division by Z (exact truncation).
POISSON_LAMBDA = 7.0
POISSON_SUPPORT = tuple(range(MIN_MIDDLE_LOOPS, MAX_MIDDLE_LOOPS + 1))
POISSON_NORMALIZATION_Z = sum(
    math.exp(-POISSON_LAMBDA) * POISSON_LAMBDA**k / math.factorial(k)
    for k in POISSON_SUPPORT
)
POISSON_Z = POISSON_NORMALIZATION_Z
POISSON_TRUNCATION_Z = POISSON_NORMALIZATION_Z
POISSON_PROBABILITIES = tuple(
    (math.exp(-POISSON_LAMBDA) * POISSON_LAMBDA**k / math.factorial(k)) / POISSON_NORMALIZATION_Z
    for k in POISSON_SUPPORT
)
SAMPLING_POLICY = "truncated_poisson"
SAMPLER_VERSION = "truncated_poisson_lambda7_support4_10_v1"
SAMPLER_KEY = "sha256_cpu_torch_generator_base_seed_rank_optimizer_step_microbatch_v1"

SOURCE_LAYER_INDICES_0BASED = (
    0, 1, 2, 3, 4,
    5, 7, 9, 11, 13, 15, 17, 19, 21, 23,
    25, 26, 27, 28, 29,
)
SOURCE_LAYER_INDICES_1BASED = tuple(i + 1 for i in SOURCE_LAYER_INDICES_0BASED)
SOURCE_MAPPING_0BASED = SOURCE_LAYER_INDICES_0BASED
SOURCE_MAPPING_1BASED = SOURCE_LAYER_INDICES_1BASED


def build_5_10xpoisson_parcae_schedule(
    middle_loop_count: int = DEFAULT_INFERENCE_MIDDLE_LOOPS,
    *, logical_layer_count: int | None = None,
    physical_layer_count: int = PHYSICAL_LAYER_COUNT,
) -> tuple[int, ...]:
    """Build prefix + middle*T + suffix for one scalar inference depth."""

    loops = int(middle_loop_count)
    if not MIN_MIDDLE_LOOPS <= loops <= MAX_MIDDLE_LOOPS:
        raise ValueError(f"middle_loop_count must be in [4, 10], got {loops}")
    if int(physical_layer_count) != PHYSICAL_LAYER_COUNT:
        raise ValueError(f"5-10xpoisson-parcae requires 20 physical layers, got {physical_layer_count}")
    schedule = tuple(range(5)) + tuple(range(5, 15)) * loops + tuple(range(15, 20))
    expected = 5 + 10 * loops + 5
    if logical_layer_count is not None and int(logical_layer_count) != expected:
        raise ValueError(f"logical depth {logical_layer_count} does not match T={loops}; expected {expected}")
    if len(schedule) != expected or not MIN_LOGICAL_LAYER_COUNT <= len(schedule) <= MAX_LOGICAL_LAYER_COUNT:
        raise AssertionError("5-10xpoisson-parcae schedule has invalid logical depth")
    return schedule


def build_5_10xpoisson_parcae_source_mapping(num_hidden_layers: int) -> tuple[int, ...]:
    if int(num_hidden_layers) != 30:
        raise ValueError(f"5-10xpoisson-parcae conversion requires source num_hidden_layers=30; got {num_hidden_layers}")
    return SOURCE_LAYER_INDICES_0BASED


# Short aliases are useful for scripts and make the schedule contract explicit.
build_schedule = build_5_10xpoisson_parcae_schedule
build_source_mapping = build_5_10xpoisson_parcae_source_mapping
LOGICAL_TO_PHYSICAL = build_schedule(MAX_MIDDLE_LOOPS)
LOGICAL_TO_PHYSICAL_SCHEDULE = LOGICAL_TO_PHYSICAL
RECURSIVE_LOGICAL_TO_PHYSICAL = LOGICAL_TO_PHYSICAL
RECURSIVE_LOGICAL_TO_PHYSICAL_SCHEDULE = LOGICAL_TO_PHYSICAL


def poisson_probabilities(*, lam: float = POISSON_LAMBDA, support: Sequence[int] = POISSON_SUPPORT) -> tuple[float, ...]:
    """Return exact standard Poisson probabilities after support truncation."""

    lam = float(lam)
    support = tuple(int(k) for k in support)
    if lam <= 0 or not support:
        raise ValueError("lambda must be positive and support must be non-empty")
    raw = tuple(math.exp(-lam) * lam**k / math.factorial(k) for k in support)
    z = sum(raw)
    return tuple(value / z for value in raw)


def poisson_metadata() -> dict[str, Any]:
    return {
        "sampling_policy": SAMPLING_POLICY,
        "sampler_version": SAMPLER_VERSION,
        "sampler_key": SAMPLER_KEY,
        "poisson_lambda": POISSON_LAMBDA,
        "poisson_support": list(POISSON_SUPPORT),
        "poisson_normalization_z": POISSON_NORMALIZATION_Z,
        "lambda": POISSON_LAMBDA,
        "support": list(POISSON_SUPPORT),
        "Z": POISSON_NORMALIZATION_Z,
        "poisson_probabilities": list(POISSON_PROBABILITIES),
    }


def _stable_sampler_seed(base_seed: int, rank: int, optimizer_step: int, microbatch_index: int) -> int:
    payload = f"{int(base_seed)}:{int(rank)}:{int(optimizer_step)}:{int(microbatch_index)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def sample_middle_loop_counts(
    base_seed: int,
    rank: int,
    optimizer_step: int,
    microbatch_index: int,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.LongTensor:
    """Sample one independently-derived truncated-Poisson depth per sequence.

    Sampling is per local microbatch (not per optimizer step), uses a private
    CPU generator, and never changes the process-global RNG state.  No rank
    communication or rank-0 broadcast is involved.
    """

    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_sampler_seed(base_seed, rank, optimizer_step, microbatch_index))
    probabilities = torch.tensor(POISSON_PROBABILITIES, dtype=torch.float64, device="cpu")
    indices = torch.multinomial(probabilities, batch_size, replacement=True, generator=generator)
    return torch.tensor(POISSON_SUPPORT, dtype=torch.long, device="cpu")[indices]


def normalize_middle_loop_counts(
    middle_loop_counts: torch.Tensor | Sequence[int], batch_size: int, *, device: torch.device
) -> torch.LongTensor:
    counts = torch.as_tensor(middle_loop_counts, dtype=torch.long, device=device).flatten()
    if counts.numel() != int(batch_size):
        raise ValueError(f"training middle_loop_counts must have one T_i per sequence ({batch_size}), got {counts.numel()}")
    if bool(torch.any(counts < MIN_MIDDLE_LOOPS)) or bool(torch.any(counts > MAX_MIDDLE_LOOPS)):
        raise ValueError("every training T_i must be in support 4..10")
    return counts


def left_alignment_tau(middle_loop_counts: torch.Tensor | Sequence[int], local_tmax: int | None = None) -> torch.LongTensor:
    counts = torch.as_tensor(middle_loop_counts, dtype=torch.long)
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("middle_loop_counts must be a non-empty vector")
    tmax = int(counts.max().item()) if local_tmax is None else int(local_tmax)
    if tmax != int(counts.max().item()):
        raise ValueError("local_tmax must equal the maximum depth on this rank")
    if bool(torch.any(counts < MIN_MIDDLE_LOOPS)) or bool(torch.any(counts > MAX_MIDDLE_LOOPS)):
        raise ValueError("every T_i must be in support 4..10")
    tau_i = tmax - counts
    return tau_i


def no_op_mask(tau: torch.Tensor, aligned_step: int) -> torch.BoolTensor:
    """True means this sample performs no operation at this aligned step."""

    return torch.as_tensor(aligned_step, device=tau.device) < tau


def _config_value(config: Any, name: str, default: Any) -> Any:
    value = getattr(config, name, default)
    return default if value is None else value


def _assert_supported_llama_api(layer: nn.Module) -> None:
    import transformers

    installed = str(getattr(transformers, "__version__", "unknown"))
    if installed != "4.54.1":
        warnings.warn(f"5-10xpoisson-parcae was designed for transformers==4.54.1; installed={installed}", RuntimeWarning)
    params = inspect.signature(layer.forward).parameters
    if not ({"past_key_value", "past_key_values"} & set(params)):
        raise RuntimeError("Unsupported LlamaDecoderLayer cache signature")


class LogicalSlotCacheView:
    """Map one physical layer call to its logical cache slot."""

    def __init__(self, cache: Any, *, physical_index: int, logical_slot: int) -> None:
        self._cache, self._physical_index, self._logical_slot = cache, int(physical_index), int(logical_slot)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: Mapping[str, Any] | None = None, **kwargs: Any):
        if int(layer_idx) != self._physical_index:
            raise ValueError("physical decoder layer index does not match cache view")
        params = inspect.signature(self._cache.update).parameters
        if "cache_kwargs" in params:
            return self._cache.update(key_states, value_states, self._logical_slot, cache_kwargs=cache_kwargs, **kwargs)
        return self._cache.update(key_states, value_states, self._logical_slot, **kwargs)

    def get_seq_length(self, layer_idx: int = 0, cache_position: torch.LongTensor | None = None) -> int:
        del layer_idx
        method = self._cache.get_seq_length
        if "cache_position" in inspect.signature(method).parameters:
            return int(method(self._logical_slot, cache_position=cache_position))
        return int(method(self._logical_slot))

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        del layer_idx
        method = getattr(self._cache, "get_usable_length", None)
        if method is None:
            return self.get_seq_length()
        names = list(inspect.signature(method).parameters)
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


def _bind_cache_middle_loop_count(cache: Any, middle_loop_count: int) -> None:
    """Prevent reusing a scalar inference cache under another schedule."""

    if cache is None:
        return
    marker = "_rsmol_5_10xpoisson_parcae_middle_loop_count"
    current = getattr(cache, marker, None)
    if current is not None and int(current) != int(middle_loop_count):
        raise ValueError(
            "past_key_values/cache is already bound to a different middle_loop_count: "
            f"bound={current} requested={middle_loop_count}"
        )
    try:
        setattr(cache, marker, int(middle_loop_count))
    except Exception as exc:
        if current is None:
            raise TypeError("past_key_values must permit schedule binding for 5-10xpoisson-parcae") from exc


def _validate_cache_capacity(cache: Any, logical_layer_count: int) -> None:
    if cache is None or isinstance(cache, DynamicCache):
        return
    try:
        capacity = len(cache)
    except TypeError:
        return
    if capacity < int(logical_layer_count):
        raise ValueError(f"past_key_values capacity={capacity} is smaller than logical depth={logical_layer_count}")


def _cache_seq_length(cache: Any) -> int:
    if cache is None:
        return 0
    try:
        return int(cache.get_seq_length())
    except (IndexError, KeyError):
        return 0


def _causal_mask(*, attention_mask: torch.Tensor | None, batch_size: int, query_length: int, past_length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    total = past_length + query_length
    if attention_mask is not None and attention_mask.ndim == 4:
        return attention_mask.to(device=device, dtype=dtype)
    if attention_mask is None:
        valid = torch.ones((batch_size, total), dtype=torch.bool, device=device)
    else:
        provided = attention_mask.to(device=device).bool()
        if provided.shape[1] == total:
            valid = provided
        elif provided.shape[1] == query_length:
            valid = torch.cat((torch.ones((batch_size, past_length), dtype=torch.bool, device=device), provided), dim=1)
        else:
            raise ValueError("attention_mask must cover query or total sequence length")
    q = torch.arange(past_length, total, device=device).view(1, 1, query_length, 1)
    k = torch.arange(total, device=device).view(1, 1, 1, total)
    allowed = (k <= q) & valid.view(batch_size, 1, 1, total)
    return torch.zeros((batch_size, 1, query_length, total), dtype=dtype, device=device).masked_fill(~allowed, torch.finfo(dtype).min)


def _call_decoder_layer(layer: nn.Module, hidden_states: torch.Tensor, *, attention_mask: torch.Tensor, position_ids: torch.LongTensor, cache: Any, use_cache: bool, cache_position: torch.LongTensor | None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[Any, ...]:
    params = inspect.signature(layer.forward).parameters
    kwargs: dict[str, Any] = {"attention_mask": attention_mask, "position_ids": position_ids, "use_cache": use_cache}
    if "output_attentions" in params:
        kwargs["output_attentions"] = False
    if "past_key_value" in params:
        kwargs["past_key_value"] = cache
    elif "past_key_values" in params:
        kwargs["past_key_values"] = cache
    if "cache_position" in params:
        kwargs["cache_position"] = cache_position
    if "position_embeddings" in params:
        kwargs["position_embeddings"] = position_embeddings
    result = layer(hidden_states, **kwargs)
    return tuple(result) if isinstance(result, tuple) else (result,)


class PreludeNorm(LlamaRMSNorm):
    """Dedicated PreludeNorm aligned with the Llama/SmolLM2 ``config.Norm``."""

    norm_type = "LlamaRMSNorm"


class ParcaeInjection(nn.Module):
    """Exact Parcae diagonal injection ``h*decay + dt*(PN(e) @ B.T)``."""

    def __init__(self, hidden_size: int, *, ssm_decay: float = DEFAULT_SSM_DECAY) -> None:
        super().__init__()
        if not 0.0 < float(ssm_decay) < 1.0:
            raise ValueError("ssm_decay must be in (0, 1)")
        self.ssm_decay = float(ssm_decay)
        self.target_product = -math.log(self.ssm_decay)
        self.A_log = nn.Parameter(torch.zeros(hidden_size))
        # inverse-softplus(target_product), with A=exp(A_log)=1 at init.
        self.dt_bias = nn.Parameter(torch.full((hidden_size,), math.log(math.expm1(self.target_product))))
        # Parcae identity B init keeps the initial input map well-conditioned.
        self.B = nn.Parameter(torch.eye(hidden_size))
        for parameter in (self.A_log, self.dt_bias, self.B):
            parameter._no_weight_decay = True

    def reset_parameters(self) -> None:
        """Re-apply the exact Parcae initialization after HF ``post_init``."""

        with torch.inference_mode():
            self.A_log.zero_()
            self.dt_bias.fill_(math.log(math.expm1(self.target_product)))
            self.B.copy_(torch.eye(self.B.shape[0], device=self.B.device, dtype=self.B.dtype))
        for parameter in (self.A_log, self.dt_bias, self.B):
            parameter._no_weight_decay = True

    def forward(self, h_t: torch.Tensor, pn_e: torch.Tensor) -> torch.Tensor:
        dt = F.softplus(self.dt_bias)
        A = torch.exp(self.A_log)
        decay = torch.exp(-dt * A)
        Abar_h = h_t * decay
        Bbar_PN_e = dt * torch.matmul(pn_e, self.B.transpose(-1, -2))
        # Parcae's injection is a linear/diagonal addition, never concat:
        # u_t = Abar(h_t) + Bbar(PN(e)).
        u_t = Abar_h + Bbar_PN_e
        return u_t

    def initialization_audit(self) -> dict[str, Any]:
        with torch.inference_mode():
            dt = F.softplus(self.dt_bias)
            A = torch.exp(self.A_log)
            decay = torch.exp(-dt * A)
        return {
            "ssm_decay_target": self.ssm_decay,
            "target_product": self.target_product,
            "initial_dt": float(dt.mean().item()),
            "initial_decay": float(decay.mean().item()),
            "A_log_zero": bool(torch.allclose(self.A_log, torch.zeros_like(self.A_log))),
            "B_identity": bool(torch.allclose(self.B, torch.eye(self.B.shape[0], device=self.B.device, dtype=self.B.dtype))),
            "decay_strictly_between_zero_one": bool(torch.all((decay > 0) & (decay < 1))),
            "no_weight_decay": all(bool(getattr(parameter, "_no_weight_decay", False)) for parameter in (self.A_log, self.dt_bias, self.B)),
        }


DiagonalInjection = ParcaeInjection


class MiddleBlockStack(nn.Module):
    """The ten shared physical decoder blocks."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx=5 + i) for i in range(MIDDLE_LAYER_COUNT)])

    def forward(self, hidden_states: torch.Tensor, *, attention_mask: torch.Tensor, position_ids: torch.LongTensor, cache_views: Sequence[Any] | None, use_cache: bool, cache_position: torch.LongTensor | None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            cache = None if cache_views is None else cache_views[index]
            hidden_states = _call_decoder_layer(layer, hidden_states, attention_mask=attention_mask, position_ids=position_ids, cache=cache, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)[0]
        return hidden_states


class ParcaeRecurrentUnit(nn.Module):
    """Injection plus shared ten-layer middle stack."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.injection = ParcaeInjection(config.hidden_size)
        self.middle = MiddleBlockStack(config)

    def forward(self, h_t: torch.Tensor, pn_e: torch.Tensor, *, attention_mask: torch.Tensor, position_ids: torch.LongTensor, cache_views: Sequence[Any] | None, use_cache: bool, cache_position: torch.LongTensor | None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        # Explicitly retain the semantic sequence A h + B PN(e) before the block stack.
        u_t = self.injection(h_t, pn_e)
        return self.middle(u_t, attention_mask=attention_mask, position_ids=position_ids, cache_views=cache_views, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)


def _detached_recurrent_call(unit: ParcaeRecurrentUnit, h_t: torch.Tensor, pn_e: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Suppress only recurrent parameter edges while retaining hidden autograd."""

    try:
        from torch.func import functional_call
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("selective Parcae BPTT requires torch.func.functional_call") from exc
    detached_parameters = {name: parameter.detach() for name, parameter in unit.named_parameters()}
    return functional_call(unit, detached_parameters, (h_t, pn_e), kwargs=kwargs)


class RecursiveLlama5_10xpoisson_parcaeModel(LlamaPreTrainedModel):
    """Llama decoder with local vector-depth training and scalar inference."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        required = {
            "recursive_source_num_hidden_layers": 30,
            "recursive_source_layer_count": 30,
            "recursive_layer_count": 20,
            "num_hidden_layers": MAX_LOGICAL_LAYER_COUNT,
            "recursive_min_middle_loops": MIN_MIDDLE_LOOPS,
            "recursive_max_middle_loops": MAX_MIDDLE_LOOPS,
            "recursive_default_inference_middle_loops": DEFAULT_INFERENCE_MIDDLE_LOOPS,
            "recursive_parameter_gradient_tail_loops": PARAMETER_GRADIENT_TAIL_LOOPS,
            "recursive_poisson_lambda": POISSON_LAMBDA,
            "recursive_learned_h0": False,
            "recursive_backward_policy": BACKWARD_POLICY,
            "recursive_training_loop_mode": "per_local_microbatch_per_sequence_truncated_poisson",
            "recursive_local_tmax": True,
            "recursive_noop_left_alignment": True,
            "recursive_prefix_layer_count": PREFIX_LAYER_COUNT,
            "recursive_middle_layer_count": MIDDLE_LAYER_COUNT,
            "recursive_suffix_layer_count": SUFFIX_LAYER_COUNT,
            "recursive_min_logical_layer_count": MIN_LOGICAL_LAYER_COUNT,
            "recursive_max_logical_layer_count": MAX_LOGICAL_LAYER_COUNT,
            "recursive_injection_no_weight_decay": True,
            "recursive_B_init": "identity",
        }
        for key, expected in required.items():
            actual = getattr(config, key, None)
            if actual is None or (isinstance(expected, bool) and (type(actual) is not bool or actual is not expected)) or (isinstance(expected, float) and (type(actual) not in (int, float) or float(actual) != expected)) or (not isinstance(expected, (float, bool)) and (actual != expected)):
                raise ValueError(f"5-10xpoisson-parcae config field {key} must be exactly {expected!r}, got {actual!r}")
        if tuple(getattr(config, "recursive_poisson_support", ())) != POISSON_SUPPORT:
            raise ValueError("recursive_poisson_support must be exactly 4..10")
        config_probabilities = tuple(float(value) for value in getattr(config, "recursive_poisson_probabilities", ()))
        if len(config_probabilities) != len(POISSON_PROBABILITIES) or any(abs(a - b) > 1e-14 for a, b in zip(config_probabilities, POISSON_PROBABILITIES)):
            raise ValueError("recursive_poisson_probabilities do not match exact truncated Poisson")
        if tuple(getattr(config, "recursive_source_layer_indices_0based", ())) != SOURCE_LAYER_INDICES_0BASED:
            raise ValueError("recursive_source_layer_indices_0based does not match 30->20 mapping")
        if abs(float(getattr(config, "recursive_poisson_normalization_z", -1.0)) - POISSON_NORMALIZATION_Z) > 1e-14:
            raise ValueError("recursive_poisson_normalization_z does not match exact truncated Poisson Z")
        if abs(float(getattr(config, "recursive_poisson_Z", -1.0)) - POISSON_NORMALIZATION_Z) > 1e-14:
            raise ValueError("recursive_poisson_Z does not match exact truncated Poisson Z")
        if abs(float(getattr(config, "recursive_ssm_decay", -1.0)) - DEFAULT_SSM_DECAY) > 1e-14:
            raise ValueError("recursive_ssm_decay must match sqrt(1/5)")
        if abs(float(getattr(config, "recursive_initial_decay", -1.0)) - DEFAULT_SSM_DECAY) > 1e-14:
            raise ValueError("recursive_initial_decay must match sqrt(1/5)")
        if abs(float(getattr(config, "recursive_target_product", -1.0)) - DEFAULT_TARGET_PRODUCT) > 1e-14:
            raise ValueError("recursive_target_product must match -log(sqrt(1/5))")
        if abs(float(getattr(config, "recursive_initial_dt", -1.0)) - DEFAULT_TARGET_PRODUCT) > 1e-14:
            raise ValueError("recursive_initial_dt must match inverse-softplus target product")
        if float(getattr(config, "recursive_state_init_std", 0.0)) <= 0:
            raise ValueError("recursive_state_init_std must be positive")
        if float(getattr(config, "recursive_embedding_scale", 0.0)) <= 0:
            raise ValueError("recursive_embedding_scale must be positive")
        exact_fields = {
            "recursive_mapping_policy": MAPPING_POLICY,
            "recursive_sampling_policy": SAMPLING_POLICY,
            "recursive_sampler_version": SAMPLER_VERSION,
            "recursive_sampler_key": SAMPLER_KEY,
            "recursive_prelude_norm": "LlamaRMSNorm",
            "recursive_state_init": "like-init",
            "recursive_injection_init": "parcae_exact_ssm_decay_sqrt_1_over_5_identity_B_no_weight_decay",
            "recursive_injection_formula": "h*decay + dt*(PN(e) @ B.T)",
        }
        for key, expected in exact_fields.items():
            if getattr(config, key, None) != expected:
                raise ValueError(f"5-10xpoisson-parcae config field {key} must be exactly {expected!r}")
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.logical_layer_count = int(getattr(config, "num_hidden_layers", MAX_LOGICAL_LAYER_COUNT))
        if self.logical_layer_count != MAX_LOGICAL_LAYER_COUNT:
            raise ValueError("target num_hidden_layers must be maximum logical depth 110")
        self.recursive_layer_count = PHYSICAL_LAYER_COUNT
        self.recursive_loops = MAX_MIDDLE_LOOPS
        self.min_middle_loops = MIN_MIDDLE_LOOPS
        self.max_middle_loops = MAX_MIDDLE_LOOPS
        self.default_inference_middle_loops = DEFAULT_INFERENCE_MIDDLE_LOOPS
        self.parameter_gradient_tail_loops = PARAMETER_GRADIENT_TAIL_LOOPS
        self.prefix_layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx=i) for i in range(PREFIX_LAYER_COUNT)])
        self.recurrent = ParcaeRecurrentUnit(config)
        self.suffix_layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx=15 + i) for i in range(SUFFIX_LAYER_COUNT)])
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.prelude_norm = PreludeNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        # Parcae state is fresh per forward: a random, non-learned tensor with
        # the same shape as e, initialized by truncated normal like-init.
        self.state_init = "like-init"
        self.state_init_std = float(getattr(config, "initializer_range", 0.02))
        self.embedding_scale = float(getattr(config, "embedding_scale", getattr(config, "embed_scale", 1.0)))
        self.injection_init = {
            "ssm_decay": DEFAULT_SSM_DECAY,
            "target_product": DEFAULT_TARGET_PRODUCT,
            "initial_dt": DEFAULT_TARGET_PRODUCT,
            "initial_decay": DEFAULT_SSM_DECAY,
            "B": "identity",
            "no_weight_decay": True,
        }
        self.gradient_checkpointing = False
        _assert_supported_llama_api(self.prefix_layers[0])
        self._collect_middle_gradient_audit = False
        self._last_middle_gradient_audit: list[dict[str, Any]] = []
        self._last_forward_audit: dict[str, Any] = {}
        self.post_init()
        self.recurrent.injection.reset_parameters()
        for parameter in (self.recurrent.injection.A_log, self.recurrent.injection.dt_bias, self.recurrent.injection.B):
            parameter._no_weight_decay = True

    @property
    def layers(self) -> list[nn.Module]:
        """Compatibility view in source-order physical module numbering."""
        return list(self.prefix_layers) + list(self.recurrent.middle.layers) + list(self.suffix_layers)

    def _validate_config_contract(self) -> None:
        if tuple(getattr(self.config, "recursive_source_layer_indices_0based", SOURCE_LAYER_INDICES_0BASED)) != SOURCE_LAYER_INDICES_0BASED:
            raise ValueError("Invalid 5-10xpoisson-parcae source mapping metadata")

    def _middle_cache_views(self, cache: Any, logical_slot: int) -> list[Any] | None:
        if cache is None:
            return None
        return [LogicalSlotCacheView(cache, physical_index=5 + i, logical_slot=logical_slot + i) for i in range(MIDDLE_LAYER_COUNT)]

    def _initialize_state(self, e: torch.Tensor) -> torch.Tensor:
        """Create the per-forward Parcae like-init state, not a parameter."""

        state = torch.empty(e.shape, dtype=e.dtype, device=e.device)
        std = self.state_init_std
        with torch.no_grad():
            nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        if torch.is_inference_mode_enabled():
            # Inference tensors cannot be promoted to autograd leaves; the
            # hidden-input edge is only required by training selective BPTT.
            return state * self.embedding_scale
        # Make a normal autograd leaf after initialization: early recurrent
        # calls must retain the hidden-input path without making h0 a model
        # parameter or allowing initialization to enter the graph.
        return (state.detach().requires_grad_(True)) * self.embedding_scale

    def forward(self, input_ids: torch.LongTensor | None = None, attention_mask: torch.Tensor | None = None, position_ids: torch.LongTensor | None = None, past_key_values: Any | None = None, inputs_embeds: torch.FloatTensor | None = None, use_cache: bool | None = None, output_attentions: bool | None = None, output_hidden_states: bool | None = None, return_dict: bool | None = None, cache_position: torch.LongTensor | None = None, middle_loop_count: int | None = None, middle_loop_counts: torch.Tensor | Sequence[int] | None = None, **kwargs: Any) -> BaseModelOutputWithPast | tuple[Any, ...]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if kwargs:
            raise TypeError(f"Unsupported 5-10xpoisson-parcae arguments: {sorted(kwargs)}")
        if self.training:
            if middle_loop_count is not None:
                raise ValueError("training uses vector middle_loop_counts; scalar middle_loop_count is inference-only")
            if middle_loop_counts is None:
                raise ValueError("training requires one middle_loop_counts T_i per local sequence")
        else:
            if middle_loop_counts is not None:
                raise ValueError("inference uses scalar middle_loop_count; vector middle_loop_counts is training-only")
        use_cache = bool(_config_value(self.config, "use_cache", True) if use_cache is None else use_cache)
        if self.training and use_cache:
            raise ValueError("training must use use_cache=False; cache is inference-only")
        output_attentions = bool(output_attentions) if output_attentions is not None else False
        output_hidden_states = bool(output_hidden_states) if output_hidden_states is not None else False
        return_dict = bool(_config_value(self.config, "use_return_dict", True) if return_dict is None else return_dict)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))
        hidden_states = inputs_embeds
        batch_size, query_length = hidden_states.shape[:2]
        if self.training:
            counts = normalize_middle_loop_counts(middle_loop_counts, batch_size, device=hidden_states.device)
            local_tmax = int(counts.max().item())
            tau = left_alignment_tau(counts, local_tmax).to(hidden_states.device)
        else:
            scalar = self.default_inference_middle_loops if middle_loop_count is None else int(middle_loop_count)
            if not MIN_MIDDLE_LOOPS <= scalar <= MAX_MIDDLE_LOOPS:
                raise ValueError("inference middle_loop_count must be an explicit scalar in [4, 10]")
            counts = torch.full((batch_size,), scalar, dtype=torch.long, device=hidden_states.device)
            local_tmax, tau = scalar, torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)
        schedule = build_5_10xpoisson_parcae_schedule(local_tmax)
        cache = past_key_values if use_cache else None
        if use_cache and cache is None:
            cache = make_dynamic_cache()
        if use_cache:
            _bind_cache_middle_loop_count(cache, local_tmax)
            _validate_cache_capacity(cache, len(schedule))
        past_length = _cache_seq_length(cache)
        if cache_position is not None:
            cache_position = cache_position.to(hidden_states.device)
            if cache_position.ndim != 1 or cache_position.numel() != query_length:
                raise ValueError("cache_position must match query length")
        if position_ids is None:
            positions = cache_position if cache_position is not None else torch.arange(past_length, past_length + query_length, device=hidden_states.device)
            position_ids = positions.unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = position_ids.to(hidden_states.device)
        if cache_position is None:
            cache_position = position_ids[0]
        causal_mask = _causal_mask(attention_mask=attention_mask, batch_size=batch_size, query_length=query_length, past_length=past_length, dtype=hidden_states.dtype, device=hidden_states.device)
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        all_hidden_states: tuple[torch.Tensor, ...] = ()
        for layer in self.prefix_layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = _call_decoder_layer(layer, hidden_states, attention_mask=causal_mask, position_ids=position_ids, cache=None if cache is None else LogicalSlotCacheView(cache, physical_index=layer.layer_idx, logical_slot=layer.layer_idx), use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)[0]
        e = hidden_states
        pn_e = self.prelude_norm(e)  # PN(e) is the sole recurrent input injection.
        h = self._initialize_state(e)
        self._last_state_init = h
        self._last_pn_e = pn_e
        self._last_pn_e_ids: list[int] = []
        self._last_middle_gradient_audit = []
        self._last_forward_audit = {"middle_loop_counts": [int(x) for x in counts.detach().cpu().tolist()], "local_tmax": local_tmax, "tau": [int(x) for x in tau.detach().cpu().tolist()], "logical_layer_count": len(schedule), "parameter_gradient_tail_loops": PARAMETER_GRADIENT_TAIL_LOOPS, "cache_enabled": bool(use_cache), "state_init": self.state_init, "state_init_std": self.state_init_std, "embedding_scale": self.embedding_scale, "prelude_norm": "LlamaRMSNorm", "prelude_norm_calls": 1, "pn_e_reused": True, "injection_init": self.injection_init, "sampler": poisson_metadata(), "state_shape": list(h.shape), "state_nonzero": bool(torch.any(h != 0))}
        for aligned_step in range(local_tmax):
            live_parameters = aligned_step >= local_tmax - PARAMETER_GRADIENT_TAIL_LOOPS
            self._last_pn_e_ids.append(id(pn_e))
            cache_views = self._middle_cache_views(cache, PREFIX_LAYER_COUNT + aligned_step * MIDDLE_LAYER_COUNT)
            audit_input = h if self._collect_middle_gradient_audit else None
            if live_parameters:
                h_candidate = self.recurrent(h, pn_e, attention_mask=causal_mask, position_ids=position_ids, cache_views=cache_views, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
            else:
                # Detached parameters preserve the hidden-input path.
                h_candidate = _detached_recurrent_call(self.recurrent, h, pn_e, attention_mask=causal_mask, position_ids=position_ids, cache_views=cache_views, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)
            inactive = no_op_mask(tau, aligned_step).view(batch_size, 1, 1)
            h = torch.where(inactive, h, h_candidate)  # no-op/copy keeps hidden shape and DDP usage identical.
            if self._collect_middle_gradient_audit:
                self._last_middle_gradient_audit.append({"aligned_step": aligned_step, "parameter_grad_enabled": live_parameters, "input": audit_input, "output": h, "inactive": inactive})
        hidden_states = h
        for index, layer in enumerate(self.suffix_layers):
            logical_slot = PREFIX_LAYER_COUNT + MIDDLE_LAYER_COUNT * local_tmax + index
            hidden_states = _call_decoder_layer(layer, hidden_states, attention_mask=causal_mask, position_ids=position_ids, cache=None if cache is None else LogicalSlotCacheView(cache, physical_index=15 + index, logical_slot=logical_slot), use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings)[0]
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
            return values
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=present, hidden_states=all_hidden_states if output_hidden_states else None, attentions=None)


class RecursiveLlama5_10xpoisson_parcaeForCausalLM(LlamaForCausalLM):
    """Causal-LM wrapper with separate scalar inference/vector training APIs."""

    def __init__(self, config: LlamaConfig) -> None:
        LlamaPreTrainedModel.__init__(self, config)
        self.model = RecursiveLlama5_10xpoisson_parcaeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        # The outer causal-LM ``post_init`` also touches nested parameters;
        # restore Parcae's exact injection values after that pass.
        self.model.recurrent.injection.reset_parameters()
        for parameter in (self.model.recurrent.injection.A_log, self.model.recurrent.injection.dt_bias, self.model.recurrent.injection.B):
            parameter._no_weight_decay = True

    def _prepare_cache_for_generation(self, generation_config: Any, model_kwargs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        if model_kwargs.get("past_key_values") is None and bool(getattr(generation_config, "use_cache", True)):
            model_kwargs["past_key_values"] = make_dynamic_cache()
        return super()._prepare_cache_for_generation(generation_config, model_kwargs, *args, **kwargs)

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs: Any) -> dict[str, Any]:
        middle_loop_count = kwargs.pop("middle_loop_count", None)
        model_inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        if middle_loop_count is not None:
            model_inputs["middle_loop_count"] = int(middle_loop_count)
        return model_inputs

    def forward(self, input_ids: torch.LongTensor | None = None, attention_mask: torch.Tensor | None = None, position_ids: torch.LongTensor | None = None, past_key_values: Any | None = None, inputs_embeds: torch.FloatTensor | None = None, labels: torch.LongTensor | None = None, use_cache: bool | None = None, output_attentions: bool | None = None, output_hidden_states: bool | None = None, return_dict: bool | None = None, cache_position: torch.LongTensor | None = None, middle_loop_count: int | None = None, middle_loop_counts: torch.Tensor | Sequence[int] | None = None, logits_to_keep: int | torch.Tensor = 0, **kwargs: Any) -> CausalLMOutputWithPast | tuple[Any, ...]:
        loss_kwargs = {}
        if "num_items_in_batch" in kwargs:
            loss_kwargs["num_items_in_batch"] = kwargs.pop("num_items_in_batch")
        if kwargs:
            raise TypeError(f"Unsupported 5-10xpoisson-parcae ForCausalLM arguments: {sorted(kwargs)}")
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True, cache_position=cache_position, middle_loop_count=middle_loop_count, middle_loop_counts=middle_loop_counts)
        # HF convention: logits_to_keep=0 means keep the complete sequence.
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)
        result = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        if return_dict is None:
            return_dict = bool(_config_value(self.config, "use_return_dict", True))
        return result if return_dict else result.to_tuple()


RecursiveLlamaForCausalLM = RecursiveLlama5_10xpoisson_parcaeForCausalLM


def register_auto_class() -> None:
    try:
        AutoModelForCausalLM.register(LlamaConfig, RecursiveLlama5_10xpoisson_parcaeForCausalLM, exist_ok=True)
    except TypeError:
        AutoModelForCausalLM.register(LlamaConfig, RecursiveLlama5_10xpoisson_parcaeForCausalLM)


def parameter_audit(model: nn.Module) -> dict[str, Any]:
    recursive_model = getattr(model, "model", model)
    names = list(model.named_parameters(remove_duplicate=False))
    by_id: dict[int, list[str]] = defaultdict(list)
    for name, parameter in names:
        by_id[id(parameter)].append(name)
    unique_numel = sum(next(parameter for _, parameter in names if id(parameter) == key).numel() for key in by_id)
    return {"architecture_contract": ARCHITECTURE_CONTRACT, "parameter_count_unique": int(unique_numel), "parameter_count_references": int(sum(parameter.numel() for _, parameter in names)), "source_logical_layer_count": 30, "source_physical_layer_count": 30, "physical_layer_count": PHYSICAL_LAYER_COUNT, "logical_layer_count": LOGICAL_LAYER_COUNT, "logical_cache_slot_count": LOGICAL_LAYER_COUNT, "min_middle_loops": MIN_MIDDLE_LOOPS, "max_middle_loops": MAX_MIDDLE_LOOPS, "default_inference_middle_loops": DEFAULT_INFERENCE_MIDDLE_LOOPS, "parameter_gradient_tail_loops": PARAMETER_GRADIENT_TAIL_LOOPS, "backward_policy": BACKWARD_POLICY, "mapping_policy": MAPPING_POLICY, "middle_recurrent_count": MIDDLE_LAYER_COUNT, "schedule_min": list(build_schedule(MIN_MIDDLE_LOOPS)), "schedule_max": list(build_schedule(MAX_MIDDLE_LOOPS)), "shared_middle_physical_indices": list(range(5, 15)), "model_physical_module_count": len(recursive_model.layers), "prelude_norm": "LlamaRMSNorm", "state_init": getattr(recursive_model, "state_init", "missing"), "state_init_std": getattr(recursive_model, "state_init_std", None), "embedding_scale": getattr(recursive_model, "embedding_scale", None), "injection_init": getattr(recursive_model, "injection_init", None), "has_learned_h0": any(name.endswith("h0") for name, _ in names), "sampler": poisson_metadata()}


__all__ = [
    "MODEL_LABEL", "ARCHITECTURE_CONTRACT", "MAPPING_POLICY", "BACKWARD_POLICY",
    "PHYSICAL_LAYER_COUNT", "PREFIX_LAYER_COUNT", "MIDDLE_LAYER_COUNT", "SUFFIX_LAYER_COUNT",
    "MIN_MIDDLE_LOOPS", "MAX_MIDDLE_LOOPS", "DEFAULT_INFERENCE_MIDDLE_LOOPS", "DEFAULT_INFERENCE_R", "PARAMETER_GRADIENT_TAIL_LOOPS",
    "MIN_LOGICAL_LAYER_COUNT", "MAX_LOGICAL_LAYER_COUNT", "LOGICAL_LAYER_COUNT", "RECURSIVE_LOOPS",
    "POISSON_LAMBDA", "POISSON_SUPPORT", "POISSON_NORMALIZATION_Z", "POISSON_Z", "POISSON_TRUNCATION_Z", "POISSON_PROBABILITIES",
    "SAMPLING_POLICY", "SAMPLER_VERSION", "SAMPLER_KEY", "SOURCE_LAYER_INDICES_0BASED", "SOURCE_LAYER_INDICES_1BASED",
    "SOURCE_MAPPING_0BASED", "SOURCE_MAPPING_1BASED", "LOGICAL_TO_PHYSICAL", "LOGICAL_TO_PHYSICAL_SCHEDULE", "RECURSIVE_LOGICAL_TO_PHYSICAL", "RECURSIVE_LOGICAL_TO_PHYSICAL_SCHEDULE", "build_5_10xpoisson_parcae_schedule", "build_schedule",
    "build_5_10xpoisson_parcae_source_mapping", "build_source_mapping", "poisson_probabilities", "poisson_metadata",
    "sample_middle_loop_counts", "normalize_middle_loop_counts", "left_alignment_tau", "no_op_mask",
    "PreludeNorm", "ParcaeInjection", "DiagonalInjection", "MiddleBlockStack", "ParcaeRecurrentUnit",
    "RecursiveLlama5_10xpoisson_parcaeModel", "RecursiveLlama5_10xpoisson_parcaeForCausalLM", "RecursiveLlamaForCausalLM",
    "make_dynamic_cache", "_bind_cache_middle_loop_count", "_validate_cache_capacity", "parameter_audit", "register_auto_class",
]
