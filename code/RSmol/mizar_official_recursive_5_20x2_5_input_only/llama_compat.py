"""Reference Llama calling/causal-mask compatibility helpers, unchanged."""
import inspect
import warnings
from typing import Any
import torch
from torch import nn
from transformers.cache_utils import DynamicCache
SUPPORTED_TRANSFORMERS_VERSION = "4.54.1"

def _assert_supported_llama_api(layer: nn.Module) -> None:
    """Validate the singular-cache Llama API used by Transformers 4.54.1."""

    import transformers

    installed = str(getattr(transformers, "__version__", "unknown"))
    if installed != SUPPORTED_TRANSFORMERS_VERSION:
        warnings.warn(
            "RecursiveLlamaForCausalLM 5-10-5 was designed for "
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
