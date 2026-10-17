"""Load only the trained MeSH text weights, retaining formal30 runtime settings."""

import contextlib
import gc
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from route_contract import BASELINE_TEXT, validate_mesh_config, validate_sources


@contextlib.contextmanager
def preserve_rng():
    # Loading happens on CPU. Also preserve Python/NumPy and initialized CUDA
    # generators so added audits cannot perturb bridge/dropout/worker seeds.
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    python = random.getstate()
    numpy = np.random.get_state()
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        random.setstate(python)
        np.random.set_state(numpy)


def load_recursive_text_model(model_path):
    validate_sources(model_path, BASELINE_TEXT)
    # Execute the exact ordinary-text load used by formal30, retaining its RNG
    # consumption. This is also used for rank-0's preliminary load. The extra
    # recursive construction below must not advance the training RNG stream.
    reference = AutoModelForCausalLM.from_pretrained(str(BASELINE_TEXT))
    reference_config = reference.config
    reference_attention = getattr(reference_config, "_attn_implementation", None)
    if not reference_attention:
        raise ValueError("cannot establish baseline attention implementation")
    if any(p.dtype != torch.float32 for p in reference.parameters() if p.is_floating_point()):
        raise ValueError("baseline text model is not FP32; comparison contract changed")
    del reference
    gc.collect()
    with preserve_rng():
        route_root = Path(__file__).resolve().parents[1]
        if str(route_root) not in sys.path:
            sys.path.insert(0, str(route_root))
        from recursive_model_perceiver_5_10x2_5_mesh import RecursiveLlamaForCausalLM, parameter_audit
        # Do not register LlamaConfig globally: the reference AutoModel must
        # continue to resolve to the ordinary SmolLM2 class in this process.
        model, loading = RecursiveLlamaForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.float32,
            attn_implementation=reference_attention,
            output_loading_info=True,
        )
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if loading.get(key):
                raise ValueError(f"recursive checkpoint did not load completely: {key}={loading[key]}")
        validate_mesh_config(model.config)
        if getattr(model.config, "_attn_implementation", None) != reference_attention:
            raise ValueError("loaded recursive attention implementation differs from baseline")
        for field in (
            "hidden_size", "vocab_size", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "rms_norm_eps", "rope_theta", "rope_scaling",
            "max_position_embeddings", "attention_dropout", "hidden_act",
            "attention_bias", "mlp_bias", "tie_word_embeddings", "pad_token_id",
        ):
            if getattr(model.config, field, None) != getattr(reference_config, field, None):
                raise ValueError(f"non-architectural text config differs from baseline: {field}")
        # formal30 passes neither cache flags nor output flags; match the
        # reference defaults rather than importing the old MeSH adapter policy.
        for field in ("use_cache", "return_dict", "output_attentions", "output_hidden_states"):
            setattr(model.config, field, getattr(reference_config, field))
        audit = parameter_audit(model)
        for field, value in (("logical_layer_count", 30), ("physical_layer_count", 20),
                             ("memory_slots", 5), ("router_count", 6)):
            if audit.get(field) != value:
                raise ValueError(f"invalid recursive parameter audit: {audit}")
        if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
            raise ValueError("recursive text parameters must all be FP32")
        if model.get_input_embeddings().weight.shape != model.get_output_embeddings().weight.shape:
            raise ValueError("recursive embedding/lm_head shapes differ")
        model.formal30_loading_audit = {
            "parameter_audit": audit, "dtype": "float32",
            "attention_implementation": model.config._attn_implementation,
            "use_cache": model.config.use_cache, "loading_info": loading,
            "rng_policy": "retain_baseline_load_consumption_isolate_recursive_load",
        }
    return model
