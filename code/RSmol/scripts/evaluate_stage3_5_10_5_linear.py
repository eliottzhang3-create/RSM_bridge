#!/usr/bin/env python3
"""Stage 3 lm-eval entry point for the isolated 5-10-5 linear baseline.

The benchmark protocol and task overlays are inherited from the already
validated 5-10-5 evaluator.  Only the model contract/audit is different:
this checkpoint has twenty physical layers, twenty logical executions, and
no recurrent middle loop.  Keeping this adapter separate prevents the
recursive evaluator from accepting a linear checkpoint accidentally.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(name: str, path: Path) -> Any:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load(
    "rsmol_stage3_linear_base", SCRIPT_ROOT / "scripts" / "evaluate_stage3_5_10_5.py"
)

DEFAULT_MODEL = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-10-5-linear"
)
MODEL_LABEL = "linear_5_10_5"
ARCHITECTURE_CONTRACT = "logical_20_physical_20_5_10_5_linear_loops_1"
LOGICAL_LAYER_COUNT = 20
PHYSICAL_LAYER_COUNT = 20
LOGICAL_TO_PHYSICAL = tuple(range(20))
SOURCE_MAPPING_0BASED = (
    0, 1, 2, 3, 4, 5, 7, 9, 11, 13,
    15, 17, 19, 21, 23, 25, 26, 27, 28, 29,
)
EXPECTED_ARCHITECTURES = ("SmolLM2_5_10_5LinearForCausalLM",)


def _load_linear() -> Any:
    return _load(
        "code.RSmol.recursive_model_5_10_5_linear",
        SCRIPT_ROOT / "recursive_model_5_10_5_linear.py",
    )


def _exact_tuple(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None


def inspect_model_artifacts_linear(path: Path) -> dict[str, Any]:
    model_dir = BASE.ensure_external_path(path, label="5-10-5 linear model")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"linear model is not an existing directory: {model_dir}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"linear model is missing config.json: {config_path}")
    config = BASE._read_json(config_path)
    if config.get("model_type") != "llama":
        raise ValueError(f"linear checkpoint must retain model_type='llama', got {config.get('model_type')!r}")
    architectures = tuple(str(x) for x in config.get("architectures", ()))
    logical = int(config.get("num_hidden_layers", 0))
    physical = int(config.get("recursive_layer_count", 0))
    loops = int(config.get("recursive_loops", 0))
    source_layer_count = int(config.get("recursive_source_layer_count", 30))
    loops_scope = str(config.get("recursive_loops_scope", ""))
    schedule = _exact_tuple(config.get("logical_to_physical"))
    source = _exact_tuple(config.get("recursive_source_layer_indices_0based"))
    if (logical, physical, loops, source_layer_count, loops_scope) != (20, 20, 1, 30, "none"):
        raise ValueError(
            "linear architecture contract mismatch: "
            f"logical={logical} physical={physical} loops={loops} "
            f"source_layer_count={source_layer_count} loops_scope={loops_scope!r}"
        )
    if architectures != EXPECTED_ARCHITECTURES:
        raise ValueError(f"linear architectures mismatch: {architectures}")
    if schedule != LOGICAL_TO_PHYSICAL or source != SOURCE_MAPPING_0BASED:
        raise ValueError(f"linear mapping mismatch: schedule={schedule} source={source}")
    model_files = BASE._model_file_manifest(model_dir)
    tokenizer_files = BASE._tokenizer_file_manifest(model_dir)
    if not model_files:
        raise FileNotFoundError(f"linear model has no local weights: {model_dir}")
    if "tokenizer_config.json" not in tokenizer_files or not ({"tokenizer.json", "tokenizer.model"} & set(tokenizer_files)):
        raise FileNotFoundError("linear checkpoint is missing tokenizer artifacts")
    return {
        "path": str(model_dir),
        "label": MODEL_LABEL,
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "config": config,
        "config_vocab_size": config.get("vocab_size"),
        "architectures": list(architectures),
        "logical_layer_count": logical,
        "physical_layer_count": physical,
        "recursive_loops": loops,
        "recursive_loops_scope": loops_scope,
        "logical_to_physical": list(schedule),
        "source_mapping_0based": list(source),
        "model_files": model_files,
        "tokenizer_files": tokenizer_files,
    }


def _finite(name: str, tensor: Any) -> None:
    import torch
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"{name} contains non-finite values")


def load_and_audit_linear(model_path: Path, *, tokenizer: Any, device: str, dtype: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    module = _load_linear()
    module.register_auto_class()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=getattr(torch, dtype), low_cpu_mem_usage=True
    )
    model.to(torch.device(device))
    try:
        expected_class = module.SmolLM2_5_10_5LinearForCausalLM
        if not isinstance(model, expected_class):
            raise TypeError(f"AutoModel resolved wrong linear class: {type(model)!r}")
        config = model.config
        layers = list(model.model.layers)
        if len(layers) != PHYSICAL_LAYER_COUNT or len({id(x) for x in layers}) != PHYSICAL_LAYER_COUNT:
            raise RuntimeError("linear physical layer object audit failed")
        if int(config.num_hidden_layers) != LOGICAL_LAYER_COUNT:
            raise RuntimeError(f"linear logical depth mismatch: {config.num_hidden_layers}")
        if tuple(model.model.logical_to_physical) != LOGICAL_TO_PHYSICAL:
            raise RuntimeError("linear runtime schedule mismatch")
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
        trace: list[int] = []
        hooks = [layer.register_forward_hook(lambda _m, _i, _o, i=i: trace.append(i)) for i, layer in enumerate(layers)]
        model.eval()
        with torch.inference_mode():
            no_cache = model(input_ids=input_ids, use_cache=False)
        for hook in hooks:
            hook.remove()
        if trace != list(range(20)):
            raise RuntimeError(f"linear forward trace mismatch: {trace}")
        with torch.inference_mode():
            with_cache = model(input_ids=input_ids, use_cache=True)
        _finite("linear no-cache logits", no_cache.logits)
        _finite("linear cache logits", with_cache.logits)
        diff = float((no_cache.logits - with_cache.logits).abs().max().item())
        if not torch.allclose(no_cache.logits, with_cache.logits, atol=BASE.CACHE_ATOL, rtol=BASE.CACHE_RTOL):
            raise RuntimeError(f"linear cache/no-cache mismatch: max_diff={diff}")
        encoded = tokenizer("Linear models are", return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=1, do_sample=False, use_cache=True, pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id)
        if generated.shape[1] <= encoded["input_ids"].shape[1]:
            raise RuntimeError("linear generation produced no new token")
        return {
            "status": "PASS", "model_class": type(model).__name__,
            "architecture_contract": ARCHITECTURE_CONTRACT,
            "logical_layer_count": LOGICAL_LAYER_COUNT,
            "physical_layer_count": PHYSICAL_LAYER_COUNT,
            "recursive_loops": 1, "loops_scope": "none",
            "logical_to_physical": list(LOGICAL_TO_PHYSICAL),
            "forward_trace": trace, "expected_forward_trace": list(range(20)),
            "cache_prefill_max_diff": diff,
            "generation_output_shape": list(generated.shape),
        }
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Patch only the imported base module.  Its lm-eval task protocol, reporting,
# offline overlays, and CLI remain identical, while all architecture-specific
# checks resolve to this linear implementation.
BASE.DEFAULT_MODEL = DEFAULT_MODEL
BASE.MODEL_LABEL = MODEL_LABEL
BASE.ARCHITECTURE_CONTRACT = ARCHITECTURE_CONTRACT
BASE.LOGICAL_LAYER_COUNT = LOGICAL_LAYER_COUNT
BASE.PHYSICAL_LAYER_COUNT = PHYSICAL_LAYER_COUNT
BASE.RECURSIVE_LOOPS = 1
BASE.LOOPS_SCOPE = "none"
BASE.LOGICAL_TO_PHYSICAL = LOGICAL_TO_PHYSICAL
BASE.SOURCE_MAPPING_0BASED = SOURCE_MAPPING_0BASED
BASE.EXPECTED_ARCHITECTURES = EXPECTED_ARCHITECTURES
BASE._load_recursive_5_10_5 = _load_linear
BASE.inspect_model_artifacts_5_10_5 = inspect_model_artifacts_linear
BASE.load_and_audit_recursive_model_5_10_5 = load_and_audit_linear
BASE.run_evaluation_5_10_5.__globals__["inspect_model_artifacts_5_10_5"] = inspect_model_artifacts_linear
BASE.run_evaluation_5_10_5.__globals__["load_and_audit_recursive_model_5_10_5"] = load_and_audit_linear


if __name__ == "__main__":
    raise SystemExit(BASE.main())
