#!/usr/bin/env python3
"""Offline CPU conversion: retain all 30 source layers, execute 5 + 20x2 + 5.

Publishes only after strict weight, forward-trace, source-parity and reload
checks pass. Existing output is refused; failed staging is retained for diagnosis.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

FORBIDDEN_CHECKOUT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM")
LOCAL_CHECKOUT = SCRIPT_ROOT.parents[1]
DEFAULT_OUTPUT_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2-5-20x2-5")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
TOKENIZER_NAMES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "tokenizer.model", "spiece.model", "vocab.json", "merges.txt",
    "added_tokens.json", "generation_config.json",
}
SOURCE_LAYER_INDICES_0BASED = tuple(range(30))
LOGICAL_TO_PHYSICAL = tuple(range(25)) + tuple(range(5, 30))
MAPPING_POLICY = "explicit_5_20x2_5_source_layers"
TEXT_ROUTE = "text_5_20x2_5_gbs512"
DEFAULT_SOURCE_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2")


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def reject_forbidden_output(path: Path) -> Path:
    candidate = _resolved(path)
    for forbidden in (_resolved(FORBIDDEN_CHECKOUT), _resolved(LOCAL_CHECKOUT)):
        try:
            candidate.relative_to(forbidden)
        except ValueError:
            continue
        raise ValueError(
            "Refusing to write model parameters inside a Git checkout: "
            f"output={candidate} forbidden_root={forbidden}"
        )
    return candidate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args(argv)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def json_safe(value: Any) -> Any:
    try:
        import torch
    except ImportError:
        torch = None
    if isinstance(value, Path):
        return str(value)
    if torch is not None and isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def source_config_summary(config: Any) -> dict[str, Any]:
    keys = (
        "model_type", "architectures", "num_hidden_layers", "hidden_size",
        "intermediate_size", "num_attention_heads", "num_key_value_heads",
        "vocab_size", "max_position_embeddings", "torch_dtype", "bos_token_id",
        "eos_token_id", "pad_token_id",
    )
    return {key: json_safe(getattr(config, key, None)) for key in keys}


def copy_module_checked(source: Any, target: Any, name: str) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    extra = sorted(set(source_state) - set(target_state))
    if missing or extra:
        raise ValueError(f"{name} state keys differ: missing_source={missing} extra_source={extra}")
    for key, target_tensor in target_state.items():
        source_tensor = source_state[key]
        if source_tensor.shape != target_tensor.shape or source_tensor.dtype != target_tensor.dtype:
            raise ValueError(
                f"{name}.{key} mismatch: source={tuple(source_tensor.shape)}/{source_tensor.dtype} "
                f"target={tuple(target_tensor.shape)}/{target_tensor.dtype}"
            )
    target.load_state_dict(source_state, strict=True)


def copy_non_layer_modules(source: Any, target: Any) -> None:
    for name in ("embed_tokens", "norm"):
        copy_module_checked(getattr(source.model, name), getattr(target.model, name), f"model.{name}")
    copy_module_checked(source.lm_head, target.lm_head, "lm_head")


def copy_tokenizer_files(source_dir: Path, target_dir: Path) -> list[str]:
    copied: list[str] = []
    for source_file in source_dir.iterdir():
        if not source_file.is_file():
            continue
        name = source_file.name
        if name == "config.json":
            continue
        is_tokenizer = name in TOKENIZER_NAMES or name.startswith("tokenizer") or name.endswith((".model", ".jinja"))
        if is_tokenizer and not name.endswith(WEIGHT_SUFFIXES):
            shutil.copy2(source_file, target_dir / name)
            copied.append(name)
    return sorted(set(copied))


def build_source_mapping(num_hidden_layers: int = 30) -> tuple[int, ...]:
    if int(num_hidden_layers) != 30:
        raise ValueError("Conversion requires an original 30-layer source")
    return SOURCE_LAYER_INDICES_0BASED


def build_target_config(source_config: Any) -> Any:
    build_source_mapping(source_config.num_hidden_layers)
    if getattr(source_config, "recursive_layer_count", None) is not None:
        raise ValueError("Recursive sources are forbidden; initialize from original SmolLM2")
    target = copy.deepcopy(source_config)
    target.num_hidden_layers = 50  # logical executions, not stored modules
    target.recursive_layer_count = 30
    target.recursive_loops = 2
    target.recursive_prefix_layer_count = 5
    target.recursive_middle_layer_count = 20
    target.recursive_suffix_layer_count = 5
    target.middle_recurrent_count = 20
    target.recursive_loops_scope = "middle_only"
    for key in ("logical_to_physical", "recursive_logical_to_physical",
                "logical_to_physical_schedule", "recursive_logical_to_physical_schedule"):
        setattr(target, key, list(LOGICAL_TO_PHYSICAL))
    for suffix, mapping in (("0based", list(range(30))), ("1based", list(range(1, 31)))):
        setattr(target, f"recursive_source_layer_indices_{suffix}", mapping)
        setattr(target, f"recursive_source_mapping_{suffix}", list(mapping))
    target.recursive_prefix_source_layers_1based = list(range(1, 6))
    target.recursive_middle_source_layers_1based = list(range(6, 26))
    target.recursive_suffix_source_layers_1based = list(range(26, 31))
    target.recursive_mapping_policy = MAPPING_POLICY
    target.text_route = TEXT_ROUTE
    target.architectures = ["RecursiveLlama5_20x2_5ForCausalLM"]
    return target


def assert_weights_equal(source: Any, target: Any) -> int:
    import torch

    left, right = source.state_dict(), target.state_dict()
    if set(left) != set(right):
        raise AssertionError("Full model state keys differ")
    for name, value in left.items():
        other = right[name]
        if value.dtype != other.dtype or value.shape != other.shape or not torch.equal(value, other):
            raise AssertionError(f"Inherited weight changed: {name}")
    return len(left)


def assert_tied_weights(model: Any) -> None:
    tied = model.model.embed_tokens.weight is model.lm_head.weight
    if tied != bool(model.config.tie_word_embeddings):
        raise AssertionError("Embedding/lm_head sharing differs from source config")


def audit_forward(source: Any, target: Any) -> tuple[dict[str, Any], Any]:
    """Check ordinary 30-layer parity, then the actual shared 50-entry route."""
    import torch

    source.eval()
    target.eval()
    ids = (torch.arange(12).reshape(2, 6) + 3) % source.config.vocab_size
    mask = torch.ones_like(ids)
    mask[1, -2:] = 0
    original_schedule = target.model.logical_to_physical
    try:
        target.model.logical_to_physical = tuple(range(30))
        with torch.no_grad():
            reference = source(input_ids=ids, attention_mask=mask, use_cache=False).logits
            single = target(input_ids=ids, attention_mask=mask, use_cache=False).logits
        # Only valid query positions enter the comparison (padding is not supervised).
        torch.testing.assert_close(single[mask.bool()], reference[mask.bool()], rtol=1e-5, atol=1e-6)
    finally:
        target.model.logical_to_physical = original_schedule
    trace: list[int] = []
    handles = [layer.register_forward_pre_hook(lambda _m, _a, i=i: trace.append(i))
               for i, layer in enumerate(target.model.layers)]
    try:
        with torch.no_grad():
            recurrent = target(input_ids=ids, attention_mask=mask, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()
    if trace != list(LOGICAL_TO_PHYSICAL):
        raise AssertionError(f"Recursive execution trace differs: {trace}")
    if not torch.isfinite(recurrent).all():
        raise FloatingPointError("Nonfinite converted logits")
    return {"source_single_pass_parity": True, "single_pass_max_abs_error":
            float((single[mask.bool()] - reference[mask.bool()]).abs().max()),
            "recursive_trace": trace, "finite_logits": True}, recurrent


def convert(args: argparse.Namespace) -> dict[str, Any]:
    source = _resolved(args.source_checkpoint)
    output = reject_forbidden_output(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing existing conversion output: {output}")
    if source == output or source in output.parents:
        raise ValueError("Conversion output must be separate from the source model")
    if not source.is_dir():
        raise FileNotFoundError(source)
    raw = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if type(raw.get("num_hidden_layers")) is not int or raw["num_hidden_layers"] != 30:
        raise ValueError("Original source config must have num_hidden_layers=30")
    if raw.get("model_type") != "llama" or raw.get("architectures") != ["LlamaForCausalLM"]:
        raise ValueError("Original Llama/SmolLM2 source required")
    if any(key.startswith(("recursive_", "mesh_", "lora_")) or key == "text_route" for key in raw):
        raise ValueError("Recursive/adapted source is forbidden")
    if any((source / name).exists() for name in ("training_state.pt", "checkpoint_complete.json",
                                                "recursive_conversion_metadata.json")):
        raise ValueError("Training/converted checkpoint cannot be used as the original source")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    import torch
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
    from recursive_model_5_20x2_5 import RecursiveLlama5_20x2_5ForCausalLM, parameter_audit

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    source_config = AutoConfig.from_pretrained(source, local_files_only=True)
    # Explicit classes keep process-global AutoModel registrations irrelevant.
    source_model, loading = LlamaForCausalLM.from_pretrained(
        source, local_files_only=True, use_safetensors=True, output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Source strict loading failed: {loading}")
    if len(source_model.model.layers) != 30:
        raise ValueError("Source must store exactly 30 decoder modules")
    # Use the loaded configuration so its resolved attention implementation is preserved.
    target_config = build_target_config(source_model.config)
    target_model = RecursiveLlama5_20x2_5ForCausalLM(target_config)
    target_model.to(dtype=next(source_model.parameters()).dtype)
    copy_non_layer_modules(source_model, target_model)
    for index in build_source_mapping(source_config.num_hidden_layers):
        copy_module_checked(source_model.model.layers[index], target_model.model.layers[index],
                            f"model.layers[{index}]")
    target_model.tie_weights()
    checked_tensors = assert_weights_equal(source_model, target_model)
    assert_tied_weights(source_model)
    assert_tied_weights(target_model)
    audit, logits = audit_forward(source_model, target_model)
    audit.update(parameter_audit(target_model))
    audit["equal_inherited_tensors"] = checked_tensors
    audit["all_parameters_trainable"] = all(p.requires_grad for p in target_model.parameters())
    if not audit["all_parameters_trainable"]:
        raise AssertionError("Conversion unexpectedly froze parameters")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        target_model.save_pretrained(staging, safe_serialization=True)
        copied = copy_tokenizer_files(source, staging)
        reloaded, loading = RecursiveLlama5_20x2_5ForCausalLM.from_pretrained(
            staging, local_files_only=True, output_loading_info=True,
            torch_dtype=next(target_model.parameters()).dtype,
            attn_implementation=target_model.config._attn_implementation,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Converted strict reload failed: {loading}")
        assert_weights_equal(target_model, reloaded)
        assert_tied_weights(reloaded)
        reloaded.eval()
        ids = (torch.arange(12).reshape(2, 6) + 3) % source_config.vocab_size
        mask = torch.ones_like(ids)
        mask[1, -2:] = 0
        with torch.no_grad():
            actual = reloaded(input_ids=ids, attention_mask=mask, use_cache=False).logits
        torch.testing.assert_close(actual, logits, rtol=0, atol=0)
        restored_tokenizer = AutoTokenizer.from_pretrained(staging, local_files_only=True, use_fast=True)
        if tokenizer.get_vocab() != restored_tokenizer.get_vocab() or tokenizer.special_tokens_map != restored_tokenizer.special_tokens_map:
            raise AssertionError("Saved tokenizer differs")
        audit["reload_weights_logits_tokenizer_equal"] = True
        metadata = {
            "status": "PASS", "conversion": "SmolLM2-5-20x2-5", "text_route": TEXT_ROUTE,
            "source_checkpoint": str(source), "target_output": str(output),
            "source_config": source_config_summary(source_config), "source_config_json": raw,
            "target_config": target_model.config.to_dict(), "source_layers": 30,
            "physical_layer_count": 30, "logical_layer_count": 50,
            "source_layer_indices_0based": list(range(30)),
            "logical_to_physical": list(LOGICAL_TO_PHYSICAL), "mapping_policy": MAPPING_POLICY,
            "parameter_audit": audit, "copied_tokenizer_files": copied,
            "seed": args.seed, "cpu_threads": args.threads, "code_commit": git_commit(),
            "torch": torch.__version__, "transformers": package_version("transformers"),
            "python": sys.version, "platform": platform.platform(),
            "conversion_time_utc": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "recursive_conversion_metadata.json").write_text(
            json.dumps(json_safe(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Conversion target appeared during conversion: {output}")
        staging.replace(output)
    except Exception as exc:
        (staging / "conversion_failure.json").write_text(json.dumps({
            "status": "FAIL", "error": repr(exc), "staging": str(staging),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"[conversion] Failed staging preserved: {staging}", file=sys.stderr, flush=True)
        raise
    return metadata


def main(argv: list[str] | None = None) -> None:
    metadata = convert(parse_args(argv))
    print(json.dumps(json_safe(metadata), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


