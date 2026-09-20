#!/usr/bin/env python3
"""Convert Mellow-v0 into an eval-only compact AudioSmolLM2 artifact.

The converted artifact deliberately contains only Mellow's trained SmolLM2,
c2l, and projection/bridge weights.  HTSAT is not copied: the existing
AudioSmolLM2 MMAU/MMAR evaluators keep loading the repository's configured
external HTSAT checkpoint at runtime.

This is a CPU-only conversion.  The output is not a training checkpoint and
contains no optimizer, scheduler, RNG, cursor, or training completion state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_MELLOW_CHECKPOINT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/Mellow-v0/"
    "models--soham97--mellow/snapshots/83672db0dae28764e283210d5bb732621e903d8a/"
    "v0.ckpt"
)
DEFAULT_BASE_SMOLLM2 = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2")
DEFAULT_OUTPUT_DIR = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/converted/"
    "mellow_v0_audio_smollm2_compact_eval"
)

ARTIFACT_CONTRACT = "mellow_v0_to_audio_smollm2_compact_eval_v1"
CONFIG_FILENAME = "mellow_audio_smollm2_eval_config.json"
MARKER_FILENAME = "artifact_complete.json"

SOURCE_PREFIXES = {
    "htsat": "audio_encoder.base.htsat.",
    "c2l": "audio_encoder.base.c2l.",
    "bridge": "audio_encoder.projection.",
    "text": "caption_decoder.lm.",
}
BRIDGE_KEY_MAP = {
    "linear1.weight": "linear1.weight",
    "linear2.weight": "linear2.weight",
    "layer_norm.weight": "norm.weight",
    "layer_norm.bias": "norm.bias",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_state_dict(payload: Any) -> dict[str, Any]:
    import torch

    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping) or not payload:
        raise RuntimeError("Mellow checkpoint does not contain a non-empty state dictionary")
    state: dict[str, Any] = {}
    for raw_key, value in payload.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module."):]
        if key in state:
            raise RuntimeError(f"duplicate Mellow state key after normalization: {key}")
        if not torch.is_tensor(value):
            raise RuntimeError(f"Mellow state value is not a tensor: {key} ({type(value).__name__})")
        state[key] = value.detach().cpu()
    return state


def _extract_group(state: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def _load_source_groups(checkpoint: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    import torch

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with an older PyTorch while remaining CPU-only.
        payload = torch.load(checkpoint, map_location="cpu")
    state = _canonical_state_dict(payload)
    groups = {
        name: _extract_group(state, prefix)
        for name, prefix in SOURCE_PREFIXES.items()
    }
    empty = [name for name, group in groups.items() if not group]
    if empty:
        raise RuntimeError(f"Mellow checkpoint is missing required parameter groups: {empty}")
    known = tuple(SOURCE_PREFIXES.values())
    unknown = sorted(key for key in state if not key.startswith(known))
    if unknown:
        raise RuntimeError(f"Mellow checkpoint contains unrecognized state keys: {unknown[:20]}")
    return groups, {
        "total_tensor_count": len(state),
        "group_tensor_counts": {name: len(group) for name, group in groups.items()},
        "ignored_htsat_tensor_count": len(groups["htsat"]),
    }


def _convert_bridge(source: Mapping[str, Any]) -> dict[str, Any]:
    from audio_5_10x2_5_mesh_mellow.model import AudioBridge

    if set(source) != set(BRIDGE_KEY_MAP):
        missing = sorted(set(BRIDGE_KEY_MAP) - set(source))
        unexpected = sorted(set(source) - set(BRIDGE_KEY_MAP))
        raise RuntimeError(
            f"Mellow projection state differs from the AudioBridge contract: "
            f"missing={missing} unexpected={unexpected}"
        )
    target_state = {target: source[source_key] for source_key, target in BRIDGE_KEY_MAP.items()}
    bridge = AudioBridge(768, 576, kernel=8, dropout=0.5)
    bridge.load_state_dict(target_state, strict=True)
    return {key: value.detach().cpu() for key, value in bridge.state_dict().items()}


def _convert_c2l(source: Mapping[str, Any]) -> dict[str, Any]:
    from torch import nn

    layer = nn.Linear(527, 768)
    layer.load_state_dict(dict(source), strict=True)
    return {key: value.detach().cpu() for key, value in layer.state_dict().items()}


def _convert_text_and_tokenizer(
    source: Mapping[str, Any],
    base_smollm2: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from audio_smollm2_135m_mellow.model import validate_original_smollm2

    text_model = AutoModelForCausalLM.from_pretrained(base_smollm2, local_files_only=True)
    initial_contract = validate_original_smollm2(text_model)
    input_weight = source.get("model.embed_tokens.weight")
    output_weight = source.get("lm_head.weight")
    if input_weight is None or output_weight is None or not torch.equal(input_weight, output_weight):
        raise RuntimeError(
            "Mellow SmolLM2 input embedding and LM head are not an exact tied-weight pair"
        )
    text_model.load_state_dict(dict(source), strict=True)
    loaded_contract = validate_original_smollm2(text_model)
    if initial_contract["vocab_size"] != loaded_contract["vocab_size"]:
        raise RuntimeError("SmolLM2 vocabulary size changed while loading Mellow weights")

    tokenizer = AutoTokenizer.from_pretrained(base_smollm2, local_files_only=True)
    tokenizer.add_special_tokens({"pad_token": "!"})
    separator_id = tokenizer.convert_tokens_to_ids("!")
    vocab_size = int(text_model.config.vocab_size)
    if separator_id is None or int(separator_id) < 0 or int(separator_id) >= vocab_size:
        raise RuntimeError(
            f"Mellow separator token '!' is outside the text-model vocabulary: "
            f"id={separator_id} vocab_size={vocab_size}"
        )
    if len(tokenizer) != vocab_size:
        raise RuntimeError(
            "adding Mellow's existing '!' pad token unexpectedly changed tokenizer vocabulary: "
            f"tokenizer={len(tokenizer)} model={vocab_size}"
        )

    text_model.eval()
    text_model.save_pretrained(output_root / "text_model", safe_serialization=True)
    tokenizer.save_pretrained(output_root / "tokenizer")
    return loaded_contract, {
        "pad_token": tokenizer.pad_token,
        "pad_token_id": int(tokenizer.pad_token_id),
        "separator_token": "!",
        "separator_token_id": int(separator_id),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": int(tokenizer.eos_token_id),
        "tokenizer_size": len(tokenizer),
    }


def _weight_files(path: Path) -> list[Path]:
    return sorted([
        *path.glob("pytorch_model*.bin"),
        *path.glob("model*.safetensors"),
    ])


def convert(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    checkpoint = args.mellow_checkpoint.resolve()
    base_smollm2 = args.base_smollm2.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Mellow-v0 checkpoint not found: {checkpoint}")
    if not (base_smollm2 / "config.json").is_file():
        raise FileNotFoundError(f"local SmolLM2 model is incomplete: {base_smollm2}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing conversion artifact: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        suffix=".tmp",
        dir=str(output_dir.parent),
    ))
    published = False
    try:
        groups, state_audit = _load_source_groups(checkpoint)
        bridge_state = _convert_bridge(groups["bridge"])
        c2l_state = _convert_c2l(groups["c2l"])
        text_contract, tokenizer_audit = _convert_text_and_tokenizer(
            groups["text"], base_smollm2, temporary
        )
        torch.save({"bridge": bridge_state, "c2l": c2l_state}, temporary / "audio_bridge.pt")

        model_weights = _weight_files(temporary / "text_model")
        if not model_weights:
            raise RuntimeError("converted artifact has no saved text-model weights")
        required = [
            "text_model/config.json",
            "tokenizer/tokenizer_config.json",
            "audio_bridge.pt",
            CONFIG_FILENAME,
            MARKER_FILENAME,
        ]
        config = {
            "artifact_contract": ARTIFACT_CONTRACT,
            "artifact_kind": "eval_only_model",
            "source_model": "soham97/mellow:v0",
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": _sha256(checkpoint),
            "base_smollm2": str(base_smollm2),
            "architecture_contract": "original_smollm2_135m_standard_llama_30_layers_hidden576_audio_mellow",
            "mapper_contract": (
                "mellow_c2l_527x768__concat_cls_frames__projection_768x576x576_"
                "biasfree_dropout0.5__cls_preserving_avgpool8"
            ),
            "compact_single_audio_prefix": True,
            "audio_tokens_per_clip": 129,
            "prefix_tokens": {"single": 130, "dual": 260},
            "sample_rate": 32000,
            "audio_seconds": 10,
            "max_prompt_tokens": 129,
            "max_context_length": 768,
            "external_htsat": {
                "included": False,
                "required_at_runtime": True,
                "policy": "use evaluator --htsat-checkpoint and --mellow-root",
            },
            "standard_text_contract": text_contract,
            "tokenizer": tokenizer_audit,
            "source_state_audit": state_audit,
            "converted_state_tensor_counts": {
                "text": len(groups["text"]),
                "bridge": len(bridge_state),
                "c2l": len(c2l_state),
            },
            "training_state_included": False,
        }
        config_path = temporary / CONFIG_FILENAME
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        marker = {
            "status": "complete",
            "artifact_contract": ARTIFACT_CONTRACT,
            "artifact_kind": "eval_only_model",
            "required": required,
            "config_sha256": _sha256(config_path),
            "text_model_weight_files": [path.name for path in model_weights],
            "htsat_included": False,
        }
        (temporary / MARKER_FILENAME).write_text(
            json.dumps(marker, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        missing = [name for name in required if not (temporary / name).is_file()]
        if missing:
            raise RuntimeError(f"refusing to publish incomplete eval-only artifact: {missing}")
        temporary.replace(output_dir)
        published = True
        return {
            "status": "PASS",
            "artifact_contract": ARTIFACT_CONTRACT,
            "output_dir": str(output_dir),
            "source_checkpoint_sha256": config["source_checkpoint_sha256"],
            "text_model_weight_files": [str(output_dir / "text_model" / path.name) for path in model_weights],
            "htsat_included": False,
            "compact_single_audio_prefix": True,
        }
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mellow-checkpoint", type=Path, default=DEFAULT_MELLOW_CHECKPOINT)
    parser.add_argument("--base-smollm2", type=Path, default=DEFAULT_BASE_SMOLLM2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = convert(parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
