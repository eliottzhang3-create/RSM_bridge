#!/usr/bin/env python3
"""Create a variable-depth initialization artifact from the fixed T=2 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recursive_model_5_10x2_5_mesh import (  # noqa: E402
    MODEL_ARCHITECTURE_CONTRACT as SOURCE_ARCHITECTURE_CONTRACT,
    RecursiveLlamaForCausalLM as FixedRecursiveLlamaForCausalLM,
)
from recursive_model_5_10x2to10_5_mesh import (  # noqa: E402
    MAX_LOGICAL_LAYER_COUNT, MAX_RECURSIVE_DEPTH, MIN_RECURSIVE_DEPTH,
    MODEL_ARCHITECTURE_CONTRACT, RecursiveLlamaForCausalLM,
)

DEFAULT_SOURCE = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "audio_5_10x2_5_mesh_mellow_shared_store/"
    "formal_fixed260_3ep_20260922_v1/checkpoint-011343"
)
MIGRATION_CONTRACT = "fixed_t2_checkpoint_to_uniform_t2_10_router7_v1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_fixed260_prefix_contract(value: Any) -> bool:
    """Accept both the current mapping schema and the older scalar schema.

    Current checkpoints serialize prefix_tokens as a mapping with single and
    dual values. Older artifacts may use scalar 260. Compact 130/260 artifacts
    must still be rejected.
    """
    if isinstance(value, dict):
        if set(value) != {"single", "dual"}:
            return False
        try:
            return int(value["single"]) == 260 and int(value["dual"]) == 260
        except (TypeError, ValueError):
            return False
    try:
        return int(value) == 260
    except (TypeError, ValueError):
        return False


def _validate_source(source: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    required = (
        source / "mesh_model" / "config.json", source / "tokenizer" / "tokenizer_config.json",
        source / "audio_bridge.pt", source / "training_state.pt",
        source / "audio_mesh_config.json", source / "checkpoint_complete.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source checkpoint is incomplete: {missing}")
    marker = _json(source / "checkpoint_complete.json")
    audio_config = _json(source / "audio_mesh_config.json")
    mesh_config = _json(source / "mesh_model" / "config.json")
    training_state = torch.load(
        source / "training_state.pt", map_location="cpu", weights_only=False, mmap=True,
    )
    if marker.get("status") != "complete" or int(marker.get("global_step", -1)) != 11343:
        raise RuntimeError(f"unexpected source completion marker: {marker}")
    if int(training_state.get("global_step", -1)) != 11343:
        raise RuntimeError("source training_state global_step is not 11343")
    if audio_config.get("architecture_contract") not in {
        SOURCE_ARCHITECTURE_CONTRACT, SOURCE_ARCHITECTURE_CONTRACT + "_audio_mellow",
        "logical_30_physical_20_5_10x2_5_mesh_audio_mellow",
    }:
        raise RuntimeError("source is not the audited fixed 5-10x2-5 Audio MeSH architecture")
    prefix_tokens = audio_config.get("prefix_tokens")
    if not _is_fixed260_prefix_contract(prefix_tokens):
        raise RuntimeError(
            "source checkpoint does not prove the fixed 260-token two-slot contract: "
            f"prefix_tokens={prefix_tokens!r}"
        )
    try:
        prefix_with_separators = int(audio_config.get("audio_prefix_tokens_with_separators", -1))
    except (TypeError, ValueError):
        prefix_with_separators = -1
    if prefix_with_separators != 260:
        raise RuntimeError(
            "source checkpoint has an invalid total audio-prefix length: "
            f"audio_prefix_tokens_with_separators={audio_config.get('audio_prefix_tokens_with_separators')!r}"
        )
    if (int(mesh_config.get("num_hidden_layers", -1)),
            int(mesh_config.get("recursive_layer_count", -1)),
            int(mesh_config.get("recursive_loops", -1))) != (30, 20, 2):
        raise RuntimeError("source mesh config is not logical30/physical20/T2")
    return {
        "source": str(source), "global_step": 11343,
        "training_contract": audio_config.get("contract"),
        "architecture_contract": audio_config.get("architecture_contract"),
        "prefix_tokens": prefix_tokens,
        "audio_prefix_tokens_with_separators": prefix_with_separators,
        "mesh_config_sha256": _sha256(source / "mesh_model" / "config.json"),
        "audio_bridge_sha256": _sha256(source / "audio_bridge.pt"),
    }


def _migrate_state(source_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    replacements = {
        "model.write_routers.0.": "model.pre_write.",
        "model.read_routers.0.": "model.pre_read.",
        "model.write_routers.1.": "model.loop1_write.",
        "model.read_routers.1.": "model.loop1_read.",
        "model.write_routers.2.": "model.refine_write.",
        "model.read_routers.2.": "model.out_read.",
    }
    migrated: dict[str, torch.Tensor] = {}
    provenance: dict[str, str] = {}
    for key, value in source_state.items():
        destination = key
        for old, new in replacements.items():
            if key.startswith(old):
                destination = new + key[len(old):]
                provenance[destination] = key
                break
        migrated[destination] = value.detach().clone()
    for suffix in ("weight", "bias"):
        source_key = f"model.read_routers.1.{suffix}"
        destination = f"model.refine_read.{suffix}"
        migrated[destination] = source_state[source_key].detach().clone()
        provenance[destination] = source_key
    return migrated, provenance


@torch.inference_mode()
def _parity(source_model: Any, target_model: Any) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(20260925)
    vocab = int(source_model.config.vocab_size)
    input_ids = torch.randint(0, vocab, (2, 11), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    source_model.eval()
    target_model.eval()
    old = source_model(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=False, return_dict=True).logits
    new = target_model(input_ids=input_ids, attention_mask=attention_mask,
                       recursive_depth=2, use_cache=False, return_dict=True).logits
    absolute = (old.float() - new.float()).abs()
    max_abs = float(absolute.max().item())
    mean_abs = float(absolute.mean().item())
    passed = bool(torch.equal(old, new) or max_abs <= 1e-6)
    if not passed:
        raise RuntimeError(f"T=2 migration parity failed: max_abs={max_abs} mean_abs={mean_abs}")
    return {"status": "PASS", "shape": list(old.shape), "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs}


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_checkpoint.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite initialization artifact: {output}")
    source_audit = _validate_source(source)
    fixed = FixedRecursiveLlamaForCausalLM.from_pretrained(
        source / "mesh_model", torch_dtype=torch.float32, low_cpu_mem_usage=False,
    )
    config = fixed.config
    config.num_hidden_layers = MAX_LOGICAL_LAYER_COUNT
    config.recursive_layer_count = 20
    config.recursive_min_depth = MIN_RECURSIVE_DEPTH
    config.recursive_max_depth = MAX_RECURSIVE_DEPTH
    config.recursive_default_depth = 2
    config.recursive_loops = 2
    config.architectures = ["RecursiveLlama5_10x2to10_5MeshForCausalLM"]
    config.model_architecture_contract = MODEL_ARCHITECTURE_CONTRACT
    for stale in ("logical_to_physical", "logical_layer_count"):
        if hasattr(config, stale):
            delattr(config, stale)
    target = RecursiveLlamaForCausalLM(config)
    migrated, router_provenance = _migrate_state(fixed.state_dict())
    result = target.load_state_dict(migrated, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict migration mismatch: {result}")
    parity = _parity(fixed, target)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent))
    published = False
    try:
        target.save_pretrained(temporary / "mesh_model", safe_serialization=False)
        tokenizer = AutoTokenizer.from_pretrained(source / "tokenizer", local_files_only=True)
        tokenizer.save_pretrained(temporary / "tokenizer")
        shutil.copy2(source / "audio_bridge.pt", temporary / "audio_bridge.pt")
        report = {
            "status": "PASS", "migration_contract": MIGRATION_CONTRACT,
            "source": source_audit, "target_architecture_contract": MODEL_ARCHITECTURE_CONTRACT,
            "recursive_depth_range": [MIN_RECURSIVE_DEPTH, MAX_RECURSIVE_DEPTH],
            "default_training_epochs": 7, "router_provenance": router_provenance,
            "t2_exact_parity": parity,
            "not_copied": ["optimizer", "scheduler", "training cursor", "rank RNG"],
        }
        (temporary / "variable_depth_init_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        marker = {
            "status": "complete", "contract": MIGRATION_CONTRACT,
            "source_global_step": 11343,
            "required": ["mesh_model", "tokenizer", "audio_bridge.pt",
                         "variable_depth_init_report.json"],
        }
        (temporary / "artifact_complete.json").write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
        published = True
    finally:
        del fixed, target
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)
    return {"status": "PASS", "output_dir": str(output), **source_audit, "parity": parity}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
