"""Model/projector-only initialization from this route's completed audio files.

The source runtime and saved header must agree. Its historical LR, code hash,
optimizer, scheduler, scaler, GradNorm, RNG and cursor never become the new run's
training state. Full continuation remains the separate --resume-checkpoint path.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re

from contract import (ROUTE, SOURCE_COUNTS, TEXT_MODEL_ARCHITECTURE,
                      TEXT_MODEL_CONTRACT, TEXT_MODEL_SCHEDULE, digest, geometry)


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def file_signature(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def validate_source_contract(contract):
    if not isinstance(contract, dict):
        raise ValueError("audio initialization requires a source contract object")
    expected = {
        "route": ROUTE, "text_model_contract": TEXT_MODEL_CONTRACT,
        "text_model_architecture": TEXT_MODEL_ARCHITECTURE,
        "text_model_schedule": list(TEXT_MODEL_SCHEDULE),
        "geometry": geometry(sum(SOURCE_COUNTS.values())), "source_rows": SOURCE_COUNTS,
        "prefix": [126, 1, 256], "answer_tokens": 300, "sample_rate": 32000,
        "attention_mask": None, "forward_dtype": "float32", "autocast": False,
        "pad_token": "!", "eos_suffix": " <|endoftext|>",
        "audio_policy": "mono_resample_center_crop_right_pad_fixed20_split10",
    }
    for key, value in expected.items():
        if key not in contract or contract[key] != value:
            raise ValueError(f"audio initialization source contract differs: {key}")
    sections = {
        "audio_encoder": {"name": "CEDSmall", "frozen": True, "mode": "eval", "hidden_cache": False},
        "mapper": {"name": "freq_merge_126", "resample": False, "dropout": .05,
                   "hidden_dim": 384, "frequency_bins": 4, "output_dim": 576,
                   "external_projection": "Identity"},
    }
    for section, fields in sections.items():
        if not isinstance(contract.get(section), dict) or any(
                contract[section].get(key) != value for key, value in fields.items()):
            raise ValueError(f"audio initialization source contract differs: {section}")
    # A strict tensor inventory below also rejects every additional branch.
    for key in ("layer_injection", "audio_listen", "audio_read2", "audio_fusion", "loop_lora",
                "audio_cross_attention", "audio_reinjection", "text_memory"):
        if key in contract:
            raise ValueError(f"audio initialization cannot import a branched model: {key}")


def inspect_audio_source(checkpoint, *, fingerprint=False):
    """Only JSON/stat/hash on CPU; never constructs a model or unpickles tensors."""
    path = Path(checkpoint).expanduser().resolve(strict=True)
    match = re.fullmatch(r"model--step-([1-9][0-9]*)\.ckpt", path.name)
    if not path.is_file() or not match or path.parent.name != "checkpoints":
        raise ValueError("audio initialization expects <run>/formal/checkpoints/model--step-N.ckpt")
    step = int(match[1])
    if not 0 < step <= 34626:
        raise ValueError("audio initialization source step outside this three-epoch horizon")
    runtime_path = path.parent.parent / "runtime.json"
    before = file_signature(path)
    runtime_before = file_signature(runtime_path)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("route") != ROUTE or runtime.get("mode") != "formal":
        raise ValueError("audio initialization requires this route's formal runtime.json")
    contract = runtime.get("contract")
    validate_source_contract(contract)
    if runtime.get("contract_sha256") != digest(contract):
        raise ValueError("source runtime contract SHA256 mismatch")
    result = {
        "mode": "audio_checkpoint_weights_only", "checkpoint": str(path),
        "runtime_config": str(runtime_path), "source_checkpoint_step": step,
        "source_contract_sha256": runtime["contract_sha256"],
        "source_checkpoint_signature": before,
        "source_runtime_sha256": file_sha256(runtime_path),
        "start_optimizer_step": 0, "restore_optimizer_scheduler_rng_cursor": False,
        "loaded_components": ["frozen_ced", "trained_projector", "full_lm", "trained_routers"],
    }
    if fingerprint:
        result["source_checkpoint_sha256"] = file_sha256(path)
    if before != file_signature(path) or runtime_before != file_signature(runtime_path):
        raise ValueError("audio initialization source changed during inspection")
    return result


def validate_saved_header(saved, source):
    """Require the serialized payload to match its own historical runtime."""
    if not isinstance(saved, Mapping):
        raise ValueError("audio initialization checkpoint must be a mapping")
    expected = {"schema_version": 1, "route_variant": ROUTE, "mode": "formal",
                "total_step": source["source_checkpoint_step"], "num_epochs": 3,
                "contract_sha256": source["source_contract_sha256"],
                "loss_reduction": "mean_of_rank_local_microbatch_token_means_then_DDP_mean",
                "batch_geometry": {"world_size": 8, "microbatch_size": 4,
                                   "gradient_accumulation_steps": 4, "global_batch_size": 128}}
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"audio initialization checkpoint header mismatch: {key}")
    contract = saved.get("contract")
    validate_source_contract(contract)
    if digest(contract) != source["source_contract_sha256"]:
        raise ValueError("audio initialization checkpoint/runtime identity mismatch")
    if not isinstance(saved.get("state_dict"), Mapping) or not saved["state_dict"]:
        raise ValueError("audio initialization lacks the complete model state_dict")
    # No source optimizer/scaler/scheduler/RNG/cursor is required or restored.


def inherit_model_state(model, saved, source):
    import torch
    validate_saved_header(saved, source)
    state, target = saved["state_dict"], model.state_dict()
    if state.keys() != target.keys():
        raise ValueError(f"audio initialization tensor inventory mismatch: missing={sorted(target.keys()-state.keys())}, "
                         f"extra={sorted(state.keys()-target.keys())}")
    mapper, routers = [], []
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != target[name].shape
                or value.dtype != target[name].dtype or not bool(torch.isfinite(value).all())):
            raise ValueError(f"audio initialization tensor shape/dtype/finite mismatch: {name}")
        if name.startswith("audio_encoder.base.mapper."):
            mapper.append(name)
        if ".write_routers." in name or ".read_routers." in name:
            routers.append(name)
        if name.startswith("audio_encoder.base.ced.") and not torch.equal(value, target[name]):
            raise ValueError(f"source frozen CED differs from the configured pretrained assets: {name}")
    embed, head = "caption_decoder.lm.model.embed_tokens.weight", "caption_decoder.lm.lm_head.weight"
    if not torch.equal(state[embed], state[head]):
        raise ValueError("audio initialization has contradictory tied embedding/head tensors")
    if not mapper or len(routers) != 12:
        raise ValueError("audio initialization must include a trained projector and all twelve router tensors")
    model.load_state_dict(state, strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value, state[name]):
            raise ValueError(f"audio initialization inheritance check failed: {name}")
    if model.caption_decoder.lm.get_input_embeddings().weight is not model.caption_decoder.lm.lm_head.weight:
        raise ValueError("embedding/head Parameter binding was lost")
    return {**source, "all_model_tensors_equal": True, "model_tensors": len(state),
            "projector_tensors": len(mapper), "trained_router_tensors": len(routers),
            "restored_training_state": False}


def initialize_audio_weights(model, source):
    """mmap CPU storages; access model tensors only, before creating optimizer."""
    import torch
    observed = inspect_audio_source(source["checkpoint"])
    if any(observed[key] != value for key, value in source.items() if key in observed):
        raise ValueError("audio initialization source changed after resolved config")
    saved = torch.load(source["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    report = inherit_model_state(model, saved, source)
    del saved
    if observed != inspect_audio_source(source["checkpoint"]):
        raise ValueError("audio initialization source changed while loading weights")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", required=True)
    args = parser.parse_args()
    print(json.dumps({"event": "AUDIO_INIT_METADATA_PASS", "source": inspect_audio_source(args.init_checkpoint)},
                     ensure_ascii=False), flush=True)
