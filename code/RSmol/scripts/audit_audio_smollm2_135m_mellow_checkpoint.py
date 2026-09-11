#!/usr/bin/env python3
"""Standalone one-GPU reload and forward/backward audit for baseline checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_smollm2_135m_mellow.data import ReasonAQADataset, collate_reasonaqa  # noqa: E402
from audio_smollm2_135m_mellow.model import (  # noqa: E402
    AUDIO_PREFIX_TOKENS,
    AUDIO_TOKENS_PER_CLIP,
    MAPPER_CONTRACT,
    ORIGINAL_SMOLLM2_CONTRACT,
    AudioSmolLM2Config,
    AudioSmolLM2Model,
    _load_mellow_wrapper,
    validate_original_smollm2,
)
from train_audio_smollm2_135m_mellow_ddp import (  # noqa: E402
    DEFAULT_HTSAT,
    DEFAULT_MELLOW,
    DEFAULT_MODEL,
    DEFAULT_TRAIN_MANIFEST,
    DEFAULT_VAL_MANIFEST,
    _audit_saved_checkpoint,
    _check_answer_labels,
    _load_training_state,
    _make_scheduler,
    _validate_parameter_change_audit,
    _validate_optimizer_coverage,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-gate", choices=("STAGE7", "FORMAL"), default="STAGE7")
    parser.add_argument("--expected-step", type=int, default=22)
    parser.add_argument(
        "--expected-parent-step",
        type=int,
        help="expected parent global step; defaults to 20 for STAGE7 and is inferred from FORMAL lineage",
    )
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--manifest", type=Path, default=Path(DEFAULT_TRAIN_MANIFEST))
    parser.add_argument("--val-manifest", type=Path, default=Path(DEFAULT_VAL_MANIFEST))
    parser.add_argument("--htsat-checkpoint", type=Path, default=Path(DEFAULT_HTSAT))
    parser.add_argument("--mellow-root", type=Path, default=Path(DEFAULT_MELLOW))
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-path", type=Path)
    return parser.parse_args(argv)


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device), "python": random.getstate()}


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"], device=device)
    if state.get("python") is not None:
        random.setstate(state["python"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_PARENT_CONTRACT_PATH_KEYS = {
    "text_model_source_path",
    "train_manifest",
    "val_manifest",
    "htsat_checkpoint",
    "mellow_root",
}
_PARENT_CONTRACT_KEYS = (
    "gate", "run_kind", "architecture_contract", "mapper_contract",
    "text_model_source_path", "text_model_runtime_class", "text_model_config_sha256", "text_model_config",
    "text_model_type", "text_model_hidden_size", "text_model_num_hidden_layers", "embedding_lm_head_tied",
    "audio_tokens_per_clip", "audio_prefix_tokens_with_separators", "sample_rate", "audio_seconds",
    "max_prompt_tokens", "max_answer_tokens", "max_context_length",
    "train_manifest", "val_manifest", "manifest_sha256", "manifest_hashes",
    "htsat_checkpoint", "mellow_root", "mellow_provenance",
    "world_size", "micro_batch_size", "gradient_accumulation_steps", "effective_global_batch_size",
    "epochs", "max_lr", "min_lr", "scheduler", "warmup_steps", "total_steps", "schedule_total_steps",
    "seed", "num_workers", "optimizer", "optimizer_betas",
    "weight_decay", "gradient_clip_norm", "autocast_dtype", "ddp_broadcast_buffers",
    "ddp_find_unused_parameters", "save_every", "checkpoint_retention", "trainable_parameter_names",
    "frozen_audio_encoder", "periodic_validation",
)


_VOLATILE_TEXT_CONFIG_KEYS = {"_name_or_path", "_commit_hash", "transformers_version"}


def _canonical_contract_value(key: str, value: Any) -> Any:
    """Normalize loader metadata that legitimately changes on checkpoint reload."""
    if key == "text_model_config" and isinstance(value, dict):
        return {name: item for name, item in value.items() if name not in _VOLATILE_TEXT_CONFIG_KEYS}
    return value


def _assert_parent_contract(child_config: dict[str, Any], parent_config: dict[str, Any]) -> None:
    """Require an exact immutable training contract across a resume edge."""
    contract_keys = list(_PARENT_CONTRACT_KEYS)
    if child_config.get("gate") == "FORMAL":
        contract_keys.extend(("run_target_step", "execution_max_steps"))
    missing = [key for key in contract_keys if key not in child_config or key not in parent_config]
    if missing:
        raise RuntimeError(f"parent/child checkpoint contract is missing fields: {missing}")
    for key in contract_keys:
        child_value = child_config[key]
        parent_value = parent_config[key]
        if key in _PARENT_CONTRACT_PATH_KEYS:
            child_value = str(Path(str(child_value)).resolve())
            parent_value = str(Path(str(parent_value)).resolve())
        child_value = _canonical_contract_value(key, child_value)
        parent_value = _canonical_contract_value(key, parent_value)
        if child_value != parent_value:
            raise RuntimeError(
                f"parent/child immutable contract mismatch for {key}: "
                f"child={child_config[key]!r} parent={parent_config[key]!r}"
            )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "stage": "standalone_audio_smollm2_135m_mellow_checkpoint_audit",
        "status": "FAIL",
        "configuration": vars(args),
        "checks": [],
        "hard_failures": [],
    }
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint audit requires CUDA; submit it through the 5090 audit wrapper")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    artifact = _audit_saved_checkpoint(args.checkpoint)
    config = json.loads((args.checkpoint / "audio_smollm2_config.json").read_text(encoding="utf-8"))
    if int(artifact["global_step"]) != int(args.expected_step):
        raise RuntimeError(f"checkpoint global step {artifact['global_step']} != expected {args.expected_step}")
    expected_run_kind = "formal" if args.expected_gate == "FORMAL" else "stage7"
    if config.get("gate") != args.expected_gate or config.get("run_kind") != expected_run_kind:
        raise RuntimeError(
            f"checkpoint gate/run_kind mismatch: saved=({config.get('gate')!r}, {config.get('run_kind')!r}) "
            f"expected=({args.expected_gate!r}, {expected_run_kind!r})"
        )
    expected_parent_step = args.expected_parent_step
    if args.expected_gate == "STAGE7":
        expected_parent_step = 20 if expected_parent_step is None else int(expected_parent_step)
        if config.get("resume_from") is None:
            raise RuntimeError("standalone STAGE7 step22 audit requires a resumed checkpoint with parent lineage")
        if int(config.get("resume_start_step", -1)) != expected_parent_step:
            raise RuntimeError(f"checkpoint resume_start_step={config.get('resume_start_step')} != expected parent step {expected_parent_step}")
        if int(config.get("parent_checkpoint_global_step", -1)) != expected_parent_step:
            raise RuntimeError("checkpoint parent_checkpoint_global_step is inconsistent")
        if int(config.get("run_target_step", -1)) != int(args.expected_step):
            raise RuntimeError("checkpoint run_target_step does not match the audited STAGE7 execution target")
    else:
        if int(config.get("run_target_step", -1)) < int(args.expected_step):
            raise RuntimeError("checkpoint run_target_step is below the audited FORMAL checkpoint step")
        if int(config.get("execution_max_steps", -1)) != int(config.get("run_target_step", -2)):
            raise RuntimeError("FORMAL checkpoint execution_max_steps disagrees with run_target_step")
        if config.get("resume_from") is None:
            if int(config.get("resume_start_step", -1)) != 0 or config.get("parent_checkpoint_global_step") is not None:
                raise RuntimeError("fresh FORMAL checkpoint has invalid resume lineage")
        else:
            if not Path(str(config["resume_from"])).is_absolute():
                raise RuntimeError("FORMAL resume_from lineage is not absolute")
            if expected_parent_step is None:
                expected_parent_step = int(config.get("resume_start_step", -1))
            if int(config.get("resume_start_step", -1)) != int(expected_parent_step):
                raise RuntimeError("FORMAL checkpoint resume_start_step differs from the requested parent step")
            if int(config.get("parent_checkpoint_global_step", -1)) != int(expected_parent_step):
                raise RuntimeError("FORMAL checkpoint parent_checkpoint_global_step is inconsistent")
    parent_path = None
    parent_artifact = None
    parent_contract_verified = False
    if config.get("resume_from") is None and args.parent_checkpoint is not None:
        raise RuntimeError("--parent-checkpoint was supplied for a fresh checkpoint with no resume lineage")
    if config.get("resume_from") is not None:
        if not Path(str(config["resume_from"])).is_absolute():
            raise RuntimeError("checkpoint resume_from lineage is not absolute")
        configured_parent = Path(str(config["resume_from"])).resolve()
        parent_path = args.parent_checkpoint.resolve() if args.parent_checkpoint is not None else configured_parent
        if configured_parent != parent_path:
            raise RuntimeError("checkpoint parent path differs from --parent-checkpoint")
        parent_artifact = _audit_saved_checkpoint(parent_path)
        if expected_parent_step is not None and int(parent_artifact["global_step"]) != int(expected_parent_step):
            raise RuntimeError(
                f"parent checkpoint global step {parent_artifact['global_step']} "
                f"!= expected {expected_parent_step}"
            )
        parent_config = json.loads((parent_path / "audio_smollm2_config.json").read_text(encoding="utf-8"))
        if parent_artifact.get("gate") != config.get("gate") or parent_artifact.get("run_kind") != config.get("run_kind"):
            raise RuntimeError(
                "parent/child gate/run_kind mismatch: "
                f"child=({config.get('gate')!r}, {config.get('run_kind')!r}) "
                f"parent=({parent_artifact.get('gate')!r}, {parent_artifact.get('run_kind')!r})"
            )
        _assert_parent_contract(config, parent_config)
        parent_contract_verified = True
    parameter_change_audit = None
    if config.get("resume_from") is not None:
        parameter_change_audit = _validate_parameter_change_audit(config.get("parameter_change_audit"))
    if config["architecture_contract"] != ORIGINAL_SMOLLM2_CONTRACT or config["mapper_contract"] != MAPPER_CONTRACT:
        raise RuntimeError("checkpoint architecture contract is not the original baseline contract")
    saved_source = Path(str(config["text_model_source_path"])).resolve()
    current_source = args.model_path.resolve()
    if saved_source != current_source:
        raise RuntimeError(f"saved text model source differs from --model-path: saved={saved_source} current={current_source}")
    source_config_path = current_source / "config.json"
    saved_source_config_sha = config.get("text_model_config_sha256")
    if not saved_source_config_sha:
        raise RuntimeError("checkpoint has no text_model_config_sha256 for original SmolLM2 provenance")
    if not source_config_path.is_file() or _sha256(source_config_path) != saved_source_config_sha:
        raise RuntimeError("saved text_model_source_path/config SHA does not match the current original SmolLM2 model")
    if Path(str(config["train_manifest"])).resolve() != args.manifest.resolve():
        raise RuntimeError("saved train manifest path differs from --manifest")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")
    train_hash = _sha256(args.manifest)
    if train_hash != config["manifest_sha256"] or train_hash != config.get("manifest_hashes", {}).get("train"):
        raise RuntimeError("audit train manifest SHA256 differs from the checkpoint provenance")
    saved_val_path = Path(str(config["val_manifest"])).resolve()
    if saved_val_path != args.val_manifest.resolve():
        raise RuntimeError("saved validation manifest path differs from --val-manifest")
    saved_val_hash = config.get("manifest_hashes", {}).get("val")
    if saved_val_hash is None:
        if args.val_manifest.is_file():
            raise RuntimeError("checkpoint has val=None despite a canonical validation manifest being present")
    else:
        if not args.val_manifest.is_file():
            raise FileNotFoundError(f"validation manifest not found: {args.val_manifest}")
        if _sha256(args.val_manifest) != saved_val_hash:
            raise RuntimeError("audit validation manifest SHA256 differs from the checkpoint provenance")
    if Path(str(config["htsat_checkpoint"])).resolve() != args.htsat_checkpoint.resolve():
        raise RuntimeError("HTSAT checkpoint path differs from the saved provenance")
    if Path(str(config["mellow_root"])).resolve() != args.mellow_root.resolve():
        raise RuntimeError("Mellow root differs from the saved provenance")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text_model = AutoModelForCausalLM.from_pretrained(args.checkpoint / "text_model", local_files_only=True)
    text_contract = validate_original_smollm2(text_model)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / "tokenizer", local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper, htsat, provenance = _load_mellow_wrapper(args.mellow_root, args.htsat_checkpoint, device)
    saved_provenance = config.get("mellow_provenance", {})
    saved_mellow_sha = saved_provenance.get("mellow_htsat_sha256")
    if not saved_mellow_sha or provenance.get("mellow_htsat_sha256") != saved_mellow_sha:
        raise RuntimeError("Mellow source provenance differs from the checkpoint")
    model = AudioSmolLM2Model(text_model.to(device), tokenizer, wrapper, htsat, AudioSmolLM2Config())
    audio_state = torch.load(args.checkpoint / "audio_bridge.pt", map_location=device, weights_only=False)
    model.bridge.load_state_dict(audio_state["bridge"], strict=True)
    model.htsat_wrapper.c2l.load_state_dict(audio_state["c2l"], strict=True)
    model.train()
    trainable_audit = model.trainable_parameter_audit()
    if not trainable_audit["training_mode_contract"]:
        raise RuntimeError(f"checkpoint model training mode contract failed: {trainable_audit}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")
    manifest_hash = _sha256(args.manifest)
    if manifest_hash != config["manifest_sha256"] or manifest_hash != config.get("manifest_hashes", {}).get("train"):
        raise RuntimeError("audit manifest SHA256 differs from the checkpoint provenance")
    dataset = ReasonAQADataset(args.manifest, tokenizer)
    loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=False, drop_last=True, collate_fn=lambda rows: collate_reasonaqa(rows, tokenizer))
    batch = next(iter(loader))
    moved = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items() if key not in {"row_indices", "audio2_reused"}}
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["max_lr"]),
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scheduler = _make_scheduler(
        optimizer,
        max_lr=float(config["max_lr"]),
        min_lr=float(config["min_lr"]),
        warmup_steps=int(config["warmup_steps"]),
        total_steps=int(config["total_steps"]),
    )
    state = _load_training_state(args.checkpoint, optimizer, scheduler)
    _validate_optimizer_coverage(state, model)
    if int(state["global_step"]) != int(args.expected_step):
        raise RuntimeError("training_state global_step disagrees with expected checkpoint step")
    if not state["optimizer"].get("state") or not state["scheduler"]:
        raise RuntimeError("checkpoint optimizer/scheduler state is incomplete")
    saved_rng = state["rng_states_by_rank"].get("0")
    if saved_rng:
        _restore_rng(saved_rng, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**moved)
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("reloaded baseline forward produced a nonfinite loss")
    _check_answer_labels(model, moved)
    output.loss.backward()
    gradient_audit = model.runtime_gradient_audit()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
    if not optimizer.state_dict()["state"]:
        raise RuntimeError("checkpoint optimizer state is empty after a completed training step")
    report.update({
        "status": "PASS",
        "device": str(device),
        "artifact": artifact,
        "gate": config["gate"],
        "run_kind": config["run_kind"],
        "standard_text_contract": text_contract,
        "trainable_audit": trainable_audit,
        "gradient_audit": gradient_audit,
        "forward_backward": True,
        "answer_only_labels": True,
        "audio_tokens_per_clip": AUDIO_TOKENS_PER_CLIP,
        "audio_prefix_tokens_with_separators": AUDIO_PREFIX_TOKENS,
        "loss": float(output.loss.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
        "optimizer_state_entries": len(optimizer.state_dict()["state"]),
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "scheduler_state_loaded": True,
        "external_audio_provenance_verified": True,
        "resume_lineage": {
            "gate": config["gate"],
            "run_kind": config["run_kind"],
            "resume_from": config.get("resume_from"),
            "resume_start_step": int(config["resume_start_step"]),
            "parent_checkpoint_global_step": config.get("parent_checkpoint_global_step"),
            "run_target_step": int(config["run_target_step"]),
            "expected_parent_step": expected_parent_step,
            "parent_checkpoint": str(parent_path) if parent_path is not None else None,
            "parent_artifact": parent_artifact,
            "parent_contract_verified": parent_contract_verified,
        },
        "parameter_change_audit": parameter_change_audit,
    })
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit(args)
    except Exception as exc:
        report = {
            "stage": "standalone_audio_smollm2_135m_mellow_checkpoint_audit",
            "status": "FAIL",
            "configuration": vars(args),
            "hard_failures": [{"error": repr(exc), "traceback": traceback.format_exc()}],
        }
    report_root = args.output_dir or args.checkpoint.parent
    report_path = args.report_path or report_root / f"{args.checkpoint.name}_standalone_baseline_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"stage": report["stage"], "status": report["status"], "report": str(report_path)}, default=str))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
