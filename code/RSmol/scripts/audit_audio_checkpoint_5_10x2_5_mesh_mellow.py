"""Standalone single-GPU integrity and reload audit for an audio checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_5_10x2_5_mesh_mellow.data import ReasonAQADataset, collate_reasonaqa  # noqa: E402
from train_audio_5_10x2_5_mesh_mellow_ddp import (  # noqa: E402
    _audit_saved_checkpoint,
    _load_model,
    _load_training_state,
    _make_scheduler,
)


DEFAULT_HTSAT = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt"
DEFAULT_MELLOW = "/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--resume-from", dest="checkpoint", required=True, type=Path)
    parser.add_argument("--htsat-checkpoint", required=True, type=Path, default=Path(DEFAULT_HTSAT))
    parser.add_argument("--mellow-root", required=True, type=Path, default=Path(DEFAULT_MELLOW))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_contract(path: Path) -> dict[str, Any]:
    required = (
        "mesh_model/config.json",
        "tokenizer/tokenizer_config.json",
        "audio_bridge.pt",
        "training_state.pt",
        "audio_mesh_config.json",
        "checkpoint_complete.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"checkpoint missing required files: {missing}")
    marker = _json(path / "checkpoint_complete.json")
    if marker.get("status") != "complete":
        raise RuntimeError(f"checkpoint completion marker is invalid: {marker}")
    config = _json(path / "audio_mesh_config.json")
    training = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    audio = torch.load(path / "audio_bridge.pt", map_location="cpu", weights_only=False)
    for key in ("optimizer", "scheduler", "global_step", "epoch", "batch_in_epoch", "rng_states_by_rank"):
        if key not in training:
            raise RuntimeError(f"training_state.pt missing {key}")
    if not training["rng_states_by_rank"]:
        raise RuntimeError("training_state.pt contains no RNG states")
    if not isinstance(audio.get("bridge"), dict) or not isinstance(audio.get("c2l"), dict):
        raise RuntimeError("audio_bridge.pt must contain bridge and c2l state dictionaries")
    step_values = {int(marker.get("global_step", -1)), int(config.get("global_step", -1)), int(training["global_step"])}
    if len(step_values) != 1:
        raise RuntimeError(f"checkpoint step mismatch: marker/config/training={step_values}")
    if int(marker["global_step"]) <= 0:
        raise RuntimeError("checkpoint global_step must be positive")
    if not training["optimizer"].get("state"):
        raise RuntimeError("optimizer state is empty; checkpoint cannot prove optimizer restoration")
    if not training["scheduler"]:
        raise RuntimeError("scheduler state is empty")
    return {
        "passed": True,
        "required_files": list(required),
        "global_step": int(training["global_step"]),
        "epoch": int(training["epoch"]),
        "batch_in_epoch": int(training["batch_in_epoch"]),
        "rng_ranks": sorted(str(key) for key in training["rng_states_by_rank"]),
        "optimizer_state_entries": len(training["optimizer"]["state"]),
        "scheduler_keys": sorted(training["scheduler"]),
        "architecture_contract": config.get("architecture_contract"),
    }


def _audit_labels(model: Any, batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    labels = model.last_labels
    prefix_length = int(model.last_prefix_length or 0)
    if labels is None or prefix_length <= 0:
        raise RuntimeError("model did not expose labels/prefix length")
    text_ids = batch["text_ids"].to(device)
    prompt_lengths = batch["prompt_lengths"].to(device)
    answer_lengths = batch["answer_lengths"].to(device)
    expected = int(answer_lengths.sum().item())
    actual = 0
    for row_index in range(text_ids.shape[0]):
        prompt_length = int(prompt_lengths[row_index].item())
        answer_length = int(answer_lengths[row_index].item())
        start = prefix_length + prompt_length
        end = start + answer_length
        if bool((labels[row_index, :start] != -100).any()) or bool((labels[row_index, end:] != -100).any()):
            raise RuntimeError(f"non-answer label detected at row {row_index}")
        expected_ids = text_ids[row_index, prompt_length:prompt_length + answer_length]
        if not torch.equal(labels[row_index, start:end], expected_ids):
            raise RuntimeError(f"answer labels are misaligned at row {row_index}")
        actual += answer_length
    if actual != expected:
        raise RuntimeError(f"answer label count mismatch: actual={actual} expected={expected}")
    return {"passed": True, "unified_text_padding": True, "prefix_length": prefix_length, "answer_supervised_tokens": actual}


def run(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "stage": "audio_checkpoint_integrity_5_10x2_5_mesh_mellow",
        "status": "FAIL",
        "checkpoint": str(args.checkpoint),
        "configuration": vars(args),
        "checks": [],
        "hard_failures": [],
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint reload audit requires CUDA")
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive")
        if not args.htsat_checkpoint.is_file():
            raise FileNotFoundError(f"HTSAT checkpoint not found: {args.htsat_checkpoint}")
        if not args.mellow_root.is_dir():
            raise FileNotFoundError(f"Mellow source root not found: {args.mellow_root}")
        saved_config = _json(args.checkpoint / "audio_mesh_config.json")
        saved_htsat = Path(str(saved_config.get("htsat_checkpoint", "")))
        if not saved_htsat or saved_htsat.resolve() != args.htsat_checkpoint.resolve():
            raise RuntimeError(f"external HTSAT checkpoint mismatch: saved={saved_htsat} requested={args.htsat_checkpoint}")
        trainer_artifact = _audit_saved_checkpoint(args.checkpoint)
        artifact = _artifact_contract(args.checkpoint)
        artifact["trainer_artifact_audit"] = trainer_artifact
        artifact["external_htsat_verified"] = str(args.htsat_checkpoint)
        artifact["mellow_root_verified"] = str(args.mellow_root)
        report["checks"].append({"name": "artifact_contract", **artifact})

        device = torch.device("cuda", 0)
        load_args = argparse.Namespace(
            resume_from=args.checkpoint,
            model_path=args.checkpoint / "mesh_model",
            tokenizer_path=None,
            htsat_checkpoint=args.htsat_checkpoint,
            mellow_root=args.mellow_root,
        )
        model, tokenizer = _load_model(load_args, device)
        model.train()
        dataset = ReasonAQADataset(args.manifest, tokenizer)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=lambda rows: collate_reasonaqa(rows, tokenizer))
        batch = next(iter(loader))
        moved = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
        optimizer_config = saved_config
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(optimizer_config["max_lr"]), betas=(0.9, 0.95), weight_decay=0.1)
        scheduler = _make_scheduler(optimizer, max_lr=float(optimizer_config["max_lr"]), min_lr=float(optimizer_config.get("min_lr", 0.0)), warmup_steps=int(optimizer_config["warmup_steps"]), total_steps=max(1, int(optimizer_config["total_steps"])))
        state = _load_training_state(args.checkpoint, optimizer, scheduler)
        if int(state["global_step"]) != int(artifact["global_step"]):
            raise RuntimeError("reloaded training state global_step mismatch")
        report["checks"].append({"name": "model_optimizer_scheduler_reload", "passed": True, "global_step": int(state["global_step"]), "optimizer_state_loaded": bool(optimizer.state_dict()["state"]), "scheduler_last_epoch": int(scheduler.last_epoch)})

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**{key: value for key, value in moved.items() if key not in {"row_indices", "audio2_reused"}})
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("reloaded checkpoint produced a nonfinite or missing loss")
        label_audit = _audit_labels(model, moved, device)
        output.loss.backward()
        trainable = model.trainable_parameter_audit()
        if not trainable["htsat_frozen"] or not trainable["bridge_trainable"] or not trainable["mesh_trainable"]:
            raise RuntimeError(f"trainability contract failed: {trainable}")
        bridge_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for name, p in model.named_parameters() if name.startswith("bridge."))
        c2l = getattr(model.htsat_wrapper, "c2l", None)
        c2l_grad = c2l is not None and any(p.grad is not None and torch.isfinite(p.grad).all() for p in c2l.parameters())
        if not bridge_grad or not c2l_grad:
            raise RuntimeError(f"audio trainable gradients missing: bridge={bridge_grad} c2l={c2l_grad}")
        report.update({"status": "PASS", "device": str(device), "loss": float(output.loss.detach().cpu()), "logits_shape": list(output.logits.shape), "label_audit": label_audit, "trainable_audit": trainable, "gradient_audit": {"bridge_finite": bridge_grad, "c2l_finite": c2l_grad}, "checks": report["checks"] + [{"name": "forward_backward", "passed": True}, {"name": "answer_only_unified_labels", **label_audit}, {"name": "trainable_gradients", "passed": True}]})
    except Exception as exc:
        report["hard_failures"].append({"name": "checkpoint_audit_exception", "detail": repr(exc), "traceback": traceback.format_exc()})
    report["summary"] = {"checks": len(report["checks"]), "hard_failures": len(report["hard_failures"])}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_path or args.output_dir / "checkpoint_integrity_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run(args)
    print(json.dumps({"stage": report["stage"], "status": report["status"], "summary": report["summary"], "report": str(args.report_path or args.output_dir / "checkpoint_integrity_audit.json")}, default=str))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
