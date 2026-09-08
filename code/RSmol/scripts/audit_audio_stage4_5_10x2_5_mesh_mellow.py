#!/usr/bin/env python3
"""Single-GPU forward/backward audit for the isolated MeSH audio route."""
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
from audio_5_10x2_5_mesh_mellow.model import AudioMeshConfig, AudioMeshModel, _load_mellow_wrapper  # noqa: E402
from recursive_model_5_10x2_5_mesh import RecursiveLlamaForCausalLM, parameter_audit, register_auto_class  # noqa: E402


def _load(args: argparse.Namespace, device: torch.device) -> tuple[AudioMeshModel, Any]:
    register_auto_class()
    from transformers import AutoTokenizer
    mesh = RecursiveLlamaForCausalLM.from_pretrained(args.mesh_checkpoint, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.mesh_checkpoint, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper, htsat, provenance = _load_mellow_wrapper(args.mellow_root, args.htsat_checkpoint, device)
    model = AudioMeshModel(mesh.to(device), tokenizer, wrapper, htsat, AudioMeshConfig())
    model._audio_provenance = provenance
    model.to(device)
    return model, tokenizer


def run(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {"stage": "stage4_audio_5_10x2_5_mesh_mellow", "status": "FAIL", "cuda_required": True, "configuration": vars(args), "checks": [], "warnings": [], "hard_failures": []}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Stage 4 requires CUDA; run on a GPU node")
        device = torch.device("cuda", 0)
        model, tokenizer = _load(args, device)
        model.train()
        dataset = ReasonAQADataset(args.manifest, tokenizer)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=lambda rows: collate_reasonaqa(rows, tokenizer))
        batch = next(iter(loader))
        moved = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
        output = model(**{key: value for key, value in moved.items() if key not in {"row_indices", "audio2_reused"}})
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("nonfinite or missing multimodal loss")
        labels = model.last_labels
        prefix_length = int(model.last_prefix_length or 0)
        if labels is None or prefix_length <= 0:
            raise RuntimeError("model did not expose labels/prefix length for loss audit")
        text_ids = moved["text_ids"]
        prompt_lengths = moved["prompt_lengths"]
        answer_lengths = moved["answer_lengths"]
        prefix_labels = labels[:, :prefix_length]
        if bool((prefix_labels != -100).any()):
            raise RuntimeError("answer-only audit failed: multimodal prefix labels are supervised")
        expected_answer_tokens = int(answer_lengths.sum().item())
        actual_answer_tokens = 0
        for row_index in range(text_ids.shape[0]):
            prompt_length = int(prompt_lengths[row_index].item())
            answer_length = int(answer_lengths[row_index].item())
            answer_start = prefix_length + prompt_length
            answer_end = answer_start + answer_length
            if bool((labels[row_index, :answer_start] != -100).any()) or bool((labels[row_index, answer_end:] != -100).any()):
                raise RuntimeError(f"answer-only audit failed: non-answer label at row {row_index}")
            expected = text_ids[row_index, prompt_length:prompt_length + answer_length]
            if not torch.equal(labels[row_index, answer_start:answer_end], expected):
                raise RuntimeError(f"answer-only audit failed: answer token mismatch at row {row_index}")
            actual_answer_tokens += answer_length
        if actual_answer_tokens != expected_answer_tokens:
            raise RuntimeError(f"answer-only audit failed: actual={actual_answer_tokens} expected={expected_answer_tokens}")
        loss = output.loss
        loss.backward()
        audit = model.trainable_parameter_audit()
        if not audit["htsat_frozen"] or not audit["bridge_trainable"] or not audit["mesh_trainable"]:
            raise RuntimeError(f"trainability contract failed: {audit}")
        if not any(p.grad is not None and torch.isfinite(p.grad).all() for n, p in model.named_parameters() if n.startswith("bridge.")):
            raise RuntimeError("bridge received no finite gradient")
        c2l = getattr(model.htsat_wrapper, "c2l", None)
        c2l_grad = c2l is not None and any(p.grad is not None and torch.isfinite(p.grad).all() for p in c2l.parameters())
        if not c2l_grad:
            raise RuntimeError("c2l received no finite gradient")
        report.update({"status": "PASS", "device": str(device), "loss": float(loss.detach().cpu()), "logits_shape": list(output.logits.shape), "parameter_audit": parameter_audit(model.mesh_model), "trainable_audit": audit, "c2l_gradient": True, "audio2_reused": bool(moved.get("audio2") is None), "label_audit": {"unified_text_padding": True, "prefix_length": prefix_length, "prefix_non_ignore_count": int((prefix_labels != -100).sum().item()), "answer_supervised_tokens": actual_answer_tokens, "expected_answer_tokens": expected_answer_tokens}, "checks": [{"name": "cuda_available", "passed": True}, {"name": "forward_loss_finite", "passed": True}, {"name": "unified_text_padding", "passed": True, "detail": "prompt and answer were concatenated before batch padding"}, {"name": "answer_only_labels", "passed": True, "detail": "verified exact answer intervals from model.last_labels"}, {"name": "backward_gradients", "passed": True}]})
    except Exception as exc:
        report["hard_failures"].append({"name": "stage4_exception", "detail": repr(exc), "traceback": traceback.format_exc()})
    report["summary"] = {"checks": len(report["checks"]), "warnings": len(report["warnings"]), "hard_failures": len(report["hard_failures"])}
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-checkpoint", "--model-path", required=True, type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--htsat-checkpoint", required=True, type=Path)
    parser.add_argument("--mellow-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--report-path", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps({"stage": report["stage"], "status": report["status"], "summary": report["summary"], "report": str(args.report_path)}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
