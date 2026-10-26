"""Isolated HTSAT Mizar S1 GBS256 ablation: fixed20 online with gradient accumulation.

The active loss/optimizer/LambdaLR/GradNorm sequence is ported from Mizar S1.
An independent entry keeps the historical ReasonAQA routes unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import time
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from contract import (ACCUMULATION_STEPS, ANSWER_TOKENS, BATCH_SIZE, EPOCHS,
                      PREFIX_TOKENS, RANK_UPDATE_SAMPLES, ROUTE, SEED, WORLD_SIZE,
                      TEXT_MODEL_CONTRACT, TEXT_MODEL_SCHEDULE, code_digest,
                      digest, lr_factor, rank_seed, recursive_model_source_sha256,
                      validate_checkpoint_tokenizer, validate_tokenizer_contract,
                      write_json)
from data.audiotext_dataset import AudioTextDataset, collate_fn
from data.sampler import CustomDistributedSampler
from data.staging import StagingGate
from models.mellow import Mellow
from optim import GradNormTracker, build_optimizer, configure_trainability
from initialization import load_trained_initialization
from audits import (ExecutionAudit, assert_state_exact, changed_parameters, frozen_digest,
                    gradient_audit, module_mode_audit, trainability_audit,
                    trainable_snapshot)


def collective_require(condition, message, device):
    flag = torch.tensor(int(bool(condition)), device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise RuntimeError(message)


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def move_batch(batch, device):
    return {"audio1": batch["audio1"].to(device), "audio2": batch["audio2"].to(device),
            "input": {k: v.to(device) for k, v in batch["input"].items()},
            "answer": {k: v.to(device) for k, v in batch["answer"].items()}}


def answer_loss(output, inputs, ignore_index):
    logits = output.logits
    prefix_len = logits.shape[1] - inputs["answer"]["input_ids"].shape[1]
    if prefix_len != PREFIX_TOKENS or logits.shape[1] != PREFIX_TOKENS + ANSWER_TOKENS:
        raise ValueError(f"unexpected prefix/total sequence: {prefix_len}, {logits.shape[1]}")
    # Match Mizar's _answer_token_loss: the tokenizer attention mask and the
    # PAD target check jointly define valid answer targets. Prompt/audio
    # contribute context only; each rank averages its own valid answer tokens.
    logits = logits[:, prefix_len - 1:-1]
    answer = inputs["answer"]
    targets = answer["input_ids"]
    attention_mask = answer.get("attention_mask")
    if attention_mask is None or attention_mask.shape != targets.shape:
        raise ValueError("Mizar answer padding requires an attention_mask matching input_ids")
    valid = attention_mask.to(dtype=torch.bool) & targets.ne(ignore_index)
    token_loss = F.cross_entropy(
        logits.transpose(1, 2), targets, reduction="none", ignore_index=ignore_index
    )
    valid_count = valid.sum()
    if valid_count.item() == 0:
        raise ValueError("answer batch contains no valid target tokens")
    return (token_loss * valid).sum() / valid_count


def inspect_tokenizer(dataset, model):
    tokenizer = dataset.tokenizer
    audit = validate_tokenizer_contract(
        tokenizer, dataset.pad_id, model.caption_decoder.lm.get_input_embeddings().num_embeddings)
    if (not getattr(model.caption_decoder, "is_recursive_5_10_5", False)
            or getattr(model.caption_decoder, "text_model_contract", "") != TEXT_MODEL_CONTRACT):
        raise ValueError("decoder is not the required non-MeSH recursive 5-10-5 text model")
    if model.caption_decoder.lm_embedding_size != 576:
        raise ValueError("recursive 5-10-5 hidden size must remain 576")
    config = model.caption_decoder.lm.config
    eos_id = getattr(config, "eos_token_id", None)
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if len(eos_id) == 1 else None
    if eos_id != 0:
        raise ValueError(f"recursive 5-10-5 model eos_token_id must be 0, got {eos_id!r}")
    vocab_size = int(getattr(config, "vocab_size", -1))
    if vocab_size != audit["embedding_rows"]:
        raise ValueError(
            f"recursive 5-10-5 vocab_size={vocab_size} differs from embedding rows={audit['embedding_rows']}"
        )
    schedule = getattr(config, "logical_to_physical", None)
    if schedule is None:
        schedule = getattr(config, "logical_to_physical_schedule", ())
    if tuple(schedule) != tuple(TEXT_MODEL_SCHEDULE):
        raise ValueError("recursive 5-10-5 logical-to-physical schedule mismatch")
    short = dataset.tokenize("hello", 256)
    long = dataset.tokenize("hello " * 2000, 256)
    return {**audit, "ignore_index": dataset.pad_id,
            "eos_suffix_token_ids": tokenizer.encode(" <|endoftext|>", add_special_tokens=False),
            "short_prompt_attention_tokens": int(short["attention_mask"].sum()),
            "short_prompt_pad_tokens": int(short["attention_mask"].eq(0).sum()),
            "long_prompt_length": int(long["input_ids"].shape[1]),
            "long_prompt_last_id": int(long["input_ids"][0, -1]),
            "padding_sent_to_attention": "all prompt/answer PADs remain visible; LM attention_mask=None",
            "eos_policy": "append before encode/truncate, no forced EOS reinsertion"}


def audit_checkpoint_tokenizer(text_model_dir, model_vocab_size):
    """Audit source metadata before assigning Mizar's ``!`` PAD token.

    A recursive checkpoint may serialize EOS (``<|endoftext|>``) as its
    tokenizer-level PAD.  That source setting is not the training contract:
    ``AudioTextDataset`` deliberately assigns the existing ``!`` vocabulary
    token as PAD without resizing the model.  Keep the source PAD metadata in
    the audit, but validate the invariants that must survive that override.
    """
    tokenizer = AutoTokenizer.from_pretrained(text_model_dir, local_files_only=True)
    return validate_checkpoint_tokenizer(tokenizer, model_vocab_size)


def probe_sources(model, dataset, prepared, device):
    saved = rng_state(device)
    result = []
    model.eval()
    try:
        with torch.no_grad():
            for source, indices in prepared["probe_rows"].items():
                # Decode through the training loader, then run the real
                # HTSAT -> mapper -> LM -> CE on GPU; staging only copies bytes.
                batch = collate_fn([dataset.original_item(indices[0])])
                inputs = move_batch(batch, device)
                output = model(inputs)
                loss = answer_loss(output, inputs, dataset.pad_id)
                if not torch.isfinite(loss) or not torch.isfinite(output.logits).all():
                    raise ValueError(f"nonfinite source probe: {source}")
                result.append({"source": source, "row_index": indices[0],
                               "loss": loss.item(), "layout": dict(model.caption_decoder.last_layout),
                               "answer_nonpad_tokens": int(inputs["answer"]["input_ids"].ne(dataset.pad_id).sum())})
                del inputs, output, loss, batch
    finally:
        model.train()
        restore_rng(saved, device)
    return result


def save_checkpoint(path, model, optimizer, scheduler, scaler, tracker, config, step, epoch, next_optimizer_step, device):
    states = [None] * WORLD_SIZE
    dist.all_gather_object(states, rng_state(device))
    if dist.get_rank() == 0:
        payload = {"schema_version": 1, "route_variant": ROUTE, "mode": config["mode"],
                   "contract": config["contract"], "contract_sha256": config["contract_sha256"],
                   "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "grad_scaler": scaler.state_dict(),
                   "grad_norm_tracker": tracker.state_dict(), "random_state_by_rank": states,
                   "total_step": step, "epoch": epoch, "next_optimizer_step": next_optimizer_step,
                   "num_epochs": EPOCHS,
                   "loss_reduction": "mean_of_rank_local_microbatch_token_means_then_DDP_mean",
                   "batch_geometry": {"world_size": WORLD_SIZE, "microbatch_size": BATCH_SIZE,
                                      "gradient_accumulation_steps": ACCUMULATION_STEPS,
                                      "global_batch_size": WORLD_SIZE * RANK_UPDATE_SAMPLES}}
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
    dist.barrier()


def reload_audit(path, model, optimizer, scheduler, scaler, tracker, config, step, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    required = {"state_dict", "optimizer", "scheduler", "grad_scaler", "grad_norm_tracker", "random_state_by_rank",
                "epoch", "next_optimizer_step", "total_step", "contract_sha256", "mode", "schema_version", "route_variant"}
    if not required.issubset(saved) or saved["contract_sha256"] != config["contract_sha256"]:
        raise ValueError("checkpoint reload lost full state/identity")
    if saved["total_step"] != step or saved["scheduler"]["last_epoch"] != step or len(saved["random_state_by_rank"]) != WORLD_SIZE:
        raise ValueError("checkpoint reload step/RNG mismatch")
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor.detach().cpu(), saved["state_dict"][name]):
            raise ValueError(f"checkpoint tensor reload mismatch: {name}")
    if any(int(value["step"]) != step for value in saved["optimizer"]["state"].values()):
        raise ValueError("optimizer did not perform the required real updates")
    for name, state in (("optimizer", optimizer.state_dict()), ("scheduler", scheduler.state_dict()),
                        ("grad_scaler", scaler.state_dict()), ("grad_norm_tracker", tracker.state_dict())):
        assert_state_exact(state, saved[name], name)
    assert_state_exact(rng_state(device), saved["random_state_by_rank"][0], "rank0_rng")
    return {"status": "PASS", "path": str(path), "optimizer_steps": step,
            "model_tensors_equal": True, "optimizer_state_entries": len(saved["optimizer"]["state"]),
            "optimizer_state_exact": True, "scheduler_state_exact": True,
            "grad_scaler_state_exact": True, "grad_norm_tracker_state_exact": True,
            "rank0_rng_exact": True, "scheduler_step": saved["scheduler"]["last_epoch"],
            "rng_ranks": WORLD_SIZE}


def train(config, rank, device):
    run_dir = Path(config["run_dir"])
    contract = config["contract"]
    geometry = contract["geometry"]
    if contract["code_sha256"] != code_digest() or config["contract_sha256"] != digest(contract):
        raise ValueError("recipe/source changed after writing resolved config")
    if contract.get("recursive_model_source_sha256") != recursive_model_source_sha256():
        raise ValueError("shared recursive model source changed after writing resolved config")
    if contract.get("route") != ROUTE or contract.get("text_model_contract") != TEXT_MODEL_CONTRACT:
        raise ValueError("resolved recipe does not belong to the non-MeSH recursive Core SFT route")
    if (geometry["world_size"] != WORLD_SIZE or geometry["microbatch"] != BATCH_SIZE
            or geometry["gradient_accumulation_steps"] != ACCUMULATION_STEPS
            or geometry["global_batch_size"] != WORLD_SIZE * RANK_UPDATE_SAMPLES):
        raise ValueError("this route requires 8 GPUs x microbatch8 x accumulation4 (global batch 256)")
    prepared = json.loads(Path(config["prepared_report"]).read_text(encoding="utf-8"))
    if prepared["mode"] != config["mode"]:
        raise ValueError("missing/mismatched successful staging")
    staging = StagingGate(prepared, log=rank == 0)
    if staging.info:
        staging.wait_for_batch(staging.info["epoch"], staging.info["start_batch"])
    local_seed = rank_seed(rank)
    random.seed(local_seed)
    np.random.seed(local_seed & 0xFFFFFFFF)
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    # Forward is deliberately FP32: the reference trainer enables GradScaler
    # but never wraps its forward in autocast, despite its float16 config label.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_tokenizer_audit = audit_checkpoint_tokenizer(
        contract["text_model_dir"],
        contract["recursive_text_checkpoint"]["config"]["vocab_size"],
    )
    if rank == 0:
        print(json.dumps({"event": "CHECKPOINT_TOKENIZER_PASS",
                          "tokenizer": source_tokenizer_audit}), flush=True)
    dataset = AudioTextDataset(prepared["metadata"], prepared["stage_root"], contract["text_model_dir"],
                               read_policy=prepared["read_policy"])
    if len(dataset) != geometry["rows"]:
        raise ValueError("prepared row count differs from the resolved recipe")
    model = Mellow(audioenc_name="HTSAT", d_in=768, d_out=576,
                   text_decoder=contract["text_model_dir"], prefix_length=PREFIX_TOKENS,
                   freeze_text_decoder_weights=False, use_pretrained_audioencoder=True,
                   freeze_audio_encoder_weights=True, pretrained_audioencoder_path=contract["htsat_root"])
    initialization = load_trained_initialization(model, config["init_checkpoint"])
    freeze_audit = configure_trainability(model)
    model.float().to(device)
    if any(p.requires_grad for p in model.audio_encoder.base.parameters()):
        raise ValueError("the complete old audio branch, including c2l, must be frozen")
    if any(p.requires_grad for p in model.audio_encoder.projection.parameters()):
        raise ValueError("the old audio projection must be frozen")
    tokenizer_audit = inspect_tokenizer(dataset, model)
    tokenizer_audit["checkpoint_source"] = source_tokenizer_audit
    model.train()
    core = model.audio_encoder.base.htsat
    if any(m.training for m in core.modules()):
        raise ValueError("frozen HTSAT has active BN/dropout/augmentation")
    if model.audio_encoder.projection.training:
        raise ValueError("frozen projection must remain in eval mode")
    trainability = trainability_audit(model)
    optimizer, scheduler, groups = build_optimizer(model, geometry["total_steps"])
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    tracker = GradNormTracker()
    step, start_epoch, start_batch = 0, 0, 0
    resume_rng = None
    if config["resume_checkpoint"]:
        saved = torch.load(config["resume_checkpoint"], map_location="cpu", weights_only=False)
        if (saved.get("schema_version") != 1 or saved.get("route_variant") != ROUTE
                or saved.get("mode") != "formal" or saved.get("contract_sha256") != config["contract_sha256"]):
            raise ValueError("resume requires a full formal checkpoint from this identical route/code/data/recipe")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["grad_scaler"])
        tracker.load_state_dict(saved["grad_norm_tracker"])
        step, start_epoch, start_batch = saved["total_step"], saved["epoch"], saved["next_optimizer_step"]
        if (step != start_epoch * geometry["steps_per_epoch"] + start_batch
                or scheduler.last_epoch != step or not 0 <= start_batch < geometry["steps_per_epoch"]
                or not 0 <= start_epoch <= EPOCHS):
            raise ValueError("invalid resume sampler/scheduler cursor")
        resume_rng = saved["random_state_by_rank"][rank]
        del saved
    if staging.info and (staging.info["epoch"], staging.info["start_batch"]) != (start_epoch, start_batch):
        raise ValueError("background staging cursor differs from the actual training resume cursor")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    source_probes = probe_sources(model, dataset, prepared, device) if rank == 0 else []
    dist.barrier()
    # Attach after the deterministic probes so the first four real microbatches
    # prove that every scheduled physical layer, including both middle loops,
    # participates in forward and backward computation.
    execution_audit = ExecutionAudit(model) if step == 0 else None
    ddp = DistributedDataParallel(model, device_ids=[device.index], output_device=device.index,
                                 broadcast_buffers=False, find_unused_parameters=False)
    if resume_rng is not None:
        restore_rng(resume_rng, device)
    # DDP broadcasts rank-zero initialization. Measure updates against that
    # shared state, never against each rank's independently initialized model.
    before = trainable_snapshot(model)
    frozen_before = frozen_digest(model)
    audit = {"status": "PREFLIGHT_PASS", "resolved_config": config, "tokenizer": tokenizer_audit,
             "optimizer_groups": groups, "source_probes": source_probes,
             "initialization": initialization, "freeze_audit": freeze_audit,
             "trainability": trainability, "frozen_parameters_sha256_before": frozen_before,
             "model_parameters": sum(p.numel() for p in model.parameters()),
             "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
             "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
             "grad_scaler_enabled": scaler.is_enabled(), "autocast": False, "tf32": False,
             "htsat_all_eval": True, "world_size": dist.get_world_size(),
             "seed_by_rank": [rank_seed(r) for r in range(WORLD_SIZE)],
             "initial_optimizer_lrs": [g["lr"] for g in optimizer.param_groups],
             "resume_step": step, "cuda_device": torch.cuda.get_device_name(device),
             "packages": contract["packages"],
             "execution_audit": ({"skipped": True, "resume_step": step}
                                 if execution_audit is None else None)}
    if rank == 0:
        write_json(run_dir / "config_audit.json", audit)
        (run_dir / "trainable_params.txt").write_text("\n".join(
            f"{name}\t{p.numel()}\t{list(p.shape)}" for name, p in model.named_parameters() if p.requires_grad) + "\n", encoding="utf-8")
        print(json.dumps({"event": "PREFLIGHT_PASS", "geometry": geometry,
                          "layout": "audio[129+129] -> SEP[1] -> prompt[256] -> answer[300]",
                          "tokenizer": {key: tokenizer_audit[key] for key in
                                        ("pad_id", "sep_id", "eos_id", "ignore_index",
                                         "tokenizer_vocab_rows", "embedding_rows")},
                          "checkpoint_tokenizer": source_tokenizer_audit,
                          "optimizer_lrs": audit["initial_optimizer_lrs"]}), flush=True)
    sampler = CustomDistributedSampler(dataset, WORLD_SIZE, rank, shuffle=True, seed=SEED, drop_last=True)
    max_steps = config["max_optimizer_steps"]
    if config["mode"] != "formal" or max_steps != geometry["total_steps"]:
        raise ValueError("the baseline requires the complete two-epoch formal schedule")
    optimizer.zero_grad(set_to_none=True)
    first_rows, last_checkpoint = [], None
    gradient_checks = []
    local_switch_step = None
    start_time = time.monotonic()
    cursor_epoch, cursor_batch = start_epoch, start_batch
    metrics = (run_dir / "metrics.jsonl").open("a", encoding="utf-8") if rank == 0 else None
    for epoch in range(start_epoch, EPOCHS):
        if step >= max_steps:
            break
        sampler.set_epoch(epoch)
        skipped_batches = start_batch if epoch == start_epoch else 0
        sampler.start_index = skipped_batches * RANK_UPDATE_SAMPLES
        # Iterator creation has an isolated CPU generator so exact resume does
        # not consume model RNG.
        generator = torch.Generator().manual_seed(SEED + epoch)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True,
                            num_workers=0, persistent_workers=False, collate_fn=collate_fn, generator=generator)
        iterator = iter(loader)
        for update_index in range(skipped_batches, geometry["steps_per_epoch"]):
            if step >= max_steps:
                break
            # Check copy health without waiting for an audio watermark.
            staging.wait_for_batch(epoch, update_index)
            if not dataset.audio_paths.local:
                ready = torch.tensor(int(staging.complete), device=device)
                dist.all_reduce(ready, op=dist.ReduceOp.MIN)
                if ready.item():
                    dataset.audio_paths.use_local()
                    local_switch_step = step
                    if rank == 0:
                        print(json.dumps({"event": "AUDIO_READ_SWITCH", "completed_optimizer_steps": step,
                                          "next_optimizer_step": step + 1, "from": "shared", "to": "local"}), flush=True)
            lr_used = [float(g["lr"]) for g in optimizer.param_groups]
            expected_factor = lr_factor(step, geometry["total_steps"])
            collective_require(all(math.isclose(g["lr"], g["initial_lr"] * expected_factor, rel_tol=1e-12)
                                   for g in optimizer.param_groups), "optimizer LR/scheduler cursor mismatch", device)
            update_losses, update_rows, update_sources = [], [], []
            for micro_index in range(ACCUMULATION_STEPS):
                batch = next(iterator)
                if epoch == 0 and update_index < 8:
                    first_rows.extend(batch["row_index"])
                    offset = update_index * RANK_UPDATE_SAMPLES + micro_index * BATCH_SIZE
                    expected = prepared["expected_first8_rows"][str(rank)][offset:offset + BATCH_SIZE]
                    collective_require(batch["row_index"] == expected,
                                       "first-eight optimizer-update sampler rows differ from staging plan", device)
                update_rows.extend(batch["row_index"])
                update_sources.extend(batch["source"])
                inputs = move_batch(batch, device)
                sync_context = (ddp.no_sync() if micro_index + 1 < ACCUMULATION_STEPS else nullcontext())
                with sync_context:
                    output = ddp(inputs)
                    loss = answer_loss(output, inputs, dataset.pad_id)
                    collective_require(bool(torch.isfinite(loss).item()), "nonfinite answer-only CE", device)
                    update_losses.append(loss.detach())
                    # Each microbatch computes a rank-local valid-token mean.
                    # Average the four means to keep the accumulated gradient scale stable.
                    scaler.scale(loss / ACCUMULATION_STEPS).backward()
                del output, loss, inputs, batch
            scaler.unscale_(optimizer)
            if step == 0 and execution_audit is not None:
                audit_result = execution_audit.report_and_remove(ACCUMULATION_STEPS)
                audit["execution_audit"] = audit_result
            if step in {0, max_steps - 1}:
                gradient_checks.append({"step": step + 1, **gradient_audit(model)})
            total_norm, min_scale = tracker.track_and_clip_(model.named_parameters())
            collective_require(math.isfinite(total_norm) and total_norm < 1e6 and math.isfinite(min_scale) and min_scale > 0,
                               "nonfinite/emergency gradient norm; optimizer update aborted", device)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            collective_require(scaler.get_scale() >= scale_before, "GradScaler skipped an optimizer update", device)
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            cursor_epoch, cursor_batch = ((epoch + 1, 0) if update_index + 1 == geometry["steps_per_epoch"]
                                          else (epoch, update_index + 1))
            reduced = torch.stack(update_losses).mean()
            dist.all_reduce(reduced)
            if rank == 0:
                record = {"step": step, "epoch": epoch + 1, "optimizer_update": update_index + 1,
                          "rank_mean_loss": reduced.item() / WORLD_SIZE,
                          "lr_used": lr_used, "lr_next": [float(g["lr"]) for g in optimizer.param_groups],
                          "gradient_l2_before_clip": total_norm, "gradient_common_scale": min_scale,
                          "grad_scaler": scaler.get_scale(), "seconds": time.monotonic() - start_time,
                          "rank0_rows": update_rows, "rank0_sources": update_sources,
                          "gpu_peak_bytes": torch.cuda.max_memory_allocated(device)}
                record["staging_wait_seconds"] = staging.wait_seconds
                record["staging_complete"] = staging.complete
                record["audio_read_mode"] = "local" if dataset.audio_paths.local else "shared"
                record["audio_read_files_rank0"] = dict(dataset.audio_paths.read_files)
                record["local_switch_step"] = local_switch_step
                metrics.write(json.dumps(record) + "\n")
                if step % 25 == 0:
                    metrics.flush()
                    print(json.dumps(record), flush=True)
            del reduced, update_losses, update_rows, update_sources
            if step in config["checkpoint_steps"] or step == max_steps:
                last_checkpoint = checkpoint_dir / f"model--step-{step}.ckpt"
                save_checkpoint(last_checkpoint, model, optimizer, scheduler, scaler, tracker,
                                config, step, cursor_epoch, cursor_batch, device)
        start_batch = 0
    if metrics:
        metrics.close()
    staging.wait_complete()
    collective_require(step == max_steps, "training ended before the requested real optimizer updates", device)
    if execution_audit is not None and execution_audit.handles:
        raise ValueError("recursive execution audit was not completed on the first optimizer update")
    last_checkpoint = checkpoint_dir / "model--epo-2.ckpt"
    save_checkpoint(last_checkpoint, model, optimizer, scheduler, scaler, tracker,
                    config, step, cursor_epoch, cursor_batch, device)
    changed = changed_parameters(model, before)
    if not any(item["changed"] for item in changed.values() if item["role"] == "core"):
        raise ValueError("the trainable shared Core layers did not update")
    frozen_after = frozen_digest(model)
    collective_require(frozen_before == frozen_after, "a frozen parameter changed during training", device)
    mode_after = module_mode_audit(model)
    finite = all(bool(torch.isfinite(p).all().item()) for p in model.parameters())
    collective_require(finite, "nonfinite final parameters", device)
    rank_check = {"rank": rank, "seed": local_seed, "optimizer_steps": step, "first8_rows": first_rows,
                  "trainable_weight_changes": changed, "gradient_audits": gradient_checks,
                  "parameters_finite": finite,
                  "htsat_all_eval": all(not m.training for m in core.modules()),
                  "frozen_modules_all_eval": mode_after["frozen_modules_all_eval"]}
    rank_check["staging_wait_seconds"] = staging.wait_seconds
    rank_check["staging_complete"] = staging.complete
    rank_check["audio_read_files"] = dataset.audio_paths.read_files
    rank_check["local_switch_step"] = local_switch_step
    rank_checks = [None] * WORLD_SIZE
    dist.all_gather_object(rank_checks, rank_check)
    if rank == 0:
        checkpoint_audit = reload_audit(last_checkpoint, model, optimizer, scheduler, scaler, tracker,
                                        config, step, device)
        report = {"status": "PASS", "route": ROUTE, "mode": config["mode"], "optimizer_steps": step,
                  "contract_sha256": config["contract_sha256"], "code_sha256": contract["code_sha256"], "geometry": geometry,
                  "rank_checks": rank_checks, "layout": model.caption_decoder.last_layout,
                  "initialization": initialization, "freeze_audit": freeze_audit,
                  "trainability": trainability, "frozen_parameters_exact": frozen_before == frozen_after,
                  "execution_audit": audit["execution_audit"], "module_mode": mode_after,
                  "checkpoint": checkpoint_audit, "elapsed_seconds": time.monotonic() - start_time,
                  "config_audit": str(run_dir / "config_audit.json"),
                  "decode_scope": "1 staged row/source through training loader/model + all consumed training batches; no staging decode"}
        write_json(run_dir / "training_report.json", report)
        print(json.dumps({"event": "PASS", "mode": config["mode"], "optimizer_steps": step,
                          "report": str(run_dir / "training_report.json")}), flush=True)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != WORLD_SIZE or not torch.cuda.is_available():
        raise RuntimeError("this route must be launched by torchrun on eight CUDA GPUs")
    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    try:
        train(config, rank, device)
    except Exception:
        write_json(Path(config["run_dir"]) / f"failure_rank{rank}.json",
                   {"status": "FAIL", "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
