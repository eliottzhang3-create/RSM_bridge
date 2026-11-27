"""StrongAC one-epoch training: L10-14 x3, A9 gate, bias3, and nonrecurrent L15."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from assets import verify_bundle
from checkpoint import save, sha_frozen, validate_saved
from contract import (BATCH_SIZE, CORE_LAYERS, EPOCHS, TRAINABLE_ROLES, PREFIX_TOKENS, ROUTE,
                      RANK_UPDATE_SAMPLES, SCHEDULE, SEED, WORLD_SIZE,
                      accumulation_for_update, code_digest, digest, group_lr,
                      parameter_role, rank_seed, rank_microbatch_size, training_cursor, validate_tokenizer_contract, write_json)
from data.audiotext_dataset import AudioTextDataset, collate_fn
from data.sampler import CustomDistributedSampler
from data.staging import StagingGate
from model import load_official, validate_trainability
from journal import recover as recover_journal
from optim import GradNormTracker, build_optimizer
from training_helpers import answer_loss, move_batch, restore_rng, rng_state, tensor_digest

PHASE = {"phase":"startup","completed_updates":0}


def collective_run(phase, action, device):
    """Synchronize ordinary errors BEFORE another rank enters DDP backward."""
    PHASE["phase"] = phase
    value, failure = None, None
    try:
        value = action()
    except Exception:
        failure = traceback.format_exc()
    ok = torch.tensor(int(failure is None),device=device)
    dist.all_reduce(ok,op=dist.ReduceOp.MIN)
    if not ok.item():
        failures = [None]*dist.get_world_size()
        dist.all_gather_object(failures,failure)
        raise RuntimeError(f"collective failure in {phase}: " + json.dumps(failures))
    return value


def gradient_audit(model):
    nonzero = {role:False for role in TRAINABLE_ROLES}
    for name,p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:
                raise ValueError(f"frozen parameter has a gradient: {name}")
            continue
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):
            raise ValueError(f"missing/nonfinite task gradient: {name}")
        if bool(p.grad.ne(0).any()):
            nonzero[parameter_role(name)] = True
        if parameter_role(name)=="loop_bias" and not all(bool(row.ne(0).any()) for row in p.grad):
            raise ValueError("one of the three loop bias vectors has no task gradient")
    if not all(nonzero.values()):
        raise ValueError(f"missing role task gradient before clip/weight decay: {nonzero}")
    return {"status":"PASS","nonzero_task_gradients":nonzero,"frozen_gradients":False}


def forward_loss(ddp, inputs, ignore_index):
    output = ddp(inputs)
    model = ddp.module if hasattr(ddp,"module") else ddp
    core = model.caption_decoder.lm.model
    if core.last_layer_trace != SCHEDULE or core.last_loop_trace!=(0,1,2):
        raise ValueError("40-call recurrence or three loop-entry biases changed")
    loss = answer_loss(output,inputs,ignore_index)
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite answer-only CE")
    return loss


def optimizer_update(ddp, batches, device, optimizer, scaler, tracker, ignore_index, *, dummy_batch=None):
    """The actual production GA path, also usable with CPU DDP in tests."""
    losses = []
    count = len(batches)
    if not count:
        raise ValueError("empty optimizer window")
    model = ddp.module if hasattr(ddp,"module") else ddp
    optimizer.zero_grad(set_to_none=True)
    fallback = next((b for b in batches if b is not None),dummy_batch)
    if fallback is None:
        raise ValueError("an empty tail rank requires a zero-loss collective context")
    for micro,batch in enumerate(batches):
        active = torch.tensor(int(batch is not None),device=device)
        dist.all_reduce(active,op=dist.ReduceOp.SUM)
        if not active.item():
            raise ValueError("optimizer microbatch contains no training rows on any rank")
        inputs = move_batch(fallback if batch is None else batch,device)
        context = ddp.no_sync() if hasattr(ddp,"no_sync") and micro+1<count else nullcontext()
        with context:
            loss = collective_run("forward",lambda:forward_loss(ddp,inputs,ignore_index),device)
            # Empty tail ranks enter the same DDP collectives with exactly zero
            # loss. Real losses average over active ranks; no row is padded into
            # the sampler, counted twice, or assigned a duplicate task gradient.
            loss = loss * (dist.get_world_size()/active.item() if batch is not None else 0.)
            losses.append(loss.detach())
            # Last optimizer window has one microbatch, not four.
            scaler.scale(loss/count).backward()
        del loss,inputs
    scaler.unscale_(optimizer)
    audit = collective_run("unscaled_task_gradient_audit",lambda:gradient_audit(model),device)
    norm,scale = tracker.track_and_clip_(model.named_parameters())
    def require_norm():
        if not (math.isfinite(norm) and norm<1e6 and math.isfinite(scale) and scale>0):
            raise ValueError(f"nonfinite/emergency gradient norm: norm={norm}, scale={scale}")
    collective_run("gradient_norm",require_norm,device)
    before = scaler.get_scale()
    scaler.step(optimizer);scaler.update()
    collective_run("optimizer",lambda:None if scaler.get_scale()>=before else (_ for _ in ()).throw(ValueError("GradScaler skipped update")),device)
    return torch.stack(losses).mean(),norm,scale,audit


def role_digests(model):
    result = {}
    for role in TRAINABLE_ROLES:
        result[role] = tensor_digest((name,p) for name,p in model.named_parameters() if parameter_role(name)==role)
    return result


def inspect_tokenizer(dataset,model):
    result = validate_tokenizer_contract(dataset.tokenizer,dataset.pad_id,
                 model.caption_decoder.lm.get_input_embeddings().num_embeddings)
    result["eos_policy"] = "append space-EOT before fixed256/300 right truncate/pad; no EOS reinsertion"
    result["padding_attention"] = "original causal attention; prompt/answer PAD visible; answer PAD excluded from CE"
    return result


def train(config,rank,device):
    run = Path(config["run_dir"])
    c,g = config["contract"],config["contract"]["geometry"]
    def check_config():
        if c["route"]!=ROUTE or c["code_sha256"]!=code_digest() or config["contract_sha256"]!=digest(c):
            raise ValueError("source/recipe changed after configuration")
        if g["world_size"]!=WORLD_SIZE or g["microbatch"]!=4 or g["gradient_accumulation_steps"]!=4 or g["epochs"]!=EPOCHS:
            raise ValueError("requires single-node8, micro4, GA4, one epoch")
    collective_run("config",check_config,device)
    collective_run("official_bundle",lambda:verify_bundle(c["official_bundle"]) if rank==0 else None,device)
    seed = rank_seed(rank)
    random.seed(seed);np.random.seed(seed & 0xffffffff);torch.manual_seed(seed)
    if device.type=="cuda": torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    prepared = collective_run("staging_report",lambda:json.loads(Path(config["prepared_report"]).read_text(encoding="utf-8")),device)
    staging = collective_run("staging_owner",lambda:StagingGate(prepared,log=rank==0),device)
    model_dir = Path(c["model_dir"])
    dataset = collective_run("dataset",lambda:AudioTextDataset(prepared["metadata"],prepared["stage_root"],
                      model_dir/"assets/smollm2-135m",read_policy=prepared["read_policy"]),device)
    if len(dataset)!=g["rows"]:
        raise ValueError("dataset row count differs from recipe")
    model = collective_run("strict_full_official_load",lambda:load_official(model_dir).to(device),device)
    tokenizer_audit = collective_run("tokenizer",lambda:inspect_tokenizer(dataset,model),device)
    optimizer,scheduler,groups = build_optimizer(model,g["total_steps"])
    scaler = torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    tracker = GradNormTracker()
    initial_frozen = sha_frozen(model)
    step,resume_rng = 0,None
    if config["resume_checkpoint"]:
        def restore():
            saved = torch.load(config["resume_checkpoint"],map_location="cpu",weights_only=False)
            cursor = validate_saved(saved,model,optimizer,config)
            model.load_state_dict(saved["state_dict"],strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["grad_scaler"])
            tracker.load_state_dict(saved["grad_norm_tracker"])
            return cursor,saved["random_state_by_rank"][rank]
        step,resume_rng = collective_run("full_resume",restore,device)
    if staging.info and (staging.info["epoch"],staging.info["start_batch"])!=training_cursor(step,g):
        raise ValueError("staging/optimizer resume cursor mismatch")
    start_step = step
    collective_run("metrics_recovery",lambda:recover_journal(run/"metrics.jsonl",step) if rank==0 else None,device)
    model.train();validate_trainability(model)
    # Real per-source probes use this job's loader/audio/model; preserve all RNG.
    saved_rng = rng_state(device)
    def probes():
        result=[]
        if rank==0:
            model.eval()
            with torch.no_grad():
                for source,indices in prepared["probe_rows"].items():
                    batch=collate_fn([dataset.original_item(indices[0])])
                    loss=forward_loss(model,move_batch(batch,device),dataset.pad_id)
                    result.append({"source":source,"row_index":indices[0],"loss":float(loss)})
        return result
    source_probes=collective_run("source_probes",probes,device)
    model.train();restore_rng(saved_rng,device)
    ddp=DistributedDataParallel(model,device_ids=[device.index] if device.type=="cuda" else None,
                               broadcast_buffers=False,find_unused_parameters=False)
    before=role_digests(model)
    if resume_rng is not None:restore_rng(resume_rng,device)
    sampler=CustomDistributedSampler(dataset,WORLD_SIZE,rank,shuffle=True,seed=SEED,drop_last=False)
    iterator=None;current_epoch=None
    # Isolated iterator generator does not consume model RNG; retain the exact
    # resume snapshot nevertheless for any future loader implementation change.
    if resume_rng is not None:restore_rng(resume_rng,device)
    (run/"checkpoints").mkdir(exist_ok=True)
    if rank==0:
        write_json(run/"config_audit.json",{"status":"PREFLIGHT_PASS","config":config,
             "trainability":validate_trainability(model),"tokenizer":tokenizer_audit,
             "optimizer_groups":groups,"source_probes":source_probes,"frozen_sha256":initial_frozen,
             "resume_step":step,"model_source":"official final454, complete FP32 strict inheritance"})
        (run/"trainable_params.txt").write_text("\n".join(f"{n}\t{p.numel()}\t{list(p.shape)}" for n,p in model.named_parameters() if p.requires_grad)+"\n",encoding="utf-8")
        print(json.dumps({"event":"PREFLIGHT_PASS","geometry":g,"resume_step":step,
                          "trainability":validate_trainability(model),"frozen_sha256":initial_frozen}),flush=True)
    started=time.monotonic();local_switch=None;consumed=0;first_audit=None
    # A dataset position is unique even when two source views share a recording.
    coverage=[]
    metrics=(run/"metrics.jsonl").open("a",encoding="utf-8") if rank==0 else None
    try:
        for update in range(step,g["total_steps"]):
            epoch,epoch_update=training_cursor(update,g)
            if epoch!=current_epoch:
                sampler.set_epoch(epoch);sampler.start_index=epoch_update*RANK_UPDATE_SAMPLES
                loader=DataLoader(dataset,batch_size=BATCH_SIZE,sampler=sampler,drop_last=False,num_workers=0,
                    collate_fn=collate_fn,generator=torch.Generator().manual_seed(SEED+epoch))
                iterator=iter(loader);current_epoch=epoch
                if rank==0:print(json.dumps({"event":"EPOCH_START","epoch":epoch+1,"next_epoch_update":epoch_update,"completed_updates":step}),flush=True)
            PHASE.update(completed_updates=step)
            collective_run("staging_health",lambda:staging.wait_for_batch(epoch,epoch_update),device)
            if not dataset.audio_paths.local:
                ready=torch.tensor(int(staging.complete),device=device)
                dist.all_reduce(ready,op=dist.ReduceOp.MIN)
                if ready.item():
                    dataset.audio_paths.use_local();local_switch=step
                    if rank==0:print(json.dumps({"event":"AUDIO_READ_SWITCH","completed_updates":step,"to":"local"}),flush=True)
            micro=accumulation_for_update(epoch_update,g["rows"])
            def decode_window():
                result=[]
                for m in range(micro):
                    expected=rank_microbatch_size(g["rows"],rank,epoch_update,m)
                    batch=next(iterator) if expected else None
                    if batch is not None and len(batch["row_index"])!=expected:
                        raise ValueError("rank microbatch differs from the exact no-padding shard")
                    result.append(batch)
                return result
            batches=collective_run("decode_window",decode_window,device)
            rows=[i for batch in batches if batch is not None for i in batch["row_index"]]
            dummy=collective_run("empty_tail_context",lambda:collate_fn([dataset.original_item(0)])
                if all(b is None for b in batches) else None,device)
            lr_used=[x["lr"] for x in optimizer.param_groups]
            if any(not math.isclose(group["lr"],group_lr(group["lr_group"],step,g["total_steps"]),rel_tol=1e-12) for group in optimizer.param_groups):
                raise ValueError("actual optimizer LR differs from scheduler position")
            loss,norm,scale,audit=optimizer_update(ddp,batches,device,optimizer,scaler,tracker,dataset.pad_id,dummy_batch=dummy)
            if first_audit is None:
                first_audit=audit;write_json(run/f"first_gradient_rank{rank}.json",audit)
            del batches
            scheduler.step();step+=1;consumed+=len(rows);coverage.extend((epoch,row) for row in rows)
            PHASE.update(completed_updates=step)
            reduced=loss.clone();dist.all_reduce(reduced)
            global_rows=torch.tensor(len(rows),device=device);dist.all_reduce(global_rows)
            record={"step":step,"epoch":epoch+1,"epoch_update":epoch_update+1,"epoch_fraction":(epoch_update+1)/g["steps_per_epoch"],"total_steps":g["total_steps"],"rank_mean_loss":float(reduced)/WORLD_SIZE,
                    "lr_used":lr_used,"lr_next":[x["lr"] for x in optimizer.param_groups],
                    "lr_used_by_group":{group["lr_group"]:used for group,used in zip(optimizer.param_groups,lr_used)},
                    "lr_next_by_group":{group["lr_group"]:group["lr"] for group in optimizer.param_groups},
                    "microbatches":micro,"effective_global_batch":int(global_rows.item()),
                    "gradient_l2_before_clip":norm,"gradient_common_scale":scale,
                    "grad_scaler":scaler.get_scale(),"rank0_rows":rows,
                    "audio_read_mode":"local" if dataset.audio_paths.local else "shared",
                    "staging_complete":staging.complete,"local_switch_step":local_switch,
                    "elapsed_seconds":time.monotonic()-started}
            if metrics:
                metrics.write(json.dumps(record)+"\n");metrics.flush()
                if step%25==0 or step==g["total_steps"]:print(json.dumps(record),flush=True)
            if epoch_update+1==g["steps_per_epoch"]:
                def finish_epoch():
                    try:next(iterator)
                    except StopIteration:pass
                    else:raise ValueError("unconsumed rows at epoch boundary")
                    actual=[row for ep,row in coverage if ep==epoch]
                    expected=[dataset.order[i] for i in sampler]
                    if actual!=expected:raise ValueError("epoch coverage differs from exact resumed no-padding shard")
                    write_json(run/f"epoch{epoch+1}_rank{rank}.json",{"status":"PASS","epoch":epoch+1,
                        "rows_consumed":len(actual),"unique_rows":len(set(actual)),"resumed_start_update":sampler.start_index//RANK_UPDATE_SAMPLES,
                        "coverage_sha256":digest(actual)})
                collective_run("epoch_coverage",finish_epoch,device)
            if step in config["checkpoint_steps"]:
                collective_run("frozen_and_trainability_check",lambda:require_frozen(model,initial_frozen),device)
                PHASE["phase"]="checkpoint"
                save(run/"checkpoints"/f"model--step-{step}.ckpt",model,optimizer,scheduler,scaler,tracker,config,step,initial_frozen,device)
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("unconsumed rows after the complete epoch")
        collective_run("final_staging",staging.wait_complete,device)
        collective_run("final_frozen",lambda:require_frozen(model,initial_frozen),device)
        start_epoch,start_update=training_cursor(start_step,g)
        expected_consumed=max(0,g["rank_samples"][rank]-start_update*RANK_UPDATE_SAMPLES)+(EPOCHS-start_epoch-1)*g["rank_samples"][rank]
        if consumed != expected_consumed:
            raise ValueError("wrong number of newly consumed rows after resume")
        rank_result={"rank":rank,"optimizer_steps":step,"new_rows_consumed":consumed,
                     "new_unique_row_indices":len(set(coverage)),"coverage_sha256":digest(coverage),
                     "frozen_sha256":sha_frozen(model),"first_task_gradient_audit":first_audit,
                     "local_switch_step":local_switch,"audio_read_files":dataset.audio_paths.read_files,
                     "role_parameters_changed":{k:v!=before[k] for k,v in role_digests(model).items()}}
        write_json(run/f"rank{rank}_completion.json",rank_result)
        results=[None]*WORLD_SIZE;dist.all_gather_object(results,rank_result)
        # Rank-local sampler coverage is checked against the exact resumed
        # deterministic shard, not merely against a total count.
        def verify_coverage():
            expected=[]
            for ep in range(start_epoch,EPOCHS):
                sampler.set_epoch(ep);sampler.start_index=start_update*RANK_UPDATE_SAMPLES if ep==start_epoch else 0
                expected.extend((ep,dataset.order[i]) for i in sampler)
            if coverage!=expected:
                raise ValueError("consumed rows differ from the exact deterministic no-replacement shard")
        collective_run("complete_epoch_coverage",verify_coverage,device)
        def final_report():
            if rank==0:
                final=run/"checkpoints"/f"model--step-{step}.ckpt"
                saved=torch.load(final,map_location="cpu",weights_only=False)
                validate_saved(saved,model,optimizer,config,allow_complete=True)
                for name,value in model.state_dict().items():
                    if not torch.equal(value.detach().cpu(),saved["state_dict"][name]):
                        raise ValueError(f"final saved model differs: {name}")
                write_json(run/"training_report.json",{"status":"PASS","route":ROUTE,"optimizer_steps":step,
                    "contract_sha256":config["contract_sha256"],"geometry":g,"rank_checks":results,
                    "checkpoint":str(final),"checkpoint_reload":"PASS","epochs":EPOCHS,"full_epoch_no_drop_no_repeat":True,
                    "elapsed_seconds":time.monotonic()-started})
        collective_run("final_reload_report",final_report,device)
        if rank==0:print(f"[training] PASS: {step} updates; {run/'training_report.json'}",flush=True)
    finally:
        if metrics:metrics.close()


def require_frozen(model,expected):
    validate_trainability(model)
    if sha_frozen(model)!=expected:
        raise ValueError("a frozen official parameter or buffer changed")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True,type=Path)
    a=p.parse_args();config=json.loads(a.config.read_text(encoding="utf-8"))
    rank,local=int(os.environ["RANK"]),int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"])!=WORLD_SIZE or not torch.cuda.is_available():
        raise RuntimeError("launch this route through its eight-GPU vc wrapper")
    torch.set_num_threads(1);torch.cuda.set_device(local);device=torch.device("cuda",local)
    dist.init_process_group("nccl",timeout=timedelta(minutes=60))
    try:
        train(config,rank,device)
    except Exception:
        write_json(Path(config["run_dir"])/f"failure_rank{rank}.json",
                   {"status":"FAILED","rank":rank,"attempt_id":config.get("attempt_id"),
                    **PHASE,"traceback":traceback.format_exc()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
