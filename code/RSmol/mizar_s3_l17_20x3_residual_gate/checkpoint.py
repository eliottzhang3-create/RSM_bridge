"""Full-state checkpoints, strict named optimizer inventory, no frozen drift."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import torch
import torch.distributed as dist
from contract import ROUTE, GROUP_LRS, EPOCHS, KEEP_CHECKPOINTS, training_cursor, code_digest, digest, group_lr, parameter_role
from training_helpers import rng_state, tensor_digest


def frozen_items(model):
    return [(name,tensor) for name,tensor in model.state_dict().items() if parameter_role(name) == "frozen"]


def inventory(model, optimizer):
    names = {id(p):name for name,p in model.named_parameters()}
    return [[{"name":names[id(p)],"shape":list(p.shape),"dtype":str(p.dtype)} for p in group["params"]]
            for group in optimizer.param_groups]


def validate_saved(saved, model, optimizer, config, *, allow_complete=False):
    contract = config["contract"]
    if (saved.get("schema_version") != 1 or saved.get("route_variant") != ROUTE
            or saved.get("mode") != "formal" or saved.get("contract") != contract
            or saved.get("contract_sha256") != digest(contract)
            or contract["code_sha256"] != code_digest()):
        raise ValueError("resume identity differs: route/source/model/data/environment/recipe")
    if saved.get("optimizer_inventory") != inventory(model, optimizer):
        raise ValueError("optimizer named parameter inventory/order changed")
    total = contract["geometry"]["total_steps"]
    step = saved.get("total_step")
    if type(step) is not int or not 1 <= step <= total or (step == total and not allow_complete):
        raise ValueError("invalid or already-completed checkpoint cursor")
    if (saved.get("epoch"), saved.get("next_optimizer_step")) != training_cursor(step,contract["geometry"]):
        raise ValueError("checkpoint epoch/update cursor contradicts its step")
    if saved.get("num_epochs") != EPOCHS or saved.get("scheduler",{}).get("last_epoch") != step:
        raise ValueError("scheduler/epoch horizon changed")
    schedule = saved["scheduler"]
    expected_lrs = [group_lr(g["lr_group"],step,total) for g in optimizer.param_groups]
    if (schedule.get("_step_count") != step + 1
            or schedule.get("base_lrs") != [GROUP_LRS[g["lr_group"]][0] for g in optimizer.param_groups]
            or schedule.get("_last_lr") != expected_lrs
            or schedule.get("lr_lambdas") != [None] * len(optimizer.param_groups)):
        raise ValueError("scheduler state differs from the actual update cursor")
    states = saved.get("random_state_by_rank")
    if not isinstance(states,list) or len(states) != contract["geometry"]["world_size"]:
        raise ValueError("checkpoint lacks every rank RNG")
    for value in states:
        if not isinstance(value,dict) or set(value) != {"python","numpy","torch","cuda"}:
            raise ValueError("incomplete rank RNG")
        if (not torch.is_tensor(value["torch"]) or value["torch"].dtype != torch.uint8
                or (next(model.parameters()).device.type == "cuda"
                    and (not torch.is_tensor(value["cuda"]) or value["cuda"].dtype != torch.uint8))):
            raise ValueError("invalid CPU/CUDA RNG tensor")
    current, state = model.state_dict(), saved.get("state_dict",{})
    if set(state) != set(current):
        raise ValueError("model state inventory changed")
    for name,tensor in current.items():
        value = state[name]
        if (not torch.is_tensor(value) or value.shape != tensor.shape or value.dtype != tensor.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"invalid model tensor: {name}")
        if parameter_role(name) == "frozen" and not torch.equal(tensor.detach().cpu(),value.cpu()):
            raise ValueError(f"frozen official state changed: {name}")
    a,b = "caption_decoder.lm.model.embed_tokens.weight","caption_decoder.lm.lm_head.weight"
    if not torch.equal(state[a],state[b]):
        raise ValueError("tied embedding/head conflict")
    expected_frozen = tensor_digest(frozen_items(model))
    if saved.get("frozen_sha256") != expected_frozen:
        raise ValueError("frozen official state digest changed")
    opt = saved.get("optimizer",{})
    groups = opt.get("param_groups",[])
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("wrong optimizer group count")
    expected_ids = []
    current_groups = optimizer.state_dict()["param_groups"]
    for sg,cg,ig in zip(groups,optimizer.param_groups,current_groups):
        if len(sg["params"]) != len(cg["params"]):
            raise ValueError("optimizer parameter count mismatch")
        if sg["params"] != ig["params"]:
            raise ValueError("optimizer parameter ID order changed")
        for key in ("weight_decay","betas","eps","initial_lr","min_lr","lr_group","role"):
            if sg.get(key) != cg.get(key):
                raise ValueError(f"optimizer configuration changed: {key}")
        if not math.isclose(sg["lr"],group_lr(cg["lr_group"],step,total),rel_tol=1e-12,abs_tol=0.):
            raise ValueError("optimizer next LR differs from scheduler cursor")
        for key,param in zip(sg["params"],cg["params"]):
            expected_ids.append(key)
            moments = opt.get("state",{}).get(key,{})
            counter = moments.get("step")
            if (set(moments) != {"step","exp_avg","exp_avg_sq"}
                    or not torch.is_tensor(counter) or counter.numel() != 1
                    or not bool(torch.isfinite(counter).all()) or float(counter) != step):
                raise ValueError("optimizer moment/step inventory changed")
            for name in ("exp_avg","exp_avg_sq"):
                value = moments[name]
                if value.shape != param.shape or value.dtype != param.dtype or not bool(torch.isfinite(value).all()):
                    raise ValueError("invalid optimizer moments")
    if len(expected_ids) != len(set(expected_ids)) or set(opt["state"]) != set(expected_ids):
        raise ValueError("duplicated/extra/missing optimizer parameter IDs")
    names = {item["name"] for group in saved["optimizer_inventory"] for item in group}
    tracker = saved.get("grad_norm_tracker",{})
    if set(tracker) != names or any(len(v)!=2 or not all(math.isfinite(x) and x>=0 for x in v) for v in tracker.values()):
        raise ValueError("invalid GradNormTracker state")
    scaler = saved.get("grad_scaler")
    if not isinstance(scaler,dict) or any(isinstance(v,float) and not math.isfinite(v) for v in scaler.values()):
        raise ValueError("invalid GradScaler state")
    if scaler and (set(scaler) != {"scale","growth_factor","backoff_factor","growth_interval","_growth_tracker"}
            or scaler["scale"] <= 0 or scaler["growth_factor"] != 2.
            or scaler["backoff_factor"] != .5 or scaler["growth_interval"] != 2000
            or type(scaler["_growth_tracker"]) is not int or not 0 <= scaler["_growth_tracker"] < 2000):
        raise ValueError("incompatible GradScaler policy/state")
    return step


def save(path, model, optimizer, scheduler, scaler, tracker, config, step, frozen_sha, device):
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states,rng_state(device))
    failure = None
    if dist.get_rank() == 0:
        try:
            if sha_frozen(model) != frozen_sha:
                raise ValueError("frozen model changed before checkpoint publication")
            payload = {"schema_version":1,"route_variant":ROUTE,"mode":"formal",
                       "contract":config["contract"],"contract_sha256":config["contract_sha256"],
                       "state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),
                       "optimizer_inventory":inventory(model,optimizer),"scheduler":scheduler.state_dict(),
                       "grad_scaler":scaler.state_dict(),"grad_norm_tracker":tracker.state_dict(),
                       "random_state_by_rank":states,"total_step":step,"num_epochs":EPOCHS,
                       "epoch":training_cursor(step,config["contract"]["geometry"])[0],
                       "next_optimizer_step":training_cursor(step,config["contract"]["geometry"])[1],
                       "frozen_sha256":frozen_sha}
            path = Path(path)
            temporary = path.with_name(path.name+".partial")
            with temporary.open("wb") as handle:
                torch.save(payload,handle);handle.flush();os.fsync(handle.fileno())
            os.replace(temporary,path)
            # Keep the four most recent complete optimizer-step checkpoints.
            previous = sorted(path.parent.glob("model--step-*.ckpt"),key=lambda p:int(p.stem.rsplit("-",1)[1]))
            for old in previous[:-KEEP_CHECKPOINTS]:
                old.unlink()
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    errors = [failure]
    dist.broadcast_object_list(errors,src=0)
    if errors[0]:
        raise RuntimeError(f"collective checkpoint publication failed: {errors[0]}")
    dist.barrier()


def sha_frozen(model):
    return tensor_digest(frozen_items(model))
