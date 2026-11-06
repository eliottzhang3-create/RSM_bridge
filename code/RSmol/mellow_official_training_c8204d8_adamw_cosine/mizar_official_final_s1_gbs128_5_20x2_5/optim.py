"""Two requested LR levels and the original Mizar S1 AdamW/GradNormTracker."""
import math
import torch

from contract import lr_factor, model_parameter_role


def parameter_role(name):
    if name.startswith("caption_decoder.lm.model.layers.") and (".lora_A." in name or ".lora_B." in name):
        parts = name.split(".")
        if (len(parts) != 9 or not parts[4].isdigit() or not 5 <= int(parts[4]) < 25
                or ".".join(parts[5:7]) not in {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
                or parts[-2] not in {"lora_A", "lora_B"} or parts[-1] not in {"0", "1"}):
            raise ValueError(f"invalid middle-loop LoRA parameter: {name}")
        return f"loop{int(parts[-1]) + 1}_{parts[-2]}"
    if name.startswith("caption_decoder.lm.model.audio_linear."):
        return "reinject_linear"
    if name.startswith("caption_decoder.lm.model.audio_gate."):
        return "reinject_gate"
    for layer in (4, 8):
        if name.startswith(f"audio_encoder.base.layer_mlps.{layer}."):
            return f"layer{layer}_mlp"
        if name.startswith(f"caption_decoder.lm.model.audio_gates.{layer}."):
            return f"layer{layer}_gate"
    if name.startswith("audio_encoder.base.mapper."):
        return "map"
    base_role, _ = model_parameter_role(name)
    if base_role in {"recurrent_base", "outer_lm", "other_inherited_lm"}:
        return base_role
    raise ValueError(f"unexpected trainable parameter outside mapper/full LM: {name}")


def build_optimizer(model, total_steps):
    groups, names, seen = {}, {}, set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            raise ValueError(f"duplicate optimizer parameter: {name}")
        seen.add(id(param))
        role = parameter_role(name)
        lr = 3e-4 if role in {"map", "outer_lm", "other_inherited_lm"} else 6e-4
        if model_parameter_role(name)[1] != lr:
            raise ValueError(f"optimizer LR differs from converted model contract: {name}")
        decay = 0. if param.ndim <= 1 or name.endswith(".bias") else 0.01
        key = (role, lr, decay)
        groups.setdefault(key, []).append(param)
        names.setdefault(key, []).append(name)
    if {key[0] for key in groups} != {"map", "recurrent_base", "outer_lm", "other_inherited_lm", "layer4_mlp", "layer8_mlp", "layer4_gate", "layer8_gate", "reinject_linear", "reinject_gate",
                                       "loop1_lora_A", "loop1_lora_B", "loop2_lora_A", "loop2_lora_B"}:
        raise ValueError("expected mapper, full LM, serial audio branches and both loop LoRA A/B pairs")
    audit = [{"role": role, "base_lr": lr, "max_lr": lr, "weight_decay": wd,
              "numel": sum(p.numel() for p in groups[(role, lr, wd)]), "names": names[(role, lr, wd)]}
             for role, lr, wd in sorted(groups)]
    optimizer = torch.optim.AdamW(
        [{"params": groups[key], "lr": key[1], "weight_decay": key[2],
          "role": key[0], "param_names": names[key], "max_lr": key[1]} for key in sorted(groups)],
        lr=3e-4, weight_decay=.01, betas=(.9, .999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, total_steps))
    return optimizer, scheduler, audit


def validate_optimizer_state(saved, optimizer, step, total_steps):
    """Reject reordered/missing moments or altered LR/decay before applying state."""
    state = saved["optimizer"]
    groups = state.get("param_groups", [])
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("resume optimizer group count differs")
    seen = set()
    for old, new in zip(groups, optimizer.param_groups):
        for key in ("role", "param_names", "max_lr", "weight_decay", "betas", "eps", "initial_lr",
                    "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"):
            if old.get(key) != new.get(key):
                raise ValueError(f"resume optimizer group field differs: {key}")
        if (len(old["params"]) != len(new["params"])
                or not math.isclose(old["lr"], new["initial_lr"] * lr_factor(step, total_steps), rel_tol=1e-12)):
            raise ValueError("resume optimizer parameter count/LR cursor differs")
        for index, p in zip(old["params"], new["params"]):
            if index in seen or index not in state["state"]:
                raise ValueError("resume optimizer missing/duplicate moment entry")
            seen.add(index)
            entry = state["state"][index]
            if int(entry["step"]) != step or not torch.isfinite(torch.as_tensor(entry["step"])).all():
                raise ValueError("resume optimizer moment step differs")
            for key in ("exp_avg", "exp_avg_sq"):
                t = entry.get(key)
                if (not torch.is_tensor(t) or t.shape != p.shape or t.dtype != p.dtype
                        or not torch.isfinite(t).all()):
                    raise ValueError(f"invalid resume optimizer moment: {key}")
    schedule = saved["scheduler"]
    if (set(state["state"]) != seen or schedule.get("base_lrs") != [g["initial_lr"] for g in optimizer.param_groups]
            or schedule.get("last_epoch") != step or schedule.get("_step_count") != step + 1
            or schedule.get("_last_lr") != [g["lr"] for g in groups]):
        raise ValueError("resume optimizer/scheduler identities differ")
    expected_names = {n for g in optimizer.param_groups for n in g["param_names"]}
    norms = saved["grad_norm_tracker"]
    if (set(norms) != expected_names
            or any(len(pair) != 2 or any(not math.isfinite(float(x)) or float(x) <= 0 for x in pair)
                   for pair in norms.values())):
        raise ValueError("resume GradNormTracker lost per-parameter state")


class GradNormTracker:
    """Numerical port of runtime/stage1/utils/utils.py, without unused history."""
    def __init__(self, initial_l2_norm=1., initial_max_norm=10., overdrive_factor=2.5, momentum=.995):
        self.initial_norm = (initial_l2_norm, initial_max_norm)
        self.overdrive_factor, self.momentum = overdrive_factor, momentum
        self.running_norm = {}

    def _clip_grad(self, norm, running, scale):
        limit = running * self.overdrive_factor
        if norm > limit:
            scale = min(scale, limit / norm)
            norm = min(norm, limit * self.overdrive_factor)
        return self.momentum * running + (1 - self.momentum) * norm, scale

    def track_and_clip_(self, parameters):
        parameters = list(parameters)
        scale, l2_norms = 1., []
        for name, parameter in parameters:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().flatten()
            running_l2, running_max = self.running_norm.get(name, self.initial_norm)
            l2 = torch.linalg.norm(gradient, 2.).item()
            maximum = torch.linalg.norm(gradient, float("inf")).item()
            l2_norms.append(l2)
            running_l2, scale = self._clip_grad(l2, running_l2, scale)
            running_max, scale = self._clip_grad(maximum, running_max, scale)
            self.running_norm[name] = (running_l2, running_max)
        if scale < 1.:
            factor = torch.tensor(scale)
            for _, parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.detach().mul_(factor.to(parameter.grad.device))
        return torch.linalg.norm(torch.tensor(l2_norms), 2.).item(), scale

    def state_dict(self):
        return dict(self.running_norm)

    def load_state_dict(self, state):
        self.running_norm = dict(state)
