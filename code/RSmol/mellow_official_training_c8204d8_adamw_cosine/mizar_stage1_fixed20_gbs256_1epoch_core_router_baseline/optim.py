"""One-LR AdamW for the Core+router baseline."""
import torch

from contract import MAX_LR, MIN_LR, lr_factor


def parameter_role(name):
    prefix = "caption_decoder.lm.model."
    if not name.startswith(prefix):
        raise ValueError(f"parameter outside authorized Core/router set: {name}")
    local = name[len(prefix):]
    if local.startswith(("write_routers.", "read_routers.")):
        return "mesh_routers"
    if local.startswith("layers."):
        try:
            layer = int(local.split(".", 2)[1])
        except (IndexError, ValueError):
            layer = -1
        if 5 <= layer < 15:
            return "core"
    raise ValueError(f"parameter outside authorized Core/router set: {name}")


def configure_trainability(model):
    """Freeze everything, then enable only physical Core layers 5..14 and routers."""
    audit = {"trainable": {}, "frozen_numel": 0}
    for name, parameter in model.named_parameters():
        try:
            role = parameter_role(name)
        except ValueError:
            role = None
        parameter.requires_grad_(role is not None)
        if role:
            audit["trainable"][role] = audit["trainable"].get(role, 0) + parameter.numel()
        else:
            audit["frozen_numel"] += parameter.numel()
    if set(audit["trainable"]) != {"core", "mesh_routers"}:
        raise ValueError(f"missing trainable baseline component: {audit}")
    model.audio_encoder.eval()
    return audit


def build_optimizer(model, total_steps):
    groups, names = {}, {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        role = parameter_role(name)
        lr = MAX_LR
        decay = 0. if param.ndim <= 1 or name.endswith(".bias") else 0.01
        key = (role, lr, decay)
        groups.setdefault(key, []).append(param)
        names.setdefault(key, []).append(name)
    if {key[0] for key in groups} != {"core", "mesh_routers"}:
        raise ValueError("expected only Core layers 5..14 and six original routers")
    audit = [{"role": role, "base_lr": lr, "min_lr": MIN_LR, "weight_decay": wd,
              "numel": sum(p.numel() for p in groups[(role, lr, wd)]), "names": names[(role, lr, wd)]}
             for role, lr, wd in sorted(groups)]
    optimizer = torch.optim.AdamW(
        [{"params": groups[key], "lr": key[1], "weight_decay": key[2]} for key in sorted(groups)],
        lr=MAX_LR, weight_decay=.01, betas=(.9, .999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, total_steps))
    return optimizer, scheduler, audit


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
