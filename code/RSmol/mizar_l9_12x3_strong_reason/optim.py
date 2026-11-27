import torch
from contract import GROUP_LRS, TRAINABLE_ROLES, lr_factor, parameter_role, rate_group


def build_optimizer(model, total_steps):
    groups, names = {}, {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        role = parameter_role(name)
        if role == "frozen":
            raise ValueError(f"frozen parameter entered optimizer: {name}")
        decay = 0. if role=="loop_bias" or p.ndim <= 1 or name.endswith(".bias") else .01
        groups.setdefault((role, decay), []).append(p)
        names.setdefault((role, decay), []).append(name)
    if {key[0] for key in groups} != set(TRAINABLE_ROLES):
        raise ValueError("optimizer must contain five shared layers, L15, three biases and the shared gate")
    audit = [{"role": role, "lr_group":rate_group(role),"base_lr": GROUP_LRS[rate_group(role)][0],
              "min_lr":GROUP_LRS[rate_group(role)][1],"weight_decay": wd,
              "names": names[(role,wd)], "numel": sum(p.numel() for p in groups[(role,wd)])}
             for role,wd in sorted(groups)]
    optimizer = torch.optim.AdamW([{"params":groups[key],"role":key[0],"lr_group":rate_group(key[0]),
        "lr":GROUP_LRS[rate_group(key[0])][0],"min_lr":GROUP_LRS[rate_group(key[0])][1],"weight_decay":key[1]}
        for key in sorted(groups)], betas=(.9,.999), eps=1e-8)
    functions=[(lambda step, ratio=group["min_lr"]/group["lr"]:lr_factor(step,total_steps,ratio))
               for group in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, functions)
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
