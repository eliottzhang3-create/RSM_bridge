"""Role-specific AdamW and cosine floors; original GradNormTracker is copied below."""
import torch
from contract import LM_MAX_LR,MAP_MAX_LR,lr_factor


def parameter_role(name):
    if name.startswith('audio_encoder.base.mapper.'): return 'map'
    if name.startswith('caption_decoder.lm.'): return 'default'
    raise ValueError(f'unexpected trainable parameter:{name}')


def build_optimizer(model,total_steps):
    groups,names={},{}
    for name,param in model.named_parameters():
        if not param.requires_grad: continue
        role=parameter_role(name)
        maximum=MAP_MAX_LR if role=='map' else LM_MAX_LR
        decay=0. if param.ndim<=1 or name.endswith('.bias') else .01
        key=(role,maximum,decay)
        groups.setdefault(key,[]).append(param)
        names.setdefault(key,[]).append(name)
    if {key[0] for key in groups}!={'map','default'}:
        raise ValueError('expected complete trainable HRM and CED mapper')
    keys=sorted(groups)
    optimizer=torch.optim.AdamW([dict(params=groups[key],lr=key[1],weight_decay=key[2],role=key[0]) for key in keys],
        betas=(.9,.999),eps=1e-8,weight_decay=.01)
    # Capture role by value, never the loop variable used by a late-bound lambda.
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
        [lambda step,role=key[0]:lr_factor(step,total_steps,role) for key in keys])
    audit=[dict(role=role,base_lr=lr,weight_decay=wd,numel=sum(p.numel() for p in groups[(role,lr,wd)]),
        names=names[(role,lr,wd)]) for role,lr,wd in keys]
    return optimizer,scheduler,audit

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
