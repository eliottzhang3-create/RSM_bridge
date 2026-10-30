"""Mizar Stage 1 discriminative AdamW and its adaptive GradNormTracker."""
import torch

from contract import lr_factor


def parameter_role(name):
    prefix = "caption_decoder.lm.model.audio_crossattn."
    if name.startswith(prefix):
        component = name[len(prefix):].split(".")[0]
        if component not in {"projections", "gate1", "gate2", "cross_attention", "round_bias",
                             "audio_input_norms", "audio_output_norms", "text_norm"}:
            raise ValueError(f"unexpected cross attention component: {name}")
        return "crossattn_" + component
    if name.startswith("audio_encoder.base.mapper."):
        return "map"
    if name.startswith("caption_decoder.lm."):
        return "default"
    raise ValueError(f"unexpected trainable parameter outside mapper/full LM: {name}")


def build_optimizer(model, total_steps):
    groups, names = {}, {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        role = parameter_role(name)
        lr = 3e-4 if role == "default" else 6e-4
        decay = 0. if param.ndim <= 1 or name.endswith((".bias", "_bias")) else 0.01
        key = (role, lr, decay)
        groups.setdefault(key, []).append(param)
        names.setdefault(key, []).append(name)
    if {key[0] for key in groups} != {"map", "default", "crossattn_projections", "crossattn_gate1",
                                       "crossattn_gate2", "crossattn_cross_attention", "crossattn_round_bias",
                                       "crossattn_audio_input_norms", "crossattn_audio_output_norms",
                                       "crossattn_text_norm"}:
        raise ValueError("expected trainable CED mapper and full language model")
    audit = [{"role": role, "base_lr": lr, "weight_decay": wd,
              "numel": sum(p.numel() for p in groups[(role, lr, wd)]), "names": names[(role, lr, wd)]}
             for role, lr, wd in sorted(groups)]
    optimizer = torch.optim.AdamW(
        [{"params": groups[key], "lr": key[1], "weight_decay": key[2]} for key in sorted(groups)],
        lr=3e-4, weight_decay=.01, betas=(.9, .999), eps=1e-8)
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
