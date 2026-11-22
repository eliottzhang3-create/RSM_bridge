"""Task-gradient checks before weight decay/optimizer updates, one record per layer."""
from __future__ import annotations
from text_hrm_h2l3.train import gradient_audit as original_gradient_audit


def prelude_parameters(model):
    import torch
    result = {}
    for index in range(5):
        parameters = list(model.model.layers[index].parameters())
        present = all(p.requires_grad and p.grad is not None for p in parameters)
        finite = present and all(bool(torch.isfinite(p.grad).all()) for p in parameters)
        norm = sum(float(p.grad.double().square().sum()) for p in parameters if p.grad is not None)**0.5
        result[str(index)] = dict(trainable=all(p.requires_grad for p in parameters),
            present=present,finite=finite,l2=norm)
    return dict(passed=all(v['trainable'] and v['present'] and v['finite'] and v['l2']>0 for v in result.values()),
                layers=result,phase='task_gradient_before_optimizer_and_weight_decay')


def gradient_audit(model,bp):
    if bp != 5:
        raise ValueError('this route requires fixed BP5')
    result = original_gradient_audit(model,bp)
    result['prelude_per_layer'] = prelude_parameters(model)
    result['passed'] = result['passed'] and result['prelude_per_layer']['passed']
    return result
