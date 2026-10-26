"""Formal baseline freeze, gradient and checkpoint audits."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from optim import parameter_role
from contract import TEXT_MODEL_SCHEDULE


def assert_state_exact(expected, actual, path="checkpoint"):
    if isinstance(expected, torch.Tensor):
        matches = (isinstance(actual, torch.Tensor) and expected.dtype == actual.dtype
                   and torch.equal(expected.detach().cpu(), actual.detach().cpu()))
    elif isinstance(expected, np.ndarray):
        matches = (isinstance(actual, np.ndarray) and expected.dtype == actual.dtype
                   and np.array_equal(expected, actual))
    elif isinstance(expected, dict):
        matches = isinstance(actual, dict) and expected.keys() == actual.keys()
        if matches:
            for key in expected:
                assert_state_exact(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        matches = type(expected) is type(actual) and len(expected) == len(actual)
        if matches:
            for index, (left, right) in enumerate(zip(expected, actual)):
                assert_state_exact(left, right, f"{path}[{index}]")
    else:
        matches = expected == actual
    if not matches:
        raise ValueError(f"serialized state differs: {path}")


def frozen_digest(model):
    result = hashlib.sha256()
    trainable = {name for name, p in model.named_parameters(remove_duplicate=False) if p.requires_grad}
    for name, tensor in model.state_dict().items():
        if name in trainable:
            continue
        result.update(name.encode() + b"\0")
        result.update(str(tensor.dtype).encode() + b"\0")
        result.update(str(tuple(tensor.shape)).encode() + b"\0")
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def trainable_snapshot(model):
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            stride = max(1, parameter.numel() // 4096)
            result[name] = (stride, parameter.detach().flatten()[::stride].cpu().clone())
    return result


def changed_parameters(model, before):
    params = dict(model.named_parameters())
    result = {}
    for name, (stride, old) in before.items():
        current = params[name].detach().flatten()[::stride].cpu()
        result[name] = {
            "role": parameter_role(name),
            "changed": not torch.equal(old, current),
            "max_abs_delta": float((old - current).abs().max()),
        }
    return result


def gradient_audit(model):
    roles, missing, nonfinite, frozen = {}, [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                frozen.append(name)
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        role = parameter_role(name)
        roles[role] = roles.get(role, 0.) + float(parameter.grad.detach().square().sum())
    if missing or nonfinite or frozen:
        raise ValueError(f"gradient contract failed: missing={missing}, nonfinite={nonfinite}, frozen={frozen}")
    return {"role_gradient_l2": {name: value ** .5 for name, value in roles.items()},
            "frozen_grads_absent": True, "all_trainable_grads_present_finite": True}


def trainability_audit(model):
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    expected = set()
    for name, _ in model.named_parameters():
        if name.startswith("caption_decoder.lm.model.layers."):
            parts = name.split(".")
            if len(parts) > 4 and parts[4].isdigit() and 5 <= int(parts[4]) < 15:
                expected.add(name)
    if trainable != expected:
        raise ValueError(f"trainable whitelist mismatch: extra={sorted(trainable - expected)[:5]}, "
                         f"missing={sorted(expected - trainable)[:5]}")
    return {"trainable_tensors": len(trainable), "trainable_numel": sum(
        p.numel() for name, p in model.named_parameters() if name in trainable),
        "only_core_5_to_14": True, **module_mode_audit(model)}


def module_mode_audit(model):
    layers = model.caption_decoder.lm.model.layers
    allowed = {id(module) for layer in layers[5:15] for module in layer.modules()}
    incorrect = [name for name, module in model.named_modules() if name
                 and bool(module.training) != (id(module) in allowed and model.training)]
    if incorrect:
        raise ValueError(f"frozen eval / Core training mode mismatch: {incorrect}")
    return {"frozen_modules_all_eval": True, "core_training": bool(model.training)}


class ExecutionAudit:
    """Observe the first real optimizer update, including both Core backward paths."""
    def __init__(self, model):
        self.records = []
        self.handles = []
        self.handles.append(model.caption_decoder.lm.register_forward_pre_hook(self._start))
        self.handles.append(model.caption_decoder.lm.register_forward_hook(self._finish))
        for index, layer in enumerate(model.caption_decoder.lm.model.layers):
            self.handles.append(layer.register_forward_hook(self._layer_hook(index)))

    def _start(self, module, inputs):
        self.records.append({"schedule": [], "backward_calls": [], "finite": True})

    def _layer_hook(self, index):
        def observe(module, inputs, output):
            record = self.records[-1]
            occurrence = len(record["schedule"])
            record["schedule"].append(index)
            if 5 <= index < 15:
                hidden = output[0] if isinstance(output, tuple) else output
                if not hidden.requires_grad:
                    raise ValueError(f"Core execution detached: physical={index} logical={occurrence}")
                def backward(gradient):
                    record["backward_calls"].append(occurrence)
                    record["finite"] &= bool(torch.isfinite(gradient).all())
                hidden.register_hook(backward)
        return observe

    def _finish(self, module, inputs, output):
        if self.records[-1]["schedule"] != list(TEXT_MODEL_SCHEDULE):
            raise ValueError("actual non-MeSH decoder execution schedule differs")

    def report_and_remove(self, expected_microbatches):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        expected = [slot for slot, layer in enumerate(TEXT_MODEL_SCHEDULE) if 5 <= layer < 15]
        if len(self.records) != expected_microbatches or any(
                sorted(record["backward_calls"]) != expected or not record["finite"]
                for record in self.records):
            raise ValueError(f"both Core loops must receive finite gradients: {self.records}")
        return {"schedule": list(TEXT_MODEL_SCHEDULE), "microbatches": len(self.records),
                "core_forward_calls_per_microbatch": 20,
                "both_core_executions_have_finite_backward": True}
