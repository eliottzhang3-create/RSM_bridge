"""Audits shared by the real five-update smoke and formal training."""
from __future__ import annotations

import hashlib

import torch

from contract import TEXT_MODEL_SCHEDULE
from optim import parameter_role


def frozen_digest(model):
    """Hash every frozen parameter byte, rather than sampling a few entries."""
    result = hashlib.sha256()
    for name, tensor in model.named_parameters():
        if tensor.requires_grad:
            continue
        result.update(name.encode() + b"\0")
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def trainable_snapshot(model):
    # New modules/routers are small; retain them in full. Sample every Core
    # matrix to keep the audit bounded without copying the full trained Core.
    result = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            role = parameter_role(name)
            stride = max(1, p.numel() // 4096) if role == "core" else 1
            result[name] = (stride, p.detach().flatten()[::stride].cpu().clone())
    return result


def changed_parameters(model, before):
    params = dict(model.named_parameters())
    result = {}
    for name, (stride, old) in before.items():
        current = params[name].detach().flatten()[::stride].cpu()
        result[name] = {"role": parameter_role(name), "changed": not torch.equal(old, current),
                        "max_abs_delta": float((old - current).abs().max())}
    return result


def assert_query_semantics(mesh):
    """Check real intermediate tensors, including the distinct Z1/H0 queries."""
    if tuple(x["physical_index"] for x in mesh.last_forward_trace) != tuple(TEXT_MODEL_SCHEDULE):
        raise ValueError("physical execution schedule changed")
    if len(mesh.last_cross_queries) != 2 or len(mesh.last_core_outputs) != 2:
        raise ValueError("expected two Core passes and exactly two audio cross-attentions")
    checks = {
        "ca0_Z0": (mesh.last_cross_queries[0], mesh.last_prefix_output),
        "ca1_Z1": (mesh.last_cross_queries[1], mesh.last_core_outputs[0]),
        "perceive0_Z0": (mesh.last_router_queries["perceive_pre"], mesh.last_prefix_output),
        "perceive1_H0": (mesh.last_router_queries["perceive_0"], mesh.last_core_inputs[0]),
    }
    for name in ("write_pre", "read_pre"):
        checks[name] = (mesh.last_router_queries[name], mesh.last_prefix_output)
    for loop in range(2):
        for kind in ("write", "read"):
            checks[f"{kind}_{loop}"] = (mesh.last_router_queries[f"{kind}_{loop}"], mesh.last_core_inputs[loop])
    if any(not torch.equal(a, b) for a, b in checks.values()):
        raise ValueError("a router/cross-attention used the wrong query")
    if len(mesh.last_router_weights) != 8:
        raise ValueError("expected six old routers and two perceive routers")
    for weights in mesh.last_router_weights.values():
        if not torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]), atol=1e-6):
            raise ValueError("router weights do not form a softmax")
    # Replay the complete memory recurrence independently from saved weights
    # and values. This also detects accidentally replacing old writes.
    memory = mesh.last_initial_memory.clone()
    values = [mesh.last_prefix_output] + mesh.last_core_outputs
    names = ["pre", "0", "1"]
    for index, (value, suffix) in enumerate(zip(values, names)):
        weights = mesh.last_router_weights[f"write_{suffix}"]
        memory = memory + value.unsqueeze(1) * weights.transpose(1, 2).unsqueeze(-1)
        if index < 2:
            weights = mesh.last_router_weights["perceive_pre" if index == 0 else "perceive_0"]
            memory = memory + mesh.last_audio_outputs[index].unsqueeze(1) * weights.transpose(1, 2).unsqueeze(-1)
        torch.testing.assert_close(memory, mesh.last_memory_write_history[index], rtol=1e-5, atol=1e-6)
        read = mesh.last_router_weights[f"read_{suffix}"]
        expected_hidden = (memory * read.transpose(1, 2).unsqueeze(-1)).sum(1)
        if index < 2:
            torch.testing.assert_close(expected_hidden, mesh.last_core_inputs[index], rtol=1e-5, atol=1e-6)
    return {name: True for name in checks} | {"memory_recurrence": True, "router_softmax": True,
                                             "audio_rounds": 2, "router_calls": 8}


def zero_init_audit(model, inputs):
    """Compare real logits to the original MeSH path with audio writes absent."""
    mesh = model.caption_decoder.lm.model
    was_training = model.training
    model.eval()
    mesh.audit_mode = True
    stage_calls = [0, 0]
    def count(index):
        def hook(*_args):
            stage_calls[index] += 1
        return hook
    htsat = model.audio_encoder.base.htsat
    handles = [htsat.layers[i].register_forward_hook(count(i - 2)) for i in (2, 3)]
    try:
        with torch.no_grad():
            for output in mesh.audio_attention.o_projs:
                if output.weight.count_nonzero() or output.bias.count_nonzero():
                    raise ValueError("both independent output projections must start at exactly zero")
            if mesh.audio_memory_builder.frequency_embedding.count_nonzero():
                raise ValueError("frequency embedding must start at exactly zero")
            enabled = model(inputs).logits.detach().cpu()
            query_checks = assert_query_semantics(mesh)
            layout = dict(mesh.audio_memory_builder.last_layout)
            if any(x.count_nonzero() for x in mesh.last_audio_outputs):
                raise ValueError("initial audio writes were nonzero")
            mesh.audio_injection_enabled = False
            disabled = model(inputs).logits.detach().cpu()
            if not torch.equal(enabled, disabled):
                delta = float((enabled - disabled).abs().max())
                raise ValueError(f"zero-init changed original-path logits (max delta {delta})")
            # Two halves in each of two forwards, with stage3 and stage4 run
            # exactly once per half. The tap must not skip the later stages.
            if stage_calls != [4, 4]:
                raise ValueError(f"HTSAT did not complete both later stages: {stage_calls}")
    finally:
        for handle in handles:
            handle.remove()
        mesh.audio_injection_enabled = True
        mesh.audit_mode = False
        model.train(was_training)
    return {"status": "PASS", "zero_init_logits_exact": True, "max_abs_delta": 0.0,
            "queries": query_checks, "audio_memory": layout, "later_htsat_stage_calls": stage_calls}


def gradient_audit(model):
    """Run after unscale, before clipping/step/zero_grad."""
    roles = {}
    missing, nonfinite, frozen = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:
                frozen.append(name)
            continue
        role = parameter_role(name)
        if p.grad is None:
            missing.append(name)
            continue
        grad = p.grad.detach()
        if not bool(torch.isfinite(grad).all()):
            nonfinite.append(name)
        roles[role] = roles.get(role, 0.0) + float(grad.square().sum())
    if missing or nonfinite or frozen:
        raise ValueError(f"gradient contract failed: missing={missing}, nonfinite={nonfinite}, frozen={frozen}")
    mesh = model.caption_decoder.lm.model
    boundaries = []
    for kind, refs in (("H", mesh.last_core_input_refs), ("Z", mesh.last_core_output_refs)):
        for index, tensor in enumerate(refs):
            valid = tensor.grad is not None and bool(torch.isfinite(tensor.grad).all()) and bool(tensor.grad.count_nonzero())
            if not valid:
                raise ValueError(f"gradient did not reach Core boundary {kind}{index}")
            boundaries.append({"boundary": f"{kind}{index}", "gradient_norm": float(tensor.grad.norm())})
    if len(boundaries) != 4:
        raise ValueError("both shared Core executions must retain input/output gradients")
    return {"role_gradient_l2": {k: v ** 0.5 for k, v in roles.items()},
            "frozen_grads_absent": True, "all_trainable_grads_present_finite": True,
            "core_boundary_gradients": boundaries}
