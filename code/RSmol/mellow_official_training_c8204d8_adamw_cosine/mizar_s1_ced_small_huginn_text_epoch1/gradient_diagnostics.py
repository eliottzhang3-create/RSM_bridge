"""Failure-only gradient inspection before clipping; no updates or RNG use."""
import math

import torch


def json_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def inspect_gradients(parameters, total_norm, clip_scale):
    """Recompute norms in FP64 on the original, unscaled, unclipped gradients."""
    records, missing = [], []
    for name, parameter in parameters:
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        gradient = parameter.grad.detach()
        finite = torch.isfinite(gradient)
        bad = int((~finite).sum().item())
        norm64 = torch.linalg.vector_norm(gradient, ord=2, dtype=torch.float64).item()
        maximum = gradient.abs().max().item()
        records.append({"name": name, "dtype": str(gradient.dtype), "nonfinite_elements": bad,
                        "l2_fp64": json_number(norm64), "max_abs": json_number(maximum)})
    nonfinite = [record for record in records if record["nonfinite_elements"]]
    finite_norms = [record["l2_fp64"] for record in records if isinstance(record["l2_fp64"], float)]
    total64 = math.hypot(*finite_norms) if len(finite_norms) == len(records) else float("nan")
    if nonfinite:
        reason = "nonfinite_gradient_values"
    elif not math.isfinite(total_norm) and math.isfinite(total64):
        reason = "fp32_norm_overflow_with_finite_gradients"
    elif math.isfinite(total_norm) and total_norm >= 1e6:
        reason = "finite_gradient_norm_reached_emergency_threshold"
    else:
        reason = "invalid_norm_or_clip_scale"
    top = sorted((r for r in records if isinstance(r["l2_fp64"], float)),
                 key=lambda r: r["l2_fp64"], reverse=True)[:20]
    return {"reason": reason, "gradient_l2_fp32_before_clip": json_number(total_norm),
            "gradient_l2_fp64_before_clip": json_number(total64),
            "gradient_common_scale": json_number(clip_scale), "emergency_threshold": 1e6,
            "nonfinite_parameters": nonfinite, "missing_gradients": missing,
            "largest_gradients_fp64": top, "parameter_count_with_gradients": len(records),
            "inspection_point": "after_unscale_before_clipping"}
