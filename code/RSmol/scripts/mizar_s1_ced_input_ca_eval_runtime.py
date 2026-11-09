"""Evaluation observers for three-depth audio CA and official input-only.

Import the trained implementations unchanged. Hooks only record calls/shapes;
no model outputs, gates, weights or CUDA visibility are changed here.
"""
from __future__ import annotations

import copy


def generate_prefix(model, input_dict, spec, audit):
    """Normalize the real two/three-item APIs and observe one CED encoding."""
    encoder = model.audio_encoder.base
    blocks, encodings, shapes = [], [], []
    handles = []

    def block(index):
        def hook(_module, _inputs, output):
            blocks.append(index)
            shapes.append(tuple(output.shape))
        return hook

    def encoded(_module, _inputs, _output):
        encodings.append(1)

    try:
        handles.append(encoder.register_forward_hook(encoded))
        for index, module in enumerate(encoder.ced.blocks):
            handles.append(module.register_forward_hook(block(index)))
        result = model.generate_prefix_inference(input_dict)
    finally:
        for handle in handles:
            handle.remove()
    if encodings != [1] or blocks != list(range(12)) or shapes != [(2, 252, 384)] * 12:
        raise RuntimeError("requires one fixed20 CED encoding and twelve original block calls")
    layout = encoder.last_audio_layout
    if (not isinstance(layout, dict) or layout.get("embedding") != [1, 126, 576]
            or layout.get("hidden") != [1, 504, 384] or layout.get("grid") != [1, 126, 4, 384]
            or layout.get("waveform_samples") != 640000 or not layout.get("encoder_eval")):
        raise RuntimeError("CED fixed20 prefix layout differs from the training implementation")
    input_only = spec["kind"] == "official_input_only"
    if not isinstance(result, (tuple, list)) or len(result) != (2 if input_only else 3):
        raise ValueError("prefix result does not match the selected training API")
    if input_only:
        prefix, output = result
        context = None
        if (layout.get("intermediate_capture") is not False or layout.get("encoder_feature_calls") != 1
                or "layer_grids" in output):
            raise RuntimeError("input-only must retain one CED feature pass without intermediate capture")
    else:
        prefix, output, context = result
        if (layout.get("tap_layers") != [4, 8, 12] or layout.get("tap_norm") != "pre_final_norm"
                or layout.get("layer_grids") != [1, 3, 126, 1536]
                or not isinstance(context, dict) or set(context) != {"audio_grids", "prompt_mask"}
                or context["audio_grids"] is not output.get("layer_grids")
                or context["prompt_mask"] is not input_dict["input"]["attention_mask"]):
            raise RuntimeError("CA4812 requires this sample's raw ordered grids and original prompt mask")
    previous = audit.get("prefix_execution_audit", {})
    audit["prefix_execution_audit"] = {
        "status": "PASS", "checked_samples": previous.get("checked_samples", 0) + 1,
        "encoder_calls_per_sample": 1, "ced_block_calls": blocks,
        "prefix_api_items": len(result), "ced_layout": copy.deepcopy(layout),
    }
    return prefix, output, context


def install_trace_audit(lm, spec, *, audio_tokens=126, width=576):
    """Verify every full forward's physical calls and actual branch invocations."""
    core = lm.model
    ca = spec["kind"] == "audio_ca4812"
    schedule = list(spec["schedule"])
    boundaries = (5, 15) if ca else (5, 25)
    trace, events, preparations, rounds = [], [], [], []
    expected_preparations = ([f"{name}{index}" for index in range(3)
                              for name in ("audio_input_norm", "audio_projection", "audio_output_norm")]
                             + ["input_projection"] if ca else ["input_projection"])
    expected_events = [[boundary, name] for boundary in boundaries
                       for name in (("input", "depth", "query_norm", "cross_attention", "ca_gate", "interaction")
                                    if ca else ("input",))]
    audit = {"status": "PENDING", "checked_forwards": 0, "use_cache": False,
             "expected_schedule": schedule, "expected_branch_indices": list(boundaries),
             "audio_memory_slots": 0, "text_memory_slots": 0,
             "memory_scope": "none", "memory_policy": "none"}
    sequence_tokens = 0

    def begin(_module, _inputs, kwargs):
        nonlocal sequence_tokens
        trace.clear()
        events.clear()
        preparations.clear()
        rounds.clear()
        value = kwargs.get("inputs_embeds")
        if (kwargs.get("use_cache") is not False or kwargs.get("past_key_values") is not None
                or kwargs.get("attention_mask") is not None or kwargs.get("labels") is not None
                or value is None or len(value.shape) != 3 or value.shape[0] != 1 or value.shape[2] != width
                or not 383 <= value.shape[1] <= 682):
            raise RuntimeError("requires batch1/full-prefix recomputation without masks or KV cache")
        sequence_tokens = int(value.shape[1])

    def layer_call(index):
        def hook(_module, inputs, _output):
            if tuple(inputs[0].shape) != (1, sequence_tokens, width):
                raise RuntimeError("every physical layer must see the complete growing prefix")
            trace.append(index)
        return hook

    def preparation(name, input_width, output_width):
        def hook(_module, inputs, output):
            if (trace or tuple(inputs[0].shape) != (1, audio_tokens, input_width)
                    or tuple(output.shape) != (1, audio_tokens, output_width)):
                raise RuntimeError("audio/input banks must be prepared once before the first physical layer")
            preparations.append(name)
        return hook

    def event(name, output_width=None):
        def hook(_module, inputs, output):
            if tuple(inputs[0].shape) != (1, audio_tokens, width):
                raise RuntimeError("audio interaction must query exactly the first126 positions")
            if output_width is not None and tuple(output.shape) != (1, audio_tokens, output_width):
                raise RuntimeError("audio gate or query geometry mismatch")
            if name == "cross_attention" and any(tuple(v.shape) != (1, audio_tokens, width) for v in inputs[:3]):
                raise RuntimeError("CA4812 Q/K/V must all cover the complete126-token audio bank")
            events.append([len(trace), name])
        return hook

    def interaction(_module, inputs, output):
        hidden, bank, residual, round_index = inputs
        if (tuple(hidden.shape) != (1, sequence_tokens, width) or tuple(output.shape) != tuple(hidden.shape)
                or tuple(bank.shape) != (1, 3, audio_tokens, width)
                or tuple(residual.shape) != (1, audio_tokens, width)):
            raise RuntimeError("CA4812 must use three prepared depths and the original input residual")
        rounds.append(int(round_index))
        events.append([len(trace), "interaction"])

    def finish(_module, _inputs, output):
        if (trace != schedule or output.past_key_values is not None
                or tuple(output.logits.shape[:2]) != (1, sequence_tokens)):
            raise RuntimeError("actual physical layer schedule, full logits or no-cache behavior mismatch")
        if events != expected_events or preparations != expected_preparations:
            raise RuntimeError(f"input/CA execution order or bank preparation count mismatch: {events}, {preparations}")
        if ca:
            if (tuple(core.last_crossattn_trace) != boundaries or rounds != [0, 1]
                    or core.audio_crossattn.last_injection_trace != ((0, "input", "cross_attention"), (1, "input", "cross_attention"))):
                raise RuntimeError("CA4812 internal traces disagree with observed input then CA calls")
        elif (tuple(core.last_layer_trace) != tuple(schedule)
              or tuple(core.last_input_injection_trace) != boundaries or core.last_input_projection_calls != 1):
            raise RuntimeError("input-only internal traces disagree with observed fifty calls/two injections/one projection")
        audit.update(status="PASS", checked_forwards=audit["checked_forwards"] + 1,
                     last_sequence_tokens=sequence_tokens, last_physical_schedule=list(trace),
                     last_branch_events=copy.deepcopy(events), bank_preparations=list(preparations),
                     input_projection_calls=preparations.count("input_projection"))

    lm.register_forward_pre_hook(begin, with_kwargs=True)
    for index, module in enumerate(core.layers):
        module.register_forward_hook(layer_call(index))
    if ca:
        branch = core.audio_crossattn
        for index in range(3):
            branch.audio_input_norms[index].register_forward_hook(preparation(f"audio_input_norm{index}", 1536, 1536))
            branch.projections[index].register_forward_hook(preparation(f"audio_projection{index}", 1536, width))
            branch.audio_output_norms[index].register_forward_hook(preparation(f"audio_output_norm{index}", width, width))
        branch.input_projection.register_forward_hook(preparation("input_projection", width, width))
        for module, name, out_width in ((branch.input_gate, "input", 1), (branch.gate1, "depth", 3),
                                        (branch.query_norm, "query_norm", width),
                                        (branch.cross_attention, "cross_attention", None), (branch.gate2, "ca_gate", 1)):
            module.register_forward_hook(event(name, out_width))
        branch.register_forward_hook(interaction)
    else:
        core.audio_linear.register_forward_hook(preparation("input_projection", width, width))
        core.audio_gate.register_forward_hook(event("input", 1))
    lm.register_forward_hook(finish)
    return audit
