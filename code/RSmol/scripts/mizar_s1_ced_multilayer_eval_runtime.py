"""Isolated CED interaction routes, using the original CED scoring protocol.

Import the actual training models; never patch their forward methods or relax
their source identities. Per-sample context is explicit and never stored on a
module. Learned projections/compression remain inside each full LM forward.
"""
from __future__ import annotations

import copy
import importlib
import json
import os
import random
from pathlib import Path

import mizar_s1_ced_eval_runtime as reference

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "mellow_official_training_c8204d8_adamw_cosine"
REMOTE_OUTPUT_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol")
PROMPT_TOKENS, MAX_NEW_TOKENS, AUDIO_TOKENS = 256, 300, 126
PREFIX_TOKENS, HIDDEN_SIZE = 383, 576
TEXT_MODEL_SCHEDULE = tuple(list(range(15)) + list(range(5, 15)) + list(range(15, 20)))
LOOP_LORA_TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
TRAINING_LOSS_REDUCTION = reference.TRAINING_LOSS_REDUCTION
PROTOCOL = copy.deepcopy(reference.PROTOCOL)
sha256_file = reference.sha256_file
parse_training_geometry = reference.parse_training_geometry
audit_assets = reference.audit_assets
import_training_route = reference.import_training_route

ROUTE_SPECS = {}
for _family, _suffix, _version, _stamp, _kind, _count in (
    ("recursive_5_10_5", "layer36912_crossattn", 2, "20261030_140459065784533", "crossattn", 4892741),
    ("recursive_5_10_5", "layer36912_query32_crossattn", 1, "20261030_162028139130855", "query32", 6241733),
    ("recursive_mesh", "layer4_8_inject", 1, None, "inject48", 2444930),
    ("recursive_mesh", "layer4_8_listen", 1, "20261030_172450546025072", "listen48", 2450700),
    ("recursive_mesh", "layer36912_listen", 2, "20261101_113133327944502", "listen36912", 3552788),
    ("recursive_mesh", "layer4_8_audio7slot_split3", 1, "20261101_170211458358358", "audio7split3", 2468010),
    ("recursive_mesh", "layer4_8_audio7slot_split3_inject", 1, "20261102_002211827036990", "audio7split3_inject", 2469164),
    ("recursive_mesh", "layer8_4_inject", 1, "20261102_112040402999384", "inject84", 2444930),
    ("recursive_mesh", "layer4_8_audio7slot_split3_prompt_mix", 1, "20261102_132232677352730", "prompt_mix", 2802093),
    ("recursive_5_10_5", "layer8_4_inject_audio_reinject", 1, "20261102_112121702891801", "serial84", 2777859),
    ("recursive_5_10_5", "layer8_4_inject_audio_reinject_loop_lora_r8", 1, "20261103_165410438279511", "serial84", 4406019),
):
    _variant = f"{_family}_{_suffix}"
    _mesh = _family == "recursive_mesh"
    ROUTE_SPECS[_variant] = {
        "route": f"ced_small_mizar_s1_fixed20_gbs256_{_variant}_v{_version}",
        "route_root": TRAINING_ROOT / f"mizar_stage1_fixed20_gbs256_3epochs_{_family}_ced_small_{_suffix}",
        "run_root": (REMOTE_OUTPUT_ROOT / f"mizar_s1_ced_small_gbs256_{_variant}_5090" / f"formal_{_stamp}"
                     if _stamp else None),
        "kind": _kind, "mesh": _mesh, "new_parameters": _count,
        "loop_lora": _suffix.endswith("_loop_lora_r8"),
        "context_kind": ("seeds" if _kind == "prompt_mix" else "pair"
                         if _kind in ("inject48", "inject84", "serial84", "listen48", "audio7split3", "audio7split3_inject") else "grids"),
        "split_memory": _kind.startswith("audio7") or _kind == "prompt_mix",
        "memory_slots": 5 if _mesh else 0,
        "audio_memory_slots": 7 if _kind.startswith("audio7") or _kind == "prompt_mix" else 5 if _mesh else 0,
        "text_memory_slots": 5 if _mesh else 0,
        "loops": 2, "schedule": TEXT_MODEL_SCHEDULE,
        "default_step": 17313, "default_geometry": (8, 8, 4),
        "interaction_layers": ((8, 4) if _kind in ("inject84", "serial84") else (4, 8)
                               if _kind in ("inject48", "listen48", "audio7split3_inject") else ()),
        "implementation": ("models.recursive_layer_inject" if _kind in ("inject48", "inject84", "serial84") else
                           "models.recursive_mesh_listen" if _mesh else "models.recursive_crossattn"),
        "text_contract": "logical_30_physical_20_5_10x2_5" if _mesh else "logical_30_physical_20_5_10_5_loops_2_no_mesh",
        "text_architecture": "logical_30_physical_20_5_10x2_5_mesh" if _mesh else "logical_30_physical_20_5_10_5_loops_2_no_mesh",
    }

ROUTE_SPECS["recursive_mesh_5_10x4_5"] = {
    "route": "ced_small_mizar_s1_fixed20_gbs128_recursive_mesh_5_10x4_5_v1",
    "route_root": TRAINING_ROOT / "mizar_stage1_fixed20_gbs128_3epochs_recursive_mesh_5_10x4_5_ced_small",
    "run_root": REMOTE_OUTPUT_ROOT / "mizar_s1_ced_small_gbs128_recursive_mesh_5_10x4_5_5090" / "formal_20261103_093224843011222",
    "kind": "mesh_x4", "mesh": True, "new_parameters": 0,
    "context_kind": "none", "split_memory": False,
    "memory_slots": 7, "audio_memory_slots": 7, "text_memory_slots": 7,
    "loops": 4, "schedule": tuple(list(range(5)) + list(range(5, 15)) * 4 + list(range(15, 20))),
    "default_step": 34626, "default_geometry": (8, 4, 4), "interaction_layers": (),
    "implementation": "recursive_model_5_10x4_5_mesh",
    "text_contract": "logical_50_physical_20_5_10x4_5",
    "text_architecture": "logical_50_physical_20_5_10x4_5_mesh",
}


def variant_spec(eval_variant):
    if eval_variant not in ROUTE_SPECS:
        raise ValueError(f"unsupported multilayer CED evaluation variant: {eval_variant!r}")
    return ROUTE_SPECS[eval_variant]


def branch_contract(route_contract, spec):
    kind = spec["kind"]
    if kind in ("crossattn", "query32"):
        return "audio_cross_attention", route_contract.cross_attention_contract()
    if kind in ("inject48", "inject84", "serial84"):
        return "layer_injection", dict(route_contract.INJECTION_CONTRACT)
    if kind in ("listen48", "audio7split3", "audio7split3_inject", "prompt_mix"):
        return "audio_listen", dict(route_contract.LISTEN_CONTRACT)
    if kind == "mesh_x4":
        return None, None
    return "audio_listen", route_contract.audio_listen_contract()


def branch_contracts(route_contract, spec):
    key, value = branch_contract(route_contract, spec)
    result = {key: value} if key is not None else {}
    if spec["kind"] == "prompt_mix":
        result["prompt_mix"] = dict(route_contract.PROMPT_MIX_CONTRACT)
    if spec["kind"] == "serial84":
        result.update(audio_reinjection=dict(route_contract.AUDIO_REINJECTION),
                      serial_audio_injection=dict(route_contract.SERIAL_INJECTION_CONTRACT))
    if spec.get("loop_lora"):
        result["loop_lora"] = dict(route_contract.LOOP_LORA_CONTRACT)
    return result


def implementation_path(spec, route_root=None):
    relative = spec["implementation"].replace(".", "/") + ".py"
    return (Path(route_root or spec["route_root"]) if spec["implementation"].startswith("models.") else ROOT) / relative


def validate_checkpoint(saved, runtime, route_contract, expected_step,
                        expected_geometry=None, *, eval_variant):
    """Evaluation accepts completed artifacts; it does not restore training state."""
    spec = variant_spec(eval_variant)
    route = spec["route"]
    if route_contract.ROUTE != route:
        raise ValueError(f"selected training route does not match variant {eval_variant}")
    contract = saved.get("contract", {})
    if (saved.get("schema_version") != 1 or saved.get("route_variant") != route
            or saved.get("mode") != "formal" or not saved.get("state_dict")):
        raise ValueError("expected a complete formal checkpoint for the selected CED route")
    identity = route_contract.digest(contract)
    if saved.get("contract_sha256") != identity:
        raise ValueError("checkpoint embedded CED contract SHA256 mismatch")
    if (runtime.get("contract_sha256") != identity or runtime.get("contract") != contract
            or runtime.get("mode") != "formal" or runtime.get("route") != route):
        raise ValueError("runtime.json does not belong to this formal CED checkpoint")
    if contract.get("code_sha256") != route_contract.code_digest():
        raise ValueError("CED training source differs from the source recorded in the checkpoint")
    if contract.get("recursive_model_source_sha256") != route_contract.recursive_model_source_sha256():
        raise ValueError("recursive model source differs from the source recorded in the checkpoint")
    expected = {
        "route": route, "text_model_contract": spec["text_contract"],
        "text_model_architecture": spec["text_architecture"],
        "text_model_schedule": list(spec["schedule"]), "sample_rate": 32000,
        "audio_policy": "mono_resample_center_crop_right_pad_fixed20_split10",
        "prefix": [126, 1, 256], "answer_tokens": 300, "attention_mask": None,
        "pad_token": "!", "eos_suffix": " <|endoftext|>",
        "forward_dtype": "float32", "autocast": False,
    }
    for key, value in expected.items():
        if key not in contract or contract[key] != value:
            raise ValueError(f"CED training contract mismatch: {key}: {contract.get(key)!r}")
    if (route_contract.TEXT_MODEL_CONTRACT != spec["text_contract"]
            or route_contract.TEXT_MODEL_ARCHITECTURE != spec["text_architecture"]
            or tuple(route_contract.TEXT_MODEL_SCHEDULE) != spec["schedule"]):
        raise ValueError("selected route does not implement the required text architecture")
    if not isinstance(contract.get("recursive_text_checkpoint"), dict) or not contract["recursive_text_checkpoint"]:
        raise ValueError("CED checkpoint has no recursive text checkpoint identity")
    if (type(expected_step) is not int or type(saved.get("total_step")) is not int
            or saved.get("total_step") != expected_step
            or saved.get("num_epochs") != 3 or saved.get("loss_reduction") != TRAINING_LOSS_REDUCTION):
        raise ValueError("CED checkpoint step / epoch horizon / training loss contract mismatch")
    geometry = contract.get("geometry", {})
    expected_geometry = spec["default_geometry"] if expected_geometry is None else expected_geometry
    actual = tuple(geometry.get(key) for key in ("world_size", "microbatch", "gradient_accumulation_steps"))
    if actual != parse_training_geometry(expected_geometry):
        raise ValueError(f"expected training geometry {expected_geometry}, got {actual}")
    if not 0 < expected_step <= geometry.get("total_steps", 0):
        raise ValueError("CED checkpoint step is outside the recorded training horizon")
    encoder = contract.get("audio_encoder", {})
    if any(encoder.get(key) != value for key, value in {
            "name": "CEDSmall", "hidden_cache": False, "frozen": True, "mode": "eval"}.items()):
        raise ValueError("CED evaluation requires the online frozen CEDSmall feature contract")
    if any(contract.get("mapper", {}).get(key) != value for key, value in {
            "name": "freq_merge_126", "hidden_dim": 384, "frequency_bins": 4,
            "output_dim": 576, "resample": False, "dropout": 0.05,
            "max_temporal_tokens": 512, "external_projection": "Identity"}.items()):
        raise ValueError("unsupported CED mapper contract")
    branches = branch_contracts(route_contract, spec)
    for key, branch in branches.items():
        if contract.get(key) != branch:
            raise ValueError(f"CED {key} branch contract mismatch")
    for other in {"audio_cross_attention", "layer_injection", "audio_listen", "audio_reinjection",
                  "prompt_mix", "serial_audio_injection", "loop_lora"} - branches.keys():
        if other in contract:
            raise ValueError(f"unexpected CED branch contract: {other}")
    return contract


def audit_recursive_route(route_contract, contract, model, *, eval_variant):
    spec = variant_spec(eval_variant)
    route_root = Path(route_contract.__file__).resolve().parent
    actual = route_contract.recursive_checkpoint_contract(Path(contract["text_model_dir"]))
    if actual != contract["recursive_text_checkpoint"]:
        raise ValueError("recursive text checkpoint changed since training")
    loader = importlib.import_module("models.recursive_text")
    implementation = importlib.import_module(spec["implementation"])
    for module, expected_path in ((loader, route_root / "models/recursive_text.py"),
                                  (implementation, implementation_path(spec, route_root))):
        if Path(module.__file__).resolve() != expected_path.resolve():
            raise RuntimeError("recursive loader or implementation escaped the selected training route")
    decoder, lm = model.caption_decoder, model.caption_decoder.lm
    if (not isinstance(lm, implementation.RecursiveLlamaForCausalLM)
            or not getattr(decoder, "is_recursive_mesh" if spec["mesh"] else "is_recursive_5_10_5", False)
            or decoder.text_model_contract != spec["text_contract"] or lm.config.hidden_size != HIDDEN_SIZE):
        raise ValueError("loaded decoder does not match the selected hidden576 architecture")
    parameters = implementation.parameter_audit(lm)
    loader.validate_recursive_config(lm.config)
    if hasattr(loader, "validate_loaded_model"):
        loader.validate_loaded_model(lm, parameters)
    core = lm.model
    if (len(core.layers) != 20 or len({id(layer) for layer in core.layers}) != 20
            or parameters.get("logical_layer_count") != len(spec["schedule"])
            or parameters.get("physical_layer_count") != 20
            or parameters.get("recursive_loops") != spec["loops"]
            or parameters.get("schedule") != list(spec["schedule"])):
        raise ValueError("CED physical layers, recursive loops or scheduled executions mismatch")
    if getattr(core, "ablate_after_two_loops", False):
        raise ValueError("evaluation requires the complete trained loop schedule without ablation")
    for name in ("audit_mode", "gradient_audit_mode", "routing_stats_mode"):
        if getattr(core, name, False):
            raise ValueError("inference must retain the ordinary forward without tensor audit modes")
    routers = []
    if spec["mesh"]:
        routers = list(core.write_routers) + list(core.read_routers)
        slots, groups = spec["memory_slots"], spec["loops"] + 1
        if (core.memory_slots != slots or len(core.write_routers) != groups or len(core.read_routers) != groups
                or len({id(r) for r in routers}) != 2 * groups
                or any(tuple(r.weight.shape) != (slots, HIDDEN_SIZE) or tuple(r.bias.shape) != (slots,) for r in routers)
                or parameters.get("memory_persistent") is not False):
            raise ValueError("MeSH transient slots or independent original router geometry mismatch")
        if any("memory_slot" in name or name.endswith(".memory") for name in lm.state_dict()):
            raise ValueError("MeSH memory must not persist in the checkpoint")
    kind = spec["kind"]
    if kind in ("crossattn", "query32"):
        branch = loader.validate_audio_cross_attention(lm)
        extra = [core.audio_crossattn]
        if core.audio_crossattn.collect_diagnostics:
            raise ValueError("cross attention diagnostic mode must remain disabled")
    elif kind == "listen36912":
        branch = loader.validate_audio_listen(lm)
        extra = [core.audio_listen]
        routers += list(core.audio_listen.listen_routers)
        if core.audio_listen.collect_diagnostics:
            raise ValueError("audio listen diagnostic mode must remain disabled")
    elif kind == "mesh_x4":
        branch, extra = {}, []
        if (parameters.get("logical_cache_slot_count") != 50
                or parameters.get("router_objects_independent") is not True):
            raise ValueError("four-loop MeSH requires all 50 logical executions and ten independent routers")
    else:
        if kind in ("inject48", "inject84", "serial84"):
            branch = (loader.validate_serial_injection(lm) if kind == "serial84"
                      else loader.validate_layer_injection(lm))
        elif spec["split_memory"]:
            # Trained router rows/gates must not be checked against initialization.
            branch = loader.validate_audio_listen(lm, fresh=False)
        else:
            branch = loader.validate_audio_listen(lm)
        mlps = model.audio_encoder.base.layer_mlps
        if set(mlps) != {"4", "8"} or mlps["4"] is mlps["8"]:
            raise ValueError("CED4/8 requires two independent audio MLPs")
        extra = [mlps]
        if spec["interaction_layers"]:
            extra.append(core.audio_gates)
        if spec["split_memory"]:
            extra += [core.audio_listen_write_routers, core.audio_listen_read_routers]
            routers += list(core.audio_listen_write_routers) + list(core.audio_listen_read_routers)
        if kind == "listen48":
            extra.append(core.listen_routers)
            routers += list(core.listen_routers.values())
        if kind == "prompt_mix":
            mixer = decoder.prompt_mix_mlp
            if (tuple(mixer[0].weight.shape) != (HIDDEN_SIZE, HIDDEN_SIZE)
                    or tuple(mixer[0].bias.shape) != (HIDDEN_SIZE,)
                    or tuple(mixer[2].weight.shape) != (3, HIDDEN_SIZE)
                    or tuple(mixer[2].bias.shape) != (3,)):
                raise ValueError("prompt mixture requires the trained hidden576/GELU/3 MLP")
            extra.append(mixer)
        if kind == "serial84":
            extra += [core.audio_linear, core.audio_gate]
            if getattr(core, "collect_reinject_stats", True):
                raise ValueError("serial input reinjection diagnostic mode must remain disabled")
        if getattr(core, "collect_injection_stats" if kind in ("inject48", "inject84", "serial84") else "collect_listen_stats", True):
            raise ValueError("CED4/8 diagnostic mode must remain disabled")
    count = sum(p.numel() for module in extra for p in module.parameters())
    lora_audit = None
    if spec.get("loop_lora"):
        lora = importlib.import_module("models.loop_lora")
        if Path(lora.__file__).resolve() != (route_root / "models/loop_lora.py").resolve():
            raise RuntimeError("LoRA implementation escaped the selected training route")
        if tuple(lora.TARGETS) != LOOP_LORA_TARGETS or getattr(core, "gradient_checkpointing", True):
            raise ValueError("evaluation requires all seven LoRA projections without gradient checkpointing")
        # Training validator checks scope, shapes, independent storage, rank/scale
        # and the original trainable W/A/B. Run before the inference-only freeze.
        lora_audit = lora.validate_loop_lora(core)
        lora_count = sum(p.numel() for name, p in core.named_parameters()
                         if ".lora_A." in name or ".lora_B." in name)
        if lora_count != 1628160 or lora_audit["parameter_count"] != lora_count:
            raise ValueError("expected two independent loop LoRAs with 1,628,160 parameters")
        # Count only A/B: the wrapped Linear W/bias still belong to the backbone.
        count += lora_count
    if count != spec["new_parameters"]:
        raise ValueError(f"CED branch parameter count mismatch: {count}")
    if routers and len({id(r) for r in routers}) != len(routers):
        raise ValueError("original and added routers must be independent")
    return {"checkpoint": actual, "parameters": parameters, "branch": branch,
            "loop_lora": lora_audit,
            "implementation": spec["implementation"], "new_parameter_count": count,
            "implementation_sha256": contract["recursive_model_source_sha256"],
            "memory_slots": spec["memory_slots"],
            "audio_memory_slots": spec["audio_memory_slots"],
            "text_memory_slots": spec["text_memory_slots"],
            "interaction_layers": list(spec["interaction_layers"]), "routers": len(routers)}


def install_recursive_trace_audit(lm, *, eval_variant):
    """Observe physical calls and real branch invocations without changing outputs."""
    spec = variant_spec(eval_variant)
    trace, events, compression, router_events, projections = [], [], [], [], []
    lora_calls, lora_modules = [], []
    kind, core = spec["kind"], lm.model
    expected_schedule = list(spec["schedule"])
    layers = spec["interaction_layers"]
    if kind == "serial84":
        expected_events = [[5, "input"], [5, "layer8"], [15, "input"], [15, "layer4"]]
    elif layers:
        expected_events = [[boundary, layer] for boundary, layer in zip((5, 15), layers)]
    else:
        expected_events = [[5, 0], [15, 1]] if kind in ("crossattn", "query32", "listen36912") else []
    audit = {"status": "PENDING", "checked_forwards": 0, "use_cache": False,
             "expected_schedule": expected_schedule,
             "expected_branch_indices": sorted({event[0] for event in expected_events}),
             "audio_memory_slots": spec["audio_memory_slots"],
             "text_memory_slots": spec["text_memory_slots"],
             "memory_policy": "fresh_per_full_sequence_forward" if spec["mesh"] else "none"}

    def check_lora_idle():
        if spec.get("loop_lora") and (not getattr(core, "loop_lora_enabled", False)
                or getattr(core, "gradient_checkpointing", True)
                or any(module.loop_index is not None for _, _, module in lora_modules)):
            raise RuntimeError("LoRA must be enabled, reset between forwards and free of gradient checkpointing")

    def begin(_module, _inputs, kwargs):
        trace.clear()
        events.clear()
        compression.clear()
        router_events.clear()
        projections.clear()
        lora_calls.clear()
        check_lora_idle()
        if (kwargs.get("use_cache") is not False or kwargs.get("past_key_values") is not None
                or kwargs.get("attention_mask") is not None or kwargs.get("labels") is not None):
            raise RuntimeError("CED evaluation requires full recomputation, no KV cache and attention_mask=None")
        if getattr(core, "ablate_after_two_loops", False):
            raise RuntimeError("CED evaluation requires all four trained MeSH loops")

    def record(index):
        def hook(_module, _inputs, _output):
            trace.append(index)
        return hook

    def interaction(_module, inputs, _output):
        round_index = inputs[4] if spec["kind"] == "listen36912" else inputs[3]
        events.append([len(trace), int(round_index)])

    def pair_event(layer):
        def hook(_module, _inputs, _output):
            events.append([len(trace), layer])
        return hook

    def compressed(_module, inputs, _output):
        compression.append([list(inputs[0].shape), list(inputs[1].shape)])

    def projected(_module, inputs, _output):
        projections.append([len(trace), list(inputs[0].shape)])

    def router_event(name):
        def hook(_module, _inputs, _output):
            router_events.append([len(trace), name])
        return hook

    def lora_projection(physical, target):
        def hook(module, inputs):
            logical = len(trace)
            loop = 0 if 5 <= logical < 15 else 1 if 15 <= logical < 25 else None
            if (loop is None or physical != expected_schedule[logical]
                    or module.loop_index != loop):
                raise RuntimeError("LoRA projection used the wrong physical layer or loop adapter")
            lora_calls.append((logical, physical, target, module.loop_index, tuple(inputs[0].shape)))
        return hook

    def finish(module, _inputs, output):
        if trace != expected_schedule:
            raise RuntimeError(f"CED actual physical schedule mismatch: {trace}")
        if output.past_key_values is not None:
            raise RuntimeError("CED evaluation unexpectedly returned a KV cache")
        if spec.get("loop_lora"):
            check_lora_idle()
            expected_lora = [(logical, expected_schedule[logical], target,
                              0 if logical < 15 else 1,
                              (1, int(output.logits.shape[1]), width))
                             for logical in range(5, 25)
                             for target, width in lora_input_widths]
            if sorted(lora_calls) != sorted(expected_lora):
                raise RuntimeError("LoRA requires exactly 140 full-sequence projection calls per forward")
            expected_trace = [(logical, expected_schedule[logical], 0 if logical < 15 else 1)
                              for logical in range(5, 25)]
            if list(core.last_loop_lora_trace) != expected_trace:
                raise RuntimeError("LoRA internal loop trace disagrees with observed projections")
            audit.update(loop_lora_projection_calls=len(lora_calls),
                         loop_lora_calls_per_loop=[sum(call[3] == loop for call in lora_calls) for loop in (0, 1)],
                         last_loop_lora_trace=[list(item) for item in expected_trace], loop_lora_indices_reset=True)
        if events != expected_events:
            raise RuntimeError(f"CED actual interaction order mismatch: {events}")
        if kind == "serial84":
            if ([list(item) for item in core.last_serial_injection_trace] != expected_events
                    or [list(item) for item in core.last_injection_trace] != [[5, 8], [15, 4]]
                    or list(core.last_reinject_trace) != [5, 15]):
                raise RuntimeError("CED serial input/layer injection trace mismatch")
            if projections != [[0, [1, AUDIO_TOKENS, HIDDEN_SIZE]]]:
                raise RuntimeError("serial input audio must be projected once before each full LM forward")
        elif layers:
            internal = getattr(module.model, "last_listen_trace" if spec["kind"] == "listen48" else "last_injection_trace")
            if [list(item) for item in internal] != expected_events:
                raise RuntimeError("CED layer trace disagrees with observed gates")
        elif spec["kind"] in ("crossattn", "query32") and tuple(module.model.last_crossattn_trace) != (5, 15):
            raise RuntimeError("CED cross attention trace mismatch")
        if spec["mesh"]:
            expected_routers = []
            for index in range(spec["loops"] + 1):
                boundary = 5 + 10 * index
                if spec["split_memory"]:
                    expected_routers += [[boundary, f"audio_write_{index}"], [boundary, f"audio_read_{index}"]]
                expected_routers += [[boundary, f"write_{index}"], [boundary, f"read_{index}"]]
                if index < 2 and spec["kind"].startswith("listen"):
                    expected_routers.append([boundary, f"listen_{index}"])
            if router_events != expected_routers:
                raise RuntimeError(f"CED MeSH router invocation order mismatch: {router_events}")
            if spec["kind"] == "listen36912" and core.audio_listen.last_rounds != [0, 1]:
                raise RuntimeError("CED four-depth listen round trace mismatch")
        if spec["kind"] == "query32":
            if compression != [[[4, 32, HIDDEN_SIZE], [4, 126, HIDDEN_SIZE]]]:
                raise RuntimeError(f"query32 must compress four depths independently once per LM forward: {compression}")
        audit.update(status="PASS", checked_forwards=audit["checked_forwards"] + 1,
                     last_sequence_tokens=int(output.logits.shape[1]),
                     last_physical_schedule=list(trace), last_branch_events=copy.deepcopy(events),
                     query_compression=copy.deepcopy(compression),
                     input_audio_projections=copy.deepcopy(projections),
                     last_router_events=copy.deepcopy(router_events))

    if spec.get("loop_lora"):
        for physical in range(5, 15):
            for target in LOOP_LORA_TARGETS:
                projection = core.layers[physical].get_submodule(target)
                lora_modules.append((physical, target, projection))
                projection.register_forward_pre_hook(lora_projection(physical, target))
        # All layers have the same projection geometry; down_proj consumes the
        # intermediate MLP width rather than hidden_size.
        lora_input_widths = [(target, module.in_features) for physical, target, module in lora_modules if physical == 5]
        check_lora_idle()
    lm.register_forward_pre_hook(begin, with_kwargs=True)
    for index, layer in enumerate(lm.model.layers):
        layer.register_forward_hook(record(index))
    if spec["mesh"]:
        for index, router in enumerate(core.write_routers):
            router.register_forward_hook(router_event(f"write_{index}"))
        for index, router in enumerate(core.read_routers):
            router.register_forward_hook(router_event(f"read_{index}"))
        if spec["split_memory"]:
            for index, router in enumerate(core.audio_listen_write_routers):
                router.register_forward_hook(router_event(f"audio_write_{index}"))
            for index, router in enumerate(core.audio_listen_read_routers):
                router.register_forward_hook(router_event(f"audio_read_{index}"))
    if kind in ("crossattn", "query32"):
        core.audio_crossattn.register_forward_hook(interaction)
        if kind == "query32":
            core.audio_crossattn.compression_attention.register_forward_hook(compressed)
    elif kind == "listen36912":
        core.audio_listen.register_forward_hook(interaction)
        for index, router in enumerate(core.audio_listen.listen_routers):
            router.register_forward_hook(router_event(f"listen_{index}"))
    elif spec["interaction_layers"]:
        if kind == "serial84":
            core.audio_gate.register_forward_hook(pair_event("input"))
            core.audio_linear.register_forward_hook(projected)
        for layer in (4, 8):
            core.audio_gates[str(layer)].register_forward_hook(pair_event(f"layer{layer}" if kind == "serial84" else layer))
        if kind == "listen48":
            for index, layer in enumerate((4, 8)):
                core.listen_routers[str(layer)].register_forward_hook(router_event(f"listen_{index}"))
    lm.register_forward_hook(finish)
    return audit


def load_runtime(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    variant_spec(args.eval_variant)
    if not torch.cuda.is_available():
        raise RuntimeError("CED Mizar S1 evaluation requires one CUDA GPU")
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    modules = import_training_route(args.route_root)
    route_contract = modules["contract"]
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    contract = validate_checkpoint(saved, runtime, route_contract, args.expected_step,
                                   args.expected_geometry, eval_variant=args.eval_variant)
    assets = audit_assets(contract)
    model = modules["models.mellow"].Mellow(
        audioenc_name="CEDSmall", d_in=384, d_out=HIDDEN_SIZE,
        text_decoder=contract["text_model_dir"], prefix_length=PREFIX_TOKENS,
        freeze_text_decoder_weights=False, use_pretrained_audioencoder=True,
        freeze_audio_encoder_weights=True, pretrained_audioencoder_path=contract["ced_model_dir"])
    model.float()
    state_dict = saved["state_dict"]
    keys = list(state_dict)
    ddp = bool(keys and all(str(key).startswith("module.") for key in keys))
    if ddp:
        state_dict = {str(key)[len("module."):]: value for key, value in state_dict.items()}
    elif any(str(key).startswith("module.") for key in keys):
        raise ValueError("checkpoint has a mixed DDP state_dict namespace")
    model.load_state_dict(state_dict, strict=True)
    recursive_audit = audit_recursive_route(route_contract, contract, model, eval_variant=args.eval_variant)
    model.to(device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(contract["text_model_dir"], local_files_only=True)
    checkpoint_token_audit = route_contract.validate_checkpoint_tokenizer(tokenizer, model.caption_decoder.lm.config.vocab_size)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    pad_id = tokenizer.encode(tokenizer.pad_token)[0]
    token_audit = route_contract.validate_tokenizer_contract(
        tokenizer, pad_id, model.caption_decoder.lm.get_input_embeddings().num_embeddings)
    token_audit["checkpoint_source"] = checkpoint_token_audit
    if any(module.training for module in model.modules()):
        raise RuntimeError("all CED inference modules must be in eval mode")
    if {p.dtype for p in model.parameters()} != {torch.float32}:
        raise RuntimeError("all CED inference parameters must be float32")
    text_encoder = modules["data.audiotext_dataset"].AudioTextDataset.__new__(modules["data.audiotext_dataset"].AudioTextDataset)
    text_encoder.tokenizer = tokenizer
    trace_audit = install_recursive_trace_audit(model.caption_decoder.lm, eval_variant=args.eval_variant)
    audit = {
        "status": "PASS", "eval_variant": args.eval_variant,
        "checkpoint_step": saved["total_step"], "checkpoint_epoch": saved.get("epoch"),
        "contract_sha256": saved["contract_sha256"], "training_code_sha256": contract["code_sha256"],
        "strict_state_dict": True, "ddp_state_dict_prefix_normalized": ddp,
        "state_dict_source": "training_checkpoint_state_dict", "training_route": saved["route_variant"],
        "training_geometry": contract["geometry"], "training_loss_reduction": saved["loss_reduction"],
        "expected_geometry": list(parse_training_geometry(args.expected_geometry)),
        "tokenizer": token_audit, "assets": assets, "all_modules_eval": True,
        "ced_all_eval": all(not module.training for module in model.audio_encoder.base.ced.modules()),
        "ced_parameters_frozen": all(not p.requires_grad for p in model.audio_encoder.base.ced.parameters()),
        "dtype": "float32", "deterministic_greedy": True, "tf32": False,
        "recursive_audit": recursive_audit, "recursive_forward_audit": trace_audit,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
    }
    del saved, state_dict
    return model, tokenizer, device, text_encoder, modules["data.fixed20"], audit, modules


def sample_context(eval_variant, audio_output, context, text_input):
    """Respect each training prefix API, including four-depth listen's None."""
    spec = variant_spec(eval_variant)
    kind = spec["kind"]
    if spec["context_kind"] == "none":
        if context is not None:
            raise ValueError("four-loop MeSH prefix must not supply an audio interaction context")
        return {}
    if spec["context_kind"] == "seeds":
        return {"audio_memory_seeds": context}
    if spec["context_kind"] == "pair":
        return {"audio_information": context}
    if kind == "listen36912":
        if context is not None:
            raise ValueError("unexpected four-depth listen prefix context")
        return {"audio_grids": audio_output["layer_grids"], "prompt_mask": text_input["attention_mask"]}
    if not isinstance(context, dict) or set(context) != {"audio_grids", "prompt_mask"}:
        raise ValueError("cross attention requires explicit audio_grids and tokenizer prompt_mask")
    return dict(context)


def validate_context(eval_variant, context, prefix):
    import torch

    def tensor(value, shape):
        return (isinstance(value, torch.Tensor) and tuple(value.shape) == shape
                and value.device == prefix.device and value.dtype == prefix.dtype
                and bool(torch.isfinite(value).all()))

    spec = variant_spec(eval_variant)
    if not isinstance(context, dict):
        raise ValueError("explicit per-sample CED context is required")
    if spec["context_kind"] == "none":
        if context:
            raise ValueError("four-loop MeSH uses only the original full-sequence input embeddings")
    elif spec["context_kind"] in ("pair", "seeds"):
        key = "audio_information" if spec["context_kind"] == "pair" else "audio_memory_seeds"
        count = 2 if spec["context_kind"] == "pair" else 3
        pair = context.get(key)
        if (set(context) != {key} or not isinstance(pair, (tuple, list)) or len(pair) != count
                or any(not tensor(value, (1, AUDIO_TOKENS, HIDDEN_SIZE)) for value in pair)):
            raise ValueError(f"CED context requires this sample's {count} finite aligned FP32 tensors in {key}")
    else:
        if (set(context) != {"audio_grids", "prompt_mask"}
                or not tensor(context["audio_grids"], (1, 4, AUDIO_TOKENS, 1536))):
            raise ValueError("four-depth CED context requires finite aligned [1,4,126,1536] grids")
        mask = context["prompt_mask"]
        if (not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (1, PROMPT_TOKENS)
                or mask.device != prefix.device or not bool(((mask == 0) | (mask == 1)).all())
                or not bool((mask.sum(dim=1) > 0).all())):
            raise ValueError("prompt_mask must be the nonempty binary tokenizer fixed256 mask")
    return context


def greedy_decode(model, tokenizer, prefix, max_new_tokens=MAX_NEW_TOKENS, *, eval_variant, context):
    import torch

    if max_new_tokens != MAX_NEW_TOKENS:
        raise ValueError("both benchmarks use exactly max_new_tokens=300")
    if tuple(prefix.shape) != (1, PREFIX_TOKENS, HIDDEN_SIZE) or prefix.dtype != torch.float32:
        raise ValueError("CED prefix shape/dtype differs from the FP32 training layout")
    forward_kwargs = validate_context(eval_variant, context, prefix)
    generated_ids, generated = [], prefix
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        output = model.caption_decoder.lm(
            inputs_embeds=generated, attention_mask=None, labels=None, use_cache=False,
            **forward_kwargs)
        logits = output.logits[:, -1, :]
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite CED generation logits")
        next_id = int(torch.argmax(logits, dim=-1).item())
        generated_ids.append(next_id)
        del output, logits
        if next_id == int(tokenizer.eos_token_id):
            stop_reason = "eos_token"
            break
        ids = torch.tensor([[next_id]], device=prefix.device, dtype=torch.long)
        generated = torch.cat((generated, model.caption_decoder.lm.get_input_embeddings()(ids)), dim=1)
    text = tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {"generated_token_ids": generated_ids, "generated_text": text,
            "generated_text_raw": text, "generated_token_count": len(generated_ids),
            "stop_reason": stop_reason, "max_new_tokens": max_new_tokens}
