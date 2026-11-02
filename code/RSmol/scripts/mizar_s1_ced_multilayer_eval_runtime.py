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
    ("recursive_mesh", "layer4_8_audio7slot_split3_inject", 1, None, "audio7split3_inject", 2469164),
    ("recursive_mesh", "layer8_4_inject", 1, None, "inject84", 2444930),
):
    _variant = f"{_family}_{_suffix}"
    _mesh = _family == "recursive_mesh"
    ROUTE_SPECS[_variant] = {
        "route": f"ced_small_mizar_s1_fixed20_gbs256_{_variant}_v{_version}",
        "route_root": TRAINING_ROOT / f"mizar_stage1_fixed20_gbs256_3epochs_{_family}_ced_small_{_suffix}",
        "run_root": (REMOTE_OUTPUT_ROOT / f"mizar_s1_ced_small_gbs256_{_variant}_5090" / f"formal_{_stamp}"
                     if _stamp else None),
        "kind": _kind, "mesh": _mesh, "new_parameters": _count,
        "context_kind": "pair" if _kind in ("inject48", "inject84", "listen48", "audio7split3", "audio7split3_inject") else "grids",
        "audio_memory_slots": 7 if _kind.startswith("audio7") else 5 if _mesh else 0,
        "interaction_layers": ((8, 4) if _kind == "inject84" else (4, 8)
                               if _kind in ("inject48", "listen48", "audio7split3_inject") else ()),
        "implementation": ("models.recursive_layer_inject" if _kind in ("inject48", "inject84") else
                           "models.recursive_mesh_listen" if _mesh else "models.recursive_crossattn"),
        "text_contract": "logical_30_physical_20_5_10x2_5" if _mesh else "logical_30_physical_20_5_10_5_loops_2_no_mesh",
        "text_architecture": "logical_30_physical_20_5_10x2_5_mesh" if _mesh else "logical_30_physical_20_5_10_5_loops_2_no_mesh",
    }


def variant_spec(eval_variant):
    if eval_variant not in ROUTE_SPECS:
        raise ValueError(f"unsupported multilayer CED evaluation variant: {eval_variant!r}")
    return ROUTE_SPECS[eval_variant]


def branch_contract(route_contract, spec):
    kind = spec["kind"]
    if kind in ("crossattn", "query32"):
        return "audio_cross_attention", route_contract.cross_attention_contract()
    if kind in ("inject48", "inject84"):
        return "layer_injection", dict(route_contract.INJECTION_CONTRACT)
    if kind in ("listen48", "audio7split3", "audio7split3_inject"):
        return "audio_listen", dict(route_contract.LISTEN_CONTRACT)
    return "audio_listen", route_contract.audio_listen_contract()


def validate_checkpoint(saved, runtime, route_contract, expected_step,
                        expected_geometry=(8, 8, 4), *, eval_variant):
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
        "text_model_schedule": list(TEXT_MODEL_SCHEDULE), "sample_rate": 32000,
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
            or tuple(route_contract.TEXT_MODEL_SCHEDULE) != TEXT_MODEL_SCHEDULE):
        raise ValueError("selected route does not implement the required text architecture")
    if not isinstance(contract.get("recursive_text_checkpoint"), dict) or not contract["recursive_text_checkpoint"]:
        raise ValueError("CED checkpoint has no recursive text checkpoint identity")
    if (type(expected_step) is not int or type(saved.get("total_step")) is not int
            or saved.get("total_step") != expected_step
            or saved.get("num_epochs") != 3 or saved.get("loss_reduction") != TRAINING_LOSS_REDUCTION):
        raise ValueError("CED checkpoint step / epoch horizon / training loss contract mismatch")
    geometry = contract.get("geometry", {})
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
    key, branch = branch_contract(route_contract, spec)
    if contract.get(key) != branch:
        raise ValueError(f"CED {key} branch contract mismatch")
    for other in {"audio_cross_attention", "layer_injection", "audio_listen", "audio_reinjection"} - {key}:
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
    for module, relative in ((loader, "models/recursive_text.py"),
                             (implementation, spec["implementation"].replace(".", "/") + ".py")):
        if Path(module.__file__).resolve() != route_root / relative:
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
            or parameters.get("logical_layer_count") != 30
            or parameters.get("physical_layer_count") != 20
            or parameters.get("recursive_loops") != 2
            or parameters.get("schedule") != list(TEXT_MODEL_SCHEDULE)):
        raise ValueError("CED requires 20 distinct physical layers and 30 scheduled executions")
    for name in ("audit_mode", "gradient_audit_mode", "routing_stats_mode"):
        if getattr(core, name, False):
            raise ValueError("inference must retain the ordinary forward without tensor audit modes")
    routers = []
    if spec["mesh"]:
        routers = list(core.write_routers) + list(core.read_routers)
        if (core.memory_slots != 5 or len(core.write_routers) != 3 or len(core.read_routers) != 3
                or len({id(r) for r in routers}) != 6
                or any(tuple(r.weight.shape) != (5, HIDDEN_SIZE) or tuple(r.bias.shape) != (5,) for r in routers)
                or parameters.get("memory_persistent") is not False):
            raise ValueError("MeSH requires five transient slots and six independent original routers")
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
    else:
        if kind in ("inject48", "inject84"):
            branch = loader.validate_layer_injection(lm)
        elif spec["audio_memory_slots"] == 7:
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
        if spec["audio_memory_slots"] == 7:
            extra += [core.audio_listen_write_routers, core.audio_listen_read_routers]
            routers += list(core.audio_listen_write_routers) + list(core.audio_listen_read_routers)
        if kind == "listen48":
            extra.append(core.listen_routers)
            routers += list(core.listen_routers.values())
        if getattr(core, "collect_injection_stats" if kind in ("inject48", "inject84") else "collect_listen_stats", True):
            raise ValueError("CED4/8 diagnostic mode must remain disabled")
    count = sum(p.numel() for module in extra for p in module.parameters())
    if count != spec["new_parameters"]:
        raise ValueError(f"CED branch parameter count mismatch: {count}")
    if routers and len({id(r) for r in routers}) != len(routers):
        raise ValueError("original and added routers must be independent")
    return {"checkpoint": actual, "parameters": parameters, "branch": branch,
            "implementation": spec["implementation"], "new_parameter_count": count,
            "implementation_sha256": contract["recursive_model_source_sha256"],
            "memory_slots": 5 if spec["mesh"] else 0,
            "audio_memory_slots": spec["audio_memory_slots"],
            "text_memory_slots": 5 if spec["mesh"] else 0,
            "interaction_layers": list(spec["interaction_layers"]), "routers": len(routers)}


def install_recursive_trace_audit(lm, *, eval_variant):
    """Observe physical calls and real branch invocations without changing outputs."""
    spec = variant_spec(eval_variant)
    trace, events, compression, router_events = [], [], [], []
    audit = {"status": "PENDING", "checked_forwards": 0, "use_cache": False,
             "expected_schedule": list(TEXT_MODEL_SCHEDULE),
             "expected_branch_indices": [] if spec["kind"] == "audio7split3" else [5, 15],
             "audio_memory_slots": spec["audio_memory_slots"],
             "text_memory_slots": 5 if spec["mesh"] else 0,
             "memory_policy": "fresh_per_full_sequence_forward" if spec["mesh"] else "none"}

    def begin(_module, _inputs, kwargs):
        trace.clear()
        events.clear()
        compression.clear()
        router_events.clear()
        if (kwargs.get("use_cache") is not False or kwargs.get("past_key_values") is not None
                or kwargs.get("attention_mask") is not None):
            raise RuntimeError("CED evaluation requires full recomputation, no KV cache and attention_mask=None")

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

    def router_event(name):
        def hook(_module, _inputs, _output):
            router_events.append([len(trace), name])
        return hook

    def finish(module, _inputs, output):
        if trace != list(TEXT_MODEL_SCHEDULE):
            raise RuntimeError(f"CED actual physical schedule mismatch: {trace}")
        if output.past_key_values is not None:
            raise RuntimeError("CED evaluation unexpectedly returned a KV cache")
        layers = spec["interaction_layers"]
        expected = ([[boundary, layer] for boundary, layer in zip((5, 15), layers)] if layers else
                    [] if spec["kind"] == "audio7split3" else [[5, 0], [15, 1]])
        if events != expected:
            raise RuntimeError(f"CED actual interaction order mismatch: {events}")
        if layers:
            internal = getattr(module.model, "last_listen_trace" if spec["kind"] == "listen48" else "last_injection_trace")
            if [list(item) for item in internal] != expected:
                raise RuntimeError("CED layer trace disagrees with observed gates")
        elif spec["kind"] in ("crossattn", "query32") and tuple(module.model.last_crossattn_trace) != (5, 15):
            raise RuntimeError("CED cross attention trace mismatch")
        if spec["mesh"]:
            expected_routers = []
            for boundary, index in ((5, 0), (15, 1), (25, 2)):
                if spec["audio_memory_slots"] == 7:
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
                     last_router_events=copy.deepcopy(router_events))

    lm.register_forward_pre_hook(begin, with_kwargs=True)
    for index, layer in enumerate(lm.model.layers):
        layer.register_forward_hook(record(index))
    kind, core = spec["kind"], lm.model
    if spec["mesh"]:
        for index, router in enumerate(core.write_routers):
            router.register_forward_hook(router_event(f"write_{index}"))
        for index, router in enumerate(core.read_routers):
            router.register_forward_hook(router_event(f"read_{index}"))
        if spec["audio_memory_slots"] == 7:
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
        for layer in (4, 8):
            core.audio_gates[str(layer)].register_forward_hook(pair_event(layer))
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
    if spec["context_kind"] == "pair":
        pair = context.get("audio_information")
        if (set(context) != {"audio_information"} or not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(not tensor(value, (1, AUDIO_TOKENS, HIDDEN_SIZE)) for value in pair)):
            raise ValueError("CED4/8 requires this sample's finite FP32 (A4,A8) in original order")
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
