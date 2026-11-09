"""Evaluation-only loading for official-final S1 30-physical/50-call Mizar.

Keep the training and conversion sources immutable. In particular, importing
the GPU-safe training runtime must never call the CPU conversion dependencies()
entry point, which intentionally hides CUDA. Decoding/scoring stay shared with
the original CED evaluator.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


def import_training_route(route_root):
    root = Path(route_root).resolve()
    if not (root / "runtime.py").is_file():
        raise FileNotFoundError(root / "runtime.py")
    for name in list(sys.modules):
        if any(name == parent or name.startswith(parent + ".")
               for parent in ("models", "data", "utils", "contract", "runtime")):
            del sys.modules[name]
    sys.path.insert(0, str(root))
    rsmol = root.parents[1]
    if str(rsmol) not in sys.path:
        sys.path.append(str(rsmol))
    importlib.invalidate_caches()
    names = ("contract", "runtime", "data.fixed20", "data.audiotext_dataset")
    modules = {name: importlib.import_module(name) for name in names}
    for name, module in modules.items():
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f"official training import escaped selected route: {name}")
    return modules


def validate_checkpoint(saved, runtime, route, expected_step, expected_geometry, spec):
    """Accept completed checkpoints while retaining the full training identity."""
    from mizar_s1_ced_eval_runtime import parse_training_geometry, TRAINING_LOSS_REDUCTION

    contract = saved.get("contract", {})
    if (route.ROUTE != spec["route"] or saved.get("schema_version") != 1
            or saved.get("route_variant") != route.ROUTE or saved.get("mode") != "formal"
            or not saved.get("state_dict") or saved.get("num_epochs") != 3
            or saved.get("loss_reduction") != TRAINING_LOSS_REDUCTION):
        raise ValueError("requires this official recursive S1 route's complete formal checkpoint")
    identity = route.digest(contract)
    if (saved.get("contract_sha256") != identity or runtime.get("contract_sha256") != identity
            or runtime.get("contract") != contract or runtime.get("mode") != "formal"
            or runtime.get("route") != route.ROUTE):
        raise ValueError("official checkpoint/runtime contract SHA or route mismatch")
    if (contract.get("code_sha256") != route.code_digest()
            or contract.get("recursive_model_source_sha256") != route.recursive_model_source_sha256()):
        raise ValueError("official training or shared recursive source changed since training")
    geometry = contract.get("geometry", {})
    expected_geometry = spec["default_geometry"] if expected_geometry is None else expected_geometry
    actual = tuple(geometry.get(key) for key in ("world_size", "microbatch", "gradient_accumulation_steps"))
    if actual != parse_training_geometry(expected_geometry):
        raise ValueError("official checkpoint training geometry mismatch")
    if (type(expected_step) is not int or type(saved.get("total_step")) is not int
            or saved["total_step"] != expected_step or not 0 < expected_step <= geometry.get("total_steps", 0)):
        raise ValueError("official checkpoint step mismatch or outside training horizon")
    # Rebuild the semantic recipe without reading/staging the training dataset.
    # This checks architecture, family contracts, preprocessing and all geometry,
    # including the trained mapper's inherited-official-final identity.
    prepared = {"rows": geometry.get("rows"), "source_rows": contract.get("source_rows"),
                "rows_canonical_sha256": contract.get("rows_canonical_sha256"),
                "mapping_sha256": contract.get("mapping_sha256"), "read_policy": "shared_until_complete"}
    if not contract.get("model_dir") or not contract.get("initialization_artifact"):
        raise ValueError("official checkpoint has no complete initialization artifact identity")
    expected = route.semantic_contract(prepared, contract["model_dir"], contract["initialization_artifact"])
    for key, value in expected.items():
        if key not in contract or contract[key] != value:
            raise ValueError(f"official training contract mismatch: {key}")
    if (contract["text_model_contract"] != spec["text_contract"]
            or contract["architecture"].get("schedule") != list(spec["schedule"])):
        raise ValueError("official 30-physical/50-call architecture mismatch")
    return contract


def audit_assets(route, contract):
    actual = route.artifact_identity(contract["model_dir"])
    if actual != contract["initialization_artifact"]:
        raise ValueError("official initialization artifact changed since training")
    return {"status": "PASS", "manifest_sha256": actual["manifest_sha256"],
            "verified_files": len(actual["files"]), "initialization_only": True,
            "evaluated_weights": "training_checkpoint_state_dict"}


def audit_model(route, contract, model, spec):
    root = Path(route.__file__).resolve().parent
    runtime = importlib.import_module("runtime")
    implementation = importlib.import_module(spec["implementation"])
    input_only = spec["kind"] == "official_input_only"
    expected = root.parents[1] / (spec["implementation"].replace(".", "/") + ".py")
    model_class = implementation.MizarInputOnlyRecursive if input_only else implementation.MizarRecursive
    if (Path(runtime.__file__).resolve() != root / "runtime.py"
            or Path(implementation.__file__).resolve() != expected.resolve()
            or not isinstance(model, model_class)):
        raise RuntimeError("official recursive loader or model escaped selected implementation")
    # Run before the evaluation freeze; the original audit checks training
    # roles but does not require learned B/gates to retain zero initialization.
    report = runtime.model_audit(model)
    core = model.caption_decoder.lm.model
    if (tuple(core.logical_to_physical) != spec["schedule"]
            or report.get("parameter_count") != (159632769 if input_only else 165334019)
            or len(model.state_dict()) != (458 if input_only else 1038)):
        raise ValueError("official complete state or schedule structure mismatch")
    if not input_only and any(report["loop_lora"].get(key) != value for key, value in spec["loop_lora_config"].items()):
        raise ValueError("official loop LoRA structure mismatch")
    return {"parameters": report, "implementation": spec["implementation"],
            "implementation_sha256": contract["recursive_model_source_sha256"],
            "new_parameter_count": spec["new_parameters"], "memory_slots": 0,
            "audio_memory_slots": 0, "text_memory_slots": 0,
            "interaction_layers": list(spec["interaction_layers"]), "routers": 0}
