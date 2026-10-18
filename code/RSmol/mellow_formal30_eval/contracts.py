"""Dependency-light contracts for the unified Mellow audio evaluator."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RSMOL = ROOT.parent
ROUTE_PARENT = RSMOL / "mellow_official_training_c8204d8_adamw_cosine"
SHARED_ROUTE = ROUTE_PARENT / "reasonaqa_global_batch256_lr_ablation_5epochs"
BASELINE_ROUTE = "reasonaqa_global_batch256_formal_30epochs"
LR_ABLATION_ROUTE = "reasonaqa_global_batch256_lr_ablation_10epochs"
LR_ABLATION_ROUTE_VARIANT = "formal_10epochs_lr_ablation_random_audio2"
PROTOCOL_VERSION = "compact_eos_reused_audio2_fp32_raw_dual_score_v5_10choice"
MAX_PROMPT_TOKENS = 129
MAX_NEW_TOKENS = 300
AUDIO_PREFIX_TOKENS = 260
MAX_CHOICES = 10
HIDDEN_SIZE = 576
SEED = 0
POLICIES = {
    "route_variant": "formal_30epochs_compact",
    "attention_mask_policy": "compact_text_right_pad_after_answer_eos",
    "htsat_freeze_policy": "eval_no_grad_base_only",
    "audio_slot_policy": "zero_if_missing_audio2",
    "gradient_clip_policy": "global_l2_hard_clip",
    "max_grad_norm": 0.5,
}
LR_ABLATION_POLICIES = {
    **POLICIES,
    "route_variant": LR_ABLATION_ROUTE_VARIANT,
    "audio_slot_policy": "random_fill_if_missing_audio2",
}
ARCHITECTURES = {
    "smollm2": {
        "route": BASELINE_ROUTE, "route_contract": None, "text_model_contract": None,
        "physical_layers": 30, "logical_layers": 30, "loops": 1,
        "memory_slots": 0, "routers": 0,
    },
    "mesh_x2": {
        "route": BASELINE_ROUTE + "_5_10x2_5_mesh",
        "route_contract": "mellow_formal30_gbs256_5_10x2_5_mesh_v1",
        "text_model_contract": "logical_30_physical_20_5_10x2_5",
        "physical_layers": 20, "logical_layers": 30, "loops": 2,
        "memory_slots": 5, "routers": 6,
    },
    "mesh_x4": {
        "route": BASELINE_ROUTE + "_5_10x4_5_mesh",
        "route_contract": "mellow_formal30_gbs256_5_10x4_5_mesh_v1",
        "text_model_contract": "logical_50_physical_20_5_10x4_5",
        "physical_layers": 20, "logical_layers": 50, "loops": 4,
        "memory_slots": 7, "routers": 10,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def source_fingerprints(paths: Sequence[Path]) -> dict[str, str]:
    """Pin executed Python semantics, without pinning shell/node/staging metadata."""
    result = {}
    for path in sorted(set(paths)):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        semantic = ast.dump(tree, include_attributes=False)
        result[path.relative_to(RSMOL).as_posix()] = hashlib.sha256(semantic.encode()).hexdigest()
    return result


def checkpoint_aliases() -> dict[str, str]:
    return dict(line.split("\t", 1) for line in
                (ROOT / "checkpoints.tsv").read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#"))


def _require(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(f"{field}: expected {expected!r}, got {actual!r}")


def route_for_architecture(architecture: str, route_variant: str | None = None) -> str:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture: {architecture}")
    if route_variant == LR_ABLATION_ROUTE_VARIANT:
        if architecture != "smollm2":
            raise ValueError("the 10-epoch LR-ablation route only produces ordinary SmolLM2 checkpoints")
        return LR_ABLATION_ROUTE
    if route_variant not in (None, POLICIES["route_variant"]):
        raise ValueError(f"unsupported training route variant: {route_variant!r}")
    return ARCHITECTURES[architecture]["route"]


def validate_contract(checkpoint: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[str, dict]:
    """No staging, hostname, PID, RNG, optimizer restoration or source-init re-audit."""
    route_variant = checkpoint.get("route_variant")
    if route_variant == LR_ABLATION_ROUTE_VARIANT:
        policies, expected_epochs, flexible_lr = LR_ABLATION_POLICIES, 10, True
    elif route_variant == POLICIES["route_variant"]:
        policies, expected_epochs, flexible_lr = POLICIES, 30, False
    else:
        raise ValueError(f"unsupported Mellow training route: {route_variant!r}")
    for field, expected in {"schema_version": 2, "num_epochs": expected_epochs,
                            "loss_reduction": "global_token_mean", **policies}.items():
        _require(checkpoint.get(field), expected, f"checkpoint.{field}")
    epoch_completed = int(checkpoint.get("epoch_completed", 0))
    if not 1 <= epoch_completed <= expected_epochs:
        raise ValueError(f"checkpoint.epoch_completed must be in [1, {expected_epochs}], got {epoch_completed}")
    for field in policies:
        if field not in {"gradient_clip_policy", "max_grad_norm"}:
            _require(config.get(field), policies[field], f"runtime.{field}")
    if not isinstance(checkpoint.get("state_dict"), Mapping) or not checkpoint["state_dict"]:
        raise ValueError("checkpoint.state_dict must be a nonempty mapping")
    _require(checkpoint.get("batch_geometry"), {
        "per_rank_batch_size": 8, "world_size": 8, "gradient_accumulation_steps": 4,
    }, "checkpoint.batch_geometry")
    identity = (checkpoint.get("route_contract"), checkpoint.get("text_model_contract"))
    matches = [name for name, spec in ARCHITECTURES.items()
               if identity == (spec["route_contract"], spec["text_model_contract"])]
    if len(matches) != 1:
        raise ValueError(f"unrecognized Mellow text-backbone contract: {identity!r}")
    architecture = matches[0]
    if flexible_lr and architecture != "smollm2":
        raise ValueError("the 10-epoch LR-ablation checkpoint must contain ordinary SmolLM2 weights")
    spec = ARCHITECTURES[architecture]
    for field in ("route_contract", "text_model_contract"):
        _require(config.get(field), spec[field], f"runtime.{field}")
    state = checkpoint["state_dict"]
    layer_prefix = "caption_decoder.lm.model.layers."
    layers = {int(key[len(layer_prefix):].split(".")[0])
              for key in state if key.startswith(layer_prefix)}
    _require(layers, set(range(spec["physical_layers"])), "checkpoint physical layers")
    routers = set()
    for family in ("write_routers", "read_routers"):
        prefix = f"caption_decoder.lm.model.{family}."
        routers.update((family, int(key[len(prefix):].split(".")[0]))
                       for key in state if key.startswith(prefix))
    _require(len(routers), spec["routers"], "checkpoint router count")
    optimizer = checkpoint.get("optimizer_contract") or {}
    _require(optimizer.get("type"), "AdamW", "checkpoint optimizer")
    _require(list(optimizer.get("betas", ())), [0.9, 0.95], "checkpoint optimizer betas")
    _require(optimizer.get("weight_decay"), 1e-4, "checkpoint optimizer weight decay")
    scheduler = checkpoint.get("scheduler") or {}
    _require(scheduler.get("scheduler_type"), "step_cosine_warmup", "checkpoint.scheduler.scheduler_type")
    try:
        max_lr, min_lr = float(scheduler["max_lr"]), float(scheduler["min_lr"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint scheduler must contain numeric max_lr and min_lr") from exc
    if not math.isfinite(max_lr) or not math.isfinite(min_lr) or max_lr <= 0.0 or min_lr <= 0.0:
        raise ValueError("checkpoint scheduler learning rates must be positive and finite")
    if flexible_lr:
        if not math.isclose(min_lr, 0.1 * max_lr, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"10-epoch checkpoint min_lr must equal 0.1 * max_lr, got {max_lr}, {min_lr}")
    else:
        _require(max_lr, 4e-5, "checkpoint.scheduler.max_lr")
        _require(min_lr, 4e-6, "checkpoint.scheduler.min_lr")
    total = int(scheduler.get("total_steps", 0))
    current_step = int(checkpoint.get("total_step", -1))
    if (total <= 0 or total % expected_epochs or not 1 <= current_step <= total
            or scheduler.get("last_step") != current_step
            or scheduler.get("warmup_steps") != math.ceil(total * 0.05)):
        raise ValueError(f"incomplete {expected_epochs}-epoch scheduler: {scheduler!r}")
    train = config.get("train") or {}
    for field, expected in {"batch_size": 8, "gradient_accumulation_steps": 4,
                            "num_epochs": expected_epochs, "gradient_clip_policy": policies["gradient_clip_policy"],
                            "max_grad_norm": 0.5}.items():
        _require(train.get(field), expected, f"runtime.train.{field}")
    _require((train.get("mixed_precision") or {}).get("use_mixed_precision"), False,
             "runtime mixed precision")
    opt = train.get("optimizer") or {}
    for field, expected in {"optimizer_type": "AdamW", "betas": [0.9, 0.95],
                            "weight_decay": 1e-4, "warmup_ratio": 0.05,
                            "scheduler": "step_cosine_warmup"}.items():
        _require(opt.get(field), expected, f"runtime.optimizer.{field}")
    runtime_max_lr = opt.get("max_lr")
    runtime_min_lr = opt.get("min_lr")
    runtime_learning_rate = opt.get("learning_rate")
    for field, value in (("max_lr", runtime_max_lr), ("min_lr", runtime_min_lr),
                         ("learning_rate", runtime_learning_rate)):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"runtime.optimizer.{field} must be numeric") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"runtime.optimizer.{field} must be finite")
    if not (math.isclose(float(runtime_max_lr), max_lr, rel_tol=1e-12, abs_tol=0.0)
            and math.isclose(float(runtime_learning_rate), max_lr, rel_tol=1e-12, abs_tol=0.0)
            and math.isclose(float(runtime_min_lr), min_lr, rel_tol=1e-12, abs_tol=0.0)):
        raise ValueError("runtime and checkpoint learning-rate bounds differ")
    if not flexible_lr and (runtime_max_lr != 4e-5 or runtime_min_lr != 4e-6):
        raise ValueError("formal30 runtime learning-rate bounds changed")
    data = config.get("data") or {}
    for field, expected in {"sampling_rate": 32000, "segment_seconds": 10,
                            "ip_text_len": 129, "op_text_len": 250}.items():
        _require(data.get(field), expected, f"runtime.data.{field}")
    model = config.get("model") or {}
    _require(model.get("model_type"), "Mellow", "runtime model_type")
    encoder, decoder = model.get("encoder") or {}, model.get("decoder") or {}
    for field, expected in {"audioenc_name": "HTSAT", "out_emb": 768, "d_proj": 576,
                            "freeze_audio_encoder_weights": True,
                            "use_pretrained_audioencoder": True}.items():
        _require(encoder.get(field), expected, f"runtime.encoder.{field}")
    for field, expected in {"prefix_length": 40, "total_prefix_length": 260,
                            "freeze_gpt_weights": False}.items():
        _require(decoder.get(field), expected, f"runtime.decoder.{field}")
    # Only fields needed to construct/identify the model participate in resume.
    # Training datapath/datafiles point at deleted per-job /dev/shm staging.
    model_contract = {"architecture": architecture, "encoder": dict(encoder),
                      "decoder": dict(decoder), "tokenizer": data.get("tokenizer_type"),
                      "training_steps": total, "training_epochs": expected_epochs,
                      "training_route_variant": route_variant,
                      "max_lr": max_lr, "min_lr": min_lr, "policies": dict(policies)}
    return architecture, model_contract


def build_prompt(question: str, choices: Sequence[Any]) -> str:
    if not 2 <= len(choices) <= MAX_CHOICES:
        raise ValueError(f"expected 2..{MAX_CHOICES} choices, got {len(choices)}")
    # Preserve body case and punctuation; only add ordered, lowercase labels.
    return str(question).strip() + " " + " ".join(
        f"{chr(97 + index)}) {str(choice).strip()}" for index, choice in enumerate(choices))


def prepare_official_prediction(value: Any) -> str:
    """Pass model text through without stripping labels, punctuation or whitespace."""
    return str(value)


def protocol(task: str) -> dict[str, Any]:
    if task not in {"mmau_test_mini", "mmar"}:
        raise ValueError(f"unknown task: {task}")
    return {
        "version": PROTOCOL_VERSION, "task": task, "seed": SEED,
        "dtype": "float32", "tf32": False, "batch_size": 1,
        "sample_rate": 32000, "audio_seconds": 10, "audio_prefix_tokens": 260,
        "audio2": "reuse_audio1_waveform_encoded_independently",
        "audio1": ("official_id_wav_preferred_else_parquet_embedded; training_first_two_channel_mean; mellow_repeat_or_per_id_seeded_random_crop" if task == "mmau_test_mini"
                   else "mono_first10s_right_zero_pad"),
        "prompt_case": "preserve_question_and_choice_bodies",
        "choice_labels": "fixed_order_lowercase_a_to_j_close_paren",
        "prompt": "route_tokenize_text_padding_false_force_final_eos",
        "max_prompt_tokens_including_eos": MAX_PROMPT_TOKENS,
        "prompt_padding": False, "attention_mask": "all_real_prefix_and_prompt_tokens_valid",
        "max_new_tokens": MAX_NEW_TOKENS, "top_p": 0.8, "temperature": 1.0,
        "do_sample": False, "use_cache": False,
        "decoder": "top_p_filter_then_argmax_full_recompute_stop_on_generated_eos",
        "choice_label_score": "raw_prediction.split(')')[0].lower_exact_no_trim",
        "prediction_preprocessing": "none",
        "official_score": "raw_generated_text_unchanged_to_official_evaluation_py",
        "expected_rows": 1000,
    }
