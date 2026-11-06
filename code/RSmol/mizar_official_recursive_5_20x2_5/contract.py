"""Dependency-free, fixed conversion and future-training contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RSMOL = HERE.parent
RELEASE_FILE = RSMOL / "mizar_official_eval/release_manifest.json"
DEFAULT_SOURCE = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M")
DEFAULT_OUTPUT = DEFAULT_SOURCE.with_name("Mizar159M-5-20x2-5-serial84-looplora-r8")
ROUTE = "mizar_official_final_5_20x2_5_serial84_loop_lora_r8_gbs128_v1"
SCHEMA = "mizar_recursive_model_only_initialization_v1"
SCHEDULE = tuple(range(25)) + tuple(range(5, 25)) + tuple(range(25, 30))
SINGLE_PASS_SCHEDULE = tuple(range(30))
MIDDLE_LAYERS = tuple(range(5, 25))
INJECTION_INDICES = (5, 25)
SERIAL_TRACE = ((5, "input"), (5, "layer8"), (25, "input"), (25, "layer4"))
TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
RANK, ALPHA, LOOPS, AUDIO_TOKENS = 8, 16, 2, 126
BASE_PARAMETERS = 159299840
LORA_PARAMETERS, AUDIO_PARAMETERS = 3256320, 2777859
TOTAL_PARAMETERS = BASE_PARAMETERS + LORA_PARAMETERS + AUDIO_PARAMETERS
NEW_TENSORS = 560 + 24
TOKENIZER_FILES = ("config.json", "generation_config.json", "merges.txt",
                   "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json")


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
                          + "\n", encoding="utf-8", newline="\n")


def implementation_identity():
    files = {p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
             for p in sorted(HERE.glob("*.py"))}
    for name in ("LICENSE", "NOTICE", "licenses/Mizar-ClearBSD.txt"):
        files[name] = hashlib.sha256((HERE / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    files["../mizar_official_eval/release_manifest.json"] = sha256_file(RELEASE_FILE)
    return files


def architecture():
    return {"physical_layers": 30, "logical_layers": 50, "loops": 2,
            "prelude": [0, 4], "middle": [5, 24], "coda": [25, 29],
            "schedule": list(SCHEDULE), "source_layer_mapping": list(range(30)),
            "memory_slots": 0, "routers": 0, "hidden_size": 576,
            "lora": {"rank": RANK, "alpha": ALPHA, "scale": 2., "dropout": 0.,
                     "targets": list(TARGETS), "independent_loops": 2,
                     "physical_layers": list(MIDDLE_LAYERS), "positions": "full_sequence",
                     "parameters": LORA_PARAMETERS, "matrices": 560,
                     "A_init": "kaiming_uniform_a_sqrt5", "B_init": "zeros"},
            "injection": {"indices": list(INJECTION_INDICES), "ced_blocks": [8, 4],
                          "trace": [list(x) for x in SERIAL_TRACE], "audio_tokens": AUDIO_TOKENS,
                          "input_source": "original_final_mapper_audio_embeddings",
                          "input_projection_and_gate_shared": True,
                          "input_projection_once_per_lm_forward": True,
                          "ced_gate_condition": "post_input_residual_hidden",
                          "gate_initialization": "zero_weight_and_bias",
                          "parameters": AUDIO_PARAMETERS},
            "use_cache": False, "gradient_checkpointing": False}


def training_intent():
    """Record user choices; conversion is not a completed/resumable train run."""
    return {"implemented": False, "rows": 1477492, "epochs": 3, "world_size": 8,
            "microbatch": 4, "accumulation": 4, "global_batch": 128,
            "steps_per_epoch": 11542, "total_steps": 34626, "warmup_steps": 1731,
            "warmup_rounding": "round_5pct_like_reference", "precision": "float32",
            "lr": {"recurrent_base": 6e-4, "new_parameters": 6e-4,
                   "mapper": 3e-4, "other_inherited_lm": 3e-4},
            "ced": "frozen_eval", "optimizer_state": "not_in_initialization_artifact"}


def validate_source_text_config(config):
    expected = {"model_type": "llama", "num_hidden_layers": 30, "hidden_size": 576,
                "intermediate_size": 1536, "num_attention_heads": 9,
                "num_key_value_heads": 3, "vocab_size": 49152, "tie_word_embeddings": True}
    problems = {key: (config.get(key), value) for key, value in expected.items()
                if config.get(key) != value}
    if problems:
        raise ValueError(f"official source text configuration mismatch: {problems}")


def build_text_config(source):
    import copy
    value = copy.deepcopy(source)
    validate_source_text_config(value)
    value.update(num_hidden_layers=50, recursive_layer_count=30, recursive_loops=2,
                 recursive_prefix_layer_count=5, recursive_middle_layer_count=20,
                 recursive_suffix_layer_count=5, logical_to_physical=list(SCHEDULE),
                 recursive_source_layer_indices_0based=list(range(30)),
                 recursive_loops_scope="middle_only", use_cache=False,
                 torch_dtype="float32", architectures=["MizarRecursiveForCausalLM"])
    return value


def validate_config_preservation(value, source):
    """Keep the exact persistence contract; report which top-level fields differ."""
    expected = build_text_config(source)
    if value != expected:
        extra = sorted(set(value) - set(expected))
        missing = sorted(set(expected) - set(value))
        changed = sorted(key for key in set(value) & set(expected) if value[key] != expected[key])
        raise ValueError("recursive config must preserve every original source config field; "
                         f"extra={extra}; missing={missing}; changed={changed}")


def validate_text_config(value, *, production=True):
    expected = {"num_hidden_layers": 50, "recursive_layer_count": 30,
                "recursive_loops": 2, "recursive_prefix_layer_count": 5,
                "recursive_middle_layer_count": 20, "recursive_suffix_layer_count": 5,
                "logical_to_physical": list(SCHEDULE), "use_cache": False,
                "recursive_loops_scope": "middle_only",
                "recursive_source_layer_indices_0based": list(range(30))}
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("requires the exact logical50 / physical30 / 5+20x2+5 contract")
    if production:
        source = dict(value, num_hidden_layers=30)
        validate_source_text_config(source)


def parameter_role(name):
    if name.startswith("audio_encoder.base.ced."):
        return "frozen_ced", None
    if name.startswith("audio_encoder.base.mapper."):
        return "mapper", 3e-4
    if name.startswith("audio_encoder.base.layer_mlps."):
        return "new_audio", 6e-4
    if name.startswith("caption_decoder.lm.model.audio_"):
        return "new_audio", 6e-4
    if name.startswith("caption_decoder.lm.model.layers."):
        parts = name.split(".")
        if not parts[4].isdigit() or not 0 <= int(parts[4]) < 30:
            raise ValueError(f"invalid physical layer parameter: {name}")
        middle = int(parts[4]) in MIDDLE_LAYERS
        if ".lora_" in name:
            if (not middle or len(parts) != 9 or ".".join(parts[5:7]) not in TARGETS
                    or parts[-2] not in ("lora_A", "lora_B") or parts[-1] not in ("0", "1")):
                raise ValueError(f"invalid loop LoRA parameter: {name}")
            return "new_lora", 6e-4
        return ("recurrent_base", 6e-4) if middle else ("outer_lm", 3e-4)
    if name.startswith(("caption_decoder.lm.model.embed_tokens.",
                        "caption_decoder.lm.model.norm.", "caption_decoder.lm.lm_head.")):
        return "other_inherited_lm", 3e-4
    raise ValueError(f"unclassified model parameter: {name}")
