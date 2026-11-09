"""Dependency-light identity for Mizar final -> recurrent text-only training."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUTE = "mizar_text_5_10x2_5_gbs1024_v1"
ARCHITECTURE = "logical_30_physical_20_5_10x2_5_loops_2"
SOURCE_LAYERS = (0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29)
SCHEDULE = tuple(range(15)) + tuple(range(5, 20))
DEFAULT_SOURCE = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M")
DEFAULT_MODEL = DEFAULT_SOURCE.with_name("Mizar159M-text-5-10x2-5")
DEFAULT_OUTPUT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_5_10x2_5_gbs1024")
LM_PREFIX = "caption_decoder.lm."


def release_manifest():
    return json.loads((HERE.parent / "mizar_official_eval/release_manifest.json").read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity():
    release = release_manifest()
    return {"checkpoint_relative_path": release["checkpoint"]["path"],
            "checkpoint_sha256": release["checkpoint"]["sha256"],
            "release": copy.deepcopy(release["release"]), "lm_prefix": LM_PREFIX}


def build_target_config(source_config):
    if source_config.num_hidden_layers != 30:
        raise ValueError("Mizar text source must have 30 physical layers")
    result = copy.deepcopy(source_config)
    fields = {
        "num_hidden_layers": 30, "recursive_layer_count": 20, "recursive_loops": 2,
        "recursive_loops_scope": "middle_only", "recursive_prefix_layer_count": 5,
        "recursive_middle_layer_count": 10, "recursive_suffix_layer_count": 5,
        "middle_recurrent_count": 10, "recursive_mapping_policy": "explicit_5_10_5_source_layers",
        "recursive_prefix_source_layers_1based": [1, 2, 3, 4, 5],
        "recursive_middle_source_layers_1based": [6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        "recursive_suffix_source_layers_1based": [26, 27, 28, 29, 30],
        "architectures": ["RecursiveLlamaForCausalLM"], "text_route": ROUTE,
        "mizar_text_source": source_identity(),
    }
    for name in ("logical_to_physical", "recursive_logical_to_physical",
                 "logical_to_physical_schedule", "recursive_logical_to_physical_schedule"):
        fields[name] = list(SCHEDULE)
    for suffix, values in (("indices_0based", SOURCE_LAYERS), ("mapping_0based", SOURCE_LAYERS),
                           ("indices_1based", [i + 1 for i in SOURCE_LAYERS]),
                           ("mapping_1based", [i + 1 for i in SOURCE_LAYERS])):
        fields["recursive_source_layer_" + suffix if suffix.startswith("indices")
               else "recursive_source_" + suffix] = list(values)
    for name, value in fields.items():
        setattr(result, name, value)
    return result


def validate_model_config(raw):
    expected = {"text_route": ROUTE, "architectures": ["RecursiveLlamaForCausalLM"],
                "num_hidden_layers": 30, "recursive_layer_count": 20,
                "recursive_loops": 2, "recursive_loops_scope": "middle_only",
                "recursive_prefix_layer_count": 5, "recursive_middle_layer_count": 10,
                "recursive_suffix_layer_count": 5,
                "recursive_source_layer_indices_0based": list(SOURCE_LAYERS),
                "logical_to_physical": list(SCHEDULE), "hidden_size": 576, "vocab_size": 49152,
                "tie_word_embeddings": True, "mizar_text_source": source_identity()}
    for name, value in expected.items():
        if raw.get(name) != value:
            raise ValueError(f"Mizar text model contract mismatch: {name}")


def inspect_initialization(path, *, resume=False):
    """Read JSON identity only: weights-only initialization never loads training_state.pt."""
    path = Path(path).resolve()
    raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
    validate_model_config(raw)
    trained = (path / "checkpoint_complete.json").is_file()
    if trained:
        marker = json.loads((path / "checkpoint_complete.json").read_text(encoding="utf-8"))
        step = marker.get("optimizer_step")
        if (marker.get("complete") is not True or marker.get("mode") != "formal" or marker.get("text_route") != ROUTE
                or marker.get("architecture") != ARCHITECTURE or marker.get("world_size") != 8
                or marker.get("data_cursors_by_rank") != [str(i) for i in range(8)]
                or type(step) is not int or not 0 <= step <= 9244
                or path.name != f"checkpoint-step-{step:06d}"
                or not (path / "training_state.pt").is_file()):
            raise ValueError("Incomplete or wrong-route Mizar text training checkpoint")
        if not resume and step != 9244:
            raise ValueError("New epoch weights-only initialization requires the final step-009244 checkpoint")
    elif resume:
        raise ValueError("Full resume requires a complete training checkpoint")
    else:
        marker = json.loads((path / "conversion_complete.json").read_text(encoding="utf-8"))
        if (marker.get("complete") is not True or marker.get("text_route") != ROUTE
                or marker.get("source") != source_identity()
                or marker.get("physical_layers") != 20 or marker.get("logical_layers") != 30):
            raise ValueError("Incomplete or wrong-route Mizar text conversion")
        step = None
    if not any(path.glob("*.safetensors")) and not (path / "pytorch_model.bin").is_file():
        raise FileNotFoundError(f"Model weights missing: {path}")
    if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise FileNotFoundError(f"Tokenizer missing: {path}")
    return {"mode": "full_resume" if resume else "weights_only_new_epoch",
            "model_path": str(path), "source_training_step": step,
            "start_optimizer_step": step if resume else 0,
            "restore_optimizer_scheduler_rng_cursor": bool(resume),
            "fresh_seed": None if resume else 0, "source": source_identity()}
