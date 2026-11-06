"""Read-only handoff from completed seven-slot v2 text training; no Torch import."""
import json
from pathlib import Path

from contract import TEXT_MODEL_CONTRACT, TEXT_MODEL_SCHEDULE

TEXT_TRAINING_CONTRACT = "text_mesh_x2_7slot_v2_bf16_full_epoch_v1"
TEXT_PRECISION_POLICY = {
    "parameter_dtype": "float32", "optimizer_state_dtype": "float32",
    "autocast_dtype": "bfloat16", "loss_dtype": "float32",
    "router_softmax_dtype": "float32", "grad_scaler_enabled": False,
    "tf32_enabled": False,
}


def final_text_checkpoint(checkpoint):
    """Validate final text provenance without restoring any text training state.

    Model/config/tokenizer content hashes are subsequently bound by
    write_config.asset_identity. The full audio resume contract includes both.
    """
    root = Path(checkpoint).expanduser().resolve(strict=True)

    def read(name):
        try:
            value = json.loads((root / name).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"final seven-slot v2 text checkpoint missing/invalid {name}: {root}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"text checkpoint {name} must contain an object")
        return value

    marker = read("checkpoint_complete.json")
    manifest = read("checkpoint_manifest.json")
    metadata = read("mesh_checkpoint_metadata.json")
    data = read("data_manifest.json")
    for value in (marker, manifest, metadata, data):
        if (value.get("architecture_contract") != TEXT_MODEL_CONTRACT
                or value.get("optimizer_step") != 9244):
            raise ValueError("audio initialization requires this route's completed 9244-step text checkpoint")
    if (marker.get("status") != "complete" or marker.get("complete_marker") is not True
            or manifest.get("status") != "complete"):
        raise ValueError("text checkpoint must have complete markers")
    expected = {
        "training_contract": TEXT_TRAINING_CONTRACT, "precision": "bf16",
        "precision_policy": TEXT_PRECISION_POLICY, "steps_per_epoch": 9244,
        "epochs": 1, "memory_slots": 7, "router_groups": 3,
        "router_module_count": 6, "router_parameters_in_optimizer": True,
        "logical_to_physical": list(TEXT_MODEL_SCHEDULE),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"final text checkpoint contract mismatch: {key}")
    names = manifest.get("files")
    required = {"config.json", "training_state.pt", "mesh_checkpoint_metadata.json",
                "data_manifest.json", "checkpoint_manifest.json", "checkpoint_complete.json"}
    if (not isinstance(names, list) or not all(isinstance(name, str) for name in names)
            or len(names) != len(set(names)) or not required.issubset(names)
            or marker.get("files") != names):
        raise ValueError("text checkpoint file manifests disagree or omit required files")
    for name in names:
        path = root / name
        if (Path(name).name != name or not path.is_file() or path.stat().st_size == 0
                or path.resolve().parent != root):
            raise ValueError(f"text checkpoint file missing/empty/outside checkpoint: {name}")
    return {"training_contract": TEXT_TRAINING_CONTRACT, "optimizer_step": 9244,
            "precision_policy": TEXT_PRECISION_POLICY,
            "initialization": "full_lm_trained_routers_tokenizer_only",
            "text_optimizer_scheduler_rng_cursor_restored": False}
