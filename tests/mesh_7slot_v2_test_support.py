"""Synthetic checkpoint metadata for dependency-light audio contract checks."""
import json
from pathlib import Path


def final_text_fixture(root):
    root = Path(root)
    architecture = "logical_30_physical_20_5_10x2_5_mesh_7slot_v2_3router"
    policy = {"parameter_dtype": "float32", "optimizer_state_dtype": "float32",
              "autocast_dtype": "bfloat16", "loss_dtype": "float32",
              "router_softmax_dtype": "float32", "grad_scaler_enabled": False,
              "tf32_enabled": False}
    def write(name, value):
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    metadata = dict(architecture_contract=architecture, optimizer_step=9244,
                    training_contract="text_mesh_x2_7slot_v2_bf16_full_epoch_v1",
                    precision="bf16", precision_policy=policy, steps_per_epoch=9244,
                    epochs=1, memory_slots=7, router_groups=3, router_module_count=6,
                    router_parameters_in_optimizer=True,
                    logical_to_physical=[*range(15), *range(5, 15), *range(15, 20)])
    write("mesh_checkpoint_metadata.json", metadata)
    write("data_manifest.json", {"architecture_contract": architecture, "optimizer_step": 9244})
    (root / "training_state.pt").write_bytes(b"synthetic-text-state-never-loaded-by-audio")
    for name in ("model.safetensors", "tokenizer.json"):
        if not (root / name).exists():
            (root / name).write_bytes(b"synthetic-fixture")
    names = sorted(["config.json", "model.safetensors", "tokenizer.json", "training_state.pt",
                    "mesh_checkpoint_metadata.json", "data_manifest.json",
                    "checkpoint_manifest.json", "checkpoint_complete.json"])
    manifest = dict(status="complete", architecture_contract=architecture,
                    optimizer_step=9244, files=names)
    write("checkpoint_manifest.json", manifest)
    write("checkpoint_complete.json", dict(manifest, complete_marker=True))
