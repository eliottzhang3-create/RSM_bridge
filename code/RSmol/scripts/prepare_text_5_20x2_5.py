#!/usr/bin/env python3
"""Optional manual model-path check; the formal launcher does not call this.

Kept for existing commands. Never scans/tokenizes the dataset or starts DDP.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def trainer_module():
    spec = importlib.util.spec_from_file_location("text_5_20x2_5_preflight", SCRIPTS / "train_stage4_5_20x2_5_ddp.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, help="Accepted for old commands; no audit is run")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args(argv)
    trainer = trainer_module()
    output = trainer.ensure_external_output(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty training output: {output}")
    model_dir = args.resume_from or args.model_dir
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if (raw.get("text_route") != trainer.TEXT_ROUTE or raw.get("num_hidden_layers") != 50
            or raw.get("recursive_layer_count") != 30 or raw.get("recursive_loops") != 2
            or raw.get("logical_to_physical") != list(trainer.LOGICAL_TO_PHYSICAL)
            or raw.get("recursive_source_layer_indices_0based") != list(range(30))
            or raw.get("architectures") != ["RecursiveLlama5_20x2_5ForCausalLM"]):
        raise ValueError("Model config is not the isolated pure-text 5-20x2-5 conversion")
    if args.resume_from:
        trainer.ensure_external_resume(args.resume_from)
        for name in ("training_state.pt", "checkpoint_complete.json"):
            if not (model_dir / name).is_file():
                raise FileNotFoundError(model_dir / name)
        marker = json.loads((model_dir / "checkpoint_complete.json").read_text(encoding="utf-8"))
        if marker.get("complete") is not True or marker.get("mode") != "formal" or marker.get("world_size") != 8:
            raise ValueError("Resume requires a complete eight-rank FORMAL checkpoint")
        print("[preflight] Complete checkpoint selected; trainer will verify saved recipe/state/manifest", flush=True)
        return 0
    metadata = json.loads((model_dir / "recursive_conversion_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "PASS" or metadata.get("text_route") != trainer.TEXT_ROUTE:
        raise ValueError("Successful conversion metadata required for fresh training")
    print("[preflight] Model config checked; dataset audit disabled. Formal launcher submits directly.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
