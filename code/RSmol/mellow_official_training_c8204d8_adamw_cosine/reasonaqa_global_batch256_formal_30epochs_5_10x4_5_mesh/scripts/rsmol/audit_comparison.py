#!/usr/bin/env python3
"""Audit formal-job inputs or the final checkpoint; never run training."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comparison import (audit_runtime_config, audit_source_equivalence, make_identity,
                        validate_comparison_identity, validate_resume_identity, write_json)
from route_contract import LOGICAL_TO_PHYSICAL, sha256_file


def audit_recursive_checkpoint(checkpoint):
    states = checkpoint["state_dict"]
    layers = {int(k.split(".layers.")[1].split(".")[0]) for k in states
              if k.startswith("caption_decoder.lm.model.layers.")}
    if layers != set(range(20)):
        raise ValueError("final checkpoint has wrong recursive physical layer indices")
    for kind in ("write", "read"):
        prefix = f"caption_decoder.lm.model.{kind}_routers."
        expected = {f"{prefix}{i}.{field}" for i in range(5) for field in ("weight", "bias")}
        if {key for key in states if key.startswith(prefix)} != expected:
            raise ValueError(f"final x4 checkpoint requires five complete {kind} routers")
        for key in expected:
            shape = (7, 576) if key.endswith(".weight") else (7,)
            if tuple(states[key].shape) != shape:
                raise ValueError(f"wrong seven-slot router shape: {key}")
    audit = checkpoint.get("initialization_audit", {}).get("lm", {}).get("parameter_audit", {})
    for field, expected in (("logical_layer_count", 50), ("physical_layer_count", 20),
                            ("recursive_loops", 4), ("logical_cache_slot_count", 50),
                            ("memory_slots", 7), ("router_count", 10)):
        if audit.get(field) != expected:
            raise ValueError(f"final checkpoint x4 initialization audit mismatch: {field}")
    if tuple(audit.get("schedule", ())) != LOGICAL_TO_PHYSICAL:
        raise ValueError("final checkpoint x4 execution schedule mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--mapping-jsonl", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.sources_only:
        # Run before copying audio. No ML imports, runtime config or stage tree.
        report = audit_source_equivalence()
        if args.output_report:
            write_json(args.output_report, report)
        print(json.dumps({"status": "PASS", "scope": "protected_training_python",
                          "provenance_only_changes": report["provenance_only_changes"]}), flush=True)
        return
    if not args.runtime_config or not args.output_report:
        parser.error("--runtime-config and --output-report are required unless --sources-only")
    import yaml
    config = yaml.safe_load(args.runtime_config.read_text(encoding="utf-8"))
    report = {"status": "PASS", "runtime": audit_runtime_config(config),
              "source_equivalence": audit_source_equivalence(),
              "comparison_identity": config["comparison_identity"],
              "runtime_config_sha256": sha256_file(args.runtime_config)}
    if args.checkpoint:
        import torch
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        validate_resume_identity(checkpoint, config)
        if checkpoint.get("epoch_completed") != 30 or checkpoint.get("num_epochs") != 30:
            raise ValueError("expected a completed 30-epoch checkpoint")
        if checkpoint.get("initialization_audit", {}).get("reference_audio_and_cpu_rng") != "exact_match":
            raise ValueError("missing exact baseline initialization comparison")
        states = checkpoint["state_dict"]
        if any(v.is_floating_point() and v.dtype != torch.float32 for v in states.values()):
            raise ValueError("final model contains non-FP32 floating tensors")
        audit_recursive_checkpoint(checkpoint)
        report.update(checkpoint=str(args.checkpoint), total_step=checkpoint["total_step"],
                      initialization=checkpoint["initialization_audit"])
    else:
        validate_comparison_identity(config["comparison_identity"], make_identity())
        if not args.audit_report or not args.mapping_jsonl:
            raise ValueError("startup audit requires the baseline data audit and mapping")
        report["data_inputs"] = {str(p): sha256_file(p) for p in
                                  (args.audit_report, args.mapping_jsonl,
                                   Path(config["data"]["datafiles"][0]))}
        htsat = Path(config["model"]["encoder"]["pretrained_audioencoder_path"]) / "HTSAT_AudioSet_Saved_1.ckpt"
        report["htsat_sha256"] = sha256_file(htsat)
    write_json(args.output_report, report)
    print(json.dumps({"status": "PASS", "report": str(args.output_report)}), flush=True)


if __name__ == "__main__":
    main()
