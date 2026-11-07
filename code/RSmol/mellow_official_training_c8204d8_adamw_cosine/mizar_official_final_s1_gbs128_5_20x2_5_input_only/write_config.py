"""Resolve the training recipe after metadata readiness, without waiting for audio copy."""
import argparse
import json
from pathlib import Path

from contract import (ROUTE, artifact_identity, checkpoint_steps, digest, resolved_contract,
                      validate_ced_assets, validate_resume_payload, write_json)
from data.staging import StagingGate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("formal",), required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    if prepared.get("mode") != args.mode or prepared.get("read_policy") != "shared_until_complete":
        raise ValueError("requires this run's shared-until-complete staging metadata")
    staging = StagingGate(prepared)
    if not staging.info:
        raise ValueError("formal job requires a current background staging producer")
    staging.wait_for_batch(staging.info["epoch"], staging.info["start_batch"])
    artifact = artifact_identity(args.model_dir)
    validate_ced_assets(args.model_dir / "ced-small")
    contract = resolved_contract(prepared, args.model_dir, artifact)
    config = {"route": ROUTE, "mode": args.mode, "contract": contract, "contract_sha256": digest(contract),
              "prepared_report": str(args.prepared.resolve()), "run_dir": str(args.run_dir.resolve()),
              "max_optimizer_steps": contract["geometry"]["total_steps"],
              "resume_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else "",
              "checkpoint_steps": checkpoint_steps(),
              "staging": {key: prepared[key] for key in
                          ("read_policy", "index_dir", "planned_files", "planned_bytes", "copy_workers")},
              "official_reference": "Mizar S1; official final initialization; input-only 5+20x2+5",
              "implementation": "independent trainer; existing conversion package and artifacts immutable"}
    expected_cursor = (0, 0)
    if args.resume_checkpoint:
        if args.resume_checkpoint.resolve().parent != args.run_dir.resolve() / "checkpoints":
            raise ValueError("full resume must use the original output root and its formal/checkpoints state")
        import torch
        saved = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        _, epoch, batch = validate_resume_payload(saved, config)
        expected_cursor = (epoch, batch)
        del saved
    if (staging.info["epoch"], staging.info["start_batch"]) != expected_cursor:
        raise ValueError("staging and checkpoint sampler cursors differ")
    write_json(args.output, config)
    print(json.dumps({"event": "CONFIG_READY", "config": str(args.output),
                      "contract_sha256": config["contract_sha256"], "geometry": contract["geometry"]}), flush=True)


if __name__ == "__main__":
    main()
