"""Write the fully resolved, immutable recipe after staging on the compute node."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from contract import ROUTE, digest, semantic_contract, validate_usad2_assets, write_json
from data.staging import StagingGate


def file_sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def asset_identity(text_root, usad2_root):
    text_root, usad2_root = Path(text_root).resolve(), Path(usad2_root).resolve()
    weights = sorted(set(text_root.glob("*.safetensors")) | set(text_root.glob("pytorch_model*.bin")))
    if not weights:
        raise FileNotFoundError(f"no local language model weights in {text_root}")
    validation = validate_usad2_assets(usad2_root)
    required = validation["files"]
    text_files = sorted(set(weights) | set(text_root.glob("*.json")) | set(text_root.glob("*.txt"))
                        | set(text_root.glob("*.model")))
    return {
        "text": [{"name": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in text_files],
        "usad2_small": {
            **validation,
            "files": [{"name": name, "bytes": (usad2_root / name).stat().st_size,
                       "sha256": file_sha256(usad2_root / name)} for name in required],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("formal",), required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--text-model-dir", required=True, type=Path)
    parser.add_argument("--usad2-model-dir", required=True, type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    if prepared.get("mode") != args.mode:
        raise ValueError("staging mode differs from the requested run mode")
    staging = StagingGate(prepared)
    if staging.info:
        staging.wait_for_batch(staging.info["epoch"], staging.info["start_batch"])
    contract = semantic_contract(prepared, args.text_model_dir.resolve(), args.usad2_model_dir.resolve())
    contract["assets"] = asset_identity(args.text_model_dir, args.usad2_model_dir)
    contract["packages"] = {name: importlib.metadata.version(name) for name in
                            ("torch", "torchaudio", "transformers", "safetensors", "numpy", "soundfile")}
    contract["python"] = platform.python_version()
    contract_sha = digest(contract)
    config = {"route": ROUTE, "mode": args.mode, "contract": contract, "contract_sha256": contract_sha,
              "prepared_report": str(args.prepared.resolve()), "run_dir": str(args.run_dir.resolve()),
              "max_optimizer_steps": contract["geometry"]["total_steps"],
              "resume_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else "",
              "staging": {key: prepared[key] for key in
                          ("read_policy", "index_dir", "initial_steps", "startup_budgets", "planned_files", "planned_bytes", "copy_workers")
                          if key in prepared},
              "checkpoint_steps": [round(contract["geometry"]["total_steps"] * i / 6) for i in range(1, 7)],
              "official_reference": "Mizar_159M/configs/s1.yaml.in + runtime/stage1 + pipeline/build_ced_cache.py",
              "intentional_differences": ["frozen USAD2-Small 384-d output; mean global token + Mellow 384->576 projection + valid-frame temporal mean groups of 4",
                   "one fixed20 waveform and one USAD2 call: 999 encoder frames -> 251 audio tokens -> 508-token prefix",
                   "shared raw audio decoded online, collective switch to staged bytes after full copy; no CED cache",
                   "8 GPUs x microbatch8 x accumulation4; global batch256 ablation",
                   "hybrid 1477492-row mixture; MusicCaps and all SpeechCraft splits retained; LibriTTS clean100/360 plus LibriTTS-R train-other-500"]}
    write_json(args.output, config)
    print(json.dumps({"config": str(args.output), "mode": args.mode, "contract_sha256": contract_sha,
                      "geometry": contract["geometry"]}), flush=True)


if __name__ == "__main__":
    main()
