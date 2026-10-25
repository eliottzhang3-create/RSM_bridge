"""Write the fully resolved, immutable recipe after staging on the compute node."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from contract import (ROUTE, digest, recursive_checkpoint_contract,
                      semantic_contract, write_json)
from data.staging import StagingGate


def file_sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def asset_identity(text_root, htsat_root):
    text_root, htsat_root = Path(text_root).resolve(), Path(htsat_root).resolve()
    weights = sorted(
        path for path in text_root.rglob("*")
        if path.is_file() and path.suffix in {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
        and ("model" in path.name or "pytorch" in path.name or path.suffix == ".safetensors")
    )
    if not weights:
        raise FileNotFoundError(f"no local language model weights in {text_root}")
    htsat = htsat_root / "HTSAT_AudioSet_Saved_1.ckpt"
    if not htsat.is_file():
        raise FileNotFoundError(htsat)
    text_files = sorted(set(weights) | {
        path for path in text_root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".txt", ".model"}
    })
    return {"text": [{"name": p.relative_to(text_root).as_posix(),
                       "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in text_files],
            "htsat": {"name": htsat.name, "bytes": htsat.stat().st_size, "sha256": file_sha256(htsat)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("formal",), required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--text-model-dir", required=True, type=Path)
    parser.add_argument("--htsat-root", required=True, type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    if prepared.get("mode") != args.mode:
        raise ValueError("staging mode differs from the requested run mode")
    staging = StagingGate(prepared)
    if staging.info:
        staging.wait_for_batch(staging.info["epoch"], staging.info["start_batch"])
    recursive_text = recursive_checkpoint_contract(args.text_model_dir)
    contract = semantic_contract(prepared, args.text_model_dir.resolve(), args.htsat_root.resolve())
    contract["recursive_text_checkpoint"] = recursive_text
    contract["assets"] = asset_identity(args.text_model_dir, args.htsat_root)
    contract["packages"] = {name: importlib.metadata.version(name) for name in
                            ("torch", "torchaudio", "transformers", "numpy", "soundfile", "torchlibrosa")}
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
              "intentional_differences": [
                  "only text decoder initialization changes from SmolLM2 to the validated 5-10x4-5 MeSH recursive checkpoint",
                  "HTSAT(129+129) plus existing c2l/projection instead of CED freq_merge_126",
                  "shared raw audio decoded online, collective switch to staged bytes after full copy; no CED cache",
                  "8 GPUs x microbatch4 x accumulation4; global batch128 after microbatch reduction for OOM",
                  "hybrid 1477492-row mixture; MusicCaps and all SpeechCraft splits retained; LibriTTS clean100/360 plus LibriTTS-R train-other-500"]}
    write_json(args.output, config)
    print(json.dumps({"config": str(args.output), "mode": args.mode, "contract_sha256": contract_sha,
                      "geometry": contract["geometry"]}), flush=True)


if __name__ == "__main__":
    main()
