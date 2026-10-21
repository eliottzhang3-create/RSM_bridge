"""Write the fully resolved, immutable recipe after staging on the compute node."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from contract import ROUTE, digest, rank_seed, semantic_contract, write_json


def file_sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def asset_identity(text_root, htsat_root):
    text_root, htsat_root = Path(text_root).resolve(), Path(htsat_root).resolve()
    weights = sorted(set(text_root.glob("*.safetensors")) | set(text_root.glob("pytorch_model*.bin")))
    if not weights:
        raise FileNotFoundError(f"no local language model weights in {text_root}")
    htsat = htsat_root / "HTSAT_AudioSet_Saved_1.ckpt"
    if not htsat.is_file():
        raise FileNotFoundError(htsat)
    text_files = sorted(set(weights) | set(text_root.glob("*.json")) | set(text_root.glob("*.txt"))
                        | set(text_root.glob("*.model")))
    return {"text": [{"name": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in text_files],
            "htsat": {"name": htsat.name, "bytes": htsat.stat().st_size, "sha256": file_sha256(htsat)}}


def validate_smoke(report, contract_sha):
    if (report.get("status") != "PASS" or report.get("mode") != "smoke"
            or report.get("optimizer_steps") != 8 or report.get("contract_sha256") != contract_sha
            or len(report.get("rank_checks", [])) != 8):
        raise ValueError("formal run requires a PASS eight-update smoke report matching current code/data/assets/recipe")
    checks = report["rank_checks"]
    if {check.get("rank") for check in checks} != set(range(8)):
        raise ValueError("smoke report does not cover all eight distinct ranks")
    for check in checks:
        changes = check.get("trainable_weight_changes", {})
        if (check.get("optimizer_steps") != 8 or check.get("seed") != rank_seed(check["rank"])
                or check.get("frozen_htsat_equal") is not True
                or check.get("parameters_finite") is not True or check.get("htsat_all_eval") is not True
                or len(check.get("first8_rows", [])) != 32
                or set(check.get("gradients", {})) != {"1", "8"}
                or set(changes) != {"c2l", "projection", "lm"}
                or not all(value.get("changed") is True for value in changes.values())):
            raise ValueError(f"smoke rank {check.get('rank')} did not pass the complete update/freeze checks")
        for gradients in check["gradients"].values():
            if set(gradients) != {"c2l", "projection", "lm"} or not all(
                    x.get("finite") is True and x.get("nonzero_parameters", 0) > 0 for x in gradients.values()):
                raise ValueError("smoke report is missing finite nonzero component gradients")
    checkpoint = report.get("checkpoint", {})
    if (checkpoint.get("status") != "PASS" or checkpoint.get("optimizer_steps") != 8
            or checkpoint.get("model_tensors_equal") is not True or checkpoint.get("rng_ranks") != 8):
        raise ValueError("smoke checkpoint did not pass its full-state reload audit")
    if report.get("layout", {}).get("total") != 815 or report["layout"].get("prefix") != 515:
        raise ValueError("smoke layout differs from HTSAT Mizar S1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--text-model-dir", required=True, type=Path)
    parser.add_argument("--htsat-root", required=True, type=Path)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    if prepared.get("status") != "PASS" or prepared.get("mode") != args.mode:
        raise ValueError("staging did not complete for the requested run mode")
    contract = semantic_contract(prepared, args.text_model_dir.resolve(), args.htsat_root.resolve())
    contract["assets"] = asset_identity(args.text_model_dir, args.htsat_root)
    contract["packages"] = {name: importlib.metadata.version(name) for name in
                            ("torch", "torchaudio", "transformers", "numpy", "soundfile", "torchlibrosa")}
    contract["python"] = platform.python_version()
    contract_sha = digest(contract)
    if args.mode == "formal":
        if not args.smoke_report:
            parser.error("formal requires --smoke-report from this recipe's completed eight-update smoke")
        validate_smoke(json.loads(args.smoke_report.read_text(encoding="utf-8")), contract_sha)
    if args.resume_checkpoint and args.mode != "formal":
        parser.error("smoke is always a fresh eight-update run; resume is only for this formal route")
    config = {"route": ROUTE, "mode": args.mode, "contract": contract, "contract_sha256": contract_sha,
              "prepared_report": str(args.prepared.resolve()), "run_dir": str(args.run_dir.resolve()),
              "max_optimizer_steps": 8 if args.mode == "smoke" else contract["geometry"]["total_steps"],
              "resume_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else "",
              "smoke_report": str(args.smoke_report.resolve()) if args.smoke_report else "",
              "checkpoint_steps": [round(contract["geometry"]["total_steps"] * i / 6) for i in range(1, 7)],
              "official_reference": "Mizar_159M/configs/s1.yaml.in + runtime/stage1 + pipeline/build_ced_cache.py",
              "intentional_differences": ["HTSAT(129+129) plus existing c2l/projection instead of CED freq_merge_126",
                   "raw audio staged then decoded online instead of hidden.f16 feature cache",
                   "8 GPUs x microbatch4 x accumulation1; same global batch32",
                   "filtered 1311334-row mixture; original LibriTTS clean100/360 replaces LibriTTS-R"]}
    write_json(args.output, config)
    print(json.dumps({"config": str(args.output), "mode": args.mode, "contract_sha256": contract_sha,
                      "geometry": contract["geometry"]}), flush=True)


if __name__ == "__main__":
    main()
