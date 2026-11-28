"""Resolve this job's exact data/source/optimizer identity after staging starts."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from assets import audit_bundle
from contract import ROUTE, checkpoint_steps, semantic_contract, digest, write_json


def build_contract(prepared, model_dir):
    from prepare_index import tokenizer_identity
    if prepared["text_audit"]["tokenizer_files"] != tokenizer_identity(Path(model_dir)/"assets/smollm2-135m"):
        raise ValueError("training tokenizer differs from the full-pool text audit")
    contract = semantic_contract(prepared, Path(model_dir).resolve(), audit_bundle(model_dir))
    contract["packages"] = {name:importlib.metadata.version(name) for name in
                            ("torch","torchaudio","transformers","numpy","soundfile","safetensors")}
    if contract["packages"]["transformers"] != "4.54.1":
        raise ValueError("this route requires Transformers4.54.1; do not modify the shared environment")
    contract["python"] = platform.python_version()
    return contract


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--resume-checkpoint", type=Path)
    a = p.parse_args()
    prepared = json.loads(a.prepared.read_text(encoding="utf-8"))
    contract = build_contract(prepared,a.model_dir)
    total = contract["geometry"]["total_steps"]
    config = {"route":ROUTE,"mode":"formal","contract":contract,"contract_sha256":digest(contract),
              "prepared_report":str(a.prepared.resolve()),"run_dir":str(a.run_dir.resolve()),
              "resume_checkpoint":str(a.resume_checkpoint.resolve()) if a.resume_checkpoint else "",
              "max_optimizer_steps":total,"checkpoint_steps":checkpoint_steps(prepared["rows"]),
              "attempt_id":str(time.time_ns())}
    write_json(a.output, config)
    print(json.dumps({"event":"CONFIG_READY","geometry":contract["geometry"],"config":str(a.output)}),flush=True)


if __name__ == "__main__":
    main()
