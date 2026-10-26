"""Audit that the isolated SFT route keeps the shared Mizar implementation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROUTE_ROOT = Path(__file__).resolve().parent
MOTHER_ROOT = ROUTE_ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5")

# These files implement the unchanged data, audio, staging and reference
# training mechanics. Model identity and the explicit SFT freeze policy are
# isolated in the other files.
SHARED_FILES = (
    "data/audiotext_dataset.py", "data/audio_paths.py", "data/fixed20.py",
    "data/indexed_jsonl.py", "data/prepared_index.py", "data/sampler.py",
    "data/staging.py", "data/template.py", "models/audio.py", "models/config.py",
    "models/htsat.py", "utils/audio_io.py", "prepare_index.py", "stage_data.py",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_alignment(route_root=ROUTE_ROOT, mother_root=MOTHER_ROOT):
    problems, hashes = [], {}
    for relative in SHARED_FILES:
        left, right = Path(mother_root, relative), Path(route_root, relative)
        if not left.is_file() or not right.is_file():
            problems.append(f"missing shared file: {relative}")
            continue
        left_hash, right_hash = sha256(left), sha256(right)
        hashes[relative] = {"mother": left_hash, "route": right_hash}
        if left_hash != right_hash:
            problems.append(f"shared implementation differs: {relative}")

    contract = (route_root / "contract.py").read_text(encoding="utf-8")
    train = (route_root / "train.py").read_text(encoding="utf-8")
    optim = (route_root / "optim.py").read_text(encoding="utf-8")
    outer = (route_root.parents[1] / "run_mizar_s1_recursive_5_10_5_core_sft_gbs256_5090.sh").read_text(encoding="utf-8")
    required = {
        "two_epochs": "WORLD_SIZE, BATCH_SIZE, ACCUMULATION_STEPS, EPOCHS, SEED = 8, 8, 4, 2, 1234" in contract,
        "max_lr": "MAX_LR, MIN_LR = 5e-5, 5e-6" in contract,
        "core_layers_only": '"physical_layers_5_to_14"' in contract and '"mesh_routers"' not in optim,
        "execution_schedule": "ExecutionAudit" in train and "TEXT_MODEL_SCHEDULE" in train,
        "formal_steps": "11542" in outer,
        "source_checkpoint": "model--step-17313.ckpt" in outer,
    }
    problems.extend(name for name, passed in required.items() if not passed)
    return {"status": "PASS" if not problems else "FAIL", "mother": str(mother_root),
            "route": str(route_root), "shared_file_hashes": hashes,
            "required_contract_checks": required, "problems": problems}


if __name__ == "__main__":
    report = audit_alignment()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "PASS" else 1)

