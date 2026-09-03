#!/usr/bin/env python3
"""Ten-step real-data Stage 4 smoke for 5-10xpoisson-parcae."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts.train_stage4_5_10xpoisson_parcae_ddp import (  # noqa: E402
    MIDDLE_LAYER_COUNT,
    MAX_MIDDLE_LOOPS,
    MIN_MIDDLE_LOOPS,
    Stage4Config,
    build_schedule,
    poisson_metadata,
    run_training,
    sample_middle_loop_counts,
    validate_schedule_contract,
)


def audit_all_supported_T() -> dict[str, object]:
    audits = {}
    for T in range(MIN_MIDDLE_LOOPS, MAX_MIDDLE_LOOPS + 1):
        schedule = build_schedule(T)
        expected_entries = T * MIDDLE_LAYER_COUNT
        middle_entries = schedule[5 : 5 + expected_entries]
        if len(middle_entries) != expected_entries or tuple(middle_entries) != tuple(range(5, 15)) * T:
            raise AssertionError(f"T={T} middle loop schedule is invalid")
        audits[str(T)] = {"logical_depth": len(schedule), "expected_entries": expected_entries, "middle_entries": list(middle_entries)}
    return audits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    schedule_audit = audit_all_supported_T()
    smoke_config = Stage4Config(gate="D", model_path=args.model_path, tokenizer_path=args.tokenizer_path, data_dir=args.data_dir, output_dir=args.output_dir, max_optimizer_steps=10, scheduler_total_steps=10, warmup_steps=1, device=args.device, seed=args.seed)
    report = run_training(smoke_config)
    report["smoke"] = {"real_data": True, "optimizer_steps": 10, "all_supported_T_audited": schedule_audit, "sampling_contract": poisson_metadata(), "cache_use": "inference-only", "training_use_cache": False}
    # run_training destroys its process group before returning, so only rank 0
    # owns the report write; eight torchrun workers must not race on the file.
    if int(os.environ.get("RANK", "0")) == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "stage4_10step_smoke_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
