#!/usr/bin/env python3
"""Prepare and audit deterministic ReasonAQA audio manifests on CPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from audio_5_10_5_mellow.manifest import (  # noqa: E402
    build_reasonaqa_manifests,
    canonical_json_hash,
    write_json,
    write_jsonl,
)


DEFAULT_REASONAQA_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa"
DEFAULT_AUDIOCAPS_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2"
DEFAULT_CLOTHO_AUDIO_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2_1"
DEFAULT_CLOTHO_AQA_AUDIO_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_audio"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasonaqa-root", "--reasonaqa_root", type=Path, default=Path(DEFAULT_REASONAQA_ROOT))
    parser.add_argument("--train-json", "--train_json", type=Path)
    parser.add_argument("--val-json", "--val_json", type=Path)
    parser.add_argument("--test-json", "--test_json", type=Path)
    parser.add_argument("--audiocaps-root", "--audiocaps_root", type=Path, default=Path(DEFAULT_AUDIOCAPS_ROOT))
    parser.add_argument("--clotho-audio-root", "--clotho_audio_root", "--clotho-root", "--clotho_root", type=Path, default=Path(DEFAULT_CLOTHO_AUDIO_ROOT))
    parser.add_argument("--clotho-aqa-audio-root", "--clotho_aqa_audio_root", type=Path, default=Path(DEFAULT_CLOTHO_AQA_AUDIO_ROOT))
    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True)
    parser.add_argument("--report-path", "--report_path", type=Path)
    parser.add_argument("--manifest-path", "--manifest_path", type=Path, help="Optional combined JSONL path")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--dry-run", "--audit-only", action="store_true", help="Audit and hash, but do not write JSONL")
    output_mode.add_argument("--write-manifest", action="store_true", help="Write split JSONL manifests (default when not dry-run)")
    unresolved_mode = parser.add_mutually_exclusive_group()
    unresolved_mode.add_argument("--allow-missing", action="store_true", help="Write unresolved rows with empty paths and report warnings")
    unresolved_mode.add_argument("--drop-unresolved", action="store_true", help="Omit unresolved rows from manifests and report them as warnings")
    return parser.parse_args(argv)


def _split_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "train": args.train_json or args.reasonaqa_root / "train.json",
        "val": args.val_json or args.reasonaqa_root / "val.json",
        "test": args.test_json or args.reasonaqa_root / "test.json",
    }


def _write_outputs(args: argparse.Namespace, manifests: dict[str, list[dict[str, Any]]], report: dict[str, Any]) -> None:
    report_path = args.report_path or args.output_dir / "reasonaqa_manifest_audit.json"
    report["output"] = {"report_path": str(report_path), "dry_run": bool(args.dry_run)}
    if not args.dry_run:
        split_paths: dict[str, str] = {}
        split_hashes: dict[str, str] = {}
        for split, rows in manifests.items():
            path = args.output_dir / f"reasonaqa_{split}.jsonl"
            split_hashes[split] = write_jsonl(path, rows)
            split_paths[split] = str(path)
        if args.manifest_path:
            combined: list[dict[str, Any]] = []
            for split in ("train", "val", "test"):
                combined.extend(manifests.get(split, []))
            split_hashes["combined"] = write_jsonl(args.manifest_path, combined)
            split_paths["combined"] = str(args.manifest_path)
        report["output"].update({"manifest_paths": split_paths, "manifest_hashes": split_hashes})
    report["output"]["report_hash"] = canonical_json_hash(report)
    write_json(report_path, report)


def run(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "stage": "stage1_reasonaqa_manifest_5_10_5_mellow",
        "configuration": {
            "reasonaqa_root": str(args.reasonaqa_root),
            "audiocaps_root": str(args.audiocaps_root),
            "clotho_audio_root": str(args.clotho_audio_root),
            "clotho_aqa_audio_root": str(args.clotho_aqa_audio_root),
            "allow_missing": bool(args.allow_missing),
            "drop_unresolved": bool(args.drop_unresolved),
            "waveform_loaded": False,
            "formal_world_size": 8,
            "formal_micro_batch_per_gpu": 4,
            "formal_global_micro_batch": 32,
        },
        "traceback": None,
    }
    try:
        manifests, audit = build_reasonaqa_manifests(
            _split_paths(args),
            (args.audiocaps_root, args.clotho_audio_root, args.clotho_aqa_audio_root),
            allow_missing=args.allow_missing,
            drop_unresolved=args.drop_unresolved,
        )
        report.update(audit)
        report["status"] = audit["status"]
        _write_outputs(args, manifests, report)
    except Exception as exc:  # noqa: BLE001
        report.update({"status": "FAIL", "hard_failures": [{"name": "manifest_exception", "passed": False, "detail": f"{type(exc).__name__}: {exc}"}], "warnings": [], "traceback": traceback.format_exc()})
        report["summary"] = {"records": 0, "hard_failures": 1, "warnings": 0}
        _write_outputs(args, {}, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps({"stage": report["stage"], "status": report["status"], "summary": report.get("summary", {}), "report": str(args.report_path or args.output_dir / "reasonaqa_manifest_audit.json")}, ensure_ascii=False))
    return 0 if report["status"] in {"PASS", "PASS_WITH_WARNINGS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
