"""CPU-only offline MMSU re-scoring; no weights, audio, parquet or inference imports."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from . import dataset, parsing, scoring, storage

DEFAULT_SCORER = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/MMSU/MMSU code/mmsu_evaluation.py")
SUBDIRECTORY = "rescore_mmsu_choice_v2"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluation-script", type=Path, default=DEFAULT_SCORER)
    parser.add_argument("--check-args", action="store_true")
    args = parser.parse_args(argv)
    args.source_output_dir = args.source_output_dir.resolve()
    args.output_dir = (args.output_dir or args.source_output_dir / SUBDIRECTORY).resolve()
    args.evaluation_script = args.evaluation_script.resolve()
    if args.output_dir == args.source_output_dir or args.output_dir in args.source_output_dir.parents:
        parser.error("offline scoring needs a separate output directory, never the source or its ancestor")
    return args


def read_source(directory):
    """Prefer complete, hash-validated raw; fall back only when raw is absent."""
    raw = directory / "raw_generations.jsonl"
    predictions = directory / "predictions_official.jsonl"
    path = raw if raw.is_file() else predictions
    before = storage.file_sha(path)
    values = parsing.read_complete_jsonl(path)
    records = []
    if path == raw:
        for index, saved in enumerate(values):
            record, generation = saved.get("official_record"), saved.get("generation")
            if (not isinstance(record, dict) or not isinstance(generation, dict)
                    or saved.get("row_index") != index or saved.get("id") != record.get("id")
                    or saved.get("record_sha256") != storage.object_sha(record)
                    or saved.get("generation_sha256") != storage.object_sha(generation)):
                raise ValueError(f"saved MMSU row identity/content mismatch at index {index}")
            records.append({**{key: record[key] for key in dataset.METADATA},
                            "response": generation.get("generated_text")})
        if predictions.is_file():
            other = parsing.read_complete_jsonl(predictions)
            parsing.validate_predictions(other)
            other = [{key: row[key] for key in (*dataset.METADATA, "response")} for row in other]
            if other != records:
                raise ValueError("raw generations and existing raw predictions disagree; source preserved")
    else:
        records = values
    parsing.validate_predictions(records)
    if storage.file_sha(path) != before:
        raise ValueError("MMSU source changed during read")
    provenance = {"input": str(path), "input_sha256": before, "rows": len(records),
                  "kind": "hash_validated_raw_generations" if path == raw else "raw_predictions_only",
                  "original_artifacts": {}}
    for name in ("run_config.json", "model_audit.json", "evaluation_report.json", "predictions_official.jsonl"):
        artifact = directory / name
        if artifact.is_file():
            provenance["original_artifacts"][name] = storage.file_sha(artifact)
    return records, provenance


def run(args):
    started = time.monotonic()
    claimed = False
    report = {"status": "FAILED", "benchmark": "mmsu", "mode": "offline_rescore",
              "gpu_inference": False, "response_policy": parsing.POLICY, "fatal_error": None}
    try:
        if not args.source_output_dir.is_dir():
            raise FileNotFoundError(f"MMSU source directory missing: {args.source_output_dir}")
        # Cooperate with the existing generator's lock. No raw/progress recovery
        # is called: source JSONL and historical reports are never modified.
        with storage.output_lock(args.source_output_dir):
            records, provenance = read_source(args.source_output_dir)
            identity = {"mode": "offline_rescore", "source": provenance,
                        "policy": parsing.POLICY, "scorer": scoring.audit_scorer(args.evaluation_script),
                        "code": {name: storage.text_sha(Path(__file__).parent / name) for name in
                                 ("rescore.py", "parsing.py", "scoring.py", "dataset.py", "storage.py")}}
            identity["identity_sha256"] = storage.object_sha(identity)
            report["source"] = provenance
            with storage.output_lock(args.output_dir):
                storage.claim_output(args.output_dir, identity)
                claimed = True
                try:
                    report["response_parsing"], report["official_evaluation"] = parsing.score_records(
                        records, args.evaluation_script, args.output_dir)
                    report["status"] = report["official_evaluation"]["status"]
                except Exception as exc:
                    report["fatal_error"] = {"error": repr(exc), "traceback": traceback.format_exc()}
                finally:
                    report["elapsed_seconds"] = time.monotonic() - started
                    storage.write_json(args.output_dir / "evaluation_report.json", report)
    except Exception as exc:
        report["status"] = "FAILED"
        report["fatal_error"] = {"error": repr(exc), "traceback": traceback.format_exc()}
    report["report"] = str(args.output_dir / "evaluation_report.json") if claimed else None
    return report


def main(argv=None):
    args = parse_args(argv)
    if args.check_args:
        print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}))
        return 0
    report = run(args)
    parsed = report.get("response_parsing", {})
    print(json.dumps({"status": report["status"], "source": str(args.source_output_dir),
                      "recognized_rows": parsed.get("recognized_rows"), "invalid_rows": parsed.get("invalid_rows"),
                      "official_evaluation": report.get("official_evaluation"),
                      "fatal_error": report["fatal_error"], "report": report["report"]}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
