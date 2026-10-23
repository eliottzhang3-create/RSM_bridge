#!/usr/bin/env python3
"""Rescore MMAU prediction JSON or raw generation JSONL with the label metric.

This command performs no model loading or inference.  It isolates the effect
of the scorer only; it cannot retrofit the author-reply prompt, audio, or
generation protocol onto predictions produced by an older run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as mmau  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path,
                        help="prediction JSON array or raw_generations.jsonl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-rows", type=int, choices=(5, 1000))
    parser.add_argument(
        "--source-checkpoint", type=Path,
        help="optional provenance only; the checkpoint is never loaded or hashed",
    )
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir or (
        args.predictions.parent / "choice_label_rescore"
        if args.predictions.suffix.lower() == ".jsonl" else args.predictions.parent
    )
    return args


def load_raw_predictions(path: Path, expected_rows: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use embedded official metadata and unmodified generated_text, with full coverage."""
    by_index: dict[int, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    blank_lines = 0
    metadata_normalization_counts = {"question": 0, "choices": 0}
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                blank_lines += 1
                continue
            where = f"{path}: line {line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or row.get("status") != "generated":
                raise ValueError(f"{where}: expected a generated record")
            index, sample_id = row.get("row_index"), row.get("id")
            if type(index) is not int or index < 0:
                raise ValueError(f"{where}: invalid row_index: {index!r}")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"{where}: missing or invalid id")
            if index in by_index:
                raise ValueError(f"{where}: duplicate row_index: {index}")
            if sample_id in seen_ids:
                raise ValueError(f"{where}: duplicate id: {sample_id!r}")
            text = row.get("generated_text")
            if not isinstance(text, str):
                raise ValueError(f"{where}: generated_text must be a string")
            official = row.get("official_record")
            if not isinstance(official, dict) or official.get("id") != sample_id:
                raise ValueError(f"{where}: official_record is missing or its id differs")
            choices = official.get("choices")
            if (not isinstance(choices, list) or len(choices) < 2
                    or not all(isinstance(choice, str) for choice in choices)):
                raise ValueError(f"{where}: official_record has invalid choices")
            for field in ("question", "answer", "task", "difficulty"):
                if not isinstance(official.get(field), str) or not official[field]:
                    raise ValueError(f"{where}: official_record has invalid {field}")
            # Inference extracts stripped display fields but saves the untouched
            # official_record alongside them. Compare using the same extractors.
            display_fields = {
                "question": mmau.extract_record_question(official),
                "choices": mmau._coerce_choices(official["choices"]),
            }
            for field, expected in display_fields.items():
                if field in row:
                    if row[field] != expected:
                        raise ValueError(
                            f"{where}: {field} differs from inference-normalized official_record; "
                            f"actual={row[field]!r}, expected={expected!r}"
                        )
                    if row[field] != official[field]:
                        metadata_normalization_counts[field] += 1
            if (official["task"] not in {"sound", "music", "speech"}
                    or official["difficulty"] not in {"easy", "medium", "hard"}):
                raise ValueError(f"{where}: invalid MMAU task or difficulty")
            # Ignore any model_output field: it may contain a processed prediction.
            by_index[index] = {**official, "model_output": text}
            seen_ids.add(sample_id)
    if len(by_index) != expected_rows:
        raise ValueError(
            f"prediction denominator mismatch: expected={expected_rows} actual={len(by_index)}"
        )
    expected_indices = set(range(expected_rows))
    if set(by_index) != expected_indices:
        raise ValueError(
            "raw generation row_index coverage differs: "
            f"missing={sorted(expected_indices - set(by_index))[:20]} "
            f"unexpected={sorted(set(by_index) - expected_indices)[:20]}"
        )
    return [by_index[index] for index in range(expected_rows)], {
        "status": "PASS", "expected_rows": expected_rows,
        "generated_rows": len(by_index), "unique_ids": len(seen_ids),
        "blank_lines": blank_lines, "row_index_coverage": f"0..{expected_rows - 1}",
        "prediction_field": "generated_text", "metadata_field": "official_record",
        "metadata_comparison_policy": "same_extractors_as_inference",
        "metadata_normalization_counts": metadata_normalization_counts,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.predictions.is_file():
        raise FileNotFoundError(f"predictions file not found: {args.predictions}")
    raw_jsonl = args.predictions.suffix.lower() == ".jsonl"
    input_audit = None
    if raw_jsonl:
        predictions, input_audit = load_raw_predictions(
            args.predictions, args.expected_rows if args.expected_rows is not None else 1000
        )
    else:
        predictions = json.loads(args.predictions.read_text(encoding="utf-8-sig"))
        if not isinstance(predictions, list):
            raise ValueError("predictions JSON must contain a list")
    if args.expected_rows is not None and len(predictions) != args.expected_rows:
        raise ValueError(
            f"prediction denominator mismatch: expected={args.expected_rows} actual={len(predictions)}"
        )
    # The same scorer works for GBS32 and GBS256. Carry over the evaluator's
    # recorded checkpoint identity without touching the large checkpoint file.
    run_config_path = args.predictions.parent / "run_config.json"
    source_run = None
    source_checkpoint = args.source_checkpoint
    if run_config_path.is_file():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8-sig"))
        recorded_checkpoint = run_config.get("checkpoint")
        if recorded_checkpoint:
            recorded_checkpoint = Path(recorded_checkpoint)
            if (source_checkpoint is not None
                    and source_checkpoint.resolve() != recorded_checkpoint.resolve()):
                raise ValueError("--source-checkpoint differs from the prediction run_config.json")
            source_checkpoint = recorded_checkpoint
        source_run = {
            "run_config": str(run_config_path.resolve()),
            "run_config_sha256": _sha256(run_config_path),
            **{key: run_config.get(key) for key in (
                "checkpoint", "checkpoint_sha256", "runtime_config", "route_root",
                "expected_step", "expected_geometry", "identity_sha256")},
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score = mmau.write_mellow_author_reply_evaluation(args.output_dir, predictions)
    report = {
        "status": "PASS",
        "operation": "scorer_only_no_inference",
        "source_predictions": str(args.predictions.resolve()),
        "source_predictions_sha256": _sha256(args.predictions),
        "source_checkpoint": (
            str(source_checkpoint.resolve()) if source_checkpoint else None
        ),
        "source_evaluation_run": source_run,
        "input_format": "raw_generations_jsonl" if raw_jsonl else "predictions_json",
        "input_audit": input_audit,
        "prediction_count": len(predictions),
        "scorer": mmau.MELLOW_AUTHOR_REPLY_SCORER,
        "scoring_rule": "prediction.split(')')[0].lower() == labeled_answer.split(')')[0].lower()",
        "trim_whitespace": False,
        "official_mmau_scorer": False,
        "score": score["total"],
        "task": score["task"],
        "difficulty": score["difficulty"],
        "record_errors": score["record_errors"],
        "output_dir": str(args.output_dir.resolve()),
        "warning": (
            "This result changes only the scoring rule. It is not a matched-protocol "
            "Mellow comparison unless the source run already used the author-reply prompt, "
            "audio preprocessing, max_len=300, and decoding protocol."
        ),
    }
    mmau._write_json(args.output_dir / "mellow_author_reply_rescore_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
