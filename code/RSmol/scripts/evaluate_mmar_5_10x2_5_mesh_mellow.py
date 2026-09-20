#!/usr/bin/env python3
"""Evaluate the partition-v2 Audio MeSH checkpoint on official MMAR."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as common  # noqa: E402


DEFAULT_CHECKPOINT = common.DEFAULT_CHECKPOINT
DEFAULT_DATASET_DIR = "/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR"
DEFAULT_AUDIO_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR/mmar-audio"
DEFAULT_EVALUATION_SCRIPT = f"{DEFAULT_DATASET_DIR}/code/evaluation.py"
DEFAULT_HTSAT = common.DEFAULT_HTSAT
DEFAULT_MELLOW = common.DEFAULT_MELLOW
SMOKE_ROWS = 5
EXPECTED_FULL_ROWS = 1000
MMAR_MAX_PROMPT_TOKENS = (
    common.DEFAULT_MAX_CONTEXT_LENGTH
    - common.DEFAULT_AUDIO_PREFIX_TOKENS
    - common.DEFAULT_MAX_NEW_TOKENS
)
MMAR_OFFICIAL_COMMIT = "3bce090a967db576c8ad433b290a8da582d6d2a0"
MMAR_GITHUB_METADATA_SHA256 = "1c9a343e7bebf1037482b935c54eacd43cc63df7b2c668724b5560caf1f84fad"
MMAR_CORE_CANONICAL_SHA256 = "fc0527faaf8599f26e2b8dfcbe9c2efc59b3408b6c9996f24197192be57dff08"
MMAR_EVALUATION_SHA256 = "a3c57b829e40e67e3f7f0bbe7d54a112ea06ed7a781534241ecc5d3751486198"
MMAR_CORE_KEYS = (
    "id",
    "audio_path",
    "question",
    "choices",
    "answer",
    "modality",
    "category",
    "sub-category",
    "language",
    "source",
    "url",
    "timestamp",
)
EXPECTED_MODALITIES = {
    "mix-music-speech": 82,
    "mix-sound-music": 11,
    "mix-sound-music-speech": 24,
    "mix-sound-speech": 218,
    "music": 206,
    "sound": 165,
    "speech": 294,
}
EXPECTED_CATEGORIES = {
    "Cultural Layer": 141,
    "Perception Layer": 404,
    "Semantic Layer": 412,
    "Signal Layer": 43,
}


def load_mmar_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        records = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or any(not isinstance(item, Mapping) for item in records):
        raise ValueError("MMAR metadata must be a JSON/JSONL sequence of objects")
    return [dict(item) for item in records]


def _record_id(record: Mapping[str, Any]) -> str:
    return str(record.get("id", "")).strip()


def resolve_mmar_audio_path(audio_root: Path, value: Any) -> Path:
    raw_text = str(value).strip()
    if not raw_text:
        raise ValueError(f"MMAR audio_path must be a non-empty relative path: {value!r}")
    raw = Path(raw_text)
    if raw.is_absolute():
        raise ValueError(f"MMAR audio_path must be a non-empty relative path: {value!r}")
    root = audio_root.resolve()
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"MMAR audio_path escapes audio root: {value!r}") from exc
    return candidate


def audit_mmar_artifacts(
    metadata_path: Path,
    audio_root: Path,
    evaluation_script: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit HF JSON against the current GitHub MMAR accuracy core."""

    ids = [_record_id(record) for record in records]
    modality_counts = Counter(str(record.get("modality", "")) for record in records)
    category_counts = Counter(str(record.get("category", "")) for record in records)
    core = [{key: record.get(key) for key in MMAR_CORE_KEYS} for record in records]
    core_sha256 = common._canonical_sha256(core)
    missing_audio: list[str] = []
    invalid_records: list[dict[str, Any]] = []
    resolved_audio: list[Path] = []
    for index, record in enumerate(records):
        choices = record.get("choices")
        answer = str(record.get("answer", ""))
        try:
            audio_path = resolve_mmar_audio_path(audio_root, record.get("audio_path", ""))
            resolved_audio.append(audio_path)
            if not audio_path.is_file():
                missing_audio.append(str(audio_path))
        except Exception as exc:
            invalid_records.append({"row_index": index, "id": ids[index], "error": str(exc)})
        if (
            not ids[index]
            or not str(record.get("question", "")).strip()
            or not isinstance(choices, list)
            or not 2 <= len(choices) <= 6
            or any(not str(choice).strip() for choice in choices or [])
            or not answer
            or answer not in [str(choice) for choice in choices or []]
        ):
            invalid_records.append({"row_index": index, "id": ids[index], "error": "invalid MCQ schema"})

    scorer_sha256 = common._sha256(evaluation_script)
    failures: list[str] = []
    if len(records) != EXPECTED_FULL_ROWS or len(set(ids)) != EXPECTED_FULL_ROWS or any(not value for value in ids):
        failures.append(
            f"metadata ID coverage differs: rows={len(records)} unique_nonempty={len({value for value in ids if value})}"
        )
    if core_sha256 != MMAR_CORE_CANONICAL_SHA256:
        failures.append(
            "MMAR core canonical SHA256 differs from official GitHub metadata: "
            f"expected={MMAR_CORE_CANONICAL_SHA256} actual={core_sha256}"
        )
    if dict(sorted(modality_counts.items())) != EXPECTED_MODALITIES:
        failures.append(f"modality distribution differs: {dict(sorted(modality_counts.items()))}")
    if dict(sorted(category_counts.items())) != EXPECTED_CATEGORIES:
        failures.append(f"category distribution differs: {dict(sorted(category_counts.items()))}")
    if invalid_records:
        failures.append(f"invalid records: {invalid_records[:10]}")
    if missing_audio:
        failures.append(f"missing audio files: count={len(missing_audio)} first={missing_audio[:10]}")
    if scorer_sha256 != MMAR_EVALUATION_SHA256:
        failures.append(
            "evaluation.py byte SHA256 differs from the current official scorer: "
            f"expected={MMAR_EVALUATION_SHA256} actual={scorer_sha256}"
        )
    report = {
        "status": "PASS" if not failures else "FAIL",
        "official_commit": MMAR_OFFICIAL_COMMIT,
        "official_github_metadata_sha256": MMAR_GITHUB_METADATA_SHA256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": common._sha256(metadata_path),
        "core_canonical_sha256": core_sha256,
        "audio_root": str(audio_root),
        "audio_files_resolved": len(resolved_audio),
        "unique_audio_files": len(set(resolved_audio)),
        "missing_audio_count": len(missing_audio),
        "evaluation_script": str(evaluation_script),
        "evaluation_sha256": scorer_sha256,
        "rows": len(records),
        "unique_ids": len(set(ids)),
        "modality_counts": dict(sorted(modality_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "failures": failures,
    }
    return report


def _prepare_sample(record: Mapping[str, Any], row_index: int, audio_root: Path) -> dict[str, Any]:
    choices = [str(choice) for choice in record["choices"]]
    audio_path = resolve_mmar_audio_path(audio_root, record["audio_path"])
    try:
        waveform, audio_info = common.decode_and_normalize_audio(
            str(audio_path),
            base_dir=audio_root,
        )
    except Exception as exc:
        raise common.RowSkip("audio", "audio_decode_failed", str(exc), audio_path=str(audio_path)) from exc
    question = str(record["question"])
    return {
        "id": _record_id(record),
        "row_index": int(row_index),
        "question": question,
        "choices": choices,
        "answer": str(record["answer"]),
        "prompt": common.build_fixed_order_prompt(question, choices),
        "official_record": copy.deepcopy(dict(record)),
        "waveform": waveform,
        "audio_payload_source": "audio_path",
        "audio_path_resolved": str(audio_path),
        "audio2_reused": True,
        **audio_info,
    }


def materialize_predictions(
    state: Sequence[Mapping[str, Any]],
    official_records: Sequence[Mapping[str, Any]],
    *,
    full: bool,
) -> list[dict[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in state:
        if str(item.get("status", "")) != "generated":
            continue
        sample_id = str(item.get("id", "")).strip()
        if sample_id in by_id:
            raise RuntimeError(f"multiple terminal generations for MMAR ID: {sample_id}")
        by_id[sample_id] = item
    source = official_records if full else official_records[:SMOKE_ROWS]
    predictions: list[dict[str, Any]] = []
    for official in source:
        item = by_id.get(_record_id(official))
        prediction = copy.deepcopy(dict(official))
        prediction["answer_prediction"] = "" if item is None else str(item.get("answer_prediction", ""))
        predictions.append(prediction)
    return predictions


def _write_outputs(
    store: common.ProgressStore,
    output_dir: Path,
    official_records: Sequence[Mapping[str, Any]],
    *,
    full: bool,
) -> list[dict[str, Any]]:
    predictions = materialize_predictions(list(store.state.values()), official_records, full=full)
    common._write_json(output_dir / "predictions_answer_prediction.json", predictions)
    smoke = sorted(
        (item for item in store.state.values() if int(item.get("row_index", -1)) < SMOKE_ROWS),
        key=lambda item: int(item.get("row_index", -1)),
    )
    with (output_dir / "smoke_first5.jsonl").open("w", encoding="utf-8") as handle:
        for item in smoke:
            handle.write(json.dumps(item, ensure_ascii=False, default=common._json_default) + "\n")
    return predictions


def _run_official_evaluation(args: argparse.Namespace, predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    log_path = args.output_dir / "official_evaluation.txt"
    result: dict[str, Any] = {
        "requested": bool(args.run_official_evaluation),
        "prediction_count": len(predictions),
        "path": str(log_path),
    }
    if not args.run_official_evaluation:
        log_path.write_text("Official evaluation was not requested.\n", encoding="utf-8")
        return {**result, "status": "NOT_REQUESTED"}
    command = [
        sys.executable,
        str(args.evaluation_script),
        "--input",
        str(args.output_dir / "predictions_answer_prediction.json"),
    ]
    completed = subprocess.run(
        command,
        cwd=str(args.evaluation_script.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.write_text(
        f"$ {' '.join(command)}\nreturncode: {completed.returncode}\n\n"
        f"===== STDOUT =====\n{completed.stdout}\n===== STDERR =====\n{completed.stderr}\n",
        encoding="utf-8",
    )
    totals = [int(value) for value in re.findall(r"\bover\s+(\d+)\s+samples\b", completed.stdout)]
    reported_total = totals[-1] if totals else None
    accuracy_matches = re.findall(
        r"Total Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%\s+over\s+(\d+)\s+samples",
        completed.stdout,
    )
    total_accuracy_percent = float(accuracy_matches[-1][0]) if accuracy_matches else None
    passed = completed.returncode == 0 and reported_total == len(predictions)
    if completed.returncode != 0:
        error = f"official scorer exited with return code {completed.returncode}"
    elif reported_total != len(predictions):
        error = (
            "official scorer did not report the requested denominator: "
            f"reported={reported_total} expected={len(predictions)}"
        )
    else:
        error = None
    return {
        **result,
        "status": "PASS" if passed else "FAILED",
        "returncode": completed.returncode,
        "reported_total": reported_total,
        "total_accuracy_percent": total_accuracy_percent,
        **({} if error is None else {"error": error}),
    }


def _ensure_output_dir(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    immutable = {
        "checkpoint": str(args.checkpoint),
        "metadata_json": str(args.metadata_json),
        "audio_root": str(args.audio_root),
        "evaluation_script": str(args.evaluation_script),
        "htsat_checkpoint": str(args.htsat_checkpoint),
        "mellow_root": str(args.mellow_root),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "prompt_format": common.PROMPT_FORMAT,
        "prediction_format": common.PREDICTION_FORMAT,
        "protocol": "official order; ReasonAQA lowercase labels; compact single-audio prefix; single cuda:0; bf16; greedy",
    }
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        mismatches = {
            key: {"existing": existing.get(key), "requested": value}
            for key, value in immutable.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"existing output directory belongs to another MMAR run: {mismatches}")
        old_mode = str(existing.get("mode", ""))
        if old_mode == "smoke" and args.mode == "full":
            existing["mode"] = "full"
            existing["mode_history"] = ["smoke", "full"]
            common._write_json(config_path, existing)
        elif old_mode not in {"", args.mode}:
            raise RuntimeError(f"MMAR run mode cannot change from {old_mode!r} to {args.mode!r}")
    else:
        entries = list(args.output_dir.iterdir())
        if entries:
            raise FileExistsError(f"refusing non-empty unowned output directory: {args.output_dir}")
        common._write_json(config_path, {**immutable, "mode": args.mode})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR))
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--evaluation-script", type=Path)
    parser.add_argument("--htsat-checkpoint", type=Path, default=Path(DEFAULT_HTSAT))
    parser.add_argument("--mellow-root", type=Path, default=Path(DEFAULT_MELLOW))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=MMAR_MAX_PROMPT_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=common.DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--run-official-evaluation", action="store_true")
    args = parser.parse_args(argv)
    args.metadata_json = args.metadata_json or args.dataset_dir / "MMAR-meta.json"
    args.audio_root = args.audio_root or args.dataset_dir / "mmar-audio"
    args.evaluation_script = args.evaluation_script or args.dataset_dir / "code" / "evaluation.py"
    if args.max_prompt_tokens != MMAR_MAX_PROMPT_TOKENS:
        parser.error(f"--max-prompt-tokens is fixed at {MMAR_MAX_PROMPT_TOKENS}")
    if args.max_new_tokens != common.DEFAULT_MAX_NEW_TOKENS:
        parser.error(f"--max-new-tokens is fixed at {common.DEFAULT_MAX_NEW_TOKENS}")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    _ensure_output_dir(args)
    report: dict[str, Any] = {
        "stage": "mmar_audio_mesh_official_accuracy",
        "status": "FAILED",
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "metadata_json": str(args.metadata_json),
        "audio_root": str(args.audio_root),
        "evaluation_script": str(args.evaluation_script),
        "protocol": {
            "metadata_order": "official MMAR order",
            "choice_order": "official fixed order",
            "prompt_format": common.PROMPT_FORMAT,
            "prediction_format": common.PREDICTION_FORMAT,
            "shuffle": False,
            "mode_limit": SMOKE_ROWS if args.mode == "smoke" else None,
            "audio_sample_rate": common.DEFAULT_SAMPLE_RATE,
            "audio_seconds": common.DEFAULT_AUDIO_SECONDS,
            "long_audio_policy": "first 10 seconds",
            "audio_prefix_tokens": common.DEFAULT_AUDIO_PREFIX_TOKENS,
            "max_prompt_tokens_without_truncation": MMAR_MAX_PROMPT_TOKENS,
            "max_new_tokens": common.DEFAULT_MAX_NEW_TOKENS,
            "do_sample": False,
            "use_cache": False,
            "prediction_key": "answer_prediction",
        },
        "records": {},
        "resumption": {},
        "official_artifact_audit": {},
        "official_evaluation": {},
        "warnings": [],
        "fatal_error": None,
    }
    records: list[dict[str, Any]] = []
    with common.ProgressStore(args.output_dir) as store:
        report["resumption"]["recovered_jsonl_errors"] = store.load_errors
        try:
            for path, description in (
                (args.checkpoint, "checkpoint"),
                (args.audio_root, "MMAR audio root"),
            ):
                if not path.is_dir():
                    raise FileNotFoundError(f"{description} directory not found: {path}")
            if not args.metadata_json.is_file():
                raise FileNotFoundError(f"MMAR metadata not found: {args.metadata_json}")
            if not args.evaluation_script.is_file():
                raise FileNotFoundError(f"MMAR official scorer not found: {args.evaluation_script}")
            records = load_mmar_records(args.metadata_json)
            report["official_artifact_audit"] = audit_mmar_artifacts(
                args.metadata_json,
                args.audio_root,
                args.evaluation_script,
                records,
            )
            if report["official_artifact_audit"]["status"] != "PASS":
                raise RuntimeError(
                    "official MMAR artifact audit failed: "
                    f"{report['official_artifact_audit']['failures']}"
                )
            model, tokenizer, device, checkpoint_config = common._load_runtime_model(args)
            limit = SMOKE_ROWS if args.mode == "smoke" else len(records)
            for row_index, official in enumerate(records[:limit]):
                sample_id = _record_id(official)
                report["records"]["rows_read"] = int(report["records"].get("rows_read", 0) + 1)
                if store.has_terminal(row_index, sample_id):
                    report["records"]["resumed_rows"] = int(report["records"].get("resumed_rows", 0) + 1)
                    continue
                try:
                    sample = _prepare_sample(official, row_index, args.audio_root)
                    generation = common._run_model_generation(
                        model,
                        tokenizer,
                        device,
                        sample,
                        max_prompt_tokens=args.max_prompt_tokens,
                        max_new_tokens=args.max_new_tokens,
                    )
                    answer_prediction = str(generation.get("generated_text", ""))
                    record = {
                        "status": "generated",
                        "row_index": row_index,
                        "id": sample_id,
                        "row_key": common.row_key(row_index, sample_id),
                        "question": sample["question"],
                        "choices": sample["choices"],
                        "prompt": sample["prompt"],
                        "official_record": sample["official_record"],
                        "answer_prediction": answer_prediction,
                        "audio2_reused": True,
                        "single_audio_slot": True,
                        **{key: value for key, value in sample.items() if key.startswith("audio_")},
                        **generation,
                    }
                    store.append_raw(record)
                    print(
                        json.dumps(
                            {
                                "row_index": row_index,
                                "id": sample_id,
                                "status": record["status"],
                                "answer_prediction": record["answer_prediction"],
                                "generated_text": generation.get("generated_text", ""),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                except common.RowSkip as exc:
                    store.append_skip({
                        "status": "skipped",
                        "row_index": row_index,
                        "id": sample_id,
                        "row_key": common.row_key(row_index, sample_id),
                        "stage": exc.stage,
                        "reason": exc.reason,
                        "error": exc.message,
                        **exc.details,
                    })
                except Exception as exc:
                    store.append_skip({
                        "status": "skipped",
                        "row_index": row_index,
                        "id": sample_id,
                        "row_key": common.row_key(row_index, sample_id),
                        "stage": "inference",
                        "reason": "sample_exception",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(limit=8),
                    })
            del model
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            predictions = _write_outputs(
                store,
                args.output_dir,
                records,
                full=args.mode == "full",
            )
            report["records"].update(common._counts(store))
            report["records"]["official_scoring_denominator"] = len(predictions)
            report["resumption"]["rows_not_repeated"] = int(report["records"].get("resumed_rows", 0))
            report["checkpoint_config"] = checkpoint_config
            expected = EXPECTED_FULL_ROWS if args.mode == "full" else SMOKE_ROWS
            if int(report["records"].get("rows_read", 0)) != expected:
                raise RuntimeError(
                    f"MMAR metadata traversal is incomplete: "
                    f"{report['records'].get('rows_read')} / {expected}"
                )
            if len(predictions) != expected or int(report["records"].get("terminal_records", 0)) != expected:
                raise RuntimeError(
                    f"MMAR coverage mismatch: predictions={len(predictions)} "
                    f"terminal={report['records'].get('terminal_records')} expected={expected}"
                )
            skipped = int(report["records"].get("skipped", 0))
            report["records"]["official_empty_predictions_from_skips"] = skipped
            if skipped:
                report["warnings"].append({
                    "name": "skipped_rows_scored_as_incorrect",
                    "count": skipped,
                    "reasons": report["records"].get("skip_reasons", {}),
                    "detail": (
                        "Every metadata row was visited. Skipped rows retain an empty "
                        "answer_prediction and are counted as incorrect by the official scorer."
                    ),
                })
            report["official_evaluation"] = _run_official_evaluation(args, predictions)
            report["official_evaluation"]["empty_predictions_from_skips"] = skipped
            report["status"] = "PASS" if report["official_evaluation"]["status"] != "FAILED" else "FAILED"
        except Exception as exc:
            report["fatal_error"] = {"error": repr(exc), "traceback": traceback.format_exc()}
            predictions = _write_outputs(
                store,
                args.output_dir,
                records,
                full=args.mode == "full",
            ) if records else []
            report["records"].update(common._counts(store))
            report["records"]["official_scoring_denominator"] = len(predictions)
            report["official_evaluation"] = common._block_official_evaluation(
                args,
                args.output_dir,
                len(predictions),
                exc,
            )
            report["status"] = "FAILED"
        report["elapsed_seconds"] = time.time() - started
        common._write_json(args.output_dir / "evaluation_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps({
        "stage": report.get("stage"),
        "status": report.get("status"),
        "mode": report.get("mode"),
        "records": report.get("records"),
        "official_evaluation": report.get("official_evaluation"),
        "report": str(args.output_dir / "evaluation_report.json"),
    }, ensure_ascii=False, default=common._json_default))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
