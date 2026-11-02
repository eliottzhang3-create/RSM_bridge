#!/usr/bin/env python3
"""Official Mizar Stage3 Q4 final weights; unchanged project MMAU/MMAR protocol."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
for import_root in (HERE.parent, HERE.parent / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
from mizar_official_eval import runtime as runtime_adapter  # noqa: E402
import evaluate_mizar_s1_ced_benchmark as base  # noqa: E402

EXPECTED_ROWS = base.EXPECTED_ROWS
DEFAULT_MMAU_DIR, DEFAULT_MMAR_DIR = base.DEFAULT_MMAU_DIR, base.DEFAULT_MMAR_DIR
OUTPUT_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
                   "mizar_official_stage3_q4_seed20260905_matched_ced_v1")
mmau, mmar = base.mmau, base.mmar
# Function objects are shared: no fork of the audio, prompt, decoding or scoring path.
claim_output = base.claim_output
scorer_prediction_key = base.scorer_prediction_key
_sha256_json = base._sha256_json
make_mmau_audio_decoder = base.make_mmau_audio_decoder
generate = base.generate
_metadata_records = base._metadata_records
_prepare_mmar = base._prepare_mmar
_materialize = base._materialize
_write_predictions = base._write_predictions
run_official_scorer = base.run_official_scorer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("mmau", "mmar"))
    parser.add_argument("--model-dir", type=Path, default=runtime_adapter.DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path,
                        help="optional relocated copy of the same pinned official final checkpoint")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--evaluation-script", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parquet-batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    args.dataset_dir = (args.dataset_dir or (DEFAULT_MMAU_DIR if args.benchmark == "mmau" else DEFAULT_MMAR_DIR)).resolve()
    args.metadata_json = (args.metadata_json or args.dataset_dir / ("mmau-test-mini.json" if args.benchmark == "mmau" else "MMAR-meta.json")).resolve()
    args.evaluation_script = (args.evaluation_script or args.dataset_dir / ("evaluation.py" if args.benchmark == "mmau" else "code/evaluation.py")).resolve()
    if args.benchmark == "mmau":
        args.parquet = (args.parquet or args.dataset_dir / "test_mini.parquet").resolve()
        args.audio_root = (args.audio_root or args.dataset_dir / "test-mini-audios").resolve()
    else:
        args.parquet = None
        args.audio_root = (args.audio_root or args.dataset_dir / "mmar-audio").resolve()
    args.model_dir = args.model_dir.resolve()
    args.checkpoint = (args.checkpoint or args.model_dir /
                       runtime_adapter.release_manifest()["checkpoint"]["path"]).resolve()
    args.output_dir = (args.output_dir or OUTPUT_ROOT / (
        "mmau_test_mini" if args.benchmark == "mmau"
        else "mmar")).resolve()
    if args.parquet_batch_size < 1:
        parser.error("--parquet-batch-size must be positive")
    return args


def _identity(args, scorer_key):
    args.bundle_audit = runtime_adapter.audit_bundle(args.model_dir, args.checkpoint)
    sources = {
        "official_runner": Path(__file__),
        "official_loader": Path(runtime_adapter.__file__),
        "base_runner": Path(base.__file__),
        "base_runtime": Path(base.runtime_adapter.__file__),
        "mmau_reference_io": Path(mmau.__file__),
        "mmar_reference_io": Path(mmar.__file__),
    }
    model_sources = {path.relative_to(runtime_adapter.ROUTE_ROOT).as_posix():
                     _sha256_json(path.read_text(encoding="utf-8").replace("\r\n", "\n"))
                     for path in sorted(runtime_adapter.ROUTE_ROOT.rglob("*.py"))}
    return {
        "benchmark": args.benchmark, "model": "official_mizar_stage3_q4_seed20260905",
        "checkpoint": str(args.checkpoint), "model_dir": str(args.model_dir),
        "official_release": args.bundle_audit,
        "model_implementation": str(runtime_adapter.ROUTE_ROOT),
        "model_sources": model_sources,
        "evaluation_code": {name: runtime_adapter.sha256_file(path) for name, path in sources.items()},
        "dataset_dir": str(args.dataset_dir), "metadata_json": str(args.metadata_json),
        "metadata_sha256": runtime_adapter.sha256_file(args.metadata_json),
        "parquet": str(args.parquet) if args.parquet else None,
        "parquet_sha256": runtime_adapter.sha256_file(args.parquet) if args.parquet else None,
        "audio_root": str(args.audio_root), "evaluation_script": str(args.evaluation_script),
        "evaluation_sha256": runtime_adapter.sha256_file(args.evaluation_script),
        "prediction_key": scorer_key, "generation": dict(runtime_adapter.PROTOCOL),
        "identity_sha256": None,
    }


def _call_model_loader(args):
    return runtime_adapter.load_runtime(args, args.bundle_audit)


def _validate_input_paths(args):
    required = [(args.model_dir, "official model directory"), (args.checkpoint, "official checkpoint"),
                (runtime_adapter.ROUTE_ROOT, "model implementation"),
                (args.metadata_json, "official metadata"), (args.evaluation_script, "official scorer")]
    if args.benchmark == "mmar":
        required.append((args.audio_root, "audio root"))
    for path, label in required:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.benchmark == "mmau":
        if not args.parquet.is_file():
            raise FileNotFoundError(f"MMAU parquet not found: {args.parquet}")
        if args.audio_root.exists() and not args.audio_root.is_dir():
            raise NotADirectoryError(f"MMAU audio root is not a directory: {args.audio_root}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    report: dict[str, Any] = {"status": "FAILED", "benchmark": args.benchmark,
                              "mode": "full", "protocol": dict(runtime_adapter.PROTOCOL),
                              "official_artifact_audit": {}, "model_audit": {},
                              "records": {}, "official_evaluation": {}, "fatal_error": None}
    store = None
    output_claimed = False
    official_records: list[dict[str, Any]] = []
    try:
        _validate_input_paths(args)
        if args.benchmark == "mmau":
            import pyarrow.parquet as pq
            parquet_rows = int(pq.ParquetFile(args.parquet).metadata.num_rows)
            if parquet_rows != EXPECTED_ROWS:
                raise ValueError(f"expected {EXPECTED_ROWS} MMAU parquet rows, got {parquet_rows}")
        output_key = scorer_prediction_key(args.evaluation_script)
        identity = _identity(args, output_key)
        if args.benchmark == "mmau":
            identity["audio_source_policy"] = "prefer_id_wav_if_present_else_parquet_embedded"
            identity["audio_root_available"] = args.audio_root.is_dir()
        identity["identity_sha256"] = _sha256_json({k: v for k, v in identity.items() if k != "identity_sha256"})
        claim_output(args, identity)
        output_claimed = True
        report["run_identity"] = identity
        official_records, artifact_audit, output_key = _metadata_records(args)
        if len(official_records) != EXPECTED_ROWS:
            raise ValueError(f"expected exactly {EXPECTED_ROWS} official rows, got {len(official_records)}")
        if output_key != identity["prediction_key"]:
            raise ValueError(f"official scorer prediction key changed during audit: {output_key!r}")
        report["official_artifact_audit"] = artifact_audit
        report["protocol"]["prediction_key"] = output_key
        runtime = _call_model_loader(args)
        report["model_audit"] = runtime[5]
        if runtime[5]["release"] != runtime_adapter.release_manifest()["release"]:
            raise ValueError("loaded model is not the pinned official final release")
        route_modules = runtime[6]
        store = mmau.ProgressStore(args.output_dir)
        report["resumption"] = {"recovered_jsonl_errors": store.load_errors}

        if args.benchmark == "mmau":
            metadata_index = mmau.build_metadata_index(official_records)
            audio_decoder = make_mmau_audio_decoder(route_modules)
            row_iter = mmau.iter_parquet_rows(args.parquet, batch_size=args.parquet_batch_size)
        else:
            row_iter = enumerate(official_records)

        rows_read = 0
        for row_index, row in row_iter:
            rows_read += 1
            if args.benchmark == "mmau":
                sample_id, _ = mmau.extract_row_id(row)
            else:
                sample_id = str(row.get("id", "")).strip()
            if store.has_terminal(row_index, sample_id):
                continue
            try:
                if args.benchmark == "mmau":
                    sample = mmau._prepare_metadata_row(
                        row, metadata_index=metadata_index, dataset_dir=args.dataset_dir,
                        prompt_builder=mmau.build_fixed_order_prompt,
                        audio_decoder=audio_decoder, audio_root=args.audio_root,
                        prefer_official_audio_file=True)
                else:
                    sample = _prepare_mmar(row, row_index, args.audio_root)
                generation = generate(runtime, sample)
                sample_record = {
                    "status": "generated", "row_index": row_index, "id": sample["id"],
                    "row_key": mmau.row_key(row_index, sample["id"]),
                    "question": sample["question"], "choices": sample["choices"],
                    "prompt": str(sample["prompt"]).lower(),
                    "official_record": sample["official_record"],
                    "audio_payload_source": sample.get("audio_payload_source"),
                    **{key: value for key, value in sample.items() if key.startswith("audio_")},
                    "generated_text": generation["generated_text"],
                    "generation_contract": dict(runtime_adapter.PROTOCOL), **generation,
                }
                store.append_raw(sample_record)
            except Exception as exc:
                report["fatal_error"] = {"row_index": row_index, "id": sample_id,
                                         "error": repr(exc), "traceback": traceback.format_exc()}
                raise RuntimeError(f"generation failed at row {row_index}, ID={sample_id!r}") from exc
            if rows_read % 10 == 0 or rows_read == EXPECTED_ROWS:
                print(json.dumps({"benchmark": args.benchmark, "rows_read": rows_read,
                                  "generated": len(store.state), "last_id": sample["id"],
                                  "last_tokens": generation["generated_token_count"]}), flush=True)

        if rows_read != EXPECTED_ROWS:
            raise RuntimeError(f"benchmark traversal incomplete: rows={rows_read}, expected={EXPECTED_ROWS}")
        counts = mmau._counts(store)
        report["records"] = {**counts, "rows_read": rows_read,
                             "official_scoring_denominator": EXPECTED_ROWS}
        if counts.get("generated", 0) != EXPECTED_ROWS or counts.get("skipped", 0):
            raise RuntimeError(f"inference coverage is incomplete: {counts}")
        predictions = _materialize(store.state.values(), official_records, output_key, args.benchmark)
        if len(predictions) != EXPECTED_ROWS:
            raise RuntimeError(f"official prediction denominator changed: {len(predictions)}")
        prediction_path = _write_predictions(args, predictions)
        report["inference_coverage"] = {"status": "PASS", "rows": rows_read,
                                        "generated": counts["generated"], "skipped": 0}
        report["official_evaluation"] = run_official_scorer(args, prediction_path, EXPECTED_ROWS)
        report["status"] = "PASS" if report["official_evaluation"]["status"] == "PASS" else "FAILED"
    except Exception as exc:
        report["fatal_error"] = report.get("fatal_error") or {
            "error": repr(exc), "traceback": traceback.format_exc()}
        print(json.dumps({"event": "EVALUATION_FAILURE", "benchmark": args.benchmark,
                          "output_claimed": output_claimed,
                          "fatal_error": report["fatal_error"]},
                         ensure_ascii=False), file=sys.stderr, flush=True)
        report["inference_coverage"] = {"status": "FAILED", "error": repr(exc)}
        if store is not None:
            report["records"] = mmau._counts(store)
            report["resumption"]["completed_rows"] = len(store.state)
        report["status"] = "FAILED"
    finally:
        if store is not None:
            store.close()
        report["elapsed_seconds"] = time.time() - started
        if output_claimed:
            mmau._write_json(args.output_dir / "evaluation_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps({"benchmark": args.benchmark, "status": report["status"],
                      "records": report.get("records"),
                      "official_evaluation": report.get("official_evaluation"),
                      "fatal_error": report.get("fatal_error"),
                      "report": (str(args.output_dir / "evaluation_report.json")
                                 if (args.output_dir / "evaluation_report.json").is_file() else None)},
                     ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
