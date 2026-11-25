#!/usr/bin/env python3
"""Full 5000-row MMSU evaluation, compact lowercase prompt and existing generation."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mizar_s1_ced_mmsu_eval import dataset, parsing, scoring, storage  # noqa: E402
from . import mmsu_runtime as runtime
from . import arguments, runtime as model_runtime

DEFAULT_MMSU = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/MMSU")


def parse_args(argv=None):
    return arguments.parse_args('mmsu', argv)



def file_identity(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": storage.file_sha(path)}


def identity(args, paths):
    model_runtime.read_checkpoint(args)
    selected, family, _ = runtime.adapter(args.eval_variant)
    sources = [*Path(__file__).parent.glob("*.py"), *Path(__file__).parent.glob("*.sh"),
               ROOT / "run_mizar_s1_ced_small_mmsu_eval_5090.sh",
               Path(selected.__file__), Path(runtime.base.__file__)]
    sources.extend((ROOT / "scripts").glob("mizar_s1_ced_*eval_runtime.py"))
    # Several existing route modules import shared recursive implementations.
    sources.extend(ROOT.glob("recursive_model*.py"))
    if family == "multilayer" and selected.variant_spec(args.eval_variant).get("official"):
        for directory in ROOT.glob("mizar_official_recursive_5_20x2_5*"):
            sources.extend(directory.rglob("*.py"))
    model_sources = {p.relative_to(args.route_root).as_posix(): storage.text_sha(p)
                     for p in sorted(args.route_root.rglob("*.py"))}
    if not model_sources:
        raise FileNotFoundError(f"missing selected training implementation: {args.route_root}")
    result = {"benchmark": "mmsu", "mode": "full", "expected_rows": dataset.EXPECTED_ROWS,
              "eval_variant": args.eval_variant, "runtime_family": family,
              "training_route": model_runtime.variant_spec(args.eval_variant)['route'],
              "source_revision": model_runtime.variant_spec(args.eval_variant)['revision'],
              "checkpoint": file_identity(args.checkpoint),
              "runtime_config": file_identity(args.runtime_config) if args.runtime_config else None,
              "expected_step": args.expected_step,
              "expected_geometry": list(args.expected_geometry) if args.expected_geometry else None,
              "route_root": str(args.route_root), "model_sources": model_sources,
              "evaluation_code": model_runtime.evaluation_sources(),
              "dataset_dir": str(args.dataset_dir), "shards": [file_identity(p) for p in paths],
              "scorer": scoring.audit_scorer(args.evaluation_script),
              "protocol": runtime.PROTOCOL,
              "audio_source_policy": "embedded_bytes_before_path_no_hub_no_export"}
    if family == "official":
        result["official_release"] = selected.audit_bundle(args.model_dir, args.checkpoint)
        result["evaluation_code"]["mizar_official_eval/release_manifest.json"] = storage.text_sha(selected.MANIFEST_PATH)
    result["identity_sha256"] = storage.object_sha(result)
    return result


def summarize(saved):
    generations = [row["generation"] for row in saved]
    return {"generated": len(saved), "skipped": 0,
            "prompt_truncated_rows": sum(bool(g.get("prompt_truncated")) for g in generations),
            "audio_center_cropped_rows": sum(g.get("audio_cropped_samples", 0) > 0 for g in generations),
            "audio_right_padded_rows": sum(g.get("audio_right_pad_samples", 0) > 0 for g in generations),
            "max_original_prompt_tokens": max((g.get("prompt_original_token_count", 0)
                                                for g in generations), default=0)}


def execute(args, report):
    paths = dataset.shard_paths(args.dataset_dir)
    run_identity = identity(args, paths)
    storage.claim_output(args.output_dir, run_identity)
    report["output_claimed"] = True
    report["run_identity"] = run_identity
    records, report["dataset_audit"] = dataset.preflight(paths, args.parquet_batch_size)
    store = storage.GenerationStore(args.output_dir, records)
    report["resumed_rows"] = len(store.state)
    report["records"] = summarize(store.state)
    # Re-scoring a complete run does not load weights or consume GPU inference.
    loaded = None
    if len(store.state) < dataset.EXPECTED_ROWS:
        loaded = runtime.load_runtime(args)
        report["model_audit"] = loaded[5]
        # Preserve loader/forward audits across a complete-run re-score.
        storage.write_json(args.output_dir / "model_audit.json", loaded[5])
        rows_read = 0
        try:
            for index, row in dataset.iter_rows(paths, audio=True, batch_size=args.parquet_batch_size):
                if index != rows_read or index >= len(records):
                    raise ValueError(f"unexpected MMSU global row index: {index}")
                rows_read += 1
                original = {key: row[key] for key in dataset.METADATA}
                if original != records[index]:
                    raise ValueError(f"MMSU metadata changed between preflight and inference: {index}")
                if index < len(store.state):
                    continue
                waveform, audio_audit = runtime.decode_audio(row["audio"], args.dataset_dir)
                generation = runtime.generate(loaded, {"prompt": dataset.prompt(original),
                    "waveform": waveform, "audio_audit": audio_audit}, variant=args.eval_variant)
                store.append(index, original, generation)
                if len(store.state) % 25 == 0:
                    print(json.dumps({"benchmark": "mmsu", "generated": len(store.state),
                                      "total": dataset.EXPECTED_ROWS, "last_id": original["id"]}), flush=True)
            if rows_read != dataset.EXPECTED_ROWS:
                raise RuntimeError(f"incomplete MMSU traversal: {rows_read}")
        finally:
            report["records"] = summarize(store.state)
            storage.write_json(args.output_dir / "model_audit.json", loaded[5])
    else:
        report["model_audit"] = json.loads((args.output_dir / "model_audit.json").read_text(encoding="utf-8"))
    if len(store.state) != dataset.EXPECTED_ROWS:
        raise RuntimeError("inference coverage incomplete; official scoring withheld")
    report["inference_coverage"] = {"status": "PASS", "rows": len(store.state), "skipped": 0}
    predictions = args.output_dir / "predictions_official.jsonl"
    scoring.write_predictions(predictions, store.state)
    # Keep the v1 raw artifact; adapt only a separate scorer input copy.
    report["response_parsing"], report["official_evaluation"] = parsing.score_records(
        parsing.read_complete_jsonl(predictions), args.evaluation_script, args.output_dir)
    report["status"] = report["official_evaluation"]["status"]


def run(args):
    started = time.monotonic()
    report = {"status": "FAILED", "benchmark": "mmsu", "mode": "full",
              "protocol": runtime.PROTOCOL, "output_claimed": False,
              "inference_coverage": {"status": "FAILED"},
              "official_evaluation": {"execution_status": "NOT_RUN", "status": "NOT_RUN"},
              "records": {}, "fatal_error": None}
    # Lock ownership encloses both exception handling and final report writing.
    try:
        with storage.output_lock(args.output_dir):
            try:
                execute(args, report)
            except Exception as exc:
                report["status"] = "FAILED"
                report["fatal_error"] = {"error": repr(exc), "traceback": traceback.format_exc()}
                print(json.dumps({"event": "MMSU_EVALUATION_FAILURE", **report["fatal_error"]}),
                      file=sys.stderr, flush=True)
            finally:
                report["elapsed_seconds"] = time.monotonic() - started
                if report["output_claimed"]:
                    storage.write_json(args.output_dir / "evaluation_report.json", report)
    except Exception as exc:
        report["status"] = "FAILED"
        report["fatal_error"] = {"error": repr(exc), "traceback": traceback.format_exc()}
    return report


def main(argv=None):
    args = parse_args(argv)
    if args.check_args:
        print(json.dumps(arguments.printable(args), default=str))
        return 0
    try:
        report = run(args)
        print(json.dumps({'status': report['status'], 'records': report['records'],
                          'official_evaluation': report['official_evaluation'],
                          'fatal_error': report['fatal_error'],
                          'report': str(args.output_dir / 'evaluation_report.json')
                          if report['output_claimed'] else None}), flush=True)
        return 0 if report['status'] == 'PASS' else 1
    finally:
        vars(args).pop('_checkpoint_payload', None)



if __name__ == "__main__":
    raise SystemExit(main())
