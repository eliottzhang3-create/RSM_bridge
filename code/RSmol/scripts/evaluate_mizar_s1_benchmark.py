#!/usr/bin/env python3
"""Full deterministic MMAU test-mini or MMAR evaluation for Mizar S1."""
from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as mmau  # noqa: E402
import evaluate_mmar_5_10x2_5_mesh_mellow as mmar  # noqa: E402
import mizar_s1_eval_runtime as runtime_adapter  # noqa: E402

EXPECTED_ROWS = 1000
PREDICTION_KEYS = {"model_output", "model_prediction", "answer_prediction"}
DEFAULT_MMAU_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini")
DEFAULT_MMAR_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAR")
OUTPUT_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_htsat_5090/formal_async_v3/eval")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("mmau", "mmar"))
    parser.add_argument("--checkpoint", type=Path, default=runtime_adapter.DEFAULT_CHECKPOINT)
    parser.add_argument("--runtime-config", type=Path, default=runtime_adapter.DEFAULT_RUNTIME)
    parser.add_argument("--route-root", type=Path, default=runtime_adapter.ROUTE_ROOT)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--evaluation-script", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parquet-batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    args.expected_step = 122937
    args.dataset_dir = (args.dataset_dir or (DEFAULT_MMAU_DIR if args.benchmark == "mmau" else DEFAULT_MMAR_DIR)).resolve()
    args.metadata_json = (args.metadata_json or args.dataset_dir / ("mmau-test-mini.json" if args.benchmark == "mmau" else "MMAR-meta.json")).resolve()
    args.evaluation_script = (args.evaluation_script or args.dataset_dir / ("evaluation.py" if args.benchmark == "mmau" else "code/evaluation.py")).resolve()
    if args.benchmark == "mmau":
        args.parquet = (args.parquet or args.dataset_dir / "test_mini.parquet").resolve()
        args.audio_root = (args.audio_root or args.dataset_dir / "test-mini-audios").resolve()
    else:
        args.parquet = None
        args.audio_root = (args.audio_root or args.dataset_dir / "mmar-audio").resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.runtime_config = args.runtime_config.resolve()
    args.route_root = args.route_root.resolve()
    args.output_dir = (args.output_dir or OUTPUT_ROOT / ("mmau_test_mini_step122937" if args.benchmark == "mmau" else "mmar_step122937")).resolve()
    if args.parquet_batch_size < 1:
        parser.error("--parquet-batch-size must be positive")
    return args


def scorer_prediction_key(path: Path) -> str:
    """Read the field actually consumed by the pinned official scorer."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = {
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "output_key" for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    if len(values) != 1 or not values <= PREDICTION_KEYS:
        raise ValueError(f"cannot identify one supported official output_key in {path}: {sorted(values)}")
    return next(iter(values))


def _sha256_json(value: Any) -> str:
    import hashlib

    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def claim_output(args: argparse.Namespace, identity: Mapping[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "run_config.json"
    if path.is_file():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior != dict(identity):
            raise RuntimeError(f"output directory belongs to another checkpoint/dataset/protocol: {args.output_dir}")
    elif any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing a non-empty unowned output directory: {args.output_dir}")
    else:
        mmau._write_json(path, identity)


def _unwrap_path(payload: Any) -> Any:
    if isinstance(payload, Mapping) and payload.get("path"):
        return payload["path"]
    return payload


def make_mmau_audio_decoder(route_modules):
    """Decode full audio at 32 kHz; fixed20 cropping happens in generation."""
    fixed20 = route_modules["data.fixed20"]
    half_samples = 10 * 32000

    def decode(payload: Any, *, base_dir: Path):
        import torch
        import torchaudio

        value = _unwrap_path(payload)
        if isinstance(value, (str, Path)):
            path = mmau._resolve_audio_path(str(value), base_dir)
            if path.is_file():
                waveform = fixed20.load_mono(path)
                source = f"training_soundfile:{path}"
                original_rate = None
            else:
                raise FileNotFoundError(f"MMAU audio file not found: {path}")
        else:
            waveform, original_rate, source = mmau._decode_audio_payload(
                payload, base_dir=base_dir, default_rate=mmau.DEFAULT_SOURCE_SAMPLE_RATE)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim != 2:
                raise ValueError(f"embedded MMAU audio has invalid dimensions: {tuple(waveform.shape)}")
            waveform = waveform.float().mean(dim=0, keepdim=True)
            if original_rate != 32000:
                waveform = torchaudio.functional.resample(waveform, original_rate, 32000)
            waveform = waveform[0].contiguous()
        if waveform.ndim != 1 or waveform.numel() == 0 or not bool(torch.isfinite(waveform).all()):
            raise ValueError("MMAU audio decoder produced invalid mono waveform")
        duration = waveform.numel() / 32000
        return waveform, {
            "audio_source": source,
            "audio_original_sample_rate": original_rate,
            "audio_original_duration_seconds": duration,
            "audio_target_sample_rate": 32000,
            "audio_full_waveform_samples": int(waveform.numel()),
            "audio_fixed20_half_samples": half_samples,
            "audio_crop_policy": "center_crop_before_split_or_right_pad_to20s",
            "audio_mono": True,
        }

    return decode


def _fixed20_pair(runtime: tuple, sample: Mapping[str, Any]):
    import torch

    fixed20 = runtime[4]
    audio_path = sample.get("audio_path_resolved")
    if audio_path:
        return fixed20.load_fixed20(audio_path)
    waveform = sample["waveform"]
    if waveform.ndim == 2:
        if waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform[0]
    if waveform.ndim != 1:
        raise ValueError(f"expected a mono waveform, got {tuple(waveform.shape)}")
    waveform = torch.as_tensor(waveform, dtype=torch.float32).contiguous()
    whole = fixed20.fixed_len(waveform, 20 * 32000)
    return whole[:10 * 32000].contiguous(), whole[10 * 32000:].contiguous()


def generate(runtime: tuple, sample: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    model, tokenizer, device, text_encoder, _, _ = runtime
    prompt = str(sample["prompt"]).lower()
    text_input = text_encoder.tokenize(prompt, runtime_adapter.PROMPT_TOKENS)
    prompt_original_count = len(tokenizer.encode(prompt + " <|endoftext|>", add_special_tokens=True))
    first, second = _fixed20_pair(runtime, sample)
    if (first.numel(), second.numel()) != (320000, 320000):
        raise RuntimeError("fixed20 audio halves must each contain exactly 10 seconds at 32 kHz")
    started = time.monotonic()
    with torch.inference_mode():
        prefix, _, _ = model.generate_prefix_inference({
            "audio1": first.unsqueeze(0).to(device),
            "audio2": second.unsqueeze(0).to(device),
            "input": {key: value.to(device) for key, value in text_input.items()},
        })
        generated = runtime_adapter.greedy_decode(model, tokenizer, prefix)
    generated.update({
        "prefix_shape": list(prefix.shape),
        "prefix_layout": "audio129+129_then_one_sep_then_prompt256",
        "answer_start_position": 515,
        "first_answer_logit_position": 514,
        "prompt_original_token_count": prompt_original_count,
        "prompt_truncated": prompt_original_count > 256,
        "prompt_padding_visible": True,
        "prompt_pad_tokens": int(text_input["attention_mask"].eq(0).sum()),
        "prompt_padding_token_id": int(tokenizer.pad_token_id),
        "audio_half_samples": [int(first.numel()), int(second.numel())],
        "audio_policy": "fixed20_center_crop_or_right_pad_then_split10",
        "audio2_reused": False,
        "generation_seconds": time.monotonic() - started,
    })
    return generated


def _metadata_records(args: argparse.Namespace):
    if args.benchmark == "mmau":
        records = mmau.load_official_records(args.metadata_json)
        audit = mmau.audit_mmau_v051525(args.metadata_json, args.evaluation_script, records)
        if audit["status"] != "PASS":
            raise ValueError(f"MMAU official metadata/scorer audit failed: {audit['failures']}")
        return records, audit, scorer_prediction_key(args.evaluation_script)
    records = mmar.load_mmar_records(args.metadata_json)
    audit = mmar.audit_mmar_artifacts(args.metadata_json, args.audio_root,
                                      args.evaluation_script, records)
    if audit["status"] != "PASS":
        raise ValueError(f"MMAR official metadata/scorer audit failed: {audit['failures']}")
    return records, audit, str(audit["evaluation_semantic_audit"]["output_key"])


def _prepare_mmar(record: Mapping[str, Any], row_index: int, audio_root: Path) -> dict[str, Any]:
    choices = [str(choice) for choice in record["choices"]]
    path = mmar.resolve_mmar_audio_path(audio_root, record["audio_path"])
    return {"id": str(record.get("id", "")).strip(), "row_index": row_index,
            "question": str(record["question"]), "choices": choices,
            "answer": str(record["answer"]),
            "prompt": mmau.build_fixed_order_prompt(str(record["question"]), choices),
            "official_record": copy.deepcopy(dict(record)),
            "audio_path_resolved": str(path), "audio_payload_source": "official_audio_path"}


def _materialize(state, official_records, output_key: str, benchmark: str) -> list[dict[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in state:
        if item.get("status") != "generated":
            continue
        sample_id = str(item.get("id", "")).strip()
        if sample_id in by_id:
            raise RuntimeError(f"multiple completed predictions for benchmark ID {sample_id!r}")
        by_id[sample_id] = item
    predictions = []
    for record in official_records:
        sample_id = (mmau._official_id(record) if benchmark == "mmau" else str(record.get("id", "")).strip())
        completed = by_id.get(sample_id)
        output = copy.deepcopy(dict(record))
        # Preserve the actual generation string byte-for-byte at the Python
        # string level. Skipped records cannot be scored as if generated.
        output[output_key] = "" if completed is None else str(completed["generated_text"])
        predictions.append(output)
    return predictions


def _write_predictions(args, predictions):
    filename = "predictions_fixed_order.json" if args.benchmark == "mmau" else "predictions_official.json"
    mmau._write_json(args.output_dir / filename, predictions)
    return args.output_dir / filename


def run_official_scorer(args, prediction_path: Path, expected_rows: int) -> dict[str, Any]:
    command = [sys.executable, str(args.evaluation_script), "--input", str(prediction_path)]
    completed = subprocess.run(command, cwd=args.evaluation_script.parent,
                               capture_output=True, text=True, check=False)
    log = args.output_dir / "official_evaluation.txt"
    log.write_text(f"$ {' '.join(command)}\nreturncode: {completed.returncode}\n\n"
                   f"===== STDOUT =====\n{completed.stdout}\n===== STDERR =====\n{completed.stderr}\n",
                   encoding="utf-8")
    totals = [int(value) for value in re.findall(r"over\s+(\d+)\s+samples", completed.stdout)]
    scores = re.findall(r"Total Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%\s+over\s+(\d+)\s+samples", completed.stdout)
    reported = totals[-1] if totals else None
    passed = completed.returncode == 0 and reported == expected_rows and bool(scores)
    return {"status": "PASS" if passed else "FAILED", "returncode": completed.returncode,
            "prediction_path": str(prediction_path), "prediction_key": scorer_prediction_key(args.evaluation_script),
            "expected_rows": expected_rows, "reported_rows": reported,
            "total_accuracy_percent": float(scores[-1][0]) if scores else None,
            "official_scorer": str(args.evaluation_script), "official_only": True}


def _identity(args: argparse.Namespace, scorer_key: str) -> dict[str, Any]:
    evaluation_code = {
        "benchmark_runner_sha256": runtime_adapter.sha256_file(Path(__file__)),
        "runtime_adapter_sha256": runtime_adapter.sha256_file(Path(runtime_adapter.__file__)),
        "mmau_reference_io_sha256": runtime_adapter.sha256_file(Path(mmau.__file__)),
        "mmar_reference_io_sha256": runtime_adapter.sha256_file(Path(mmar.__file__)),
    }
    return {
        "benchmark": args.benchmark, "checkpoint": str(args.checkpoint),
        "checkpoint_size": args.checkpoint.stat().st_size,
        "checkpoint_mtime_ns": args.checkpoint.stat().st_mtime_ns,
        "checkpoint_sha256": runtime_adapter.sha256_file(args.checkpoint),
        "runtime_config": str(args.runtime_config),
        "runtime_config_sha256": runtime_adapter.sha256_file(args.runtime_config),
        "route_root": str(args.route_root), "dataset_dir": str(args.dataset_dir),
        "metadata_json": str(args.metadata_json), "metadata_sha256": runtime_adapter.sha256_file(args.metadata_json),
        "parquet": str(args.parquet) if args.parquet else None,
        "parquet_sha256": runtime_adapter.sha256_file(args.parquet) if args.parquet else None,
        "audio_root": str(args.audio_root), "evaluation_script": str(args.evaluation_script),
        "evaluation_sha256": runtime_adapter.sha256_file(args.evaluation_script),
        "evaluation_code": evaluation_code,
        "prediction_key": scorer_key, "generation": dict(runtime_adapter.PROTOCOL),
        "identity_sha256": None,
    }


def _call_model_loader(args):
    return runtime_adapter.load_runtime(args)


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
        for path, label in ((args.checkpoint, "checkpoint"), (args.runtime_config, "runtime config"),
                            (args.route_root, "training route"), (args.audio_root, "audio root"),
                            (args.metadata_json, "official metadata"),
                            (args.evaluation_script, "official scorer")):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        if args.benchmark == "mmau" and not args.parquet.is_file():
            raise FileNotFoundError(f"MMAU parquet not found: {args.parquet}")
        if args.benchmark == "mmau":
            import pyarrow.parquet as pq
            parquet_rows = int(pq.ParquetFile(args.parquet).metadata.num_rows)
            if parquet_rows != EXPECTED_ROWS:
                raise ValueError(f"expected {EXPECTED_ROWS} MMAU parquet rows, got {parquet_rows}")
        output_key = scorer_prediction_key(args.evaluation_script)
        identity = _identity(args, output_key)
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
        if runtime[5]["checkpoint_step"] != args.expected_step:
            raise ValueError("loaded checkpoint step is not the requested final training step")
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
                      "report": str(args.output_dir / "evaluation_report.json")},
                     ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
