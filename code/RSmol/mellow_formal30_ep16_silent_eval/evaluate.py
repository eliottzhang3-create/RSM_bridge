#!/usr/bin/env python3
"""Isolated MMAU/MMAR evaluation of formal30 epoch-16, silent-slot checkpoints."""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))
    __package__ = ROOT.name
for path in (ROOT, ROOT.parent / "scripts", ROOT.parent):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from . import contracts as C
from . import audio as eval_audio
from . import runtime
import evaluate_mmau_test_mini_5_10x2_5_mesh_mellow as mmau
import evaluate_mmar_5_10x2_5_mesh_mellow as mmar


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("mmau_test_mini", "mmar"), required=True)
    parser.add_argument("--checkpoint-file", required=True,
                        help="full .ckpt path, or smollm2 / mesh_x2 / mesh_x4 alias")
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    args = parser.parse_args(argv)
    aliases = C.checkpoint_aliases()
    args.checkpoint_argument = args.checkpoint_file
    args.requested_alias = args.checkpoint_file if args.checkpoint_file in aliases else None
    args.requested_alias_architecture = args.requested_alias
    args.checkpoint_file = Path(aliases.get(args.checkpoint_file, args.checkpoint_file)).expanduser().resolve(strict=True)
    run_root = args.checkpoint_file.parent.parent.parent
    if args.runtime_config is None:
        args.runtime_config = run_root / "runtime_30epochs.yaml"
    try:
        args.runtime_config = args.runtime_config.resolve(strict=True)
    except FileNotFoundError as exc:
        parser.error(f"runtime config does not exist: {exc.filename}")
    if args.runtime_config.parent != run_root:
        parser.error("runtime config must belong to the checkpoint run root")
    args.output_dir = (args.output_dir or run_root / "eval_formal30_ep16_silent_audio2_v2" / args.task /
                       args.checkpoint_file.parent.name / args.checkpoint_file.stem / "full").resolve()
    args.dataset_dir = (args.dataset_dir or Path("/hpc_stor03/sjtu_home/jinwei.zhang/data") /
                        ("MMAU_test_mini" if args.task == "mmau_test_mini" else "MMAR")).resolve(strict=True)
    args.metadata_json = args.dataset_dir / ("mmau-test-mini.json" if args.task == "mmau_test_mini" else "MMAR-meta.json")
    args.evaluation_script = args.dataset_dir / ("evaluation.py" if args.task == "mmau_test_mini" else "code/evaluation.py")
    args.audio_root = args.dataset_dir / ("test-mini-audios" if args.task == "mmau_test_mini" else "mmar-audio")
    args.parquet = args.dataset_dir / "test_mini.parquet"
    args.mode, args.dtype = "full", "fp32"
    args.max_prompt_tokens, args.max_new_tokens = C.MAX_PROMPT_TOKENS, C.MAX_NEW_TOKENS
    args.parquet_batch_size = 8
    args.run_official_evaluation = True
    args.checkpoint = args.checkpoint_file.parent  # existing traversal expects a directory
    return args


def prepare(args: Any) -> None:
    import torch
    import yaml

    print(f"[ep16-silent-eval] checkpoint argument: {args.checkpoint_argument}", flush=True)
    print(f"[ep16-silent-eval] validating checkpoint: {args.checkpoint_file}", flush=True)
    config = yaml.safe_load(args.runtime_config.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint_file, map_location="cpu", weights_only=False)
    if not isinstance(config, dict) or not isinstance(checkpoint, dict):
        raise ValueError("runtime and checkpoint must both be mappings")
    args.architecture, args.model_contract = C.validate_contract(checkpoint, config)
    if args.requested_alias_architecture and args.architecture != args.requested_alias_architecture:
        raise ValueError("checkpoint alias does not match the embedded architecture contract")
    args.runtime_config_data = config
    args.loaded_checkpoint = checkpoint
    args.route_root = C.ROUTE_PARENT / C.route_for_architecture(
        args.architecture, args.model_contract["training_route_variant"])
    args.mellow_root = args.route_root
    args.htsat_checkpoint = Path(config["model"]["encoder"]["pretrained_audioencoder_path"]) / "HTSAT_AudioSet_Saved_1.ckpt"
    args.loaded_model_audit = {}
    args.checkpoint_audit = {
        "schema_version": checkpoint["schema_version"], "epoch_completed": checkpoint["epoch_completed"],
        "num_epochs": checkpoint["num_epochs"], "total_step": checkpoint["total_step"],
        "batch_geometry": checkpoint["batch_geometry"], "loss_reduction": checkpoint["loss_reduction"],
        "route_contract": checkpoint.get("route_contract"),
        "text_model_contract": checkpoint.get("text_model_contract"),
        "route_variant": checkpoint.get("route_variant"),
        "max_lr": checkpoint.get("scheduler", {}).get("max_lr"),
        "min_lr": checkpoint.get("scheduler", {}).get("min_lr"),
    }
    print(f"[ep16-silent-eval] architecture={args.architecture} "
          f"epoch_completed={checkpoint['epoch_completed']} num_epochs={checkpoint['num_epochs']} "
          f"current_step={checkpoint['total_step']} scheduler_total_steps={checkpoint['scheduler']['total_steps']}", flush=True)


def audit_dataset(args: Any) -> tuple[list[dict], dict, dict]:
    """Load scoring metadata without walking or hashing benchmark audio."""
    if args.task == "mmau_test_mini":
        records = mmau.load_official_records(args.metadata_json)
        audio_source_policy = "prefer_official_id_wav_else_parquet_embedded"
    else:
        records = mmar.load_mmar_records(args.metadata_json)
        audio_source_policy = "official_audio_path_required"
    # The task evaluator performs metadata/scorer checks once, then resolves,
    # decodes, and hashes each waveform only when that sample is processed.
    # Avoid opening every WAV or the complete MMAU parquet during startup.
    identity = {
        "metadata_sha256": C.sha256_file(args.metadata_json),
        "evaluation_script_sha256": C.sha256_file(args.evaluation_script),
        "audio_source_policy": audio_source_policy,
        "audio_integrity_policy": "per_sample_decode_and_waveform_sha256",
        "audio_preflight": "deferred_until_sample_processing",
    }
    args.dataset_audio_audit = {
        "policy": audio_source_policy,
        "preflight": "deferred_until_sample_processing",
        "full_audio_file_hashing": False,
        "full_parquet_hashing": False,
        "metadata_rows": len(records),
    }
    return records, {"status": "DEFERRED_TO_TASK_EVALUATOR"}, identity


def audit_mmar_artifacts_without_audio_scan(
    metadata_path: Path,
    audio_root: Path,
    evaluation_script: Path,
    records: Sequence[dict],
) -> dict:
    """Keep MMAR metadata/scorer checks while deferring audio paths to traversal."""
    ids = [mmar._record_id(record) for record in records]
    modality_counts = Counter(str(record.get("modality", "")) for record in records)
    category_counts = Counter(str(record.get("category", "")) for record in records)
    core = [{key: record.get(key) for key in mmar.MMAR_CORE_KEYS} for record in records]
    core_sha256 = mmar.common._canonical_sha256(core)
    core_by_id = [{key: record.get(key) for key in mmar.MMAR_CORE_KEYS}
                  for record in sorted(records, key=mmar._record_id)]
    core_by_id_sha256 = mmar.common._canonical_sha256(core_by_id)
    id_set_sha256 = mmar.common._canonical_sha256(sorted(ids))
    invalid_records = []
    empty_choice_records = []
    answers_not_exact_choices = []
    for index, record in enumerate(records):
        choices = record.get("choices")
        answer = record.get("answer")
        if (not ids[index] or not str(record.get("question", "")).strip()
                or not isinstance(choices, list) or not 2 <= len(choices) <= 6
                or any(not isinstance(choice, str) for choice in choices or [])
                or not isinstance(answer, str) or not answer.strip()):
            invalid_records.append({"row_index": index, "id": ids[index],
                                    "error": "record is incompatible with the official string scorer"})
        elif any(not choice.strip() for choice in choices):
            empty_choice_records.append({"row_index": index, "id": ids[index]})
        if isinstance(choices, list) and isinstance(answer, str) and answer not in choices:
            answers_not_exact_choices.append({"row_index": index, "id": ids[index]})

    scorer_sha256 = mmar.common._sha256(evaluation_script)
    scorer_semantics = mmar._audit_mmar_scorer_semantics(evaluation_script)
    failures = []
    warnings = []
    unique_nonempty_ids = {value for value in ids if value}
    if len(records) != mmar.EXPECTED_FULL_ROWS or len(unique_nonempty_ids) != mmar.EXPECTED_FULL_ROWS:
        failures.append(f"metadata ID coverage differs: rows={len(records)} unique_nonempty={len(unique_nonempty_ids)}")
    if id_set_sha256 != mmar.MMAR_ID_SET_SHA256:
        failures.append("MMAR ID set differs from the official 1000-question release: "
                        f"expected={mmar.MMAR_ID_SET_SHA256} actual={id_set_sha256}")
    if core_sha256 != mmar.MMAR_CORE_CANONICAL_SHA256:
        warnings.append("metadata row order/fields differ from the GitHub JSONL publication; "
                        f"ordered_core_sha256={core_sha256}")
    if core_by_id_sha256 != mmar.MMAR_CORE_BY_ID_CANONICAL_SHA256:
        warnings.append("metadata core content differs from the GitHub publication after ID sorting; "
                        f"core_by_id_sha256={core_by_id_sha256}")
    if dict(sorted(modality_counts.items())) != mmar.EXPECTED_MODALITIES:
        failures.append(f"modality distribution differs: {dict(sorted(modality_counts.items()))}")
    if dict(sorted(category_counts.items())) != mmar.EXPECTED_CATEGORIES:
        failures.append(f"category distribution differs: {dict(sorted(category_counts.items()))}")
    if invalid_records:
        failures.append(f"invalid records: {invalid_records[:10]}")
    if scorer_semantics["status"] != "PASS":
        failures.extend(scorer_semantics["failures"])
    if scorer_sha256 not in {mmar.MMAR_EVALUATION_SHA256, mmar.MMAR_HF_EVALUATION_SHA256}:
        warnings.append("evaluation.py byte SHA256 is an unregistered packaging variant; "
                        f"semantic audit passed with sha256={scorer_sha256}")
    elif scorer_sha256 == mmar.MMAR_HF_EVALUATION_SHA256:
        warnings.append("using the official Hugging Face evaluation.py packaging variant")
    if empty_choice_records:
        warnings.append(f"official metadata contains {len(empty_choice_records)} records with empty distractor choices")
    if answers_not_exact_choices:
        warnings.append(f"official metadata contains {len(answers_not_exact_choices)} answers not byte-equal to a choice")
    return {
        "status": "PASS" if not failures else "FAIL",
        "official_commit": mmar.MMAR_OFFICIAL_COMMIT,
        "official_github_metadata_sha256": mmar.MMAR_GITHUB_METADATA_SHA256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": mmar.common._sha256(metadata_path),
        "core_canonical_sha256": core_sha256,
        "core_by_id_canonical_sha256": core_by_id_sha256,
        "id_set_sha256": id_set_sha256,
        "audio_root": str(audio_root),
        "audio_files_resolved": 0,
        "unique_audio_files": 0,
        "missing_audio_count": None,
        "audio_preflight": "deferred_to_sample_processing",
        "evaluation_script": str(evaluation_script),
        "evaluation_sha256": scorer_sha256,
        "evaluation_semantic_audit": scorer_semantics,
        "rows": len(records),
        "unique_ids": len(set(ids)),
        "modality_counts": dict(sorted(modality_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "empty_choice_records": empty_choice_records,
        "answers_not_exact_choices": answers_not_exact_choices,
        "warnings": warnings,
        "failures": failures,
    }


@contextlib.contextmanager
def deferred_mmar_audio_preflight():
    """Scope the lightweight MMAR audit to this isolated evaluator invocation."""
    original_audit = mmar.audit_mmar_artifacts
    mmar.audit_mmar_artifacts = audit_mmar_artifacts_without_audio_scan
    try:
        yield
    finally:
        mmar.audit_mmar_artifacts = original_audit


def build_identity(args: Any, dataset_identity: dict) -> dict:
    evaluation_files = [ROOT / name for name in ("contracts.py", "runtime.py", "evaluate.py", "audio.py")]
    evaluation_files.extend([Path(mmau.__file__), Path(mmar.__file__)])
    evaluation_sources = C.source_fingerprints(evaluation_files)
    model_sources = C.source_fingerprints(runtime.model_source_files(
        args.architecture, args.model_contract["training_route_variant"]))
    # Paths, hostname, PID, timestamps, output dir and training /dev/shm paths
    # are provenance, not scientific identity. Same bytes/semantics can resume.
    model_contract = copy.deepcopy(args.model_contract)
    model_contract["encoder"].pop("pretrained_audioencoder_path", None)
    model_contract["decoder"].pop("text_decoder", None)
    model_contract.pop("tokenizer", None)
    import transformers
    import torch
    import torchaudio
    import numpy
    import soundfile
    import scipy
    import pyarrow
    import torchlibrosa
    software = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                "torchaudio": torchaudio.__version__, "numpy": numpy.__version__,
                "soundfile": soundfile.__version__, "scipy": scipy.__version__,
                "pyarrow": pyarrow.__version__, "torchlibrosa": torchlibrosa.__version__,
                "cuda": torch.version.cuda}
    return {
        "protocol": C.protocol(args.task), "checkpoint_file_identity": file_stat_identity(args.checkpoint_file),
        "checkpoint_epoch_completed": args.checkpoint_audit["epoch_completed"],
        "training_num_epochs": args.checkpoint_audit["num_epochs"],
        "model_contract": model_contract, "model_sources": model_sources,
        "text_config": json.loads((Path(args.runtime_config_data["model"]["decoder"]["text_decoder"]) / "config.json").read_text(encoding="utf-8")),
        "baseline_text_config": json.loads((Path(args.runtime_config_data["data"]["tokenizer_type"]) / "config.json").read_text(encoding="utf-8")),
        "tokenizer_files": {path.name: C.sha256_file(path)
                            for path in sorted(Path(args.runtime_config_data["data"]["tokenizer_type"]).iterdir())
                            if path.is_file() and (path.name.startswith("tokenizer") or path.name in {
                                "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"})},
        "dataset": dataset_identity, "evaluation_sources": evaluation_sources, "software": software,
    }


def claim_output(output_dir: Path, identity: dict) -> None:
    marker = output_dir.parent / f".{output_dir.name}.ep16_silent_eval_identity.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous != identity:
            changed = sorted(key for key in set(previous) | set(identity) if previous.get(key) != identity.get(key))
            raise ValueError(f"existing evaluation belongs to different model/data/protocol: {changed}; use a new output directory")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("refusing nonempty evaluation directory without Mellow evaluation identity")
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(identity, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def file_stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@contextlib.contextmanager
def output_lock(output_dir: Path):
    import fcntl

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.ep16_silent_eval.lock"
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another evaluator is writing this output: {output_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def score_both(args: Any, records: list[dict], report: dict) -> dict:
    raw = {}
    raw_path = args.output_dir / "raw_generations.jsonl"
    if raw_path.is_file():
        entries, _ = mmau._read_jsonl(raw_path)
        for entry in entries:
            if entry.get("status") == "generated":
                sample_id = str(entry["id"])
                if sample_id in raw:
                    raise ValueError(f"duplicate raw generation for ID {sample_id}")
                raw[sample_id] = str(entry.get("generated_text", ""))
    key = "model_output" if args.task == "mmau_test_mini" else str(
        report["official_artifact_audit"]["evaluation_semantic_audit"]["output_key"])
    official_predictions, prefix_predictions = [], []
    for record in records:
        text = raw.get(str(record["id"]), "")
        item = copy.deepcopy(record)
        item[key] = text
        official_predictions.append(item)
        item = copy.deepcopy(record)
        item[key] = text
        prefix_predictions.append(item)
    filename = "predictions_fixed_order.json" if args.task == "mmau_test_mini" else "predictions_official.json"
    # The official traversal and prefix scorer must receive the same complete
    # generated text, including choice labels, parentheses and whitespace.
    actual = json.loads((args.output_dir / filename).read_text(encoding="utf-8"))
    if actual != official_predictions:
        raise ValueError("official scorer predictions differ from the unchanged raw generated text")
    mmau._write_json(args.output_dir / "predictions_choice_label_prefix.json", prefix_predictions)
    if args.task == "mmau_test_mini":
        score = mmau.evaluate_mellow_author_reply_predictions(prefix_predictions)
        mmau._write_json(args.output_dir / "choice_label_prefix_evaluation.json", score)
        lines = [f"MMAU choice-label prefix accuracy: {score['total']['accuracy_percent']:.6f}% over {score['total']['total']} samples"]
        for group in ("task", "difficulty"):
            lines.extend(f"{group}/{name}: {item['accuracy_percent']:.6f}% over {item['total']} samples"
                         for name, item in score[group].items())
        (args.output_dir / "choice_label_prefix_evaluation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        score = mmar.write_choice_label_prefix_evaluation(args.output_dir, prefix_predictions, output_key=key)
    return score


def run(args: Any) -> dict:
    prepare(args)
    records, _, dataset_identity = audit_dataset(args)
    identity = build_identity(args, dataset_identity)
    with output_lock(args.output_dir):
        claim_output(args.output_dir, identity)

        def generation(model, tokenizer, device, sample, *, max_prompt_tokens, max_new_tokens):
            return runtime.run_generation(args, mmau, model, tokenizer, device, sample,
                                          max_prompt_tokens=max_prompt_tokens, max_new_tokens=max_new_tokens)

        common_kwargs = dict(
            load_runtime_model=runtime.load_model, run_model_generation=generation,
            prepare_prediction=C.prepare_official_prediction,
            prediction_format="raw_generated_text_unchanged_v3",
            audio_prefix_tokens=C.AUDIO_PREFIX_TOKENS,
            record_static_fields={"architecture": args.architecture, "audio2_reused": False,
                                  "audio2_waveform_materialized": True,
                                  "audio_encoder_invocations": 2, "single_audio_slot": False,
                                  "audio2_policy": C.AUDIO2_POLICY},
            stage="formal30_ep16_silent_" + args.task,
        )
        if args.task == "mmau_test_mini":
            report = mmau.run(
                args, **common_kwargs, prompt_builder=C.build_prompt,
                audio_decoder=eval_audio.decode_training_audio,
                audio_root=args.audio_root, prefer_official_audio_file=True,
                prompt_format="preserve_body_case_fixed_order_a_to_j",
                audio_format="training_first_two_channel_mean_mellow_repeat_or_per_id_seeded_random_crop",
                protocol_contract=C.PROTOCOL_VERSION,
                generation_protocol=C.protocol(args.task),
                logical_trace=args.architecture,
            )
        else:
            # MMAR's fixed-order prompt helper preserves case and has identical
            # formatting; generation asserts exact equality to C.build_prompt.
            with deferred_mmar_audio_preflight():
                report = mmar.run(
                    args, **common_kwargs,
                    protocol_description=C.PROTOCOL_VERSION + ": " + json.dumps(C.protocol(args.task), sort_keys=True),
                )
        report["protocol"] = {**report.get("protocol", {}), **C.protocol(args.task)}
        report["protocol"]["prompt_format"] = "preserve_body_case_fixed_order_a_to_j"
        report["protocol"].pop("max_prompt_tokens_without_truncation", None)
        report["dtype"], report["device"] = "float32", "cuda:0"
        report["checkpoint_file"] = str(args.checkpoint_file)
        report["checkpoint_argument"] = str(getattr(args, "checkpoint_argument", args.checkpoint_file))
        report["runtime_config"] = str(args.runtime_config)
        report["architecture"] = args.architecture
        report["checkpoint_contract"] = args.checkpoint_audit
        report["loaded_model_audit"] = args.loaded_model_audit
        report["dataset_audio_audit"] = getattr(args, "dataset_audio_audit", {})
        if args.task == "mmau_test_mini":
            observed_sources = report.get("records", {}).get("audio", {}).get("payload_sources", {})
            report["dataset_audio_audit"]["observed_payload_sources"] = observed_sources
        report["evaluation_identity"] = identity
        report["provenance"] = {
            "runtime_config_sha256": C.sha256_file(args.runtime_config),
            "runtime_path": str(args.runtime_config), "route_root": str(args.route_root),
            "training_staging_checked": False,
        }
        # This is the same for all three models on one benchmark. Model code,
        # checkpoint and architecture are separately identified above.
        shared = {key: identity[key] for key in ("protocol", "dataset", "evaluation_sources", "software", "tokenizer_files")}
        shared["checkpoint_training_family"] = {
            "route_variant": args.checkpoint_audit["route_variant"],
            "epoch_completed": args.checkpoint_audit["epoch_completed"],
            "num_epochs": args.checkpoint_audit["num_epochs"],
            "batch_geometry": args.checkpoint_audit["batch_geometry"],
            "loss_reduction": args.checkpoint_audit["loss_reduction"],
        }
        model_audit = args.loaded_model_audit
        shared["loaded_model_shared_settings"] = {key: model_audit.get(key) for key in (
            "shared_lm_config", "attention_implementation", "tokenizer_vocab_sha256", "eos_token_id", "pad_token_id")}
        audio_settings = copy.deepcopy(args.model_contract["encoder"])
        audio_settings.pop("pretrained_audioencoder_path", None)
        shared["audio_encoder_settings"] = audio_settings
        report["comparison_protocol_sha256"] = C.fingerprint(shared)
        coverage = report.get("inference_coverage", {}).get("status") == "PASS"
        if coverage:
            try:
                report["choice_label_prefix_evaluation"] = score_both(args, records, report)
            except Exception as exc:
                report["status"] = "FAILED"
                report["fatal_error"] = {"error": repr(exc), "stage": "dual_scoring"}
        failures = int(report.get("records", {}).get("skip_reasons", {}).get("sample_exception", 0))
        prefix = report.get("choice_label_prefix_evaluation") or {}
        comparable = bool(coverage and failures == 0 and report.get("status") == "PASS"
                          and report.get("official_artifact_audit", {}).get("status") == "PASS"
                          and report.get("official_evaluation", {}).get("status") == "PASS"
                          and prefix.get("total", {}).get("total") == 1000)
        report["comparable"] = comparable
        report["comparable_official_score"] = comparable
        report["dual_scoring"] = {
            "official": report.get("official_evaluation", {}).get("status"),
            "choice_label_prefix": prefix.get("status"),
            "raw_generation_shared": True,
            "official_input_policy": "raw_generated_text_no_label_strip_no_trim",
            "prefix_input_policy": "raw_generated_text_no_label_strip_no_trim",
            "denominator": 1000,
        }
        if not comparable:
            report["status"] = "FAILED"
            report.setdefault("comparison_errors", []).append(
                f"Incomplete or failed comparable evaluation; generation exceptions={failures}. Inspect fatal_error/skipped/scorer logs.")
        mmau._write_json(args.output_dir / "evaluation_report.json", report)
        mmau._write_json(args.output_dir / "summary.json", {
            "status": report["status"], "comparable": comparable, "task": args.task,
            "architecture": args.architecture, "checkpoint_file": str(args.checkpoint_file),
            "checkpoint_argument": str(getattr(args, "checkpoint_argument", args.checkpoint_file)),
            "epoch_completed": args.checkpoint_audit["epoch_completed"],
            "num_epochs": args.checkpoint_audit["num_epochs"],
            "comparison_protocol_sha256": report["comparison_protocol_sha256"],
            "choice_label_prefix": prefix.get("total", {}),
            "official": report.get("official_evaluation", {}),
            "records": report.get("records", {}),
        })
        return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    print(json.dumps({"status": report["status"], "comparable": report["comparable"],
                      "architecture": args.architecture, "task": args.task,
                      "report": str(args.output_dir / "evaluation_report.json")}, ensure_ascii=False))
    return 0 if report["comparable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
