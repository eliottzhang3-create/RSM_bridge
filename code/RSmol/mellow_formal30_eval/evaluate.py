#!/usr/bin/env python3
"""Full MMAU/MMAR evaluation of supported Mellow audio checkpoints."""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT.parent / "scripts", ROOT.parent):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import contracts as C
import audio as eval_audio
import runtime
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
    args.requested_alias_architecture = {
        "smollm2_epo15": "smollm2",
        "smollm2_epo25": "smollm2",
    }.get(args.requested_alias, args.requested_alias)
    args.checkpoint_file = Path(aliases.get(args.checkpoint_file, args.checkpoint_file)).expanduser().resolve(strict=True)
    run_root = args.checkpoint_file.parent.parent.parent
    if args.runtime_config is None:
        candidates = [run_root / name for name in ("runtime_30epochs.yaml", "runtime_10epochs.yaml")
                      if (run_root / name).is_file()]
        if len(candidates) != 1:
            parser.error("could not uniquely infer runtime config; pass --runtime-config from the checkpoint run root")
        args.runtime_config = candidates[0]
    try:
        args.runtime_config = args.runtime_config.resolve(strict=True)
    except FileNotFoundError as exc:
        parser.error(f"runtime config does not exist: {exc.filename}")
    if args.runtime_config.parent != run_root:
        parser.error("runtime config must belong to the checkpoint run root")
    args.output_dir = (args.output_dir or run_root / "eval_mellow_unified_v5" / args.task /
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

    print(f"[formal30-eval] checkpoint argument: {args.checkpoint_argument}", flush=True)
    print(f"[formal30-eval] validating checkpoint: {args.checkpoint_file}", flush=True)
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


def audit_dataset(args: Any) -> tuple[list[dict], dict, dict]:
    """Validate benchmark artifacts before opening a resumable generation store."""
    if args.task == "mmau_test_mini":
        records = mmau.load_official_records(args.metadata_json)
        audit = mmau.audit_mmau_v051525(args.metadata_json, args.evaluation_script, records)
    else:
        records = mmar.load_mmar_records(args.metadata_json)
        audit = mmar.audit_mmar_artifacts(args.metadata_json, args.audio_root, args.evaluation_script, records)
    if audit["status"] != "PASS":
        raise ValueError(f"official artifact audit failed: {audit['failures']}")
    audio_hashes = []
    official_audio_rows = 0
    fallback_audio_rows = 0
    for record in records:
        choices = mmau._coerce_choices(record.get("choices"))
        C.build_prompt(mmau.extract_record_question(record), choices)
        if args.task == "mmau_test_mini":
            path = args.audio_root / f"{mmau._official_id(record)}.wav"
        else:
            path = mmar.resolve_mmar_audio_path(args.audio_root, record["audio_path"])
        if path.is_file():
            official_audio_rows += 1
            audio_hashes.append([str(record["id"]), "official_wav", C.sha256_file(path)])
        elif args.task == "mmau_test_mini":
            # The historical MMAU evaluator deliberately preferred the
            # official ``<id>.wav`` file, but fell back to the embedded
            # parquet ``context.audio`` payload when that file was absent.
            # Keep that compatibility rule: many remote dataset mirrors do
            # not materialize all 1,000 convenience WAVs, while the parquet
            # benchmark remains complete.  The row-level decoder will still
            # record a skip if the embedded payload is absent or undecodable.
            fallback_audio_rows += 1
            audio_hashes.append([str(record["id"]), "parquet_embedded_audio_fallback"])
        else:
            raise FileNotFoundError(f"official benchmark audio is missing: {path}")
    identity = {
        "metadata_sha256": C.sha256_file(args.metadata_json),
        "evaluation_script_sha256": C.sha256_file(args.evaluation_script),
        "audio_files_sha256": C.fingerprint(audio_hashes),
        "audio_files": official_audio_rows if args.task == "mmau_test_mini" else len(audio_hashes),
        "audio_source_policy": ("prefer_official_id_wav_else_parquet_embedded"
                                 if args.task == "mmau_test_mini" else "official_audio_path_required"),
    }
    if args.task == "mmau_test_mini":
        identity["parquet_sha256"] = C.sha256_file(args.parquet)
        identity["official_audio_files"] = official_audio_rows
        identity["parquet_fallback_audio_rows"] = fallback_audio_rows
        identity["official_audio_missing_rows"] = fallback_audio_rows
    args.dataset_audio_audit = {
        "policy": identity["audio_source_policy"],
        "official_audio_files": official_audio_rows,
        "parquet_fallback_audio_rows": fallback_audio_rows,
        "metadata_rows": len(records),
    }
    return records, audit, identity


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
        "protocol": C.protocol(args.task), "checkpoint_sha256": C.sha256_file(args.checkpoint_file),
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
    marker = output_dir.parent / f".{output_dir.name}.mellow_eval_identity.json"
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


@contextlib.contextmanager
def output_lock(output_dir: Path):
    import fcntl

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.mellow_eval.lock"
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
    print("[formal30-eval] auditing metadata/scorer and benchmark audio (MMAU parquet fallback enabled)", flush=True)
    records, artifact_audit, dataset_identity = audit_dataset(args)
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
            record_static_fields={"architecture": args.architecture, "audio2_reused": True,
                                  "audio2_equals_audio1": True,
                                  "audio2_waveform_materialized": False,
                                  "audio_encoder_invocations": 2, "single_audio_slot": False,
                                  "audio2_policy": "reuse_audio1_waveform_encoded_independently"},
            stage="mellow_unified_" + args.task,
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
            report = mmar.run(args, **common_kwargs,
                              protocol_description=C.PROTOCOL_VERSION + ": " + json.dumps(C.protocol(args.task), sort_keys=True))
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
                          and artifact_audit["status"] == "PASS"
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
