#!/usr/bin/env python3
"""CPU-only preflight for the official Mellow 8-GPU training route.

The audit intentionally does not open audio waveforms, load language-model
weights, initialize CUDA, or launch distributed processes. It validates the
already-PASS ReasonAQA path audit, local SmolLM2 tokenizer/config assets, the
HTSAT checkpoint schema expected by upstream Mellow, cluster resource limits,
and important training-code contracts before an 8-GPU staging/DDP smoke is
submitted.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import time
from collections import Counter
from pathlib import Path
from typing import Any


CONTRACT = "mellow_official_c8204d8_8gpu_preflight_v1"
PATH_AUDIT_CONTRACT = "reasonaqa_raw_audio_path_mapping_v1"
DEFAULT_TEXT_MODEL = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2")
DEFAULT_HTSAT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-audit-report", type=Path, required=True)
    parser.add_argument("--mapping-jsonl", type=Path)
    parser.add_argument("--text-model-dir", type=Path, default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--htsat-checkpoint", type=Path, default=DEFAULT_HTSAT)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--memory-gib", type=int, default=256)
    parser.add_argument("--workers-per-rank", type=int, default=4)
    parser.add_argument("--candidate-batch-size-per-rank", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--staging-margin-gib", type=float, default=10.0)
    parser.add_argument("--mapping-source-samples", type=int, default=48)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, status: str, detail: Any) -> None:
        if status not in {"PASS", "WARN", "FAIL"}:
            raise ValueError(f"invalid check status: {status}")
        self.checks.append({"name": name, "status": status, "detail": detail})

    def require(self, name: str, condition: bool, detail: Any) -> bool:
        self.add(name, "PASS" if condition else "FAIL", detail)
        return condition

    def warn(self, name: str, detail: Any) -> None:
        self.add(name, "WARN", detail)

    @property
    def hard_failures(self) -> list[dict[str, Any]]:
        return [check for check in self.checks if check["status"] == "FAIL"]

    @property
    def warnings(self) -> list[dict[str, Any]]:
        return [check for check in self.checks if check["status"] == "WARN"]


def existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def existing_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def inspect_mapping(
    path: Path,
    sample_count: int,
) -> dict[str, Any]:
    rng = random.Random(20260929)
    reservoir: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    count = 0
    logical_seen: set[str] = set()
    groups: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    staged_payload_bytes = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"mapping line {line_number} is not an object")
            logical = str(item.get("logical_path", ""))
            source = str(item.get("source_path", ""))
            size = int(item.get("source_size_bytes", -1))
            if not logical or not source or size <= 0:
                raise ValueError(f"invalid mapping line {line_number}: {item}")
            if logical in logical_seen:
                raise ValueError(f"duplicate logical path in mapping: {logical!r}")
            logical_seen.add(logical)
            groups[str(item.get("group", "unknown"))] += 1
            methods[str(item.get("method", "unknown"))] += 1
            staged_payload_bytes += size
            count += 1
            if first is None:
                first = item
            last = item
            if len(reservoir) < sample_count:
                reservoir.append(item)
            elif sample_count > 0:
                replacement = rng.randrange(count)
                if replacement < sample_count:
                    reservoir[replacement] = item

    selected: dict[str, dict[str, Any]] = {}
    for item in [first, last, *reservoir]:
        if item is not None:
            selected[str(item["logical_path"])] = item

    sample_results = []
    for logical, item in sorted(selected.items()):
        source = Path(str(item["source_path"]))
        expected = int(item["source_size_bytes"])
        exists = source.is_file()
        actual = source.stat().st_size if exists else None
        sample_results.append(
            {
                "logical_path": logical,
                "source_path": str(source),
                "expected_size_bytes": expected,
                "actual_size_bytes": actual,
                "status": "PASS" if exists and actual == expected else "FAIL",
            }
        )

    return {
        "line_count": count,
        "group_counts": dict(groups),
        "method_counts": dict(methods),
        "staged_payload_bytes": staged_payload_bytes,
        "staged_payload_gib": staged_payload_bytes / (1024**3),
        "sample_count": len(sample_results),
        "source_samples": sample_results,
    }


def inspect_text_model(model_dir: Path) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer

    raw_config_path = model_dir / "config.json"
    raw_config = json.loads(raw_config_path.read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    tokenizer_length_before = len(tokenizer)
    added_tokens = tokenizer.add_special_tokens({"pad_token": "!"})
    tokenizer_length_after = len(tokenizer)

    weight_files = sorted(
        [
            *model_dir.glob("*.safetensors"),
            *model_dir.glob("pytorch_model*.bin"),
            *model_dir.glob("*.safetensors.index.json"),
            *model_dir.glob("pytorch_model*.bin.index.json"),
        ],
        key=lambda path: path.name,
    )
    tokenizer_files = sorted(
        [
            path
            for path in model_dir.iterdir()
            if path.is_file()
            and (
                path.name.startswith("tokenizer")
                or path.name in {"vocab.json", "merges.txt", "special_tokens_map.json"}
            )
        ],
        key=lambda path: path.name,
    )
    return {
        "path": str(model_dir),
        "path_routes_to_smollm2_branch": "smollm2" in str(model_dir).casefold(),
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "raw_model_type": raw_config.get("model_type"),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
        "tokenizer_length_before": tokenizer_length_before,
        "tokenizer_length_after": tokenizer_length_after,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "pad_token_added_count": added_tokens,
        "weight_files": [path.name for path in weight_files],
        "tokenizer_files": [path.name for path in tokenizer_files],
    }


def inspect_htsat_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"HTSAT checkpoint payload is not a dictionary: {type(payload).__name__}")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("HTSAT checkpoint has no non-empty state_dict")

    keys = list(state)
    prefix_counts = Counter(key[:10] for key in keys)
    stripped = [key[10:] for key in keys]
    stripped_collisions = len(stripped) - len(set(stripped))
    tensor_entries = 0
    tensor_elements = 0
    for value in state.values():
        if isinstance(value, torch.Tensor):
            tensor_entries += 1
            tensor_elements += value.numel()
    return {
        "path": str(path),
        "basename": path.name,
        "size_bytes": path.stat().st_size,
        "state_entries": len(state),
        "tensor_entries": tensor_entries,
        "tensor_elements": tensor_elements,
        "all_keys_start_sed_model": all(key.startswith("sed_model.") for key in keys),
        "first_ten_character_prefix_counts": dict(prefix_counts.most_common(12)),
        "stripped_key_collisions": stripped_collisions,
        "key_preview": keys[:12],
    }


def inspect_upstream_source(route_root: Path) -> dict[str, Any]:
    paths = {
        "trainer": route_root / "training" / "trainer.py",
        "distributed": route_root / "distributed" / "torch.py",
        "dataset": route_root / "data" / "audiotext_dataset.py",
        "audio_io": route_root / "utils" / "audio_io.py",
        "model": route_root / "models" / "mellow.py",
        "decoder": route_root / "models" / "decoder.py",
        "entrypoint": route_root / "train.py",
    }
    texts = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}

    contracts = {
        "dataset_joins_datapath": "os.path.join(self.data_path, file_path1)" in texts["dataset"],
        "dataset_loads_raw_audio": "load_audio(file_path1" in texts["dataset"],
        "audio_io_uses_soundfile": "sf.read(" in texts["audio_io"],
        "htsat_checkpoint_fixed_basename": "HTSAT_AudioSet_Saved_1.ckpt" in texts["model"],
        "htsat_keys_strip_first_ten": "new_ckpt[key[10:]]" in texts["model"],
        "decoder_path_string_routes_smollm2": 'elif "smollm2" in self.text_decoder' in texts["decoder"],
        "ddp_converts_sync_batchnorm": "convert_sync_batchnorm(model)" in texts["distributed"],
        "ddp_find_unused_false": "find_unused_parameters=False" in texts["distributed"],
        "cosine_scheduler_is_epoch_based": (
            "CosineAnnealingLR(optimizer, self.config[\"train\"][\"num_epochs\"])"
            in texts["trainer"]
        ),
        "checkpoint_saves_raw_state_dict": (
            "torch.save(self.distributed.get_distributed_model_state(model), f)"
            in texts["trainer"]
        ),
        "resume_reads_nested_state_dict": "checkpoint = checkpoint['state_dict']" in texts["trainer"],
        "resume_loads_model_state_only": "model.load_state_dict(checkpoint, strict=False)" in texts["trainer"],
        "job_id_supports_shared_env": (
            "MELLOW_JOB_ID" in texts["entrypoint"]
            and "datetime.now().strftime" in texts["entrypoint"]
        ),
        "gradient_accumulation_supported": (
            "gradient_accumulation_steps" in texts["trainer"]
            and "loss / gradient_accumulation_steps" in texts["trainer"]
            and "no_sync()" in texts["trainer"]
        ),
    }
    return {
        "files": {name: str(path) for name, path in paths.items()},
        "contracts": contracts,
    }


def main() -> int:
    args = parse_args()
    audit = Audit()
    started = time.time()
    route_root = Path(__file__).resolve().parents[2]

    report: dict[str, Any] = {
        "status": "FAIL",
        "contract": CONTRACT,
        "hostname": socket.gethostname(),
        "started_unix": started,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }

    try:
        path_audit_path = existing_file(args.path_audit_report, "path audit report")
        path_audit = json.loads(path_audit_path.read_text(encoding="utf-8"))
        audit.require(
            "path_audit_status",
            path_audit.get("status") == "PASS",
            {"path": str(path_audit_path), "status": path_audit.get("status")},
        )
        audit.require(
            "path_audit_contract",
            path_audit.get("contract") == PATH_AUDIT_CONTRACT,
            {
                "actual": path_audit.get("contract"),
                "expected": PATH_AUDIT_CONTRACT,
            },
        )

        recorded_mapping = existing_file(Path(str(path_audit["mapping_path"])), "recorded mapping")
        mapping_path = (
            existing_file(args.mapping_jsonl, "mapping argument")
            if args.mapping_jsonl is not None
            else recorded_mapping
        )
        audit.require(
            "mapping_report_pairing",
            mapping_path == recorded_mapping,
            {"recorded": str(recorded_mapping), "argument": str(mapping_path)},
        )
        mapping = inspect_mapping(mapping_path, args.mapping_source_samples)
        audit.require(
            "mapping_logical_count",
            mapping["line_count"] == int(path_audit["unique_logical_paths_resolved"]),
            {
                "mapping": mapping["line_count"],
                "audit": path_audit["unique_logical_paths_resolved"],
            },
        )
        failed_samples = [
            item for item in mapping["source_samples"] if item["status"] != "PASS"
        ]
        audit.require(
            "mapping_source_size_samples",
            not failed_samples,
            {
                "checked": mapping["sample_count"],
                "failures": failed_samples,
            },
        )
        train_json = existing_file(Path(str(path_audit["train_json"])), "ReasonAQA train JSON")
        audit.add(
            "reasonaqa_metadata_present",
            "PASS",
            {"path": str(train_json), "size_bytes": train_json.stat().st_size},
        )

        text_model_dir = existing_directory(args.text_model_dir, "text model directory")
        text_model = inspect_text_model(text_model_dir)
        audit.require(
            "text_model_routes_official_smollm2_branch",
            text_model["path_routes_to_smollm2_branch"],
            {
                "path": text_model["path"],
                "reason": "official decoder dispatches by substring in the path string",
            },
        )
        audit.require(
            "text_model_config",
            text_model["model_type"] == "llama"
            and text_model["vocab_size"] == 49152
            and text_model["hidden_size"] == 576,
            {
                "model_type": text_model["model_type"],
                "vocab_size": text_model["vocab_size"],
                "hidden_size": text_model["hidden_size"],
            },
        )
        audit.require(
            "text_model_weights_present",
            bool(text_model["weight_files"]),
            text_model["weight_files"],
        )
        audit.require(
            "tokenizer_local_load",
            text_model["tokenizer_class"] == "GPT2TokenizerFast"
            and text_model["tokenizer_vocab_size"] == 49152
            and text_model["pad_token"] == "!",
            text_model,
        )

        htsat_path = existing_file(args.htsat_checkpoint, "HTSAT checkpoint")
        htsat = inspect_htsat_checkpoint(htsat_path)
        audit.require(
            "htsat_official_basename",
            htsat["basename"] == "HTSAT_AudioSet_Saved_1.ckpt",
            htsat["basename"],
        )
        audit.require(
            "htsat_official_key_stripping_compatible",
            htsat["all_keys_start_sed_model"] and htsat["stripped_key_collisions"] == 0,
            {
                "all_keys_start_sed_model": htsat["all_keys_start_sed_model"],
                "stripped_key_collisions": htsat["stripped_key_collisions"],
                "prefix_counts": htsat["first_ten_character_prefix_counts"],
            },
        )

        resource_details = {
            "gpus": args.gpus,
            "cpus": args.cpus,
            "memory_gib": args.memory_gib,
            "cpu_limit": args.gpus * 8,
            "memory_limit_gib": args.gpus * 32,
        }
        audit.require("resource_world_size", args.gpus == 8, resource_details)
        audit.require(
            "resource_queue_limits",
            args.cpus <= args.gpus * 8 and args.memory_gib <= args.gpus * 32,
            resource_details,
        )
        audit.require(
            "worker_cpu_budget",
            args.workers_per_rank * args.gpus <= args.cpus,
            {
                "workers_per_rank": args.workers_per_rank,
                "total_workers": args.workers_per_rank * args.gpus,
                "requested_cpus": args.cpus,
            },
        )
        estimated_required = (
            mapping["staged_payload_bytes"]
            + train_json.stat().st_size
            + mapping_path.stat().st_size
            + path_audit_path.stat().st_size
            + int(args.staging_margin_gib * 1024**3)
        )
        audit.require(
            "requested_memory_covers_staging_estimate",
            estimated_required < args.memory_gib * 1024**3,
            {
                "estimated_required_gib": estimated_required / (1024**3),
                "requested_memory_gib": args.memory_gib,
                "note": "the actual /dev/shm free space must still be checked inside the job",
            },
        )

        upstream = inspect_upstream_source(route_root)
        core_contract_names = {
            "dataset_joins_datapath",
            "dataset_loads_raw_audio",
            "audio_io_uses_soundfile",
            "htsat_checkpoint_fixed_basename",
            "htsat_keys_strip_first_ten",
            "decoder_path_string_routes_smollm2",
            "ddp_converts_sync_batchnorm",
            "ddp_find_unused_false",
            "cosine_scheduler_is_epoch_based",
            "checkpoint_saves_raw_state_dict",
            "resume_loads_model_state_only",
            "job_id_supports_shared_env",
            "gradient_accumulation_supported",
        }
        missing_contracts = [
            name
            for name in core_contract_names
            if not upstream["contracts"].get(name, False)
        ]
        audit.require(
            "upstream_source_contracts",
            not missing_contracts,
            {"missing": missing_contracts, "contracts": upstream["contracts"]},
        )
        resume_schema_compatible = not (
            upstream["contracts"].get("checkpoint_saves_raw_state_dict")
            and upstream["contracts"].get("resume_reads_nested_state_dict")
        )
        audit.add(
            "checkpoint_resume_schema_compatibility",
            "PASS" if resume_schema_compatible else "WARN",
            {
                "checkpoint_saves_raw_state_dict": upstream["contracts"].get(
                    "checkpoint_saves_raw_state_dict"
                ),
                "resume_reads_nested_state_dict": upstream["contracts"].get(
                    "resume_reads_nested_state_dict"
                ),
                "detail": (
                    "fresh training is unaffected; do not use resume_checkpoint until "
                    "the schema mismatch is explicitly adapted"
                    if not resume_schema_compatible
                    else "save and resume schemas agree"
                ),
            },
        )

        audit.warn(
            "candidate_batch_size_requires_8gpu_smoke",
            {
                "per_rank": args.candidate_batch_size_per_rank,
                "gradient_accumulation": args.gradient_accumulation_steps,
                "global_effective": (
                    args.candidate_batch_size_per_rank
                    * args.gpus
                    * args.gradient_accumulation_steps
                ),
                "reason": "3090 memory fit is intentionally deferred to the real 8-GPU smoke",
            },
        )
        audit.warn(
            "checkpoint_resume_limitation",
            "upstream checkpoints contain only a raw model state_dict; optimizer, scheduler, epoch, sampler, and RNG state are not saved, and the training resume path currently expects a nested 'state_dict' key",
        )
        audit.warn(
            "epoch_boundary_broadcast_requires_runtime_validation",
            "upstream broadcasts model and optimizer tensor states after every epoch; contiguity and rank agreement need an 8-GPU runtime audit",
        )
        audit.warn(
            "sync_batchnorm_config_is_not_a_switch",
            "distributed/torch.py converts to SyncBatchNorm unconditionally whenever DDP is used",
        )
        audit.warn(
            "unused_or_non_stepwise_config_fields",
            {
                "warm_up_steps": "not used by the trainer",
                "reduce_lr_steps": "not used by the trainer",
                "scheduler": "cosine advances once per epoch, not per optimizer step",
                "gradient_accumulation": (
                    "implemented in trainer.py; optimizer/scheduler semantics remain "
                    "official except updates occur after each accumulation window"
                ),
            },
        )

        report.update(
            {
                "path_audit": path_audit,
                "mapping": mapping,
                "text_model": text_model,
                "htsat_checkpoint": htsat,
                "upstream_source": upstream,
                "resource_contract": resource_details,
                "estimated_staging_required_bytes": estimated_required,
            }
        )
    except Exception as exc:
        audit.add(
            "preflight_exception",
            "FAIL",
            {"type": type(exc).__name__, "message": str(exc)},
        )

    report["checks"] = audit.checks
    report["hard_failures"] = audit.hard_failures
    report["warnings"] = audit.warnings
    report["status"] = "PASS" if not audit.hard_failures else "FAIL"
    report["completed_unix"] = time.time()
    report["elapsed_seconds"] = report["completed_unix"] - started
    write_json(args.report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
