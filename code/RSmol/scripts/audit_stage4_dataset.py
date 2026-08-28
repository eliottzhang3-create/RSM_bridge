#!/usr/bin/env python3
"""Run the Stage 4 Gate B dataset audit as a CPU-only local process.

This entry point deliberately owns no multi-process or training setup.  The
Parquet discovery, footer/content audit, shard assignment, and local
tokenizer validation are shared with :mod:`train_stage4_ddp` so that Gate B
and the subsequent multi-card pilots consume the same audit semantics.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_shared_stage4_api() -> Any:
    """Load the dependency-light helpers without importing the package init.

    ``code.RSmol.__init__`` exports the model and therefore imports torch.  A
    file-path import of the Stage 4 entry point executes only its standard
    library top level; its audit/tokenizer helpers import optional packages
    lazily when called.
    """

    source = SCRIPT_ROOT / "scripts" / "train_stage4_ddp.py"
    spec = importlib.util.spec_from_file_location("stage4_dataset_shared", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load shared Stage 4 helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SHARED = _load_shared_stage4_api()
DEFAULT_CONTEXT_LENGTH = _SHARED.DEFAULT_CONTEXT_LENGTH
DEFAULT_FORMAL_GLOBAL_SAMPLES = _SHARED.DEFAULT_FORMAL_GLOBAL_SAMPLES
DEFAULT_MAX_OPTIMIZER_STEPS = _SHARED.DEFAULT_MAX_OPTIMIZER_STEPS
DEFAULT_TARGET_SAMPLES_PER_RANK = _SHARED.DEFAULT_TARGET_SAMPLES_PER_RANK
DEFAULT_WORLD_SIZE = _SHARED.DEFAULT_WORLD_SIZE
EXPECTED_PARQUET_COUNT = _SHARED.EXPECTED_PARQUET_COUNT
DATASET_NUM_PROC = _SHARED.DATASET_NUM_PROC
NUM_WORKERS = _SHARED.NUM_WORKERS
PERSISTENT_WORKERS = _SHARED.PERSISTENT_WORKERS
PIN_MEMORY = _SHARED.PIN_MEMORY
_load_tokenizer_only = _SHARED._load_tokenizer_only
_write_json = _SHARED._write_json
assign_shards = _SHARED.assign_shards
audit_parquet_shards = _SHARED.audit_parquet_shards
discover_parquet_files = _SHARED.discover_parquet_files
ensure_external_output = _SHARED.ensure_external_output


class AuditConfig:
    """CLI values for a deterministic, eight-rank-compatible pre-audit."""

    data_dir = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset")
    output_dir = Path(
        "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4/gate-b-cpu"
    )

    def __init__(
        self,
        model_path: Path | None = None,
        tokenizer_path: Path | None = None,
        data_dir: Path = data_dir,
        output_dir: Path = output_dir,
        report_path: Path | None = None,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        world_size: int = DEFAULT_WORLD_SIZE,
        seed: int = 0,
    ) -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.report_path = report_path
        self.context_length = context_length
        self.world_size = world_size
        self.seed = seed


def _parse_args(argv: Sequence[str] | None = None) -> AuditConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=AuditConfig.data_dir,
        help="Directory containing the exact 85 train-xxxxx-of-00085.parquet shards",
    )
    parser.add_argument("--output-dir", type=Path, default=AuditConfig.output_dir)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    config = AuditConfig(**vars(args))
    if config.model_path is None and config.tokenizer_path is None:
        raise ValueError(
            "Gate B CPU audit requires --tokenizer-path or --model-path; "
            "when --tokenizer-path is omitted, the tokenizer is loaded from --model-path."
        )
    if not (0 < config.context_length <= DEFAULT_CONTEXT_LENGTH):
        raise ValueError(
            f"Gate B context_length must be in [1, {DEFAULT_CONTEXT_LENGTH}], "
            f"got {config.context_length}"
        )
    if config.world_size <= 0:
        raise ValueError(f"world_size must be positive, got {config.world_size}")
    if config.report_path is None:
        config.report_path = config.output_dir / "stage4_gate_B_audit.json"
    return config


def _canonical_output_paths(config: AuditConfig) -> tuple[Path, Path]:
    output_dir = ensure_external_output(config.output_dir)
    report_path = ensure_external_output(config.report_path or output_dir / "stage4_gate_B_audit.json")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Gate B CPU audit refuses to overwrite a non-empty output directory: {output_dir}"
        )
    if report_path.exists():
        raise FileExistsError(
            f"Gate B CPU audit refuses to overwrite an existing report: {report_path}"
        )
    return output_dir, report_path


def _rank_statistics(audit: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    by_name = {item["name"]: item for item in audit["shards"]}
    rank_raw_rows: dict[str, int] = {}
    rank_valid_rows: dict[str, int] = {}
    for rank, shard_paths in manifest["rank_shards"].items():
        rank_raw_rows[rank] = sum(int(by_name[Path(path).name]["num_rows"]) for path in shard_paths)
        rank_valid_rows[rank] = sum(
            int(by_name[Path(path).name]["content"]["valid_trainable_rows"])
            for path in shard_paths
        )
    manifest["rank_raw_rows"] = rank_raw_rows
    manifest["rank_valid_trainable_rows"] = rank_valid_rows
    manifest["target_samples_per_rank"] = DEFAULT_TARGET_SAMPLES_PER_RANK
    manifest["rank_has_target_samples"] = {
        rank: rows >= DEFAULT_TARGET_SAMPLES_PER_RANK
        for rank, rows in rank_valid_rows.items()
    }
    manifest["raw_rows"] = int(audit["total_rows"])
    manifest["effective_trainable_rows"] = int(audit["content"]["valid_trainable_rows"])
    # Keep formal consumption tied to the requested topology.  The default
    # world_size=8 remains exactly 9,465,856 samples.
    manifest["formal_global_samples"] = int(
        DEFAULT_FORMAL_GLOBAL_SAMPLES * manifest["world_size"] // DEFAULT_WORLD_SIZE
    )
    manifest["formal_remaining_raw_rows"] = int(
        audit["total_rows"] - manifest["formal_global_samples"]
    )
    return manifest


def run(config: AuditConfig) -> dict[str, Any]:
    """Audit all shards and write one complete external JSON report."""

    output_dir, report_path = _canonical_output_paths(config)
    tokenizer_source = config.tokenizer_path or config.model_path
    if tokenizer_source is None:  # guarded by _parse_args; useful for callers
        raise ValueError("Gate B CPU audit requires a local tokenizer path")

    # This helper validates a local tokenizer directory and loads only the
    # tokenizer.  No model/checkpoint artifact is opened by this process.
    tokenizer = _load_tokenizer_only(tokenizer_source)
    files = discover_parquet_files(config.data_dir)
    audit = audit_parquet_shards(
        files,
        tokenizer=tokenizer,
        context_length=config.context_length,
        content=True,
    )
    if len(files) != EXPECTED_PARQUET_COUNT:
        raise ValueError(f"Gate B requires {EXPECTED_PARQUET_COUNT} shards, got {len(files)}")

    manifest = assign_shards(files, world_size=config.world_size, seed=config.seed)
    _rank_statistics(audit, manifest)
    rank_failures = [
        rank
        for rank, ok in manifest["rank_has_target_samples"].items()
        if not ok
    ]
    report = {
        "status": "PASS" if not rank_failures else "FAIL",
        "gate": "B",
        "configuration": {
            "model_path": str(config.model_path.expanduser().resolve()) if config.model_path else None,
            "tokenizer_path": str(tokenizer_source.expanduser().resolve()),
            "data_dir": str(config.data_dir.expanduser().resolve()),
            "output_dir": str(output_dir),
            "report_path": str(report_path),
            "context_length": config.context_length,
            "world_size": config.world_size,
            "seed": config.seed,
        },
        "dataset_audit": audit,
        "manifest": manifest,
        "rank_shards": manifest["rank_shards"],
        "rank_shard_counts": manifest["rank_shard_counts"],
        "rank_raw_rows": manifest["rank_raw_rows"],
        "rank_valid_trainable_rows": manifest["rank_valid_trainable_rows"],
        "target_samples_per_rank": DEFAULT_TARGET_SAMPLES_PER_RANK,
        "rank_has_target_samples": manifest["rank_has_target_samples"],
        "raw_rows": audit["total_rows"],
        "effective_rows": audit["content"]["valid_trainable_rows"],
        "formal_global_samples": manifest["formal_global_samples"],
        "formal_remaining_raw_rows": manifest["formal_remaining_raw_rows"],
        "target_sample_contract": {
            "optimizer_steps": DEFAULT_MAX_OPTIMIZER_STEPS,
            "samples_per_rank": DEFAULT_TARGET_SAMPLES_PER_RANK,
            "all_ranks_reach_target": not rank_failures,
            "failing_ranks": rank_failures,
        },
        "cpu_only": True,
        "model_loaded": False,
        "distributed_processes_started": False,
        "cache_files_audited": True,
        "cache_files_paths": [],
        "cache_policy": "datasets library not used; no HF Arrow cache created",
        "datasets_library_used": False,
        "data_loader_contract": {
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
            "persistent_workers": PERSISTENT_WORKERS,
            "dataset_num_proc": DATASET_NUM_PROC,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    if rank_failures:
        raise RuntimeError(
            "Gate B fail-fast: rank(s) lack the required "
            f"{DEFAULT_TARGET_SAMPLES_PER_RANK:,} effective samples: "
            f"{rank_failures}; valid={manifest['rank_valid_trainable_rows']}"
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        report = run(config)
        print(f"[result] status={report['status']}", flush=True)
        return 0 if report["status"] == "PASS" else 1
    except Exception as exc:
        print(f"[result] status=FAIL error={exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
