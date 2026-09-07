#!/usr/bin/env python3
"""CPU-only environment and remote-artifact audit for the Mellow audio route.

The command only inspects paths, JSON metadata, package versions, and the
HTSAT checkpoint with ``map_location='cpu'``.  It never opens an audio
waveform, submits a GPU job, or writes into a remote source tree.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MELLOW_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main"
DEFAULT_HTSAT_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/code/HTS-Audio-Transformer-main"
DEFAULT_REASONAQA_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa"
DEFAULT_AUDIOCAPS_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2"
DEFAULT_CLOTHO_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2"
DEFAULT_HTSAT_CHECKPOINT = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt"
DEFAULT_BASE_CHECKPOINT = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10_5/formal-epoch2-continue-20260902_184936/checkpoint-step-009244"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mellow-root", "--mellow_root", type=Path, default=Path(DEFAULT_MELLOW_ROOT))
    parser.add_argument("--htsat-root", "--htsat_root", type=Path, default=Path(DEFAULT_HTSAT_ROOT))
    parser.add_argument("--reasonaqa-root", "--reasonaqa_root", type=Path, default=Path(DEFAULT_REASONAQA_ROOT))
    parser.add_argument("--audiocaps-root", "--audiocaps_root", type=Path, default=Path(DEFAULT_AUDIOCAPS_ROOT))
    parser.add_argument("--clotho-root", "--clotho_root", type=Path, default=Path(DEFAULT_CLOTHO_ROOT))
    parser.add_argument("--htsat-checkpoint", "--htsat_checkpoint", type=Path, default=Path(DEFAULT_HTSAT_CHECKPOINT))
    parser.add_argument("--base-checkpoint", "--base_checkpoint", type=Path, default=Path(DEFAULT_BASE_CHECKPOINT))
    parser.add_argument("--report-path", "--report_path", type=Path, required=True)
    parser.add_argument("--skip-checkpoint-load", action="store_true", help="Only inspect checkpoint readability, do not torch.load it")
    return parser.parse_args(argv)


def _check(report: dict[str, Any], name: str, passed: bool, detail: Any = None, *, warning: bool = False) -> None:
    item: dict[str, Any] = {"name": name, "passed": bool(passed)}
    if detail is not None:
        item["detail"] = detail
    bucket = "checks" if passed else ("warnings" if warning else "hard_failures")
    report[bucket].append(item)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "<not-installed>"
    except Exception as exc:  # noqa: BLE001
        return f"<error:{type(exc).__name__}:{exc}>"


def _module_probe(name: str) -> dict[str, Any]:
    try:
        if importlib.util.find_spec(name) is None:
            return {"installed": False, "version": _package_version(name), "error": "module spec not found"}
    except Exception as exc:  # noqa: BLE001
        return {"installed": False, "version": _package_version(name), "error": f"find_spec:{type(exc).__name__}: {exc}"}
    try:
        module = importlib.import_module(name)
        return {"installed": True, "version": str(getattr(module, "__version__", _package_version(name))), "file": str(getattr(module, "__file__", "<builtin>"))}
    except Exception as exc:  # noqa: BLE001
        return {"installed": False, "version": _package_version(name), "error": f"{type(exc).__name__}: {exc}"}


def _path_summary(path: Path, *, kind: str) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "kind": kind, "exists": path.exists(), "is_file": path.is_file(), "is_dir": path.is_dir()}
    if path.exists():
        try:
            result["readable"] = os.access(path, os.R_OK)
            if path.is_file():
                result["size_bytes"] = path.stat().st_size
        except OSError as exc:
            result["readable"] = False
            result["error"] = f"{type(exc).__name__}: {exc}"
    else:
        result["readable"] = False
    return result


def _key_files(root: Path, names: Iterable[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name in names:
        direct = root / name
        candidates = [direct] if direct.exists() else sorted(root.rglob(name)) if root.is_dir() else []
        results.append({"name": name, "matches": [str(item) for item in candidates[:20]], "match_count": len(candidates)})
    return results


def _json_probe(path: Path) -> dict[str, Any]:
    result = _path_summary(path, kind="json")
    if not path.is_file():
        return result
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        result.update({"json_parseable": True, "top_level_type": type(payload).__name__, "record_count": len(payload) if isinstance(payload, (list, dict)) else None})
    except Exception as exc:  # noqa: BLE001
        result.update({"json_parseable": False, "error": f"{type(exc).__name__}: {exc}"})
    return result


def _checkpoint_probe(path: Path, *, load: bool) -> dict[str, Any]:
    result = _path_summary(path, kind="checkpoint")
    if not path.is_file() or not result.get("readable"):
        return result
    try:
        with path.open("rb") as handle:
            result["header_bytes"] = handle.read(16).hex()
        if load:
            import torch
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            result["torch_load_ok"] = True
            result["payload_type"] = type(payload).__name__
            if isinstance(payload, dict):
                result["top_level_keys"] = sorted(str(key) for key in payload.keys())[:100]
        else:
            result["torch_load_skipped"] = True
    except Exception as exc:  # noqa: BLE001
        result.update({"torch_load_ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "stage": "stage0_audio_5_10_5_mellow",
        "status": "FAIL",
        "cuda_required": False,
        "configuration": {"python": sys.executable, "platform": platform.platform(), "formal_world_size": 8, "formal_micro_batch_per_gpu": 4, "formal_global_micro_batch": 32},
        "checks": [], "warnings": [], "hard_failures": [], "traceback": None,
    }
    try:
        package_names = ("torch", "torchaudio", "torchlibrosa", "librosa", "transformers", "soundfile")
        report["dependencies"] = {name: {"distribution_version": _package_version(name), "probe": _module_probe(name)} for name in package_names}
        torch_probe = report["dependencies"]["torch"]["probe"]
        if torch_probe.get("installed"):
            try:
                import torch
                report["dependencies"]["torch"]["cuda_available"] = bool(torch.cuda.is_available())
            except Exception as exc:  # noqa: BLE001
                report["dependencies"]["torch"]["cuda_probe_error"] = f"{type(exc).__name__}: {exc}"
        report["paths"] = {
            "mellow_root": _path_summary(args.mellow_root, kind="source_dir"),
            "htsat_root": _path_summary(args.htsat_root, kind="source_dir"),
            "reasonaqa_root": _path_summary(args.reasonaqa_root, kind="dataset_dir"),
            "audiocaps_root": _path_summary(args.audiocaps_root, kind="dataset_dir"),
            "clotho_root": _path_summary(args.clotho_root, kind="dataset_dir"),
            "htsat_checkpoint": _checkpoint_probe(args.htsat_checkpoint, load=not args.skip_checkpoint_load),
            "base_checkpoint": _path_summary(args.base_checkpoint, kind="model_checkpoint_dir"),
        }
        for name in ("mellow_root", "htsat_root"):
            _check(report, f"{name}_visible", report["paths"][name]["is_dir"] and report["paths"][name]["readable"], report["paths"][name])
        checkpoint_ok = report["paths"]["htsat_checkpoint"].get("readable", False)
        if not args.skip_checkpoint_load:
            checkpoint_ok = checkpoint_ok and report["paths"]["htsat_checkpoint"].get("torch_load_ok", False)
        _check(report, "htsat_checkpoint_readable", checkpoint_ok, report["paths"]["htsat_checkpoint"])
        _check(report, "base_checkpoint_visible", report["paths"]["base_checkpoint"]["is_dir"] and report["paths"]["base_checkpoint"].get("readable", False), report["paths"]["base_checkpoint"])
        if report["paths"]["base_checkpoint"]["is_dir"]:
            report["base_checkpoint_files"] = _key_files(args.base_checkpoint, ("config.json", "training_state.pt", "pytorch_model.bin", "model.safetensors"))
            base_file_map = {item["name"]: item for item in report["base_checkpoint_files"]}
            _check(report, "base_checkpoint_config", base_file_map["config.json"]["match_count"] > 0, base_file_map["config.json"])
            _check(report, "base_checkpoint_weights", any(base_file_map[name]["match_count"] > 0 for name in ("pytorch_model.bin", "model.safetensors")), {name: base_file_map[name] for name in ("pytorch_model.bin", "model.safetensors")})
        report["source_key_files"] = {
            "mellow": _key_files(args.mellow_root, ("example.py", "requirements.txt", "mellow", "mellow/model/htsat.py", "README.md")),
            "htsat": _key_files(args.htsat_root, ("htsat.py", "HTSAT.py", "model/htsat.py", "config.py", "dataloader.py", "models")),
        }
        for split in ("train", "val", "test"):
            path = args.reasonaqa_root / f"{split}.json"
            report.setdefault("reasonaqa_splits", {})[split] = _json_probe(path)
            _check(report, f"reasonaqa_{split}_json", bool(report["reasonaqa_splits"][split].get("json_parseable")), report["reasonaqa_splits"][split])
        report["remote_audio_scan"] = {"audiocaps_root_visible": args.audiocaps_root.is_dir(), "clotho_root_visible": args.clotho_root.is_dir(), "waveform_loaded": False}
        if report["hard_failures"]:
            report["status"] = "FAIL"
        elif report["warnings"]:
            report["status"] = "PASS_WITH_WARNINGS"
        else:
            report["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        report["traceback"] = traceback.format_exc()
        report["hard_failures"].append({"name": "audit_exception", "passed": False, "detail": f"{type(exc).__name__}: {exc}"})
    report["summary"] = {"checks": len(report["checks"]), "warnings": len(report["warnings"]), "hard_failures": len(report["hard_failures"])}
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    write_report(args.report_path, report)
    print(json.dumps({"stage": report["stage"], "status": report["status"], "summary": report["summary"], "report": str(args.report_path)}, ensure_ascii=False))
    return 0 if report["status"] in {"PASS", "PASS_WITH_WARNINGS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
