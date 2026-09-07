#!/usr/bin/env python3
"""CUDA-only single-audio HTSAT forward audit for the Mellow route.

Remote Mellow and official HTSAT trees are imported dynamically from CLI
paths.  The script selects one implementation, loads the supplied checkpoint
strictly, preprocesses one 32 kHz/10 s mono waveform, and records every
observable output shape.  A missing CUDA device or missing audio is reported
as an incomplete/failed audit; it is never represented as a fabricated pass.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import sys
import traceback
import types
import wave
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MELLOW_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main"
DEFAULT_HTSAT_ROOT = "/hpc_stor03/sjtu_home/jinwei.zhang/code/HTS-Audio-Transformer-main"
DEFAULT_HTSAT_CHECKPOINT = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mellow-root", "--mellow_root", type=Path, default=Path(DEFAULT_MELLOW_ROOT))
    parser.add_argument("--htsat-root", "--htsat_root", type=Path, default=Path(DEFAULT_HTSAT_ROOT))
    parser.add_argument("--htsat-checkpoint", "--htsat_checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--audio-path", "--audio_path", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--report-path", "--report_path", type=Path, required=True)
    parser.add_argument("--implementation", choices=("auto", "mellow", "official"), default="auto")
    parser.add_argument("--device", default="cuda", help="Must be CUDA; explicit GPU submission is required")
    parser.add_argument("--construction-only", action="store_true", help="Import/build/checkpoint audit without audio forward; status is INCOMPLETE")
    return parser.parse_args(argv)


def _record(report: dict[str, Any], name: str, passed: bool, detail: Any = None, *, warning: bool = False) -> None:
    item: dict[str, Any] = {"name": name, "passed": bool(passed)}
    if detail is not None:
        item["detail"] = detail
    report["checks" if passed else ("warnings" if warning else "hard_failures")].append(item)


def _summary(report: dict[str, Any]) -> None:
    report["summary"] = {"checks": len(report["checks"]), "warnings": len(report["warnings"]), "hard_failures": len(report["hard_failures"])}


def _safe_shape(value: Any) -> list[int] | None:
    try:
        return [int(item) for item in value.shape]
    except Exception:
        return None


def _module_shape(value: Any) -> list[int] | None:
    shape = _safe_shape(value)
    if shape is not None:
        return shape
    try:
        parameter = next(value.parameters())
        return _safe_shape(parameter)
    except Exception:
        return None


def _tensor_outputs(value: Any, prefix: str = "output") -> Iterable[tuple[str, Any]]:
    if value is None:
        return
    if hasattr(value, "detach") and hasattr(value, "shape"):
        yield prefix, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _tensor_outputs(item, f"{prefix}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensor_outputs(item, f"{prefix}[{index}]")
    else:
        for key in ("embedding", "embeddings", "latent", "framewise_output", "clipwise_output", "logits", "x"):
            if hasattr(value, key):
                yield from _tensor_outputs(getattr(value, key), f"{prefix}.{key}")


def _forward_model(model: Any, candidate: Any) -> tuple[Any, str]:
    """Call a Mellow wrapper or official HTSAT with its supported API."""

    attempts = [(candidate,), (candidate, None), (candidate, None, True)]
    errors: list[str] = []
    for args in attempts:
        try:
            return model(*args), f"args={len(args)}"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"args={len(args)}:{type(exc).__name__}: {exc}")
    raise RuntimeError("HTSAT forward call failed: " + " | ".join(errors))


def _module_candidates(root: Path, implementation: str) -> list[tuple[str, Path | None]]:
    if not root.is_dir():
        return []
    files = sorted(
        (path for path in root.rglob("*.py") if "__pycache__" not in path.parts),
        key=lambda path: str(path).lower(),
    )
    preferred = [path for path in files if path.name.lower() in {"htsat.py", "htsat_model.py", "model.py"} or "htsat" in path.name.lower()]
    if not preferred:
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")[:256_000]
            except OSError:
                continue
            if "HTSAT" in text or "htsat" in text:
                preferred.append(path)
    if implementation == "mellow":
        preferred = [path for path in preferred if "mellow" in str(path).lower()] + preferred
    elif implementation == "official":
        preferred = [path for path in preferred if "mellow" not in str(path).lower()] + preferred
    seen: set[str] = set()
    result: list[tuple[str, Path | None]] = []
    for path in preferred:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            result.append((path.stem, path))
    return result


def _import_module(root: Path, name: str, path: Path | None) -> tuple[types.ModuleType, str]:
    root_string = str(root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    for child in sorted(root.glob("*/")):
        child_string = str(child.resolve())
        if child_string not in sys.path:
            sys.path.insert(0, child_string)
    errors: list[str] = []
    import_names = [name, name.lower(), f"model.{name.lower()}", f"models.{name.lower()}", f"mellow.{name.lower()}"]
    for import_name in import_names:
        try:
            module = importlib.import_module(import_name)
            return module, str(getattr(module, "__file__", import_name))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{import_name}: {type(exc).__name__}: {exc}")
    if path is not None:
        module_name = f"rsmol_audio_remote_{abs(hash(str(path.resolve())))}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError("no import spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module, str(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    raise ImportError("; ".join(errors[-8:]))


def _add_remote_root(root: Path) -> None:
    root_string = str(root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    for child in sorted(root.glob("*/")):
        child_string = str(child.resolve())
        if child_string not in sys.path:
            sys.path.insert(0, child_string)


def _import_explicit_mellow(root: Path) -> tuple[types.ModuleType, str, type[Any], type[Any] | None]:
    """Import the Mellow adapter/backbone by its documented module path."""

    _add_remote_root(root)
    module = importlib.import_module("mellow.model.htsat")
    wrapper_cls = getattr(module, "HTSATWrapper", None)
    backbone_cls = getattr(module, "HTSAT_Swin_Transformer", None)
    if not inspect.isclass(wrapper_cls):
        raise ImportError("mellow.model.htsat.HTSATWrapper is missing")
    if not inspect.isclass(backbone_cls):
        backbone_cls = None
    return module, str(getattr(module, "__file__", "mellow.model.htsat")), wrapper_cls, backbone_cls


def _find_model_class(module: types.ModuleType) -> type[Any]:
    candidates: list[type[Any]] = []
    for _, value in vars(module).items():
        if not inspect.isclass(value) or value.__module__ != module.__name__:
            continue
        name = value.__name__.lower()
        if "htsat" in name or "swin" in name:
            candidates.append(value)
    if not candidates:
        raise ImportError(f"no HTSAT/Swin model class found in {module.__name__}")
    candidates.sort(key=lambda item: ("htsat" not in item.__name__.lower(), item.__name__))
    return candidates[0]


def _htsat_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        sample_rate=32000, window_size=1024, hop_size=320,
        mel_bins=64, fmin=50, fmax=14000, classes_num=527,
        htsat_spec_size=256, htsat_patch_size=4, htsat_window_size=8,
        htsat_depth=[2, 2, 6, 2], htsat_dim=96, htsat_stride=4,
        htsat_num_head=[4, 8, 16, 32],
    )


def _construct_official_model(cls: type[Any]) -> Any:
    """Construct official HTSAT with the complete AudioSet configuration."""

    config = _htsat_config()
    values: dict[str, Any] = {
        "spec_size": 256, "patch_size": 4, "in_chans": 1,
        "num_classes": 527, "classes_num": 527, "species_num": 527,
        "window_size": 8, "config": config,
        "depths": [2, 2, 6, 2], "embed_dim": 96,
        "patch_stride": 4, "num_heads": [4, 8, 16, 32],
        "mlp_ratio": 4.0, "qkv_bias": True, "qk_scale": None,
        "drop_rate": 0.0, "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1, "ape": False, "patch_norm": True,
        "use_checkpoint": False,
    }
    signature = inspect.signature(cls)
    kwargs = {
        name: values[name]
        for name, parameter in signature.parameters.items()
        if name in values and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return cls(**kwargs)


def _construct_mellow_wrapper(wrapper_cls: type[Any], backbone_cls: type[Any] | None) -> tuple[Any, dict[str, Any]]:
    """Build the explicit Mellow wrapper and its configured HTSAT child."""

    config = _htsat_config()
    backbone = _construct_official_model(backbone_cls) if backbone_cls is not None else None
    signature = inspect.signature(wrapper_cls)
    values: dict[str, Any] = {
        "htsat": backbone, "sed_model": backbone, "backbone": backbone,
        "config": config, "dataset": None,
    }
    kwargs = {
        name: values[name]
        for name, parameter in signature.parameters.items()
        if name in values and values[name] is not None and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    wrapper = wrapper_cls(**kwargs)
    child = getattr(wrapper, "htsat", None)
    if child is None:
        child = getattr(wrapper, "sed_model", None)
    if child is None:
        child = getattr(wrapper, "backbone", None)
    if child is None:
        raise AttributeError("HTSATWrapper has no htsat/sed_model/backbone child")
    return wrapper, {"config": vars(config), "backbone_constructor": backbone_cls.__name__ if backbone_cls is not None else None, "wrapper_constructor_kwargs": sorted(kwargs)}


def _state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and candidate and all(hasattr(item, "shape") for item in candidate.values()):
                return dict(candidate)
        if payload and all(hasattr(item, "shape") for item in payload.values()):
            return dict(payload)
        for candidate in payload.values():
            try:
                found = _state_dict(candidate)
                if found:
                    return found
            except ValueError:
                continue
    raise ValueError("checkpoint does not contain a tensor state_dict")


def _normalise_state_dict(state: dict[str, Any], model_keys: Iterable[str]) -> tuple[dict[str, Any], str]:
    model_key_set = set(model_keys)
    prefixes = ("module.", "model.", "net.", "htsat.", "backbone.", "sed_model.")
    best_state, best_prefix, best_overlap = state, "", len(set(state) & model_key_set)
    for prefix in prefixes:
        candidate = {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in state.items()}
        overlap = len(set(candidate) & model_key_set)
        if overlap > best_overlap:
            best_state, best_prefix, best_overlap = candidate, prefix, overlap
    return best_state, best_prefix


def _mellow_backbone_state(state: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Apply Mellow's checkpoint convention: remove fixed ``sed_model.``."""

    stripped: dict[str, Any] = {}
    removed = 0
    for key, value in state.items():
        new_key = str(key)
        if new_key.startswith("module.sed_model."):
            new_key = new_key[len("module.sed_model."):]
            removed += 1
        elif new_key.startswith("sed_model."):
            new_key = new_key[len("sed_model."):]
            removed += 1
        stripped[new_key] = value
    return stripped, "sed_model." if removed else "<none>"


def _load_audio(path: Path, torch: Any, target_sr: int = 32000, seconds: int = 10) -> tuple[Any, dict[str, Any]]:
    waveform: Any = None
    sample_rate: int | None = None
    loader = ""
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(str(path))
        loader = "torchaudio"
    except Exception:
        try:
            import soundfile as sf
            data, sample_rate = sf.read(str(path), always_2d=True)
            waveform = torch.from_numpy(data.T)
            loader = "soundfile"
        except Exception:
            with wave.open(str(path), "rb") as handle:
                sample_rate = int(handle.getframerate())
                channels = int(handle.getnchannels())
                frames = int(handle.getnframes())
                sample_width = int(handle.getsampwidth())
                raw = handle.readframes(frames)
            if sample_width != 2:
                raise RuntimeError("wave fallback only supports 16-bit PCM; install torchaudio or soundfile")
            import numpy as np
            data = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels).astype("float32") / 32768.0
            waveform = torch.from_numpy(data.T)
            loader = "wave"
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.float()
    source_channels, source_samples = int(waveform.shape[0]), int(waveform.shape[-1])
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if int(sample_rate) != target_sr:
        try:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, int(sample_rate), target_sr)
        except Exception:
            target_samples = max(1, round(waveform.shape[-1] * target_sr / int(sample_rate)))
            waveform = torch.nn.functional.interpolate(waveform.unsqueeze(0), size=target_samples, mode="linear", align_corners=False).squeeze(0)
    desired = target_sr * seconds
    before_crop = int(waveform.shape[-1])
    if before_crop >= desired:
        waveform = waveform[..., :desired]
        operation = "crop" if before_crop > desired else "none"
    else:
        waveform = torch.nn.functional.pad(waveform, (0, desired - before_crop))
        operation = "pad"
    info = {"path": str(path), "loader": loader, "source_sample_rate": int(sample_rate), "target_sample_rate": target_sr, "source_channels": source_channels, "source_samples": source_samples, "pre_crop_samples": before_crop, "operation": operation, "waveform_shape": _safe_shape(waveform)}
    return waveform, info


def _resolve_manifest_audio(path: Path) -> Path:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    candidate = row.get("audio1_path") or row.get("filepath1")
                    if candidate:
                        candidate_path = Path(candidate)
                        return candidate_path if candidate_path.is_absolute() else path.parent / candidate_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("records", payload.get("data", [])) if isinstance(payload, dict) else []
    if rows and isinstance(rows[0], dict):
        candidate = rows[0].get("audio1_path") or rows[0].get("filepath1")
        if candidate:
            candidate_path = Path(candidate)
            return candidate_path if candidate_path.is_absolute() else path.parent / candidate_path
    raise ValueError(f"manifest has no audio1_path: {path}")


def _config_observation(model: Any) -> dict[str, Any]:
    expected = {"sample_rate": 32000, "window_size": 1024, "hop_size": 320, "mel_bins": 64, "fmin": 50, "fmax": 14000, "classes_num": 527}
    actual: dict[str, Any] = {}
    for name in expected:
        for owner in (model, getattr(model, "config", None)):
            if owner is not None and hasattr(owner, name):
                value = getattr(owner, name)
                if isinstance(value, (int, float, str)):
                    actual[name] = value
                    break
    checks = {name: (name not in actual or actual[name] == value) for name, value in expected.items()}
    return {"expected": expected, "actual": actual, "checks": checks, "all_observed_values_match": all(checks.values())}


def run(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "stage": "stage2_htsat_5_10_5_mellow",
        "status": "FAIL",
        "cuda_required": True,
        "configuration": {"mellow_root": str(args.mellow_root), "htsat_root": str(args.htsat_root), "checkpoint": str(args.htsat_checkpoint), "implementation_requested": args.implementation, "device_requested": args.device, "construction_only": bool(args.construction_only), "target_sample_rate": 32000, "target_seconds": 10, "formal_world_size": 8, "formal_micro_batch_per_gpu": 4, "formal_global_micro_batch": 32},
        "checks": [], "warnings": [], "hard_failures": [], "traceback": None,
    }
    try:
        if args.device != "cuda":
            _record(report, "explicit_cuda_device", False, "Stage 2 requires --device cuda; CPU is construction/audit-only", warning=False)
            report["status"] = "FAIL"
            _summary(report)
            return report
        import torch
        report["torch"] = {"version": str(getattr(torch, "__version__", "unknown")), "cuda_available": bool(torch.cuda.is_available()), "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0}
        if not torch.cuda.is_available():
            _record(report, "cuda_available", False, "CUDA is unavailable. Submit this script through the GPU wrapper and rerun with --device cuda")
            report["next_step"] = "Run code/RSmol/scripts/audit_audio_stage2_htsat_5_10_5_mellow.sh on a CUDA node"
            report["status"] = "FAIL"
            _summary(report)
            return report
        _record(report, "cuda_available", True, report["torch"])
        label = ""
        root = args.mellow_root
        module: types.ModuleType
        source_path = ""
        model: Any
        backbone: Any
        model_class: type[Any]
        mellow_constructor: dict[str, Any] = {}
        mellow_error: str | None = None
        if args.implementation in {"auto", "mellow"}:
            try:
                module, source_path, wrapper_class, backbone_class = _import_explicit_mellow(args.mellow_root)
                model, mellow_constructor = _construct_mellow_wrapper(wrapper_class, backbone_class)
                backbone = getattr(model, "htsat", None)
                if backbone is None:
                    backbone = getattr(model, "sed_model", None)
                if backbone is None:
                    backbone = getattr(model, "backbone", None)
                if backbone is None:
                    raise AttributeError("HTSATWrapper has no accessible HTSAT child")
                label = "mellow"
                root = args.mellow_root
                model_class = wrapper_class
            except Exception as exc:  # noqa: BLE001
                mellow_error = f"{type(exc).__name__}: {exc}"
                if args.implementation == "mellow":
                    raise
        if not label:
            import_errors: list[str] = []
            for module_name, module_path in _module_candidates(args.htsat_root, "official"):
                try:
                    module, source_path = _import_module(args.htsat_root, module_name, module_path)
                    model_class = _find_model_class(module)
                    backbone = _construct_official_model(model_class)
                    model = backbone
                    label = "official_fallback" if mellow_error else "official"
                    root = args.htsat_root
                    break
                except Exception as exc:  # noqa: BLE001
                    import_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            if not label:
                reason = (f" Mellow import failed: {mellow_error}." if mellow_error else "")
                raise ImportError("No explicit Mellow or official HTSAT implementation found." + reason + " " + " | ".join(import_errors[-10:]))
        report["implementation"] = {"selected": label, "root": str(root), "module": module.__name__, "source_path": source_path, "class": model_class.__name__, "mellow_fallback_reason": mellow_error, "constructor": mellow_constructor}
        report["configuration_observed_before_load"] = _config_observation(backbone)
        try:
            payload = torch.load(args.htsat_checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(args.htsat_checkpoint, map_location="cpu")
        state = _state_dict(payload)
        if label == "mellow":
            normalised, prefix = _mellow_backbone_state(state)
            target = backbone
        else:
            normalised, prefix = _normalise_state_dict(state, backbone.state_dict().keys())
            target = backbone
        load_result = target.load_state_dict(normalised, strict=False)
        missing = sorted(str(item) for item in load_result.missing_keys)
        unexpected = sorted(str(item) for item in load_result.unexpected_keys)
        report["checkpoint"] = {"path": str(args.htsat_checkpoint), "payload_type": type(payload).__name__, "state_key_count": len(state), "normalised_prefix": prefix, "target": "wrapper.htsat" if label == "mellow" else "official_htsat", "missing_keys": missing, "unexpected_keys": unexpected, "strict_compatible": not missing and not unexpected}
        _record(report, "htsat_backbone_strict_compatible", not missing and not unexpected, report["checkpoint"])
        if missing or unexpected:
            report["status"] = "FAIL"
            _summary(report)
            return report
        c2l = getattr(model, "c2l", None) if label == "mellow" else None
        if label == "mellow":
            report["c2l"] = {"present": c2l is not None, "random_initialised": c2l is not None, "shape": _module_shape(c2l) if c2l is not None else None, "checkpoint_included": False, "note": "AudioSet checkpoint contains HTSAT backbone only; c2l is intentionally not loaded"}
        model = model.to(torch.device("cuda"))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
        trainable_count = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
        report["model"] = {"parameter_count": parameter_count, "trainable_parameter_count": trainable_count, "frozen_parameter_count": parameter_count - trainable_count, "training": bool(model.training), "device": str(next(model.parameters()).device) if any(True for _ in model.parameters()) else "cuda"}
        observed_config = _config_observation(model)
        report["configuration"]["observed"] = observed_config
        _record(report, "htsat_config_values_match", observed_config["all_observed_values_match"], observed_config)
        missing_config = sorted(set(observed_config["expected"]) - set(observed_config["actual"]))
        if missing_config:
            _record(report, "htsat_config_values_observed", False, {"missing": missing_config}, warning=True)
        _record(report, "model_frozen_eval", not model.training and trainable_count == 0, report["model"])
        if args.construction_only:
            report["status"] = "INCOMPLETE"
            report["next_step"] = "Provide --audio-path or --manifest to run the required single-audio CUDA forward"
            _summary(report)
            return report
        audio_path = args.audio_path or _resolve_manifest_audio(args.manifest) if args.manifest else args.audio_path
        if audio_path is None:
            _record(report, "audio_input_present", False, "Provide --audio-path or --manifest; no waveform was fabricated")
            report["status"] = "INCOMPLETE"
            _summary(report)
            return report
        if not audio_path.is_file():
            _record(report, "audio_input_present", False, str(audio_path))
            report["status"] = "FAIL"
            _summary(report)
            return report
        waveform, waveform_info = _load_audio(audio_path, torch)
        report["audio"] = waveform_info
        waveform_cuda = waveform.to("cuda")
        report["audio"]["cuda_waveform_shape"] = _safe_shape(waveform_cuda)
        spec_observation: dict[str, Any] = {}
        for name in ("spectrogram_extractor", "logmel_extractor"):
            extractor = getattr(model, name, None)
            if callable(extractor):
                try:
                    with torch.no_grad():
                        spec = extractor(waveform_cuda)
                    spec_observation[name] = {"shape": _safe_shape(spec), "dtype": str(getattr(spec, "dtype", "unknown")), "finite": bool(torch.isfinite(spec).all())}
                except Exception as exc:  # noqa: BLE001
                    spec_observation[name] = {"error": f"{type(exc).__name__}: {exc}"}
        report["spectrogram"] = spec_observation
        forward_errors: list[str] = []
        output: Any = None
        input_used = "waveform[B,T]"
        for candidate, label_name in ((waveform_cuda, "waveform[B,T]"), (waveform_cuda.unsqueeze(1), "waveform[B,1,T]")):
            try:
                with torch.no_grad():
                    output, call_signature = _forward_model(model, candidate)
                input_used = f"{label_name};{call_signature}"
                break
            except Exception as exc:  # noqa: BLE001
                forward_errors.append(f"{label_name}: {type(exc).__name__}: {exc}")
        if output is None:
            raise RuntimeError("HTSAT forward failed for supported waveform layouts: " + " | ".join(forward_errors))
        tensors = {name: {"shape": _safe_shape(value), "dtype": str(value.dtype), "finite": bool(torch.isfinite(value).all()), "numel": int(value.numel())} for name, value in _tensor_outputs(output)}
        embedding_items = {name: item for name, item in tensors.items() if any(token in name.lower() for token in ("embedding", "latent", "framewise", "clipwise", "logit"))}
        if not embedding_items and "output" in tensors:
            embedding_items["embedding"] = tensors["output"]
        embedding_shape = next(iter(embedding_items.values()), {}).get("shape")
        report["embedding"] = {"shape": embedding_shape, "dimension": embedding_shape[-1] if embedding_shape else None, "length": embedding_shape[-2] if embedding_shape and len(embedding_shape) >= 3 else None, "finite": bool(embedding_items) and all(item["finite"] for item in embedding_items.values())}
        report["forward"] = {"input_used": input_used, "output_type": type(output).__name__, "output_keys": sorted(tensors), "tensor_outputs": tensors, "embedding_latent_framewise": embedding_items}
        _record(report, "forward_outputs_finite", bool(tensors) and all(item["finite"] for item in tensors.values()), report["forward"])
        _record(report, "waveform_contract", report["audio"]["waveform_shape"] == [1, 320000] and report["audio"]["target_sample_rate"] == 32000, report["audio"])
        report["status"] = "PASS" if not report["hard_failures"] else "FAIL"
    except Exception as exc:  # noqa: BLE001
        report["traceback"] = traceback.format_exc()
        report["hard_failures"].append({"name": "stage2_exception", "passed": False, "detail": f"{type(exc).__name__}: {exc}"})
        report["status"] = "FAIL"
    _summary(report)
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
    print(json.dumps({"stage": report["stage"], "status": report["status"], "summary": report.get("summary", {}), "report": str(args.report_path)}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
