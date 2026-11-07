"""CED asset validation copied from the reference route; no training contract."""
import json
from pathlib import Path
from .contract import AUDIO_TOKENS

CED_ASSET_FILES = ("config.json", "configuration_ced.py", "modeling_ced.py",
                   "feature_extraction_ced.py", "preprocessor_config.json", "model.safetensors")


CED_EXPECTED_CONFIG = {
    "embed_dim": 384, "depth": 12, "num_heads": 6, "n_mels": 64,
    "patch_size": 16, "patch_stride": 16, "target_length": 1012,
    "n_fft": 512, "win_size": 512, "hop_size": 160,
    "center": True, "f_min": 0, "f_max": 8000,
    "pooling": "mean", "pad_last": True, "outputdim": 527,
}


def ced_label_metadata(config):
    """Read the real local labels, retaining duplicate display names by index."""
    raw = config.get("id2label")
    error = "incompatible CED-Small assets: config.json needs complete local id2label metadata (indices 0..526)"
    if not isinstance(raw, dict) or len(raw) != 527:
        raise ValueError(error)
    labels = {}
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            raise ValueError(error) from None
        if str(index) != str(key) or not isinstance(value, str) or not value.strip():
            raise ValueError(error)
        labels[index] = value
    if set(labels) != set(range(527)):
        raise ValueError(error)
    labels = dict(sorted(labels.items()))
    reverse = config.get("label2id")
    if reverse is None:
        reverse = {label: index for index, label in labels.items()}
    if (not isinstance(reverse, dict) or set(reverse) != set(labels.values())
            or any(type(index) is not int or index not in labels or labels[index] != label
                   for label, index in reverse.items())):
        raise ValueError("incompatible CED-Small assets: inconsistent local label2id metadata")
    return labels, dict(reverse)


def validate_ced_config(config):
    """Reject incompatible feature geometry instead of reshaping/truncating it."""
    if not isinstance(config, dict):
        raise ValueError("incompatible CED-Small assets: config.json must contain an object")
    problems = [f"{name}={config.get(name)!r}, expected {expected!r}"
                for name, expected in CED_EXPECTED_CONFIG.items()
                if config.get(name) != expected]
    if config.get("model_type") != "ced":
        problems.append(f"model_type={config.get('model_type')!r}, expected 'ced'")
    auto_map = config.get("auto_map", {})
    if not isinstance(auto_map, dict):
        raise ValueError("incompatible CED-Small assets: auto_map must contain an object")
    for name, expected in (("AutoConfig", "configuration_ced.CedConfig"),
                           ("AutoModelForAudioClassification", "modeling_ced.CedForAudioClassification")):
        if auto_map.get(name) != expected:
            problems.append(f"auto_map.{name}={auto_map.get(name)!r}, expected {expected!r}")
    if problems:
        raise ValueError("incompatible CED-Small assets: " + "; ".join(problems))
    labels, _ = ced_label_metadata(config)
    return {**CED_EXPECTED_CONFIG, "input_sampling_rate": 32000, "ced_sampling_rate": 16000,
            "mel_frames": 2001, "segments": 2, "hidden_tokens_per_segment": 252,
            "hidden_tokens": 504, "temporal_tokens": AUDIO_TOKENS, "output_dim": 576,
            "mapper": "freq_merge_126", "mapper_dropout": 0.05, "mapper_resample": False,
            "label_metadata": "local_config_json", "label_count": len(labels)}


def validate_ced_assets(root):
    root = Path(root).resolve()
    for name in CED_ASSET_FILES:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing/empty local CED asset: {path}")
    return validate_ced_config(json.loads((root / "config.json").read_text(encoding="utf-8")))
