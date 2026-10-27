"""Tokenizer/asset comparison with the ordinary USAD2 baseline; no Torch import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from contract import digest, validate_checkpoint_tokenizer


TOKENIZER_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json", "tokenizer.model",
                   "spiece.model", "vocab.json", "merges.txt"}


def tokenizer_asset_identity(directory):
    root = Path(directory).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"comparison tokenizer config missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("hidden_size") != 576 or config.get("num_hidden_layers") != 30
            or config.get("recursive_layer_count") is not None):
        raise ValueError("comparison directory must be the ordinary 30-layer, 576-d SmolLM2 baseline")
    if not isinstance(config.get("vocab_size"), int) or config["vocab_size"] <= 0:
        raise ValueError("comparison config has no valid vocab_size")
    files = sorted(p for p in root.iterdir() if p.is_file() and p.name in TOKENIZER_FILES)
    if not any(p.name in {"tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json"} for p in files):
        raise FileNotFoundError(f"comparison tokenizer vocabulary files missing: {root}")
    if any(p.stat().st_size == 0 for p in files):
        raise ValueError(f"comparison tokenizer contains an empty asset: {root}")
    return {"root": str(root), "vocab_size": config["vocab_size"],
            "files": [{"name": p.name, "bytes": p.stat().st_size,
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}


def compare_tokenizers(mesh, baseline, model_vocab_size):
    """Compare complete token IDs plus the actual dataset encode_plus contract."""
    mesh_audit = validate_checkpoint_tokenizer(mesh, model_vocab_size)
    baseline_audit = validate_checkpoint_tokenizer(baseline, model_vocab_size)
    if mesh_audit["mizar_pad_id"] != 17 or baseline_audit["mizar_pad_id"] != 17:
        raise ValueError("comparison requires the baseline ! PAD ID 17")
    mesh_vocab, baseline_vocab = mesh.get_vocab(), baseline.get_vocab()
    if mesh_vocab != baseline_vocab:
        raise ValueError("MeSH and ordinary USAD2 baseline token-to-ID dictionaries differ")
    for field in ("padding_side", "truncation_side"):
        if getattr(mesh, field) != getattr(baseline, field):
            raise ValueError(f"tokenizer {field} differs from the ordinary USAD2 baseline")
    fixtures = ("hello", "Question A", "Keep CASE!", "你好，声音。", "!",
                "<think>A</think><answer>B</answer>", "<|endoftext|>", "hello " * 2000)
    encodings = []
    for length in (256, 300):
        for index, sentence in enumerate(fixtures):
            kwargs = dict(text=sentence + " <|endoftext|>", add_special_tokens=True,
                          max_length=length, padding="max_length", truncation=True, return_tensors="pt")
            actual, expected = mesh.encode_plus(**kwargs), baseline.encode_plus(**kwargs)
            normalized = []
            for value in (actual, expected):
                normalized.append({key: (value[key].tolist() if hasattr(value[key], "tolist") else value[key])
                                   for key in ("input_ids", "attention_mask")})
            if normalized[0] != normalized[1]:
                raise ValueError(f"tokenizer encoding differs: fixture={index}, length={length}")
            ids, mask = normalized[0]["input_ids"][0], normalized[0]["attention_mask"][0]
            if len(ids) != length or len(mask) != length:
                raise ValueError("tokenizer did not produce fixed-length prompt/answer tensors")
            if any(not valid and token != 17 for token, valid in zip(ids, mask)):
                raise ValueError("answer mask would exclude a non-PAD target; CE objectives would differ")
            encodings.append(normalized[0])
    return {"status": "PASS", "vocabulary_sha256": digest(mesh_vocab),
            "vocabulary_rows": len(mesh_vocab), "encoding_comparisons": len(encodings),
            "encodings_sha256": digest(encodings), "pad_id": 17, "eos_sep_id": 0,
            "mesh_tokenizer": mesh_audit, "baseline_tokenizer": baseline_audit,
            "loss_targets": "answer_mask AND non-PAD equals non-PAD for tokenizer-produced masks"}
