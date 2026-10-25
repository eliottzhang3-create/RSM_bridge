"""Compare canonical and non-MeSH checkpoint tokenizers without loading weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from contract import (DEFAULT_TEXT_MODEL_DIR, recursive_checkpoint_contract,
                      validate_checkpoint_tokenizer)


def vocab_hash(vocab):
    data = json.dumps(vocab, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def compare_tokenizers(reference, recursive, model_vocab_size):
    old, new = reference.get_vocab(), recursive.get_vocab()
    reference_audit = validate_checkpoint_tokenizer(reference, len(reference))
    checkpoint_audit = validate_checkpoint_tokenizer(recursive, model_vocab_size)
    components = ("model", "normalizer", "pre_tokenizer", "post_processor", "decoder")
    backends = [json.loads(t.backend_tokenizer.to_str()) for t in (reference, recursive)]
    backend_equal = {key: backends[0].get(key) == backends[1].get(key) for key in components}
    metadata = ("eos_token_id", "bos_token_id", "unk_token_id", "pad_token_id",
                "padding_side", "truncation_side", "add_bos_token", "add_eos_token")
    metadata_equal = {key: getattr(reference, key, None) == getattr(recursive, key, None)
                      for key in metadata}
    probes = []
    for length in (256, 300):
        for text in ("", "hello", "! <|endoftext|>", "What sound can be heard? A dog barking.",
                     "音乐、语音与环境声 🎵\n", "hello " * 2000):
            args = {"return_tensors": None, "max_length": length,
                    "padding": "max_length", "truncation": True}
            left = dict(reference(text + " <|endoftext|>", **args))
            right = dict(recursive(text + " <|endoftext|>", **args))
            probes.append(left == right)
    checks = {
        "vocab_exactly_equal": old == new,
        "reference_vocab_unchanged_after_pad": old == reference.get_vocab(),
        "checkpoint_vocab_unchanged_after_pad": new == recursive.get_vocab(),
        "backend_components_equal": all(backend_equal.values()),
        "tokenizer_metadata_equal_after_pad": all(metadata_equal.values()),
        "padded_truncated_probe_encodings_equal": all(probes),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "reference_vocab_hash": vocab_hash(old), "checkpoint_vocab_hash": vocab_hash(new),
            "only_in_reference_count": len(old.keys() - new.keys()),
            "only_in_checkpoint_count": len(new.keys() - old.keys()),
            "different_id_count": sum(old[key] != new[key] for key in old.keys() & new.keys()),
            "backend_components_equal": backend_equal, "metadata_equal": metadata_equal,
            "probe_count": len(probes), "reference_mizar_pad": reference_audit,
            "checkpoint_mizar_pad": checkpoint_audit,
            "scope": "CPU tokenizer/config only; no model weights, CUDA or training executed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_TEXT_MODEL_DIR))
    parser.add_argument("--reference-tokenizer", type=Path, required=True)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    checkpoint = recursive_checkpoint_contract(args.checkpoint)
    from transformers import AutoTokenizer
    reference = AutoTokenizer.from_pretrained(args.reference_tokenizer, local_files_only=True)
    recursive = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    report = compare_tokenizers(reference, recursive, checkpoint["config"]["vocab_size"])
    report.update(checkpoint=str(args.checkpoint), reference_tokenizer=str(args.reference_tokenizer),
                  model_config=checkpoint["config"])
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
