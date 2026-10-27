"""Synthetic tokenizer fixtures exercise comparison logic without Transformers."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_usad2_small"
sys.path.insert(0, str(ROUTE))
from comparison import compare_tokenizers, tokenizer_asset_identity
from contract import digest, write_json


class SyntheticTokenizer:
    pad_token, pad_token_id, eos_token_id = "<|endoftext|>", 0, 0
    padding_side = truncation_side = "right"

    def __init__(self):
        self.vocab = {"!" if i == 17 else f"t{i}": i for i in range(32)}
        self.calls = []
        self.bad_encoding = self.bad_mask = False

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab.copy()

    def add_special_tokens(self, mapping):
        self.pad_token, self.pad_token_id = mapping["pad_token"], 17
        return 1  # Existing special-token registration can report one without growth.

    def encode(self, text, add_special_tokens=True):
        return [17] if text == "!" else [0]

    def encode_plus(self, **kwargs):
        self.calls.append(kwargs)
        length = kwargs["max_length"]
        ids = [17 if char == "!" else ord(char) % 13 + 1 for char in kwargs["text"]][:length]
        mask = [1] * len(ids) + [0] * (length - len(ids))
        ids += [17] * (length - len(ids))
        if self.bad_encoding:
            ids[0] = 31
        if self.bad_mask:
            mask[0] = 0
        return {"input_ids": [ids], "attention_mask": [mask]}


class MeSHUSAD2TokenizerTests(unittest.TestCase):
    def test_full_vocabulary_and_actual_dataset_encoding_options_are_compared(self):
        mesh, baseline = SyntheticTokenizer(), SyntheticTokenizer()
        result = compare_tokenizers(mesh, baseline, 32)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["vocabulary_sha256"], digest(mesh.vocab))
        self.assertEqual(result["encoding_comparisons"], 16)
        self.assertEqual(result["mesh_tokenizer"]["added_tokens"], 1)
        self.assertEqual(result["mesh_tokenizer"]["vocabulary_growth"], 0)
        self.assertEqual(mesh.calls, baseline.calls)
        self.assertEqual({row["max_length"] for row in mesh.calls}, {256, 300})
        self.assertTrue(all(row["return_tensors"] == "pt" and row["padding"] == "max_length"
                            and row["truncation"] and row["text"].endswith(" <|endoftext|>")
                            for row in mesh.calls))

    def test_same_vocabulary_size_with_different_ids_is_rejected(self):
        mesh, baseline = SyntheticTokenizer(), SyntheticTokenizer()
        baseline.vocab["t1"], baseline.vocab["t2"] = baseline.vocab["t2"], baseline.vocab["t1"]
        with self.assertRaisesRegex(ValueError, "token-to-ID"):
            compare_tokenizers(mesh, baseline, 32)

    def test_identical_vocabulary_with_different_encoding_or_padding_is_rejected(self):
        for kind in ("encoding", "padding", "truncation"):
            mesh, baseline = SyntheticTokenizer(), SyntheticTokenizer()
            if kind == "encoding":
                mesh.bad_encoding = True
            else:
                setattr(mesh, kind + "_side", "left")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                compare_tokenizers(mesh, baseline, 32)

    def test_mask_that_excludes_nonpad_targets_is_rejected_even_if_both_tokenizers_match(self):
        mesh, baseline = SyntheticTokenizer(), SyntheticTokenizer()
        mesh.bad_mask = baseline.bad_mask = True
        with self.assertRaisesRegex(ValueError, "non-PAD target"):
            compare_tokenizers(mesh, baseline, 32)

    def test_actual_vocabulary_growth_is_rejected(self):
        class GrowingTokenizer(SyntheticTokenizer):
            def add_special_tokens(self, mapping):
                self.vocab["new_token"] = 32
                return super().add_special_tokens(mapping)
        with self.assertRaisesRegex(ValueError, "changed vocabulary rows"):
            compare_tokenizers(GrowingTokenizer(), SyntheticTokenizer(), 32)

    def test_reference_asset_changes_affect_identity_and_recursive_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(hidden_size=576, num_hidden_layers=30, vocab_size=32)
            write_json(root / "config.json", config)
            write_json(root / "tokenizer.json", {"isolated_synthetic_fixture": True})
            first = tokenizer_asset_identity(root)
            write_json(root / "tokenizer.json", {"isolated_synthetic_fixture": "changed"})
            self.assertNotEqual(first, tokenizer_asset_identity(root))
            write_json(root / "config.json", dict(config, recursive_layer_count=20))
            with self.assertRaisesRegex(ValueError, "ordinary 30-layer"):
                tokenizer_asset_identity(root)


if __name__ == "__main__":
    unittest.main()
