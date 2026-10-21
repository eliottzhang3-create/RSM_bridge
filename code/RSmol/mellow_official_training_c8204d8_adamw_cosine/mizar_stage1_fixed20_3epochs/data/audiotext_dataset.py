"""Online raw audio + released Mizar S1 text/tokenization semantics."""
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from contract import ANSWER_TOKENS, PROMPT_TOKENS, data_order
from data.fixed20 import load_fixed20
from data.indexed_jsonl import IndexedRows
from data.template import BOTH, FIRST, SECOND, EMOTION, WORDONLY, LONGLINEONLY, DETAILONLY


def text_pair(row):
    """Port of runtime/stage1/data/audiotext_dataset.py _create_answer_input."""
    prompt = row["input"]

    def metadata(key):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        nested = row.get("metadata", {})
        value = nested.get(key) if isinstance(nested, dict) else None
        return value.strip() if isinstance(value, str) and value.strip() else ""

    def answer_text():
        value = row.get("answer")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("missing nonempty Mizar answer")
        return value.strip()

    if prompt == "caption both audios":
        prompt = random.choice(BOTH)
        a, b = metadata("caption1"), metadata("caption2")
        answer = f"The audio 1 is {a.lower()}. The audio 2 is {b.lower()}." if a and b else answer_text()
        answer = answer.replace("..", ".")
    elif prompt in {"caption first audio", "caption second audio"}:
        first = prompt == "caption first audio"
        prompt = random.choice(FIRST if first else SECOND)
        caption = metadata("caption1" if first else "caption2")
        answer = f"The audio {1 if first else 2} is {caption.lower()}." if caption else answer_text()
        answer = answer.replace("..", ".")
    elif prompt in {"explain the difference in few words", "explain the difference in a sentence", "explain the difference in detail"}:
        templates = {"explain the difference in few words": WORDONLY,
                     "explain the difference in a sentence": LONGLINEONLY,
                     "explain the difference in detail": DETAILONLY}
        prompt, answer = random.choice(templates[prompt]), row["answer"]
    elif "emo_emo_emo" in prompt:
        prompt, answer = prompt.replace("emo_emo_emo", random.choice(EMOTION)), row["answer"]
    elif row.get("prompt_contract") == "ke_omni_r_cot_v1":
        answer = row["answer"]
    else:
        prompt, answer = prompt.lower(), row["answer"].lower()
    return prompt, answer


class AudioTextDataset(Dataset):
    def __init__(self, metadata_path, stage_root, tokenizer_path):
        self.rows = IndexedRows(metadata_path)
        self.order = data_order(len(self.rows))
        self.stage_root = Path(stage_root).resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.tokenizer.add_special_tokens({"pad_token": "!"})
        self.pad_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        if (self.tokenizer.pad_token != "!" or self.pad_id != self.tokenizer.pad_token_id
                or self.pad_id == self.tokenizer.eos_token_id):
            raise ValueError(
                "invalid Mizar pad token: expected a non-EOS ! token; "
                f"pad_token={self.tokenizer.pad_token!r} pad_id={self.pad_id} "
                f"tokenizer_pad_id={self.tokenizer.pad_token_id} eos_id={self.tokenizer.eos_token_id}")

    def __len__(self):
        return len(self.rows)

    def tokenize(self, sentence, length):
        return self.tokenizer.encode_plus(
            text=sentence + " <|endoftext|>", add_special_tokens=True,
            max_length=length, padding="max_length", truncation=True, return_tensors="pt")

    def original_item(self, original_index):
        row = self.rows[original_index]
        paths = [(self.stage_root / name).resolve() for name in row["s1_raw_paths"]]
        if any(not path.is_relative_to(self.stage_root) for path in paths):
            raise ValueError("audio path escaped the job's staging directory")
        try:
            first, second = load_fixed20(paths[0], paths[1] if len(paths) == 2 else "")
        except Exception as exc:
            raise RuntimeError(f"decode failed: row={original_index} source={row['s1_source']} raw={paths}") from exc
        prompt, answer = text_pair(row)
        return {"audio1": first, "audio2": second,
                "input": self.tokenize(prompt, PROMPT_TOKENS),
                "answer": self.tokenize(answer, ANSWER_TOKENS),
                "row_index": original_index, "source": row["s1_source"]}

    def __getitem__(self, index):
        return self.original_item(self.order[index])


def collate_fn(items):
    return {
        "audio1": torch.stack([x["audio1"] for x in items]),
        "audio2": torch.stack([x["audio2"] for x in items]),
        "input": {k: torch.cat([x["input"][k] for x in items]) for k in items[0]["input"]},
        "answer": {k: torch.cat([x["answer"][k] for x in items]) for k in items[0]["answer"]},
        "row_index": [x["row_index"] for x in items],
        "source": [x["source"] for x in items],
    }
