#!/usr/bin/env python3
"""Run fixed-order MMAU test-mini inference for the audio MeSH checkpoint.

The evaluator deliberately keeps benchmark I/O independent from the training
dataset implementation.  MMAU test-mini is read from its parquet file in
physical row order, while the official JSON is used as the authoritative
record (and as a small id index) for prompt metadata and scoring fields.

Only the GPU/model path needs the project's torch/Transformers/Mellow
environment.  Metadata helpers, the JSONL resume store, and output
materializer are dependency-light so they can be tested on a CPU-only
checkout.  Model text is passed unchanged to the official scorer.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
import unicodedata
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
# The model packages live under ``code/RSmol`` while the established loader
# and generation helpers live beside this evaluator under ``scripts``.  Add
# both explicitly so direct execution and import-by-path behave identically.
for import_root in (SCRIPT_DIR, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


DEFAULT_CHECKPOINT = (
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "audio_5_10x2_5_mesh_mellow/partition_formal_answer_eos_v2_10epochs_20260918/"
    "checkpoint-037810"
)
DEFAULT_DATASET_DIR = "/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini"
DEFAULT_HTSAT = "/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt"
DEFAULT_MELLOW = "/hpc_stor03/sjtu_home/jinwei.zhang/code/mellow-main"
DEFAULT_SAMPLE_RATE = 32_000
DEFAULT_SOURCE_SAMPLE_RATE = 16_000
DEFAULT_AUDIO_SECONDS = 10
DEFAULT_MAX_PROMPT_TOKENS = 129
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_AUDIO_PREFIX_TOKENS = 130
DEFAULT_MAX_CONTEXT_LENGTH = 768
SMOKE_ROWS = 5
EXPECTED_FULL_ROWS = 1000
MMAU_VERSION = "MMAU-v05.15.25"
MMAU_OFFICIAL_COMMIT = "110127f54c0dfba3faa5ec9feee4a7e4148679c5"
MMAU_METADATA_SHA256 = "9f18fda99f8dbc2bc5ecd6323fb5063309f54969810b5c6f1caeb3b8d904cf1c"
MMAU_METADATA_CANONICAL_SHA256 = "04c6b38739179ec7f3044b05435f5a734970f774ac60c86b00ac5a65ee439859"
MMAU_EVALUATION_SHA256 = "85480e1c0dfe8ee1406e9c6e598eff0dca9e0216701f076faf69081c6aab1558"
PROMPT_FORMAT = "reasonaqa_lowercase_labels_no_choices_prefix_v1"
PREDICTION_FORMAT = "official_generated_text_strip_leading_abcd_label_v2"
INFERENCE_ONLY_STATUS = "INFERENCE_ONLY"
MELLOW_AUTHOR_REPLY_PROMPT_FORMAT = "mellow_github_issue5_lowercase_questionmark_labels_v1"
MELLOW_AUTHOR_REPLY_AUDIO_FORMAT = "mellow_wrapper_repeat_or_independent_random_crop_v1"
MELLOW_AUTHOR_REPLY_SCORER = "mellow_github_issue5_choice_label_prefix_exact_v1"
MELLOW_AUTHOR_REPLY_CONTEXT = {
    "source": "soham97/mellow GitHub issue #5 author reply",
    "issue_opened": "2025-04-16",
    "evaluated_benchmark": MMAU_VERSION,
    "evaluated_benchmark_release": "2025-05-15",
    "paper_score_exact_reproduction": False,
    "reason": (
        "the author-reply protocol predates MMAU-v05.15.25; the current benchmark revised "
        "questions, answers, and audio, so this is protocol reproduction on the revised set"
    ),
}


class RowSkip(Exception):
    """A normal per-row data/inference condition that must not stop a run."""

    def __init__(self, stage: str, reason: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.stage = str(stage)
        self.reason = str(reason)
        self.message = str(message)
        self.details = details


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return records, errors
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL record is not an object")
                records.append(value)
            except Exception as exc:
                # A process killed during a write can leave one partial final
                # line.  It is safe to ignore that line and continue from the
                # flushed records that precede it.
                errors.append(f"{path.name}:{line_number}: {exc}")
    return records, errors


class JsonlAppender:
    """Append-and-flush JSONL records used by the resumable evaluator."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")

    def append(self, record: Mapping[str, Any]) -> None:
        self.handle.write(json.dumps(dict(record), ensure_ascii=False, default=_json_default))
        self.handle.write("\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "JsonlAppender":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def row_key(row_index: int, sample_id: Any) -> str:
    """Return a stable key that remains unique for duplicate benchmark IDs."""

    normalized_id = "" if sample_id is None else str(sample_id).strip()
    return f"{int(row_index)}\t{normalized_id}"


class ProgressStore:
    """Recover and persist row completion state without re-running inference."""

    TERMINAL_STATUSES = frozenset({"generated", "skipped"})

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.progress_path = output_dir / "progress.jsonl"
        self.raw_path = output_dir / "raw_generations.jsonl"
        self.skipped_path = output_dir / "skipped.jsonl"
        self.state: dict[str, dict[str, Any]] = {}
        self.load_errors: list[str] = []
        self._load_existing()
        self.raw = JsonlAppender(self.raw_path)
        self.skipped = JsonlAppender(self.skipped_path)
        self.progress = JsonlAppender(self.progress_path)

    @staticmethod
    def _key_from_record(record: Mapping[str, Any]) -> str | None:
        if "row_index" not in record:
            return None
        try:
            return row_key(int(record["row_index"]), record.get("id"))
        except Exception:
            return None

    def _remember(self, record: Mapping[str, Any]) -> None:
        key = str(record.get("row_key") or self._key_from_record(record) or "")
        status = str(record.get("status", ""))
        if key and status in self.TERMINAL_STATUSES:
            self.state[key] = dict(record)

    def _load_existing(self) -> None:
        progress, errors = _read_jsonl(self.progress_path)
        self.load_errors.extend(errors)
        # Backward/recovery path for a run killed after raw/skipped append but
        # before the progress event.  Both files are append-only and are
        # de-duplicated by row index + ID in memory.  Always merge all three
        # streams: a kill between the raw append and progress append must not
        # cause that row to be inferred again on restart.
        raw, raw_errors = _read_jsonl(self.raw_path)
        skipped, skipped_errors = _read_jsonl(self.skipped_path)
        self.load_errors.extend(raw_errors)
        self.load_errors.extend(skipped_errors)
        for record in raw:
            self._remember(record)
        for record in skipped:
            self._remember(record)
        # Progress is the last writer in the normal path and therefore wins if
        # a future implementation records more than one terminal event.
        for event in progress:
            record = event.get("record") if isinstance(event.get("record"), dict) else event
            if isinstance(record, dict):
                self._remember(record)

    def has_terminal(self, row_index: int, sample_id: Any) -> bool:
        return row_key(row_index, sample_id) in self.state

    def get(self, row_index: int, sample_id: Any) -> dict[str, Any] | None:
        return self.state.get(row_key(row_index, sample_id))

    def append_raw(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        self.raw.append(payload)
        self.state[row_key(int(payload["row_index"]), payload.get("id"))] = payload
        self.progress.append({"status": payload["status"], "record": payload})

    def append_skip(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        self.skipped.append(payload)
        self.state[row_key(int(payload["row_index"]), payload.get("id"))] = payload
        self.progress.append({"status": "skipped", "record": payload})

    def close(self) -> None:
        self.raw.close()
        self.skipped.close()
        self.progress.close()

    def __enter__(self) -> "ProgressStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = " ".join(text.split())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`"}:
        text = text[1:-1].strip()
    return text


def _choice_label_index(value: Any) -> int | None:
    """Return an explicit option-label index without consuming answer text.

    Parenthesized/bracketed labels are unambiguous.  Bare labels such as
    ``A.`` or ``B)`` require following whitespace so real content including
    ``F. Scott Fitzgerald``, ``J.D. Salinger``, ``E-guitar`` and ``B:maj/1``
    is not treated as an option label.
    """

    match = re.match(
        r"^\s*(?:\(([A-Za-z])\)|\[([A-Za-z])\]|([A-Da-d])[.)\]:-](?=\s))",
        str(value),
    )
    if not match:
        return None
    label = next(group for group in match.groups() if group is not None).upper()
    return ord(label) - ord("A")


def _strip_choice_label(value: Any, *, expected_index: int | None = None) -> str:
    text = str(value).strip()
    if expected_index is None:
        # Without positional context, only bracketed labels are safe to strip.
        # A bare ``F. `` may be the beginning of a person's name.
        return re.sub(
            r"^\s*(?:\([A-Za-z]\)|\[[A-Za-z]\])\s*",
            "",
            text,
            count=1,
        )
    label_index = _choice_label_index(text)
    if label_index is None or label_index != expected_index:
        return text
    return re.sub(
        r"^\s*(?:\([A-Za-z]\)|\[[A-Za-z]\]|[A-Da-d][.)\]:-](?=\s))\s*",
        "",
        text,
        count=1,
    )


def choices_match_fixed_order(left: Sequence[Any], right: Sequence[Any]) -> bool:
    """Compare parquet choices to canonical official choices in fixed order."""

    if len(left) != len(right):
        return False
    for index, (left_choice, right_choice) in enumerate(zip(left, right)):
        if _normalize_text(_strip_choice_label(left_choice, expected_index=index)) != _normalize_text(right_choice):
            return False
    return True


def build_fixed_order_prompt(question: str, choices: Sequence[Any]) -> str:
    """Build the ReasonAQA-style prompt without a ``Choices:`` prefix."""

    lines = [
        f"{chr(ord('a') + index)}) {str(choice).strip()}"
        for index, choice in enumerate(choices)
    ]
    question_text = str(question).strip()
    return f"{question_text} {' '.join(lines)}".strip()


def build_mellow_author_reply_prompt(question: str, choices: Sequence[Any]) -> str:
    """Reproduce the prompt construction posted by Mellow's authors in issue #5."""

    question_text = str(question)
    question_text = question_text[:-1] + "? "
    choices_text = " ".join(
        f"{chr(ord('a') + index)}) {str(choice)}"
        for index, choice in enumerate(choices)
    )
    return (question_text + choices_text).lower()


def mellow_author_reply_labeled_answer(answer: Any, choices: Sequence[Any]) -> str:
    """Attach the choice letter exactly as in the Mellow author's reply code."""

    answer_text = str(answer).lower()
    matches = [
        index
        for index, choice in enumerate(choices)
        if answer_text == str(choice).lower()
    ]
    if not matches:
        raise ValueError(
            "Mellow author-reply scorer requires the answer to match a choice: "
            f"answer={answer!r} choices={list(choices)!r} matches={matches}"
        )
    return f"{chr(ord('a') + matches[0])}) {answer_text}"


def mellow_author_reply_choice_is_correct(prediction: Any, labeled_answer: Any) -> bool:
    """Use the exact label-prefix comparison from the Mellow GitHub response."""

    return (
        str(prediction).split(")")[0].lower()
        == str(labeled_answer).split(")")[0].lower()
    )


def evaluate_mellow_author_reply_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score fixed-order predictions with the Mellow author's published helper."""

    task_metrics = {name: [0, 0] for name in ("sound", "music", "speech")}
    difficulty_metrics = {name: [0, 0] for name in ("easy", "hard", "medium")}
    correct = 0
    scored_rows: list[dict[str, Any]] = []
    for row_index, record in enumerate(predictions):
        task = str(record.get("task", ""))
        difficulty = str(record.get("difficulty", ""))
        if task not in task_metrics or difficulty not in difficulty_metrics:
            raise ValueError(
                f"invalid MMAU metadata at row {row_index}: task={task!r} difficulty={difficulty!r}"
            )
        choices = _coerce_choices(record.get("choices"))
        labeled_answer = mellow_author_reply_labeled_answer(record.get("answer", ""), choices)
        prediction = str(record.get("model_output", ""))
        matched = mellow_author_reply_choice_is_correct(prediction, labeled_answer)
        if matched:
            task_metrics[task][0] += 1
            difficulty_metrics[difficulty][0] += 1
            correct += 1
        task_metrics[task][1] += 1
        difficulty_metrics[difficulty][1] += 1
        scored_rows.append({
            "row_index": row_index,
            "id": record.get("id"),
            "prediction": prediction,
            "labeled_answer": labeled_answer,
            "correct": matched,
            "task": task,
            "difficulty": difficulty,
        })

    def summarize(metrics: Mapping[str, Sequence[int]]) -> dict[str, Any]:
        return {
            name: {
                "correct": int(values[0]),
                "total": int(values[1]),
                "accuracy_percent": (
                    float(values[0]) / float(values[1]) * 100.0 if values[1] else 0.0
                ),
            }
            for name, values in metrics.items()
        }

    total = len(predictions)
    return {
        "status": "PASS",
        "scorer": MELLOW_AUTHOR_REPLY_SCORER,
        "total": {
            "correct": correct,
            "total": total,
            "accuracy_percent": correct / total * 100.0 if total else 0.0,
        },
        "task": summarize(task_metrics),
        "difficulty": summarize(difficulty_metrics),
        "rows": scored_rows,
    }


def write_mellow_author_reply_evaluation(
    output_dir: Path,
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Persist the author-reply score separately from MMAU-v05.15.25 scoring."""

    score = evaluate_mellow_author_reply_predictions(predictions)
    _write_json(output_dir / "mellow_author_reply_evaluation.json", score)
    lines = ["Mellow author-reply choice-label evaluation", "", "Task-wise Accuracy:"]
    for name in ("sound", "music", "speech"):
        item = score["task"][name]
        lines.append(
            f"{name} : {item['accuracy_percent']:.2f}% over {item['total']} samples"
        )
    lines.extend(["", "Difficulty-wise Accuracy:"])
    for name in ("easy", "hard", "medium"):
        item = score["difficulty"][name]
        lines.append(
            f"{name} : {item['accuracy_percent']:.2f}% over {item['total']} samples"
        )
    total = score["total"]
    lines.extend([
        "",
        f"Total Accuracy: {total['accuracy_percent']:.2f}% over {total['total']} samples",
        "",
    ])
    (output_dir / "mellow_author_reply_evaluation.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return score


def prepare_model_output_for_official_scorer(value: Any) -> str:
    """Remove only a leading ReasonAQA a)-d) label before official scoring."""

    return re.sub(r"^\s*[a-d]\)\s*", "", str(value), count=1, flags=re.IGNORECASE)


def _unbox(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "as_py"):
        try:
            return value.as_py()
        except Exception:
            pass
    return value


def _jsonish(value: Any) -> Any:
    """Decode JSON encoded Arrow/HuggingFace struct fields when necessary.

    The downloaded MMAU parquet exposes ``other_attributes`` as a JSON string
    in common pyarrow versions.  Keeping this conversion local means that
    ordinary answer/choice strings are not accidentally parsed or rewritten.
    """

    value = _unbox(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = bytes(value).decode("utf-8")
        except Exception:
            return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in {"{", "["}:
            try:
                return json.loads(stripped)
            except Exception:
                try:
                    # Some pandas/HF exports stringify Python containers with
                    # single quotes.  literal_eval is deliberately limited to
                    # Python literals and never executes arbitrary code.
                    return ast.literal_eval(stripped)
                except Exception:
                    return value
    return value


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    value = _jsonish(value)
    return value if isinstance(value, Mapping) else None


def _with_sampling_rate(value: Any, sampling_rate: Any) -> Any:
    """Carry a parent audio sampling rate into a nested audio payload."""

    value = _jsonish(value)
    if value is None:
        return None
    if sampling_rate is None:
        return value
    mapping = _as_mapping(value)
    if mapping is None:
        # A raw array in ``context.audio`` has no place to carry the parent
        # context's sampling rate unless we wrap it.  Encoded bytes and paths
        # carry their own rate (or are decoded as files), so leave those as-is.
        if isinstance(value, (bytes, bytearray, memoryview, str, Path)):
            return value
        return {"array": value, "sampling_rate": _unbox(sampling_rate)}
    if mapping.get("sampling_rate") is not None or mapping.get("sample_rate") is not None:
        return value
    result = dict(mapping)
    result["sampling_rate"] = _unbox(sampling_rate)
    return result


def _string_or_empty(value: Any) -> str:
    value = _jsonish(value)
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return ""
    return str(value).strip()


def _coerce_choices(value: Any) -> list[str]:
    value = _jsonish(value)
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            parsed = _jsonish(stripped)
            if parsed is value or isinstance(parsed, str):
                # Keep malformed/container-looking text as one option; the
                # caller will reject a one-option choice list, while literal
                # option text such as ``[foo]`` is not silently erased.
                return [stripped] if stripped else []
            value = parsed
        else:
            return [stripped] if stripped else []
    if isinstance(value, Mapping):
        # A struct is not a valid option sequence, but accepting a common
        # ``choices: {A: ..., B: ...}`` shape makes the failure deterministic.
        value = list(value.values())
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        try:
            value = value.tolist()
        except Exception:
            pass
    if not isinstance(value, (list, tuple)):
        return []
    return [_string_or_empty(item) for item in value]


def load_official_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        for key in ("data", "records", "samples"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        else:
            # A few exports use an object keyed by sample ID.  Accept that
            # shape without treating arbitrary metadata mappings as records.
            keyed = []
            for key, value in payload.items():
                if isinstance(value, Mapping):
                    record = dict(value)
                    record.setdefault("id", key)
                    keyed.append(record)
            if not keyed:
                raise ValueError("official MMAU JSON must be a list of records")
            records = keyed
    else:
        raise ValueError("official MMAU JSON must contain a record list")
    result = [dict(record) for record in records if isinstance(record, Mapping)]
    if len(result) != len(records):
        raise ValueError("official MMAU JSON contains a non-object record")
    return result


def audit_mmau_v051525(
    metadata_path: Path,
    evaluation_script: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require the exact current official MMAU test-mini metadata/scorer."""

    metadata_sha256 = _sha256(metadata_path)
    canonical_sha256 = _canonical_sha256([dict(record) for record in records])
    evaluation_sha256 = _sha256(evaluation_script)
    task_counts = Counter(str(record.get("task", "")) for record in records)
    difficulty_counts = Counter(str(record.get("difficulty", "")) for record in records)
    ids = [_official_id(record) for record in records]
    failures: list[str] = []
    if metadata_sha256 != MMAU_METADATA_SHA256:
        failures.append(
            f"metadata byte SHA256 differs: expected={MMAU_METADATA_SHA256} actual={metadata_sha256}"
        )
    if canonical_sha256 != MMAU_METADATA_CANONICAL_SHA256:
        failures.append(
            "metadata canonical SHA256 differs: "
            f"expected={MMAU_METADATA_CANONICAL_SHA256} actual={canonical_sha256}"
        )
    if evaluation_sha256 != MMAU_EVALUATION_SHA256:
        failures.append(
            f"evaluation.py SHA256 differs: expected={MMAU_EVALUATION_SHA256} actual={evaluation_sha256}"
        )
    if len(records) != EXPECTED_FULL_ROWS or len(set(ids)) != EXPECTED_FULL_ROWS or any(not value for value in ids):
        failures.append(
            f"metadata ID coverage differs: rows={len(records)} unique_nonempty={len({value for value in ids if value})}"
        )
    expected_tasks = {"music": 334, "sound": 333, "speech": 333}
    expected_difficulties = {"easy": 224, "hard": 236, "medium": 540}
    if dict(sorted(task_counts.items())) != expected_tasks:
        failures.append(f"task distribution differs: {dict(sorted(task_counts.items()))}")
    if dict(sorted(difficulty_counts.items())) != expected_difficulties:
        failures.append(f"difficulty distribution differs: {dict(sorted(difficulty_counts.items()))}")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "version": MMAU_VERSION,
        "official_commit": MMAU_OFFICIAL_COMMIT,
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_sha256,
        "metadata_canonical_sha256": canonical_sha256,
        "evaluation_script": str(evaluation_script),
        "evaluation_sha256": evaluation_sha256,
        "rows": len(records),
        "unique_ids": len(set(ids)),
        "task_counts": dict(sorted(task_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "failures": failures,
    }
    return report


def _official_id(record: Mapping[str, Any]) -> str:
    sample_id = _string_or_empty(record.get("id"))
    if sample_id:
        return sample_id
    attrs = _as_mapping(record.get("other_attributes"))
    if attrs is not None:
        return _string_or_empty(attrs.get("id"))
    for key in ("sample_id", "uid", "uuid"):
        sample_id = _string_or_empty(record.get(key))
        if sample_id:
            return sample_id
    return ""


def build_metadata_index(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    ambiguous: set[str] = set()
    for record in records:
        sample_id = _official_id(record)
        if not sample_id:
            # A malformed official record cannot be joined to a parquet row;
            # leave it out so the affected row is skipped rather than making
            # a single bad record abort the whole test-mini stream.
            continue
        if sample_id in ambiguous:
            continue
        if sample_id in index:
            # Never choose arbitrarily between duplicate metadata records.
            # Remove the ID from the usable index so every corresponding
            # parquet row is deterministically skipped as ambiguous.
            index.pop(sample_id, None)
            ambiguous.add(sample_id)
            continue
        index[sample_id] = dict(record)
    return index


def extract_row_id(row: Mapping[str, Any]) -> tuple[str, str]:
    attrs = _as_mapping(row.get("other_attributes"))
    if attrs is not None:
        sample_id = _string_or_empty(attrs.get("id"))
        if sample_id:
            return sample_id, "other_attributes.id"
    sample_id = _string_or_empty(row.get("id"))
    if sample_id:
        return sample_id, "id"
    return "", "missing"


def extract_row_question(row: Mapping[str, Any]) -> str:
    for key in ("instruction", "question", "prompt", "input"):
        value = _string_or_empty(row.get(key))
        if value:
            return value
    return ""


def extract_record_question(record: Mapping[str, Any]) -> str:
    """Return the benchmark question across JSON/parquet naming variants."""

    for key in ("question", "instruction", "prompt", "input"):
        value = _string_or_empty(record.get(key))
        if value:
            return value
    return ""


def extract_row_choices(row: Mapping[str, Any]) -> list[str]:
    return _coerce_choices(row.get("choices"))


def extract_row_answer(row: Mapping[str, Any]) -> str:
    for key in ("answer", "target", "output"):
        value = _string_or_empty(row.get(key))
        if value:
            return value
    return ""


def extract_record_answer(record: Mapping[str, Any]) -> str:
    for key in ("answer", "target", "output"):
        value = _string_or_empty(record.get(key))
        if value:
            return value
    return ""


def extract_record_field(record: Mapping[str, Any], *keys: str) -> str:
    """Read a scalar metadata field from flat or nested JSON exports."""

    for key in keys:
        value = _string_or_empty(record.get(key))
        if value:
            return value
    attrs = _as_mapping(record.get("other_attributes"))
    if attrs is not None:
        for key in keys:
            value = _string_or_empty(attrs.get(key))
            if value:
                return value
    return ""


def _mapping_with_audio(value: Any) -> Any:
    value = _jsonish(value)
    if isinstance(value, Mapping):
        if "audio" in value:
            return _unbox(value["audio"])
        if any(key in value for key in ("bytes", "array", "path", "sampling_rate", "data")):
            return value
    return value


def extract_audio_payload(row: Mapping[str, Any]) -> tuple[Any, str]:
    # HF's MMAU conversion uses context.audio.  The additional shapes cover
    # pyarrow/HF versions that flatten the audio column or expose a path.
    context = _jsonish(row.get("context"))
    context_mapping = _as_mapping(context)
    if context_mapping is not None:
        context_rate = context_mapping.get("sampling_rate") or context_mapping.get("sample_rate")
        context_rate = context_rate or row.get("sampling_rate") or row.get("sample_rate")
        if "audio" in context_mapping:
            payload = _with_sampling_rate(
                context_mapping.get("audio"),
                context_rate,
            )
            if payload is not None:
                return payload, "context.audio"
        payload = _mapping_with_audio(context_mapping)
        if payload is not None:
            return payload, "context"
    elif context is not None:
        payload = _with_sampling_rate(
            _mapping_with_audio(context),
            row.get("sampling_rate") or row.get("sample_rate"),
        )
        if payload is not None:
            return payload, "context"
    for key in ("audio", "audio1", "audio_data", "audio_bytes"):
        if key in row and row.get(key) is not None:
            raw_payload = _mapping_with_audio(row[key])
            payload = _with_sampling_rate(
                raw_payload,
                row.get("sampling_rate") or row.get("sample_rate"),
            )
            return payload, key
    # ``audio_id`` is metadata, not an audio payload.  Do not feed an ID into
    # the decoder as though it were a path or waveform; benchmark rows are
    # expected to carry the embedded context.audio value.
    return None, "missing"


def iter_parquet_rows(path: Path, *, batch_size: int = 8, limit: int | None = None) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield rows in physical parquet order without materializing the table."""

    if batch_size <= 0:
        raise ValueError("parquet batch size must be positive")
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    parquet = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
        for row in batch.to_pylist():
            if limit is not None and row_index >= limit:
                return
            if not isinstance(row, dict):
                row = dict(row)
            yield row_index, row
            row_index += 1


def _resolve_audio_path(value: str, base_dir: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    candidates = (base_dir / candidate, Path.cwd() / candidate, base_dir.parent / candidate)
    for item in candidates:
        if item.is_file():
            return item
    return candidates[0]


def _decode_wave_bytes(raw: bytes) -> tuple[Any, int, str]:
    """Decode common WAV/audio bytes into a torch tensor and source rate."""

    import torch

    soundfile_error: Exception | None = None
    try:
        import numpy as np
        import soundfile as sf

        array, source_rate = sf.read(io.BytesIO(raw), always_2d=True, dtype="float32")
        return torch.from_numpy(np.asarray(array, dtype="float32").T), int(source_rate), "bytes:soundfile"
    except Exception as exc:
        soundfile_error = exc
    try:
        import torchaudio

        waveform, source_rate = torchaudio.load(io.BytesIO(raw))
        return waveform.float(), int(source_rate), "bytes:torchaudio"
    except Exception:
        pass
    try:
        with wave.open(io.BytesIO(raw), "rb") as handle:
            source_rate = int(handle.getframerate())
            channels = int(handle.getnchannels())
            width = int(handle.getsampwidth())
            frames = int(handle.getnframes())
            payload = handle.readframes(frames)
        if width != 2:
            raise RuntimeError(f"wave fallback supports 16-bit PCM only, got sample width {width}")
        import numpy as np

        array = np.frombuffer(payload, dtype=np.int16).reshape(-1, channels).astype("float32") / 32768.0
        return torch.from_numpy(array.T), source_rate, "bytes:wave"
    except Exception as wave_error:
        raise RuntimeError(f"unable to decode embedded audio bytes: soundfile={soundfile_error!r}; wave={wave_error!r}") from wave_error


def _array_to_waveform(value: Any, *, default_rate: int) -> tuple[Any, int, str]:
    import torch

    value = _jsonish(value)
    if isinstance(value, Mapping):
        rate_value = _unbox(value.get("sampling_rate") or value.get("sample_rate") or default_rate)
        try:
            source_rate = int(rate_value)
        except Exception as exc:
            raise RuntimeError(f"invalid audio sampling rate: {rate_value!r}") from exc
        if value.get("array") is not None:
            value = _jsonish(value["array"])
        elif value.get("data") is not None:
            value = _jsonish(value["data"])
        else:
            raise RuntimeError("audio mapping has neither array nor data")
    else:
        source_rate = default_rate
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _decode_wave_bytes(bytes(value))
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        try:
            value = value.tolist()
        except Exception:
            pass
    try:
        waveform = torch.as_tensor(value)
    except Exception as exc:
        raise RuntimeError(f"audio array cannot be converted to tensor: {type(value).__name__}") from exc
    if waveform.ndim not in (1, 2):
        raise RuntimeError(f"audio array must be 1D or 2D, got rank {waveform.ndim}")
    if waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1] and waveform.shape[1] <= 8:
        waveform = waveform.transpose(0, 1)
    if not waveform.is_floating_point():
        info = torch.iinfo(waveform.dtype)
        scale = float(max(abs(info.min), abs(info.max))) or 1.0
        waveform = waveform.float() / scale
    else:
        waveform = waveform.float()
    return waveform, source_rate, "array"


def _decode_audio_payload(payload: Any, *, base_dir: Path, default_rate: int) -> tuple[Any, int, str]:
    import torch

    payload = _jsonish(payload)
    if isinstance(payload, Mapping):
        rate_value = _unbox(payload.get("sampling_rate") or payload.get("sample_rate") or default_rate)
        try:
            sampling_rate = int(rate_value)
        except Exception as exc:
            raise RuntimeError(f"invalid audio sampling rate: {rate_value!r}") from exc
        if payload.get("bytes") is not None:
            raw = _unbox(payload["bytes"])
            if not isinstance(raw, (bytes, bytearray, memoryview)):
                raise RuntimeError(f"audio bytes field is not bytes: {type(raw).__name__}")
            waveform, source_rate, source = _decode_wave_bytes(bytes(raw))
            return waveform, source_rate or sampling_rate, source
        if payload.get("array") is not None or payload.get("data") is not None:
            return _array_to_waveform(payload, default_rate=sampling_rate)
        if payload.get("path"):
            payload = _unbox(payload["path"])
        else:
            raise RuntimeError("audio mapping has no bytes, array, data, or path")
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return _decode_wave_bytes(bytes(payload))
    if isinstance(payload, str) or isinstance(payload, Path):
        path = _resolve_audio_path(str(payload), base_dir)
        if not path.is_file():
            raise FileNotFoundError(f"audio path does not exist: {path}")
        try:
            import soundfile as sf
            import numpy as np

            array, source_rate = sf.read(str(path), always_2d=True, dtype="float32")
            return torch.from_numpy(np.asarray(array, dtype="float32").T), int(source_rate), f"path:soundfile:{path}"
        except Exception as soundfile_error:
            try:
                import torchaudio

                waveform, source_rate = torchaudio.load(str(path))
                return waveform.float(), int(source_rate), f"path:torchaudio:{path}"
            except Exception as torchaudio_error:
                raise RuntimeError(f"unable to decode audio path {path}: soundfile={soundfile_error!r}; torchaudio={torchaudio_error!r}") from torchaudio_error
    return _array_to_waveform(payload, default_rate=default_rate)


def _resample_waveform(waveform: Any, source_rate: int, target_rate: int) -> Any:
    import torch.nn.functional as F

    if source_rate == target_rate:
        return waveform
    try:
        import torchaudio

        return torchaudio.functional.resample(waveform, source_rate, target_rate)
    except Exception:
        target_samples = max(1, round(int(waveform.shape[-1]) * target_rate / source_rate))
        return F.interpolate(waveform.unsqueeze(0), size=target_samples, mode="linear", align_corners=False).squeeze(0)


def decode_and_normalize_audio(
    payload: Any,
    *,
    base_dir: Path,
    source_rate: int = DEFAULT_SOURCE_SAMPLE_RATE,
    target_rate: int = DEFAULT_SAMPLE_RATE,
    seconds: int = DEFAULT_AUDIO_SECONDS,
) -> tuple[Any, dict[str, Any]]:
    """Decode, mono-convert, resample, crop/pad exactly to the train contract."""

    import torch
    import torch.nn.functional as F

    waveform, actual_rate, source = _decode_audio_payload(payload, base_dir=base_dir, default_rate=source_rate)
    if actual_rate <= 0:
        raise RuntimeError(f"invalid source sampling rate: {actual_rate}")
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise RuntimeError(f"decoded waveform must be [channels,samples], got {tuple(waveform.shape)}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.float()
    if not bool(torch.isfinite(waveform).all()):
        raise RuntimeError("decoded waveform contains non-finite values")
    original_samples = int(waveform.shape[-1])
    original_duration = original_samples / float(actual_rate)
    was_resampled = actual_rate != target_rate
    waveform = _resample_waveform(waveform, actual_rate, target_rate)
    target_samples = int(target_rate * seconds)
    resampled_samples = int(waveform.shape[-1])
    was_cropped = resampled_samples > target_samples
    was_padded = resampled_samples < target_samples
    if was_cropped:
        waveform = waveform[..., :target_samples]
    elif was_padded:
        waveform = F.pad(waveform, (0, target_samples - resampled_samples))
    if tuple(waveform.shape) != (1, target_samples):
        raise RuntimeError(f"normalized waveform shape mismatch: {tuple(waveform.shape)}")
    return waveform, {
        "audio_source": source,
        "audio_original_sample_rate": int(actual_rate),
        "audio_original_num_samples": original_samples,
        "audio_original_duration_seconds": original_duration,
        "audio_resampled": bool(was_resampled),
        "audio_resampled_num_samples": resampled_samples,
        "audio_was_cropped": bool(was_cropped),
        "audio_was_padded": bool(was_padded),
        "audio_target_sample_rate": int(target_rate),
        "audio_target_num_samples": target_samples,
        "audio_target_duration_seconds": float(seconds),
        "audio_mono": True,
    }


def decode_mellow_author_reply_audio(
    payload: Any,
    *,
    base_dir: Path,
    source_rate: int = DEFAULT_SOURCE_SAMPLE_RATE,
    target_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[Any, dict[str, Any]]:
    """Decode/resample before MellowWrapper-style repeat or random cropping."""

    import torch

    waveform, actual_rate, source = _decode_audio_payload(
        payload,
        base_dir=base_dir,
        default_rate=source_rate,
    )
    if actual_rate <= 0:
        raise RuntimeError(f"invalid source sampling rate: {actual_rate}")
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise RuntimeError(f"decoded waveform must be [channels,samples], got {tuple(waveform.shape)}")
    waveform = waveform.float()
    if not bool(torch.isfinite(waveform).all()):
        raise RuntimeError("decoded waveform contains non-finite values")
    original_channels = int(waveform.shape[0])
    original_samples_per_channel = int(waveform.shape[-1])
    original_duration = original_samples_per_channel / float(actual_rate)
    waveform = _resample_waveform(waveform, actual_rate, target_rate)
    # MellowWrapper calls reshape(-1), rather than averaging channels.
    waveform = waveform.reshape(1, -1).contiguous()
    if waveform.shape[-1] <= 0:
        raise RuntimeError("decoded waveform is empty")
    return waveform, {
        "audio_source": source,
        "audio_original_sample_rate": int(actual_rate),
        "audio_original_num_samples": original_samples_per_channel,
        "audio_original_channels": original_channels,
        "audio_original_duration_seconds": original_duration,
        "audio_resampled": bool(actual_rate != target_rate),
        "audio_resampled_num_samples": int(waveform.shape[-1]),
        "audio_target_sample_rate": int(target_rate),
        "audio_target_duration_seconds": float(DEFAULT_AUDIO_SECONDS),
        "audio_channel_policy": "mellow_wrapper_reshape_channels",
        "audio_segment_policy": MELLOW_AUTHOR_REPLY_AUDIO_FORMAT,
    }


def mellow_author_reply_audio_segment(
    waveform: Any,
    *,
    target_rate: int = DEFAULT_SAMPLE_RATE,
    seconds: int = DEFAULT_AUDIO_SECONDS,
    rng: Any = random,
) -> tuple[Any, dict[str, Any]]:
    """Apply MellowWrapper's exact repeat-or-random-crop length policy."""

    import torch

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise RuntimeError(f"Mellow audio source must be [1,samples], got {tuple(waveform.shape)}")
    source_samples = int(waveform.shape[-1])
    if source_samples <= 0:
        raise RuntimeError("Mellow audio source is empty")
    target_samples = int(target_rate * seconds)
    repeat_factor = 1
    crop_start = 0
    if target_samples >= source_samples:
        repeat_factor = int((target_samples + source_samples - 1) // source_samples)
        segment = waveform.repeat(1, repeat_factor)[..., :target_samples]
        policy = "repeat_then_trim"
    else:
        crop_start = int(rng.randrange(source_samples - target_samples))
        segment = waveform[..., crop_start:crop_start + target_samples]
        policy = "random_crop"
    if tuple(segment.shape) != (1, target_samples):
        raise RuntimeError(f"Mellow audio segment shape mismatch: {tuple(segment.shape)}")
    if not bool(torch.isfinite(segment).all()):
        raise RuntimeError("Mellow audio segment contains non-finite values")
    return segment.contiguous(), {
        "source_samples": source_samples,
        "target_samples": target_samples,
        "policy": policy,
        "repeat_factor": repeat_factor,
        "crop_start": crop_start,
        "crop_end": crop_start + target_samples,
    }


def _prepare_metadata_row(
    row: Mapping[str, Any],
    *,
    metadata_index: Mapping[str, Mapping[str, Any]],
    dataset_dir: Path,
    prompt_builder: Any = build_fixed_order_prompt,
    audio_decoder: Any = decode_and_normalize_audio,
    audio_root: Path | None = None,
    prefer_official_audio_file: bool = False,
) -> dict[str, Any]:
    sample_id, id_source = extract_row_id(row)
    if not sample_id:
        raise RowSkip("data", "missing_id", "parquet row has no other_attributes.id or id")
    official = metadata_index.get(sample_id)
    if official is None:
        raise RowSkip("data", "id_not_found_in_official_json", f"parquet id is absent from official JSON: {sample_id}", id_source=id_source)
    row_question = extract_row_question(row)
    question = extract_record_question(official) or row_question
    choices = _coerce_choices(official.get("choices"))
    answer = extract_record_answer(official)
    task = extract_record_field(official, "task")
    difficulty = extract_record_field(official, "difficulty")
    if not question:
        raise RowSkip("data", "missing_question", "official JSON record has no question")
    if len(choices) < 2 or any(not choice for choice in choices):
        raise RowSkip("data", "invalid_choices", "official JSON record has invalid choices")
    if not answer:
        raise RowSkip("data", "missing_answer", "official JSON record has no answer")
    if not task:
        raise RowSkip("data", "missing_task", "official JSON record has no task")
    if not difficulty:
        raise RowSkip("data", "missing_difficulty", "official JSON record has no difficulty")
    if task not in {"sound", "music", "speech"}:
        raise RowSkip("data", "unsupported_task", f"official JSON task is not supported by evaluation.py: {task!r}")
    if difficulty not in {"easy", "medium", "hard"}:
        raise RowSkip("data", "unsupported_difficulty", f"official JSON difficulty is not supported by evaluation.py: {difficulty!r}")

    if not row_question:
        raise RowSkip("data", "missing_question", "parquet row has no instruction/question field")
    row_choices = extract_row_choices(row)
    if len(row_choices) < 2 or any(not choice for choice in row_choices):
        raise RowSkip("data", "missing_choices", "parquet row has no valid choices field")
    if not choices_match_fixed_order(row_choices, choices):
        raise RowSkip(
            "data",
            "choices_order_mismatch",
            "parquet choices do not match official JSON fixed order",
            parquet_choices=row_choices,
            official_choices=choices,
        )
    row_answer = extract_row_answer(row)
    if not row_answer:
        raise RowSkip("data", "missing_answer", "parquet row has no answer field")
    effective_base_dir = dataset_dir
    if prefer_official_audio_file and audio_root is not None:
        # Match the Mellow issue #5 code exactly: ``data[i]["id"] + ".wav"``.
        # Do not silently substitute another metadata field for the filename.
        filename = f"{sample_id}.wav"
        official_audio_path = audio_root / filename
        if official_audio_path.is_file():
            payload = official_audio_path
            payload_source = "official_id_wav"
            effective_base_dir = audio_root
        else:
            payload, payload_source = extract_audio_payload(row)
            payload_source = f"{payload_source}:fallback_missing_official_wav"
    else:
        payload, payload_source = extract_audio_payload(row)
    if payload is None:
        raise RowSkip("audio", "audio_missing", "MMAU row has neither official WAV nor embedded audio")
    try:
        waveform, audio_info = audio_decoder(
            payload,
            base_dir=effective_base_dir,
        )
    except Exception as exc:
        raise RowSkip("audio", "audio_decode_failed", str(exc), audio_payload_source=payload_source) from exc

    prompt = prompt_builder(question, choices)
    return {
        "id": sample_id,
        "id_source": id_source,
        "question": question,
        "choices": choices,
        "answer": answer,
        "parquet_question": row_question,
        "parquet_answer": row_answer,
        "task": task,
        "difficulty": difficulty,
        "official_record": copy.deepcopy(dict(official)),
        "prompt": prompt,
        "waveform": waveform,
        "audio_payload_source": payload_source,
        "audio2_reused": True,
        **audio_info,
    }


def _token_rows(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(token) for token in value]


def _tokenize_without_truncation(tokenizer: Any, prompt: str, *, max_prompt_tokens: int) -> tuple[Any, int]:
    encoded = tokenizer(
        prompt,
        truncation=False,
        padding=False,
        add_special_tokens=True,
        return_tensors="pt",
    )
    token_rows = _token_rows(encoded["input_ids"])
    token_count = len(token_rows)
    if token_count <= 0:
        raise RowSkip("prompt", "empty_prompt_tokens", "tokenizer returned an empty prompt")
    if token_count > max_prompt_tokens:
        raise RowSkip(
            "prompt",
            "prompt_exceeds_max_tokens",
            f"prompt has {token_count} tokens; contract limit is {max_prompt_tokens}",
            prompt_token_count=token_count,
            max_prompt_tokens=max_prompt_tokens,
        )
    import torch

    return torch.tensor([token_rows], dtype=torch.long), token_count


def tokenize_mellow_author_reply_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    max_prompt_tokens: int,
) -> tuple[Any, int, bool]:
    """Mirror MellowWrapper ``encode_plus`` truncation while retaining an audit count."""

    encoded = tokenizer(
        prompt,
        truncation=True,
        padding=False,
        max_length=max_prompt_tokens,
        add_special_tokens=True,
        return_tensors="pt",
    )
    untruncated = tokenizer(
        prompt,
        truncation=False,
        padding=False,
        add_special_tokens=True,
        return_tensors="pt",
    )
    token_rows = _token_rows(encoded["input_ids"])
    original_rows = _token_rows(untruncated["input_ids"])
    if not token_rows:
        raise RowSkip("prompt", "empty_prompt_tokens", "tokenizer returned an empty prompt")
    import torch

    return (
        torch.tensor([token_rows], dtype=torch.long),
        len(original_rows),
        len(original_rows) > len(token_rows),
    )


def _load_runtime_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Audio MeSH benchmark inference requires one CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    from generate_audio_checkpoint_reasonaqa import _validate_checkpoint_contract
    from train_audio_5_10x2_5_mesh_mellow_ddp import _load_model

    checkpoint_audit = _validate_checkpoint_contract(args)
    config = dict(checkpoint_audit["audio_mesh_config"])
    saved_htsat = str(config.get("htsat_checkpoint", ""))
    saved_mellow = str(config.get("mellow_root", ""))
    if not saved_htsat:
        raise RuntimeError("checkpoint does not record its external HTSAT checkpoint")
    if not saved_mellow:
        raise RuntimeError("checkpoint does not record its external Mellow root")
    if Path(saved_htsat).resolve() != args.htsat_checkpoint.resolve():
        raise RuntimeError(f"checkpoint HTSAT path mismatch: saved={saved_htsat} requested={args.htsat_checkpoint}")
    if Path(saved_mellow).resolve() != args.mellow_root.resolve():
        raise RuntimeError(f"checkpoint Mellow root mismatch: saved={saved_mellow} requested={args.mellow_root}")

    load_args = argparse.Namespace(
        resume_from=args.checkpoint,
        model_path=args.checkpoint / "mesh_model",
        tokenizer_path=None,
        htsat_checkpoint=args.htsat_checkpoint,
        mellow_root=args.mellow_root,
        compact_single_audio_prefix=bool(
            checkpoint_audit["compact_single_audio_prefix"]
        ),
    )
    model, tokenizer = _load_model(load_args, device)
    saved_provenance = config.get("mellow_provenance") or {}
    loaded_provenance = getattr(model, "_audio_provenance", {})
    for key in ("module", "mellow_htsat_source", "mellow_htsat_sha256"):
        if saved_provenance.get(key) != loaded_provenance.get(key):
            raise RuntimeError(
                f"Mellow provenance mismatch for {key}: "
                f"saved={saved_provenance.get(key)!r} loaded={loaded_provenance.get(key)!r}"
            )
    model.eval()
    if not bool(model.config_audio.compact_single_audio_prefix):
        raise RuntimeError("Audio MeSH benchmark evaluation requires compact single-audio prefix")
    actual_context_length = int(getattr(model.config_audio, "max_context_length", 0))
    if actual_context_length != DEFAULT_MAX_CONTEXT_LENGTH:
        raise RuntimeError(
            "inference context contract mismatch: "
            f"expected={DEFAULT_MAX_CONTEXT_LENGTH} actual={actual_context_length}"
        )
    owner = model.mesh_model.model
    owner.audit_mode = False
    owner.gradient_audit_mode = False
    owner.routing_stats_mode = False
    modes = {
        "composite": bool(model.training),
        "mesh": bool(model.mesh_model.training),
        "bridge": bool(model.bridge.training),
        "wrapper": bool(model.htsat_wrapper.training),
        "htsat": bool(model.htsat_backbone.training),
        "c2l": bool(model.htsat_wrapper.c2l.training),
    }
    if any(modes.values()):
        raise RuntimeError(f"inference requires all modules in eval mode: {modes}")
    config = dict(config)
    config["checkpoint_artifact_audit"] = checkpoint_audit
    config["runtime_max_context_length"] = actual_context_length
    return model, tokenizer, device, config


def _run_model_generation(
    model: Any,
    tokenizer: Any,
    device: Any,
    sample: Mapping[str, Any],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    # These helpers are the established ReasonAQA implementation: they build
    # the same audio bridge prefix and perform MeSH-safe full recomputation,
    # including a trace check on every generation step.  The reused decoder
    # explicitly uses greedy decoding with do_sample=False, use_cache=False,
    # and logits_to_keep=1.
    from generate_audio_checkpoint_reasonaqa import _build_audio_prefix, _greedy_decode

    prompt_ids_cpu, prompt_token_count = _tokenize_without_truncation(
        tokenizer, str(sample["prompt"]), max_prompt_tokens=max_prompt_tokens
    )
    prompt_ids = prompt_ids_cpu.to(device)
    item = {
        "audio1": sample["waveform"],
        "audio2": None,
        "audio2_reused": True,
        "single_audio_slot": True,
    }
    with __import__("torch").inference_mode():
        audio_prefix, prefix_audit = _build_audio_prefix(
            model,
            item,
            device,
            autocast_enabled=True,
        )
        generated = _greedy_decode(
            model,
            tokenizer,
            audio_prefix,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            autocast_enabled=True,
        )
    generated["prompt_token_count"] = prompt_token_count
    generated.update(prefix_audit)
    return generated


def materialize_predictions(
    state: Iterable[Mapping[str, Any]],
    official_records: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Materialize predictions without ever shrinking the official denominator."""

    records: dict[str, Mapping[str, Any]] = {}
    for item in state:
        status = str(item.get("status", ""))
        if status != "generated":
            continue
        key = str(item.get("row_key") or row_key(int(item["row_index"]), item.get("id")))
        records[key] = item
    ordered = sorted(records.values(), key=lambda item: (int(item["row_index"]), str(item.get("id", ""))))
    if official_records is not None:
        by_id: dict[str, Mapping[str, Any]] = {}
        for item in ordered:
            sample_id = str(item.get("id", "")).strip()
            if not sample_id:
                continue
            if sample_id in by_id:
                raise RuntimeError(f"multiple terminal generations for official MMAU ID: {sample_id}")
            by_id[sample_id] = item
        predictions: list[dict[str, Any]] = []
        for official in official_records:
            sample_id = _official_id(official)
            item = by_id.get(sample_id)
            prediction = copy.deepcopy(dict(official))
            prediction["model_output"] = "" if item is None else str(item.get("model_output", ""))
            predictions.append(prediction)
        return predictions

    predictions: list[dict[str, Any]] = []
    for item in ordered:
        official = item.get("official_record")
        if not isinstance(official, Mapping):
            continue
        prediction = copy.deepcopy(dict(official))
        # Backward-compatible dependency-light path used by unit tests.
        prediction["model_output"] = str(item.get("model_output", ""))
        predictions.append(prediction)
    return predictions


def _materialize_from_store(
    store: ProgressStore,
    output_dir: Path,
    official_records: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    predictions = materialize_predictions(store.state.values(), official_records)
    _write_json(output_dir / "predictions_fixed_order.json", predictions)
    first_five = sorted(
        (record for record in store.state.values() if int(record.get("row_index", -1)) < SMOKE_ROWS),
        key=lambda record: int(record.get("row_index", -1)),
    )
    with (output_dir / "smoke_first5.jsonl").open("w", encoding="utf-8") as handle:
        for record in first_five:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    return predictions


def _run_official_evaluation(args: argparse.Namespace, output_dir: Path, prediction_count: int) -> dict[str, Any]:
    path = output_dir / "official_evaluation.txt"
    result: dict[str, Any] = {
        "requested": bool(args.run_official_evaluation),
        "prediction_count": int(prediction_count),
        "path": str(path),
    }
    if not args.run_official_evaluation:
        path.write_text("Official evaluation was not requested for this run.\n", encoding="utf-8")
        result["status"] = "NOT_REQUESTED"
        return result
    if not args.evaluation_script.is_file():
        message = f"official evaluation.py not found: {args.evaluation_script}"
        path.write_text(message + "\n", encoding="utf-8")
        result.update({"status": "FAILED", "returncode": None, "error": message})
        return result
    command = [sys.executable, str(args.evaluation_script), "--input", str(output_dir / "predictions_fixed_order.json")]
    try:
        completed = subprocess.run(
            command,
            cwd=str(args.evaluation_script.parent),
            capture_output=True,
            text=True,
            check=False,
        )
        content = (
            f"$ {' '.join(command)}\n"
            f"returncode: {completed.returncode}\n\n"
            "===== STDOUT =====\n"
            f"{completed.stdout}\n"
            "===== STDERR =====\n"
            f"{completed.stderr}\n"
        )
        path.write_text(content, encoding="utf-8")
        totals = [int(value) for value in re.findall(r"\bover\s+(\d+)\s+samples\b", completed.stdout)]
        reported_total = totals[-1] if totals else None
        accuracy_matches = re.findall(
            r"Total Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%\s+over\s+(\d+)\s+samples",
            completed.stdout,
        )
        total_accuracy_percent = float(accuracy_matches[-1][0]) if accuracy_matches else None
        passed = completed.returncode == 0 and reported_total == int(prediction_count)
        result.update({
            "status": "PASS" if passed else "FAILED",
            "returncode": completed.returncode,
            "reported_total": reported_total,
            "total_accuracy_percent": total_accuracy_percent,
        })
        if completed.returncode == 0 and not passed:
            result["error"] = (
                "official scorer did not report the requested denominator: "
                f"reported={reported_total} expected={prediction_count}"
            )
    except Exception as exc:
        path.write_text(f"official evaluation invocation failed: {exc!r}\n", encoding="utf-8")
        result.update({"status": "FAILED", "returncode": None, "error": repr(exc)})
    return result


def _block_official_evaluation(
    args: argparse.Namespace,
    output_dir: Path,
    prediction_count: int,
    error: BaseException,
) -> dict[str, Any]:
    """Overwrite any stale smoke score when a later run cannot be scored."""

    path = output_dir / "official_evaluation.txt"
    message = (
        "Official evaluation was blocked by pipeline failure.\n"
        f"mode: {args.mode}\n"
        f"prediction_count: {int(prediction_count)}\n"
        f"error: {error!r}\n"
    )
    path.write_text(message, encoding="utf-8")
    return {
        "requested": bool(args.run_official_evaluation),
        "status": "BLOCKED_BY_PIPELINE_FAILURE",
        "prediction_count": int(prediction_count),
        "path": str(path),
        "error": repr(error),
    }


def overall_evaluation_status(
    inference_coverage_status: str,
    official_evaluation_status: str,
) -> str:
    """Keep inference coverage distinct from completion of official scoring."""

    if inference_coverage_status != "PASS":
        return "FAILED"
    if official_evaluation_status == "PASS":
        return "PASS"
    if official_evaluation_status == "NOT_REQUESTED":
        return INFERENCE_ONLY_STATUS
    return "FAILED"


def _counts(store: ProgressStore) -> dict[str, Any]:
    statuses = Counter(str(record.get("status", "unknown")) for record in store.state.values())
    reasons = Counter(
        str(record.get("reason", "unknown"))
        for record in store.state.values()
        if record.get("status") == "skipped"
    )
    generated = [
        record
        for record in store.state.values()
        if record.get("status") == "generated"
    ]
    durations: list[float] = []
    for record in generated:
        value = record.get("audio_original_duration_seconds")
        if value is None:
            continue
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration >= 0:
            durations.append(duration)
    cropped_ids = [
        str(record.get("id"))
        for record in generated
        if bool(record.get("audio_was_cropped"))
    ]
    audio = {
        "decoded_generation_records": len(durations),
        "over_ten_seconds": sum(duration > DEFAULT_AUDIO_SECONDS for duration in durations),
        "cropped_records": len(cropped_ids),
        "cropped_ids": cropped_ids,
        "padded_records": sum(bool(record.get("audio_was_padded")) for record in generated),
        "payload_sources": dict(sorted(Counter(
            str(record.get("audio_payload_source", "unknown"))
            for record in generated
        ).items())),
    }
    if durations:
        audio.update({"minimum_duration_seconds": min(durations), "maximum_duration_seconds": max(durations)})
    return {
        "terminal_records": int(sum(statuses.values())),
        "generation_completed": int(statuses.get("generated", 0)),
        "generated": int(statuses.get("generated", 0)),
        "skipped": int(statuses.get("skipped", 0)),
        "skip_reasons": dict(sorted(reasons.items())),
        "audio": audio,
    }


def _ensure_output_dir(
    args: argparse.Namespace,
    *,
    prediction_format: str = PREDICTION_FORMAT,
    prompt_format: str = PROMPT_FORMAT,
    protocol: str | None = None,
) -> None:
    output_dir = args.output_dir
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    immutable = {
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "parquet": str(args.parquet),
        "metadata_json": str(args.metadata_json),
        "evaluation_script": str(args.evaluation_script),
        "htsat_checkpoint": str(args.htsat_checkpoint),
        "mellow_root": str(args.mellow_root),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "prompt_format": prompt_format,
        "prediction_format": prediction_format,
        "protocol": protocol or "fixed-order; ReasonAQA lowercase labels; parquet physical order; single cuda:0; bf16; no permutation vote",
    }
    if hasattr(args, "audio_root"):
        immutable["audio_root"] = str(args.audio_root)
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        existing_mode = str(existing.get("mode", ""))
        promote_smoke_to_full = existing_mode == "smoke" and args.mode == "full"
        mismatches = {
            key: {"existing": existing.get(key), "requested": value}
            for key, value in immutable.items()
            if existing.get(key) != value and not (key == "mode" and promote_smoke_to_full)
        }
        if mismatches:
            raise RuntimeError(f"existing output directory belongs to another run: {mismatches}")
        if promote_smoke_to_full:
            # A smoke run intentionally owns the same append-only progress
            # streams that the subsequent full run can continue.  Promote the
            # run contract in-place so rows 0..4 are resumed rather than
            # re-inferred, while retaining an audit trail of the transition.
            history = existing.get("mode_history")
            if not isinstance(history, list):
                history = [existing_mode]
            if not history or history[-1] != "full":
                history = [*history, "full"]
            existing["mode"] = "full"
            existing["mode_history"] = history
            _write_json(config_path, existing)
        if not promote_smoke_to_full and existing_mode not in {"", args.mode}:
            # The reverse transition would silently truncate a full run and is
            # therefore rejected.
            raise RuntimeError(
                f"existing output directory mode cannot change from {existing_mode!r} to {args.mode!r}"
            )
    else:
        # A non-empty unowned directory is never silently overwritten.  An
        # empty directory, or one containing only a harmless system entry, is
        # safe to claim for this run.
        entries = [item for item in output_dir.iterdir() if item.name not in {"." , ".."}]
        if entries:
            raise FileExistsError(
                f"refusing to use non-empty output directory without run_config.json: {output_dir}"
            )
        _write_json(config_path, {**immutable, "mode": args.mode})


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_checkpoint: str | Path = DEFAULT_CHECKPOINT,
    default_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    add_audio_root: bool = False,
    description: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--checkpoint", type=Path, default=Path(default_checkpoint))
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR))
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--evaluation-script", type=Path)
    if add_audio_root:
        parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--htsat-checkpoint", type=Path, default=Path(DEFAULT_HTSAT))
    parser.add_argument("--mellow-root", type=Path, default=Path(DEFAULT_MELLOW))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parquet-batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=default_max_new_tokens)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--run-official-evaluation", action="store_true")
    args = parser.parse_args(argv)
    args.parquet = args.parquet or args.dataset_dir / "test_mini.parquet"
    args.metadata_json = args.metadata_json or args.dataset_dir / "mmau-test-mini.json"
    args.evaluation_script = args.evaluation_script or args.dataset_dir / "evaluation.py"
    if add_audio_root:
        args.audio_root = args.audio_root or args.dataset_dir / "test-mini-audios"
    if args.max_prompt_tokens != DEFAULT_MAX_PROMPT_TOKENS:
        parser.error(f"--max-prompt-tokens is fixed at {DEFAULT_MAX_PROMPT_TOKENS} for this checkpoint")
    if args.max_new_tokens != default_max_new_tokens:
        parser.error(f"--max-new-tokens is fixed at {default_max_new_tokens}")
    if args.parquet_batch_size <= 0:
        parser.error("--parquet-batch-size must be positive")
    return args


def run(
    args: argparse.Namespace,
    *,
    load_runtime_model: Any | None = None,
    run_model_generation: Any | None = None,
    prepare_prediction: Any | None = None,
    prediction_format: str = PREDICTION_FORMAT,
    prompt_builder: Any = build_fixed_order_prompt,
    audio_decoder: Any = decode_and_normalize_audio,
    audio_root: Path | None = None,
    prefer_official_audio_file: bool = False,
    prompt_format: str = PROMPT_FORMAT,
    audio_format: str = "mono_32khz_first10s_right_zero_pad",
    protocol_contract: str | None = None,
    generation_protocol: Mapping[str, Any] | None = None,
    audio_prefix_tokens: int = DEFAULT_AUDIO_PREFIX_TOKENS,
    stage: str = "mmau_test_mini_audio_mesh_fixed_order",
    logical_trace: str = "5+10+10+5 per generation step",
) -> dict[str, Any]:
    load_runtime_model = load_runtime_model or _load_runtime_model
    run_model_generation = run_model_generation or _run_model_generation
    prepare_prediction = prepare_prediction or prepare_model_output_for_official_scorer
    started = time.time()
    _ensure_output_dir(
        args,
        prediction_format=prediction_format,
        prompt_format=prompt_format,
        protocol=protocol_contract,
    )
    generation_contract = {
        "max_new_tokens": int(args.max_new_tokens),
        "greedy": True,
        "do_sample": False,
        "use_cache": False,
    }
    if generation_protocol:
        generation_contract.update(dict(generation_protocol))
    report: dict[str, Any] = {
        "stage": stage,
        "status": "FAILED",
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "parquet": str(args.parquet),
        "metadata_json": str(args.metadata_json),
        "evaluation_script": str(args.evaluation_script),
        "device": "cuda:0",
        "dtype": "float32" if args.dtype == "fp32" else "bfloat16",
        "protocol": {
            "parquet_order": "physical row order via pyarrow.ParquetFile.iter_batches",
            "shuffle": False,
            "mode_limit": SMOKE_ROWS if args.mode == "smoke" else None,
            "choice_order": "official JSON fixed order",
            "prompt_format": prompt_format,
            "prediction_format": prediction_format,
            "permutation_majority_vote": False,
            "audio_sample_rate": DEFAULT_SAMPLE_RATE,
            "audio_seconds": DEFAULT_AUDIO_SECONDS,
            "audio_prefix_tokens": int(audio_prefix_tokens),
            "max_prompt_tokens": DEFAULT_MAX_PROMPT_TOKENS,
            "max_new_tokens": int(args.max_new_tokens),
            "audio_preprocessing": audio_format,
            "official_audio_files": bool(prefer_official_audio_file),
            "audio_root": str(audio_root) if audio_root is not None else None,
            **generation_contract,
            "logical_trace": logical_trace,
        },
        "records": {},
        "inference_coverage": {"status": "NOT_COMPLETED"},
        "resumption": {},
        "official_evaluation": {},
        "comparable_official_score": False,
        "official_artifact_audit": {},
        "warnings": [],
        "fatal_error": None,
    }
    with ProgressStore(args.output_dir) as store:
        report["resumption"]["recovered_jsonl_errors"] = store.load_errors
        official_records: list[dict[str, Any]] = []
        try:
            if not args.checkpoint.is_dir():
                raise FileNotFoundError(f"checkpoint directory not found: {args.checkpoint}")
            if not args.parquet.is_file():
                raise FileNotFoundError(f"MMAU parquet not found: {args.parquet}")
            if not args.metadata_json.is_file():
                raise FileNotFoundError(f"MMAU metadata JSON not found: {args.metadata_json}")
            if not args.evaluation_script.is_file():
                raise FileNotFoundError(f"MMAU official evaluation.py not found: {args.evaluation_script}")
            official_records = load_official_records(args.metadata_json)
            report["official_artifact_audit"] = audit_mmau_v051525(
                args.metadata_json,
                args.evaluation_script,
                official_records,
            )
            if report["official_artifact_audit"]["status"] != "PASS":
                raise RuntimeError(
                    f"{MMAU_VERSION} official artifact audit failed: "
                    f"{report['official_artifact_audit']['failures']}"
                )
            metadata_index = build_metadata_index(official_records)
            report["metadata_index_records"] = len(metadata_index)
            # Load the exact checkpoint contract before consuming rows.  The
            # smoke mode therefore always validates model loading even when a
            # malformed first-five row is skipped, while parquet data itself
            # remains streamed only once in physical order.
            model, tokenizer, device, checkpoint_config = load_runtime_model(args)
            for row_index, row in iter_parquet_rows(
                args.parquet,
                batch_size=args.parquet_batch_size,
                limit=SMOKE_ROWS if args.mode == "smoke" else None,
            ):
                report["records"]["rows_read"] = int(report["records"].get("rows_read", 0) + 1)
                sample_id, _ = extract_row_id(row)
                if store.has_terminal(row_index, sample_id):
                    report["records"]["resumed_rows"] = int(report["records"].get("resumed_rows", 0) + 1)
                    continue
                try:
                    sample = _prepare_metadata_row(
                        row,
                        metadata_index=metadata_index,
                        dataset_dir=args.dataset_dir,
                        prompt_builder=prompt_builder,
                        audio_decoder=audio_decoder,
                        audio_root=audio_root,
                        prefer_official_audio_file=prefer_official_audio_file,
                    )
                    generation = run_model_generation(
                        model,
                        tokenizer,
                        device,
                        sample,
                        max_prompt_tokens=args.max_prompt_tokens,
                        max_new_tokens=args.max_new_tokens,
                    )
                    generated_text = str(generation.get("generated_text", ""))
                    model_output = prepare_prediction(generated_text)
                    record = {
                        "status": "generated",
                        "row_index": int(row_index),
                        "parquet_row_index": int(row_index),
                        "id": sample["id"],
                        "row_key": row_key(row_index, sample["id"]),
                        "id_source": sample["id_source"],
                        "question": sample["question"],
                        "choices": sample["choices"],
                        "prompt": sample["prompt"],
                        "parquet_question": sample["parquet_question"],
                        "parquet_answer": sample["parquet_answer"],
                        "official_record": sample["official_record"],
                        "model_output": model_output,
                        "audio2_reused": True,
                        "audio_payload_source": sample["audio_payload_source"],
                        **{key: value for key, value in sample.items() if key.startswith("audio_")},
                        **generation,
                    }
                    store.append_raw(record)
                    print(
                        json.dumps(
                            {
                                "row_index": row_index,
                                "id": sample["id"],
                                "status": "generated",
                                "model_output": model_output,
                                "generated_text": generation.get("generated_text", ""),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                except RowSkip as exc:
                    skip_record = {
                        "status": "skipped",
                        "row_index": int(row_index),
                        "parquet_row_index": int(row_index),
                        "id": sample_id or None,
                        "row_key": row_key(row_index, sample_id),
                        "stage": exc.stage,
                        "reason": exc.reason,
                        "error": exc.message,
                        **exc.details,
                    }
                    store.append_skip(skip_record)
                    print(json.dumps(skip_record, ensure_ascii=False), flush=True)
                except Exception as exc:
                    skip_record = {
                        "status": "skipped",
                        "row_index": int(row_index),
                        "parquet_row_index": int(row_index),
                        "id": sample_id or None,
                        "row_key": row_key(row_index, sample_id),
                        "stage": "inference",
                        "reason": "sample_exception",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(limit=8),
                    }
                    store.append_skip(skip_record)
                    print(json.dumps(skip_record, ensure_ascii=False), flush=True)
            if model is not None:
                del model
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
            scoring_records = (
                official_records
                if args.mode == "full"
                else official_records[:SMOKE_ROWS]
            )
            predictions = _materialize_from_store(
                store,
                args.output_dir,
                scoring_records,
            )
            report["records"].update(_counts(store))
            report["records"]["official_scoring_denominator"] = len(predictions)
            report["resumption"]["rows_not_repeated"] = int(report["records"].get("resumed_rows", 0))
            report["checkpoint_config"] = checkpoint_config
            expected = EXPECTED_FULL_ROWS if args.mode == "full" else SMOKE_ROWS
            if int(report["records"].get("rows_read", 0)) != expected:
                raise RuntimeError(
                    f"{args.mode} MMAU parquet traversal is incomplete: "
                    f"{report['records'].get('rows_read')} / {expected}"
                )
            if len(predictions) != expected:
                raise RuntimeError(
                    f"{args.mode} MMAU denominator must be {expected}, got {len(predictions)}"
                )
            if int(report["records"].get("terminal_records", 0)) != expected:
                raise RuntimeError(
                    f"{args.mode} MMAU terminal coverage is incomplete: "
                    f"{report['records'].get('terminal_records')} / {expected}"
                )
            skipped = int(report["records"].get("skipped", 0))
            prompt_too_long = int(
                report["records"].get("skip_reasons", {}).get(
                    "prompt_exceeds_max_tokens", 0
                )
            )
            report["records"]["official_empty_predictions_from_skips"] = skipped
            truncated_prompts = sum(
                bool(record.get("prompt_truncated"))
                for record in store.state.values()
                if record.get("status") == "generated"
            )
            report["prompt_length_audit"] = {
                "max_prompt_tokens": int(args.max_prompt_tokens),
                "prompt_exceeds_max_tokens": prompt_too_long,
                "prompt_truncated_records": truncated_prompts,
                "status": "PASS" if prompt_too_long == 0 else "WARNING",
            }
            if skipped:
                report["warnings"].append({
                    "name": "skipped_rows_scored_as_incorrect",
                    "count": skipped,
                    "reasons": report["records"].get("skip_reasons", {}),
                    "detail": (
                        "Every parquet row was visited. Skipped rows retain an empty model_output "
                        "and are counted as incorrect by the official scorer."
                    ),
                })
            report["inference_coverage"] = {
                "status": "PASS",
                "expected_rows": expected,
                "rows_read": int(report["records"].get("rows_read", 0)),
                "terminal_records": int(report["records"].get("terminal_records", 0)),
                "generated": int(report["records"].get("generated", 0)),
                "skipped_scored_as_empty": skipped,
            }
            report["official_evaluation"] = _run_official_evaluation(args, args.output_dir, len(predictions))
            report["official_evaluation"]["empty_predictions_from_skips"] = skipped
            report["comparable_official_score"] = (
                report["official_evaluation"].get("status") == "PASS"
            )
            report["status"] = overall_evaluation_status(
                str(report["inference_coverage"].get("status")),
                str(report["official_evaluation"].get("status")),
            )
            if report["official_evaluation"].get("status") == "NOT_REQUESTED":
                report["warnings"].append({
                    "name": "official_scorer_not_requested",
                    "detail": (
                        "Inference coverage completed, but no comparable official score "
                        "was produced. Re-run with --run-official-evaluation."
                    ),
                })
        except Exception as exc:
            report["fatal_error"] = {"error": repr(exc), "traceback": traceback.format_exc()}
            report["inference_coverage"] = {
                "status": "FAILED",
                "error": repr(exc),
            }
            # Even setup failures should leave the required output files and a
            # truthful report; no failed generation is silently scored.
            predictions = _materialize_from_store(
                store,
                args.output_dir,
                (
                    official_records
                    if official_records and args.mode == "full"
                    else official_records[:SMOKE_ROWS]
                    if official_records
                    else None
                ),
            )
            report["records"].update(_counts(store))
            report["records"]["official_scoring_denominator"] = len(predictions)
            report["official_evaluation"] = _block_official_evaluation(
                args,
                args.output_dir,
                len(predictions),
                exc,
            )
            report["status"] = "FAILED"
        report["elapsed_seconds"] = time.time() - started
        _write_json(args.output_dir / "evaluation_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(
        json.dumps(
            {
                "stage": report.get("stage"),
                "status": report.get("status"),
                "mode": report.get("mode"),
                "records": report.get("records", {}),
                "official_evaluation": report.get("official_evaluation", {}),
                "report": str(args.output_dir / "evaluation_report.json"),
            },
            ensure_ascii=False,
            default=_json_default,
        )
    )
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
