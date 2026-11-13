"""GT-independent MMSU answer parsing, separate from immutable raw generation."""
from __future__ import annotations

import json
import os
import re
from collections import Counter

from . import dataset, storage

VERSION = "mmsu_explicit_choice_v2"
POLICY = {
    "version": VERSION,
    "priority": ["single_letter", "option_label", "bracketed_label", "answer_phrase", "unique_exact_option_text"],
    "text_matching": "strip_one_terminal_EOS_then_casefold_and_collapse_whitespace",
    "uses_answer_gt": False,
    "invalid_or_ambiguous": "empty_scorer_response_counted_wrong_in_full_5000_denominator",
    "raw_generation": "preserved_without_modification",
    "scoring": "untouched_pinned_official_scorer_on_separate_normalized_copy",
}
EOS = "<|endoftext|>"


def parse_response(raw, choices):
    """Recognize explicit labels or a unique full choice; never search prose for letters."""
    if not isinstance(raw, str) or len(choices) != 4:
        raise ValueError("expected a text response and four original choices")
    text = raw.strip()
    eos_removed = text.endswith(EOS)
    if eos_removed:
        text = text[:-len(EOS)].rstrip()
    result = {"choice": None, "method": "unrecognized", "eos_removed": eos_removed}
    if not text:
        return {**result, "method": "empty"}
    if re.fullmatch(r"[A-Da-d]", text):
        return {**result, "choice": text.upper(), "method": "single_letter"}
    label = re.match(r"^([A-Da-d])[.):](?:\s+|$)", text)
    if label:
        return {**result, "choice": label[1].upper(), "method": "option_label"}
    label = re.match(r"^(?:\(([A-Da-d])\)|\[([A-Da-d])\])(?:[.:]?\s+|$)", text)
    if label:
        return {**result, "choice": (label[1] or label[2]).upper(), "method": "bracketed_label"}
    phrase = re.fullmatch(
        r"(?:the\s+)?(?:answer|correct\s+answer|option|choice)(?:\s+is)?\s*[:：]?\s*"
        r"(?:\(([A-Da-d])\)|\[([A-Da-d])\]|([A-Da-d]))[.!]?", text, flags=re.IGNORECASE)
    if phrase:
        return {**result, "choice": next(value for value in phrase.groups() if value).upper(),
                "method": "answer_phrase"}
    normalized = " ".join(text.casefold().split())
    matches = [letter for letter, value in zip("ABCD", choices)
               if isinstance(value, str) and " ".join(value.casefold().split()) == normalized]
    if len(matches) == 1:
        return {**result, "choice": matches[0], "method": "unique_exact_option_text"}
    if matches:
        return {**result, "method": "ambiguous_option_text", "candidates": matches}
    return result


def read_complete_jsonl(path):
    """Offline readers never truncate source files, even an incomplete last write."""
    def bad_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    records = []
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if number > dataset.EXPECTED_ROWS or not line.endswith(b"\n"):
                raise ValueError(f"incomplete/oversized MMSU JSONL at row {number}: {path}")
            value = json.loads(line, parse_constant=bad_constant)
            if not isinstance(value, dict):
                raise ValueError(f"expected MMSU object at row {number}: {path}")
            records.append(value)
    return records


def validate_predictions(records):
    seen = set()
    if len(records) != dataset.EXPECTED_ROWS:
        raise ValueError("MMSU scoring requires exactly 5000 unique predictions")
    for row in records:
        dataset.validate_record(row)
        key = dataset.row_id(row)
        if key in seen or not isinstance(row.get("response"), str):
            raise ValueError(f"duplicate MMSU ID or non-text response: {key}")
        seen.add(key)


def prepare_predictions(records, output_dir):
    """Write an official-schema scoring copy and a full per-row parsing audit."""
    validate_predictions(records)
    predictions = output_dir / "predictions_normalized.jsonl"
    audit = output_dir / "response_parsing.jsonl"
    temporary = [p.with_name(p.name + ".tmp") for p in (predictions, audit)]
    methods = Counter()
    invalid, changed, eos = 0, 0, 0
    examples = []
    with temporary[0].open("w", encoding="utf-8", newline="\n") as normalized, \
            temporary[1].open("w", encoding="utf-8", newline="\n") as details:
        for index, record in enumerate(records):
            raw = record["response"]
            parsed = parse_response(raw, [record[key] for key in dataset.CHOICES])
            response = parsed["choice"] or ""
            methods[parsed["method"]] += 1
            invalid += parsed["choice"] is None
            changed += response != raw
            eos += parsed["eos_removed"]
            row = {key: record[key] for key in dataset.METADATA}
            row["response"] = response
            normalized.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            detail = {"row_index": index, "id": record["id"], "raw_response": raw,
                      "scorer_response": response, **parsed}
            details.write(json.dumps(detail, ensure_ascii=False, allow_nan=False) + "\n")
            if parsed["choice"] is None and len(examples) < 20:
                examples.append(detail)
        for stream in (normalized, details):
            stream.flush()
            os.fsync(stream.fileno())
    for source, destination in zip(temporary, (predictions, audit)):
        source.replace(destination)
    report = {"status": "PASS", "policy": POLICY, "rows": len(records),
              "recognized_rows": len(records) - invalid, "invalid_rows": invalid,
              "ambiguous_rows": methods["ambiguous_option_text"], "methods": dict(methods),
              "changed_response_rows": changed, "terminal_eos_removed_rows": eos,
              "invalid_counted_wrong_rows": invalid, "invalid_examples": examples,
              "normalized_predictions": str(predictions), "parsing_details": str(audit),
              "normalized_predictions_sha256": storage.file_sha(predictions),
              "parsing_details_sha256": storage.file_sha(audit)}
    storage.write_json(output_dir / "parsing_report.json", report)
    return predictions, report


def score_records(records, script, output_dir):
    from . import scoring
    predictions, report = prepare_predictions(records, output_dir)
    result = scoring.run_scorer(script, predictions, output_dir)
    result.update(response_policy=VERSION, invalid_response_policy=POLICY["invalid_or_ambiguous"])
    return report, result
