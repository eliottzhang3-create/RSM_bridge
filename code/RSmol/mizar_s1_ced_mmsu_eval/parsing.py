"""User-selected normalized GT-text containment, with a separate official input."""
from __future__ import annotations

import json
import os
import re
from collections import Counter

from . import dataset, storage

VERSION = "mmsu_answer_text_contains_v2"
POLICY = {
    "version": VERSION,
    "grading": "normalized_complete_answer_gt_text_contained_in_normalized_generation",
    "text_matching": "casefold_collapse_whitespace_preserve_punctuation_word_boundaries",
    "response_format": "remove_one_terminal_EOS_and_leading_option_label_for_text_comparison_only",
    "uses_answer_gt": "offline_grading_only_never_model_input",
    "model_option_letter": "ignored_for_correctness",
    "nonmatching_text": "empty_scorer_response_counted_wrong_in_full_5000_denominator",
    "official_conversion": "on_text_match_use_first_original_choice_exactly_equal_to_answer_gt",
    "limitation": "mentions_in_explanations_negations_or_multiple_answers_also_count_as_matches",
    "raw_generation": "preserved_without_modification",
    "scoring": "untouched_pinned_official_scorer_on_separate_normalized_copy",
}
EOS = "<|endoftext|>"


def parse_response(raw, answer_gt):
    """Grade only complete GT text presence; the generated option letter is irrelevant."""
    if not isinstance(raw, str) or not isinstance(answer_gt, str) or not answer_gt.strip():
        raise ValueError("expected a text response and a nonempty ground-truth answer text")
    text = raw.strip()
    eos_removed = text.endswith(EOS)
    if eos_removed:
        text = text[:-len(EOS)].rstrip()
    # Formatting is discarded without using the letter as a prediction. This
    # also prevents an option marker 'a.' from matching a short GT text 'a'.
    text, prefix_removed = re.subn(
        r"^(?:[A-Da-d][.):]|\([A-Da-d]\)[.:]?|\[[A-Da-d]\][.:]?)(?:\s+|$)", "", text, count=1)
    normalized = " ".join(text.casefold().split())
    target = " ".join(answer_gt.casefold().split())
    left = r"(?<!\w)" if target[0].isalnum() or target[0] == "_" else ""
    right = r"(?!\w)" if target[-1].isalnum() or target[-1] == "_" else ""
    match = re.search(left + re.escape(target) + right, normalized)
    return {"matched": match is not None,
            "method": "answer_text_present" if match else "answer_text_absent",
            "normalized_response_text": normalized, "normalized_answer_gt": target,
            "match_span": list(match.span()) if match else None,
            "eos_removed": eos_removed, "format_prefix_removed": bool(prefix_removed)}


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
    nonmatching, changed, eos, duplicated_gt = 0, 0, 0, 0
    examples = []
    with temporary[0].open("w", encoding="utf-8", newline="\n") as normalized, \
            temporary[1].open("w", encoding="utf-8", newline="\n") as details:
        for index, record in enumerate(records):
            raw = record["response"]
            parsed = parse_response(raw, record["answer_gt"])
            # Original choices are used ONLY to serialize the already-decided
            # text-match result for the unmodified official scorer.
            correct_letters = [letter for letter, key in zip("ABCD", dataset.CHOICES)
                               if record[key] == record["answer_gt"]]
            response = correct_letters[0] if parsed["matched"] else ""
            methods[parsed["method"]] += 1
            nonmatching += not parsed["matched"]
            duplicated_gt += len(correct_letters) > 1
            changed += response != raw
            eos += parsed["eos_removed"]
            row = {key: record[key] for key in dataset.METADATA}
            row["response"] = response
            normalized.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            detail = {"row_index": index, "id": record["id"], "raw_response": raw,
                      "scorer_response": response, **parsed}
            details.write(json.dumps(detail, ensure_ascii=False, allow_nan=False) + "\n")
            if not parsed["matched"] and len(examples) < 20:
                examples.append(detail)
        for stream in (normalized, details):
            stream.flush()
            os.fsync(stream.fileno())
    for source, destination in zip(temporary, (predictions, audit)):
        source.replace(destination)
    report = {"status": "PASS", "policy": POLICY, "rows": len(records),
              "matched_rows": len(records) - nonmatching, "nonmatching_rows": nonmatching,
              "multiple_gt_choice_rows": duplicated_gt, "methods": dict(methods),
              "changed_response_rows": changed, "terminal_eos_removed_rows": eos,
              "nonmatching_counted_wrong_rows": nonmatching, "nonmatching_examples": examples,
              "normalized_predictions": str(predictions), "parsing_details": str(audit),
              "normalized_predictions_sha256": storage.file_sha(predictions),
              "parsing_details_sha256": storage.file_sha(audit)}
    storage.write_json(output_dir / "parsing_report.json", report)
    return predictions, report


def score_records(records, script, output_dir):
    from . import scoring
    predictions, report = prepare_predictions(records, output_dir)
    result = scoring.run_scorer(script, predictions, output_dir)
    result.update(response_policy=VERSION, nonmatching_response_policy=POLICY["nonmatching_text"])
    if result["execution_status"] == "PASS" and (
            result.get("actual_denominator") != report["rows"]
            or result.get("overall_accuracy") != report["matched_rows"] / report["rows"]):
        raise ValueError("official scorer disagrees with normalized-answer-text match totals")
    return report, result
