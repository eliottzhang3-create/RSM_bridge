#!/usr/bin/env python3
"""Compare two MMAU raw_generations.jsonl files on CPU, without model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


OUTPUTS = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol")
DEFAULT_REPRODUCTION = OUTPUTS / "mellow_official_reasonaqa_adamw_cosine_5090/formal_30epochs_20261001_003838/eval/mmau_test_mini_official_training_v1/raw_generations.jsonl"
DEFAULT_MELLOW = OUTPUTS / "mellow_v0/mmau_test_mini_mellow_author_reply_matched_smollm2_113430_v2/raw_generations.jsonl"
DEFAULT_SCORER = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/MMAU_test_mini/evaluation.py")
DEFAULT_OUTPUT = OUTPUTS / "mmau_mellow_vs_reproduction_case_analysis_20261010"
SCORER_SHA256 = "85480e1c0dfe8ee1406e9c6e598eff0dca9e0216701f076faf69081c6aab1558"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    index_counts: Counter[int] = Counter()
    id_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    invalid: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("not a JSON object")
                index, sample_id = row.get("row_index"), row.get("id")
                if type(index) is not int or index < 0 or not isinstance(sample_id, str) or not sample_id:
                    raise ValueError("missing or invalid row_index/id")
                index_counts[index] += 1
                id_counts[sample_id] += 1
                status_counts[str(row.get("status"))] += 1
                rows.setdefault(index, row)
            except (ValueError, json.JSONDecodeError) as exc:
                invalid.append({"line": line_number, "error": str(exc)})
    duplicates = sorted(index for index, count in index_counts.items() if count != 1)
    duplicate_ids = sorted(sample_id for sample_id, count in id_counts.items() if count != 1)
    for index, row in list(rows.items()):
        if index in duplicates or row["id"] in duplicate_ids:
            del rows[index]
    return rows, {"path": str(path), "sha256": sha256(path), "status_counts": dict(status_counts),
                  "duplicate_row_indices": duplicates, "duplicate_ids": duplicate_ids,
                  "invalid_lines": invalid, "unique_candidate_rows": len(rows)}


def official_prediction(text: str) -> str:
    # Identical to prepare_model_output_for_official_scorer in the project.
    return re.sub(r"^\s*[a-d]\)\s*", "", text, count=1, flags=re.IGNORECASE)


def official_string_match(answer: str, prediction: str, choices: list[str]) -> bool:
    # Identical to MMAU-v05.15.25 evaluation.py:string_match, SHA-checked below.
    def tokenize(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    prediction_tokens, answer_tokens = tokenize(prediction), tokenize(answer)
    if not prediction_tokens:
        return False
    incorrect_tokens: set[str] = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)
    return answer_tokens.issubset(prediction_tokens) and prediction_tokens.isdisjoint(incorrect_tokens)


def comparable_metadata(row: dict[str, Any]) -> tuple[str, tuple[str, ...], str] | None:
    official = row.get("official_record")
    if not isinstance(official, dict) or official.get("id") != row.get("id"):
        return None
    question, choices, answer = official.get("question"), official.get("choices"), official.get("answer")
    if not (isinstance(question, str) and isinstance(choices, list) and choices
            and all(isinstance(choice, str) for choice in choices) and isinstance(answer, str)
            and row.get("question") == question and row.get("choices") == choices):
        return None
    return question, tuple(choices), answer


def analyze(mellow: dict[int, dict[str, Any]], reproduction: dict[int, dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    cases: dict[str, list[dict[str, Any]]] = {"letter": [], "official_text": []}
    exclusions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index in sorted(mellow.keys() | reproduction.keys()):
        left, right = mellow.get(index), reproduction.get(index)
        if left is None or right is None:
            reason = "missing_mellow" if left is None else "missing_reproduction"
        elif left["id"] != right["id"]:
            reason = "id_mismatch"
        elif left.get("status") != "generated" or right.get("status") != "generated":
            reason = "non_generated_status"
        elif comparable_metadata(left) is None or comparable_metadata(left) != comparable_metadata(right):
            reason = "metadata_mismatch_or_invalid"
        elif not isinstance(left.get("generated_text"), str) or not isinstance(right.get("generated_text"), str):
            reason = "missing_generated_text"
        else:
            reason = None
        if reason:
            exclusions.append({"row_index": index, "mellow_id": left.get("id") if left else None,
                               "reproduction_id": right.get("id") if right else None, "reason": reason})
            continue
        assert left is not None and right is not None
        details = comparable_metadata(left)
        assert details is not None
        question, choices_tuple, answer = details
        choices = list(choices_tuple)
        label = next((chr(ord("a") + i) for i, choice in enumerate(choices)
                      if answer.lower() == choice.lower()), None)
        left_text, right_text = left["generated_text"], right["generated_text"]
        item = {"question_number": index + 1, "row_index": index, "id": left["id"],
                "question": question, "choices": choices,
                "correct_option": f"{label}) {answer}" if label else answer,
                "mellow_output": left_text, "reproduction_output": right_text}
        counts["aligned_generated"] += 1
        if label is None:
            exclusions.append({"row_index": index, "id": left["id"], "reason": "gold_answer_not_in_choices_for_letter_scoring"})
        else:
            # Mellow author's exact rule: compare text before first ')' after lowercasing.
            left_ok = left_text.split(")")[0].lower() == label
            right_ok = right_text.split(")")[0].lower() == label
            counts["letter_mellow_correct"] += left_ok
            counts["letter_reproduction_correct"] += right_ok
            if left_ok and not right_ok:
                cases["letter"].append(item)
        left_ok = official_string_match(answer, official_prediction(left_text), choices)
        right_ok = official_string_match(answer, official_prediction(right_text), choices)
        counts["official_mellow_correct"] += left_ok
        counts["official_reproduction_correct"] += right_ok
        if left_ok and not right_ok:
            cases["official_text"].append(item)
    return cases, {"counts": {**dict(counts), "letter_cases": len(cases["letter"]),
                              "official_text_cases": len(cases["official_text"]),
                              "excluded_items": len(exclusions)}, "excluded": exclusions}


def write_report(path: Path, title: str, rule: str, rows: list[dict[str, Any]], aligned: int) -> None:
    lines = [f"# {title}", "", rule, "", f"命中 {len(rows)} 题；可对齐的 generated 题目 {aligned} 题。", ""]
    for row in rows:
        lines += [f"## 第 {row['question_number']} 题（row_index={row['row_index']}）", "",
                  f"- ID：{row['id']}", f"- 题目：{row['question']}",
                  f"- 正确选项：{row['correct_option']}",
                  "- 所有选项：" + "；".join(f"{chr(ord('a') + i)}) {choice}" for i, choice in enumerate(row["choices"])),
                  "- Mellow output：" + json.dumps(row["mellow_output"], ensure_ascii=False),
                  "- 复现模型 output：" + json.dumps(row["reproduction_output"], ensure_ascii=False), ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mellow", type=Path, default=DEFAULT_MELLOW)
    parser.add_argument("--reproduction", type=Path, default=DEFAULT_REPRODUCTION)
    parser.add_argument("--official-scorer", type=Path, default=DEFAULT_SCORER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    mellow_path = args.mellow.expanduser().resolve(strict=True)
    reproduction_path = args.reproduction.expanduser().resolve(strict=True)
    scorer_path = args.official_scorer.expanduser().resolve(strict=True)
    scorer_hash = sha256(scorer_path)
    if scorer_hash != SCORER_SHA256:
        parser.error(f"MMAU scorer SHA256 differs: expected={SCORER_SHA256}, actual={scorer_hash}, path={scorer_path}")
    if mellow_path == reproduction_path:
        parser.error("inputs must be different files")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir in {mellow_path.parent, reproduction_path.parent}:
        parser.error("output directory must be separate from input directories")
    mellow_rows, mellow_audit = read_rows(mellow_path)
    reproduction_rows, reproduction_audit = read_rows(reproduction_path)
    cases, comparison = analyze(mellow_rows, reproduction_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned = comparison["counts"].get("aligned_generated", 0)
    letter_report = output_dir / "mellow_correct_reproduction_wrong_choice_letter.md"
    text_report = output_dir / "mellow_correct_reproduction_wrong_official_text.md"
    write_report(letter_report, "Mellow 对、复现模型错：选项字母", "按 Mellow author-reply：首个 ) 前的文本转小写后与正确选项字母比较，不修剪空格。", cases["letter"], aligned)
    write_report(text_report, "Mellow 对、复现模型错：MMAU 官方答案文本", "去掉生成文本开头的 a)～d) 后，按 MMAU-v05.15.25 官方 token-set string_match 判分，并非字符串严格相等。", cases["official_text"], aligned)
    complete = (aligned == 1000 and not comparison["excluded"] and
                all(not audit[key] for audit in (mellow_audit, reproduction_audit)
                    for key in ("duplicate_row_indices", "duplicate_ids", "invalid_lines")))
    audit = {"status": "PASS" if complete else "INCOMPLETE_INPUT",
             "official_scorer": {"path": str(scorer_path), "sha256": scorer_hash},
             "mellow": mellow_audit, "reproduction": reproduction_audit,
             "comparison": comparison,
             "reports": {"choice_letter": str(letter_report), "official_text": str(text_report)}}
    audit_path = output_dir / "comparison_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], **comparison["counts"],
                      "reports": audit["reports"], "audit": str(audit_path)}, ensure_ascii=False))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
