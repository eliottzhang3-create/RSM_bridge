#!/usr/bin/env python3
"""Aggregate three reports only when benchmark and inference protocols agree."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def compare_reports(paths):
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    hashes = {report.get("comparison_protocol_sha256") for report in reports}
    if len(hashes) != 1 or None in hashes:
        raise ValueError("reports use different data, scorer, code, tokenizer, environment or generation protocols")
    if any(not report.get("comparable") or report.get("status") != "PASS" for report in reports):
        raise ValueError("all reports must be complete and comparable PASS")
    architectures = [report["architecture"] for report in reports]
    if set(architectures) != {"smollm2", "mesh_x2", "mesh_x4"} or len(architectures) != 3:
        raise ValueError("supply exactly one report per architecture: smollm2, mesh_x2, mesh_x4")
    attention = {report["loaded_model_audit"]["attention_implementation"] for report in reports}
    vocab = {report["loaded_model_audit"]["tokenizer_vocab_sha256"] for report in reports}
    if len(attention) != 1 or len(vocab) != 1:
        raise ValueError("loaded attention implementation or tokenizer vocabulary differs")
    configs = [report["loaded_model_audit"]["shared_lm_config"] for report in reports]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("non-architectural LM settings differ")
    rows = []
    for report in reports:
        prefix = report["choice_label_prefix_evaluation"]["total"]
        official = report["official_evaluation"]
        if prefix["total"] != 1000 or official.get("reported_total") != 1000:
            raise ValueError("both scorers must use the full 1000-row denominator")
        rows.append({
            "task": report["protocol"]["task"], "architecture": report["architecture"],
            "choice_label_accuracy_percent": prefix["accuracy_percent"],
            "official_accuracy_percent": official["total_accuracy_percent"],
            "generated": report["records"]["generated"], "skipped": report["records"]["skipped"],
            "checkpoint_file": report["checkpoint_file"],
            "comparison_protocol_sha256": report["comparison_protocol_sha256"],
        })
    # Cross-model sample-level evidence: same prompt IDs and same chosen audio.
    samples = []
    for report, path in zip(reports, paths):
        entries = [json.loads(line) for line in (path.parent / "raw_generations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        sample_map = {str(item["id"]): (item["prompt_token_ids"], item["audio1_waveform_sha256"])
                      for item in entries if item.get("status") == "generated"}
        if len(sample_map) != report["records"]["generated"]:
            raise ValueError("raw generation coverage differs from report")
        samples.append(sample_map)
    if any(sample != samples[0] for sample in samples[1:]):
        raise ValueError("models did not receive identical sample coverage, prompt tokens and audio segments")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs=3, type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    rows = compare_reports(args.reports)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
