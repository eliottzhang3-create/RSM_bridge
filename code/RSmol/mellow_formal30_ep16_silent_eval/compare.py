#!/usr/bin/env python3
"""Aggregate three reports only when benchmark and inference protocols agree."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name
from . import contracts as C


def compare_reports(paths):
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    hashes = {report.get("comparison_protocol_sha256") for report in reports}
    if len(hashes) != 1 or None in hashes:
        raise ValueError("reports use different data, scorer, code, tokenizer, environment or generation protocols")
    if any(not report.get("comparable") or report.get("status") != "PASS" for report in reports):
        raise ValueError("all reports must be complete and comparable PASS")
    for report in reports:
        contract = report.get("checkpoint_contract") or {}
        protocol = report.get("protocol") or {}
        if (contract.get("epoch_completed") != C.EVALUATION_EPOCH
                or contract.get("num_epochs") != 30
                or contract.get("route_variant") != C.POLICIES["route_variant"]):
            raise ValueError("all reports must use formal30 epoch-16 checkpoints")
        if (protocol.get("version") != C.PROTOCOL_VERSION
                or protocol.get("audio2") != C.AUDIO2_POLICY
                or protocol.get("prediction_preprocessing") != "none"):
            raise ValueError("reports do not use the isolated silent-slot/raw-scoring protocol")
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
            "epoch_completed": report["checkpoint_contract"]["epoch_completed"],
            "num_epochs": report["checkpoint_contract"]["num_epochs"],
            "comparison_protocol_sha256": report["comparison_protocol_sha256"],
        })
    # Cross-model sample-level evidence: same prompt IDs and same chosen audio.
    samples = []
    silent_hash = hashlib.sha256(bytes(320000 * 4)).hexdigest()
    for report, path in zip(reports, paths):
        entries = [json.loads(line) for line in (path.parent / "raw_generations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        generated = [item for item in entries if item.get("status") == "generated"]
        if len({str(item["id"]) for item in generated}) != len(generated):
            raise ValueError("duplicate raw generations for one sample")
        for item in generated:
            if (item.get("audio2_policy") != C.AUDIO2_POLICY
                    or item.get("audio2_nonzero_elements") != 0
                    or item.get("audio2_max_abs") != 0.0
                    or item.get("audio2_waveform_sha256") != silent_hash
                    or item.get("audio2_reused") is not False
                    or item.get("audio2_waveform_materialized") is not True
                    or item.get("audio_encoder_invocations") != 2):
                raise ValueError("raw generation does not prove a silent second waveform encoded independently")
        sample_map = {str(item["id"]): (item["prompt_token_ids"], item["audio1_waveform_sha256"],
                                        item["audio2_waveform_sha256"]) for item in generated}
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
