"""Prepare all reliably mapped S3 candidates; no sampling or Q4 selection."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from itertools import zip_longest
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

from .audio import AudioAuditor
from .io_utils import canonical, digest, identity, iter_rows, loads, publish, sha256_file, sync, write_json
from .mapping import (AUDIO_CONTRACT, AVQAMapper, AudioMCQMapper, Unmatched,
                      candidate_mcq_key, validate_candidate)


FORMAT = "mizar_s3_full_candidates_filtered_raw_audio_v1"
OFFICIAL_SHA256 = "c4b1862f58148d2728dfad10ee8346974924badca6de6b82181ffe605225cc65"
OFFICIAL_COUNTS = {"AudioCaps": 49687, "Clotho": 1992, "CompA-R": 48045,
                   "LP-MusicCaps-MTT": 7871, "MusicCaps": 637, "SpeechCraft": 123529,
                   "TACOS": 24316, "AVQA": 33875}
BASE = "/hpc_stor03/sjtu_home/jinwei.zhang"
FILES = ("train.json", "selection.jsonl", "excluded.jsonl", "resolved_paths.tsv",
         "audio_path_map.json", "audio_audit.jsonl", "dataset_report.json")
POLICY = {"sampling": "none; preserve every verified candidate in original S3 order",
          "metadata": "original S3 objects unchanged; exact file-open mapping is external",
          "audio": AUDIO_CONTRACT, "decode": "full recording, native float32, finite, CPU SoundFile",
          "avqa": "original id+ordered choices+zero-based gold+case-sensitive recording id/start",
          "audiomcq": "historical strong direct identity/options/recording, original successful audit path; S3 gold authoritative",
          "filter": "exclude each unconfirmed/conflicting/missing/undecodable sample with evidence",
          "split": "no new train/probe split; no Q4/quadrant/option permutation/CoT augmentation"}


def source_identity():
    folder = Path(__file__).resolve().parent
    paths = list(folder.glob("*.py")) + [folder.parent / "prepare_mizar_s3_data.sh"]
    return {path.name: sha256_file(path) for path in sorted(paths)}


def audio_stamp_or_missing(path):
    try:
        return AudioAuditor.stat(path)[1]
    except PermissionError:
        raise
    except (OSError, ValueError) as error:
        return {"unavailable": type(error).__name__, "message": str(error)}


def resolve_audio(row, path_map):
    """Use with train.json; the only supported lookup is its original filepath1."""
    if row.get("filepath2") != "" or row.get("filepath1") not in path_map:
        raise ValueError("training row has no verified single-recording mapping")
    return Path(path_map[row["filepath1"]])


def verify_dataset(root, *, check_audio=True):
    root = Path(root).resolve(strict=True)
    manifest = loads((root / "manifest.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in manifest.items() if k != "seal_sha256"}
    if (manifest.get("format") != FORMAT or manifest.get("status") != "PASS"
            or manifest.get("seal_sha256") != digest(body) or set(manifest.get("files", {})) != set(FILES)
            or manifest.get("policy") != POLICY):
        raise ValueError("invalid S3 dataset identity/seal/policy/inventory")
    for name, record in manifest["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"prepared file changed: {path}")
    report = loads((root / "dataset_report.json").read_text(encoding="utf-8"))
    if report["binding"] != manifest["binding"] or report["policy"] != POLICY:
        raise ValueError("report identity disagrees with manifest")
    paths = loads((root / "audio_path_map.json").read_text(encoding="utf-8"))
    decoded = {}
    for record in iter_rows(root / "audio_audit.jsonl"):
        path = record["resolved_path"]
        if path in decoded:
            raise ValueError("duplicate audio audit path")
        decoded[path] = record
        if check_audio and AudioAuditor.stat(path)[1] != record["stat"]:
            raise ValueError(f"retained audio identity changed: {path}")
    counts, reasons, seen_paths = Counter(), Counter(), {}
    selected_ids, selected_keys = set(), set()
    total = report["input_rows"]
    if type(total) is not int or total <= 0:
        raise ValueError("invalid original denominator")
    seen = bytearray(total)
    last = -1
    for row, selected in zip_longest(iter_rows(root / "train.json"), iter_rows(root / "selection.jsonl")):
        if row is None or selected is None:
            raise ValueError("manifest/selection row count disagrees")
        validate_candidate(row)
        if row["id"] in selected_ids or row["sample_key"] in selected_keys:
            raise ValueError("duplicate retained sample identity")
        selected_ids.add(row["id"])
        selected_keys.add(row["sample_key"])
        index = selected["source_row_index"]
        if (type(index) is not int or not last < index < total
                or selected["source_record_sha256"] != digest(row)
                or selected["sample_key"] != row["sample_key"]
                or selected["selected_index"] != sum(counts.values())):
            raise ValueError("metadata/order/selection provenance mismatch")
        last = index
        seen[index] = 1
        path = paths.get(row["filepath1"])
        if not path or path != selected["resolved_path"] or path not in decoded:
            raise ValueError("incomplete or conflicting exact recording mapping")
        seen_paths[row["filepath1"]] = path
        counts[row["source_dataset"]] += 1
    last = -1
    excluded_counts = Counter()
    exclusions = iter_rows(root / "excluded.jsonl") if (root / "excluded.jsonl").stat().st_size else ()
    for record in exclusions:
        index = record["source_row_index"]
        if type(index) is not int or not last < index < total or seen[index]:
            raise ValueError("invalid/duplicate excluded source index")
        last = index
        seen[index] = 1
        if record["source_record_sha256"] != digest(record["metadata"]):
            raise ValueError("excluded source metadata digest mismatch")
        excluded_counts[record["metadata"]["source_dataset"]] += 1
        reasons[record["reason"]] += 1
    if (not all(seen) or dict(counts) != report["kept_by_source"]
            or dict(excluded_counts) != report["excluded_by_source"]
            or dict(reasons) != report["excluded_by_reason"]
            or counts + excluded_counts != Counter(report["input_by_source"])
            or sum(counts.values()) != manifest["rows"]
            or report["kept_rows"] != manifest["rows"]
            or report["excluded_rows"] != sum(excluded_counts.values())
            or total != sum(counts.values()) + sum(excluded_counts.values())
            or report["unique_retained_audio"] != len(decoded)
            or seen_paths != paths or set(paths.values()) != set(decoded)):
        raise ValueError("coverage/counts/path-map inventory mismatch")
    with (root / "resolved_paths.tsv").open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    if (len(records) != len(paths) or {r["raw_path"]: r["resolved_path"] for r in records} != paths
            or any(r["field"] != "filepath1" or r["status"] != "exists" for r in records)):
        raise ValueError("resolved TSV and exact JSON map disagree")
    return manifest


def prepare(source, avqa_metadata, avqa_root, historical_manifest, historical_audit,
            output_dir, *, expected_sha=OFFICIAL_SHA256, expected_counts=None, progress_every=10000):
    """Python API permits explicit small fixture identities; CLI always pins official S3."""
    if expected_counts is None:
        expected_counts = OFFICIAL_COUNTS
    if progress_every <= 0:
        raise ValueError("progress-every must be positive")
    output = Path(output_dir).absolute()
    # No outputs or source-side lock files before checking read-only inputs.
    decoder = AudioAuditor()
    code = source_identity()
    inputs = {}
    for name, path in (("s3_candidates", source), ("avqa_metadata", avqa_metadata),
                       ("historical_manifest", historical_manifest), ("historical_audit", historical_audit)):
        print(f"[s3-prepare] fingerprinting {name}: {path}", flush=True)
        inputs[name] = identity(path)
    if inputs["s3_candidates"]["sha256"] != expected_sha:
        raise ValueError("S3 candidate SHA disagrees with official release; refusing a different training universe")
    protected = [Path(r["path"]) for r in inputs.values()]
    audio_root = Path(avqa_root).resolve(strict=True)
    resolved_output = output.resolve()
    if (any(p.is_relative_to(resolved_output) for p in protected)
            or resolved_output == audio_root or audio_root.is_relative_to(resolved_output)
            or resolved_output.is_relative_to(audio_root)):
        raise ValueError("output must not contain/replace input metadata or overlap AVQA audio root")
    avqa = AVQAMapper(avqa_metadata, audio_root)
    counts, ids, sample_keys, needed_keys = Counter(), set(), set(), set()
    for index, row in enumerate(iter_rows(source)):
        validate_candidate(row)
        if row["sample_key"] in sample_keys or row["id"] in ids:
            raise ValueError(f"duplicate S3 candidate identity at source row {index}")
        ids.add(row["id"])
        sample_keys.add(row["sample_key"])
        counts[row["source_dataset"]] += 1
        if row["source"] == "avqa":
            avqa.observe(row)
        else:
            needed_keys.add(candidate_mcq_key(row))
        if (index + 1) % progress_every == 0:
            print(f"[s3-prepare] metadata validated={index+1:,}", flush=True)
    if counts != Counter(expected_counts):
        raise ValueError(f"S3 source inventory disagrees: {dict(counts)} != {expected_counts}")
    mcq = AudioMCQMapper(historical_manifest, historical_audit, needed_keys)
    del needed_keys, ids, sample_keys
    historical_paths = {p for rows in mcq.audits.values() for status, p in rows if status == "exists" and p}
    binding = {"inputs": inputs, "source_files_sha256": code, "decoder": decoder.decoder,
               "avqa_root": str(audio_root), "avqa_inventory_sha256": avqa.inventory_sha256,
               "historical_audio_stat_sha256": digest({p: audio_stamp_or_missing(p) for p in sorted(historical_paths)})}
    if output.exists() or output.is_symlink():
        verified = verify_dataset(output)
        if verified["binding"] != binding:
            raise ValueError("existing output has different input/code/audio identity; use a new output directory")
        print(f"[s3-prepare] reusing verified dataset: {output}", flush=True)
        return verified
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".prepare.lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    lock_stat = os.fstat(fd)
    stage = None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        fd = None
        stage = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
        pending = stage / ".decisions.jsonl"
        references, errors, decode_errors = defaultdict(set), {}, {}
        with pending.open("w", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(iter_rows(source)):
                decision = {"source_row_index": index, "source_record_sha256": digest(row),
                            "sample_key": row["sample_key"], "metadata": row}
                try:
                    path, evidence = (avqa if row["source"] == "avqa" else mcq).resolve(row)
                    path = str(Path(path).resolve())
                    references[row["filepath1"]].add(path)
                    decision.update(resolved_path=path, evidence=evidence)
                    if path in decode_errors:
                        raise Unmatched("audio_decode_or_file_check_failed", decode_errors[path])
                    try:
                        before_decodes = len(decoder.cache)
                        decoder.check(path)
                        if len(decoder.cache) != before_decodes and len(decoder.cache) % 200 == 0:
                            print(f"[s3-prepare] unique audio fully decoded={len(decoder.cache):,}; "
                                  f"candidate row={index+1:,}/{sum(counts.values()):,}", flush=True)
                    except PermissionError:
                        raise
                    except (OSError, ValueError, RuntimeError) as error:
                        details = {"resolved_path": path, "error_type": type(error).__name__, "error": str(error)}
                        decode_errors[path] = details
                        raise Unmatched("audio_decode_or_file_check_failed", details) from error
                    decision["status"] = "verified"
                except Unmatched as error:
                    decision.update(status="excluded", reason=error.reason, details=error.details)
                    errors[error.reason] = errors.get(error.reason, 0) + 1
                handle.write(canonical(decision) + "\n")
                if (index + 1) % progress_every == 0:
                    print(f"[s3-prepare] mapped/decoded rows={index+1:,}/{sum(counts.values()):,}; "
                          f"unique decoded={len(decoder.cache):,}; row exclusions={sum(errors.values()):,}", flush=True)
            sync(handle)
        conflicts = {raw: sorted(paths) for raw, paths in references.items() if len(paths) != 1}
        kept, excluded, reasons, methods = Counter(), Counter(), Counter(), Counter()
        historical_gold_differences = 0
        path_map = {}
        with (stage / "train.json").open("w", encoding="utf-8", newline="\n") as train, \
                (stage / "selection.jsonl").open("w", encoding="utf-8", newline="\n") as selection, \
                (stage / "excluded.jsonl").open("w", encoding="utf-8", newline="\n") as rejected:
            train.write("[")
            for decision in iter_rows(pending):
                row = decision["metadata"]
                raw = row["filepath1"]
                if raw in conflicts:
                    decision.update(status="excluded", reason="s3_recording_path_maps_to_conflicting_files",
                                    details={"resolved_paths": conflicts[raw]})
                if decision["status"] == "excluded":
                    excluded[row["source_dataset"]] += 1
                    reasons[decision["reason"]] += 1
                    rejected.write(canonical(decision) + "\n")
                    continue
                selected_index = sum(kept.values())
                if selected_index:
                    train.write(",\n")
                train.write(canonical(row))
                path_map[raw] = decision["resolved_path"]
                kept[row["source_dataset"]] += 1
                methods[decision["evidence"]["method"]] += 1
                if decision["evidence"].get("historical_gold_matches_s3") is False:
                    historical_gold_differences += 1
                selection.write(canonical({k: v for k, v in decision.items() if k != "metadata"}
                                          | {"selected_index": selected_index}) + "\n")
            train.write("]\n")
            for handle in (train, selection, rejected):
                sync(handle)
        if not sum(kept.values()):
            raise ValueError("no verified samples remain; see building directory decisions")
        # Empty JSONL files are valid audit inventories, handled separately by
        # verifier. Empty input/candidate files remain errors in iter_rows.
        write_json(stage / "audio_path_map.json", dict(sorted(path_map.items())))
        used_audio = set(path_map.values())
        with (stage / "audio_audit.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for path in sorted(used_audio):
                handle.write(canonical(decoder.cache[path]) + "\n")
            sync(handle)
        with (stage / "resolved_paths.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["field", "raw_path", "status", "resolved_path"], delimiter="\t")
            writer.writeheader()
            for raw, path in sorted(path_map.items()):
                writer.writerow({"field": "filepath1", "raw_path": raw, "status": "exists", "resolved_path": path})
            sync(handle)
        retained = [decoder.cache[p] for p in used_audio]
        report = {"format": FORMAT, "status": "PASS", "binding": binding, "policy": POLICY,
                  "input_rows": sum(counts.values()), "input_by_source": dict(counts),
                  "kept_rows": sum(kept.values()), "kept_by_source": dict(kept),
                  "excluded_rows": sum(excluded.values()), "excluded_by_source": dict(excluded),
                  "excluded_by_reason": dict(reasons), "mapping_methods": dict(methods),
                  "retained_audiomcq_historical_gold_differences": historical_gold_differences,
                  "historical_strong_direct_by_source": dict(mcq.history_counts),
                  "avqa_annotation_rows": sum(avqa.annotation_counts.values()),
                  "avqa_annotation_relations": dict(avqa.annotation_counts),
                  "avqa_indexed_wavs": avqa.file_count, "unique_retained_audio": len(used_audio),
                  "unique_retained_audio_bytes": sum(r["stat"]["bytes"] for r in retained),
                  "native_rates": dict(Counter(str(r["samplerate"]) for r in retained)),
                  "channels": dict(Counter(str(r["channels"]) for r in retained)),
                  "all_zero_audio": sum(r["all_zero"] for r in retained),
                  "first20_will_pad_audio": sum(r["first20_will_pad"] for r in retained),
                  "first20_will_crop_audio": sum(r["first20_will_crop"] for r in retained),
                  "audio_decoder_errors": list(decode_errors.values()),
                  "recording_path_conflicts": conflicts,
                  "audio_modified_or_copied": False, "model_loaded": False,
                  "limits": ["native decoded waveform hashes are not canonical resampled first20 hashes",
                             "no near-duplicate or unseen-evaluation claim; no new split created",
                             "retained files checked by stat on reuse; force re-decode using a new output directory"]}
        write_json(stage / "dataset_report.json", report)
        for name, path in (("s3_candidates", source), ("avqa_metadata", avqa_metadata),
                           ("historical_manifest", historical_manifest), ("historical_audit", historical_audit)):
            if identity(path) != inputs[name]:
                raise ValueError(f"input changed during preparation: {name}")
        if source_identity() != code:
            raise ValueError("preparation source changed during audit")
        decoder.recheck()
        if AVQAMapper(avqa_metadata, audio_root).inventory_sha256 != avqa.inventory_sha256:
            raise ValueError("AVQA inventory changed during preparation")
        if digest({p: audio_stamp_or_missing(p) for p in sorted(historical_paths)}) != binding["historical_audio_stat_sha256"]:
            raise ValueError("historical mapped audio inventory changed during preparation")
        manifest = {"format": FORMAT, "status": "PASS", "policy": POLICY, "binding": binding,
                    "rows": sum(kept.values()), "files": {name: {"bytes": (stage / name).stat().st_size,
                    "sha256": sha256_file(stage / name)} for name in FILES}}
        manifest["seal_sha256"] = digest(manifest)
        write_json(stage / "manifest.json", manifest)
        verify_dataset(stage)
        pending.unlink()
        publish(stage, output)
        stage = None
        print(canonical({"event": "S3_DATASET_PASS", "output": str(output), "kept": manifest["rows"],
                         "excluded": report["excluded_rows"], "kept_by_source": dict(kept)}), flush=True)
        return manifest
    except BaseException as error:
        if stage is not None:
            write_json(stage / "failure.json", {"status": "FAILED", "error": str(error),
                       "traceback": traceback.format_exc(), "output_not_published": True})
            print(f"[s3-prepare] FAILED; diagnostic directory preserved: {stage}", file=sys.stderr, flush=True)
        raise
    finally:
        if fd is not None:
            os.close(fd)
        try:
            current = lock.lstat()
            if not lock.is_symlink() and (current.st_dev, current.st_ino) == (lock_stat.st_dev, lock_stat.st_ino):
                lock.unlink()
        except FileNotFoundError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(BASE + "/data/Mizar/s3_candidates.json"))
    parser.add_argument("--avqa-metadata", type=Path, default=Path(BASE + "/data/AVQA/train_r1aqa_line.json"))
    parser.add_argument("--avqa-root", type=Path, default=Path(BASE + "/data/AVQA"))
    parser.add_argument("--historical-manifest", type=Path, default=Path(BASE + "/data/Mizar/s1_stage1_hybrid_filtered.json"))
    parser.add_argument("--historical-audit", type=Path, default=Path(BASE + "/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path(BASE + "/data/Mizar/s3_full_filtered_v1"))
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--verify-only", action="store_true", help="verify published files/seal/retained audio stat; no re-decode")
    parser.add_argument("--check-args", action="store_true", help="print configuration without reading production assets")
    args = parser.parse_args(argv)
    if args.check_args:
        print(canonical({"paths": {k: str(v) for k, v in vars(args).items() if isinstance(v, Path)},
                         "expected_sha256": OFFICIAL_SHA256, "expected_sources": OFFICIAL_COUNTS,
                         "policy": POLICY, "gpu": False}))
        return
    if args.verify_only:
        verified = verify_dataset(args.output_dir)
        print(canonical({"event": "S3_DATASET_VERIFIED", "rows": verified["rows"], "output": str(args.output_dir)}))
        return
    prepare(args.source, args.avqa_metadata, args.avqa_root, args.historical_manifest,
            args.historical_audit, args.output_dir, progress_every=args.progress_every)


if __name__ == "__main__":
    main()
