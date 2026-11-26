"""CPU-only, deterministic source-row selection and raw-audio resolution.

The original S1 is read twice and never changed. Publication requires the exact
release, all requested views, complete audio resolution, and tokenizer audits.
No model is constructed and no audio is copied, decoded, or encoded here.
"""
from __future__ import annotations
import argparse
import collections
import csv
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import traceback

from contract import (DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, REASON_ROWS, SEED,
                      SOURCE_COUNTS, STRONG_COUNTS, TOTAL_ROWS, digest,
                      raw_references, sha256_file, write_json)
from manifest_io import iter_rows
from publication import atomic_publish
from previous_training import DEFAULT_PREVIOUS_MANIFEST, load_previous_training

RELEASE_ROWS = 1477976
REASON_POOL = 756968
SOURCE_SHA256 = {False: "de92b7efd1dd6b5fd30b3f62c31639a3bcff6766a088849658fdd6bc8520641d",
                 True: "111abde09c75abfd8dc6ee66c4762d7557d0b55e62e5cbde0bcc7b485176ac91"}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")


def canonical_row(row):
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def row_kind(row):
    if row.get("source") == "reasonaqa_oneaudio":
        if row.get("filepath2_original") or row.get("audio_path2"):
            return None
        if not row.get("filepath1_original"):
            raise ValueError("ReasonAQA lacks its original recording")
        return "reasonaqa"
    if row.get("source") == "audiomcq" and row.get("audio_contribution") == "strong":
        view = row.get("stageE_variant")
        if view not in STRONG_COUNTS:
            raise ValueError(f"unregistered strongAC view: {view!r}")
        if (len(row.get("choices", [])) != 4 or row.get("gold_letter") not in ("a","b","c","d")
                or not str(row.get("id", "")).strip()):
            raise ValueError("invalid strongAC choices/gold/source ID")
        if row.get("filepath2_original") or row.get("audio_path2"):
            raise ValueError("strongAC must have exactly one original recording")
        return view
    return None


def select(source, identify, *, reason_rows=REASON_ROWS, strong_counts=None,
           release_rows=RELEASE_ROWS, reason_pool=REASON_POOL, progress=True, history=None):
    """Maximize recording coverage, then round-robin unused QA within recordings."""
    expected = STRONG_COUNTS if strong_counts is None else strong_counts
    groups = collections.defaultdict(list)
    counts = collections.Counter()
    selected = set()
    seen_strong, retained_strong = collections.Counter(), collections.Counter()
    if history is not None:
        reason_rows = history.strong_total
    seen = 0
    for index, row in enumerate(iter_rows(Path(source))):
        seen += 1
        kind = row_kind(row)
        if kind:
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("input", "answer")):
                raise ValueError(f"empty text at source row {index}")
            counts[kind] += 1
            if kind == "reasonaqa":
                groups[identify(row)].append(index)
            else:
                key = digest(row)
                seen_strong[key] += 1
                if history is None or seen_strong[key] <= history.strong_records[key]:
                    selected.add(index)
                    retained_strong[kind] += 1
                elif any(ref in history.references for ref in raw_references(row)[1]):
                    raise ValueError("previous training used this audio but its strongAC row changed/is absent; refusing to silently filter previously used audio")
        if progress and seen % 100000 == 0:
            print(f"[manifest] scanned {seen:,}; ReasonAQA recordings={len(groups):,}", flush=True)
    if seen != release_rows or counts["reasonaqa"] != reason_pool:
        raise ValueError(f"wrong S1 universe: rows={seen}, eligible={dict(counts)}")
    if {k: counts[k] for k in expected} != expected:
        raise ValueError(f"strongAC view counts differ: {dict(counts)}; required={expected}")
    if counts["reasonaqa"] < reason_rows:
        raise ValueError("insufficient single-recording ReasonAQA rows")
    if history is not None and retained_strong != history.strong_counts:
        raise ValueError("previous strongAC rows are not an unchanged subset of original S1")
    rng = random.Random(SEED)
    for key in sorted(groups):
        rng.shuffle(groups[key])
    active = sorted(groups)
    chosen = collections.Counter()
    remaining, rounds = reason_rows, 0
    while remaining:
        if not active:
            raise ValueError("ReasonAQA sampling exhausted before the fixed quota")
        rng.shuffle(active)
        next_active = []
        rounds += 1
        for key in active:
            if not remaining:
                break
            selected.add(groups[key].pop())
            chosen[key] += 1
            remaining -= 1
            if groups[key]:
                next_active.append(key)
        active = next_active
    return selected, {"source_rows": seen, "candidate_counts": dict(counts),
                      "seed": SEED, "reason_recording_groups": len(groups),
                      "selected_reason_recordings": len(chosen), "sampling_rounds": rounds,
                      "reason_rows_per_recording_histogram": dict(sorted(collections.Counter(chosen.values()).items())),
                      "retained_strong_counts":dict(retained_strong),
                      "excluded_strong_counts":{k:counts[k]-retained_strong[k] for k in expected},
                      "reason_sample_rows":reason_rows,
                      "previous_training":history.describe() if history is not None else None,
                      "sampling_policy": "sorted canonical recording groups; private seed1234; shuffled QA per group; reshuffled active groups each round; no replacement"}


class Resolver:
    def __init__(self, tsv, mapping_json):
        self.tsv = Path(tsv)
        self.mapping_json = Path(mapping_json)
        self.resolutions, self.stats, self.errors = {}, {}, {}
        self.rules = None
        self.checked = set()
        self.methods = collections.Counter()
        self.audit_records = {}
        self.required_previous_references = set()
        with self.tsv.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not {"field", "raw_path", "status", "resolved_path"} <= set(reader.fieldnames or []):
                raise ValueError("invalid existing raw-audio audit TSV")
            for record in reader:
                self.audit_records[(record["field"],record["raw_path"])] = record
                if record["status"] == "exists":
                    key, path = (record["field"], record["raw_path"]), record["resolved_path"]
                    if key in self.resolutions and self.resolutions[key] != path:
                        raise ValueError(f"conflicting old resolutions: {key}")
                    self.resolutions[key] = path

    def _fallback(self, field, raw):
        if self.rules is None:
            script = Path(__file__).resolve().parents[1] / "scripts/audit_mizar_s1_raw_audio.py"
            spec = importlib.util.spec_from_file_location("mizar_raw_mapping_readonly", script)
            self.mapping = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.mapping)
            self.rules = self.mapping.load_rules(self.mapping_json)
            self.mapping.build_recursive_indexes(self.rules, progress=True)
        return self.mapping.resolve(raw, field, self.rules)

    def resolve(self, row):
        _source, refs = raw_references(row)
        if len(refs) != 1:
            raise ValueError("this training manifest permits only one original recording")
        key = refs[0]
        if key in self.errors:
            return "UNRESOLVED:" + repr(key)
        if key in self.checked:
            return self.resolutions[key]
        path = self.resolutions.get(key)
        try:
            if key in self.required_previous_references and (not path or not Path(path).is_file()):
                raise ValueError(f"previous training used this audio; its original audited path is now missing: {path!r}; restore it rather than filtering or remapping")
            if not path or not Path(path).is_file():
                result = self._fallback(*key)
                if result["status"] != "exists":
                    self.errors[key] = {"field": key[0], "raw_path": key[1], **result}
                    return "UNRESOLVED:" + repr(key)
                path = result["path"]
                self.methods[result.get("method", "unspecified")] += 1
            canonical = str(Path(path).resolve(strict=True))
            if canonical not in self.stats:
                stat = Path(canonical).stat()
                if not Path(canonical).is_file() or stat.st_size <= 0:
                    raise ValueError("missing/empty resolved audio")
                self.stats[canonical] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            self.resolutions[key] = canonical
            self.checked.add(key)
            return canonical
        except (OSError, ValueError) as exc:
            self.errors[key] = {"field": key[0], "raw_path": key[1], "error": str(exc)}
            return "UNRESOLVED:" + repr(key)


def file_records(root, names):
    return {name: {"bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)} for name in names}


def verify_dataset(root, *, total_rows=None):
    root = Path(root)
    report = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    seal = report.pop("seal_sha256")
    if seal != digest(report) or report.get("status") != "PASS" or (total_rows is not None and report.get("rows") != total_rows):
        raise ValueError("wrong or corrupted dataset manifest")
    modern = report.get("format") == "mizar_l9_12x3_strong_reason_dataset_v2"
    if not modern and report.get("format") != "mizar_l9_12x3_strong_reason_dataset_v1":
        raise ValueError("wrong dataset format")
    if modern:
        from contract import validate_data_counts
        validate_data_counts(report["rows"],report["source_counts"],report["selected_counts"])
        history = report.get("previous_training",{})
        if history.get("strong_views") != report["rows"]//2 or not history.get("sha256"):
            raise ValueError("dataset lacks previous training membership evidence")
    elif total_rows is None and report.get("rows") != TOTAL_ROWS:
        raise ValueError("legacy dataset must retain its original fixed quota")
    expected_files = {"train.jsonl", "selection.jsonl", "resolved_paths.tsv", "dataset_report.json"}
    if modern:expected_files.add("excluded_strong_rows.jsonl")
    if set(report["files"]) != expected_files:
        raise ValueError("incomplete dataset file inventory")
    for name, expected in report["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"dataset file changed: {path}")
    report["seal_sha256"] = seal
    return report


class TokenAudit:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pending = []
        self.counts = collections.defaultdict(collections.Counter)

    def add(self, kind, row):
        if kind != "reasonaqa":
            self.pending.append((kind, row["input"].lower(), row["answer"].lower()))
            if len(self.pending) >= 256:
                self.flush()

    def flush(self):
        if not self.pending:
            return
        for slot, limit in ((1, 256), (2, 300)):
            ids = self.tokenizer([r[slot] + " <|endoftext|>" for r in self.pending],
                                 add_special_tokens=True, padding=False, truncation=False)["input_ids"]
            for row, tokens in zip(self.pending, ids):
                c = self.counts[row[0]]
                name = "prompt" if slot == 1 else "answer"
                c[name + "_rows"] += 1
                c[name + "_truncated"] += len(tokens) > limit
                c[name + "_eos_retained"] += bool(tokens and tokens[-1] == 0 and len(tokens) <= limit)
        self.pending.clear()


def build(source, output, resolver, tokenizer, *, quotas=None, history=None):
    """Small explicit quotas are an in-process fixture API, never a CLI override."""
    q = quotas or {"reason_rows": REASON_ROWS, "strong_counts": STRONG_COUNTS,
                   "release_rows": RELEASE_ROWS, "reason_pool": REASON_POOL}
    selected, selection = select(source, resolver.resolve, **q, history=history)
    expected_views = {k:history.strong_counts[k] for k in STRONG_COUNTS} if history is not None else q["strong_counts"]
    reason_rows = history.strong_total if history is not None else q["reason_rows"]
    expected_rows = reason_rows + sum(expected_views.values())
    counts, kinds, subtypes = collections.Counter(), collections.Counter({k:0 for k in STRONG_COUNTS}), collections.Counter()
    excluded_refs = set()
    used_refs, audio_sets = {}, collections.defaultdict(set)
    token_audit = TokenAudit(tokenizer)
    with (output / "train.jsonl").open("w", encoding="utf-8", newline="\n") as train, \
            (output / "selection.jsonl").open("w", encoding="utf-8", newline="\n") as provenance, \
            (output / "excluded_strong_rows.jsonl").open("w", encoding="utf-8", newline="\n") as excluded:
        selected_index = 0
        for source_index, row in enumerate(iter_rows(Path(source))):
            if source_index not in selected:
                if history is not None and row_kind(row) in STRONG_COUNTS:
                    ref = raw_references(row)[1][0]
                    excluded_refs.add(ref)
                    excluded.write(json.dumps({"source_row_index":source_index,"id":row.get("id"),
                        "view":row_kind(row),"source_record_sha256":digest(row),
                        "raw_reference":ref,"previous_reference_was_used":ref in history.references,
                        "previous_audit":resolver.audit_records.get(ref),
                        "reason":"not_used_by_previous_hybrid_training"},ensure_ascii=False)+"\n")
                continue
            kind = row_kind(row)
            source_name, refs = raw_references(row)
            resolved = resolver.resolve(row)
            used_refs[refs[0]] = resolved
            audio_sets[kind].add(resolved)
            counts[source_name] += 1
            kinds[kind] += 1
            if kind == "reasonaqa":
                subtypes[row.get("subtype", "UNCLASSIFIED")] += 1
            line = canonical_row(row)
            train.write(line + "\n")
            provenance.write(json.dumps({"selected_index": selected_index, "source_row_index": source_index,
                "view": kind, "source": source_name, "source_record_sha256": digest(row),
                "raw_field": refs[0][0], "raw_path": refs[0][1], "resolved_path": resolved}, ensure_ascii=False) + "\n")
            token_audit.add(kind, row)
            selected_index += 1
        train.flush(); os.fsync(train.fileno())
        provenance.flush(); os.fsync(provenance.fileno())
    token_audit.flush()
    if resolver.errors:
        write_json(output / "unresolved_audio.json", list(resolver.errors.values()))
        raise ValueError(f"{len(resolver.errors)} unresolved/empty audio references; fixed quotas preserved, no GPU submission; see {output / 'unresolved_audio.json'}")
    if kinds != {"reasonaqa": reason_rows, **expected_views}:
        raise ValueError("selection changed between the two source passes")
    paths = set(used_refs.values())
    with (output / "resolved_paths.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["field", "raw_path", "status", "resolved_path"], delimiter="\t")
        writer.writeheader()
        for (field, raw), resolved in sorted(used_refs.items()):
            stat = Path(resolved).stat()
            if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != resolver.stats[resolved]:
                raise ValueError(f"audio changed during preparation: {resolved}")
            writer.writerow({"field": field, "raw_path": raw, "status": "exists", "resolved_path": resolved})
    strong_audio = audio_sets["audiomcq_mcq"] | audio_sets["audiomcq_cot"]
    details = {**selection, "selected_counts": dict(kinds), "source_counts": dict(counts),
               "reason_subtypes": dict(subtypes), "unique_audio_files": len(paths),
               "unique_audio_bytes": sum(resolver.stats[p]["bytes"] for p in paths),
               "reason_strong_shared_recordings": len(audio_sets["reasonaqa"] & strong_audio),
               "excluded_unique_audio_references":len(excluded_refs),
               "audio_fallback_methods": dict(resolver.methods), "tokenization": dict(token_audit.counts),
               "audio_diversity_scope": "canonical resolved file paths, not an acoustic near-duplicate claim",
               "audio_decoded": False, "audio_copied": False, "model_forward": False}
    write_json(output / "dataset_report.json", details)
    return expected_rows, details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--resolved-paths", type=Path, default=Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_s1_audio_audit_s2_hybrid_20261024_170403/resolved_paths.tsv"))
    parser.add_argument("--mapping-json", type=Path, default=Path(__file__).resolve().parents[1] / "scripts/mizar_s1_audio_mapping_s2_hybrid.json")
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--previous-training-manifest",type=Path,default=Path(DEFAULT_PREVIOUS_MANIFEST))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.verify_only:
        print(json.dumps(verify_dataset(output), ensure_ascii=False)); return
    source = args.source.resolve(strict=True)
    source_sha = sha256_file(source)
    if source_sha != SOURCE_SHA256[source.name.endswith(".gz")]:
        raise ValueError("source is not the pinned ORIGINAL S1 release; the filtered S1 cannot supply the fixed full quota")
    tokenizer_dir = args.model_dir / "assets/smollm2-135m"
    inputs = {"source_sha256": source_sha, "old_audit_sha256": sha256_file(args.resolved_paths),
              "mapping_json_sha256": sha256_file(args.mapping_json),
              "tokenizer": file_records(tokenizer_dir, TOKENIZER_FILES), "seed": SEED,
              "preparer_sha256": sha256_file(Path(__file__)),
              "resolver_sha256": sha256_file(Path(__file__).resolve().parents[1] / "scripts/audit_mizar_s1_raw_audio.py")}
    inputs["previous_training_manifest_sha256"] = sha256_file(args.previous_training_manifest)
    inputs["previous_training_reader_sha256"] = sha256_file(Path(__file__).with_name("previous_training.py"))
    if output.exists():
        report = verify_dataset(output)
        if report["inputs"] != inputs:
            raise ValueError("existing dataset has different source/mapping/tokenizer/preparer identity; use a new output directory")
        print(f"[manifest] reusing verified {output}", flush=True); return
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".prepare.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
    try:
        os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor); descriptor = None
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        tokenizer.add_special_tokens({"pad_token": "!"})
        if len(tokenizer) != 49152 or tokenizer.pad_token_id != 17 or tokenizer.eos_token_id != 0:
            raise ValueError("tokenizer must retain vocabulary49152/PAD17/EOS0")
        resolver = Resolver(args.resolved_paths, args.mapping_json)
        history = load_previous_training(args.previous_training_manifest,resolver)
        print(json.dumps({"event":"PREVIOUS_TRAINING_COUNTS","strong_views":history.strong_total,
                          "reason_sample_rows":history.strong_total,"total_rows":2*history.strong_total,
                          "strong_counts":dict(history.strong_counts)},ensure_ascii=False),flush=True)
        rows, details = build(source, stage, resolver, tokenizer, history=history)
        print(json.dumps({"event":"RETAINED_DATASET_COUNTS","selected_counts":details["selected_counts"],
                          "source_counts":details["source_counts"],"excluded_strong_counts":details["excluded_strong_counts"],
                          "excluded_unique_audio_references":details["excluded_unique_audio_references"]},ensure_ascii=False),flush=True)
        if sha256_file(source) != source_sha or sha256_file(args.resolved_paths) != inputs["old_audit_sha256"] or sha256_file(args.mapping_json) != inputs["mapping_json_sha256"]:
            raise ValueError("source or mappings changed during preparation")
        if sha256_file(args.previous_training_manifest) != inputs["previous_training_manifest_sha256"]:
            raise ValueError("previous training manifest changed during preparation")
        from contract import validate_data_counts
        validate_data_counts(rows,details["source_counts"],details["selected_counts"])
        report = {"format": "mizar_l9_12x3_strong_reason_dataset_v2", "status": "PASS", "rows": rows,
                  "inputs": inputs, "source_path": str(source), "selected_counts": details["selected_counts"],
                  "source_counts":details["source_counts"],"previous_training":history.describe(),
                  "files": file_records(stage, ("train.jsonl", "selection.jsonl", "resolved_paths.tsv", "dataset_report.json", "excluded_strong_rows.jsonl"))}
        report["seal_sha256"] = digest(report)
        write_json(stage / "manifest.json", report)
        verify_dataset(stage)
        publication = atomic_publish(stage, output)
        print(json.dumps({"status": "PASS", "dataset": str(output), "rows": rows,
                          "unique_audio_files": details["unique_audio_files"], "publication": publication}, ensure_ascii=False), flush=True)
    except BaseException:
        write_json(stage / "failure.json", {"status": "FAILED", "traceback": traceback.format_exc()})
        print(f"[manifest] failed; diagnostics preserved at {stage}", file=sys.stderr, flush=True)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock.unlink()


if __name__ == "__main__":
    main()
