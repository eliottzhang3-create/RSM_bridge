"""Identity-backed joins only; no fuzzy question or filename search across sources."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path, PurePosixPath
import re

from .io_utils import digest, iter_rows


SOURCES = {"AudioCaps", "Clotho", "CompA-R", "LP-MusicCaps-MTT",
           "MusicCaps", "SpeechCraft", "TACOS"}
AUDIO_CONTRACT = "raw_mono_resample32k_first20_pad640000_no_second_audio"


def norm(value):
    return " ".join(value.casefold().split())


def slash(value):
    return value.replace("\\", "/")


class Unmatched(ValueError):
    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.reason, self.details = reason, details or {}


def validate_candidate(row):
    for key in ("sample_key", "id", "source", "source_dataset", "source_sample_key",
                "filepath1", "input", "answer", "question", "gold_choice"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"missing/non-string S3 field {key}")
    if row["source"] == "avqa":
        if row["source_dataset"] != "AVQA":
            raise ValueError("contradictory AVQA source labels")
    elif row["source"] != "audiomcq" or row["source_dataset"] not in SOURCES:
        raise ValueError("unregistered S3 source")
    options, gold = row.get("choices"), row.get("gold_option_index")
    if (not isinstance(options, list) or len(options) != 4
            or not all(isinstance(x, str) and x.strip() for x in options)
            or type(gold) is not int or not 0 <= gold < 4):
        raise ValueError("S3 requires four nonempty options and a valid integer gold index")
    if norm(options[gold]) != norm(row["gold_choice"]) or row.get("gold_letter") != "abcd"[gold]:
        raise ValueError("S3 gold index/letter/content disagree")
    prompt = row["question"] + " " + " ".join(f"{letter}) {option}" for letter, option in zip("abcd", options))
    if (norm(row["input"]) != norm(prompt)
            or norm(row["answer"]) != norm(f"{'abcd'[gold]}) {row['gold_choice']}")):
        raise ValueError("S3 input or direct answer disagrees with its options/gold")
    if "<|endoftext|>" in row["input"] or "<|endoftext|>" in row["answer"]:
        raise ValueError("S3 metadata must leave EOS handling to the S3 loader")
    if row.get("filepath2") != "" or row.get("audio_input_contract") != AUDIO_CONTRACT:
        raise ValueError("requires original single-recording S3 first20 contract")


def mcq_key(source, sample, choices, gold=None):
    # S3 corrects some historical S1 gold labels. Recording identity must not
    # depend on the old answer, and old targets must never overwrite S3 targets.
    return source, str(sample), tuple(norm(x) for x in choices)


def candidate_mcq_key(row):
    return mcq_key(row["source_dataset"], row["source_sample_key"], row["choices"], row["gold_option_index"])


def recording_name(value):
    """Case-sensitive YouTube ID + exact integer start offset; underscores remain in ID."""
    if not isinstance(value, str):
        raise Unmatched("avqa_invalid_recording_name", {"value": value})
    name = PurePosixPath(slash(value)).name
    if name.endswith(".wav"):
        name = name[:-4]
    match = re.fullmatch(r"([A-Za-z0-9_-]{11})_(\d+)", name)
    if not match:
        raise Unmatched("avqa_invalid_recording_name", {"value": value})
    return match[1], int(match[2])


def avqa_relative(value):
    if not isinstance(value, str):
        raise Unmatched("avqa_unsafe_or_unknown_annotation_path", {"value": value})
    value = slash(value)
    for prefix in ("./Joysw909/AVQA/", "Joysw909/AVQA/", "./AVQA/", "AVQA/", "./"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    path = PurePosixPath(value)
    if (path.is_absolute() or len(path.parts) != 2
            or path.parts[0] not in {"VGG10000", "VGG20000", "VGG30000", "VGG40000"}
            or path.suffix != ".wav" or ".." in path.parts):
        raise Unmatched("avqa_unsafe_or_unknown_annotation_path", {"value": value})
    return path


class AVQAMapper:
    def __init__(self, annotation, audio_root):
        self.root = Path(audio_root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("AVQA audio root is not a directory")
        self.annotations = defaultdict(list)
        self.annotation_counts = Counter()
        for index, row in enumerate(iter_rows(annotation)):
            raw_id = row.get("id")
            if type(raw_id) is not int or raw_id < 0:
                raise ValueError(f"AVQA annotation has invalid integer id at {index}")
            self.annotations[raw_id].append((index, row))
            self.annotation_counts[str(row.get("question_relation"))] += 1
        if not self.annotations:
            raise ValueError("AVQA annotation is empty")
        self.files = defaultdict(set)
        self.file_count = 0
        inventory = []
        for folder in ("VGG10000", "VGG20000", "VGG30000", "VGG40000"):
            directory = self.root / folder
            if not directory.is_dir():
                continue
            for path in directory.rglob("*"):
                if path.suffix != ".wav" or not path.is_file():
                    continue
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(self.root):
                    raise ValueError(f"AVQA symlink escapes configured root: {path}")
                self.file_count += 1
                stat = resolved.stat()
                inventory.append((str(path.relative_to(self.root)), str(resolved),
                                  stat.st_size, stat.st_mtime_ns))
                try:
                    self.files[recording_name(path.name)].add(str(resolved))
                except Unmatched:
                    continue
        if not self.file_count:
            raise ValueError("no WAV files under the four AVQA VGG directories; check audio root")
        self.inventory_sha256 = digest(sorted(inventory))
        self.direct, self.failures, self.groups = {}, {}, defaultdict(set)
        self.anchors = defaultdict(list)

    def observe(self, row):
        """Collect trusted annotation joins before resolving shared S3 recording paths."""
        key = row["sample_key"]
        try:
            match = re.fullmatch(r"avqa_train_(\d+)", row["source_sample_key"])
            if not match:
                raise Unmatched("avqa_invalid_source_sample_key")
            records = self.annotations.get(int(match[1]), [])
            if not records:
                raise Unmatched("avqa_annotation_id_missing")
            if len(records) != 1:
                raise Unmatched("avqa_duplicate_annotation_id", {"count": len(records)})
            index, raw = records[0]
            options, gold = raw.get("multi_choice"), raw.get("answer")
            if (not isinstance(options, list) or len(options) != 4
                    or not all(isinstance(x, str) and x.strip() for x in options)
                    or type(gold) is not int or not 0 <= gold < 4):
                raise Unmatched("avqa_invalid_original_options_or_zero_based_answer")
            if ([norm(x) for x in options] != [norm(x) for x in row["choices"]]
                    or gold != row["gold_option_index"] or norm(options[gold]) != norm(row["gold_choice"])):
                raise Unmatched("avqa_options_or_gold_mismatch")
            if norm(str(raw.get("question_relation", ""))) not in {"sound", "both"}:
                raise Unmatched("avqa_original_relation_not_sound_or_both")
            relative = avqa_relative(raw.get("audio_path", ""))
            recording = recording_name(raw.get("video_name", ""))
            if recording_name(relative.name) != recording:
                raise Unmatched("avqa_video_and_audio_path_disagree")
            evidence = {"method": "avqa_original_id_options_gold_recording",
                        "annotation_id": raw["id"], "annotation_row_index": index,
                        "annotation_record_sha256": digest(raw), "video_name": raw["video_name"],
                        "annotation_audio_path": raw["audio_path"],
                        "recording_id": list(recording), "question_relation": raw["question_relation"]}
            self.direct[key] = (recording, evidence)
            self.groups[row["filepath1"]].add(recording)
            self.anchors[row["filepath1"]].append({"sample_key": key,
                "annotation_id": raw["id"], "annotation_record_sha256": digest(raw)})
        except Unmatched as error:
            self.failures[key] = error

    def resolve(self, row):
        key, old_path = row["sample_key"], row["filepath1"]
        recordings = self.groups.get(old_path, set())
        if len(recordings) > 1:
            raise Unmatched("avqa_shared_s3_path_has_conflicting_recordings")
        if key in self.direct:
            recording, evidence = self.direct[key]
        else:
            failure = self.failures[key]
            # Only missing annotations may inherit an independently validated
            # recording join. Contradictory labels are never rescued this way.
            if failure.reason != "avqa_annotation_id_missing" or len(recordings) != 1:
                raise failure
            recording = next(iter(recordings))
            evidence = {"method": "avqa_same_exact_s3_recording_path_as_validated_id",
                        "recording_id": list(recording), "anchor_s3_filepath1": old_path,
                        "validated_anchors": self.anchors[old_path]}
        matches = self.files.get(recording, set())
        if not matches:
            raise Unmatched("avqa_recording_wav_missing", {"recording_id": list(recording)})
        if len(matches) != 1:
            raise Unmatched("avqa_multiple_wavs_for_recording", {"paths": sorted(matches)})
        return next(iter(matches)), evidence


class AudioMCQMapper:
    def __init__(self, history, resolved_tsv, needed_keys, progress_every=100000):
        self.entries = defaultdict(dict)
        self.audits = defaultdict(set)
        self.history_counts = Counter()
        for index, row in enumerate(iter_rows(history)):
            if (index + 1) % progress_every == 0:
                print(f"[s3-prepare] historical rows scanned={index+1:,}", flush=True)
            source = row.get("source_dataset")
            if (row.get("source") != "audiomcq" or source not in SOURCES
                    or row.get("audio_contribution") != "strong"
                    or row.get("stageE_variant") != "audiomcq_mcq"):
                continue
            options, letter = row.get("choices"), row.get("gold_letter")
            if (not isinstance(options, list) or len(options) != 4
                    or not all(isinstance(x, str) and x.strip() for x in options)
                    or letter not in ("a", "b", "c", "d")):
                raise ValueError(f"invalid historical strongAC direct metadata at {index}")
            self.history_counts[source] += 1
            key = mcq_key(source, row.get("id"), options, "abcd".index(letter))
            if key not in needed_keys:
                continue
            raw = row.get("audio_path")
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"historical strongAC has no raw audio_path at {index}")
            entry = self.entries[key].setdefault(raw, {"historical_row_index": index,
                "historical_record_sha256": digest(row), "historical_gold_letters": []})
            if letter not in entry["historical_gold_letters"]:
                entry["historical_gold_letters"].append(letter)
        if not self.history_counts:
            raise ValueError("history contains no strongAC direct rows; wrong historical manifest")
        needed_paths = {raw for rows in self.entries.values() for raw in rows}
        with Path(resolved_tsv).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not {"field", "raw_path", "status", "resolved_path"} <= set(reader.fieldnames or []):
                raise ValueError("historical resolved TSV has incompatible columns")
            for record in reader:
                if record["field"] == "audio_path" and record["raw_path"] in needed_paths:
                    self.audits[record["raw_path"]].add((record["status"], record["resolved_path"]))

    def resolve(self, row):
        entries = self.entries.get(candidate_mcq_key(row), {})
        if not entries:
            raise Unmatched("audiomcq_no_historical_strong_direct_identity_options")
        if len(entries) != 1:
            raise Unmatched("audiomcq_conflicting_historical_raw_references", {"paths": sorted(entries)})
        raw = next(iter(entries))
        old_name = PurePosixPath(slash(raw)).name
        s3_name = PurePosixPath(slash(row["filepath1"])).name
        if old_name != s3_name:
            raise Unmatched("audiomcq_historical_and_s3_recording_filename_disagree",
                            {"historical_raw_path": raw, "s3_filepath1": row["filepath1"]})
        if row["source_dataset"] == "SpeechCraft":
            split = lambda p: re.findall(r"(?:^|/)(train-clean-100|train-clean-360|train-other-500)(?:/|$)", slash(p))
            if len(split(raw)) != 1 or split(raw) != split(row["filepath1"]):
                raise Unmatched("audiomcq_speechcraft_split_disagrees")
        records = self.audits.get(raw, set())
        if not records:
            raise Unmatched("audiomcq_historical_audit_missing", {"historical_raw_path": raw})
        if len(records) != 1:
            raise Unmatched("audiomcq_historical_audit_conflict", {"historical_raw_path": raw})
        status, path = next(iter(records))
        if status != "exists" or not path:
            raise Unmatched("audiomcq_historical_audit_not_successful", {"historical_raw_path": raw})
        return path, {"method": "historical_strong_direct_identity_options_recording_and_successful_audit",
                      "historical_raw_path": raw, **entries[raw],
                      "historical_gold_matches_s3": entries[raw]["historical_gold_letters"] == [row["gold_letter"]],
                      "gold_authority": "unchanged_s3_candidates_metadata"}
