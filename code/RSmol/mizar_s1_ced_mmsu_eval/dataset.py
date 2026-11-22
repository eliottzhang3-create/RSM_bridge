"""Read the three official HF shards without exporting audio or using the hub."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

EXPECTED_ROWS = 5000
SHARD_ROWS = (1667, 1667, 1666)
SHARD_NAMES = tuple(f"train-{i:05d}-of-00003.parquet" for i in range(3))
CHOICES = ("choice_a", "choice_b", "choice_c", "choice_d")
METADATA = ("id", "task_name", "question", *CHOICES, "answer_gt", "category",
            "sub-category", "sub-sub-category", "linguistics_sub_discipline")
INSTRUCTION = ("Choose the most suitable answer from options A, B, C, and D. "
               "You must respond with only A, B, C, or D.")
PROMPT_VERSION = "mmsu_compact_lowercase_ced_fixed20_raw_greedy_answer_text_v3"
PROMPT_POLICY = "MMSU_compact_question_abcd_parentheses_lowercase_preserve_null_positions_v1"
OUTPUT_NAMESPACE = "mmsu_compact_lowercase_v1"


def prompt(record):
    """Match MMAU's fixed-order builder followed by lower(), without instructions.

    Keep the four original slots: str(None) becomes 'none', never drop/relabel
    choices. Ground-truth/category metadata are not used to construct the input.
    """
    lines = [f"{chr(ord('a') + index)}) {str(record[key]).strip()}"
             for index, key in enumerate(CHOICES)]
    question = str(record['question']).strip()
    return f"{question} {' '.join(lines)}".strip().lower()


def official_prompt(record):
    """Historical template for diagnostic ablations only; not production input."""
    choices = "\n".join(f"{letter}. {record[key]}" for letter, key in zip("ABCD", CHOICES))
    return f"{INSTRUCTION}\n\nQuestion: {record['question']}\n\n{choices}"


def row_id(record):
    value = record["id"]
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"invalid MMSU id: {value!r}")
    return str(value)


def validate_record(record):
    if not set(METADATA) <= record.keys():
        raise ValueError(f"missing MMSU fields: {sorted(set(METADATA) - record.keys())}")
    row_id(record)
    for key in METADATA:
        value = record[key]
        if key == "id" or (key in CHOICES and value is None):
            continue
        if not isinstance(value, str):
            raise ValueError(f"MMSU {record['id']} field {key} must be text, got {type(value).__name__}")
    if not record["question"].strip() or not record["answer_gt"].strip():
        raise ValueError(f"empty question/answer_gt: {record['id']}")
    if record["answer_gt"] not in [record[key] for key in CHOICES]:
        raise ValueError(f"answer_gt does not equal an original choice: {record['id']}")


def shard_paths(dataset_dir):
    directory = Path(dataset_dir) / "data"
    paths = [directory / name for name in SHARD_NAMES]
    actual = sorted(p.name for p in directory.glob("*.parquet"))
    if actual != list(SHARD_NAMES):
        raise ValueError(f"expected exactly the three MMSU train shards in {directory}, got {actual}")
    return paths


def iter_rows(paths, *, audio, batch_size=8):
    import pyarrow.parquet as pq

    index = 0
    columns = [*METADATA, "audio"] if audio else list(METADATA)
    for path in paths:
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
                for row in batch.to_pylist():
                    yield index, row
                    index += 1


def preflight(paths, batch_size=8):
    """Audit all metadata before loading GPU weights; audio remains streamed."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    shards = []
    for path, expected in zip(paths, SHARD_ROWS, strict=True):
        with pq.ParquetFile(path) as parquet:
            schema = parquet.schema_arrow
            if set(schema.names) != {*METADATA, "audio"}:
                raise ValueError(f"unexpected MMSU schema: {path}: {schema.names}")
            audio_type = schema.field("audio").type
            if (not pa.types.is_struct(audio_type) or {f.name for f in audio_type} != {"bytes", "path"}
                    or not pa.types.is_binary(audio_type.field("bytes").type)
                    or not pa.types.is_string(audio_type.field("path").type)):
                raise ValueError(f"expected HF audio struct bytes/path: {path}: {audio_type}")
            count = parquet.metadata.num_rows
            if count != expected:
                raise ValueError(f"MMSU shard row count: {path}: {count} != {expected}")
            shards.append({"name": path.name, "rows": count, "schema": str(schema)})
    records, audit = audit_records(iter_rows(paths, audio=False, batch_size=batch_size))
    audit["shards"] = shards
    return records, audit


def audit_records(indexed_rows):
    records, seen = [], set()
    tasks, categories, nulls = Counter(), Counter(), Counter()
    ambiguous = 0
    for index, row in indexed_rows:
        if index != len(records):
            raise ValueError("noncontiguous global MMSU row index")
        validate_record(row)
        key = row_id(row)
        if key in seen:
            raise ValueError(f"duplicate MMSU id across shards: {key}")
        seen.add(key)
        records.append(row)
        tasks[row["task_name"]] += 1
        categories[f"{row['category']}/{row['sub-category']}"] += 1
        nulls.update(key for key in CHOICES if row[key] is None)
        ambiguous += sum(row[k] == row["answer_gt"] for k in CHOICES) > 1
    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"MMSU requires {EXPECTED_ROWS} metadata rows, got {len(records)}")
    return records, {"status": "PASS", "rows": len(records),
                     "task_count": len(tasks), "task_rows": dict(tasks),
                     "category_rows": dict(categories), "null_choices": dict(nulls),
                     "multiple_matching_choices": ambiguous,
                     "answer_policy": "original_text_exact_equality_no_normalization"}
