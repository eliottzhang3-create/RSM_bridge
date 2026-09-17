#!/usr/bin/env python3
"""CPU-only, zero-copy partition PLAN; never reads or copies waveform payloads.

QA rows are indivisible and assigned exactly once. Distinct-audio QA edges are
unioned; every complete connected component is placed in one partition.
Outputs are planning artifacts, NOT a dataset accepted by the training loader.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

FORMAT = "reasonaqa_zero_copy_component_plan_v1"
STORE_FORMAT = "manifest_unique_fixed_waveform_store_v1"
DEFAULT_MANIFEST = "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/audio_5_10_5_mellow/preflight/stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl"
DEFAULT_STORE = "/hpc_stor03/sjtu_home/jinwei.zhang/data/rsmol_reasonaqa_train_unique_waveforms_32k_10s_f32_v1"
DEFAULT_TOKENIZER = "/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_path(value: str) -> str:
    return os.path.normpath(os.path.expanduser(value)).replace("\\", "/")


def audio_path(row: dict[str, Any], first: bool) -> str:
    for key in (("audio1_path", "filepath1") if first else ("audio2_path", "filepath2")):
        if row.get(key):
            if not isinstance(row[key], str):
                raise ValueError(f"{key} must be a string")
            return row[key]
    return ""


def iter_rows(path: Path):
    # Nonblank-record ordinal agrees with ReasonAQADataset row_index; physical
    # line numbers are separately retained in outputs for diagnostics.
    ordinal = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield ordinal, line_number, row, line
            ordinal += 1


def load_store(store_dir: Path, manifest_sha: str):
    if (store_dir / "BUILDING").exists():
        raise ValueError("waveform store is still BUILDING")
    metadata = json.loads((store_dir / "metadata.json").read_text(encoding="utf-8"))
    expected = {"format": STORE_FORMAT, "status": "PASS", "manifest_sha256": manifest_sha,
                "sample_rate": 32000, "seconds": 10, "samples_per_audio": 320000,
                "bytes_per_audio": 1280000, "dtype": "float32", "byte_order": "little",
                "data_file": "waveforms.f32"}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"store metadata mismatch: {key}")
    if metadata.get("waveform_verification", {}).get("passed") is not True:
        raise ValueError("store waveform verification is not PASS")
    index_sha = sha256_file(store_dir / "index.jsonl")
    if index_sha != metadata.get("index_sha256"):
        raise ValueError("store index SHA256 mismatch")
    audio = []
    aliases: dict[str, int] = {}
    for _, _, entry, _ in iter_rows(store_dir / "index.jsonl"):
        aid = len(audio)
        if (entry.get("audio_id") != aid or entry.get("byte_length") != 1280000
                or entry.get("byte_offset") != aid * 1280000
                or entry.get("shape") != [1, 320000] or entry.get("dtype") != "float32"):
            raise ValueError(f"invalid store index row: audio_id={aid}")
        path = entry.get("source_path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"invalid source_path: audio_id={aid}")
        for raw in [path, *entry.get("manifest_aliases", [])]:
            alias = normalized_path(raw)
            previous = aliases.setdefault(alias, aid)
            if previous != aid:
                raise ValueError(f"ambiguous store alias: {alias}")
        audio.append(entry)
    total_bytes = len(audio) * 1280000
    if not audio or metadata.get("num_unique_audio_files") != len(audio):
        raise ValueError("store audio count mismatch")
    if metadata.get("total_waveform_bytes") != total_bytes:
        raise ValueError("store total byte count mismatch")
    if (store_dir / "waveforms.f32").stat().st_size != total_bytes:
        raise ValueError("store payload size mismatch")
    # Stored payload checksum is provenance, NOT re-verified by this planner.
    return metadata, audio, aliases, index_sha


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b] or (self.size[a] == self.size[b] and a > b):
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


@dataclass
class Component:
    component_id: int
    audio_ids: list[int]
    rows: int = 0
    waveform_bytes: int = 0
    sequence_length_sum: int = 0
    compute_proxy: int = 0
    distinct_audio_rows: int = 0
    source_rows: Counter = field(default_factory=Counter)


def collect_components(manifest: Path, audio: list[dict], aliases: dict[str, int], tokenizer=None,
                       token_batch_size: int = 1024):
    uf = UnionFind(len(audio))
    first_ids, second_ids, lengths = array("I"), array("I"), array("I")
    prompts, answers = [], []

    def flush():
        if not prompts:
            return
        if tokenizer is None:
            # Explicit fallback proxy, never described as token/compute truth.
            values = [260 + min(129, max(1, (len(p) + 3) // 4))
                      + min(250, max(1, (len(a) + 3) // 4)) for p, a in zip(prompts, answers)]
        else:
            pids = tokenizer(prompts, max_length=129, truncation=True, padding=False,
                             add_special_tokens=True)["input_ids"]
            aids = tokenizer(answers, max_length=250, truncation=True, padding=False,
                             add_special_tokens=False)["input_ids"]
            if len(pids) != len(prompts) or len(aids) != len(answers) or any(not x for x in aids):
                raise ValueError("tokenizer returned inconsistent or empty answer tokens")
            values = [260 + len(p) + len(a) for p, a in zip(pids, aids)]
        lengths.extend(values)
        prompts.clear()
        answers.clear()

    used = set()
    for ordinal, line_number, row, _ in iter_rows(manifest):
        a = audio_path(row, True)
        b = audio_path(row, False) or a
        if not a:
            raise ValueError(f"missing audio1 at manifest line {line_number}")
        try:
            a_id, b_id = aliases[normalized_path(a)], aliases[normalized_path(b)]
        except KeyError as exc:
            raise ValueError(f"audio absent from store at manifest line {line_number}: {exc}") from exc
        prompt = str(row.get("prompt") or row.get("question") or row.get("input") or "")
        answer = str(row.get("answer") or row.get("target") or row.get("output") or row.get("caption1") or "")
        if not answer:
            raise ValueError(f"missing answer at manifest line {line_number}")
        first_ids.append(a_id)
        second_ids.append(b_id)
        used.update((a_id, b_id))
        uf.union(a_id, b_id)
        prompts.append(prompt)
        answers.append(answer)
        if len(prompts) >= token_batch_size:
            flush()
        if (ordinal + 1) % 100000 == 0:
            print(f"[component-plan] scanned {ordinal + 1} QA rows", flush=True)
    flush()
    if len(used) != len(audio):
        raise ValueError("manifest does not use exactly the store audio inventory")
    by_root: dict[int, list[int]] = {}
    for aid in range(len(audio)):
        by_root.setdefault(uf.find(aid), []).append(aid)
    components = [Component(ids[0], ids, waveform_bytes=sum(audio[x]["byte_length"] for x in ids),
                            source_rows=Counter()) for ids in sorted(by_root.values(), key=lambda x: x[0])]
    audio_component = [0] * len(audio)
    for cid, component in enumerate(components):
        for aid in component.audio_ids:
            audio_component[aid] = cid
    for a, b, length in zip(first_ids, second_ids, lengths):
        cid = audio_component[a]
        if cid != audio_component[b]:
            raise AssertionError("union-find failed to contain double-audio QA")
        c = components[cid]
        c.rows += 1
        c.distinct_audio_rows += a != b
        c.sequence_length_sum += length
        c.compute_proxy += length * length
        groups = sorted({str(audio[a].get("source_group", "other")), str(audio[b].get("source_group", "other"))})
        c.source_rows["+".join(groups)] += 1
    return components, audio_component, first_ids, second_ids, lengths


def pack_components(components: list[Component], num_partitions: int = 6, *, seed: int = 20260917,
                    trials: int = 64, refine_candidates: int = 4, local_rounds: int = 8,
                    swap_attempts: int = 4000, row_tolerance: float = 0.02,
                    waveform_weight: float = 0.05, compute_weight: float = 0.05,
                    source_weight: float = 0.05):
    if num_partitions < 2 or len(components) < num_partitions:
        raise ValueError("zero-copy nonempty partitioning requires at least one component per partition")
    sources = sorted({key for c in components for key in c.source_rows})
    vectors = [[c.rows, c.waveform_bytes, c.compute_proxy,
                *[c.source_rows.get(s, 0) for s in sources]] for c in components]
    targets = [sum(v[d] for v in vectors) / num_partitions for d in range(len(vectors[0]))]
    weights = [1.0, waveform_weight, compute_weight,
               *[source_weight / max(1, len(sources)) for _ in sources]]

    def score(loads):
        deviations = [[(load[d] / targets[d] - 1) if targets[d] else 0 for d in range(len(targets))]
                      for load in loads]
        worst_rows = max(abs(x[0]) for x in deviations)
        # Lexicographic: first remove QA tolerance violations. Within tolerance,
        # QA variance weight=1; each secondary objective defaults to only 0.05.
        return (max(0.0, worst_rows - row_tolerance),
                sum(weights[d] * sum(x[d] ** 2 for x in deviations) / num_partitions
                    for d in range(len(targets))))

    def alter(load, vector, sign):
        for d, value in enumerate(vector):
            load[d] += sign * value

    rng = random.Random(seed)
    candidates = []
    for trial in range(trials):
        jitter = [1.0 if trial == 0 else rng.uniform(0.85, 1.15) for _ in components]
        order = sorted(range(len(components)), key=lambda i: (-components[i].rows * jitter[i], components[i].component_id))
        loads = [[0] * len(targets) for _ in range(num_partitions)]
        assignment = [-1] * len(components)
        counts = [0] * num_partitions
        for pos, cid in enumerate(order):
            eligible = [p for p in range(num_partitions) if counts[p] == 0] if pos < num_partitions else list(range(num_partitions))
            choices = []
            for p in eligible:
                alter(loads[p], vectors[cid], 1)
                choices.append((score(loads), loads[p][0], p))
                alter(loads[p], vectors[cid], -1)
            p = min(choices)[2]
            assignment[cid] = p
            counts[p] += 1
            alter(loads[p], vectors[cid], 1)
        candidates.append((score(loads), assignment, loads, counts))
        if (trial + 1) % 16 == 0:
            print(f"[component-plan] greedy starts {trial + 1}/{trials}", flush=True)
    candidates.sort(key=lambda x: (x[0], x[1]))
    refined = []
    for candidate_number, (_, assignment, loads, counts) in enumerate(candidates[:refine_candidates], 1):
        print(f"[component-plan] refining candidate {candidate_number}/{min(refine_candidates, trials)}", flush=True)
        current = score(loads)
        for _ in range(local_rounds):
            improved = False
            order = list(range(len(components)))
            rng.shuffle(order)
            for cid in order:
                origin = assignment[cid]
                if counts[origin] <= 1:
                    continue
                best = current
                destination = None
                for p in range(num_partitions):
                    if p == origin:
                        continue
                    alter(loads[origin], vectors[cid], -1)
                    alter(loads[p], vectors[cid], 1)
                    candidate = score(loads)
                    alter(loads[p], vectors[cid], -1)
                    alter(loads[origin], vectors[cid], 1)
                    if candidate < best:
                        best, destination = candidate, p
                if destination is not None:
                    alter(loads[origin], vectors[cid], -1)
                    alter(loads[destination], vectors[cid], 1)
                    counts[origin] -= 1
                    counts[destination] += 1
                    assignment[cid], current, improved = destination, best, True
            for _ in range(swap_attempts):
                a, b = rng.sample(range(len(components)), 2)
                pa, pb = assignment[a], assignment[b]
                if pa == pb:
                    continue
                alter(loads[pa], vectors[a], -1)
                alter(loads[pb], vectors[b], -1)
                alter(loads[pa], vectors[b], 1)
                alter(loads[pb], vectors[a], 1)
                candidate = score(loads)
                if candidate < current:
                    assignment[a], assignment[b] = pb, pa
                    current, improved = candidate, True
                else:
                    alter(loads[pa], vectors[b], -1)
                    alter(loads[pb], vectors[a], -1)
                    alter(loads[pa], vectors[a], 1)
                    alter(loads[pb], vectors[b], 1)
            if not improved:
                break
        refined.append((current, assignment))
    objective, assignment = min(refined, key=lambda x: (x[0], x[1]))
    return assignment, {"objective": list(objective), "row_balance_within_tolerance": objective[0] == 0,
                        "source_categories": sources, "targets": targets,
                        "optimality": "deterministic multistart heuristic; no global optimum guarantee"}


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def audit_written_assignments(output: Path, audio_partition: list[int], first_ids, second_ids,
                              summaries: list[dict]) -> None:
    """Re-read emitted sidecars; verify coverage, uniqueness and local references."""
    seen_audio = bytearray(len(audio_partition))
    for p in range(len(summaries)):
        count = 0
        for _, _, row, _ in iter_rows(output / f"partition_{p}_audio.jsonl"):
            aid = row["audio_id"]
            if not 0 <= aid < len(seen_audio) or seen_audio[aid] or audio_partition[aid] != p:
                raise AssertionError("emitted audio has duplicate, missing or incorrect ownership")
            seen_audio[aid] = 1
            count += 1
        if count != summaries[p]["unique_audio"]:
            raise AssertionError("emitted partition audio cardinality mismatch")
    if not all(seen_audio):
        raise AssertionError("audio omitted from emitted plan")
    seen_rows = bytearray(len(first_ids))
    counts = [0] * len(summaries)
    for _, _, row, _ in iter_rows(output / "row_assignments.jsonl"):
        rid = row["row_index"]
        if not 0 <= rid < len(seen_rows) or seen_rows[rid]:
            raise AssertionError("emitted QA ordinal is duplicated or out of range")
        a, b, p = row["audio1_id"], row["audio2_id"], row["partition_id"]
        if a != first_ids[rid] or b != second_ids[rid] or audio_partition[a] != p or audio_partition[b] != p:
            raise AssertionError("emitted QA has a nonlocal or incorrect audio reference")
        seen_rows[rid] = 1
        counts[p] += 1
    if not all(seen_rows) or counts != [s["qa_rows"] for s in summaries]:
        raise AssertionError("QA omitted from emitted plan or partition counts mismatch")


def write_plan(output: Path, manifest: Path, store: Path, metadata: dict, index_sha: str,
               manifest_sha: str, audio: list[dict], components: list[Component],
               audio_component: list[int], first_ids, second_ids, lengths,
               assignment: list[int], optimizer_report: dict, config: dict):
    # mkdir(exist_ok=False) and BUILDING make incomplete output unambiguous.
    output.mkdir(parents=True, exist_ok=False)
    (output / "BUILDING").write_text("planning; do not consume until partition_audit.json is PASS\n", encoding="utf-8")
    num = config["num_partitions"]
    audio_partition = [assignment[cid] for cid in audio_component]
    summaries = [{"partition_id": p, "qa_rows": 0, "same_audio_rows": 0, "distinct_audio_rows": 0,
                  "unique_audio": 0, "waveform_bytes": 0, "sequence_length_sum": 0,
                  "compute_proxy_sum_length_squared": 0, "source_qa_rows": Counter(),
                  "source_audio_counts": Counter(), "component_count": 0} for p in range(num)]
    expected_row_hashes = [hashlib.sha256() for _ in range(num)]
    for cid, c in enumerate(components):
        summaries[assignment[cid]]["component_count"] += 1
    with ExitStack() as stack:
        audio_files = [stack.enter_context((output / f"partition_{p}_audio.jsonl").open("w", encoding="utf-8")) for p in range(num)]
        for aid, entry in enumerate(audio):
            p = audio_partition[aid]
            s = summaries[p]
            s["unique_audio"] += 1
            s["waveform_bytes"] += entry["byte_length"]
            s["source_audio_counts"][entry.get("source_group", "other")] += 1
            audio_files[p].write(json.dumps({**entry, "partition_id": p,
                                            "component_id": components[audio_component[aid]].component_id}, ensure_ascii=False) + "\n")
        mapping = stack.enter_context((output / "row_assignments.jsonl").open("w", encoding="utf-8"))
        row_files = [stack.enter_context((output / f"partition_{p}_rows.jsonl").open("w", encoding="utf-8")) for p in range(num)]
        count = 0
        for ordinal, line_number, row, line in iter_rows(manifest):
            if ordinal >= len(first_ids):
                raise ValueError("manifest grew during planning")
            a, b, length = first_ids[ordinal], second_ids[ordinal], lengths[ordinal]
            p = audio_partition[a]
            if p != audio_partition[b]:
                raise AssertionError("double-audio QA crosses partition")
            # Keep every original training field untouched; all planner fields
            # live in a sidecar, including original Dataset ordinal.
            emitted_line = line if line.endswith("\n") else line + "\n"
            row_files[p].write(emitted_line)
            expected_row_hashes[p].update(emitted_line.encode("utf-8"))
            mapping.write(json.dumps({"row_index": ordinal, "manifest_line_number": line_number,
                                      "partition_id": p, "component_id": components[audio_component[a]].component_id,
                                      "audio1_id": a, "audio2_id": b}, separators=(",", ":")) + "\n")
            s = summaries[p]
            s["qa_rows"] += 1
            s["same_audio_rows"] += a == b
            s["distinct_audio_rows"] += a != b
            s["sequence_length_sum"] += length
            s["compute_proxy_sum_length_squared"] += length * length
            groups = sorted({str(audio[a].get("source_group", "other")), str(audio[b].get("source_group", "other"))})
            s["source_qa_rows"]["+".join(groups)] += 1
            count += 1
    if count != len(first_ids) or sha256_file(manifest) != manifest_sha:
        raise ValueError("manifest changed during planning")
    if sha256_file(store / "index.jsonl") != index_sha or (store / "BUILDING").exists():
        raise ValueError("store index/build state changed during planning")
    if json.loads((store / "metadata.json").read_text(encoding="utf-8")) != metadata:
        raise ValueError("store metadata changed during planning")
    audit_written_assignments(output, audio_partition, first_ids, second_ids, summaries)
    for p in range(num):
        rows_path = output / f"partition_{p}_rows.jsonl"
        # Text-mode newline translation is platform-dependent; normalize to
        # '\n' exactly as the input iterator did before hashing.
        digest = hashlib.sha256()
        actual_rows = 0
        with rows_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                digest.update(line.encode("utf-8"))
                actual_rows += bool(line.strip())
        if digest.hexdigest() != expected_row_hashes[p].hexdigest() or actual_rows != summaries[p]["qa_rows"]:
            raise AssertionError("partition manifest differs from assigned original QA stream")
    target_rows = count / num
    target_bytes = sum(s["waveform_bytes"] for s in summaries) / num
    target_compute = sum(s["compute_proxy_sum_length_squared"] for s in summaries) / num
    for s in summaries:
        s["waveform_gib"] = s["waveform_bytes"] / 1024**3
        s["qa_relative_deviation"] = s["qa_rows"] / target_rows - 1
        s["waveform_relative_deviation"] = s["waveform_bytes"] / target_bytes - 1
        s["compute_relative_deviation"] = s["compute_proxy_sum_length_squared"] / target_compute - 1
    artifacts = {path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                 for path in sorted(output.glob("*.jsonl"))}
    component_records = [{"component_id": c.component_id, "partition_id": assignment[cid],
                          "unique_audio": len(c.audio_ids), "qa_rows": c.rows,
                          "waveform_bytes": c.waveform_bytes, "compute_proxy": c.compute_proxy,
                          "source_qa_rows": dict(c.source_rows)} for cid, c in enumerate(components)]
    plan = {"format": FORMAT, "config": config, "manifest": str(manifest), "manifest_sha256": manifest_sha,
            "store_dir": str(store), "index_sha256": index_sha,
            "waveform_sha256_from_store_metadata": metadata.get("waveform_sha256"),
            "payload_checksum_reverified": False, "optimizer": optimizer_report,
            "partitions": summaries, "components": component_records, "artifacts": artifacts}
    write_json(output / "partition_plan.json", plan)
    audit = {"status": "PASS", "format": FORMAT, "scope": "planning integrity only; not training/memory/performance PASS",
             "qa_rows": count, "unique_audio": len(audio), "connected_components": len(components),
             "isolated_audio_components": sum(len(c.audio_ids) == 1 for c in components),
             "largest_component_audio_count": max(len(c.audio_ids) for c in components),
             "largest_component_qa_rows": max(c.rows for c in components),
             "same_audio_rows": sum(s["same_audio_rows"] for s in summaries),
             "distinct_audio_rows": sum(s["distinct_audio_rows"] for s in summaries),
             "duplicated_audio": 0, "duplicated_waveform_bytes": 0, "split_components": 0,
             "cross_partition_qa": 0, "all_rows_assigned_exactly_once": True,
             "all_audio_assigned_exactly_once": True, "all_partitions_nonempty": all(s["qa_rows"] for s in summaries),
             "row_balance_within_tolerance": optimizer_report["row_balance_within_tolerance"],
             "balance_status": "PASS" if optimizer_report["row_balance_within_tolerance"] else "NOT_MET",
             "maximum_qa_relative_deviation": max(abs(s["qa_relative_deviation"]) for s in summaries),
             "manifest_sha256": manifest_sha, "partitions": summaries,
             "partition_plan_sha256": sha256_file(output / "partition_plan.json")}
    if not audit["all_partitions_nonempty"]:
        raise AssertionError("empty partition")
    write_json(output / "partition_audit.json", audit)
    (output / "BUILDING").unlink()
    return audit


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    parser.add_argument("--waveform-store-dir", type=Path, default=Path(DEFAULT_STORE))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-partitions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--refine-candidates", type=int, default=4)
    parser.add_argument("--local-rounds", type=int, default=8)
    parser.add_argument("--swap-attempts", type=int, default=4000)
    parser.add_argument("--row-tolerance", type=float, default=0.02)
    parser.add_argument("--waveform-weight", type=float, default=0.05)
    parser.add_argument("--compute-weight", type=float, default=0.05)
    parser.add_argument("--source-weight", type=float, default=0.05)
    parser.add_argument("--tokenizer-path", type=Path, default=Path(DEFAULT_TOKENIZER))
    parser.add_argument("--skip-tokenization", action="store_true", help="stdlib-only character-length proxy instead of exact truncated token lengths")
    parser.add_argument("--token-batch-size", type=int, default=1024)
    args = parser.parse_args(argv)
    for key in ("num_partitions", "trials", "refine_candidates", "local_rounds", "token_batch_size"):
        if getattr(args, key) < (2 if key == "num_partitions" else 1):
            parser.error(f"{key} is too small")
    if args.swap_attempts < 0 or not math.isfinite(args.row_tolerance) or not 0 <= args.row_tolerance < 1:
        parser.error("invalid swap-attempts or row-tolerance")
    for key in ("waveform_weight", "compute_weight", "source_weight"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"invalid {key}")
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    manifest = args.manifest.expanduser().resolve(strict=True)
    store = args.waveform_store_dir.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing plan: {output}")
    manifest_sha = sha256_file(manifest)
    metadata, audio, aliases, index_sha = load_store(store, manifest_sha)
    tokenizer = None
    tokenizer_files = {}
    if not args.skip_tokenization:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers import AutoTokenizer
        tokenizer_dir = args.tokenizer_path.expanduser().resolve(strict=True)
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "config.json"):
            path = tokenizer_dir / name
            if path.is_file():
                tokenizer_files[name] = sha256_file(path)
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    components, audio_component, first, second, lengths = collect_components(
        manifest, audio, aliases, tokenizer, args.token_batch_size)
    if tokenizer is not None:
        for name, digest in tokenizer_files.items():
            if sha256_file(tokenizer_dir / name) != digest:
                raise ValueError("tokenizer changed during planning")
    config = {key: getattr(args, key) for key in ("num_partitions", "seed", "trials", "refine_candidates",
              "local_rounds", "swap_attempts", "row_tolerance", "waveform_weight", "compute_weight", "source_weight")}
    config.update({"compute_measurement": "character_length_proxy" if tokenizer is None else "truncated_token_length_squared_proxy",
                   "tokenizer_path": None if tokenizer is None else str(args.tokenizer_path),
                   "tokenizer_file_sha256": tokenizer_files,
                   "audio_prefix_tokens": 260, "max_prompt_tokens": 129, "max_answer_tokens": 250,
                   "row_index_semantics": "zero-based nonblank manifest record ordinal",
                   "algorithm": "whole connected components; multistart greedy + moves + swaps; no splitting"})
    options = {key: config[key] for key in ("seed", "trials", "refine_candidates", "local_rounds",
               "swap_attempts", "row_tolerance", "waveform_weight", "compute_weight", "source_weight")}
    assignment, optimizer = pack_components(components, args.num_partitions, **options)
    audit = write_plan(output, manifest, store, metadata, index_sha, manifest_sha, audio,
                       components, audio_component, first, second, lengths, assignment, optimizer, config)
    print(json.dumps({"status": audit["status"], "balance_status": audit["balance_status"],
                      "qa_rows": audit["qa_rows"], "unique_audio": audit["unique_audio"],
                      "connected_components": len(components), "duplicated_audio": 0,
                      "partitions": audit["partitions"], "elapsed_seconds": time.perf_counter() - started,
                      "audit": str(output / "partition_audit.json")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
