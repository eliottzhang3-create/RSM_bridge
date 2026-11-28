"""Build/reuse an immutable path index on the login CPU before requesting GPUs."""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from contract import code_digest, digest, write_json, sha256_file
from data.prepared_index import (INDEX_SCHEMA_VERSION, check_input_contents, input_contents,
                                 input_identity, load_index)
from stage_data import load_resolutions, prepare
from publication import atomic_publish


def tokenizer_identity(root):
    return {p.name:sha256_file(p) for p in sorted(Path(root).glob("*"))
            if p.is_file() and p.suffix in {".json",".txt"}}


def audit_text(metadata,tokenizer_path):
    from transformers import AutoTokenizer
    from data.audiotext_dataset import text_pair
    from contract import PROMPT_TOKENS,ANSWER_TOKENS,validate_checkpoint_tokenizer
    from manifest_io import iter_rows
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    validate_checkpoint_tokenizer(tokenizer,49152)
    max_prompt=max_answer=count=0
    for index,row in enumerate(iter_rows(metadata)):
        prompt,answer=text_pair(row)
        p=tokenizer.encode(prompt,add_special_tokens=True)
        a=tokenizer.encode(answer+" <|endoftext|>",add_special_tokens=True)
        if len(p)>PROMPT_TOKENS or len(a)>ANSWER_TOKENS or not a or a[-1]!=0:
            raise ValueError(f"S3 text cannot fit without truncation: row={index} prompt={len(p)} answer={len(a)}")
        max_prompt=max(max_prompt,len(p));max_answer=max(max_answer,len(a));count+=1
        if count%10000==0:print(f"[cpu-index] S3 text checked={count:,}",flush=True)
    return {"status":"PASS","rows":count,"max_prompt_tokens":max_prompt,"max_answer_eos_tokens":max_answer,
            "prompt_eos":False,"answer_eos":True,"truncations":0,"tokenizer_files":tokenizer_identity(tokenizer_path)}


def ensure_index(manifest, resolutions, cache_root, tokenizer_path):
    from prepare_manifest import verify_dataset
    dataset = verify_dataset(Path(manifest).parent)
    if Path(manifest).resolve() != (Path(manifest).parent/"train.json").resolve() or Path(resolutions).resolve() != (Path(manifest).parent/"resolved_paths.tsv").resolve():
        raise ValueError("use the sealed S3 train.json and its sibling resolved_paths.tsv")
    from contract import SOURCE_COUNTS
    expected_counts = dataset.get("source_counts",SOURCE_COUNTS)
    inputs = input_identity(manifest, resolutions)
    signature = digest({"inputs": input_contents(inputs), "code_sha256": code_digest(),
                        "dataset_seal_sha256":dataset["seal_sha256"],
                        "tokenizer":tokenizer_identity(tokenizer_path)})
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / signature
    if destination.exists():
        report = load_index(destination)
        check_input_contents(report["inputs"], inputs)
        print(f"[cpu-index] reusing {destination}", flush=True)
        return destination
    start = time.monotonic()
    print("[cpu-index] preparing metadata on CPU before GPU submission; no audio decode/copy", flush=True)
    with tempfile.TemporaryDirectory(prefix=".building-", dir=cache_root) as directory:
        temporary = Path(directory)
        report, files = prepare(manifest, load_resolutions(resolutions), temporary / "rows.jsonl", expected_counts=expected_counts)
        report["text_audit"] = audit_text(temporary/"rows.jsonl",tokenizer_path)
        report["dataset_selection"] = {"seal_sha256":dataset["seal_sha256"],
            "historical_inputs":dataset["inputs"],"selected_counts":dataset["selected_counts"]}
        write_json(temporary / "files.json", files)
        check_input_contents(inputs, input_identity(manifest, resolutions), "during CPU index preparation")
        report.update(schema_version=INDEX_SCHEMA_VERSION, status="PASS", code_sha256=code_digest(), inputs=inputs,
                      preparation_seconds=time.monotonic() - start,
                      artifact_bytes={name: (temporary / name).stat().st_size for name in
                                      ("rows.jsonl", "rows.jsonl.idx", "files.json")})
        # Absolute build-directory paths must not survive the atomic rename.
        report["artifact_sha256"] = {name: sha256_file(temporary / name) for name in report["artifact_bytes"]}
        report.pop("metadata")
        write_json(temporary / "index_report.json", report)
        try:
            atomic_publish(temporary, destination)
        except FileExistsError:
            # Another submission may have finished the same immutable index.
            if not destination.exists():
                raise
        report = load_index(destination)
        check_input_contents(report["inputs"], inputs)
    print(f"[cpu-index] ready in {time.monotonic() - start:.1f}s: {destination}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--resolved-paths", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--path-output", required=True, type=Path)
    parser.add_argument("--tokenizer-path",required=True,type=Path)
    args = parser.parse_args()
    path = ensure_index(args.manifest, args.resolved_paths, args.cache_root,args.tokenizer_path)
    args.path_output.parent.mkdir(parents=True, exist_ok=True)
    args.path_output.write_text(str(path) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
