"""Build/reuse an immutable path index on the login CPU before requesting GPUs."""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from contract import code_digest, digest, write_json
from data.prepared_index import input_identity, load_index
from stage_data import load_resolutions, prepare


def ensure_index(manifest, resolutions, cache_root):
    inputs = input_identity(manifest, resolutions)
    signature = digest({"inputs": inputs, "code_sha256": code_digest()})
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / signature
    if destination.exists():
        load_index(destination, manifest, resolutions)
        print(f"[cpu-index] reusing {destination}", flush=True)
        return destination
    start = time.monotonic()
    print("[cpu-index] preparing metadata on CPU before GPU submission; no audio decode/copy", flush=True)
    with tempfile.TemporaryDirectory(prefix=".building-", dir=cache_root) as directory:
        temporary = Path(directory)
        report, files = prepare(manifest, load_resolutions(resolutions), temporary / "rows.jsonl")
        write_json(temporary / "files.json", files)
        if inputs != input_identity(manifest, resolutions):
            raise RuntimeError("manifest/audit TSV changed during CPU index preparation")
        report.update(schema_version=1, status="PASS", code_sha256=code_digest(), inputs=inputs,
                      preparation_seconds=time.monotonic() - start,
                      artifact_bytes={name: (temporary / name).stat().st_size for name in
                                      ("rows.jsonl", "rows.jsonl.idx", "files.json")})
        # Absolute build-directory paths must not survive the atomic rename.
        report.pop("metadata")
        write_json(temporary / "index_report.json", report)
        try:
            os.rename(temporary, destination)
        except OSError:
            # Another submission may have finished the same immutable index.
            if not destination.exists():
                raise
        load_index(destination, manifest, resolutions)
    print(f"[cpu-index] ready in {time.monotonic() - start:.1f}s: {destination}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--resolved-paths", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--path-output", required=True, type=Path)
    args = parser.parse_args()
    path = ensure_index(args.manifest, args.resolved_paths, args.cache_root)
    args.path_output.parent.mkdir(parents=True, exist_ok=True)
    args.path_output.write_text(str(path) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
