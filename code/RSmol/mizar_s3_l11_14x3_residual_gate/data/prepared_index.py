"""Reusable CPU-built metadata; opening it never rescans the raw audio tree."""
import hashlib
import json
from pathlib import Path

from contract import SOURCE_COUNTS, code_digest, validate_data_counts

INDEX_SCHEMA_VERSION = 2
INPUT_NAMES = ("manifest", "resolved_paths")


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def input_identity(manifest, resolutions):
    result = {}
    for name, value in (("manifest", manifest), ("resolved_paths", resolutions)):
        path = Path(value).resolve()
        stat = path.stat()
        result[name] = {"path": str(path), "bytes": stat.st_size,
                        "sha256": file_sha256(path)}
    return result


def input_contents(inputs):
    """Paths identify provenance; only bytes and SHA256 identify contents."""
    try:
        return {name: {key: inputs[name][key] for key in ("bytes", "sha256")}
                for name in INPUT_NAMES}
    except (KeyError, TypeError) as exc:
        raise ValueError("CPU index lacks content fingerprints; rerun the submission command") from exc


def check_input_contents(expected, current, when="since CPU index preparation"):
    expected_contents, current_contents = input_contents(expected), input_contents(current)
    differences = {name: {"expected": expected[name], "current": current[name]}
                   for name in INPUT_NAMES if expected_contents[name] != current_contents[name]}
    if differences:
        raise ValueError(f"manifest/audit TSV content changed {when}: "
                         + json.dumps(differences, ensure_ascii=True, sort_keys=True))


def load_index(directory, manifest=None, resolutions=None):
    directory = Path(directory).resolve()
    report = json.loads((directory / "index_report.json").read_text(encoding="utf-8"))
    if (report.get("schema_version") != INDEX_SCHEMA_VERSION or report.get("status") != "PASS"
            or report.get("code_sha256") != code_digest()):
        raise ValueError("CPU index is incomplete or belongs to different source code; rerun the submission command")
    validate_data_counts(report["rows"],report["source_rows"],report["dataset_selection"]["selected_counts"])
    audit=report.get("text_audit",{})
    if (audit.get("status")!="PASS" or audit.get("rows")!=report["rows"]
            or audit.get("truncations")!=0 or audit.get("prompt_eos") is not False
            or audit.get("answer_eos") is not True
            or not 0<audit.get("max_prompt_tokens",0)<=256
            or not 0<audit.get("max_answer_eos_tokens",0)<=96 or not audit.get("tokenizer_files")):
        raise ValueError("CPU index lacks the complete S3 token/EOS audit")
    if set(report["artifact_bytes"]) != {"rows.jsonl","rows.jsonl.idx","files.json"} or set(report["artifact_sha256"])!=set(report["artifact_bytes"]):
        raise ValueError("CPU index artifact inventory changed")
    if manifest is not None:
        print("[cpu-index] verifying manifest/audit TSV SHA256; no raw audio scan", flush=True)
        current = input_identity(manifest, resolutions)
        check_input_contents(report["inputs"], current)
        aliases = {name: {"prepared": report["inputs"][name]["path"], "current": current[name]["path"]}
                   for name in INPUT_NAMES if report["inputs"][name]["path"] != current[name]["path"]}
        if aliases:
            print("[cpu-index] input path aliases accepted with identical contents: "
                  + json.dumps(aliases, ensure_ascii=True, sort_keys=True), flush=True)
        print("[cpu-index] manifest/audit TSV contents verified", flush=True)
    for name, size in report["artifact_bytes"].items():
        if (Path(name).name != name or (directory / name).stat().st_size != size
                or file_sha256(directory / name) != report["artifact_sha256"][name]):
            raise ValueError(f"CPU index artifact is missing/changed: {name}")
    report["metadata"] = str(directory / "rows.jsonl")
    report["files_path"] = str(directory / "files.json")
    report["index_dir"] = str(directory)
    return report
