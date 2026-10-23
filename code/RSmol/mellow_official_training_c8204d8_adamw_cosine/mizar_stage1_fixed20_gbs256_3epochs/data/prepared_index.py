"""Reusable CPU-built metadata; opening it never rescans the raw audio tree."""
import json
from pathlib import Path

from contract import SOURCE_COUNTS, code_digest


def input_identity(manifest, resolutions):
    result = {}
    for name, value in (("manifest", manifest), ("resolved_paths", resolutions)):
        path = Path(value).resolve()
        stat = path.stat()
        result[name] = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def load_index(directory, manifest=None, resolutions=None):
    directory = Path(directory).resolve()
    report = json.loads((directory / "index_report.json").read_text(encoding="utf-8"))
    if (report.get("schema_version") != 1 or report.get("status") != "PASS"
            or report.get("code_sha256") != code_digest()):
        raise ValueError("CPU index is incomplete or belongs to different source code; rerun the submission command")
    if report["source_rows"] != SOURCE_COUNTS or report["rows"] != sum(SOURCE_COUNTS.values()):
        raise ValueError("CPU index row/source counts differ from the fixed manifest")
    if manifest is not None and report["inputs"] != input_identity(manifest, resolutions):
        raise ValueError("manifest/audit TSV changed since CPU index preparation")
    for name, size in report["artifact_bytes"].items():
        if Path(name).name != name or (directory / name).stat().st_size != size:
            raise ValueError(f"CPU index artifact is missing/changed: {name}")
    report["metadata"] = str(directory / "rows.jsonl")
    report["files_path"] = str(directory / "files.json")
    report["index_dir"] = str(directory)
    return report
