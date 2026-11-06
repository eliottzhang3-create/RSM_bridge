"""Read-only release and published-artifact checks; imports no tensor libraries."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from .assets import validate_ced_assets
from .contract import (RELEASE_FILE, ROUTE, SCHEMA, TOKENIZER_FILES, architecture,
                       implementation_identity, sha256_file, training_intent,
                       validate_config_preservation, validate_source_text_config, validate_text_config)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def release_manifest():
    return read_json(RELEASE_FILE)


def checked_path(root, relative):
    """Only regular, contained, portable relative paths from manifests."""
    root = Path(root).resolve()
    part = PurePosixPath(relative)
    if (not relative or "\\" in relative or ":" in relative or part.is_absolute()
            or any(x in ("", ".", "..") for x in relative.split("/"))):
        raise ValueError(f"unsafe manifest path: {relative}")
    value = root.joinpath(*part.parts)
    if not value.resolve().is_relative_to(root) or value.is_symlink():
        raise ValueError(f"manifest path escapes root or is a symlink: {relative}")
    return value


def verify_files(root, entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest needs a nonempty file list")
    seen = set()
    for entry in entries:
        relative = entry["path"]
        if relative in seen:
            raise ValueError(f"duplicate manifest path: {relative}")
        seen.add(relative)
        path = checked_path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"missing required file: {path}")
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"SHA256/size mismatch: {path}")
    return sorted(seen)


def file_record(root, path):
    path = Path(path)
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def audit_source(source):
    root, release = Path(source).resolve(), release_manifest()
    entries = [release["checkpoint"], *release["files"]]
    files = verify_files(root, entries)
    validate_source_text_config(read_json(root / "assets/smollm2-135m/config.json"))
    ced = validate_ced_assets(root / "assets/ced-small")
    return {"status": "PASS", "scope": "source_files_only", "tensor_audit": "NOT_RUN",
            "source": str(root), "release": release["release"], "verified_files": entries,
            "verified_file_count": len(files), "ced_geometry": ced,
            "release_manifest_sha256": sha256_file(RELEASE_FILE)}


def validate_output_location(source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output: {output}")
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("output and official source must be separate trees")
    if any((parent / ".git").exists() for parent in (output, *output.parents)):
        raise ValueError("model output must be outside a Git checkout")
    return output


def artifact_asset_records():
    """The small constructor/tokenizer bundle is copied exactly, not regenerated."""
    result = []
    for entry in release_manifest()["files"]:
        relative = entry["path"]
        if relative.startswith("assets/ced-small/"):
            result.append({**entry, "path": relative.replace("assets/", "", 1)})
        elif relative.startswith("assets/smollm2-135m/") and relative.rsplit("/", 1)[1] in TOKENIZER_FILES:
            result.append({**entry, "path": relative.replace("assets/smollm2-135m/", "tokenizer/", 1)})
    return result


def audit_artifact(directory, *, _pending_reload=False):
    root = Path(directory).resolve()
    manifest = read_json(root / "manifest.json")
    if (manifest.get("schema") != SCHEMA or manifest.get("route") != ROUTE
            or manifest.get("architecture") != architecture()
            or manifest.get("training_intent") != training_intent()
            or manifest.get("implementation") != implementation_identity()
            or manifest.get("source_release_manifest_sha256") != sha256_file(RELEASE_FILE)):
        raise ValueError("artifact schema/route/architecture/implementation/source identity mismatch")
    files = verify_files(root, manifest["files"])
    expected_assets = artifact_asset_records()
    expected_names = {x["path"] for x in expected_assets} | {
        "model.ckpt", "config.json", "source_audit.json", "conversion_report.json", "LICENSE", "NOTICE",
        "licenses/Mizar-ClearBSD.txt"}
    if set(files) != expected_names:
        raise ValueError("artifact inventory differs from the complete model-only contract")
    actual_names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual_names != expected_names | {"manifest.json"}:
        raise ValueError("artifact contains unlisted files (training state is not allowed)")
    verify_files(root, expected_assets)
    validate_text_config(read_json(root / "config.json"))
    validate_config_preservation(read_json(root / "config.json"), read_json(root / "tokenizer/config.json"))
    report = read_json(root / "conversion_report.json")
    source = read_json(root / "source_audit.json")
    if (report.get("status") != "PASS" or report.get("scope") != "full_conversion"
            or source.get("scope") != "source_files_only" or source.get("status") != "PASS"
            or source.get("verified_files") != [release_manifest()["checkpoint"], *release_manifest()["files"]]
            or source.get("release_manifest_sha256") != sha256_file(RELEASE_FILE)):
        raise ValueError("artifact is missing a successful conversion/source audit")
    if not _pending_reload and (not isinstance(report.get("reload"), dict)
                                or report["reload"].get("status") != "PASS"):
        raise ValueError("artifact has not completed mandatory tensor/logits reload verification")
    return {"status": "PASS", "scope": "artifact_files_only", "tensor_audit": "NOT_RUN",
            "directory": str(root), "verified_file_count": len(files), "manifest": manifest}
