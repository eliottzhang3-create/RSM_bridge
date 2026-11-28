"""Dependency-free official release verification before requesting a GPU."""
import json
from pathlib import Path
from contract import sha256_file


def audit_bundle(model_dir):
    root = Path(model_dir).resolve()
    release = json.loads((Path(__file__).parent / "release_manifest.json").read_text(encoding="utf-8"))
    records = []
    for entry in [release["checkpoint"], *release["files"]]:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"official release missing or changed: {path}")
        records.append({"path": entry["path"], "bytes": entry["bytes"], "sha256": entry["sha256"]})
    return {"model_dir": str(root), "files": records,
            "release_manifest_sha256": sha256_file(Path(__file__).parent / "release_manifest.json")}


def verify_bundle(bundle):
    root = Path(bundle["model_dir"])
    for entry in bundle["files"]:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"official release changed after admission: {path}")


if __name__ == "__main__":
    import argparse
    from contract import write_json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    args = p.parse_args()
    write_json(args.report, audit_bundle(args.model_dir))
    print(f"[assets] official full release verified: {args.model_dir}")
