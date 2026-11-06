"""CPU-only, stdlib artifact audit before GPU submission; no tensor library import."""
import argparse
import json
from pathlib import Path

from contract import artifact_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    identity = artifact_identity(args.model_dir)
    print(json.dumps({"status": "PASS", "scope": "artifact_files_only", "tensor_audit": "NOT_RUN",
                      "model_dir": str(args.model_dir), "manifest_sha256": identity["manifest_sha256"]}))


if __name__ == "__main__":
    main()
