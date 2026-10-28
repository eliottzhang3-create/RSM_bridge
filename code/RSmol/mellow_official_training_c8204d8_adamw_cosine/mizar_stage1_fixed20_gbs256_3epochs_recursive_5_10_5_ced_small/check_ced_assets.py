"""Check local CED files/config using stdlib before requesting GPU resources."""
import argparse
import json
from pathlib import Path

from contract import CED_ASSET_FILES, validate_ced_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ced-model-dir", required=True, type=Path)
    args = parser.parse_args()
    geometry = validate_ced_assets(args.ced_model_dir)
    print(json.dumps({"status": "ASSET_CONFIG_PASS", "ced_model_dir": str(args.ced_model_dir.resolve()),
                      "files": list(CED_ASSET_FILES), "geometry": geometry,
                      "scope": "file/config validation only; no tensor loading or inference"}), flush=True)


if __name__ == "__main__":
    main()
