"""Validate the complete local USAD2-Small package using only the stdlib."""
import argparse
import json
from pathlib import Path

from contract import validate_usad2_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usad2-model-dir", required=True, type=Path)
    args = parser.parse_args()
    result = validate_usad2_assets(args.usad2_model_dir)
    print(json.dumps({"status": "ASSET_CONFIG_PASS", **result,
                      "scope": "files/config only; full safetensors load and GPU forward occur in the training job"}),
          flush=True)


if __name__ == "__main__":
    main()
