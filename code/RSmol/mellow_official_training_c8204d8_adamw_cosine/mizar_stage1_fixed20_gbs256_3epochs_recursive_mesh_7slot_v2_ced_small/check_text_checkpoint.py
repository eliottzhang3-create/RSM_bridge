"""CPU-only pre-submit check for the explicit final text checkpoint."""
import argparse
import json

from contract import recursive_checkpoint_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-model-dir", required=True)
    args = parser.parse_args()
    result = recursive_checkpoint_contract(args.text_model_dir)
    print(json.dumps({"status": "PASS_METADATA_ONLY", "text_checkpoint": result}, indent=2))


if __name__ == "__main__":
    main()
