"""Stdlib-only pre-submit checks; real tokenizer/model loading runs in the job."""
import argparse
import json

from comparison import tokenizer_asset_identity
from contract import recursive_checkpoint_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-model-dir", required=True)
    parser.add_argument("--comparison-text-model-dir", required=True)
    args = parser.parse_args()
    checkpoint = recursive_checkpoint_contract(args.text_model_dir)
    baseline = tokenizer_asset_identity(args.comparison_text_model_dir)
    if checkpoint["config"]["vocab_size"] != baseline["vocab_size"]:
        raise ValueError("MeSH and baseline config vocab_size differ")
    print(json.dumps({"status": "TEXT_ASSETS_PASS", "mesh": checkpoint["config"],
                      "comparison_tokenizer": baseline["root"]}), flush=True)


if __name__ == "__main__":
    main()
