"""CPU JSON/complete-marker check; no torch, pickle, CUDA or network access."""
import argparse
import json
from contract import recursive_checkpoint_contract

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-model-dir", required=True)
    args = parser.parse_args()
    print(json.dumps({"event": "TEXT_CHECKPOINT_METADATA_PASS",
                      "checkpoint": recursive_checkpoint_contract(args.text_model_dir)},
                     ensure_ascii=False), flush=True)
