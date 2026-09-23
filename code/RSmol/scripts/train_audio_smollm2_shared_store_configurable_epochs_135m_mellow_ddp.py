#!/usr/bin/env python3
"""Explicit configurable-epoch entry for SmolLM2 shared-store training."""
from __future__ import annotations

import train_audio_smollm2_shared_store_135m_mellow_ddp as shared_store


def main() -> None:
    args = shared_store.parse_args()
    if args.epochs is None:
        raise ValueError(
            "configurable SmolLM2 shared-store training requires explicit --epochs"
        )
    args.epochs = int(args.epochs)
    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer")
    shared_store.FORMAL_EPOCHS = args.epochs
    shared_store.run(args)


if __name__ == "__main__":
    main()
