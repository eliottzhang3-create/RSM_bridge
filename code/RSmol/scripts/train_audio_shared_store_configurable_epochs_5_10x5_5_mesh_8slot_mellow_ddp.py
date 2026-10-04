#!/usr/bin/env python3
"""Formal-only configurable-epoch entry for the isolated x5/8-slot route."""
from __future__ import annotations

import train_audio_shared_store_5_10x5_5_mesh_8slot_mellow_ddp as shared_store


def main() -> None:
    args = shared_store.parse_args()
    if args.mode != "formal":
        raise ValueError("x5/8-slot configurable route supports formal mode only")
    if args.epochs is None:
        raise ValueError("formal training requires explicit --epochs")
    args.epochs = int(args.epochs)
    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer")

    # Keep the common trainer's audited horizon calculation while making the
    # requested formal epoch count explicit in the route's checkpoint contract.
    shared_store.FORMAL_EPOCHS = args.epochs
    shared_store.run(args)


if __name__ == "__main__":
    main()
