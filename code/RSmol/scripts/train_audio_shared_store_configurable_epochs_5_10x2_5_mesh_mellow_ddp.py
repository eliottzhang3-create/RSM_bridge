#!/usr/bin/env python3
"""Configurable-epoch entry for the isolated fixed-260 shared-store trainer.

The audited trainer historically fixes its scheduler horizon to three epochs.
This entry changes only that horizon: ``--epochs`` is mandatory, must be a
positive integer, and becomes the exact horizon checked by smoke, resume, and
formal runs.  All model, data, checkpoint, resume, and gate logic remains in
the original shared-store trainer.
"""
from __future__ import annotations

import train_audio_shared_store_5_10x2_5_mesh_mellow_ddp as shared_store


def main() -> None:
    args = shared_store.parse_args()
    if args.epochs is None:
        raise ValueError("configurable shared-store training requires explicit --epochs")
    args.epochs = int(args.epochs)
    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer")

    # The original trainer deliberately uses this module constant in both the
    # training-shape calculation and formal smoke gate.  Setting it before
    # run() makes the requested horizon part of every report/checkpoint/resume
    # comparison without changing any other audited behavior.
    shared_store.FORMAL_EPOCHS = args.epochs
    shared_store.run(args)


if __name__ == "__main__":
    main()
