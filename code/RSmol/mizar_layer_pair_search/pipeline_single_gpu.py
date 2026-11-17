"""Prepare/reuse the manifest and evaluate all masks on one GPU."""
from .evaluate import main

if __name__ == '__main__':
    raise SystemExit(main(prepare_manifest=True, world_size=1))
