"""Evaluate an existing manifest inside an allocated single-GPU job."""
from .evaluate import main

if __name__ == '__main__':
    raise SystemExit(main(world_size=1))
