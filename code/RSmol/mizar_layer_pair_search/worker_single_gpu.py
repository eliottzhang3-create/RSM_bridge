"""One torchrun worker scores all 100 records for every mask."""
from .worker import main

if __name__ == '__main__':
    raise SystemExit(main(world_size=1))
