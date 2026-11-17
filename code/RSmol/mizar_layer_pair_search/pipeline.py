"""One scheduled task: create/reuse manifest, cache prefixes and search all masks."""
from .evaluate import main

if __name__ == '__main__':
    raise SystemExit(main(prepare_manifest=True))
