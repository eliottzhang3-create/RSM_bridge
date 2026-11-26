import argparse
from pathlib import Path
from . import config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Reuse first500 qualifying ReasonAQA caches; 8x5090 patching only')
    p.add_argument('--source-output-dir', type=Path, default=config.SOURCE_OUTPUT)
    p.add_argument('--output-dir', type=Path, default=config.OUTPUT)
    p.add_argument('--check-args', action='store_true')
    a = p.parse_args(argv)
    a.source_output_dir = a.source_output_dir.expanduser().resolve()
    a.output_dir = a.output_dir.expanduser().resolve()
    if (a.output_dir == a.source_output_dir or a.output_dir.is_relative_to(a.source_output_dir)
            or a.source_output_dir.is_relative_to(a.output_dir)):
        p.error('new output must be separate from the read-only source screening directory')
    return a


def forwarded(a):
    return ['--source-output-dir', str(a.source_output_dir), '--output-dir', str(a.output_dir)]
