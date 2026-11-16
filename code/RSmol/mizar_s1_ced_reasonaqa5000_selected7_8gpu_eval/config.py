"""Frozen supplemental seven targets; official_mizar is always excluded."""
import json
from pathlib import Path

from mizar_s1_ced_reasonaqa5000_eval.config import ROOT, REMOTE, DATA, DEFAULT_MANIFEST, SAMPLE_SIZE
from mizar_s1_ced_reasonaqa5000_8gpu_eval.config import (
    WORLD_SIZE, ROWS_PER_RANK, OUTPUT_NAME, PARALLEL, rank_indices, rank_dir,
)

PACKAGE = Path(__file__).resolve().parent
EXCLUDED = ('official_mizar',)
TARGETS = json.loads((PACKAGE / 'targets.json').read_text(encoding='utf-8'))
for _spec in TARGETS.values():
    _spec['route_root'] = ROOT / _spec['route_root']
    _spec['run_root'] = Path(_spec['run_root'])
    _spec['expected_geometry'] = tuple(_spec['expected_geometry'])
VARIANT_TARGETS = {spec['eval_variant']: name for name, spec in TARGETS.items()}


def target_name(value):
    if value in TARGETS:
        return value
    if value in VARIANT_TARGETS:
        return VARIANT_TARGETS[value]
    raise ValueError(f'unsupported/excluded supplemental ReasonAQA5000 target: {value}')
