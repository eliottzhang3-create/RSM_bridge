"""Frozen checkpoint bindings; this package adds no model or generation code."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
TARGETS = json.loads((PACKAGE / 'targets.json').read_text(encoding='utf-8'))
EXCLUDED = tuple(json.loads((PACKAGE / 'excluded.json').read_text(encoding='utf-8'))['eval_variants'])
for _spec in TARGETS.values():
    _spec['route_root'] = ROOT / _spec['route_root']
    _spec['run_root'] = Path(_spec['run_root'])
    _spec['expected_geometry'] = tuple(_spec['expected_geometry'])
VARIANT_TARGETS = {spec['eval_variant']: name for name, spec in TARGETS.items()}
DEFAULT_LEDGER = ROOT / 'log/mmsu_remaining4090_admissions'
DEFAULT_SUMMARY = Path('/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mmsu_remaining27_4090_v2.csv')


def target_name(value):
    if value in TARGETS:
        return value
    if value in VARIANT_TARGETS:
        return VARIANT_TARGETS[value]
    raise ValueError(f'unsupported or already-completed MMSU target: {value}')
