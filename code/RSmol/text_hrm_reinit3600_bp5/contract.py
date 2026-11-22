"""Separate training identity; the original HRM architecture/weights stay intact."""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

from text_hrm_h2l3.contract import (ARCHITECTURE, REMOTE_HOME, REPO, DEFAULT_DATA,
    canonical_hash, sha256, verify_inventory, require_external, validate_config,
    file_inventory as original_inventory, RECIPE as ORIGINAL_RECIPE)

HERE = Path(__file__).resolve().parent
ROUTE = 'hrm_text_reinit3600_bp5_seed1_cosine_v1'
FORMAT = 'hrm_text_reinit3600_full_checkpoint_v1'
SOURCE_STEP = 3600
DEFAULT_SOURCE = REMOTE_HOME / 'outputs/RSmol/text_hrm_h2l3/formal_20261122_000344_4005340/checkpoint-step-003600'
DEFAULT_INIT = REMOTE_HOME / 'models/SmolLM2-HRM-H2L3-reinit3600-raw'
DEFAULT_OUTPUT = REMOTE_HOME / 'outputs/RSmol/text_hrm_reinit3600_bp5'
RECIPE = copy.deepcopy(ORIGINAL_RECIPE)
RECIPE.update(seed=0, data_seed=1, lr=6e-4, lr_start=3e-4, lr_min=6e-5,
    lr_warmup_steps=462, lr_warmup_ratio=0.05,
    lr_policy='first_update_start_linear_inclusive_then_cosine_to_final_1based',
    bp_warmup_steps=0, bp_min=5, bp_max=5, initialization='raw_step3600_weights_only',
    prelude_audit='per_layer_first_micro_and_first_full_update_before_optimizer')


def write_json(path, value):
    import json
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',
                          encoding='utf-8',newline='\n')


def _update(update):
    if type(update) is not int or not 1 <= update <= 9244:
        raise ValueError('update must be in 1..9244')


def learning_rate(update):
    _update(update)
    if update <= 462:
        return 3e-4 + (6e-4 - 3e-4) * ((update - 1) / 461)
    return 6e-5 + (6e-4 - 6e-5) * 0.5 * (1 + math.cos(math.pi * (update - 462) / 8782))


def bp_steps(update):
    _update(update)
    return 5


def file_inventory(root, *, include_state=False):
    result = original_inventory(root,include_state=include_state)
    result.pop('initialization_complete.json',None)
    return result


def code_identity():
    files = sorted(HERE.glob('*.py')) + sorted((HERE.parent/'text_hrm_h2l3').glob('*.py'))
    files += [HERE.parent/'scripts/prepare_text_hrm_reinit3600_bp5.sh',
              HERE.parent/'scripts/train_text_hrm_reinit3600_bp5_ddp.sh',
              HERE.parent/'run_text_hrm_reinit3600_bp5_3090.sh']
    return {p.relative_to(REPO).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in files}
