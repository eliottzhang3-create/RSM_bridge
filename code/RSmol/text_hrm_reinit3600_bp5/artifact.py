"""Same model config, distinct initialization/full-resume artifact contracts."""
from __future__ import annotations
import json
from pathlib import Path

from .contract import (ROUTE, FORMAT, ARCHITECTURE, SOURCE_STEP, RECIPE, validate_config,
                       verify_inventory, file_inventory, canonical_hash)


def inspect_artifact(path, *, resume=False, production=True):
    root = Path(path).resolve()
    validate_config(json.loads((root/'config.json').read_text(encoding='utf-8')),production=production)
    name = 'checkpoint_complete.json' if resume else 'initialization_complete.json'
    if not (root/name).is_file():
        raise ValueError(f'missing required reinit3600 artifact marker: {name}')
    marker = json.loads((root/name).read_text(encoding='utf-8'))
    if (marker.get('complete') is not True or marker.get('route') != ROUTE
        or marker.get('architecture') != ARCHITECTURE or marker.get('production') is not production):
        raise ValueError('reinit3600 artifact identity/completion mismatch')
    if resume:
        step = marker.get('step')
        if (marker.get('format') != FORMAT or type(step) is not int or not 1 <= step <= 9244
            or root.name != f'checkpoint-step-{step:06d}' or marker.get('world_size') != 8):
            raise ValueError('reinit3600 checkpoint format/step/world mismatch')
        if not (root/'training_state.pt').is_file():
            raise ValueError('missing reinit3600 full training state')
        binding = json.loads((root/'binding.json').read_text(encoding='utf-8'))
        if (binding.get('route') != ROUTE or binding.get('architecture') != ARCHITECTURE
            or binding.get('recipe') != RECIPE or binding.get('world_size') != 8
            or binding.get('accumulation_steps') != 64 or binding.get('production') is not production
            or canonical_hash(binding) != marker.get('binding_sha256')):
            raise ValueError('reinit3600 binding/recipe mismatch')
    else:
        if marker.get('source_step') != SOURCE_STEP or marker.get('weights_kind') != 'raw':
            raise ValueError('requires raw step3600 initialization snapshot')
        if (root/'training_state.pt').exists() or (root/'ema').exists():
            raise ValueError('initialization snapshot must contain raw weights only')
    verify_inventory(root,marker.get('files'))
    if file_inventory(root,include_state=resume) != marker['files']:
        raise ValueError('reinit3600 artifact file inventory changed')
    return marker
