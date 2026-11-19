"""Dependency-light, isolated architecture and official-source identity."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUTE = 'mizar_final_text_7_14x3_7_mesh_6slot_v1'
ARCHITECTURE = 'logical_56_physical_28_7_14x3_7_mesh_6slot_4router_v1'
LOGICAL_LAYER_COUNT, PHYSICAL_LAYER_COUNT = 56, 28
PREFIX_LAYER_COUNT, MIDDLE_LAYER_COUNT, SUFFIX_LAYER_COUNT = 7, 14, 7
RECURSIVE_LOOPS, MEMORY_SLOT_COUNT, ROUTER_COUNT = 3, 6, 4
ROUTER_PARAMETER_COUNT = 27696
SOURCE_LAYERS = tuple(i for i in range(30) if i not in (2, 17))
SCHEDULE = tuple(range(7)) + tuple(range(7, 21)) * RECURSIVE_LOOPS + tuple(range(21, 28))
ROUTER_KEYS = frozenset(f'model.{kind}_routers.{group}.{part}'
                        for kind in ('write', 'read') for group in range(ROUTER_COUNT)
                        for part in ('weight', 'bias'))
ROUTER_INITIALIZATION = {
    'distribution': 'torch.nn.init.trunc_normal_', 'mean': 0.0,
    'std': 'sqrt(2/(6*hidden_size))', 'bounds': '[-3*std,+3*std]',
    'bias': 0.0, 'depth_scaling': False,
}
DEFAULT_SOURCE = Path('/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M')
DEFAULT_MODEL = DEFAULT_SOURCE.with_name('Mizar159M-text-7-14x3-7-mesh-6slot')
DEFAULT_OUTPUT = Path('/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_text_7_14x3_7_mesh')
DEFAULT_TRAINING_MODEL = DEFAULT_OUTPUT / 'formal_20261118_182750_1375514/checkpoint-001200'
LM_PREFIX = 'caption_decoder.lm.'


def release_manifest():
    return json.loads((HERE.parent / 'mizar_official_eval/release_manifest.json').read_text(encoding='utf-8'))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_identity():
    release = release_manifest()
    return {'checkpoint_relative_path': release['checkpoint']['path'],
            'checkpoint_sha256': release['checkpoint']['sha256'],
            'release': copy.deepcopy(release['release']), 'lm_prefix': LM_PREFIX}


def architecture_fields():
    fields = {
        'num_hidden_layers': LOGICAL_LAYER_COUNT, 'recursive_layer_count': 28, 'recursive_loops': RECURSIVE_LOOPS,
        'recursive_loops_scope': 'middle_only', 'recursive_prefix_layer_count': 7,
        'recursive_middle_layer_count': 14, 'recursive_suffix_layer_count': 7,
        'middle_recurrent_count': 14, 'memory_slots': MEMORY_SLOT_COUNT, 'router_groups': ROUTER_COUNT,
        'recursive_mapping_policy': 'mizar_final_drop_1based_3_18_keep_order',
        'recursive_prefix_source_layers_1based': [i + 1 for i in SOURCE_LAYERS[:7]],
        'recursive_middle_source_layers_1based': [i + 1 for i in SOURCE_LAYERS[7:21]],
        'recursive_suffix_source_layers_1based': [i + 1 for i in SOURCE_LAYERS[21:]],
        'deleted_source_layers_1based': [3, 18],
        'architectures': ['MizarText7_14x3_7MeshForCausalLM'], 'text_route': ROUTE,
        'architecture_contract': ARCHITECTURE, 'mizar_text_source': source_identity(),
        'router_initialization': copy.deepcopy(ROUTER_INITIALIZATION),
        'transition_router_query': 'prefix_output', 'loop_router_query': 'core_input',
        'embedding_scale': 'disabled',
    }
    for name in ('logical_to_physical', 'recursive_logical_to_physical',
                 'logical_to_physical_schedule', 'recursive_logical_to_physical_schedule'):
        fields[name] = list(SCHEDULE)
    for name in ('recursive_source_layer_indices_0based', 'recursive_source_mapping_0based'):
        fields[name] = list(SOURCE_LAYERS)
    for name in ('recursive_source_layer_indices_1based', 'recursive_source_mapping_1based'):
        fields[name] = [i + 1 for i in SOURCE_LAYERS]
    return fields


def build_target_config(source_config):
    if source_config.num_hidden_layers != 30:
        raise ValueError('Mizar source must have all 30 original layers')
    result = copy.deepcopy(source_config)
    for name, value in architecture_fields().items():
        setattr(result, name, value)
    result.use_cache = False
    return result


def validate_model_config(raw, *, production=True):
    expected = architecture_fields()
    if production:
        expected.update(hidden_size=576, vocab_size=49152, tie_word_embeddings=True, model_type='llama')
    for name, value in expected.items():
        if raw.get(name) != value:
            raise ValueError(f'Mizar 7-14x3-7 MeSH config mismatch: {name}')


def inspect_initialization(path):
    """Inspect weights-only input without deserializing any training state."""
    path = Path(path).resolve()
    validate_model_config(json.loads((path / 'config.json').read_text(encoding='utf-8')))
    if any((path / name).exists() for name in ('training_state.pt', 'checkpoint_complete.json',
                                              'mesh_checkpoint_metadata.json')):
        return _inspect_checkpoint_weights(path)
    marker = json.loads((path / 'conversion_complete.json').read_text(encoding='utf-8'))
    metadata = json.loads((path / 'recursive_conversion_metadata.json').read_text(encoding='utf-8'))
    expected = {'complete': True, 'text_route': ROUTE, 'source': source_identity(),
                'physical_layers': 28, 'logical_layers': LOGICAL_LAYER_COUNT}
    if any(marker.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong-route or incomplete conversion')
    expected_metadata = {
        'status': 'PASS', 'text_route': ROUTE, 'architecture_contract': ARCHITECTURE,
        'source': source_identity(), 'inherited_weights_exact': True,
        'final_lm_strict_load': True, 'reload_tensors_logits_tokenizer_equal': True,
        'router_initialization': ROUTER_INITIALIZATION, 'router_seed': 0,
        'physical_layers': 28, 'logical_layers': LOGICAL_LAYER_COUNT, 'memory_slots': MEMORY_SLOT_COUNT, 'router_groups': ROUTER_COUNT,
        'source_layer_indices_0based': list(SOURCE_LAYERS),
    }
    if (any(metadata.get(k) != v for k, v in expected_metadata.items())
            or set(metadata.get('fresh_router_tensors', [])) != ROUTER_KEYS):
        raise ValueError('Invalid final-LM inheritance or official router initialization provenance')
    if not any(path.glob('*.safetensors')):
        raise FileNotFoundError(f'Converted weights missing: {path}')
    if not (path / 'tokenizer.json').is_file():
        raise FileNotFoundError(f'Tokenizer missing: {path}')
    return {'mode': 'fresh_final_lm_conversion', 'start_optimizer_step': 0,
            'restore_optimizer_scheduler_rng_cursor': False, 'source': source_identity()}


def _inspect_checkpoint_weights(path):
    """Old LR, optimizer, scheduler, RNG and data cursor do not constrain a new run."""
    names = ('checkpoint_complete.json', 'checkpoint_manifest.json', 'mesh_checkpoint_metadata.json')
    if not all((path / name).is_file() for name in names):
        raise ValueError('Weights-only initialization requires a complete route checkpoint; '
                         '--resume-from is reserved for full-state continuation')
    marker, manifest, metadata = [json.loads((path / name).read_text(encoding='utf-8')) for name in names]
    if (marker.get('status') != 'complete' or marker.get('complete_marker') is not True
            or manifest.get('status') != 'complete'
            or any(item.get('architecture_contract') != ARCHITECTURE for item in (marker, manifest, metadata))):
        raise ValueError('Weights-only checkpoint has an incomplete or different architecture')
    step = metadata.get('optimizer_step')
    if (type(step) is not int or step < 0 or path.name != f'checkpoint-{step:06d}'
            or any(item.get('optimizer_step') != step for item in (marker, manifest))):
        raise ValueError('Weights-only checkpoint directory and metadata steps disagree')
    expected = {'memory_slots': MEMORY_SLOT_COUNT, 'router_groups': ROUTER_COUNT,
                'router_module_count': 2 * ROUTER_COUNT, 'logical_to_physical': list(SCHEDULE)}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Weights-only checkpoint must use this six-slot, three-loop MeSH architecture')
    if not any(path.glob('*.safetensors')) or not (path / 'tokenizer.json').is_file():
        raise FileNotFoundError(f'Checkpoint model weights or tokenizer missing: {path}')
    return {'mode': 'trained_checkpoint_weights_only', 'start_optimizer_step': 0,
            'source_checkpoint_step': step, 'loaded_model_path': str(path),
            'restore_optimizer_scheduler_rng_cursor': False, 'routers': 'loaded_from_checkpoint',
            'source': source_identity()}
