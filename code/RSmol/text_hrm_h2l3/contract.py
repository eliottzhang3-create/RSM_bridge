"""Dependency-light architecture, training recipe, and artifact identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REMOTE_REPO = Path('/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM')
REMOTE_HOME = REMOTE_REPO.parents[1]
ROUTE = 'smollm2_text_hrm_h2l3_p5_l5_h5_c5_v1'
ARCHITECTURE = 'physical20_execution50_hrm_h2l3_magicnorm_gated_gqa_v1'
DEFAULT_SOURCE = REMOTE_HOME / 'models/SmolLM2'
DEFAULT_MODEL = REMOTE_HOME / 'models/SmolLM2-HRM-H2L3-P5-L5-H5-C5'
DEFAULT_DATA = REMOTE_HOME / 'data/SmolLM2-135M-10Bsubset'
DEFAULT_OUTPUT = REMOTE_HOME / 'outputs/RSmol/text_hrm_h2l3'
SOURCE_LAYERS = tuple(range(5)) + tuple(range(5, 25, 2)) + tuple(range(25, 30))
SCHEDULE = tuple(range(5)) + (tuple(range(5, 10)) * 3 + tuple(range(10, 15))) * 2 + tuple(range(15, 20))
FORMAT = 'hrm_text_full_checkpoint_v1'
SOURCE_CONFIG = dict(model_type='llama', num_hidden_layers=30, hidden_size=576,
                     intermediate_size=1536, num_attention_heads=9,
                     num_key_value_heads=3, vocab_size=49152, tie_word_embeddings=True,
                     rms_norm_eps=1e-5, rope_theta=100000.0, attention_dropout=0.0,
                     attention_bias=False)
FIELDS = dict(text_route=ROUTE, architecture_contract=ARCHITECTURE,
              architectures=['SmolHRMForCausalLM'], hrm_h_cycles=2, hrm_l_cycles=3,
              hrm_entry_eps=1e-6, hrm_exit_eps=1e-6,
              hrm_layers=list(range(5, 15)), hrm_l_layers=list(range(5, 10)),
              hrm_h_layers=list(range(10, 15)), hrm_initial_state_seed=0,
              hrm_initial_state='official_fmod_trunc_normal_std1_fixed_buffer_fp32',
              hrm_norm_conversion='fold_gamma_into_qkv_and_gate_up',
              hrm_attention_gate='sigmoid_per_channel_zero_weight_oproj_times2',
              hrm_input='parameterless_rmsnorm_prelude_output',
              hrm_attention='causal_padding_mask', hrm_logical_layers=50,
              hrm_source_layers=list(SOURCE_LAYERS), hrm_schedule=list(SCHEDULE),
              use_cache=False)
RECIPE = dict(total_steps=9244, world_size=8, micro_batch_size=2,
              accumulation_steps=64, global_batch_size=1024, context_length=1024,
              seed=0, record_buffer_size=4096, optimizer='official_AdamATan2',
              betas=[0.9, 0.95], weight_decay=0.1, lr=3e-4, lr_warmup_steps=3000,
              lr_policy='linear_from_zero_then_constant_1based', ema=0.9999,
              bp_warmup_steps=1849, bp_min=3, bp_max=5, gradient_clipping=False,
              grad_scaler=False, loss='global_shifted_valid_token_mean',
              parameter_dtype='float32', optimizer_dtype='float32',
              forward_autocast='bfloat16', norm_ce_dtype='float32', tf32=False,
              save_every=400, retention=3, workers=0, packing=False,
              dataset_cursor='deterministic_shard_start_replay_v1')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def learning_rate(update):
    if type(update) is not int or not 1 <= update <= RECIPE['total_steps']:
        raise ValueError('update must be in 1..9244')
    return RECIPE['lr'] * min(update / RECIPE['lr_warmup_steps'], 1.0)


def bp_steps(update):
    if type(update) is not int or not 1 <= update <= RECIPE['total_steps']:
        raise ValueError('update must be in 1..9244')
    return 3 + min(2, (2 * update) // RECIPE['bp_warmup_steps'])


def validate_source_config(raw, *, production=True):
    expected = SOURCE_CONFIG if production else {'model_type': 'llama', 'num_hidden_layers': 30,
                                                'tie_word_embeddings': True, 'attention_dropout': 0.0,
                                                'attention_bias': False}
    errors = {key: {'expected': value, 'actual': raw.get(key)}
              for key, value in expected.items() if raw.get(key) != value}
    if raw.get('mlp_bias', False) or any(key in raw for key in ('text_route', 'hrm_h_cycles', 'recursive_layer_count')):
        errors['source_identity'] = 'requires original, unconverted 30-layer SmolLM2'
    if errors:
        raise ValueError(f'source config mismatch: {errors}')


def target_config(source):
    import copy
    result = copy.deepcopy(source)
    result.num_hidden_layers = 20
    for key, value in FIELDS.items():
        setattr(result, key, copy.deepcopy(value))
    return result


def validate_config(raw, *, production=True):
    errors = {key: {'expected': value, 'actual': raw.get(key)}
              for key, value in FIELDS.items() if raw.get(key) != value}
    expected = (SOURCE_CONFIG | {'num_hidden_layers': 20}) if production else {'num_hidden_layers': 20, 'tie_word_embeddings': True}
    errors.update({key: {'expected': value, 'actual': raw.get(key)}
                   for key, value in expected.items() if raw.get(key) != value})
    if raw.get('mlp_bias', False):
        errors['mlp_bias'] = 'unsupported'
    if errors:
        raise ValueError(f'HRM architecture mismatch: {errors}')


def require_external(path):
    path = Path(path).resolve()
    if any(path == root.resolve() or root.resolve() in path.parents for root in (REPO, REMOTE_REPO)):
        raise ValueError(f'model/data/output artifacts must be outside the Git checkout: {path}')
    return path


def file_inventory(root, *, include_state=False):
    root = Path(root)
    suffixes = {'.safetensors', '.bin', '.json', '.model', '.txt', '.tiktoken'}
    return {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(root.rglob('*')) if p.is_file()
            and (p.suffix in suffixes or (include_state and p.name == 'training_state.pt'))
            and p.name not in {'conversion_complete.json', 'checkpoint_complete.json', 'failure.json'}}


def verify_inventory(root, inventory):
    root = Path(root).resolve()
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('missing file inventory')
    for name, expected in inventory.items():
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file() or path.is_symlink():
            raise ValueError(f'invalid artifact inventory path: {name}')
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError(f'artifact checksum mismatch: {name}')


def code_identity():
    files = sorted(HERE.glob('*.py'))
    files += [HERE.parent / 'scripts/convert_text_hrm_h2l3.sh',
              HERE.parent / 'scripts/train_text_hrm_h2l3_ddp.sh',
              HERE.parent / 'run_text_hrm_h2l3_3090.sh']
    return {str(p.relative_to(REPO)).replace('\\', '/'): hashlib.sha256(
        p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in files}
