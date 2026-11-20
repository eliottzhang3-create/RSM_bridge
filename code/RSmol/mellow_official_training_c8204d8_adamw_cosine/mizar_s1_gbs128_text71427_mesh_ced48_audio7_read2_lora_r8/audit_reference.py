"""Read-only source inheritance audit. No model assets, downloads or GPU required."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
BASE = ROOT.with_name('mizar_s1_gbs128_text71427_mesh_ced_small')
LORA = ROOT.with_name('mizar_s1_gbs128_text71427_mesh_ced48_loop_lora_r8')


def tree(path):
    return ast.parse(Path(path).read_text(encoding='utf-8-sig'))


def find_function(module, name):
    matches = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f'expected one function {name}: {len(matches)}')
    return matches[0]


def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 'audio7-read2-source', 'exec'), scope)


def normalized(path):
    return Path(path).read_bytes().replace(b'\r\n', b'\n')


def audit():
    snapshot = json.loads((ROOT/'protected_sources.sha256').read_text(encoding='utf-8'))
    for name, expected in snapshot.items():
        if hashlib.sha256(normalized(REPO/name)).hexdigest() != expected:
            raise ValueError(f'protected source changed: {name}')
    files = ['prepare_index.py', 'stage_data.py', 'write_config.py', 'models/audio.py',
             'models/ced_offline.py', 'utils/audio_io.py']
    files += [p.relative_to(BASE).as_posix() for p in (BASE/'data').glob('*.py')]
    for name in files:
        old = normalized(BASE/name).replace(BASE.name.encode(), ROOT.name.encode())
        if normalized(ROOT/name) != old:
            raise ValueError(f'base data/staging implementation changed: {name}')
    for name in ('models/ced.py', 'models/mellow.py', 'models/decoder.py', 'models/loop_lora.py'):
        if normalized(ROOT/name) != normalized(LORA/name):
            raise ValueError(f'CED/mapper/LoRA implementation changed: {name}')
    functions = {'contract.py': ['geometry', 'lr_factor', 'data_order', 'rank_seed', 'raw_references',
                   'validate_ced_assets', 'validate_ced_config', 'validate_checkpoint_tokenizer',
                   'validate_tokenizer_contract', 'recursive_checkpoint_contract'],
                 'train.py': ['answer_loss', 'move_batch', 'rng_state', 'restore_rng', 'inspect_tokenizer'],
                 'optim.py': ['track_and_clip_', '_clip_grad']}
    for name, names in functions.items():
        for fn in names:
            if ast.dump(find_function(tree(ROOT/name), fn)) != ast.dump(find_function(tree(LORA/name), fn)):
                raise ValueError(f'inherited numerical/data contract changed: {name}:{fn}')
    return {'status': 'PASS_LOCAL_SOURCE', 'protected_files': len(snapshot),
            'unchanged_data_staging_files': files,
            'numeric_functions_equal': functions, 'gpu_training_verified': False}


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
