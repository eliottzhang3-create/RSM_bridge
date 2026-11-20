"""Read-only source inheritance audit. No model assets, downloads or GPU required."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
BASE = ROOT.with_name('mizar_s1_gbs128_text71427_mesh_ced_small')
LORA = ROOT.with_name('mizar_s1_gbs128_text71427_mesh_ced48_loop_lora_r8')
MOTHER = ROOT.with_name('mizar_s1_gbs128_text71427_mesh_ced48_audio7_read2_lora_r8')


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
    changed_files = {'README.md', 'audit_reference.py', 'protected_sources.sha256',
                     'contract.py', 'optim.py', 'train.py', 'run.sh',
                     'models/recursive_text.py', 'models/recursive_audio7_read2_ar6.py'}
    copied_files = []
    for source in sorted(MOTHER.rglob('*')):
        if not source.is_file() or '__pycache__' in source.parts or source.suffix in {'.pyc', '.pyo'}:
            continue
        name = source.relative_to(MOTHER).as_posix().replace('audio7_read2', 'audio7_read2_ar6')
        if name not in changed_files:
            if normalized(ROOT/name) != normalized(source).replace(b'audio7_read2', b'audio7_read2_ar6'):
                raise ValueError(f'unchanged mother implementation differs: {name}')
            copied_files.append(name)
    # Protect the complete training update, loss, checkpoint and RNG implementation.
    # Only these two auditing functions change to account for the four extra routers.
    parent_train = ast.parse(normalized(MOTHER/'train.py').decode().replace('audio7_read2', 'audio7_read2_ar6'))
    child_train = tree(ROOT/'train.py')
    # The existing preflight report must describe the new, larger parameter inventory.
    for node in ast.walk(parent_train):
        if isinstance(node, ast.Dict):
            for index, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == 'new_parameter_count':
                    node.values[index] = ast.parse("AUDIO_READ2_CONTRACT['added_parameters_with_lora']", mode='eval').body
    changed_functions = {'validate_trainable_model', 'router_gradient_audit'}
    for module in (parent_train, child_train):
        module.body = [node for node in module.body
                       if not isinstance(node, ast.FunctionDef) or node.name not in changed_functions]
    if ast.dump(parent_train) != ast.dump(child_train):
        raise ValueError('trainer changed outside the two explicit router audit functions')
    parent_model = tree(MOTHER/'models/recursive_audio7_read2.py')
    child_model = tree(ROOT/'models/recursive_audio7_read2_ar6.py')
    for name in ('_run_stack', '_route', '_write', '_read', 'parameter_audit'):
        if ast.dump(find_function(parent_model, name)) != ast.dump(find_function(child_model, name)):
            raise ValueError(f'Transformer/LoRA/MeSH arithmetic changed: {name}')
    parent_forward = find_function(next(node for node in parent_model.body
        if isinstance(node, ast.ClassDef) and node.name == 'MeshLlamaModel'), 'forward')
    child_forward = find_function(next(node for node in child_model.body
        if isinstance(node, ast.ClassDef) and node.name == 'MeshLlamaModel'), 'forward')
    guard = next(node for node in parent_forward.body if isinstance(node, ast.If)
        and ast.unparse(node.test) == "not hasattr(self, 'audio_read_routers')")
    guard.test = ast.parse("not hasattr(self, 'audio_read_routers') or not hasattr(self, 'audio_write_routers')", mode='eval').body
    if ast.dump(parent_forward) != ast.dump(child_forward):
        raise ValueError('model forward changed outside the required audio-router installation guard')
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
            'direct_mother': MOTHER.name, 'unchanged_mother_files': copied_files,
            'training_update_checkpoint_rng_equal': True, 'transformer_and_lora_forward_equal': True,
            'unchanged_data_staging_files': files,
            'numeric_functions_equal': functions, 'gpu_training_verified': False}


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
