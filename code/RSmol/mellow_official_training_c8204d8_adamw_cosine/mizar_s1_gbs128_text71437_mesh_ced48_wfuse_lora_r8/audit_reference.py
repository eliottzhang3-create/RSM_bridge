"""Read-only two-parent and original three-loop numerical source audit."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
BASE = ROOT.with_name('mizar_s1_gbs128_text71437_mesh_ced_small')
SOURCE = ROOT.with_name('mizar_s1_gbs128_text71427_mesh_ced48_wfuse_lora_r8')


def tree(path):
    return ast.parse(Path(path).read_text(encoding='utf-8-sig'))


def find_function(module, name):
    matches = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f'expected one function {name}: {len(matches)}')
    return matches[0]


def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 'three-loop-wfuse-source', 'exec'), scope)


def normalized(path):
    return Path(path).read_bytes().replace(b'\r\n', b'\n')


def equal_node(left, right, label):
    if ast.dump(left, include_attributes=False) != ast.dump(right, include_attributes=False):
        raise ValueError(f'inherited numerical implementation changed: {label}')


def audit_parent():
    snapshot = json.loads((ROOT / 'protected_sources.json').read_text(encoding='utf-8'))['sha256_normalized_lf']
    for name, expected in snapshot.items():
        if hashlib.sha256(normalized(REPO / name)).hexdigest() != expected:
            raise ValueError(f'protected source changed: {name}')
    files = ['prepare_index.py', 'stage_data.py', 'check_ced_assets.py', 'check_text_assets.py',
             'models/audio.py', 'models/ced_offline.py', 'utils/audio_io.py']
    files += [p.relative_to(BASE).as_posix() for p in sorted((BASE / 'data').glob('*.py'))]
    for name in files:
        if normalized(ROOT / name) != normalized(BASE / name):
            raise ValueError(f'three-loop data/staging changed: {name}')
    for name in ('models/ced.py', 'models/mellow.py', 'models/decoder.py'):
        if normalized(ROOT / name) != normalized(SOURCE / name):
            raise ValueError(f'wfuse CED/mapper/audio interface changed: {name}')
    functions = {
        'contract.py': ['geometry', 'lr_factor', 'data_order', 'rank_seed', 'raw_references',
                        'validate_ced_assets', 'validate_ced_config', 'validate_checkpoint_tokenizer',
                        'validate_tokenizer_contract', 'recursive_checkpoint_contract',
                        'recursive_model_source_sha256'],
        'train.py': ['answer_loss', 'move_batch', 'rng_state', 'restore_rng', 'inspect_tokenizer',
                     'audit_checkpoint_tokenizer', 'collective_require', 'tensor_digest', 'weights_changed'],
        'optim.py': ['track_and_clip_', '_clip_grad'],
        'write_config.py': ['file_sha256', 'asset_identity'],
    }
    for name, names in functions.items():
        for fn in names:
            equal_node(find_function(tree(ROOT / name), fn), find_function(tree(BASE / name), fn), f'{name}:{fn}')
    shared = tree(REPO / 'code/RSmol/recursive_model_mizar_text_7_14x3_7_mesh.py')
    private = tree(ROOT / 'models/recursive_mesh_wfuse_lora.py')
    helpers = ('_cfg', '_assert_llama_api', 'make_dynamic_cache', '_cache_seq_length', '_validate_cache',
               '_causal_mask', '_call_layer', '_init_router', '_write', '_read', '_route', 'parameter_audit')
    for name in helpers:
        equal_node(find_function(private, name), find_function(shared, name), f'backbone:{name}')
    for name in ('enable_audio_fusion', 'fuse_write_value', 'fusion_diagnostics'):
        equal_node(find_function(private, name), find_function(tree(SOURCE / 'models/recursive_mesh_wfuse_lora.py'), name), f'fusion:{name}')

    def epoch_loop(path):
        return next(n for n in find_function(tree(path), 'train').body
                    if isinstance(n, ast.For) and ast.unparse(n.target) == 'epoch')
    old_epoch = ast.parse(ast.unparse(epoch_loop(SOURCE / 'train.py')).replace(
        'both recursive passes require finite nonzero gradients',
        'all three recursive passes require finite nonzero gradients')).body[0]
    equal_node(epoch_loop(ROOT / 'train.py'), old_epoch, 'complete_epoch_update_loop')

    class StripFusion(ast.NodeTransformer):
        def visit_Assign(self, node):
            if ast.unparse(node.targets[0]) in ('value_pre', 'value'):
                return None
            return self.generic_visit(node)

        def visit_Name(self, node):
            if node.id in ('value_pre', 'value'):
                node.id = 'prefix_output' if node.id == 'value_pre' else 'core'
            return node

        def visit_Call(self, node):
            node.keywords = [kw for kw in node.keywords if kw.arg != 'loop_index']
            return self.generic_visit(node)

    def memory_segment(module, strip=False):
        method = next(n for cls in module.body if isinstance(cls, ast.ClassDef) and cls.name == 'MeshLlamaModel'
                      for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        begin = next(i for i,n in enumerate(method.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'memory')
        end = next(i for i,n in enumerate(method.body) if isinstance(n, ast.Assign) and ast.unparse(n.value) == 'self.norm(hidden)')
        result = ast.Module(body=copy.deepcopy(method.body[begin:end+1]), type_ignores=[])
        return StripFusion().visit(result) if strip else result

    equal_node(memory_segment(private, True), memory_segment(shared), 'all_four_original_memory_interactions')
    return {'status': 'PASS_LOCAL_SOURCE', 'protected_source_files': len(snapshot),
            'identical_files_normalized_lf': files, 'numeric_functions_equal': functions,
            'backbone_helpers_equal': list(helpers), 'complete_epoch_update_loop_equal': True,
            'two_write_value_substitutions_only': True, 'gpu_training_verified': False}


def audit():
    return audit_parent()


if __name__ == '__main__':
    print(json.dumps(audit_parent(), indent=2))
