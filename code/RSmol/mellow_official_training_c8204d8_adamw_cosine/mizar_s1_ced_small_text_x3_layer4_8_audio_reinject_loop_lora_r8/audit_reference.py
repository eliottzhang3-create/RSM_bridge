"""Optional source and semantic audit; never an automatic formal-training gate."""
import ast
import copy
import hashlib
import json
from pathlib import Path

from contract import (DEFAULT_TEXT_MODEL_DIR, LOOP_LORA_CONTRACT, ROUTE, SOURCE_COUNTS,
                      TEXT_MODEL_SCHEDULE, semantic_contract, write_json, code_digest)

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
PARENT=ROOT.with_name('mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small_layer4_8_inject_audio_reinject_loop_lora_r8')
ENTRY='run_mizar_s1_ced_small_text_x3_layer4_8_audio_reinject_loop_lora_r8_5090.sh'
EXPECTED_TEXT='/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage4_5_10x3_5/formal_1epoch_bf16_20261109_153922858684473/checkpoint-009244'
CHANGED={'audit_reference.py','contract.py','models/recursive_layer_inject.py','models/recursive_text.py',
         'models/loop_lora.py','models/decoder.py','optim.py','train.py','write_config.py','runtime_checks.py','run.sh'}


def tree(path):
    return ast.parse(path.read_text(encoding='utf-8-sig'))


def find_function(module,name):
    found=[node for node in ast.walk(module) if isinstance(node,ast.FunctionDef) and node.name==name]
    if len(found)!=1: raise ValueError(f'expected one {name}, got {len(found)}')
    return found[0]


def execute(nodes,scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),
                 '<x3-source-functions>','exec'),scope)


def require_equal(actual,expected,label):
    if actual!=expected: raise ValueError('source/recipe mismatch: '+label)


def executable_ast(node):
    """Compare implementation while allowing architecture docstrings to change."""
    node=copy.deepcopy(node)
    for item in ast.walk(node):
        if isinstance(item,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
            if item.body and isinstance(item.body[0],ast.Expr) and isinstance(item.body[0].value,ast.Constant) and isinstance(item.body[0].value.value,str):
                item.body.pop(0)
    return ast.dump(node)


def normalized_train(source):
    return source.replace('5-10x3-5','5-10-5').replace('is_recursive_5_10x3_5','is_recursive_5_10_5').replace(
        'all three LoRA A/B pairs','both LoRA A/B pairs').replace('all three loop LoRA A/B pairs','both loop LoRA A/B pairs')


def audit():
    guard=json.loads((ROOT/'source_guard.json').read_text(encoding='utf-8'))
    for name,expected in guard.items():
        require_equal(hashlib.sha256((REPO/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),expected,name)
    inventory=lambda root:{p.relative_to(root).as_posix() for p in root.rglob('*')
                           if p.suffix in {'.py','.sh'} and '__pycache__' not in p.parts}
    require_equal(inventory(ROOT),inventory(PARENT),'complete copied source inventory')
    identical=[]
    for name in sorted(inventory(PARENT)-CHANGED):
        require_equal((ROOT/name).read_bytes(),(PARENT/name).read_bytes(),name)
        identical.append(name)
    old_train=tree(PARENT/'train.py')
    new_train=ast.parse(normalized_train((ROOT/'train.py').read_text(encoding='utf-8')))
    for name in ('train','answer_loss','save_checkpoint','reload_audit','rng_state','restore_rng','main',
                 'inspect_tokenizer','audit_checkpoint_tokenizer','probe_sources','role_snapshots','weights_changed'):
        require_equal(ast.dump(find_function(new_train,name)),ast.dump(find_function(old_train,name)),'trainer '+name)
    old=tree(PARENT/'models/recursive_layer_inject.py')
    new=ast.parse((ROOT/'models/recursive_layer_inject.py').read_text(encoding='utf-8').replace('5-10x3-5','5-10-5').replace('5_10x3_5','5_10_5'))
    helpers=('LogicalSlotCacheView','_assert_supported_llama_api','_make_cache','make_dynamic_cache',
             '_cache_seq_length','_causal_mask','_call_decoder_layer')
    for name in helpers:
        select=lambda t:next(n for n in t.body if getattr(n,'name',None)==name)
        require_equal(executable_ast(select(new)),executable_ast(select(old)),'attention/cache '+name)
    for name in ('inject_input_audio','inject_audio','enable_layer_injection','reinjection_diagnostics','injection_diagnostics'):
        require_equal(ast.dump(find_function(new,name)),ast.dump(find_function(old,name)),'audio residual '+name)
    require_equal(DEFAULT_TEXT_MODEL_DIR,EXPECTED_TEXT,'explicit text checkpoint')
    require_equal(list(TEXT_MODEL_SCHEDULE),[*range(5),*list(range(5,15))*3,*range(15,20)],'forty-entry schedule')
    scope={'__file__':str(PARENT/'contract.py')}
    execute(tree(PARENT/'contract.py').body,scope)
    prepared=dict(rows=sum(SOURCE_COUNTS.values()),source_rows=SOURCE_COUNTS,rows_canonical_sha256='rows',mapping_sha256='mapping')
    parent_recipe=scope['semantic_contract'](prepared,'text','ced')
    recipe=semantic_contract(prepared,'text','ced')
    changes={'route','code_sha256','recursive_model_source_sha256','text_model_contract','text_model_architecture',
             'text_model_schedule','audio_reinjection','serial_audio_injection','loop_lora'}
    for value in (parent_recipe,recipe):
        for key in changes: value.pop(key)
    require_equal(recipe,parent_recipe,'all inherited training/data/audio-encoder settings')
    return dict(status='PASS',scope='local source and synthetic semantic audit; no production assets or GPU',
                protected_originals=dict(files=len(guard),all_unchanged=True),byte_identical_sources=identical,
                common_recipe_equal=True,training_parity=dict(entire_train_function_ast_equal=True),
                route=ROUTE,default_text=DEFAULT_TEXT_MODEL_DIR,loop_lora=LOOP_LORA_CONTRACT,
                route_code_sha256=code_digest())


if __name__=='__main__':
    result=audit()
    write_json(ROOT/'reference_audit.json',result)
    print(json.dumps(result,indent=2))
