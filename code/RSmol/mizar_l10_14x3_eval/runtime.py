"""Strict model-only inference for L10-14x3/A9 gate/bias3/train L15."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from .registry import ROOT, PACKAGE, ROUTES, EXPECTED_STEP, variant_spec
from . import sources

if str(ROOT/'scripts') not in sys.path:
    sys.path.insert(0,str(ROOT/'scripts'))
import mizar_s1_ced_eval_runtime as reference

PROMPT_TOKENS,MAX_NEW_TOKENS,AUDIO_TOKENS = 256,300,126
PREFIX_TOKENS,HIDDEN_SIZE = 383,576
PROTOCOL = copy.deepcopy(reference.PROTOCOL)
sha256_file = reference.sha256_file


def evaluation_sources():
    files = set(PACKAGE.glob('*.py')) | set(PACKAGE.glob('*.sh')) | set(PACKAGE.glob('*.json'))
    files.add(ROOT/'run_mizar_l10_14x3_eval_4090.sh')
    files.add(ROOT/'mizar_s1_ced_mmsu_eval/storage.py')
    files.add(ROOT/'mizar_s1_ced_reasonaqa5000_eval/storage.py')
    for name in ('mizar_s1_ced_eval_runtime.py','evaluate_mmau_test_mini_5_10x2_5_mesh_mellow.py',
                 'evaluate_mmar_5_10x2_5_mesh_mellow.py'):
        files.add(ROOT/'scripts'/name)
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in sorted(files) if '__pycache__' not in p.parts}


def read_checkpoint(args):
    sources.prepare(args.eval_variant,args.sources_cache,args.source_revision)
    stat = args.checkpoint.stat()
    runtime_before = args.runtime_config.stat()
    signature = (stat.st_size,stat.st_mtime_ns,runtime_before.st_size,runtime_before.st_mtime_ns)
    if hasattr(args,'_checkpoint_payload'):
        if signature != args._checkpoint_signature:
            raise ValueError('checkpoint changed after identity resolution')
        return args._checkpoint_payload
    import torch
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    after = args.checkpoint.stat()
    runtime_after = args.runtime_config.stat()
    if signature != (after.st_size,after.st_mtime_ns,runtime_after.st_size,runtime_after.st_mtime_ns):
        raise ValueError('checkpoint changed during read')
    config = json.loads(args.runtime_config.read_text(encoding='utf-8'))
    validate_checkpoint(saved,config,args.eval_variant,args.expected_step,
        sources.load_contract(args.eval_variant,args.sources_cache,args.source_revision))
    runtime_after = args.runtime_config.stat()
    if signature[2:] != (runtime_after.st_size,runtime_after.st_mtime_ns):
        raise ValueError('runtime changed during contract read')
    args.expected_step = saved['total_step']
    args._checkpoint_signature,args._checkpoint_payload = signature,saved
    return saved


def validate_checkpoint(saved,config,target,expected_step=None,registered=None):
    spec = variant_spec(target)
    if not isinstance(saved,dict) or not isinstance(saved.get('contract'),dict):
        raise ValueError('expected full formal checkpoint mapping')
    c = saved['contract']
    from mizar_s1_ced_reasonaqa5000_eval.storage import object_sha
    sha = object_sha(c)
    if (saved.get('schema_version') != 1 or saved.get('mode') != 'formal'
            or saved.get('route_variant') != spec['route'] or c.get('route') != spec['route']
            or config.get('route') != spec['route'] or config.get('mode') != 'formal'
            or config.get('contract') != c or saved.get('contract_sha256') != sha
            or config.get('contract_sha256') != sha):
        raise ValueError('checkpoint/runtime route or contract identity differs')
    if c.get('code_sha256') != sources.profile(target)['code_sha256']:
        raise ValueError('checkpoint training source differs from registered source')
    registered = registered or sources.load_contract(target)
    prepared = {key:c[key] for key in ('rows_canonical_sha256','mapping_sha256','source_rows','dataset_selection') if key in c}
    prepared['rows'] = c.get('geometry',{}).get('rows')
    selection = prepared.get('dataset_selection')
    if not isinstance(selection,dict) or not all(k in selection for k in ('seal_sha256','historical_inputs','selected_counts')):
        raise ValueError('strong-only dataset selection seal is missing')
    registered.validate_data_counts(prepared['rows'],prepared.get('source_rows'),selection['selected_counts'])
    expected = registered.semantic_contract(prepared,c.get('model_dir'),c.get('official_bundle'))
    actual = {key:value for key,value in c.items() if key not in ('packages','python')}
    if actual != expected:
        fields = sorted(key for key in set(actual)|set(expected) if actual.get(key) != expected.get(key) or (key in actual) != (key in expected))
        raise ValueError('native training semantic contract differs: '+','.join(fields))
    geometry = c['geometry']
    total = geometry.get('total_steps')
    step = saved.get('total_step')
    if (type(total) is not int or type(step) is not int or not 1 <= step <= total
            or expected_step is not None and expected_step != step or saved.get('num_epochs') != 1
            or config.get('max_optimizer_steps') != total
            or (saved.get('epoch'),saved.get('next_optimizer_step')) != registered.training_cursor(step,geometry)):
        raise ValueError('invalid checkpoint step/epoch cursor')
    if (step != EXPECTED_STEP or tuple(geometry.get(k) for k in
            ('world_size','microbatch','gradient_accumulation_steps')) != (8,4,4)):
        raise ValueError('requires selected step2542 and training geometry 8,4,4')
    state = saved.get('state_dict')
    if not isinstance(state,dict) or len(state) != spec['state_count']:
        raise ValueError('incomplete model state inventory')
    return c


def validate_model_state(model,saved,contract_module):
    import torch
    state,current = saved['state_dict'],model.state_dict()
    if state.keys() != current.keys():
        raise ValueError('model tensor names differ')
    for name,value in state.items():
        target = current[name]
        if (not torch.is_tensor(value) or value.shape != target.shape or value.dtype != target.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError('model shape/dtype/finite differs: '+name)
        if contract_module.parameter_role(name) == 'frozen' and not torch.equal(value,target):
            raise ValueError('frozen official state changed: '+name)
    a,b = 'caption_decoder.lm.model.embed_tokens.weight','caption_decoder.lm.lm_head.weight'
    if not torch.equal(state[a],state[b]):
        raise ValueError('tied embedding/head values disagree')
    # Verify actual frozen tensor digest, without creating/loading any optimizer.
    from training_helpers import tensor_digest
    frozen = [(n,t) for n,t in current.items() if contract_module.parameter_role(n) == 'frozen']
    if saved.get('frozen_sha256') != tensor_digest(frozen):
        raise ValueError('frozen official state digest differs')
    return state


def install_forward_audit(lm,target):
    spec = variant_spec(target)
    trace,gate_calls,norm_calls = [],[],[]
    audit = dict(status='PENDING',checked_forwards=0,expected_layers=list(spec['schedule']),
                 expected_loop_entries=[0,1,2],expected_gate_calls=2,expected_final_norm_calls=1,
                 use_cache=False,last_layers=[])
    def begin(module,args,kwargs):
        trace.clear();gate_calls.clear();norm_calls.clear()
        x = kwargs.get('inputs_embeds')
        if (x is None or x.shape[0] != 1 or x.shape[2] != lm.config.hidden_size
                or not 383 <= x.shape[1] <= 682 or kwargs.get('use_cache') is not False
                or kwargs.get('attention_mask') is not None or kwargs.get('past_key_values') is not None):
            raise ValueError('full-recompute inference contract differs')
    def layer(index):
        def hook(module,args):
            trace.append(index)
        return hook
    def finish(module,args,output):
        if (trace != list(spec['schedule']) or output.past_key_values is not None
                or lm.model.last_loop_trace != (0,1,2) or len(gate_calls)!=2 or len(norm_calls)!=1):
            raise ValueError('native layer/gate/bias/final norm execution differs')
        audit.update(status='PASS',checked_forwards=audit['checked_forwards']+1,last_layers=list(trace),
                     last_loop_entries=list(lm.model.last_loop_trace),last_gate_calls=len(gate_calls),
                     last_final_norm_calls=len(norm_calls))
    def gate(module,args):
        if tuple(args[0].shape)!=(1,126,lm.config.hidden_size):
            raise ValueError('gate must consume the previous round audio126 hidden states')
        gate_calls.append(1)
    lm.register_forward_pre_hook(begin,with_kwargs=True)
    for index,module in enumerate(lm.model.layers):
        module.register_forward_pre_hook(layer(index))
    lm.register_forward_hook(finish)
    lm.model.audio_gate.register_forward_pre_hook(gate)
    lm.model.norm.register_forward_hook(lambda *args:norm_calls.append(1))
    return audit


def load_runtime(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    from transformers import AutoTokenizer, __version__ as transformers_version
    if transformers_version != '4.54.1':
        raise ValueError('requires Transformers4.54.1; do not modify the shared environment')
    if not torch.cuda.is_available():
        raise RuntimeError('formal evaluation requires one CUDA GPU')
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0');torch.cuda.set_device(device)
    saved = read_checkpoint(args)
    modules = sources.import_training(args.eval_variant,args.sources_cache,args.source_revision)
    c = saved['contract']
    bundle = modules['assets'].audit_bundle(args.model_dir)
    if bundle != c['official_bundle'] or str(args.model_dir.resolve()) != c['model_dir']:
        raise ValueError('official initialization assets differ from training')
    model = modules['model'].load_official(args.model_dir)
    state = validate_model_state(model,saved,modules['contract'])
    model.load_state_dict(state,strict=True)
    modules['model'].validate_trainability(model)
    model.eval().requires_grad_(False)
    model.to(device)
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('all inference modules must be eval and frozen')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir/'assets/smollm2-135m',local_files_only=True)
    tokenizer.add_special_tokens({'pad_token':'!'})
    tokenizer.padding_side = tokenizer.truncation_side = 'right'
    tokens = modules['contract'].validate_tokenizer_contract(tokenizer,tokenizer.pad_token_id,
        model.caption_decoder.lm.get_input_embeddings().num_embeddings)
    text_encoder = modules['data.audiotext_dataset'].AudioTextDataset.__new__(modules['data.audiotext_dataset'].AudioTextDataset)
    text_encoder.tokenizer = tokenizer
    audit = dict(status='PASS',eval_variant=args.eval_variant,checkpoint_step=saved['total_step'],
        training_route=saved['route_variant'],training_geometry=c['geometry'],training_code_sha256=c['code_sha256'],
        contract_sha256=saved['contract_sha256'],strict_state_dict=True,trained_values_preserved=True,
        frozen_official_state_verified=True,all_modules_eval=True,all_parameters_frozen=True,
        state_tensor_count=len(state),parameter_count=sum(p.numel() for p in model.parameters()),
        dtype='float32',tf32=False,tokenizer=tokens,assets=bundle,
        source_revision=sources.profile(args.eval_variant)['revision'],
        training_source_root=str(args.route_root),model_execution=execution_policy(args.eval_variant),
        forward_audit=install_forward_audit(model.caption_decoder.lm,args.eval_variant),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(device))
    vars(args).pop('_checkpoint_payload',None)
    return model,tokenizer,device,text_encoder,modules['data.fixed20'],audit,modules


def generate_prefix_inference(model,inputs,*,eval_variant,audit):
    result = model.generate_prefix_inference(inputs)
    if len(result) not in (2,3):
        raise ValueError('unexpected native prefix interface')
    prefix,audio = result[:2]
    if len(result) == 3 and result[2] is not None:
        raise ValueError('unexpected external model context')
    return prefix,audio,None


def sample_context(eval_variant,audio_output,context,text_input):
    if context is not None:
        raise ValueError('these models have no external recurrent context')
    return {}


def validate_context(eval_variant,context,prefix):
    variant_spec(eval_variant)
    if context != {}:
        raise ValueError('unexpected stale sample context')
    return {}


def greedy_decode(model,tokenizer,prefix,max_new_tokens=MAX_NEW_TOKENS,*,eval_variant,context):
    import torch
    if max_new_tokens != 300 or prefix.shape != (1,383,model.caption_decoder.lm.config.hidden_size) or prefix.dtype != torch.float32:
        raise ValueError('FP32 prefix383/max300 required')
    validate_context(eval_variant,context,prefix)
    generated_ids,generated = [],prefix
    stop_reason = 'max_new_tokens'
    for _ in range(max_new_tokens):
        output = model.caption_decoder.lm(inputs_embeds=generated,attention_mask=None,labels=None,use_cache=False)
        logits = output.logits[:,-1,:]
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('nonfinite generation logits')
        next_id = int(torch.argmax(logits,dim=-1).item())
        generated_ids.append(next_id)
        del output,logits
        if next_id == int(tokenizer.eos_token_id):
            stop_reason = 'eos_token'
            break
        ids = torch.tensor([[next_id]],device=prefix.device,dtype=torch.long)
        generated = torch.cat((generated,model.caption_decoder.lm.get_input_embeddings()(ids)),dim=1)
    text = tokenizer.decode(generated_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    return dict(generated_token_ids=generated_ids,generated_text=text,generated_text_raw=text,
        generated_token_count=len(generated_ids),stop_reason=stop_reason,max_new_tokens=max_new_tokens)


def execution_policy(variant):
    spec = variant_spec(variant)
    return dict(architecture=variant,attention_mask=None,prompt_padding_visible=True,physical_layers=30,
        schedule=list(spec['schedule']),use_cache=False,state_policy='rebuild_each_full_forward',
        audio_anchor='raw_post_layer9_audio126',gate='A9+tanh(Linear(previous_audio))*previous_audio',
        shared_gate_calls=2,loop_bias_shape=[3,576],loop_bias_order='reconstruct_audio_then_add_current_bias',
        bias_broadcast='all_tokens',final_norm_calls=1)


def protocol_for(variant,base=None):
    result = copy.deepcopy(PROTOCOL if base is None else base)
    result.update(attention_mask=None,prompt_padding='right_bang_visible',model_execution=execution_policy(variant))
    return result


def waveform_sha(waveform):
    return hashlib.sha256(waveform.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
