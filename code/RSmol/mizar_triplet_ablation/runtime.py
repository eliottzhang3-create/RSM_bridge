"""Strict original model loader plus original-index three-block identity routing."""
import os
import random
from mizar_layer_pair_search import runtime as pair_reference
from mizar_layer_analysis_common import runtime as original
from mizar_letter_analysis.runtime import decision
from . import config

autocast,prefix=pair_reference.autocast,pair_reference.prefix


class TripleRouter(original.LayerRouter):
    def expected_layers(self):return [i for i in range(1,31) if i not in config.drop_layers(self.drop)]

    def set_drop(self,start):
        import torch
        excluded=config.drop_layers(start);self.drop=start
        self.lm.model.layers=torch.nn.ModuleList([m for i,m in enumerate(self.original,1) if i not in excluded])
        self.lm.config.num_hidden_layers=len(self.lm.model.layers)

    def evidence(self):
        dropped=config.drop_layers(self.drop)
        return {'window_start_1based':self.drop,'window_end_1based':None if self.drop is None else self.drop+2,
            'drop_layers_1based':dropped,'executed_original_layers_1based':list(self.trace),
            'effective_layers':30-len(dropped),'checked_forwards':self.forwards}


def load(args,bundle,binding):
    # Same strict original constructor/load, precision and native input contract
    # as mizar_layer_analysis_common.runtime.load, with a triplet controller.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    import transformers
    from mizar_official_eval import runtime as official
    from mizar_layer_pair_search.manifest import load_tokenizer
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('worker must see one assigned GPU')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 required, no fallback')
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    official.verify_bundle_unchanged(bundle,args.model_dir,args.checkpoint)
    modules=official.base.import_training_route(config.ROUTE)
    model=modules['models.mellow'].Mellow(audioenc_name='CEDSmall',d_in=384,d_out=576,
        text_decoder=str(args.model_dir/'assets/smollm2-135m'),prefix_length=383,freeze_text_decoder_weights=False,
        use_pretrained_audioencoder=True,freeze_audio_encoder_weights=True,
        pretrained_audioencoder_path=str(args.model_dir/'assets/ced-small'))
    model.float();state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    state_audit=official.validate_state_dict(state,torch);model.load_state_dict(state,strict=True);del state
    structure=official.audit_model(model);model.to(device).eval().requires_grad_(False)
    tokenizer,_=load_tokenizer(args.model_dir)
    if any(m.training for m in model.modules()) or {p.dtype for p in model.parameters()}!={torch.float32}:
        raise ValueError('expected all eval and unchanged FP32 source parameters')
    lm=model.caption_decoder.lm
    if lm.config._attn_implementation not in ('eager','sdpa'):raise ValueError('unsupported attention backend')
    controller=TripleRouter(lm)
    audit={'status':'PASS',**state_audit,**structure,'binding':binding,'strict_state_dict':True,
        'checkpoint_sha256':bundle['verified_files'][0]['sha256'],'weights_dtype':'float32','forward_dtype':'bfloat16_autocast',
        'audio_frontend_dtype':'float32','prefix_dtype':'bfloat16','tf32':False,'all_modules_eval':True,'bf16_supported':True,
        'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(device),
        'attention_implementation':lm.config._attn_implementation,'sdpa_backends':{'flash':False,'mem_efficient':False,'math':True},
        'experiment':config.SCHEMA,'task':args.task}
    return {'model':model,'lm':lm,'tokenizer':tokenizer,'device':device,'controller':controller,'audit':audit}


def logits(loaded,cached_prefix):
    import torch
    with torch.inference_mode(),autocast(loaded['device']):
        output=loaded['lm'](inputs_embeds=cached_prefix,attention_mask=None,labels=None,use_cache=False)
        value=output.logits[0,-1].float()
    if not bool(torch.isfinite(value).all()):raise ValueError('nonfinite first-answer logits')
    return value


def score(loaded,cached_prefix,record):
    import torch
    lp=logits(loaded,cached_prefix).log_softmax(-1)
    ids=torch.tensor(record['letter_token_ids'],device=lp.device,dtype=torch.long)
    values=lp.index_select(0,ids).tolist()
    scores=[float(v) if enabled else None for v,enabled in zip(values,record['choice_valid_mask'])]
    return {**decision(record,scores),'execution':loaded['controller'].evidence()}


def checks(loaded,cached_prefix,record):
    import torch
    ctl,lm=loaded['controller'],loaded['lm'];probes=[]
    try:
        ctl.set_drop(None);baseline=logits(loaded,cached_prefix).clone()
        for start in (6,15,25):
            ctl.set_drop(start);bypass=logits(loaded,cached_prefix).clone();evidence=ctl.evidence()
            ctl.set_drop(None);handles=[]
            def before(module,args,kwargs):module._triplet_identity_input=args[0] if args else kwargs['hidden_states']
            def after(module,args,output):
                hidden=module._triplet_identity_input
                return (hidden,*output[1:]) if isinstance(output,tuple) else hidden
            targets=[ctl.original[i-1] for i in config.drop_layers(start)]
            for module in targets:
                handles.extend([module.register_forward_pre_hook(before,with_kwargs=True),module.register_forward_hook(after)])
            try:identity=logits(loaded,cached_prefix).clone()
            finally:
                for handle in handles:handle.remove()
                for module in targets:
                    if hasattr(module,'_triplet_identity_input'):del module._triplet_identity_input
            if not torch.equal(bypass,identity):raise ValueError('triplet bypass differs from three executed identity-output blocks')
            probes.append({'status':'PASS','start':start,'drop_layers_1based':config.drop_layers(start),'max_abs_error':0.0,'execution':evidence})
        ctl.set_drop(None)
        if not torch.equal(logits(loaded,cached_prefix),baseline):raise ValueError('restoring baseline changed logits')
        return {'status':'PASS','baseline_restored_bitwise':True,'triplet_identity_probes':probes}
    finally:ctl.set_drop(None)
