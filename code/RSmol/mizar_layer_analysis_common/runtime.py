"""Full original model loading and an audited single-layer identity route."""
import copy
import os
import random
from mizar_layer_pair_search import runtime as reference
from . import config

autocast=reference.autocast
prefix=reference.prefix
score=reference.score
option_logits=reference.option_logits
option_score=reference.option_score


class LayerRouter:
    def __init__(self,lm,hidden_size=576,prefix_tokens=383):
        if len(lm.model.layers)!=30 or len({id(m) for m in lm.model.layers})!=30:
            raise ValueError('expected thirty original distinct blocks')
        self.lm,self.original=lm,tuple(lm.model.layers)
        self.hidden_size,self.prefix_tokens=hidden_size,prefix_tokens
        self.drop,self.trace,self.forwards=None,[],0
        lm.register_forward_pre_hook(self._begin,with_kwargs=True)
        lm.register_forward_hook(self._finish)
        for index,layer in enumerate(self.original,1): layer.register_forward_pre_hook(self._observe(index))

    def _observe(self,index):
        def hook(_module,_args): self.trace.append(index)
        return hook

    def _begin(self,_module,args,kwargs):
        x=kwargs.get('inputs_embeds')
        if (args or x is None or x.ndim!=3 or x.shape[0]!=1 or x.shape[2]!=self.hidden_size
                or not self.prefix_tokens<=x.shape[1]<=1024 or kwargs.get('use_cache') is not False
                or kwargs.get('attention_mask') is not None or kwargs.get('past_key_values') is not None):
            raise ValueError('native fixed-prefix/no-cache contract violated')
        self.trace.clear()

    def _finish(self,_module,_args,output):
        if self.trace!=self.expected_layers() or getattr(output,'past_key_values',None) is not None:
            raise ValueError('actual single-layer route/cache differs from requested condition')
        self.forwards+=1

    def expected_layers(self): return [i for i in range(1,31) if i!=self.drop]

    def set_drop(self,layer):
        if layer is not None and (type(layer) is not int or not 2<=layer<=29):
            raise ValueError('only original one-based layers 2..29 may be skipped')
        import torch
        self.drop=layer
        self.lm.model.layers=torch.nn.ModuleList([m for i,m in enumerate(self.original,1) if i!=layer])
        self.lm.config.num_hidden_layers=len(self.lm.model.layers)

    def evidence(self):
        return {'drop_layer_1based':self.drop,'executed_original_layers_1based':list(self.trace),
                'effective_layers':len(self.expected_layers()),'checked_forwards':self.forwards}


def load(args,bundle,binding):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    import transformers
    from mizar_official_eval import runtime as official
    from mizar_layer_pair_search.manifest import load_tokenizer
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('worker must see one assigned GPU')
    device=torch.device('cuda:0'); torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 required, no fallback')
    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    official.verify_bundle_unchanged(bundle,args.model_dir,args.checkpoint)
    modules=official.base.import_training_route(config.ROUTE)
    model=modules['models.mellow'].Mellow(audioenc_name='CEDSmall',d_in=384,d_out=576,
        text_decoder=str(args.model_dir/'assets/smollm2-135m'),prefix_length=383,freeze_text_decoder_weights=False,
        use_pretrained_audioencoder=True,freeze_audio_encoder_weights=True,
        pretrained_audioencoder_path=str(args.model_dir/'assets/ced-small'))
    model.float()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    state_audit=official.validate_state_dict(state,torch); model.load_state_dict(state,strict=True); del state
    structure=official.audit_model(model)
    model.to(device).eval().requires_grad_(False)
    tokenizer,_=load_tokenizer(args.model_dir)
    if any(m.training for m in model.modules()) or {p.dtype for p in model.parameters()}!={torch.float32}:
        raise ValueError('all modules must be eval with unchanged FP32 source weights')
    lm=model.caption_decoder.lm
    if lm.config._attn_implementation not in ('eager','sdpa'): raise ValueError('unsupported attention backend')
    controller=LayerRouter(lm)
    audit={'status':'PASS',**state_audit,**structure,'binding':binding,'strict_state_dict':True,
        'checkpoint_sha256':bundle['verified_files'][0]['sha256'],'weights_dtype':'float32','forward_dtype':'bfloat16_autocast',
        'audio_frontend_dtype':'float32','prefix_dtype':'bfloat16','tf32':False,'all_modules_eval':True,'bf16_supported':True,
        'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(device),'attention_implementation':lm.config._attn_implementation,
        'sdpa_backends':{'flash':False,'mem_efficient':False,'math':True}}
    return {'model':model,'lm':lm,'tokenizer':tokenizer,'device':device,'controller':controller,'audit':audit}


def check_skip_equivalence(loaded,cached_prefix,record):
    """Compare bypass to a genuinely executed block whose output is replaced by its input."""
    import torch
    ctl,lm=loaded['controller'],loaded['lm']
    ids=record['choice_token_ids'][0]
    checks=[]
    try:
        for layer in (2,29):
            ctl.set_drop(layer)
            bypass=option_logits(lm,cached_prefix,ids,loaded['device'])
            trace=ctl.evidence()
            ctl.set_drop(None)
            holder={}
            def before(_module,args,kwargs): holder['input']=args[0] if args else kwargs['hidden_states']
            def after(_module,_args,output):
                value=holder['input']
                return (value,*output[1:]) if isinstance(output,tuple) else value
            target=ctl.original[layer-1]
            handles=[target.register_forward_pre_hook(before,with_kwargs=True),target.register_forward_hook(after)]
            try: replaced=option_logits(lm,cached_prefix,ids,loaded['device'])
            finally:
                for handle in handles: handle.remove()
            torch.testing.assert_close(bypass,replaced,rtol=0,atol=0)
            checks.append({'layer_1based':layer,'status':'PASS','max_abs_error':float((bypass-replaced).abs().max()),'execution':trace})
        return {'status':'PASS','bypass_vs_identity_output':checks}
    finally: ctl.set_drop(None)
