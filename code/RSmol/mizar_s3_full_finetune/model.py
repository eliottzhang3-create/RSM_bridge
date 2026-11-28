"""Native Mizar: strict final inheritance; complete mapper and LM training."""
from pathlib import Path
import torch
from torch import nn
from models.mellow import Mellow
from contract import SCHEDULE, parameter_role


def install_trace(lm):
    trace=[]
    lm.model.last_layer_trace=()
    def begin(_module,_args,kwargs):
        if kwargs.get("use_cache") is not False or kwargs.get("attention_mask") is not None or kwargs.get("past_key_values") is not None:
            raise ValueError("S3 requires native causal attention, full recomputation and no KV state")
        trace.clear()
    def layer(index):
        def observe(_module,_args):trace.append(index)
        return observe
    def finish(_module,_args,output):
        lm.model.last_layer_trace=tuple(trace)
        if tuple(trace)!=SCHEDULE or getattr(output,"past_key_values",None) is not None:
            raise ValueError(f"native 30-layer trace changed: {trace}")
    lm.register_forward_pre_hook(begin,with_kwargs=True)
    for index,block in enumerate(lm.model.layers):block.register_forward_pre_hook(layer(index))
    lm.register_forward_hook(finish)


def configure_training(model):
    for name,p in model.named_parameters():p.requires_grad_(parameter_role(name)!="frozen")
    model.train();model.audio_encoder.base.ced.eval()
    install_trace(model.caption_decoder.lm)
    return model


def validate_trainability(model,*,production=True):
    counts={}
    for name,p in model.named_parameters():
        role=parameter_role(name)
        if p.requires_grad!=(role!="frozen") or p.dtype!=torch.float32:
            raise ValueError(f"incorrect S3 trainability/dtype: {name}")
        counts[role]=counts.get(role,0)+p.numel()
    if set(counts)!={"frozen","mapper","lm"}:raise ValueError(f"wrong parameter roles: {counts}")
    base,lm=model.audio_encoder.base,model.caption_decoder.lm
    if len(lm.model.layers)!=30 or len({id(layer) for layer in lm.model.layers})!=30:
        raise ValueError("S3 requires every original standard LM block once")
    if lm.get_input_embeddings().weight is not lm.get_output_embeddings().weight:
        raise ValueError("tied embedding/head Parameter identity lost")
    if any(m.training for m in base.ced.modules()):raise ValueError("frozen CED must remain eval")
    if base.mapper.training!=model.training or any(l.training!=model.training for l in lm.model.layers):
        raise ValueError("projector/LM modes disagree with training mode")
    if not isinstance(model.audio_encoder.projection,nn.Identity):raise ValueError("outer projection must be Identity")
    if production:
        if sum(counts.values())!=159299840 or len(model.state_dict())!=454:
            raise ValueError("official complete model inventory changed")
        if lm.config.hidden_size!=576 or lm.config.vocab_size!=49152:
            raise ValueError("official text dimensions changed")
        if base.mapper_type!="freq_merge_126" or base.mapper.resample or base.mapper.output_tokens!=126:
            raise ValueError("official complete projector geometry changed")
        if any("audio_gate" in n or "lora" in n or "router" in n or "loop_bias" in n for n in model.state_dict()):
            raise ValueError("unexpected experimental parameters in native model")
    return counts


def inherit_state(model,state,*,production=True):
    expected=model.state_dict()
    if not isinstance(state,dict) or set(state)!=set(expected):raise ValueError("official tensor inventory mismatch")
    if production and (len(state)!=454 or sum(v.numel() for v in state.values())!=187628509):
        raise ValueError("official454 inventory changed")
    for name,value in state.items():
        target=expected[name]
        if not torch.is_tensor(value) or value.shape!=target.shape or value.dtype!=target.dtype or not bool(torch.isfinite(value).all()):
            raise ValueError(f"invalid official tensor: {name}")
    a,b="caption_decoder.lm.model.embed_tokens.weight","caption_decoder.lm.lm_head.weight"
    if not torch.equal(state[a],state[b]):raise ValueError("official tied tensor values disagree")
    model.load_state_dict(state,strict=True)
    for name,value in model.state_dict().items():
        if not torch.equal(value,state[name]):raise ValueError(f"official loaded value differs: {name}")


def load_official(model_dir):
    root=Path(model_dir)
    model=Mellow(audioenc_name="CEDSmall",d_in=384,d_out=576,text_decoder=str(root/"assets/smollm2-135m"),
        prefix_length=383,freeze_text_decoder_weights=False,use_pretrained_audioencoder=True,
        freeze_audio_encoder_weights=True,pretrained_audioencoder_path=str(root/"assets/ced-small")).float()
    lm=model.caption_decoder.lm
    lm.config.use_cache=False;lm.gradient_checkpointing_disable()
    state=torch.load(root/"checkpoints/final/stage3_q4_seed20260905.ckpt",map_location="cpu",weights_only=True,mmap=True)
    inherit_state(model,state);del state
    configure_training(model);validate_trainability(model)
    return model
