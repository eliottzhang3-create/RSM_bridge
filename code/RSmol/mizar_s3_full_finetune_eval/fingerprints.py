"""Validate actual per-question token/audio input identities."""
from mizar_s1_ced_reasonaqa5000_eval import storage

def input_fingerprint(generation):
    import re
    actual = {key:generation.get(key) for key in ('prompt_token_ids','prompt_attention_mask','audio_half_sha256')}
    ids,mask,audio = actual.values()
    if (not isinstance(ids,list) or len(ids)!=256 or any(type(x) is not int or x<0 for x in ids)
            or not isinstance(mask,list) or len(mask)!=256 or not any(mask)
            or any(type(x) is not int or x not in (0,1) for x in mask)
            or not isinstance(audio,list) or len(audio)!=2
            or any(not isinstance(x,str) or not re.fullmatch('[0-9a-f]{64}',x) for x in audio)
            or storage.object_sha(actual)!=generation.get('input_fingerprint_sha256')):
        raise ValueError('invalid per-sample model input fingerprint')
    return generation['input_fingerprint_sha256']
