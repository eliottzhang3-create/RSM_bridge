"""Each worker atomically caches only its assigned questions, once per task."""
import os
from mizar_layer_pair_search import audio,storage
from . import config,runtime


def precision(audit):
    return {k:audit[k] for k in ('torch','transformers','cuda','gpu','weights_dtype','forward_dtype',
        'audio_frontend_dtype','prefix_dtype','attention_implementation','sdpa_backends')}


def save_tensor(path,tensor):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with temporary.open('wb') as stream:
        torch.save(tensor,stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


def entry(directory,record,expected):
    import torch
    index=record['global_index']; path=directory/f'prefix_{index:04d}.pt'
    meta=storage.verify_seal(storage.read_json(directory/f'prefix_{index:04d}.json'))
    if (meta['cache_identity']!=expected or meta['record_sha256']!=storage.object_sha(record)
            or meta['file_sha256']!=storage.file_sha(path)): raise ValueError('prefix input/cache identity mismatch')
    value=torch.load(path,map_location='cpu',weights_only=True)
    if (not torch.is_tensor(value) or value.shape!=(1,383,576) or value.dtype!=torch.bfloat16
            or not bool(torch.isfinite(value).all()) or audio.tensor_sha(value)!=meta['tensor_sha256']):
        raise ValueError('invalid BF16 prefix')
    return value,meta


def prepare_local(loaded,records,output,expected,rank):
    import torch
    directory=output/'cache'/f'rank{rank:02d}'; directory.mkdir(parents=True,exist_ok=True)
    selected=[records[i] for i in config.indices(rank,len(records))]
    key={'run_identity_sha256':expected['identity_sha256'],'precision':precision(loaded['audit']),'rank':rank}
    storage.claim(directory,key)
    missing=[r for r in selected if not (directory/f'prefix_{r["global_index"]:04d}.json').exists()]
    if any(r['audio_ref']['kind']=='parquet' for r in missing): audio.prepare_selected_rows(missing)
    values,hashes={},[]
    try:
        for record in selected:
            index=record['global_index']; path=directory/f'prefix_{index:04d}.pt'; meta_path=directory/f'prefix_{index:04d}.json'
            if not meta_path.exists():
                wave,audit=audio.load_fixed20(record['audio_ref'])
                if audit!=record['audio_audit']: raise ValueError('decoded audio differs from frozen input')
                value=runtime.prefix(loaded,record,wave).cpu()
                save_tensor(path,value)
                meta=storage.seal({'cache_identity':key,'record_sha256':storage.object_sha(record),
                    'file_sha256':storage.file_sha(path),'tensor_sha256':audio.tensor_sha(value),'audio_audit':audit})
                storage.write_json(meta_path,meta)
                restored,_=entry(directory,record,key)
                if not torch.equal(value,restored): raise ValueError('serialized prefix is not bitwise exact')
            value,meta=entry(directory,record,key)
            values[index]=value.to(loaded['device']); hashes.append(meta['sha256'])
            print(__import__('json').dumps({'event':'PREFIX_READY','rank':rank,'index':index,'key':record['key']}),flush=True)
        result=storage.seal({'cache_identity':key,'indices':[r['global_index'] for r in selected],
                             'entry_meta_sha256':hashes,'status':'PASS'})
        storage.write_json(directory/'cache_manifest.json',result)
        return values,result
    finally: audio.clear_selected_rows()
