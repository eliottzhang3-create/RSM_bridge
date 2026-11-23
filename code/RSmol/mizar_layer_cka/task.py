import json
import numpy as np
from mizar_layer_pair_search import audio,storage
from mizar_audio_cutoff import frozen_manifest
from mizar_layer_analysis_common import audit,cache,config
from . import capture,metrics


def protocol(args):
    return {**config.COMMON_PROTOCOL,'task':'cka','rows':100,'layers':30,'views':list(capture.VIEWS),
        'capture_point':'raw_post_block_before_final_LM_norm','forward_input':'prefix_only_no_candidate_or_GT',
        'feature_dtype':'float32','CKA_dtype':'float64','metric':['linear_standard','linear_unbiased_HSIC'],
        'global_centering':True,'permutation_repeats':args.permutation_repeats,'permutation_seed':1234}


def prepare(args): return frozen_manifest.load(args.manifest,log=True)
runtime_check=capture.runtime_check


def feature_entry(directory,record,expected):
    import torch
    index=record['global_index']; path=directory/f'features_{index:03d}.pt'
    meta=storage.verify_seal(storage.read_json(directory/f'features_{index:03d}.json'))
    if (meta['identity_sha256']!=expected['identity_sha256'] or meta['record_sha256']!=storage.object_sha(record)
            or meta['file_sha256']!=storage.file_sha(path) or meta['index']!=index): raise ValueError('feature input/identity changed')
    value=torch.load(path,map_location='cpu',weights_only=True)
    if (not torch.is_tensor(value) or value.dtype!=torch.float32 or value.shape!=(3,30,576)
            or not bool(torch.isfinite(value).all()) or meta['tensor_sha256']!=audio.tensor_sha(value)):
        raise ValueError('invalid captured features')
    if meta['execution']['executed_original_layers_1based']!=list(range(1,31)) or meta['execution']['drop_layer_1based'] is not None:
        raise ValueError('features were not captured from full original model')
    return value,meta


def work(args,loaded,prefixes,records,expected,rank,dist):
    directory=args.output_dir/'features'; directory.mkdir(exist_ok=True)
    for index in config.indices(rank,100):
        record=records[index]; path=directory/f'features_{index:03d}.pt'
        if not (directory/f'features_{index:03d}.json').exists():
            value,_=capture.extract(loaded,prefixes[index],record['prompt_attention_mask_audit_only'])
            cache.save_tensor(path,value)
            storage.write_json(directory/f'features_{index:03d}.json',storage.seal({'index':index,'key':record['key'],
                'identity_sha256':expected['identity_sha256'],'record_sha256':storage.object_sha(record),
                'file_sha256':storage.file_sha(path),'tensor_sha256':audio.tensor_sha(value),'execution':loaded['controller'].evidence()}))
        feature_entry(directory,record,expected)
        print(json.dumps({'event':'CKA_FEATURE_READY','rank':rank,'index':index,'key':record['key']}),flush=True)
    dist.barrier()


def finalize(args,frozen,expected):
    records=frozen['records']; directory=args.output_dir
    if any(not (directory/'features'/f'features_{i:03d}.json').exists() for i in range(100)) or not audit.ranks_complete(directory):
        return None
    checks=audit.verify(directory,expected,records)
    if any(c.get('capture_native_logits_bitwise_equal') is not True for c in checks['rank_checks']):
        raise ValueError('missing actual capture/native equivalence check')
    features=np.stack([feature_entry(directory/'features',r,expected)[0].numpy() for r in records],axis=2)
    # [view,layer,sample,hidden], merged in global index order before centering.
    np.savez(directory/'pooled_features.npz',**{name:features[i] for i,name in enumerate(capture.VIEWS)})
    matrices={}
    for i,name in enumerate(capture.VIEWS):
        for debiased,label in ((False,'standard'),(True,'debiased')):
            key=f'{name}_{label}'; value=metrics.matrix(features[i],debiased)
            mean,std=metrics.permutation_null(features[i],debiased,args.permutation_repeats)
            matrices[key]=value
            for suffix,array in (('',value),('_permutation_mean',mean),('_permutation_std',std)):
                np.savetxt(directory/(key+suffix+'.csv'),array,delimiter=',',fmt='%.12g')
            print(json.dumps({'event':'CKA_MATRIX_DONE','view':name,'estimator':label,'samples':100}),flush=True)
    np.savez(directory/'cka_matrices.npz',**matrices)
    plots=plot(directory,matrices)
    artifacts=[p for p in directory.iterdir() if p.suffix in ('.npz','.csv','.png','.svg')]
    storage.write_json(directory/'artifacts.json',storage.seal({'identity_sha256':expected['identity_sha256'],
        'files':{p.name:storage.file_sha(p) for p in sorted(artifacts)}}))
    return {'status':'PASS','rows':100,'layers':30,'views':list(capture.VIEWS),'matrices':6,
            'global_input_order_sha256':storage.object_sha([r['key'] for r in records]),
            'runtime_checks_sha256':checks['sha256'],'plots':plots}


def plot(directory,matrices):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError: return {'status':'OPTIONAL_MATPLOTLIB_MISSING','data':'cka_matrices.npz'}
    for name,value in matrices.items():
        fig,ax=plt.subplots(figsize=(6,5.2),constrained_layout=True)
        im=ax.imshow(value,origin='lower',extent=(.5,30.5,.5,30.5),vmin=-1 if name.endswith('debiased') else 0,vmax=1,cmap='coolwarm')
        ax.set(xlabel='Original layer (1-based)',ylabel='Original layer (1-based)',title=name.replace('_',' '))
        fig.colorbar(im,ax=ax,label='CKA'); fig.savefig(directory/(name+'.png'),dpi=180); fig.savefig(directory/(name+'.svg')); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True)
    for name,value in matrices.items(): ax.plot(range(1,30),np.diag(value,1),label=name,linestyle='--' if name.endswith('standard') else '-')
    ax.set(xlabel='Layer L (compared with L+1)',ylabel='Adjacent-layer CKA'); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.savefig(directory/'adjacent_cka.png',dpi=180); fig.savefig(directory/'adjacent_cka.svg'); plt.close(fig)
    return {'status':'PASS','heatmaps':6,'adjacent_layer_curves':1}
