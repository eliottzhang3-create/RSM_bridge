"""Four figure types for dynamic subsets, with group boundaries and zero-N states."""
import argparse
import json
from pathlib import Path
from mizar_layer_pair_search import storage
from . import config


def make_plot(directory):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frozen=storage.verify_seal(storage.read_json(directory/'selected/manifest.json'),'manifest_sha256')
    scores=storage.read_json(directory/'layer_scores.json');n=frozen['selected'];task=frozen['task'];files=[]
    if [s['layer_1based'] for s in scores]!=(list(range(1,31)) if n else []):raise ValueError('incomplete plotting layer coverage')
    for s in scores:storage.verify_seal(s)
    x=np.arange(1,31);title=f'Mizar / {"MMAU test-mini" if task=="mmau" else "MMAR"} (selected n={n})'
    def save(fig,name):
        for ext in ('png','svg'):
            path=directory/(name+'.'+ext);fig.savefig(path,dpi=180,bbox_inches='tight');files.append(str(path))
        plt.close(fig)
    def decorate(ax):
        ax.set_xlim(1,30);ax.set_xticks([1,5,10,15,20,25,30]);ax.set_xlabel('Patched layer (1-based block output)');ax.grid(alpha=.2)
    def no_data(ax,text='No qualifying questions'):
        ax.text(.5,.5,text,ha='center',va='center',transform=ax.transAxes)
    fig,ax=plt.subplots(figsize=(10,5));decorate(ax);ax.set_ylabel('Recovery rate (unclipped)')
    if n:
        ci=np.asarray([s['mean_RR_CI95'] for s in scores]);ax.plot(x,[s['mean_RR'] for s in scores],marker='o',ms=3,label='Mean RR')
        ax.fill_between(x,ci[:,0],ci[:,1],alpha=.18,label='95% question bootstrap CI')
        ax.plot(x,[s['median_RR'] for s in scores],'--',label='Median RR');ax.axhline(0,color='grey',lw=.8);ax.axhline(1,color='grey',ls=':',lw=.8);ax.legend()
    else:no_data(ax)
    ax.set_title(title+'\nClean correct, silence wrong, positive target-probability gap')
    fig.text(.5,-.02,'Layer30 is a readout restoration control; CI bands are pointwise.',ha='center',fontsize=9);save(fig,'recovery_rate')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for g in config.GROUPS[task]:
        count=frozen['group_counts'][g]
        if count:
            means=[s['groups'][g]['mean_RR'] for s in scores];ci=np.asarray([s['groups'][g]['mean_RR_CI95'] for s in scores])
            line=axes[0].plot(x,means,label=f'{g} (n={count})')[0]
            axes[0].fill_between(x,ci[:,0],ci[:,1],color=line.get_color(),alpha=.1)
            axes[1].plot(x,[s['groups'][g]['patched_accuracy'] for s in scores],label=f'{g} (n={count})')
        else:
            for ax in axes:ax.plot([],[],label=f'{g} (n=0; no eligible questions)')
    for ax in axes:decorate(ax);ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean RR (unclipped), 95% bootstrap CI');axes[1].set_ylabel('Patched classification accuracy');axes[1].set_ylim(-.02,1.02)
    if not n:
        for ax in axes:no_data(ax)
    fig.suptitle(title+' / conditional subset by dataset group');fig.tight_layout();save(fig,'recovery_by_category')
    fig,ax=plt.subplots(figsize=(10,5));decorate(ax);ax.set_ylabel('Mean target full-vocabulary probability')
    if n:
        for key,label in [('mean_Pclean','Clean'),('mean_Pcorrupted','Silence'),('mean_Ppatched','Patched')]:ax.plot(x,[s[key] for s in scores],label=label)
        ax.legend()
    else:no_data(ax)
    ax.set_title(title+' / target probabilities');save(fig,'target_probability')
    fig,ax=plt.subplots(figsize=(12,6))
    if n:
        columns=[]
        for layer in range(1,31):
            path=directory/'layers'/f'patch_{layer:02d}'/'per_question.jsonl'
            rows=[storage.verify_seal(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines()]
            if [r['selected_index'] for r in rows]!=list(range(n)):raise ValueError('heatmap question order changed')
            columns.append([r['RR'] for r in rows])
        # Within each group keep original source order; publish the row map.
        order=[r['selected_index'] for g in config.GROUPS[task] for r in frozen['records'] if r['analysis_group']==g]
        matrix=np.asarray(columns,dtype=np.float64).T[order];limit=max(1.,float(np.abs(matrix).max()))
        im=ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap='coolwarm',vmin=-limit,vmax=limit,extent=(.5,30.5,n-.5,-.5))
        positions=[];labels=[];offset=0
        for g in config.GROUPS[task]:
            count=frozen['group_counts'][g]
            if count:
                positions.append(offset+(count-1)/2);labels.append(f'{g} (n={count})');offset+=count
                if offset<n:ax.axhline(offset-.5,color='black',lw=.6)
        ax.set_yticks(positions,labels);fig.colorbar(im,ax=ax,label='RR (linear scale, unclipped)')
        storage.write_jsonl(directory/'heatmap_row_map.jsonl',[{'heatmap_row':i,'selected_index':idx,
            'source_manifest_index':frozen['records'][idx]['source_manifest_index'],'key':frozen['records'][idx]['key'],
            'group':frozen['records'][idx]['analysis_group']} for i,idx in enumerate(order)])
    else:
        no_data(ax);storage.write_jsonl(directory/'heatmap_row_map.jsonl',[])
    ax.set_xlabel('Patched layer (1-based)');ax.set_ylabel('Questions grouped by category; original source order within group')
    ax.set_title(title+' / per-question RR');fig.tight_layout();save(fig,'recovery_heatmap')
    return files


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv);print(json.dumps(make_plot(a.output_dir.resolve())));return 0


if __name__=='__main__':raise SystemExit(main())
