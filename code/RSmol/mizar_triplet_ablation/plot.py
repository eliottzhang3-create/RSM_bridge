"""CPU window accuracy and baseline changes with explicit start/end labels."""
import argparse
import json
from pathlib import Path
from mizar_layer_pair_search import storage
from . import config


def make_plot(directory):
    scores=storage.read_json(directory/'condition_scores.json');task=scores[0]['task']
    if [s['window_start_1based'] for s in scores]!=list(config.CONDITIONS):raise ValueError('need baseline plus20 windows')
    for row in scores:
        if row['total']!=1000 or {g:v['total'] for g,v in row['groups'].items()}!=config.GROUP_COUNTS[task]:raise ValueError('group denominators changed')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files=[];x=list(range(6,26));colors=('#2563eb','#059669','#d97706','#9333ea')
    title=f'{"MMAU test-mini" if task=="mmau" else "MMAR"}: three-consecutive-layer ablation'
    def style(ax):
        ticks=[6,9,12,15,18,21,25];ax.set_xticks(ticks,[f'{i}–{i+2}' for i in ticks]);ax.set_xlim(5.7,25.3)
        ax.set_xlabel('Skipped original layer window (1-based; stride 1)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    def save(fig,stem):
        for ext in ('png','svg'):
            filename=f'{stem}.{ext}';fig.savefig(directory/filename,dpi=180,bbox_inches='tight');files.append(filename)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    baseline=scores[0]['accuracy']*100;y=[s['accuracy']*100 for s in scores[1:]]
    axes[0].plot(x,y,marker='o',ms=3,color='#111827',label='Overall (n=1000)')
    axes[0].axhline(baseline,color='grey',ls='--',label=f'Full30 baseline: {baseline:.1f}%')
    axes[0].set(ylabel='Letter accuracy (%)',ylim=(0,100),title=title)
    axes[1].plot(x,[v-baseline for v in y],marker='o',ms=3,color='#111827',label='Overall change')
    axes[1].axhline(0,color='grey',ls='--');axes[1].set(ylabel='Change (percentage points)',title='Change vs full30 baseline')
    for ax in axes:style(ax)
    save(fig,'accuracy_by_window')
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for g,color in zip(config.GROUP_COUNTS[task],colors):
        base=scores[0]['groups'][g]['accuracy']*100;y=[s['groups'][g]['accuracy']*100 for s in scores[1:]]
        label=f'{g} (n={config.GROUP_COUNTS[task][g]})'
        axes[0].plot(x,y,color=color,marker='o',ms=3,label=label);axes[0].axhline(base,color=color,ls='--',alpha=.5)
        axes[1].plot(x,[v-base for v in y],color=color,marker='o',ms=3,label=label)
    axes[0].set(ylabel='Letter accuracy (%)',ylim=(0,100),title='Group accuracy; dashed = group baseline')
    axes[1].set(ylabel='Change (percentage points)',title='Change from each group baseline');axes[1].axhline(0,color='grey',ls='--')
    for ax in axes:style(ax)
    fig.suptitle(title+' / lowercase letter loglikelihood, prompt EOT');save(fig,'accuracy_by_group')
    if task=='mmar':
        for delta,stem in ((False,'accuracy_by_category'),(True,'delta_by_category')):
            fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
            for ax,g,color in zip(axes.flat,config.GROUP_COUNTS[task],colors):
                base=scores[0]['groups'][g]['accuracy']*100;y=[s['groups'][g]['accuracy']*100 for s in scores[1:]]
                ax.plot(x,[v-base for v in y] if delta else y,color=color,marker='o',ms=3,label='Change' if delta else 'Accuracy')
                ax.axhline(0 if delta else base,color='grey',ls='--',label='No change' if delta else f'Baseline {base:.2f}%')
                ax.set_title(f'{g} (n={config.GROUP_COUNTS[task][g]})');ax.set_ylabel('Change (pp)' if delta else 'Accuracy (%)')
                if not delta:ax.set_ylim(0,100)
                style(ax)
            fig.suptitle(title);save(fig,stem)
    return {'status':'PASS','files':files}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args(argv);print(json.dumps(make_plot(a.output_dir.resolve())));return 0


if __name__=='__main__':raise SystemExit(main())
