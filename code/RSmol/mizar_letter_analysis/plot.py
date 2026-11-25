"""Scientific figures from saved lowercase-letter decisions; CPU only."""
import argparse
import json
from pathlib import Path
from mizar_layer_pair_search import storage
from . import config


def make_plot(directory):
    scores=storage.read_json(directory/'condition_scores.json')
    task=scores[0]['task']
    if [r['condition_layer_1based'] for r in scores]!=list(config.conditions(task)):
        raise ValueError('plot requires all conditions in baseline-first order')
    for row in scores:
        if row['total']!=1000 or {g:v['total'] for g,v in row['groups'].items()}!=config.GROUP_COUNTS[task]:
            raise ValueError('plot group denominators changed')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return {'status':'OPTIONAL_MATPLOTLIB_MISSING','data':'accuracy_by_group.csv'}
    x=[r['condition_layer_1based'] for r in scores[1:]]
    colors=('#2563eb','#059669','#d97706','#9333ea')
    names={'cutoff':'Audio + SEP cutoff','mmau':'MMAU single-layer ablation','mmar':'MMAR single-layer ablation'}
    files=[]

    def finish(fig,stem):
        for extension in ('png','svg'):
            name=f'{stem}.{extension}';fig.savefig(directory/name,dpi=180);files.append(name)
        plt.close(fig)

    def style(axis):
        axis.set_xlabel('Audio + SEP cutoff layer (1-based)' if task=='cutoff' else 'Skipped original decoder layer (1-based)')
        axis.set_xticks([x[0],*range(5,x[-1],5),x[-1]])
        axis.grid(alpha=.2);axis.legend(fontsize=8)

    fig,axes=plt.subplots(1,2,figsize=(13,4.7),constrained_layout=True)
    baseline=scores[0]['accuracy']*100;y=[r['accuracy']*100 for r in scores[1:]]
    axes[0].plot(x,y,color='#111827',marker='o',markersize=3,label='Overall (n=1000)')
    axes[0].axhline(baseline,linestyle='--',color='#6b7280',label=f'Full 30-layer baseline: {baseline:.1f}%')
    axes[0].set(ylabel='Letter accuracy (%)',ylim=(0,100),title=names[task])
    axes[1].plot(x,[v-baseline for v in y],color='#111827',marker='o',markersize=3,label='Overall change')
    axes[1].axhline(0,color='#6b7280',linestyle='--')
    axes[1].set(ylabel='Change (percentage points)',title='Relative to full 30-layer baseline')
    for axis in axes:style(axis)
    finish(fig,'accuracy_by_layer')

    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for group,color in zip(config.GROUP_COUNTS[task],colors):
        base=scores[0]['groups'][group]['accuracy']*100
        y=[r['groups'][group]['accuracy']*100 for r in scores[1:]]
        label=f'{group} (n={config.GROUP_COUNTS[task][group]})'
        axes[0].plot(x,y,color=color,marker='o',markersize=3,label=label)
        axes[0].axhline(base,color=color,linestyle='--',alpha=.5)
        axes[1].plot(x,[v-base for v in y],color=color,marker='o',markersize=3,label=label)
    axes[0].set(ylabel='Letter accuracy (%)',ylim=(0,100),title='Group accuracy; dashed lines are each baseline')
    axes[1].set(ylabel='Change (percentage points)',title='Change from each group baseline')
    axes[1].axhline(0,color='#6b7280',linestyle='--')
    fig.suptitle(names[task]+' — lowercase-letter loglikelihood, prompt EOT')
    for axis in axes:style(axis)
    finish(fig,'psr_accuracy_by_layer' if task=='mmau' else 'accuracy_by_group')

    if task=='mmar':
        for delta,stem in ((False,'accuracy_by_category'),(True,'delta_by_category')):
            fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
            for axis,group,color in zip(axes.flat,config.GROUP_COUNTS[task],colors):
                base=scores[0]['groups'][group]['accuracy']*100
                y=[r['groups'][group]['accuracy']*100 for r in scores[1:]]
                axis.plot(x,[v-base for v in y] if delta else y,color=color,marker='o',markersize=3,label='Change' if delta else 'Letter accuracy')
                axis.axhline(0 if delta else base,color='#6b7280',linestyle='--',label='No change' if delta else f'Baseline: {base:.2f}%')
                axis.set_title(f'{group} (n={config.GROUP_COUNTS[task][group]})')
                axis.set_ylabel('Change (percentage points)' if delta else 'Letter accuracy (%)')
                if not delta:axis.set_ylim(0,100)
                style(axis)
            fig.suptitle('MMAR lowercase-letter likelihood, EOT — '+('baseline change' if delta else 'single-layer ablation'))
            finish(fig,stem)
    return {'status':'PASS','files':files,'primary_metric':'lowercase_letter_likelihood'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--task',choices=config.TASKS)
    args=parser.parse_args(argv)
    result=make_plot(args.output_dir)
    print(json.dumps(result))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
