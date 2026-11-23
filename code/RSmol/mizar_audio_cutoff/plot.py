"""Standalone scientific curves; numerical results do not require matplotlib."""
from pathlib import Path
from mizar_layer_pair_search import storage
from . import config


def make_plot(directory):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return {'status':'OPTIONAL_DEPENDENCY_MISSING', 'dependency':'matplotlib',
                'data':'accuracy_by_cutoff.csv', 'command':'python -m mizar_audio_cutoff.plot --output-dir <output>'}
    scores = storage.read_json(directory/'condition_scores.json')
    if [row['cutoff_layer_1based'] for row in scores] != list(config.CONDITIONS):
        raise ValueError('plot requires all nineteen conditions')
    baseline, rows = scores[0], scores[1:]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    colors = ['#111827','#2563eb','#059669','#d97706','#9333ea']
    x = [row['cutoff_layer_1based'] for row in rows]
    for name, color in zip(('overall', *config.DATASETS), colors):
        def accuracy(row):
            return row['accuracy'] if name == 'overall' else row['subsets'][name]['accuracy']
        y, base = [accuracy(row)*100 for row in rows], accuracy(baseline)*100
        axes[0].plot(x, y, marker='o', markersize=3, label=name, color=color)
        axes[0].axhline(base, color=color, linewidth=.8, linestyle='--', alpha=.5)
        axes[1].plot(x, [v-base for v in y], marker='o', markersize=3, label=name, color=color)
    axes[0].set(title='Accuracy (dashed: full-audio baseline)', ylabel='Accuracy (%)', ylim=(0,100))
    axes[1].set(title='Change relative to baseline', ylabel='Accuracy change (percentage points)')
    axes[1].axhline(0, color='#6b7280', linewidth=.8, linestyle='--')
    for axis in axes:
        axis.set_xlabel('Audio + SEP cutoff layer (1-based; mask before layer)')
        axis.set_xticks(list(range(3,21,2)))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(directory/'accuracy_curves.svg')
    fig.savefig(directory/'accuracy_curves.png', dpi=180)
    plt.close(fig)
    return {'status':'PASS', 'svg':'accuracy_curves.svg', 'png':'accuracy_curves.png'}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = make_plot(args.output_dir)
    print(result)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
