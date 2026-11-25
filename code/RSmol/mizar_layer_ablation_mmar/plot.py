"""CPU-only scientific figures from saved primary metrics; no model imports."""
from pathlib import Path
import argparse
import json
from mizar_layer_pair_search import storage
from . import config


def make_plot(directory, summaries=None):
    if summaries is None:
        summaries = storage.read_json(directory/'condition_scores.json')
    if [row['drop_layer_1based'] for row in summaries] != list(config.CONDITIONS):
        raise ValueError('plot requires baseline then all original layers 2..29')
    for row in summaries:
        if row['total'] != config.ROWS or {k: v['total'] for k, v in row['groups']['category'].items()} != config.CATEGORY_COUNTS:
            raise ValueError('plot category denominators changed')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return {'status': 'OPTIONAL_MATPLOTLIB_MISSING', 'data': 'accuracy_by_category.csv'}
    x = [row['drop_layer_1based'] for row in summaries[1:]]
    colors = ('#2563eb', '#059669', '#d97706', '#9333ea')
    artifacts = []

    def save(fig, name):
        for ext in ('png', 'svg'):
            filename = f'{name}.{ext}'
            fig.savefig(directory/filename, dpi=180)
            artifacts.append(filename)
        plt.close(fig)

    def style(ax):
        ax.set_xlabel('Skipped original decoder layer (1-based)')
        ax.set_xticks([2, *range(5, 30, 5), 29])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    base = summaries[0]['accuracy']*100
    y = [row['accuracy']*100 for row in summaries[1:]]
    axes[0].plot(x, y, marker='o', markersize=3, color='#111827', label='MMAR overall (n=1000)')
    axes[0].axhline(base, color='#dc2626', linestyle='--', label=f'30-layer baseline: {base:.1f}%')
    axes[0].set(title='MMAR single-layer identity ablation', ylabel='Primary accuracy (%)', ylim=(0, 100))
    axes[1].plot(x, [value-base for value in y], marker='o', markersize=3, color='#111827', label='Overall change')
    axes[1].axhline(0, color='#6b7280', linestyle='--')
    axes[1].set(title='Change from complete 30-layer baseline', ylabel='Accuracy change (percentage points)')
    for ax in axes:
        style(ax)
    save(fig, 'accuracy_by_layer')
    for delta, filename in ((False, 'accuracy_by_category'), (True, 'delta_by_category')):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        fig.suptitle('MMAR primary accuracy: single-layer identity ablation' if not delta
                     else 'MMAR primary accuracy change from full 30-layer baseline')
        for ax, category, color in zip(axes.flat, config.CATEGORIES, colors):
            baseline = summaries[0]['groups']['category'][category]['accuracy']*100
            y = [row['groups']['category'][category]['accuracy']*100 for row in summaries[1:]]
            ax.plot(x, [v-baseline for v in y] if delta else y, color=color,
                    marker='o', markersize=3, label='Ablation change' if delta else 'Single-layer ablation')
            ax.axhline(0 if delta else baseline, color='#6b7280', linestyle='--',
                       label='No change' if delta else f'30-layer baseline: {baseline:.2f}%')
            ax.set_title(f'{category} (n={config.CATEGORY_COUNTS[category]})')
            ax.set_ylabel('Change (percentage points)' if delta else 'Accuracy (%)')
            if not delta:
                ax.set_ylim(0, 100)
            style(ax)
        save(fig, filename)
    return {'status': 'PASS', 'primary_metric_only': True, 'files': artifacts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    result = make_plot(args.output_dir)
    print(json.dumps(result))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
