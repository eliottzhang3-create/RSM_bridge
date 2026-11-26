"""CPU plotting of saved conditional RR results; no model import."""
import argparse
from pathlib import Path
from mizar_layer_pair_search import storage
from . import config


def make_plot(directory):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scores = storage.read_json(directory/'layer_scores.json')
    if [s['layer_1based'] for s in scores] != list(range(1, 31)):
        raise ValueError('need all thirty layer summaries before plotting')
    for s in scores:
        storage.verify_seal(s)
    x = np.arange(1, 31); files = []
    def save(fig, name):
        for ext in ('png', 'svg'):
            path = directory/(name+'.'+ext)
            fig.savefig(path, dpi=180, bbox_inches='tight'); files.append(str(path))
        plt.close(fig)
    def decorate(ax):
        ax.set_xlim(1, 30); ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.set_xlabel('Patched layer (1-based block output)'); ax.grid(alpha=.2)
    fig, ax = plt.subplots(figsize=(10, 5))
    mean = [s['mean_RR'] for s in scores]; ci = np.asarray([s['mean_RR_CI95'] for s in scores])
    ax.plot(x, mean, marker='o', markersize=3, label='Mean RR'); ax.fill_between(x, ci[:, 0], ci[:, 1], alpha=.18, label='95% bootstrap CI')
    ax.plot(x, [s['median_RR'] for s in scores], linestyle='--', label='Median RR')
    ax.axhline(0, color='grey', lw=.8); ax.axhline(1, color='grey', lw=.8, linestyle=':')
    ax.set_ylabel('Recovery rate (unclipped)'); decorate(ax); ax.legend()
    ax.set_title('Mizar / ReasonAQA: silence to clean text-state recovery\n500 selected: clean correct, silence wrong, positive probability gap')
    fig.text(.5, -.015, 'Layer 30 is a readout restoration control; bands are pointwise. Mean is per-question RR.', ha='center', fontsize=9)
    save(fig, 'recovery_rate')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for c in config.CATEGORIES:
        n = scores[0]['groups'][c]['total']
        if n:
            axes[0].plot(x, [s['groups'][c]['mean_RR'] for s in scores], label=f'{c} (n={n})')
            axes[1].plot(x, [s['groups'][c]['patched_accuracy'] for s in scores], label=c)
    for ax in axes:
        decorate(ax); ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean RR (unclipped)'); axes[1].set_ylabel('Patched classification accuracy')
    axes[1].set_ylim(-.02, 1.02); fig.suptitle('Conditional selected subset by ReasonAQA source category')
    fig.tight_layout(); save(fig, 'recovery_by_category')
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for field, label in [('mean_Pclean', 'Clean'), ('mean_Pcorrupted', 'Silence'), ('mean_Ppatched', 'Patched')]:
        ax.plot(x, [s[field] for s in scores], label=label)
    decorate(ax); ax.set_ylabel('Mean target full-vocabulary probability'); ax.legend()
    ax.set_title('Target probabilities (not candidate-renormalized)'); save(fig, 'target_probability')
    columns = []
    for layer in range(1, 31):
        import json
        path = directory/'layers'/f'patch_{layer:02d}'/'per_question.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if [r['selected_index'] for r in rows] != list(range(config.TARGET)):
            raise ValueError('heatmap needs exact selected question order')
        columns.append([r['RR'] for r in rows])
    matrix = np.asarray(columns, dtype=np.float64).T
    fig, ax = plt.subplots(figsize=(11, 6))
    extent = float(max(1, np.abs(matrix).max()))
    im = ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap='coolwarm', vmin=-extent, vmax=extent,
                   extent=(.5, 30.5, config.TARGET-.5, -.5))
    ax.set_xlabel('Patched layer (1-based)'); ax.set_ylabel('Selected question (source row order)')
    ax.set_title('Per-question RR (linear color scale; no clipping)')
    fig.colorbar(im, ax=ax, label='RR'); save(fig, 'recovery_heatmap')
    return files


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output-dir', type=Path, default=config.OUTPUT)
    args = p.parse_args(argv); print(make_plot(args.output_dir.resolve())); return 0


if __name__ == '__main__':
    raise SystemExit(main())
