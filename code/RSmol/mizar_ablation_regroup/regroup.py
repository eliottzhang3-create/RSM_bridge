"""Read-only source results, count-weighted PSR regrouping, and CPU plotting."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import traceback

from mizar_layer_pair_search import storage

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
DEFAULT_SOURCE = Path('/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/mizar_ablation_bf16_8gpu3090_v1')
CONDITIONS = (None, *range(2, 30))
GROUPS = ('perception', 'semantic', 'reasoning')
SCHEMA = 'mizar_ablation_psr_count_weighted_v1'


def condition_name(layer):
    return 'baseline' if layer is None else f'ablate_{layer:02d}'


def read_mapping():
    mapping = storage.read_json(PACKAGE/'mapping.json')
    if set(mapping['groups']) != set(GROUPS):
        raise ValueError('mapping must contain perception, semantic and reasoning')
    names = []
    for group, count in zip(GROUPS, (8, 8, 11)):
        subcategories = mapping['groups'][group]['subcategories']
        if len(subcategories) != count or any(not isinstance(s, str) or not s.strip() for s in subcategories):
            raise ValueError(f'invalid {group} mapping')
        names.extend(subcategories)
    if len(names) != len(set(names)):
        raise ValueError('a subcategory was assigned to more than one group')
    return mapping


def check_accuracy(value, expected, context):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)):
        raise ValueError(f'inconsistent accuracy: {context}')


def check_counts(row, context):
    correct, total = row.get('correct'), row.get('total')
    if type(correct) is not int or type(total) is not int or not 0 <= correct <= total or total <= 0:
        raise ValueError(f'invalid correct/total counts: {context}')
    check_accuracy(row.get('accuracy'), correct/total, context)


def load_completed(source, mapping):
    """Read summaries only; no source file is opened for writing or repair."""
    identities = []
    def read(relative):
        path = source/relative
        before = storage.file_identity(path)
        value = storage.read_json(path)
        storage.verify_file(before)
        identities.append(before)
        return value

    expected = storage.verify_seal(read('run_config.json'), 'identity_sha256')
    report = read('analysis_report.json')
    run_sha = expected['identity_sha256']
    if (expected.get('task') != 'ablation' or expected.get('protocol', {}).get('rows') != 1000
            or expected['protocol'].get('conditions') != list(CONDITIONS)
            or report.get('status') != 'PASS' or report.get('conditions_completed') != 29
            or report.get('decisions') != 29000 or report.get('rows_per_condition') != 1000
            or report.get('identity_sha256') != run_sha):
        raise ValueError('source must be a completed 29-condition / 29000-decision ablation run')
    scores = read('condition_scores.json')
    if (not isinstance(scores, list) or len(scores) != 29
            or [s.get('drop_layer_1based') for s in scores] != list(CONDITIONS)):
        raise ValueError('condition_scores.json lacks baseline and exact layers 2..29')
    baseline_copy = read('baseline.json')
    if baseline_copy != scores[0]:
        raise ValueError('baseline.json differs from condition_scores.json')
    names = {name for group in GROUPS for name in mapping['groups'][group]['subcategories']}
    summaries, baseline_counts = [], None
    for layer, combined in zip(CONDITIONS, scores):
        name = condition_name(layer)
        summary = storage.verify_seal(read(Path('conditions')/name/'summary.json'))
        if {k:v for k,v in summary.items() if k != 'sha256'} != combined:
            raise ValueError(f'per-condition summary differs from combined scores: {name}')
        if (summary.get('condition') != name or summary.get('drop_layer_1based') != layer
                or summary.get('identity_sha256') != run_sha or summary.get('total') != 1000):
            raise ValueError(f'wrong condition/identity/denominator: {name}')
        check_counts(summary, name)
        subsets = summary.get('groups', {}).get('sub_category', {})
        if set(subsets) != names:
            raise ValueError(f'{name}: unknown subcategories={sorted(set(subsets)-names)}; missing={sorted(names-set(subsets))}')
        for subcategory, row in subsets.items():
            check_counts(row, f'{name}/{subcategory}')
        counts = {subcategory:row['total'] for subcategory,row in subsets.items()}
        if baseline_counts is None:
            baseline_counts = counts
        if counts != baseline_counts:
            raise ValueError(f'subcategory denominators changed: {name}')
        if sum(counts.values()) != 1000 or sum(row['correct'] for row in subsets.values()) != summary['correct']:
            raise ValueError(f'subcategory counts do not reconstruct overall score: {name}')
        baseline = subsets if layer is None else summaries[0]['groups']['sub_category']
        check_accuracy(summary.get('baseline_accuracy'), scores[0]['accuracy'], name+'/baseline')
        check_accuracy(summary.get('delta_vs_baseline'), (summary['correct']-scores[0]['correct'])/1000, name+'/delta')
        for subcategory, row in subsets.items():
            check_accuracy(row.get('delta_vs_baseline'),
                           (row['correct']-baseline[subcategory]['correct'])/row['total'], name+'/'+subcategory+'/delta')
        if summary.get('baseline_rows_sha256') != scores[0].get('raw_rows_sha256'):
            raise ValueError(f'condition points to another baseline: {name}')
        summaries.append(summary)
    return summaries, identities, run_sha


def aggregate(summaries, mapping):
    rows = []
    baseline = summaries[0]['groups']['sub_category']
    for summary in summaries:
        subcategories = summary['groups']['sub_category']
        grouped = {}
        for group in GROUPS:
            names = mapping['groups'][group]['subcategories']
            correct = sum(subcategories[name]['correct'] for name in names)
            total = sum(subcategories[name]['total'] for name in names)
            base_correct = sum(baseline[name]['correct'] for name in names)
            grouped[group] = {'correct':correct, 'total':total, 'accuracy':correct/total,
                'accuracy_percent':100*correct/total, 'baseline_correct':base_correct,
                'baseline_accuracy':base_correct/total, 'delta_vs_baseline':(correct-base_correct)/total,
                'delta_percentage_points':100*(correct-base_correct)/total, 'subcategory_count':len(names)}
        if sum(g['correct'] for g in grouped.values()) != summary['correct'] or sum(g['total'] for g in grouped.values()) != 1000:
            raise ValueError('PSR aggregation did not preserve the complete denominator')
        rows.append({'condition':summary['condition'], 'drop_layer_1based':summary['drop_layer_1based'],
                     'overall_accuracy':summary['accuracy'], 'groups':grouped})
    return rows


def write_tables(directory, rows, mapping, baseline_subcategories):
    storage.write_json(directory/'group_mapping.json', mapping)
    storage.write_json(directory/'psr_scores.json', rows)
    with (directory/'psr_scores.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['condition', 'drop_layer_1based', 'overall_accuracy']
        fields += [f'{group}_{suffix}' for group in GROUPS for suffix in
                   ('correct', 'total', 'accuracy', 'accuracy_percent', 'baseline_accuracy', 'delta_percentage_points')]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k:row[k] for k in fields[:3]},
                **{f'{group}_{suffix}':row['groups'][group][suffix] for group in GROUPS for suffix in
                   ('correct', 'total', 'accuracy', 'accuracy_percent', 'baseline_accuracy', 'delta_percentage_points')}})
    with (directory/'subcategory_mapping.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['group','subcategory','total','baseline_correct','baseline_accuracy'])
        writer.writeheader()
        for group in GROUPS:
            for name in mapping['groups'][group]['subcategories']:
                counts = baseline_subcategories[name]
                writer.writerow({'group':group, 'subcategory':name, 'total':counts['total'],
                                 'baseline_correct':counts['correct'], 'baseline_accuracy':counts['accuracy']})


def draw(directory, rows, mapping):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False}):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
        colors = {'perception':'#2563eb', 'semantic':'#059669', 'reasoning':'#c026d3'}
        layers = [row['drop_layer_1based'] for row in rows[1:]]
        for group in GROUPS:
            base = rows[0]['groups'][group]
            values = [row['groups'][group] for row in rows[1:]]
            label = f"{mapping['groups'][group]['label']} (n={base['total']})"
            axes[0].plot(layers, [v['accuracy_percent'] for v in values], color=colors[group], marker='o', markersize=3,
                         linewidth=1.8, label=label)
            axes[0].axhline(base['accuracy_percent'], color=colors[group], linestyle='--', linewidth=1.3, alpha=.8,
                           label=f"{group} baseline: {base['accuracy_percent']:.2f}%")
            axes[1].plot(layers, [v['delta_percentage_points'] for v in values], color=colors[group], marker='o',
                         markersize=3, linewidth=1.8, label=label)
        axes[0].set(title='Accuracy by ability group', ylabel='Accuracy (%)', ylim=(0, 100))
        axes[1].set(title='Change relative to each group baseline', ylabel='Accuracy change (percentage points)')
        axes[1].axhline(0, color='#6b7280', linewidth=1, linestyle='--')
        for ax in axes:
            ax.set_xlabel('Skipped original layer (1-based)')
            ax.set_xticks([2, *range(5, 30, 3), 29])
            ax.set_xlim(1.5, 29.5)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc='best')
        fig.suptitle('Mizar single-layer ablation — MMAU test-mini, 1000 questions\n'
                     'User-defined groups; question-weighted accuracy from saved likelihood decisions', fontsize=12)
        try:
            for suffix in ('png', 'svg'):
                fig.savefig(directory/f'psr_accuracy_by_layer.{suffix}', dpi=200)
        finally:
            plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-args', action='store_true')
    args = parser.parse_args(argv)
    args.source_dir = args.source_dir.expanduser().resolve()
    args.output_dir = (args.output_dir or args.source_dir.with_name(args.source_dir.name+'_psr_v1')).expanduser().resolve()
    if args.output_dir.is_relative_to(args.source_dir) or args.source_dir.is_relative_to(args.output_dir):
        parser.error('source and output must be separate directories, not parent/child')
    return args


def run(args):
    mapping = read_mapping()
    summaries, inputs, source_sha = load_completed(args.source_dir, mapping)
    rows = aggregate(summaries, mapping)
    paths = [*PACKAGE.glob('*.py'), PACKAGE/'mapping.json', ROOT/'run_mizar_ablation_regroup_cpu.sh']
    identity = storage.seal({'schema':SCHEMA, 'source_dir':str(args.source_dir), 'source_identity_sha256':source_sha,
        'source_files':inputs, 'mapping_sha256':storage.object_sha(mapping),
        'implementation':{p.relative_to(ROOT).as_posix():storage.text_sha(p) for p in sorted(paths)}}, 'identity_sha256')
    with storage.output_lock(args.output_dir):
        storage.claim(args.output_dir, identity)
        report = {'schema':SCHEMA, 'status':'FAILED', 'identity_sha256':identity['identity_sha256'],
                  'source_dir':str(args.source_dir), 'output_dir':str(args.output_dir), 'source_results_modified':False,
                  'scoring':'sum correct / sum total within user-defined subcategory groups; saved likelihood decisions unchanged'}
        try:
            write_tables(args.output_dir, rows, mapping, summaries[0]['groups']['sub_category'])
            draw(args.output_dir, rows, mapping)
            for item in inputs:
                storage.verify_file(item)
            files = ['psr_scores.json','psr_scores.csv','subcategory_mapping.csv','group_mapping.json',
                     'psr_accuracy_by_layer.png','psr_accuracy_by_layer.svg']
            report.update(status='PASS', conditions=29, rows_per_condition=1000,
                group_totals={group:rows[0]['groups'][group]['total'] for group in GROUPS},
                baseline={group:rows[0]['groups'][group] for group in GROUPS},
                artifacts={name:storage.file_sha(args.output_dir/name) for name in files})
        except Exception as exc:
            report.update(error=repr(exc), traceback=traceback.format_exc())
        storage.write_json(args.output_dir/'regroup_report.json', report)
    for row in rows:
        print(json.dumps({'event':'PSR_SCORE', **row}, ensure_ascii=False), flush=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main(argv=None):
    args = parse_args(argv)
    if args.check_args:
        print(json.dumps(vars(args), default=str, indent=2))
        return 0
    try:
        return 0 if run(args)['status'] == 'PASS' else 1
    except Exception as exc:
        print(json.dumps({'status':'FAILED', 'error':repr(exc), 'traceback':traceback.format_exc()}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
