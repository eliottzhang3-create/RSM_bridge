"""Add taxonomy/task breakdowns without changing MMSU generation or grading."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from mizar_s1_ced_mmsu_eval import dataset, parsing, rescore, scoring, storage

VERSION = 'mmsu_detailed_answer_text_v2_v1'
SUBDIRECTORY = 'detailed_mmsu_answer_text_v2'
TAXONOMY = ('category', 'sub-category', 'sub-sub-category')
VIEWS = {
    'category': ('category',),
    'category_subcategory': TAXONOMY[:2],
    'taxonomy': TAXONOMY,
    'discipline': (*TAXONOMY, 'linguistics_sub_discipline'),
    'task': ('task_name',),
    'task_taxonomy': (*TAXONOMY, 'linguistics_sub_discipline', 'task_name'),
    'leaderboard_domain': ('category', 'leaderboard_domain'),
}
TITLES = {
    'category': 'Category', 'category_subcategory': 'Category / Sub-category',
    'taxonomy': 'Three-level taxonomy', 'discipline': 'Linguistics sub-discipline',
    'task': 'Task', 'task_taxonomy': 'Task with full taxonomy',
    'leaderboard_domain': 'Leaderboard domains: Semantics / Phonology / Paralinguistics',
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-output-dir', type=Path, action='append', required=True,
                        help='original full MMSU output; repeat to process multiple architectures')
    parser.add_argument('--output-dir', type=Path, help='single source only; default: independent source subdirectory')
    parser.add_argument('--evaluation-script', type=Path, default=rescore.DEFAULT_SCORER)
    parser.add_argument('--check-args', action='store_true')
    args = parser.parse_args(argv)
    args.source_output_dir = [p.resolve() for p in args.source_output_dir]
    if len(set(args.source_output_dir)) != len(args.source_output_dir):
        parser.error('duplicate source directories')
    if args.output_dir and len(args.source_output_dir) != 1:
        parser.error('--output-dir applies to a single source only')
    args.evaluation_script = args.evaluation_script.resolve()
    if args.output_dir:
        args.output_dir = args.output_dir.resolve()
    for source, output in planned(args):
        if output == source or output in source.parents:
            parser.error('detailed reports need a separate directory, never the source or its ancestor')
    return args


def planned(args):
    return [(source, args.output_dir or source / SUBDIRECTORY) for source in args.source_output_dir]


def counts(correct, total):
    return {'correct': correct, 'incorrect': total - correct, 'total': total,
            'accuracy': correct / total, 'accuracy_percent': 100 * correct / total}


def leaderboard_domain(row):
    # Preserve original labels in the other views. Only this derived view
    # merges paralinguistic third-level fields, as the official table does.
    if row['sub-category'].strip().casefold() == 'paralinguistics':
        return 'Paralinguistics'
    if row['sub-category'].strip().casefold() == 'linguistics':
        value = row['sub-sub-category'].strip().casefold()
        if value in ('semantics', 'phonology'):
            return value.title()
    return 'Unmapped'


def aggregate(records):
    parsing.validate_predictions(records)
    tables = {view: defaultdict(lambda: [0, 0]) for view in VIEWS}
    total_correct = 0
    empty_labels, unmapped = Counter(), Counter()
    for row in records:
        correct = bool(parsing.parse_response(row['response'], row['answer_gt'])['matched'])
        total_correct += correct
        labels = {**row, 'leaderboard_domain': leaderboard_domain(row)}
        if labels['leaderboard_domain'] == 'Unmapped':
            unmapped[tuple(row[key] for key in TAXONOMY)] += 1
        empty_labels.update(key for key in (*TAXONOMY, 'linguistics_sub_discipline', 'task_name')
                            if not row[key].strip())
        for view, fields in VIEWS.items():
            bucket = tables[view][tuple(labels[key] for key in fields)]
            bucket[0] += correct
            bucket[1] += 1
    groups = {view: [{'labels': dict(zip(VIEWS[view], key)), **counts(*value)}
                     for key, value in sorted(table.items())] for view, table in tables.items()}
    coverage = {view: {'correct': sum(r['correct'] for r in rows),
                       'total': sum(r['total'] for r in rows), 'groups': len(rows)}
                for view, rows in groups.items()}
    if any(value['correct'] != total_correct or value['total'] != dataset.EXPECTED_ROWS
           for value in coverage.values()):
        raise ValueError('a detailed grouping dropped or double-counted rows')
    task_count = len(groups['task'])
    warnings = []
    if task_count != 47:
        warnings.append(f'Observed {task_count} task_name values; official README describes 47 tasks. No rows omitted.')
    if empty_labels:
        warnings.append('Empty classification labels retained as explicit empty-string groups.')
    if unmapped:
        warnings.append('Unknown leaderboard taxonomy retained in Unmapped; inspect taxonomy view for exact labels.')
    return {'overall': counts(total_correct, len(records)), 'groups': groups,
            'group_coverage': coverage, 'aggregation': 'micro_accuracy_correct_divided_by_all_rows_in_group',
            'taxonomy_audit': {'task_count': task_count, 'official_readme_task_count': 47,
                               'empty_label_rows': dict(empty_labels),
                               'unmapped_leaderboard_rows': [
                                   {'labels': dict(zip(TAXONOMY, key)), 'total': value}
                                   for key, value in sorted(unmapped.items())]},
            'warnings': warnings}


def check_official(details, official):
    overall = details['overall']
    if (official.get('status') != 'PASS' or official.get('actual_denominator') != overall['total']
            or official.get('overall_accuracy') != overall['accuracy']):
        raise ValueError('detailed totals disagree with the pinned official scorer on the v2 scoring copy')
    for group in details['groups']['category']:
        other = official['category_accuracy'][group['labels']['category']]
        if (other['total_correct'] != group['correct'] or other['total_count'] != group['total']
                or other['average_accuracy'] != group['accuracy']):
            raise ValueError('category counts disagree with the pinned official scorer')
    for group in details['groups']['category_subcategory']:
        labels = group['labels']
        if official['category_subcategory_accuracy'][labels['category']][labels['sub-category']] != group['accuracy']:
            raise ValueError('category/subcategory accuracy disagrees with the pinned official scorer')


def historical_reports(source):
    result = {}
    for path in (source / 'evaluation_report.json', source / rescore.SUBDIRECTORY / 'evaluation_report.json'):
        if path.is_file():
            result[str(path)] = {'sha256': storage.file_sha(path),
                                 'report': json.loads(path.read_text(encoding='utf-8'))}
    return result


def check_history(history, details):
    checked = []
    for name, entry in history.items():
        previous = entry['report']
        official = previous.get('official_evaluation', {})
        item = {'path': name, 'sha256': entry['sha256']}
        if (previous.get('status') == 'PASS' and official.get('status') == 'PASS'
                and official.get('response_policy') == parsing.VERSION):
            accuracy = official.get('overall_accuracy')
            if (official.get('actual_denominator') != details['overall']['total']
                    or not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy)
                    or not math.isclose(accuracy, details['overall']['accuracy'], rel_tol=0, abs_tol=1e-12)
                    or previous.get('response_parsing', {}).get('matched_rows') != details['overall']['correct']):
                raise ValueError(f'historical successful v2 totals disagree with raw-response recomputation: {name}')
            item['status'] = 'AGREES'
        else:
            item.update(status='NOT_COMPARED', reason='not a successful declared answer-text-v2 score')
        checked.append(item)
    return checked


def write_tables(output, report):
    fields = ['view', *TAXONOMY, 'linguistics_sub_discipline', 'task_name', 'leaderboard_domain',
              'correct', 'incorrect', 'total', 'accuracy', 'accuracy_percent']
    csv_path = output / 'detailed_scores.csv'
    temporary = csv_path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerow({'view': 'overall', **report['overall']})
        for view, groups in report['groups'].items():
            for group in groups:
                writer.writerow({'view': view, **group['labels'],
                                 **{key: value for key, value in group.items() if key != 'labels'}})
    temporary.replace(csv_path)
    def cell(value):
        return (value if value else '<empty>').replace('|', '\\|').replace('\r', ' ').replace('\n', '<br>')
    lines = ['# MMSU detailed scores', '', f"Source: {report['source']['input']}", '',
             f"Policy: `{parsing.VERSION}`. Accuracy is sample-weighted within each group.", '',
             f"Overall: {report['overall']['correct']}/{report['overall']['total']} "
             f"({report['overall']['accuracy_percent']:.4f}%).", '',
             'Official taxonomy labels with the existing project grading policy; no new inference.', '']
    for warning in report['warnings']:
        lines.extend([f'Warning: {warning}', ''])
    for view, groups in report['groups'].items():
        keys = VIEWS[view]
        lines.extend([f'## {TITLES[view]}', '',
                      '| ' + ' | '.join([*keys, 'Correct', 'Total', 'Accuracy (%)']) + ' |',
                      '| ' + ' | '.join(['---'] * (len(keys) + 3)) + ' |'])
        for group in groups:
            values = [*(cell(group['labels'][key]) for key in keys), str(group['correct']),
                      str(group['total']), f"{group['accuracy_percent']:.4f}"]
            lines.append('| ' + ' | '.join(values) + ' |')
        lines.append('')
    text_path = output / 'detailed_scores.md'
    temporary = text_path.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(text_path)
    return {path.name: {'path': str(path), 'sha256': storage.file_sha(path)}
            for path in (csv_path, text_path)}


def code_identity():
    paths = [*Path(__file__).parent.glob('*.py'), *Path(__file__).parent.glob('*.sh'),
             *(Path(parsing.__file__).parent / name for name in
               ('dataset.py', 'parsing.py', 'rescore.py', 'scoring.py', 'storage.py'))]
    return {str(path): storage.text_sha(path) for path in sorted(paths)}


def run(source, output, script):
    source, output, script = Path(source).resolve(), Path(output).resolve(), Path(script).resolve()
    started = time.monotonic()
    report = {'status': 'FAILED', 'version': VERSION, 'benchmark': 'mmsu', 'mode': 'offline_detailed_report',
              'gpu_inference': False, 'response_policy': parsing.POLICY, 'fatal_error': None, 'report': None}
    try:
        if output == source or output in source.parents:
            raise ValueError('detailed reports need a separate directory, never the source or its ancestor')
        if not source.is_dir():
            raise FileNotFoundError(f'MMSU source output missing: {source}')
        with storage.output_lock(source):
            records, provenance = rescore.read_source(source)
            history = historical_reports(source)
            details = aggregate(records)
            history_audit = check_history(history, details)
            identity = {'version': VERSION, 'source': provenance, 'response_policy': parsing.POLICY,
                        'views': {name: list(fields) for name, fields in VIEWS.items()},
                        'historical_reports': {name: value['sha256'] for name, value in history.items()},
                        'scorer': scoring.audit_scorer(script), 'code': code_identity()}
            identity['identity_sha256'] = storage.object_sha(identity)
            with storage.output_lock(output):
                storage.claim_output(output, identity)
                try:
                    report.update(source=provenance, historical_score_comparison=history_audit)
                    parsed, official = parsing.score_records(records, script, output)
                    report.update(response_parsing=parsed, official_evaluation=official)
                    check_official(details, official)
                    # Recheck the validated raw and all provenance before publishing.
                    if rescore.read_source(source)[1] != provenance or historical_reports(source) != history:
                        raise ValueError('source MMSU artifacts changed during detailed reporting')
                    if code_identity() != identity['code'] or scoring.audit_scorer(script) != identity['scorer']:
                        raise ValueError('reporting/scoring source changed during detailed reporting')
                    report.update(details)
                    report['artifacts'] = write_tables(output, report)
                    report['status'] = 'PASS'
                except Exception as exc:
                    report['fatal_error'] = {'error': repr(exc), 'traceback': traceback.format_exc()}
                finally:
                    report['elapsed_seconds'] = time.monotonic() - started
                    report['report'] = str(output / 'detailed_report.json')
                    storage.write_json(output / 'detailed_report.json', report)
    except Exception as exc:
        report['fatal_error'] = {'error': repr(exc), 'traceback': traceback.format_exc()}
    report['elapsed_seconds'] = time.monotonic() - started
    return report


def main(argv=None):
    args = parse_args(argv)
    failed = False
    for source, output in planned(args):
        if args.check_args:
            print(json.dumps({'source_output_dir': str(source), 'output_dir': str(output),
                              'evaluation_script': str(args.evaluation_script), 'gpu_inference': False}))
            continue
        report = run(source, output, args.evaluation_script)
        failed |= report['status'] != 'PASS'
        print(json.dumps({'status': report['status'], 'source_output_dir': str(source),
                          'overall': report.get('overall'), 'report': report['report'],
                          'warnings': report.get('warnings', []), 'fatal_error': report['fatal_error']}, ensure_ascii=False))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
