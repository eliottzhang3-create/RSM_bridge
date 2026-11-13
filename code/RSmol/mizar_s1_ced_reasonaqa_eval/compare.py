"""Compare complete runs only after verifying common data, policy and model inputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import config, scoring, storage


def read_run(directory):
    def read(name):
        return json.loads((directory / name).read_text(encoding='utf-8'))

    identity, report, accuracy = read('run_config.json'), read('evaluation_report.json'), read('accuracy.json')
    identity_body = {key: value for key, value in identity.items() if key != 'identity_sha256'}
    total = identity['expected_rows']
    if (storage.object_sha(identity_body) != identity['identity_sha256']
            or identity.get('benchmark') != 'reasonaqa_close_ended' or identity.get('mode') != 'full'
            or report.get('status') != 'PASS' or report.get('run_identity') != identity
            or report.get('inference_coverage') !=
                {'status': 'PASS', 'generated': total, 'expected_rows': total, 'skipped': 0}):
        raise ValueError(f'incomplete or inconsistent ReasonAQA report: {directory}')
    # Do not recover/truncate or rewrite anything during comparison.
    raw = (directory / 'raw_generations.jsonl').read_bytes()
    if not raw.endswith(b'\n'):
        raise ValueError(f'incomplete raw JSONL: {directory}')
    saved = [json.loads(line) for line in raw.splitlines()]
    if len(saved) != total:
        raise ValueError(f'raw coverage mismatch: {directory}')
    records, fingerprints = [], []
    for index, item in enumerate(saved):
        record, generation = item['record'], item['generation']
        if (item['row_index'] != index or item['record_sha256'] != storage.object_sha(record)
                or item['generation_sha256'] != storage.object_sha(generation)):
            raise ValueError(f'raw identity/hash mismatch at row {index}: {directory}')
        storage.validate_generation(generation, record)
        records.append(record)
        fingerprints.append((record['id'], generation['input_fingerprint_sha256']))
    if storage.object_sha(records) != identity['resolved_records_sha256']:
        raise ValueError(f'resolved records mismatch: {directory}')
    recomputed, _ = scoring.build_results(saved, total)
    if recomputed != accuracy or recomputed != report.get('accuracy'):
        raise ValueError(f'full-answer score disagrees with raw: {directory}')
    common = {key: identity[key] for key in
              ('test_json', 'selection_sha256', 'resolved_records_sha256', 'expected_rows', 'protocol', 'evaluation_code')}
    return identity, accuracy, common, fingerprints


def compare_runs(directories, output):
    rows, common, fingerprints = [], None, None
    for name, directory in directories:
        identity, accuracy, current, inputs = read_run(directory)
        if common is None:
            common, fingerprints = current, inputs
        elif common != current or fingerprints != inputs:
            raise ValueError(f'data/protocol/code or per-row model inputs differ: {name}')
        row = {'target': name, 'eval_variant': identity['eval_variant'],
               'checkpoint': identity['checkpoint']['path'], 'output_dir': str(directory)}
        for label, group in [('overall', accuracy['overall']), *accuracy['by_category'].items()]:
            for field in ('correct', 'total', 'accuracy'):
                row[f'{label}_{field}'] = group[field]
        rows.append(row)
    if not rows:
        raise ValueError('no runs to compare')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    storage.write_json(output.with_suffix('.json'),
                       {'status': 'PASS', 'metric': 'normalized_full_answer_exact_match',
                        'common': common, 'input_fingerprints_sha256': storage.object_sha(fingerprints), 'rows': rows})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, action='append', help='repeat for explicit runs; defaults to the selected seven')
    parser.add_argument('--output', type=Path, default=config.REMOTE / 'outputs/RSmol/reasonaqa_selected7_step17313_v1.csv')
    args = parser.parse_args(argv)
    directories = ([(path.name + f':{index}', path.resolve()) for index, path in enumerate(args.report_dir)]
                   if args.report_dir else [(name, spec['run_root'] / 'eval' / config.OUTPUT_NAME.format(step=17313))
                                            for name, spec in config.TARGETS.items()])
    try:
        rows = compare_runs(directories, args.output.resolve())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'ReasonAQA comparison failed: {exc}\n')
    print(json.dumps({'status': 'PASS', 'runs': len(rows), 'csv': str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
