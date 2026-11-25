"""Compare complete runs only after verifying common data, policy and model inputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mizar_s1_ced_reasonaqa5000_eval import config, scoring, storage
from .registry import ROUTES
from .comparison_policy import known_variants, common_protocol, shared_evaluation_code, execution_columns


def read_run(directory):
    def read(name):
        return json.loads((directory / name).read_text(encoding='utf-8'))

    identity, report, accuracy = read('run_config.json'), read('evaluation_report.json'), read('accuracy.json')
    identity_body = {key: value for key, value in identity.items() if key != 'identity_sha256'}
    total = identity['expected_rows']
    if (storage.object_sha(identity_body) != identity['identity_sha256']
            or identity.get('benchmark') != 'reasonaqa_close_ended_sample5000' or identity.get('mode') != 'sample5000'
            or total != config.SAMPLE_SIZE or identity.get('eval_variant') not in known_variants()
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
    original_records = [{key: value for key, value in record.items() if key not in ('audio1', 'audio2')} for record in records]
    if storage.object_sha(original_records) != identity['selection_sha256']:
        raise ValueError(f'raw records disagree with fixed sampling selection: {directory}')
    recomputed, _ = scoring.build_results(saved, total)
    if recomputed != accuracy or recomputed != report.get('accuracy'):
        raise ValueError(f'full-answer score disagrees with raw: {directory}')
    common = {key: identity[key] for key in
              ('manifest_sha256', 'test_json', 'selection_sha256', 'resolved_records_sha256', 'expected_rows', 'protocol')}
    common['protocol']=common_protocol(identity,'reasonaqa5000')
    # Old/new outer runners differ; the reused data, sampling and scoring code must agree.
    common['shared_evaluation_code'] = shared_evaluation_code(identity,'reasonaqa5000')
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
               'checkpoint': identity['checkpoint']['path'], 'output_dir': str(directory), **execution_columns(identity)}
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


def compare_benchmarks(benchmark, directories, output):
    from .registry import ROWS
    from . import mmsu_runtime, runtime
    from mizar_s1_ced_mmsu_eval import parsing
    expected = ROWS[benchmark]
    rows, reference = [], None
    for name, directory in directories:
        identity = json.loads((directory / 'run_config.json').read_text(encoding='utf-8'))
        report = json.loads((directory / 'evaluation_report.json').read_text(encoding='utf-8'))
        if (report.get('status') != 'PASS' or report.get('run_identity') != identity
                or identity.get('benchmark') != benchmark or identity.get('eval_variant') not in known_variants()
                or storage.object_sha({k:v for k,v in identity.items() if k!='identity_sha256'}) != identity['identity_sha256']):
            raise ValueError(f'incomplete/inconsistent report: {directory}')
        coverage = report.get('inference_coverage', {})
        if coverage.get('status') != 'PASS' or coverage.get('rows') != expected or coverage.get('skipped') != 0:
            raise ValueError('incomplete inference coverage report')
        raw_bytes = (directory / 'raw_generations.jsonl').read_bytes()
        if not raw_bytes.endswith(b'\n'):
            raise ValueError('incomplete raw JSONL')
        raw = [json.loads(line) for line in raw_bytes.splitlines()]
        if len(raw) != expected or len({str(r['id']) for r in raw}) != expected:
            raise ValueError('raw coverage differs')
        for index, record in enumerate(raw):
            if record['row_index'] != index:
                raise ValueError('raw rows are not contiguous')
        official = report['official_evaluation']
        if official.get('status') != 'PASS':
            raise ValueError('official evaluation did not pass')
        if benchmark == 'mmsu':
            common = {k:identity[k] for k in ('shards','scorer')}
            common['protocol']=common_protocol(identity,benchmark)
            common['shared_evaluation_code']=shared_evaluation_code(identity,benchmark)
            correct = 0
            inputs = []
            for record in raw:
                original, generated = record['official_record'], record['generation']
                if (record['record_sha256'] != storage.object_sha(original)
                        or record['generation_sha256'] != storage.object_sha(generated)):
                    raise ValueError('corrupt MMSU record hashes')
                correct += parsing.parse_response(generated['generated_text'],original['answer_gt'])['matched']
                inputs.append(record['record_sha256'])
            score = correct / expected
            if official['actual_denominator'] != expected or official['overall_accuracy'] != score:
                raise ValueError('MMSU raw and score disagree')
        else:
            common = {k:identity[k] for k in ('metadata_sha256','parquet_sha256','evaluation_sha256')}
            common['generation']=common_protocol(identity,benchmark)
            common['shared_evaluation_code']=shared_evaluation_code(identity,benchmark)
            if any(r.get('status') != 'generated' for r in raw) or official['reported_rows'] != expected:
                raise ValueError('incomplete generation/scorer denominator')
            filename = 'predictions_fixed_order.json' if benchmark=='mmau' else 'predictions_official.json'
            predictions = json.loads((directory / filename).read_text(encoding='utf-8'))
            key = identity['prediction_key']
            if len(predictions)!=expected:
                raise ValueError('prediction coverage differs')
            by_id = {str(r['id']):r for r in raw}
            for prediction in predictions:
                record = by_id[str(prediction['id'])]
                if prediction[key] != record['generated_text']:
                    raise ValueError('official input does not preserve raw generation')
                if {k:v for k,v in prediction.items() if k!=key} != {k:v for k,v in record['official_record'].items() if k!=key}:
                    raise ValueError('official prediction metadata differs')
            inputs = [storage.object_sha(r['official_record']) for r in raw]
            score = official['total_accuracy_percent'] / 100
        if not 0 <= score <= 1:
            raise ValueError('invalid accuracy')
        if reference is None:
            reference = (common, inputs)
        elif reference != (common, inputs):
            raise ValueError('benchmark data/protocol/code/record order differ across models')
        rows.append({'target':name,'eval_variant':identity['eval_variant'],
                     'checkpoint': identity['checkpoint']['path'] if isinstance(identity['checkpoint'],dict) else identity['checkpoint'],
                     'total':expected,'accuracy':score,'output_dir':str(directory), **execution_columns(identity)})
    if not rows:
        raise ValueError('no runs to compare')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    storage.write_json(output.with_suffix('.json'),{'status':'PASS','benchmark':benchmark,'rows':rows})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=('reasonaqa5000', 'mmau', 'mmar', 'mmsu'), required=True)
    parser.add_argument('--report-dir', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    directories = [(path.name + f':{index}', path.resolve()) for index, path in enumerate(args.report_dir)]
    try:
        if args.benchmark == 'reasonaqa5000':
            rows = compare_runs(directories, args.output.resolve())
        else:
            rows = compare_benchmarks(args.benchmark, directories, args.output.resolve())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'Comparison failed: {exc}\n')
    print(json.dumps({'status': 'PASS', 'runs': len(rows), 'csv': str(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
