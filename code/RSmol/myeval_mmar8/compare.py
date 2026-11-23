"""CPU-only validation, rescoring and side-by-side open-answer export."""
import argparse
import csv
import json
from pathlib import Path

from . import runtime, scoring, storage
from .arguments import parse_args as runner_args
from .registry import ROOT, ROUTES, intervention


def read_result(directory):
    directory = Path(directory)
    identity = json.loads((directory / 'run_config.json').read_text(encoding='utf-8'))
    target = identity['target']
    if target not in ROUTES or identity.get('intervention') != intervention(target):
        raise ValueError('wrong target/intervention identity')
    if identity.get('evaluation_sources') != runtime.evaluation_sources():
        raise ValueError('evaluation source identity differs; use the original evaluation code')
    records = json.loads((directory / 'task_manifest.json').read_text(encoding='utf-8'))
    if storage.object_sha(records) != identity['task_manifest_sha256']:
        raise ValueError('saved task manifest is damaged')
    report = json.loads((directory / 'evaluation_report.json').read_text(encoding='utf-8'))
    if (report.get('status') != 'PASS' or report.get('run_identity') != identity
            or report.get('raw_sha256') != storage.file_sha(directory / 'raw_generations.jsonl')):
        raise ValueError('incomplete or externally modified run/raw records')
    store = storage.GenerationStore(directory, records, read_only=True, expected_intervention=identity['intervention'])
    matcher, scorer = scoring.load_scorer(directory / 'official_scorer.py')
    if scorer != identity['scorer']:
        raise ValueError('saved official scorer identity differs')
    result = scoring.summarize(records, store.state, matcher)
    saved = json.loads((directory / 'scores.json').read_text(encoding='utf-8'))
    if saved != {k: v for k, v in result.items() if k not in ('permutation_scores', 'open_answers')}:
        raise ValueError('published scores disagree with complete raw generations')
    return identity, records, store.state, result


def compare(directories, output):
    loaded = [read_result(path) for path in directories]
    by_target = {identity['target']: (identity, records, state, result)
                 for identity, records, state, result in loaded}
    if len(loaded) != 3 or set(by_target) != set(ROUTES):
        raise ValueError('need exactly base, read2, read2_zero48 completed outputs')
    base_identity, base_records, base_state, _ = by_target['base']
    common = ('annotations_sha256', 'metadata_sha256', 'audio_files', 'task_manifest_sha256',
              'scorer', 'thresholds', 'generation', 'evaluation_sources')
    for identity, records, state, _ in loaded:
        if any(identity[k] != base_identity[k] for k in common) or records != base_records:
            raise ValueError('the models did not evaluate the same tasks/data/protocol')
        for a, b in zip(base_state, state):
            if a['generation']['input_fingerprint_sha256'] != b['generation']['input_fingerprint_sha256']:
                raise ValueError('actual prompt tokens/masks/audio differ between models')
    if by_target['read2'][0]['checkpoint_sha256'] != by_target['read2_zero48'][0]['checkpoint_sha256']:
        raise ValueError('read2 and zero48 must use exactly the same checkpoint')
    output = Path(output)
    for directory in directories:
        source = Path(directory).resolve()
        if output.resolve() == source or output.resolve().is_relative_to(source):
            raise ValueError('comparison output must not be inside a source result')
    with storage.output_lock(output):
        storage.claim_output(output, {'kind': 'myeval8_comparison_v1',
            'sources': {k: storage.object_sha(v[0]) for k, v in by_target.items()}})
        summary = {k: {'dimensions': v[3]['dimensions'], 'total': v[3]['total']} for k, v in by_target.items()}
        storage.write_json(output / 'comparison.json', summary)
        with (output / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['model', 'dimension', 'correct', 'total', 'accuracy_percent'])
            for target, result in summary.items():
                for dim, metric in [*result['dimensions'].items(), ('total', result['total'])]:
                    writer.writerow([target, dim, metric['correct'], metric['total'], metric['accuracy_percent']])
        lines = ['# MyEval8 open-ended outputs', '', 'Open-ended answers are not scored.', '']
        for index, item in enumerate(by_target['base'][3]['open_answers']):
            lines += [f'## Audio {item["item_no"]}, question {item["question_number"]} ({item["dim"]})',
                      '', item['question'], '']
            for target in ROUTES:
                answer = by_target[target][3]['open_answers'][index]['generation']['generated_text']
                lines += [f'### {target}', '', *['> ' + line for line in answer.splitlines()], '']
        (output / 'open_answers_comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        combined = [{'item_id': item['item_id'], 'question_number': item['question_number'],
                     'dim': item['dim'], 'question': item['question'],
                     'answers': {target: by_target[target][3]['open_answers'][i]['generation']
                                 for target in ROUTES}}
                    for i, item in enumerate(by_target['base'][3]['open_answers'])]
        storage.write_jsonl(output / 'open_answers_comparison.jsonl', combined)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', action='append', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT.parents[1] / 'outputs/RSmol/myeval_mmar8_comparison_v1')
    args = parser.parse_args(argv)
    directories = args.result_dir or [runner_args('evaluate', ['--eval-variant', target]).output_dir for target in ROUTES]
    summary = compare(directories, args.output_dir)
    print(json.dumps({'status': 'PASS', 'output_dir': str(args.output_dir), 'models': summary}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
