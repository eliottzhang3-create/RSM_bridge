"""748 independent generations per target; durable resume, no open-answer grading."""
import json
import time
import traceback
from pathlib import Path

from . import arguments, benchmark, dataset, runtime, scoring, storage
from .registry import REQUIRED_CORRECT, intervention


def parse_args(argv=None):
    return arguments.parse_args('evaluate', argv)


def prepare(args):
    rows = dataset.load_annotations(args.annotations)
    official = benchmark.mmar.load_mmar_records(args.metadata_json)
    # Same official metadata/scorer/audio audit as the full baseline runner.
    artifact = benchmark.mmar.audit_mmar_artifacts(args.metadata_json, args.audio_root, args.evaluation_script, official)
    if artifact['status'] != 'PASS':
        raise ValueError('original MMAR metadata/audio/scorer audit failed: ' + str(artifact))
    records = dataset.expand(rows, official, args.audio_root)
    missing = [r['audio_path_resolved'] for r in records if not Path(r['audio_path_resolved']).is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    matcher, scorer = scoring.load_scorer(args.evaluation_script)
    crop = []
    for row in rows:
        def seconds(text):
            return sum(int(x) * m for x, m in zip(text.split(':'), (3600, 60, 1)))
        a, b = row['audio']['timestamp'].split(',')
        duration = seconds(b) - seconds(a)
        crop.append({'item_id': row['item_id'], 'annotated_duration_seconds': duration,
                     'annotated_center_crop_start_seconds': max(0, (duration - 20) / 2),
                     'basis': 'annotation timestamps; actual input also fingerprinted after decoding'})
    return records, artifact, matcher, scorer, crop


def identity(args, records, scorer):
    signatures = []
    for path in sorted({Path(r['audio_path_resolved']) for r in records}):
        stat = path.stat()
        signatures.append({'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    return {
        'protocol': 'myeval_mmar8_full_permutations_v1', 'target': args.eval_variant,
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': storage.file_sha(args.checkpoint),
        'runtime_config': str(args.runtime_config), 'runtime_sha256': storage.file_sha(args.runtime_config),
        'source_revision': runtime.variant_spec(args.eval_variant)['revision'],
        'annotations_sha256': dataset.normalized_sha(args.annotations),
        'metadata_sha256': storage.file_sha(args.metadata_json), 'audio_files': signatures,
        'task_manifest_sha256': storage.object_sha(records), 'scorer': scorer,
        'thresholds': {str(k): v for k, v in REQUIRED_CORRECT.items()},
        'intervention': intervention(args.eval_variant), 'generation': runtime.PROTOCOL,
        'evaluation_sources': runtime.evaluation_sources(),
    }


def run(args):
    with storage.output_lock(args.output_dir):
        return _run(args)


def _run(args):
    started = time.monotonic()
    report = {'status': 'FAILED', 'task': 'myeval8', 'target': args.eval_variant,
              'expected_generations': 748, 'generated': 0}
    claimed, store, live = False, None, None
    try:
        runtime.sources.prepare(args.eval_variant, args.sources_cache)
        records, artifact, matcher, scorer, crop = prepare(args)
        binding = identity(args, records, scorer)
        storage.claim_output(args.output_dir, binding)
        claimed = True
        manifest_path = args.output_dir / 'task_manifest.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding='utf-8')) != records:
                raise ValueError('saved task manifest changed')
        else:
            storage.write_json(manifest_path, records)
        storage.write_json(args.output_dir / 'dataset_audit.json', {
            'official_audit': artifact, 'crop_audit': crop, 'thresholds': binding['thresholds'],
            'closed_questions': 53, 'closed_permutations': 734, 'open_questions': 14,
            'original_questions_used_as_model_input': False, 'old_model_answers_used': False})
        saved_scorer = args.output_dir / 'official_scorer.py'
        scorer_text = args.evaluation_script.read_bytes().replace(b'\r\n', b'\n')
        if saved_scorer.exists() and saved_scorer.read_bytes() != scorer_text:
            raise ValueError('saved official scorer changed')
        if not saved_scorer.exists():
            saved_scorer.write_bytes(scorer_text)
        store = storage.GenerationStore(args.output_dir, records, expected_intervention=binding['intervention'])
        report['resumed_generations'] = len(store.state)
        if len(store.state) < len(records):
            live = runtime.load_runtime(args)
            report['model_audit'] = live[5]
            storage.write_json(args.output_dir / 'model_audit.json', live[5])
            for index in range(len(store.state), len(records)):
                record = records[index]
                generation = benchmark.generate(live, record)
                store.append(index, record, generation)
                report['generated'] = len(store.state)
                if (index + 1) % 10 == 0 or index + 1 == len(records):
                    print(json.dumps({'event': 'MYEVAL_PROGRESS', 'target': args.eval_variant,
                        'completed': index + 1, 'total': len(records), 'id': record['id']}), flush=True)
            storage.write_json(args.output_dir / 'model_audit.json', live[5])
        else:
            report['model_audit'] = json.loads((args.output_dir / 'model_audit.json').read_text(encoding='utf-8'))
        if identity(args, records, scorer) != binding:
            raise ValueError('checkpoint/data/audio/code identity changed during inference')
        result = scoring.summarize(records, store.state, matcher)
        scoring.publish(args.output_dir, result)
        report.update(status='PASS', generated=len(store.state), run_identity=binding,
                      total=result['total'], dimensions=result['dimensions'],
                      open_ended_scored=False, raw_sha256=storage.file_sha(store.path))
    except Exception as exc:
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        if store is not None:
            report['generated'] = len(store.state)
        print(json.dumps({'event': 'MYEVAL_FAILURE', **report}, ensure_ascii=False), flush=True)
    finally:
        vars(args).pop('_checkpoint_payload', None)
        report['elapsed_seconds'] = time.monotonic() - started
        if claimed:
            if live is not None:
                storage.write_json(args.output_dir / 'model_audit.json', live[5])
            storage.write_json(args.output_dir / 'evaluation_report.json', report)
    return report


def main(argv=None):
    args = parse_args(argv)
    if args.check_args:
        print(json.dumps(arguments.printable(args), default=str)); return 0
    report = run(args)
    print(json.dumps({k: report.get(k) for k in ('target', 'status', 'generated', 'total', 'error')}, ensure_ascii=False))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
