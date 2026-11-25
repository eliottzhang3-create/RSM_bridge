"""Coordinator owns the output lock and launches exactly one torchrun group."""
import json
import os
import subprocess
import sys
import time
import traceback
from mizar_layer_pair_search import storage
from . import arguments, audit, config, identity, prepare, results, selection


def launch(args):
    command = [sys.executable, '-u', '-B', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc-per-node=8', '--max-restarts=0', '--module', 'mizar_reasonaqa_tracing.worker', *arguments.forwarded(args)]
    print(json.dumps({'event': 'LAUNCH_TRACING', 'stage': args.stage, 'argv': command}), flush=True)
    completed = subprocess.run(command, cwd=config.ROOT, env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
    if completed.returncode:
        raise RuntimeError(f'torchrun failed ({completed.returncode}); see selection_report.json and ranks/rankXX/failure.json')


def finalize(args, pool, expected):
    directory = args.output_dir
    if not (directory/'selected/manifest.json').is_file():
        return None
    frozen = selection.load_selected(directory, pool, expected)
    if args.stage == 'select':
        return {'status': 'SELECTED', 'selected': config.TARGET, 'manifest': str(directory/'selected/manifest.json'),
                'manifest_sha256': frozen['manifest_sha256'], 'category_counts': frozen['category_counts']}
    if not all((directory/'ranks'/f'rank{rank:02d}'/'rank_report.json').is_file() for rank in range(8)):
        return None
    summaries = []
    for layer in range(1, 31):
        parts = [results.read_rank(directory, rank, frozen, expected, layer) for rank in range(8)]
        if any(len(p) != len(config.indices(rank)) for rank, p in enumerate(parts)):
            return None
        rows = results.merge(parts, frozen, expected, layer)
        old = directory/'layers'/f'patch_{layer:02d}'/'summary.json'
        elapsed = 0
        if old.exists():
            saved = storage.verify_seal(storage.read_json(old))
            if saved['identity_sha256'] != expected['identity_sha256']:
                raise ValueError('layer summary belongs to another run')
            elapsed = saved['elapsed_seconds']
        summaries.append(results.save_layer(directory, rows, frozen, expected, layer, elapsed))
    checks = audit.verify(directory, pool, frozen, expected)
    results.write_tables(directory, summaries)
    from .plot import make_plot
    plots = make_plot(directory)
    return {'status': 'PASS', 'selected': config.TARGET, 'patched_layers': 30, 'patched_decisions': config.TARGET*30,
        'manifest': str(directory/'selected/manifest.json'), 'manifest_sha256': frozen['manifest_sha256'],
        'category_counts': frozen['category_counts'], 'metric': 'mean_per_question_unclipped_probability_space_RR',
        'runtime_checks_sha256': checks['sha256'], 'plots': plots,
        'conditional_clean_accuracy': 1.0, 'conditional_corrupted_accuracy': 0.0,
        'note': 'selected conditional subset, not full5000 evaluation; layer30 is a readout restoration control'}


def run(args):
    report = {'schema': config.SCHEMA, 'status': 'FAILED', 'stage': args.stage}; started = time.monotonic()
    claimed = False
    try:
        pool = prepare.ensure(args)
        expected = identity.build(args, pool)
        with storage.output_lock(args.output_dir):
            storage.claim(args.output_dir, expected); claimed = True
            try:
                report.update(identity_sha256=expected['identity_sha256'], pool_sha256=pool['manifest_sha256'])
                storage.write_json(args.output_dir/'shuffle.json', storage.seal({'seed': config.SEED,
                    'pool_sha256': pool['manifest_sha256'], 'order': pool['shuffle_order'], 'shuffle_sha256': pool['shuffle_sha256']}))
                complete = finalize(args, pool, expected)
                if complete is None:
                    launch(args); complete = finalize(args, pool, expected)
                if complete is None:
                    raise RuntimeError('incomplete selected manifest or patch/rank coverage')
                identity.verify(args, expected, full=True)
                report.update(complete)
            except Exception as exc:
                report.update(error=repr(exc), traceback=traceback.format_exc())
            finally:
                report['elapsed_seconds'] = time.monotonic()-started
                storage.write_json(args.output_dir/('selection_only_report.json' if args.stage == 'select' else 'analysis_report.json'), report)
    except Exception as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc(), output_claimed=claimed)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return report


def main(argv=None):
    args = arguments.parse_args(argv)
    if args.check_args:
        print(json.dumps({'args': vars(args), 'protocol': config.PROTOCOL}, default=str, indent=2)); return 0
    return 0 if run(args)['status'] in ('PASS', 'SELECTED') else 1


if __name__ == '__main__':
    raise SystemExit(main())
