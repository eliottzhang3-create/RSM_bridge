"""Own a new output directory and verify all nineteen complete conditions."""
import json
import os
import subprocess
import sys
import time
import traceback
from mizar_layer_pair_search import manifest, storage
from . import arguments, config, identity, results


def launch(args):
    argv = [sys.executable, '-u', '-B', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
            f'--nproc-per-node={args.world_size}', '--max-restarts=0', '--module',
            'mizar_audio_cutoff.worker', *arguments.forwarded(args)]
    print(json.dumps({'event':'LAUNCH_AUDIO_CUTOFF', 'argv':argv}), flush=True)
    run = subprocess.run(argv, cwd=config.ROOT, env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
    if run.returncode:
        raise RuntimeError(f'torchrun failed ({run.returncode}); inspect ranks/rank*/failure.json')


def verify_artifacts(directory, expected):
    world = expected['world_size']
    checks = storage.verify_seal(storage.read_json(directory/'runtime_checks.json'))
    if (checks['identity_sha256'] != expected['identity_sha256'] or checks['status'] != 'PASS'
            or checks['world_size'] != world or len(checks['rank_audits']) != world or len(checks['rank_checks']) != world):
        raise ValueError('incomplete runtime audits')
    if len({a['binding']['worker_cuda_visible_devices'] for a in checks['rank_audits']}) != world:
        raise ValueError('duplicate GPU binding')
    for rank, (audit, check) in enumerate(zip(checks['rank_audits'], checks['rank_checks'])):
        folder = directory/'ranks'/f'rank{rank:02d}'
        for name, value in [('model_audit.json', audit), ('runtime_check.json', check)]:
            saved = storage.verify_seal(storage.read_json(folder/name))
            if {k:v for k,v in saved.items() if k != 'sha256'} != value:
                raise ValueError('rank audit changed')
        if (audit['binding']['rank'] != rank or check['rank'] != rank or check['status'] != 'PASS'
                or check['identity_sha256'] != expected['identity_sha256'] or check['baseline_native_bitwise_equal'] is not True
                or [c['cutoff_layer_1based'] for c in check['cutoff_isolation']] != [3, 20]
                or any(c['status'] != 'PASS' or c['audio_sep_intervention_max_abs_error'] != 0 for c in check['cutoff_isolation'])):
            raise ValueError('native baseline or cutoff isolation verification failed')
        report = storage.verify_seal(storage.read_json(folder/'rank_report.json'))
        if (report['status'] != 'PASS' or report['identity_sha256'] != expected['identity_sha256']
                or report['binding'] != audit['binding'] or report['conditions_completed'] != 19
                or report['rows_per_condition'] != len(config.indices(rank, world))):
            raise ValueError('rank did not complete all conditions')
    cache = storage.verify_seal(storage.read_json(directory/'cache/cache_manifest.json'))
    if (cache['sha256'] != checks['cache_manifest_sha256'] or cache['rows'] != 100
            or cache['cache_identity']['run_identity_sha256'] != expected['identity_sha256']
            or len(cache['entry_meta_sha256']) != 100):
        raise ValueError('cache identity mismatch')
    for index, digest in enumerate(cache['entry_meta_sha256']):
        meta = storage.verify_seal(storage.read_json(directory/'cache'/f'prefix_{index:03d}.json'))
        if meta['sha256'] != digest or storage.file_sha(directory/'cache'/f'prefix_{index:03d}.pt') != meta['file_sha256']:
            raise ValueError('cached prefix changed')
    return checks


def finalize(directory, frozen, expected):
    records, world, baseline, summaries = frozen['records'], expected['world_size'], None, []
    for cutoff in config.CONDITIONS:
        parts = [results.read_rank(directory, rank, records, cutoff, expected['identity_sha256'], world) for rank in range(world)]
        if any(len(rows) != len(config.indices(rank, world)) for rank, rows in enumerate(parts)):
            return None
        rows = results.merge(parts, records, cutoff, expected['identity_sha256'], world)
        if cutoff is None:
            baseline = rows
        elapsed = 0
        old = directory/'conditions'/config.condition_name(cutoff)/'summary.json'
        if old.exists():
            summary = storage.verify_seal(storage.read_json(old))
            if summary['identity_sha256'] != expected['identity_sha256']:
                raise ValueError('summary belongs to another run')
            elapsed = summary['elapsed_seconds']
        summaries.append(results.save_condition(directory, records, cutoff, rows, baseline, expected['identity_sha256'], elapsed))
    if any(not (directory/'ranks'/f'rank{rank:02d}'/'rank_report.json').exists() for rank in range(world)):
        return None
    checks = verify_artifacts(directory, expected)
    ranked = results.save_analysis(directory, summaries)
    from .plot import make_plot
    plot = make_plot(directory)
    return {'status':'PASS', 'conditions_completed':19, 'decisions':1900, 'rows_per_condition':100,
            'generation_count':0, 'world_size':world, 'baseline_accuracy':summaries[0]['accuracy'],
            'best_cutoff_accuracy':ranked[0]['accuracy'],
            'top_tied_cutoffs':[s['cutoff_layer_1based'] for s in ranked if s['rank'] == 1],
            'runtime_checks_sha256':checks['sha256'], 'plot':plot}


def run(args):
    started = time.monotonic()
    report = {'schema':config.SCHEMA, 'status':'FAILED'}
    try:
        frozen = manifest.load(args.manifest)
        expected = identity.build(args, frozen)
        with storage.output_lock(args.output_dir):
            storage.claim(args.output_dir, expected)
            try:
                report.update(identity_sha256=expected['identity_sha256'], manifest_sha256=frozen['manifest_sha256'])
                complete = finalize(args.output_dir, frozen, expected)
                if complete is None:
                    launch(args)
                    complete = finalize(args.output_dir, frozen, expected)
                if complete is None:
                    raise RuntimeError('workers exited without complete 19x100 coverage')
                identity.verify(args, expected, full=True)
                report.update(complete)
            except Exception as exc:
                report.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
            finally:
                report['elapsed_seconds'] = time.monotonic()-started
                storage.write_json(args.output_dir/'analysis_report.json', report)
    except Exception as exc:
        report.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return report


def main(argv=None):
    args = arguments.parse_args(argv)
    if args.check_args:
        print(json.dumps({**vars(args), 'protocol':config.PROTOCOL}, default=str, indent=2))
        return 0
    return 0 if run(args)['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
