"""CPU coordinator owns output; torchrun workers synchronize each candidate."""
import json
import os
import subprocess
import sys
import time
import traceback

from . import arguments, config, execution, identity, manifest, results, storage


def launch(args, world_size=8):
    execution.validate_world(world_size)
    module = 'mizar_layer_pair_search.worker_single_gpu' if world_size == 1 else 'mizar_layer_pair_search.worker'
    command = [sys.executable, '-u', '-B', '-m', 'torch.distributed.run', '--standalone',
               '--nnodes=1', f'--nproc-per-node={world_size}', '--max-restarts=0', '--module',
               module, *arguments.forwarded(args)]
    print(json.dumps({'event': 'LAUNCH_MASK_SEARCH', 'world_size': world_size, 'argv': command}), flush=True)
    completed = subprocess.run(command, cwd=config.ROOT,
        env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
    if completed.returncode:
        raise RuntimeError(f'torchrun failed with exit {completed.returncode}; see per-rank failure reports')


def finalize(directory, frozen, expected, world_size=8):
    """Return None for incomplete raw coverage; corrupt complete records fail."""
    execution.validate_world(world_size)
    records, summaries = frozen['records'], []
    for index in range(len(config.PAIRS)+1):
        parts = [results.read_rank(directory, r, records, index, expected['identity_sha256'], world_size) for r in range(world_size)]
        if any(len(rows) != len(execution.rank_indices(rank, world_size)) for rank, rows in enumerate(parts)):
            return None
        rows = results.merge_rank_rows(parts, records, index, expected['identity_sha256'], world_size)
        elapsed = 0
        old_path = directory / 'scores' / (config.mask_name(index)+'.json')
        if old_path.exists():
            old = storage.verify_seal(storage.read_json(old_path))
            if old['identity_sha256'] != expected['identity_sha256']:
                raise ValueError('summary belongs to another run')
            elapsed = old['elapsed_seconds']
        summaries.append(results.save_mask(directory, records, index, rows, expected['identity_sha256'], elapsed))
    if any(not (directory / 'ranks' / f'rank{r:02d}' / 'rank_report.json').exists() for r in range(world_size)):
        return None
    checks = verify_runtime_artifacts(directory, expected, world_size)
    ordered = results.save_ranking(directory, summaries[:378])
    storage.write_json(directory / 'baseline.json', summaries[378])
    storage.write_jsonl(directory / 'mask_scores.jsonl', summaries[:378])
    return {'status': 'PASS', 'world_size': world_size, 'candidate_masks_completed': 378, 'rows_per_mask': 100,
            'candidate_decisions': 37800, 'baseline_decisions': 100, 'first_mask_free_generations': 100,
            'baseline_accuracy': summaries[378]['accuracy'], 'top_accuracy': ordered[0]['accuracy'],
            'top_tied_masks': [{'mask_index': r['mask_index'], 'drop_pair_0based': r['drop_pair_0based']}
                               for r in ordered if r['rank'] == 1],
            'ranking': str(directory / 'ranking.csv'), 'runtime_checks_sha256': checks['sha256']}


def verify_runtime_artifacts(directory, expected, world_size=8):
    execution.validate_world(world_size)
    checks = storage.verify_seal(storage.read_json(directory / 'runtime_checks.json'))
    if (checks.get('identity_sha256') != expected['identity_sha256'] or checks.get('status') != 'PASS'
            or checks['physical_pruning_equivalence']['status'] != 'PASS'
            or checks['online_cache_check']['status'] != 'PASS' or len(checks['rank_audits']) != world_size):
        raise ValueError('missing real BF16/runtime/mask/cache verification')
    bindings = checks['rank_audits']
    if (len({a['binding']['worker_cuda_visible_devices'] for a in bindings}) != world_size
            or [a['binding']['rank'] for a in bindings] != list(range(world_size))):
        raise ValueError('runtime audit does not cover the requested GPU workers')
    for rank, audit in enumerate(bindings):
        saved = storage.verify_seal(storage.read_json(directory / 'ranks' / f'rank{rank:02d}' / 'model_audit.json'))
        if {k: v for k, v in saved.items() if k != 'sha256'} != audit:
            raise ValueError('per-rank loader audit differs from runtime checks')
        report = storage.verify_seal(storage.read_json(directory / 'ranks' / f'rank{rank:02d}' / 'rank_report.json'))
        if (report.get('status') != 'PASS' or report.get('identity_sha256') != expected['identity_sha256']
                or report.get('binding') != audit['binding'] or report.get('masks_completed') != 379
                or report.get('rows_per_mask') != len(execution.rank_indices(rank, world_size))):
            raise ValueError('worker completion report mismatch')
    cache = storage.verify_seal(storage.read_json(directory / 'cache/cache_manifest.json'))
    if (cache['sha256'] != checks['cache_manifest_sha256'] or cache['rows'] != 100
            or cache['cache_identity']['run_identity_sha256'] != expected['identity_sha256']
            or len(cache['entry_meta_sha256']) != 100):
        raise ValueError('cache manifest changed after runtime verification')
    for index in range(100):
        meta = storage.verify_seal(storage.read_json(directory / 'cache' / f'prefix_{index:03d}.json'))
        if (meta['sha256'] != cache['entry_meta_sha256'][index]
                or storage.file_sha(directory / 'cache' / f'prefix_{index:03d}.pt') != meta['file_sha256']):
            raise ValueError('prefix cache files changed')
    return checks


def run(args, *, prepare_manifest=True, world_size=8):
    started = time.monotonic()
    report, claimed = {'status': 'FAILED', 'schema': config.SCHEMA}, False
    try:
        frozen = manifest.ensure(args) if prepare_manifest else manifest.load(args.manifest)
        expected = identity.build(args, frozen, world_size)
        with storage.output_lock(args.output_dir):
            storage.claim(args.output_dir, expected)
            claimed = True
            try:
                report.update(identity_sha256=expected['identity_sha256'], manifest_sha256=frozen['manifest_sha256'])
                complete = finalize(args.output_dir, frozen, expected, world_size)
                report['workers_launched'] = complete is None
                if complete is None:
                    launch(args, world_size)
                    complete = finalize(args.output_dir, frozen, expected, world_size)
                if complete is None:
                    raise RuntimeError('workers exited without complete 378 masks/baseline/first100 generations')
                identity.verify(args, expected, full_manifest=True, world_size=world_size)
                report.update(complete)
            except Exception as exc:
                report.update(status='FAILED', fatal_error={'error': repr(exc), 'traceback': traceback.format_exc()})
            finally:
                report['elapsed_seconds'] = time.monotonic()-started
                storage.write_json(args.output_dir / 'search_report.json', report)
    except Exception as exc:
        report.update(status='FAILED', fatal_error={'error': repr(exc), 'traceback': traceback.format_exc()})
        report['output_claimed'] = claimed
    print(json.dumps(report, ensure_ascii=False, default=str, allow_nan=False), flush=True)
    return report


def main(argv=None, *, prepare_manifest=False, world_size=8):
    args = execution.parse_args(argv, world_size)
    if args.check_args:
        print(json.dumps({**vars(args), 'protocol': execution.protocol(world_size)}, default=str, ensure_ascii=False, indent=2))
        return 0
    return 0 if run(args, prepare_manifest=prepare_manifest, world_size=world_size)['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
