"""Verify old identities and select first500 without mutating or rerunning the source."""
import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from mizar_layer_pair_search import storage
from mizar_reasonaqa_tracing import config as old_config, identity as old_identity, selection as old_selection
from . import config


def original_arguments(expected):
    return SimpleNamespace(**{k: Path(v) for k, v in expected['source_arguments'].items()},
        checkpoint=Path(expected['checkpoint']), pool_manifest=Path(expected['pool_manifest']))


def read_journal(path):
    """Strict read-only: even a partial tail is reported, never truncated here."""
    rows = []
    with path.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b'\n'):
                raise ValueError(f'source journal has an incomplete tail at line{number}; source left unchanged')
            try:
                rows.append(storage.verify_seal(json.loads(line)))
            except (ValueError, UnicodeError, AttributeError) as exc:
                raise ValueError(f'invalid source journal line{number}') from exc
    return rows


def validate_history(history, pool, expected):
    accepted = 0
    for pos, value in enumerate(history):
        if pos >= len(pool['shuffle_order']):
            raise ValueError('source journal exceeds fixed5000')
        r = pool['records'][pool['shuffle_order'][pos]]
        old_selection.validate_outcome(value, r, expected)
        if (value['shuffle_position'] != pos or value['acceptance_order'] != (accepted if value['accepted'] else None)
                or accepted >= old_config.TARGET):
            raise ValueError('source screening order/acceptance sequence changed')
        accepted += int(value['accepted'])
    return accepted


def verify_cache(directory, record, outcome, source_expected, full=True):
    folder = directory/'cache'/f'source_{record["source_manifest_index"]:04d}'
    meta = storage.verify_seal(storage.read_json(folder/'entry.json'))
    if (meta['sha256'] != outcome['cache_meta_sha256'] or meta['identity_sha256'] != source_expected['identity_sha256']
            or meta['record_sha256'] != storage.object_sha(record)
            or meta['source_manifest_index'] != record['source_manifest_index']
            or meta['audio_audit'] != outcome['audio_audit']
            or meta['tensor_shapes'] != {'clean_prefix': [1,383,576], 'corrupted_prefix': [1,383,576], 'clean_states': [30,256,576]}
            or meta['tensor_dtypes']['clean_prefix'] != 'torch.bfloat16'
            or meta['tensor_dtypes']['corrupted_prefix'] != 'torch.bfloat16'
            or meta['tensor_dtypes']['clean_states'] not in ('torch.bfloat16','torch.float32')):
        raise ValueError(f'original full activation cache metadata changed: {record["key"]}')
    if not (folder/'tensors.pt').is_file():
        raise FileNotFoundError(f'original cache missing: {folder}/tensors.pt')
    if full and storage.file_sha(folder/'tensors.pt') != meta['file_sha256']:
        raise ValueError(f'original cache content changed: {record["key"]}')
    return meta


def load(directory, *, full=False, verify_caches=False):
    expected = storage.verify_seal(storage.read_json(directory/'run_config.json'), 'identity_sha256')
    # No bypass of source guards: the original1000 package is kept unchanged.
    pool = old_identity.verify(original_arguments(expected), expected, full=full)
    history = read_journal(directory/'screening/journal.jsonl')
    accepted = validate_history(history, pool, expected)
    report = storage.verify_seal(storage.read_json(directory/'selection_report.json'))
    computed = {'identity_sha256': expected['identity_sha256'], 'pool_size': len(pool['records']),
        'scanned': len(history), 'accepted': accepted, 'required': old_config.TARGET,
        'reasons': dict(Counter(r['reason'] for r in history)),
        'clean_correct': sum(r['clean']['correct'] for r in history),
        'corrupted_forwards': sum(r['corrupted'] is not None for r in history),
        'journal_sha256': storage.object_sha(history), 'shuffle_sha256': pool['shuffle_sha256']}
    if any(report.get(k) != v for k, v in computed.items()):
        raise ValueError('source selection report differs from journal')
    if report['status'] not in ('INSUFFICIENT_QUALIFYING', 'SELECTED'):
        raise ValueError('source screening has not completed with a sealed selection report')
    if report['status'] == 'INSUFFICIENT_QUALIFYING' and len(history) != len(pool['records']):
        raise ValueError('source says insufficient before exhausting fixed5000')
    if (report['status'] == 'SELECTED') != (accepted == old_config.TARGET):
        raise ValueError('source selection status contradicts its qualifying count')
    if accepted < config.TARGET:
        raise ValueError(f'only {accepted} source rows qualify, need {config.TARGET}; source not modified')
    selected = [r for r in history if r['accepted']][:config.TARGET]
    rows = []
    for outcome in sorted(selected, key=lambda v: pool['records'][v['source_manifest_index']]['source_row_index']):
        r = copy.deepcopy(pool['records'][outcome['source_manifest_index']])
        if verify_caches:
            verify_cache(directory, r, outcome, expected, full=full)
        r.update(selected_index=len(rows), shuffle_position=outcome['shuffle_position'], acceptance_order=outcome['acceptance_order'],
            screening={k: outcome[k] for k in ('clean','corrupted','probability_gap','cache_meta_sha256','audio_audit')})
        rows.append(r)
    snapshots = {name: storage.file_sha(directory/name) for name in
                 ('run_config.json','screening/journal.jsonl','selection_report.json')}
    return {'expected': expected, 'pool': pool, 'history': history, 'records': rows, 'selection_report': report,
            'snapshots': snapshots, 'stop_shuffle_position': selected[-1]['shuffle_position']}


def manifest(context, source_directory):
    return storage.seal({'schema': config.SCHEMA, 'protocol': config.PROTOCOL,
        'source_output_dir': str(source_directory), 'source_identity_sha256': context['expected']['identity_sha256'],
        'source_pool_sha256': context['pool']['manifest_sha256'], 'source_snapshots': context['snapshots'],
        'source_scanned': len(context['history']), 'source_total_qualifying': context['selection_report']['accepted'],
        'selection': 'first500_accepted_in_original_shuffle_then_source_row_ascending',
        'stop_shuffle_position': context['stop_shuffle_position'],
        'category_counts': dict(Counter(r['category'] for r in context['records'])),
        'records': context['records']}, 'manifest_sha256')
