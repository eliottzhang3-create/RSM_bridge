"""Read existing MMSU artifacts without recovery, truncation or model loading."""
from __future__ import annotations

import contextlib
import io
import json
import os
import runpy
from pathlib import Path

from mizar_s1_ced_mmsu_eval import dataset, evaluate, parsing, rescore, runtime, scoring, storage


class OutputBusy(RuntimeError):
    pass


@contextlib.contextmanager
def existing_lock(directory):
    """Cooperate with the original generator without creating its lock file."""
    path = directory / '.mmsu.lock'
    if not path.exists():
        yield
        return
    with path.open('r+b') as stream:
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OutputBusy(f'MMSU generation or scoring is running: {directory}') from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


class FileCache:
    def __init__(self):
        self.values = {}

    def identity(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.values:
            self.values[key] = evaluate.file_identity(path)
        return self.values[key]


def check_identity(args, identity, cache):
    body = {k: v for k, v in identity.items() if k != 'identity_sha256'}
    if storage.object_sha(body) != identity.get('identity_sha256'):
        raise ValueError('saved MMSU run identity digest differs')
    _, family, _ = runtime.adapter(args.eval_variant)
    for key, expected in dict(benchmark='mmsu', mode='full', expected_rows=5000,
            eval_variant=args.eval_variant, runtime_family=family, expected_step=args.expected_step,
            expected_geometry=list(args.expected_geometry), route_root=str(args.route_root),
            dataset_dir=str(args.dataset_dir)).items():
        if identity.get(key) != expected:
            raise ValueError(f'saved MMSU identity differs: {key}')
    for key in ('checkpoint', 'runtime_config'):
        if identity.get(key) != cache.identity(getattr(args, key)):
            raise ValueError(f'saved MMSU {key} content/path differs')
    paths = dataset.shard_paths(args.dataset_dir)
    if identity.get('shards') != [cache.identity(path) for path in paths]:
        raise ValueError('saved MMSU dataset shards differ')
    model_sources = {p.relative_to(args.route_root).as_posix(): storage.text_sha(p)
                     for p in sorted(args.route_root.rglob('*.py'))}
    if not model_sources or identity.get('model_sources') != model_sources:
        raise ValueError('saved training model source identity differs')
    if identity.get('scorer') != scoring.audit_scorer(args.evaluation_script):
        raise ValueError('saved official scorer identity differs')
    # CPU scoring can change only the score version, never the prompt policy.
    # Historical official-template raw must not count as completed compact input.
    shared = lambda p: {k: v for k, v in p.items() if k not in ('version', 'scoring')}
    if shared(identity.get('protocol', {})) != shared(runtime.PROTOCOL):
        raise ValueError('saved MMSU generation protocol differs')


def raw_prefix(path):
    """Validate durable rows; report an unfinished tail without modifying it."""
    rows, seen, tail = [], set(), False
    if not path.exists():
        return rows, tail
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                tail = True
                break
            row = json.loads(line)
            record, generation = row.get('official_record'), row.get('generation')
            if not isinstance(record, dict) or not isinstance(generation, dict):
                raise ValueError('invalid raw MMSU generation object')
            dataset.validate_record(record)
            key = dataset.row_id(record)
            if (row.get('row_index') != len(rows) or row.get('id') != record['id'] or key in seen
                    or row.get('record_sha256') != storage.object_sha(record)
                    or row.get('generation_sha256') != storage.object_sha(generation)
                    or not isinstance(generation.get('generated_text'), str)):
                raise ValueError(f'corrupt or out-of-order raw MMSU row {len(rows)}')
            seen.add(key)
            rows.append(row)
            if len(rows) > 5000:
                raise ValueError('too many raw MMSU rows')
    return rows, tail


def verified_score(directory, records, identity, provenance, script):
    """Recheck v2 totals using the same pinned official function, without writes."""
    report = read_json(directory / 'evaluation_report.json')
    official, parsed = report.get('official_evaluation', {}), report.get('response_parsing', {})
    if (report.get('status') != 'PASS' or official.get('status') != 'PASS'
            or official.get('execution_status') != 'PASS' or not official.get('cli_function_agreement')
            or official.get('actual_denominator') != 5000 or official.get('response_policy') != parsing.VERSION
            or parsed.get('policy') != parsing.POLICY or parsed.get('rows') != 5000):
        raise ValueError('no complete v2 official score')
    if report.get('mode') == 'offline_rescore':
        if report.get('source') != provenance:
            raise ValueError('offline score belongs to other source artifacts')
    elif (report.get('run_identity') != identity
          or report.get('inference_coverage') != {'status': 'PASS', 'rows': 5000, 'skipped': 0}):
        raise ValueError('score does not cover the original run identity')
    normalized = parsing.read_complete_jsonl(directory / 'predictions_normalized.jsonl')
    expected, correct = [], 0
    for record in records:
        matched = parsing.parse_response(record['response'], record['answer_gt'])['matched']
        letter = next(k for k, field in zip('ABCD', dataset.CHOICES) if record[field] == record['answer_gt'])
        expected.append({**{k: record[k] for k in dataset.METADATA}, 'response': letter if matched else ''})
        correct += matched
    if normalized != expected or parsed.get('matched_rows') != correct:
        raise ValueError('normalized predictions or match totals differ from raw generations')
    for filename, key in (('predictions_normalized.jsonl', 'normalized_predictions_sha256'),
                          ('response_parsing.jsonl', 'parsing_details_sha256')):
        if storage.file_sha(directory / filename) != parsed.get(key):
            raise ValueError(f'v2 scoring artifact hash differs: {filename}')
    audit = scoring.audit_scorer(script)
    if official.get('scorer') != audit:
        raise ValueError('score used a different official scorer')
    with contextlib.redirect_stdout(io.StringIO()):
        function = runpy.run_path(str(script))['calculate_accuracy_per_task_and_category']
        groups, categories, accuracy, total = function(normalized)
    if (total != 5000 or accuracy != correct / 5000 or official.get('overall_accuracy') != accuracy
            or official.get('category_accuracy') != dict(categories)
            or official.get('category_subcategory_accuracy') != {k: dict(v) for k, v in groups.items()}):
        raise ValueError('official score differs from raw-response recomputation')
    return {'score_dir': str(directory), 'correct': correct, 'total': total, 'accuracy': accuracy,
            'category_accuracy': dict(categories), 'response_policy': parsing.VERSION}


def inspect(args, cache=None):
    directory = args.output_dir
    if not directory.exists():
        return {'status': 'NEW', 'rows': 0}
    try:
        with existing_lock(directory):
            if not any(p.name != '.mmsu.lock' for p in directory.iterdir()):
                return {'status': 'NEW', 'rows': 0}
            identity = read_json(directory / 'run_config.json')
            check_identity(args, identity, cache or FileCache())
            rows, tail = raw_prefix(directory / 'raw_generations.jsonl')
            if len(rows) < 5000:
                if (directory / 'predictions_official.jsonl').exists():
                    # Only the original offline reader may use predictions-only
                    # fallback, and only if raw is entirely absent.
                    records, provenance = rescore.read_source(directory)
                else:
                    current = evaluate.identity(args, dataset.shard_paths(args.dataset_dir))
                    if current != identity:
                        raise ValueError('partial MMSU run requires its original code/protocol to resume')
                    return {'status': 'RESUME', 'rows': len(rows), 'unfinished_tail': tail}
            else:
                if tail:
                    raise ValueError('complete raw has an unfinished extra tail; source preserved')
                records, provenance = rescore.read_source(directory)
            model = read_json(directory / 'model_audit.json')
            if (model.get('status') != 'PASS' or model.get('checkpoint_step') != args.expected_step
                    or model.get('strict_state_dict') is not True):
                raise ValueError('complete generation lacks a successful strict model load audit')
            common = {k: identity[k] for k in ('shards', 'protocol', 'evaluation_code')}
            common['scoring_policy'] = parsing.POLICY
            # Prefer a separately scored v2 copy over historical scoring.
            errors = []
            for scored in (directory / rescore.SUBDIRECTORY, directory):
                try:
                    score = verified_score(scored, records, identity, provenance, args.evaluation_script)
                    return {'status': 'COMPLETE', 'rows': 5000, 'score': score, 'common': common,
                            'metadata_sha256': storage.object_sha([{k: r[k] for k in dataset.METADATA} for r in records])}
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(str(exc))
            return {'status': 'NEEDS_RESCORE', 'rows': 5000, 'score_errors': errors}
    except OutputBusy as exc:
        return {'status': 'BUSY', 'error': str(exc)}
