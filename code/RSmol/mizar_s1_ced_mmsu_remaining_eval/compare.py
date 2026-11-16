"""Summarize independently verified complete MMSU v2 results; never generate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mizar_s1_ced_mmsu_eval import storage
from . import config, inspection, submit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--output', type=Path, default=config.DEFAULT_SUMMARY)
    own, forwarded = parser.parse_known_args(argv)
    args = submit.parse_args(forwarded)
    items = submit.planned(args)
    rows, details, common, code = [], [], None, {}
    cache = inspection.FileCache()
    for name, _, resolved in items:
        row = {'target': name, 'eval_variant': resolved.eval_variant, 'checkpoint': str(resolved.checkpoint),
               'output_dir': str(resolved.output_dir), 'status': 'FAILED',
               'correct': '', 'total': '', 'accuracy': '', 'score_dir': '', 'error': ''}
        try:
            state = inspection.inspect(resolved, cache)
            details.append({'target': name, **state})
            row['status'] = state['status']
            if state['status'] == 'COMPLETE':
                current = dict(state['common'])
                current_code = current.pop('evaluation_code')
                current['protocol'] = {k: v for k, v in current['protocol'].items() if k not in ('version', 'scoring')}
                current['metadata_sha256'] = state['metadata_sha256']
                if common is not None and common != current:
                    raise ValueError('MMSU data, order or generation/scoring protocol differs across results')
                if any(path in code and code[path] != sha for path, sha in current_code.items()):
                    raise ValueError('shared evaluation source versions differ across results')
                common = current
                code.update(current_code)
                for key in ('correct', 'total', 'accuracy', 'score_dir'):
                    row[key] = state['score'][key]
        except Exception as exc:
            row.update(status='FAILED', error=repr(exc))
        rows.append(row)
    output = own.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    ready = all(row['status'] == 'COMPLETE' for row in rows)
    storage.write_json(output.with_suffix('.json'),
        {'status': 'PASS' if ready else 'INCOMPLETE', 'metric': inspection.parsing.VERSION,
         'expected_targets': len(items), 'complete_targets': sum(row['status'] == 'COMPLETE' for row in rows),
         'excluded': config.EXCLUDED, 'common': common, 'evaluation_code': code, 'rows': rows, 'details': details})
    print(json.dumps({'status': 'PASS' if ready else 'INCOMPLETE', 'complete': sum(row['status'] == 'COMPLETE' for row in rows),
                      'expected': len(items), 'csv': str(output), 'json': str(output.with_suffix('.json'))}))
    return 0 if ready else 1


if __name__ == '__main__':
    raise SystemExit(main())
