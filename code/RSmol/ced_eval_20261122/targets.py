"""Exact checkpoint/run/task bindings from the user's 2026-11-22 request."""
import json
from pathlib import Path
from .registry import PACKAGE, ROUTES, BENCHMARKS


def target_records():
    records=json.loads((PACKAGE/'targets.json').read_text(encoding='utf-8'))
    result={}
    seen=set()
    for record in records:
        if (set(record)!={'target','checkpoint','benchmarks'} or record['target'] not in ROUTES
                or record['target'] in result or not record['benchmarks']
                or len(record['benchmarks'])!=len(set(record['benchmarks']))
                or not set(record['benchmarks'])<=set(BENCHMARKS)):
            raise ValueError('invalid/repeated target or benchmark in fixed manifest')
        path=Path(record['checkpoint'])
        if path.name!='model--step-34626.ckpt' or path.parent.name!='checkpoints' or path.parent.parent.name!='formal':
            raise ValueError('fixed target must name its explicit formal step34626 checkpoint')
        for benchmark in record['benchmarks']:
            key=(record['checkpoint'],benchmark)
            if key in seen:raise ValueError('duplicate checkpoint/benchmark in requested batch')
            seen.add(key)
        result[record['target']]=record
    if set(result)!=set(ROUTES) or len(seen)!=19:
        raise ValueError('expected exactly seven targets and nineteen unique jobs')
    return result
