"""Only the two user-provided step34626 checkpoints and their eight evaluations."""
import json
from pathlib import Path
from .registry import PACKAGE, ROUTES, BENCHMARKS

def target_records():
    records=json.loads((PACKAGE/'targets.json').read_text(encoding='utf-8'))
    result={};seen=set()
    for record in records:
        if (set(record)!={'target','checkpoint','benchmarks'} or record['target'] not in ROUTES
                or record['target'] in result or record['benchmarks']!=list(BENCHMARKS)):
            raise ValueError('expected unique target with the four fixed benchmarks')
        path=Path(record['checkpoint'])
        if path.name!='model--step-34626.ckpt' or path.parent.name!='checkpoints' or path.parent.parent.name!='formal':
            raise ValueError('fixed target must name its explicit formal step34626 checkpoint')
        for benchmark in record['benchmarks']:
            key=(record['checkpoint'],benchmark)
            if key in seen:raise ValueError('duplicate checkpoint/benchmark in requested batch')
            seen.add(key)
        result[record['target']]=record
    if set(result)!=set(ROUTES) or len(seen)!=8:
        raise ValueError('expected exactly two targets and eight unique jobs')
    return result
