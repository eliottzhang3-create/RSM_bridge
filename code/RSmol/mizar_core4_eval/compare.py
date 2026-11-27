"""Read-only comparison of complete MMAU/MMAR/MMSU results."""
import argparse
import csv
import json
from pathlib import Path
from mizar_s1_ced_reasonaqa5000_eval import storage
from .comparison_policy import known_variants,common_protocol,shared_evaluation_code,execution_columns

def input_fingerprint(generation):
    import re
    actual = {key:generation.get(key) for key in ('prompt_token_ids','prompt_attention_mask','audio_half_sha256')}
    ids,mask,audio = actual.values()
    if (not isinstance(ids,list) or len(ids)!=256 or any(type(x) is not int or x<0 for x in ids)
            or not isinstance(mask,list) or len(mask)!=256 or not any(mask)
            or any(type(x) is not int or x not in (0,1) for x in mask)
            or not isinstance(audio,list) or len(audio)!=2
            or any(not isinstance(x,str) or not re.fullmatch('[0-9a-f]{64}',x) for x in audio)
            or storage.object_sha(actual)!=generation.get('input_fingerprint_sha256')):
        raise ValueError('invalid per-sample model input fingerprint')
    return generation['input_fingerprint_sha256']


def compare_benchmarks(benchmark, directories, output):
    from .registry import ROWS
    from . import mmsu_runtime, runtime
    from mizar_s1_ced_mmsu_eval import parsing
    expected = ROWS[benchmark]
    rows, reference = [], None
    for name, directory in directories:
        identity = json.loads((directory / 'run_config.json').read_text(encoding='utf-8'))
        report = json.loads((directory / 'evaluation_report.json').read_text(encoding='utf-8'))
        if (report.get('status') != 'PASS' or report.get('run_identity') != identity
                or identity.get('benchmark') != benchmark or identity.get('eval_variant') not in known_variants()
                or storage.object_sha({k:v for k,v in identity.items() if k!='identity_sha256'}) != identity['identity_sha256']):
            raise ValueError(f'incomplete/inconsistent report: {directory}')
        coverage = report.get('inference_coverage', {})
        if coverage.get('status') != 'PASS' or coverage.get('rows') != expected or coverage.get('skipped') != 0:
            raise ValueError('incomplete inference coverage report')
        raw_bytes = (directory / 'raw_generations.jsonl').read_bytes()
        if not raw_bytes.endswith(b'\n'):
            raise ValueError('incomplete raw JSONL')
        raw = [json.loads(line) for line in raw_bytes.splitlines()]
        if len(raw) != expected or len({str(r['id']) for r in raw}) != expected:
            raise ValueError('raw coverage differs')
        for index, record in enumerate(raw):
            if record['row_index'] != index:
                raise ValueError('raw rows are not contiguous')
        official = report['official_evaluation']
        if official.get('status') != 'PASS':
            raise ValueError('official evaluation did not pass')
        if benchmark == 'mmsu':
            common = {k:identity[k] for k in ('shards','scorer')}
            common['protocol']=common_protocol(identity,benchmark)
            common['shared_evaluation_code']=shared_evaluation_code(identity,benchmark)
            correct = 0
            inputs = []
            for record in raw:
                original, generated = record['official_record'], record['generation']
                if (record['record_sha256'] != storage.object_sha(original)
                        or record['generation_sha256'] != storage.object_sha(generated)):
                    raise ValueError('corrupt MMSU record hashes')
                correct += parsing.parse_response(generated['generated_text'],original['answer_gt'])['matched']
                inputs.append((record['record_sha256'],input_fingerprint(generated)))
            score = correct / expected
            if official['actual_denominator'] != expected or official['overall_accuracy'] != score:
                raise ValueError('MMSU raw and score disagree')
        else:
            common = {k:identity[k] for k in ('metadata_sha256','parquet_sha256','evaluation_sha256')}
            common['generation']=common_protocol(identity,benchmark)
            common['shared_evaluation_code']=shared_evaluation_code(identity,benchmark)
            if any(r.get('status') != 'generated' for r in raw) or official['reported_rows'] != expected:
                raise ValueError('incomplete generation/scorer denominator')
            filename = 'predictions_fixed_order.json' if benchmark=='mmau' else 'predictions_official.json'
            predictions = json.loads((directory / filename).read_text(encoding='utf-8'))
            key = identity['prediction_key']
            if (len(predictions)!=expected
                    or [str(r['id']) for r in predictions] != [str(r['id']) for r in raw]):
                raise ValueError('prediction coverage differs')
            by_id = {str(r['id']):r for r in raw}
            for prediction in predictions:
                record = by_id[str(prediction['id'])]
                if prediction[key] != record['generated_text']:
                    raise ValueError('official input does not preserve raw generation')
                if {k:v for k,v in prediction.items() if k!=key} != {k:v for k,v in record['official_record'].items() if k!=key}:
                    raise ValueError('official prediction metadata differs')
            inputs = [(storage.object_sha(r['official_record']),input_fingerprint(r)) for r in raw]
            score = official['total_accuracy_percent'] / 100
        if not 0 <= score <= 1:
            raise ValueError('invalid accuracy')
        if reference is None:
            reference = (common, inputs)
        elif reference != (common, inputs):
            raise ValueError('benchmark data/protocol/code/record order differ across models')
        rows.append({'target':name,'eval_variant':identity['eval_variant'],
                     'checkpoint': identity['checkpoint']['path'] if isinstance(identity['checkpoint'],dict) else identity['checkpoint'],
                     'total':expected,'accuracy':score,'output_dir':str(directory), **execution_columns(identity)})
    if not rows:
        raise ValueError('no runs to compare')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    storage.write_json(output.with_suffix('.json'),{'status':'PASS','benchmark':benchmark,'rows':rows})
    return rows


def main(argv=None):
    from .registry import BENCHMARKS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',choices=BENCHMARKS,required=True)
    p.add_argument('--report-dir',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    directories=[(str(i),path.resolve()) for i,path in enumerate(args.report_dir)]
    try:
        rows=compare_benchmarks(args.benchmark,directories,args.output.resolve())
    except (OSError,ValueError,KeyError,TypeError) as exc:
        p.exit(1,f'Comparison failed: {exc}\n')
    print(json.dumps(dict(status='PASS',runs=len(rows),csv=str(args.output))))
    return 0

if __name__=='__main__':raise SystemExit(main())
