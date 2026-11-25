"""One coordinator verifies and finalizes all letter-scored conditions."""
import json
import os
import subprocess
import sys
import time
import traceback
from mizar_layer_pair_search import storage
from . import arguments, audit, config, identity, manifest, results


def launch(args):
    command=[sys.executable,'-u','-B','-m','torch.distributed.run','--standalone','--nnodes=1',
        '--nproc-per-node=8','--max-restarts=0','--module','mizar_letter_analysis.worker',*arguments.forwarded(args)]
    print(json.dumps({'event':'LAUNCH_LETTER_EOT','task':args.task,'argv':command}),flush=True)
    completed=subprocess.run(command,cwd=config.ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'))
    if completed.returncode:raise RuntimeError(f'torchrun failed ({completed.returncode}); see ranks/rankXX/failure.json')


def finalize(args, frozen, expected):
    records=frozen['records'];directory=args.output_dir;baseline=None;summaries=[]
    for layer in config.conditions(args.task):
        parts=[results.read_rank(directory,r,records,args.task,layer,expected) for r in range(8)]
        if any(len(part)!=125 for part in parts):return None
        rows=results.merge(parts,records,args.task,layer,expected)
        if layer is None:baseline=rows
        elapsed=0
        path=directory/'conditions'/config.condition_name(args.task,layer)/'summary.json'
        if path.exists():
            old=storage.verify_seal(storage.read_json(path))
            if old['identity_sha256']!=expected['identity_sha256']:raise ValueError('summary belongs to another run')
            elapsed=old['elapsed_seconds']
        summaries.append(results.save_condition(directory,rows,baseline,records,args.task,layer,expected,elapsed))
    if not audit.ranks_complete(directory):return None
    checks=audit.verify(directory,expected,records)
    ranked=results.save_analysis(directory,summaries,frozen)
    from .plot import make_plot
    plots=make_plot(directory)
    return {'status':'PASS','conditions_completed':len(summaries),'decisions':config.ROWS*len(summaries),
        'rows_per_condition':config.ROWS,'baseline_accuracy':summaries[0]['accuracy'],'top_accuracy':ranked[0]['accuracy'],
        'top_tied_layers_1based':[r['condition_layer_1based'] for r in ranked if r['rank']==1],
        'group_counts':config.GROUP_COUNTS[args.task],'generation_count':0,
        'primary_metric':'bare_lowercase_letter_loglikelihood_argmax','runtime_checks_sha256':checks['sha256'],'plots':plots}


def run(args):
    report={'schema':config.SCHEMA,'task':args.task,'status':'FAILED'};started=time.monotonic()
    try:
        frozen=manifest.ensure(args)
        expected=identity.build(args,frozen)
        with storage.output_lock(args.output_dir):
            storage.claim(args.output_dir,expected)
            try:
                report.update(identity_sha256=expected['identity_sha256'],manifest_sha256=frozen['manifest_sha256'])
                complete=finalize(args,frozen,expected)
                if complete is None:
                    launch(args);complete=finalize(args,frozen,expected)
                if complete is None:raise RuntimeError('incomplete raw decision/rank coverage')
                identity.verify(args,expected,full=True)
                report.update(complete)
            except Exception as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
            finally:
                report['elapsed_seconds']=time.monotonic()-started
                storage.write_json(args.output_dir/'analysis_report.json',report)
    except Exception as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
    return report


def main(argv=None):
    args=arguments.parse_args(argv)
    if args.check_args:
        print(json.dumps({**vars(args),'protocol':config.protocol(args.task)},default=str,indent=2));return 0
    return 0 if run(args)['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
