import json
import os
import subprocess
import sys
import time
import traceback
from mizar_layer_pair_search import storage
from . import arguments,audit,config,identity,manifest,results


def failures(directory):return [{'file':str(p),**storage.read_json(p)} for p in sorted(directory.glob('ranks/rank*/failure.json'))]


def launch(a):
    command=[sys.executable,'-u','-B','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8',
        '--max-restarts=0','--module','mizar_triplet_ablation.worker',*arguments.forwarded(a)]
    print(json.dumps({'event':'LAUNCH_TRIPLET','task':a.task,'argv':command}),flush=True)
    done=subprocess.run(command,cwd=config.ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'))
    if done.returncode:
        detail='; '.join(f'rank{r.get("rank")}: {r.get("error")}' for r in failures(a.output_dir))
        raise RuntimeError(f'torchrun failed({done.returncode}); {detail or "inspect GPU job log"}')


def finalize(a,frozen,expected):
    if not all((a.output_dir/'ranks'/f'rank{r:02d}'/'rank_report.json').is_file() for r in range(8)):return None
    summaries=[];baseline=None;records=frozen['records']
    for start in config.CONDITIONS:
        parts=[results.read_rank(a.output_dir,r,records,a.task,start,expected) for r in range(8)]
        if any(len(part)!=125 for part in parts):return None
        rows=results.merge(parts,records,a.task,start,expected)
        if start is None:baseline=rows
        elapsed=0;path=a.output_dir/'conditions'/config.condition_name(a.task,start)/'summary.json'
        if path.is_file():
            saved=storage.verify_seal(storage.read_json(path))
            if saved['identity_sha256']!=expected['identity_sha256']:raise ValueError('condition summary identity changed')
            elapsed=saved['elapsed_seconds']
        summaries.append(results.save_condition(a.output_dir,rows,baseline,records,a.task,start,expected,elapsed))
    checks=audit.verify(a.output_dir,expected,records)
    ranked=results.save_analysis(a.output_dir,summaries,frozen)
    from .plot import make_plot
    return {'status':'PASS','conditions_completed':21,'decisions':21000,'rows_per_condition':1000,
        'baseline_accuracy':summaries[0]['accuracy'],'top_accuracy':ranked[0]['accuracy'],
        'top_tied_windows_1based':[r['drop_layers_1based'] for r in ranked if r['rank']==1],
        'group_counts':config.GROUP_COUNTS[a.task],'runtime_checks_sha256':checks['sha256'],'plots':make_plot(a.output_dir),
        'generation_count':0,'primary_metric':'bare_lowercase_letter_likelihood_argmax'}


def run(a):
    report={'schema':config.SCHEMA,'task':a.task,'status':'FAILED'};started=time.monotonic()
    try:
        frozen=manifest.ensure(a);expected=identity.build(a,frozen)
        with storage.output_lock(a.output_dir):
            storage.claim(a.output_dir,expected)
            try:
                report.update(identity_sha256=expected['identity_sha256'],manifest_sha256=frozen['manifest_sha256'])
                complete=finalize(a,frozen,expected)
                if complete is None:launch(a);complete=finalize(a,frozen,expected)
                if complete is None:raise RuntimeError('incomplete125x8x21 raw coverage or runtime evidence')
                identity.verify(a,expected,full=True);report.update(complete)
            except Exception as exc:report.update(error=repr(exc),traceback=traceback.format_exc(),worker_failures=failures(a.output_dir))
            finally:
                report['elapsed_seconds']=time.monotonic()-started;storage.write_json(a.output_dir/'analysis_report.json',report)
    except Exception as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
    print(json.dumps(report,ensure_ascii=False),flush=True);return report


def main(argv=None):
    a=arguments.parse_args(argv)
    if a.check_args:print(json.dumps({'args':vars(a),'protocol':config.protocol(a.task)},default=str,indent=2));return 0
    return 0 if run(a)['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
