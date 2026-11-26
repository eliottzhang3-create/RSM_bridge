import json
import os
import subprocess
import sys
import time
import traceback
from mizar_layer_pair_search import storage
from . import arguments,audit,config,identity,prepare,results,screening


def failures(directory):return [{'file':str(p),**storage.read_json(p)} for p in sorted(directory.glob('ranks/rank*/failure.json'))]


def launch(a):
    command=[sys.executable,'-u','-B','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8',
        '--max-restarts=0','--module','mizar_benchmark_tracing.worker',*arguments.forwarded(a)]
    print(json.dumps({'event':'LAUNCH_TRACING','task':a.task,'stage':a.stage,'argv':command}),flush=True)
    done=subprocess.run(command,cwd=config.ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'))
    if done.returncode:
        detail='; '.join(f'rank{r.get("rank")} {r.get("phase")}: {r.get("error")}' for r in failures(a.output_dir))
        raise RuntimeError(f'torchrun failed({done.returncode}); {detail or "inspect GPU log: no Python failure file"}')


def finalize(a,inputs,expected):
    if not (a.output_dir/'selected/manifest.json').is_file():return None
    rebuilt=screening.reconstruct(a.output_dir,inputs,expected)
    if rebuilt is None:return None
    frozen,selection=rebuilt
    if a.stage=='select':return {'status':selection['status'],'screened':1000,'selected':frozen['selected'],
        'group_counts':frozen['group_counts'],'selected_manifest':str(a.output_dir/'selected/manifest.json')}
    if not all((a.output_dir/'ranks'/f'rank{r:02d}'/'rank_report.json').is_file() for r in range(8)):return None
    summaries=[];layers=range(1,31) if frozen['selected'] else []
    for layer in layers:
        parts=[results.read_rank(a.output_dir,r,frozen,expected,layer) for r in range(8)]
        if any(len(p)!=len(config.indices(r,frozen['selected'])) for r,p in enumerate(parts)):return None
        rows=results.merge(parts,frozen,expected,layer);elapsed=0
        path=a.output_dir/'layers'/f'patch_{layer:02d}'/'summary.json'
        if path.is_file():
            old=storage.verify_seal(storage.read_json(path))
            if old['identity_sha256']!=expected['identity_sha256']:raise ValueError('summary identity changed')
            elapsed=old['elapsed_seconds']
        summaries.append(results.save_layer(a.output_dir,rows,frozen,expected,layer,elapsed))
    checks=audit.verify(a.output_dir,inputs,frozen,expected);results.tables(a.output_dir,summaries)
    from .plot import make_plot
    return {'status':'PASS' if frozen['selected'] else 'NO_QUALIFYING','screened':1000,'selected':frozen['selected'],
        'selection_quota':None,'patched_layers':len(summaries),'patched_decisions':frozen['selected']*30,
        'full1000_clean_accuracy':selection['clean_accuracy'],'full1000_silence_accuracy':selection['corrupted_accuracy'],
        'full1000_groups':selection['groups'],'selected_groups':frozen['group_counts'],
        'selected_manifest':str(a.output_dir/'selected/manifest.json'),'manifest_sha256':frozen['manifest_sha256'],
        'runtime_checks_sha256':checks['sha256'],'plots':make_plot(a.output_dir),
        'note':'RR is conditional on clean correct/silence wrong/positive gap; layer30 is a readout control'}


def run(a):
    report={'schema':config.SCHEMA,'task':a.task,'stage':a.stage,'status':'FAILED'};started=time.monotonic()
    try:
        inputs=prepare.ensure(a);expected=identity.build(a,inputs)
        with storage.output_lock(a.output_dir):
            storage.claim(a.output_dir,expected)
            try:
                report['identity_sha256']=expected['identity_sha256']
                storage.write_json(a.output_dir/'shuffle.json',storage.seal({'manifest_sha256':inputs['manifest_sha256'],
                    'seed':config.SEED,'order':inputs['shuffle_order']}))
                complete=finalize(a,inputs,expected)
                if complete is None:launch(a);complete=finalize(a,inputs,expected)
                if complete is None:raise RuntimeError('incomplete screening/patch/rank evidence')
                identity.verify(a,expected,full=True);report.update(complete)
            except Exception as exc:report.update(error=repr(exc),traceback=traceback.format_exc(),worker_failures=failures(a.output_dir))
            finally:
                report['elapsed_seconds']=time.monotonic()-started
                storage.write_json(a.output_dir/('selection_only_report.json' if a.stage=='select' else 'analysis_report.json'),report)
    except Exception as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
    print(json.dumps(report,ensure_ascii=False),flush=True);return report


def main(argv=None):
    a=arguments.parse_args(argv)
    if a.check_args:print(json.dumps({'args':vars(a),'protocol':config.protocol(a.task)},default=str,indent=2));return 0
    return 0 if run(a)['status'] in ('PASS','SELECTED','NO_QUALIFYING') else 1


if __name__=='__main__':raise SystemExit(main())
