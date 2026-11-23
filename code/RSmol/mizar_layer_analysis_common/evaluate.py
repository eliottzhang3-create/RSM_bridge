import json
import os
import subprocess
import sys
import time
import traceback
from mizar_layer_pair_search import storage
from . import arguments,config,identity
from .worker import adapter


def launch(args,task):
    command=[sys.executable,'-u','-B','-m','torch.distributed.run','--standalone','--nnodes=1',
        '--nproc-per-node=8','--max-restarts=0','--module',f'mizar_layer_{task}.worker',*arguments.forwarded(args)]
    print(json.dumps({'event':'LAUNCH','task':task,'argv':command}),flush=True)
    result=subprocess.run(command,cwd=config.ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'))
    if result.returncode: raise RuntimeError(f'torchrun failed ({result.returncode}); see ranks/rank*/failure.json')


def run(args,task):
    api=adapter(task); started=time.monotonic(); report={'task':task,'status':'FAILED'}
    try:
        frozen=api.prepare(args)
        expected=identity.build(args,frozen,task,api.protocol(args))
        with storage.output_lock(args.output_dir):
            storage.claim(args.output_dir,expected)
            try:
                report.update(identity_sha256=expected['identity_sha256'],manifest_sha256=frozen['manifest_sha256'])
                complete=api.finalize(args,frozen,expected)
                if complete is None:
                    launch(args,task); complete=api.finalize(args,frozen,expected)
                if complete is None: raise RuntimeError('incomplete raw coverage or worker completion reports')
                identity.verify(args,expected,task,api.protocol(args),full=True)
                report.update(complete)
            except Exception as exc: report.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
            finally:
                report['elapsed_seconds']=time.monotonic()-started
                storage.write_json(args.output_dir/'analysis_report.json',report)
    except Exception as exc: report.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
    return report


def main(task,argv=None):
    args=arguments.parse_args(task,argv)
    if args.check_args:
        print(json.dumps({**vars(args),'task':task,'protocol':adapter(task).protocol(args)},default=str,indent=2)); return 0
    return 0 if run(args,task)['status']=='PASS' else 1
