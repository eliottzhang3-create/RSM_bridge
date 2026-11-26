"""Independent continuation identity, original screening untouched."""
import json
import os
import subprocess
import sys
import time
import traceback
from mizar_layer_pair_search import storage
from . import arguments, audit, config, identity, manifest, results, source


def failures(directory):
    return [{'file':str(path),**storage.read_json(path)} for path in sorted(directory.glob('ranks/rank*/failure.json'))]


def launch(args):
    command=[sys.executable,'-u','-B','-m','torch.distributed.run','--standalone','--nnodes=1',
        '--nproc-per-node=8','--max-restarts=0','--module','mizar_reasonaqa_tracing500.worker',*arguments.forwarded(args)]
    print(json.dumps({'event':'LAUNCH_TRACING500','selected':500,'new_screening_forwards':0,'argv':command}),flush=True)
    completed=subprocess.run(command,cwd=config.ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1'))
    if completed.returncode:
        details=failures(args.output_dir)
        roots='; '.join(f'rank{r.get("rank","?")} {r.get("phase","?")}: {r.get("error","?")}' for r in details)
        raise RuntimeError(f'torchrun failed ({completed.returncode}); {roots or "no Python failure file; inspect GPU job log"}')


def finalize(args, frozen, expected, context):
    if not all((args.output_dir/'ranks'/f'rank{rank:02d}'/'rank_report.json').is_file() for rank in range(8)):
        return None
    summaries=[]
    for layer in range(1,31):
        parts=[results.read_rank(args.output_dir,rank,frozen,expected,layer) for rank in range(8)]
        if any(len(part)!=len(config.indices(rank)) for rank,part in enumerate(parts)):
            return None
        rows=results.merge(parts,frozen,expected,layer)
        old=args.output_dir/'layers'/f'patch_{layer:02d}'/'summary.json';elapsed=0
        if old.exists():
            saved=storage.verify_seal(storage.read_json(old))
            if saved['identity_sha256']!=expected['identity_sha256']:
                raise ValueError('saved layer summary belongs to another continuation')
            elapsed=saved['elapsed_seconds']
        summaries.append(results.save_layer(args.output_dir,rows,frozen,expected,layer,elapsed))
    checks=audit.verify(args,frozen,expected,context)
    results.write_tables(args.output_dir,summaries)
    from .plot import make_plot
    return {'status':'PASS','selected':500,'patched_layers':30,'patched_decisions':15000,'new_screening_forwards':0,
        'source_qualifying':frozen['source_total_qualifying'],'manifest':str(args.output_dir/'selected/manifest.json'),
        'manifest_sha256':frozen['manifest_sha256'],'category_counts':frozen['category_counts'],
        'runtime_checks_sha256':checks['sha256'],'plots':make_plot(args.output_dir),
        'metric':'mean_per_question_unclipped_probability_space_RR',
        'conditional_clean_accuracy':1.0,'conditional_corrupted_accuracy':0.0,
        'note':'first500 qualifying rows of original shuffled screening; layer30 is a readout restoration control'}


def run(args):
    report={'schema':config.SCHEMA,'status':'FAILED'};started=time.monotonic()
    try:
        print(json.dumps({'event':'VERIFY_SOURCE_SCREENING','source_output_dir':str(args.source_output_dir)}),flush=True)
        context=source.load(args.source_output_dir,full=True,verify_caches=True)
        frozen=source.manifest(context,args.source_output_dir);expected=identity.build(args,frozen,context)
        with storage.output_lock(args.output_dir):
            storage.claim(args.output_dir,expected)
            try:
                manifest.publish(args,frozen)
                report.update(identity_sha256=expected['identity_sha256'],source_identity_sha256=context['expected']['identity_sha256'])
                complete=finalize(args,frozen,expected,context)
                if complete is None:
                    launch(args);complete=finalize(args,frozen,expected,context)
                if complete is None:
                    raise RuntimeError('incomplete500-question coverage or missing runtime evidence')
                identity.verify(args,expected,full=True)
                report.update(complete)
            except Exception as exc:
                report.update(error=repr(exc),traceback=traceback.format_exc(),worker_failures=failures(args.output_dir))
            finally:
                report['elapsed_seconds']=time.monotonic()-started
                storage.write_json(args.output_dir/'analysis_report.json',report)
    except Exception as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc())
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def main(argv=None):
    args=arguments.parse_args(argv)
    if args.check_args:
        print(json.dumps({'args':vars(args),'protocol':config.PROTOCOL},default=str,indent=2));return 0
    return 0 if run(args)['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
