"""Offline validation; production assets/CUDA/vc submission are never executed."""
from __future__ import annotations
import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ROUTE=ROOT/'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small'
ENTRY=ROOT/'code/RSmol/run_mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small_3090.sh'


def artifact_checks():
    import hashlib
    files=[p for p in ROUTE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [ENTRY,Path(__file__),ROOT/'tests/test_audio_hrm_prefixlm_route.py',ROOT/'tests/test_audio_hrm_prefixlm_tensors.py',ROOT/'tests/test_audio_hrm_prefixlm_ddp.py']
    for path in files:
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path),feature_version=(3,10))
        if path.suffix in ('.py','.sh','.md'):
            for i,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
                if line.rstrip()!=line:raise ValueError(f'trailing whitespace:{path}:{i}')
    bash='C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else shutil.which('bash')
    for path in (ENTRY,ROUTE/'run.sh'):subprocess.run([bash,'-n',str(path)],capture_output=True,check=True)
    readme=ROUTE/'README.md'
    blocks=re.findall(r'```bash\n(.*?)```',readme.read_text(encoding='utf-8'),re.S) if readme.exists() else []
    for block in blocks:subprocess.run([bash,'-n'],input=block,text=True,capture_output=True,check=True)
    guard=json.loads((ROUTE/'source_guard.json').read_text())
    changed=[n for n,v in guard.items() if hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=v]
    if changed:raise ValueError(f'existing tracked files changed:{changed}')
    sync=sorted(p.relative_to(ROOT).as_posix() for p in files)+['audit_reports/audio_hrm_prefixlm_validation.json']
    ignored=subprocess.run(['git','check-ignore','--stdin'],input='\n'.join(sync)+'\n',cwd=ROOT,capture_output=True,text=True)
    if ignored.stdout.strip():raise ValueError('deliverables ignored by Git:'+ignored.stdout)
    subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,check=True)
    (ROUTE/'sync_files.txt').write_text('\n'.join(sorted(set(sync+[(ROUTE/'sync_files.txt').relative_to(ROOT).as_posix()])))+'\n',encoding='utf-8',newline='\n')
    return dict(status='PASS',python310=True,bash=True,readme_bash_blocks=len(blocks),
        protected_existing_files=len(guard),git_visible_files=len(set(sync)),whitespace=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-dir',type=Path,action='append',default=[])
    parser.add_argument('--require-tensors',action='store_true')
    parser.add_argument('--ddp-python',type=Path,help='Optional existing Python with a working CPU Gloo; inherits no dependency-dir override')
    parser.add_argument('--artifacts-only',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'audit_reports/audio_hrm_prefixlm_validation.json')
    args=parser.parse_args()
    if args.artifacts_only:
        report=json.loads(args.output.read_text(encoding='utf-8'));report['artifact_checks']=artifact_checks()
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps(report['artifact_checks']));return 0
    env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    if args.dependency_dir:
        paths=[str(p.resolve(strict=True)) for p in args.dependency_dir]
        env['PYTHONPATH']=os.pathsep.join(paths+[env.get('PYTHONPATH','')])
    checks=[]
    for name in ('test_audio_hrm_prefixlm_route.py','test_audio_hrm_prefixlm_tensors.py','test_audio_hrm_prefixlm_ddp.py'):
        start=time.monotonic()
        test_env=env.copy();python=sys.executable
        if name.endswith('_ddp.py') and args.ddp_python:
            python=str(args.ddp_python.resolve(strict=True));test_env.pop('PYTHONPATH',None)
        result=subprocess.run([python,'-B',str(ROOT/'tests'/name),'-v'],cwd=ROOT,env=test_env,
            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=360)
        output=result.stdout+result.stderr
        found=re.search(r'Ran (\d+) tests?',output);skips=re.search(r'skipped=(\d+)',output)
        item=dict(test_file=name,tests=int(found[1]) if found else 0,skipped=int(skips[1]) if skips else 0,
            exit_code=result.returncode,seconds=time.monotonic()-start,output=output)
        item['status']='FAIL' if result.returncode or not found else 'PASS_WITH_SKIPS' if item['skipped'] else 'PASS'
        checks.append(item);print(json.dumps({k:v for k,v in item.items() if k!='output'}),flush=True)
        if result.returncode:print(output[-10000:],flush=True)
    versions=subprocess.run([sys.executable,'-B','-c',
        'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ["torch","transformers","torchaudio","soundfile","safetensors","numpy"]}))'],
        env=env,cwd=ROOT,capture_output=True,text=True,timeout=120)
    failed=any(item['status']=='FAIL' for item in checks)
    skips=sum(item['skipped'] for item in checks)
    if args.require_tensors and skips:failed=True
    report=dict(status='FAIL' if failed else 'PASS_LOCAL_WITH_SKIPS' if skips else 'PASS_LOCAL_CPU',
        tests_run=sum(c['tests'] for c in checks),passed=sum(c['tests']-c['skipped'] for c in checks if c['status']!='FAIL'),
        skipped=skips,checks=checks,versions=json.loads(versions.stdout) if versions.returncode==0 else {'error':versions.stderr},
        production_weights_read=False,remote_jobs_submitted=False,cuda_executed=False,
        fixture_scope='Real CPU HRM/SDPA/mapper/autograd/audio; production CED encoder replaced with explicit synthetic fixture in core gradient tests.')
    if args.output.exists():
        previous=json.loads(args.output.read_text(encoding='utf-8'))
        history=previous.get('previous_runs',[])
        if previous.get('status')=='FAIL':history=history+[{k:v for k,v in previous.items() if k!='previous_runs'}]
        if history:report['previous_runs']=history
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('checks','previous_runs')},ensure_ascii=False,indent=2))
    return int(failed)


if __name__=='__main__':raise SystemExit(main())
