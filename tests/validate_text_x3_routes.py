"""Validate both isolated routes. --require-tensors makes missing CPU ML dependencies fatal."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-tensors',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'audit_reports/text_x3_routes_validation.json')
    args=parser.parse_args()
    os.chdir(ROOT); sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT/'tests'))
    bash='C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else shutil.which('bash')
    if os.name=='nt': os.environ['PATH']=str(Path(bash).parent)+os.pathsep+os.environ['PATH']
    sources=[]
    for route in ('5_10x3_5','5_10x3_5_mesh_6slot'):
        sources += [ROOT/f'code/RSmol/recursive_model_{route}.py', ROOT/f'code/RSmol/scripts/convert_stepwise_{route}.py',ROOT/f'code/RSmol/scripts/train_stage4_{route}_ddp.py']
        sources += [ROOT/f'code/RSmol/text_{route}/{name}' for name in ('README.md','source_guard.json')]
        sources += [ROOT/f'code/RSmol/{name}' for name in (f'run_stage4_{route}_3090.sh',f'run_stage4_{route}_formal_3090.sh')]
        sources += [ROOT/f'code/RSmol/scripts/{name}' for name in (f'convert_stepwise_{route}.sh',f'train_stage4_{route}_ddp.sh')]
    sources += list((ROOT/'tests').glob('*text_x3*.py'))
    sources += [ROOT/'code/RSmol/text_5_10x3_5_mesh_6slot/router_transfer.py',
                ROOT/'code/RSmol/text_5_10x3_5_mesh_6slot/router_transfer_source_guard.json']
    sources=sorted(set(sources))
    for path in sources:
        if path.suffix=='.py': compile(ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,10)),str(path),'exec')
        if path.suffix=='.sh':
            if b'\r' in path.read_bytes(): raise RuntimeError(f'Bash requires LF: {path}')
            subprocess.run([bash,'-n',str(path)],check=True,capture_output=True)
        if path.name=='README.md':
            for block in re.findall(r'```bash\n(.*?)```',path.read_text(encoding='utf-8'),re.S):
                subprocess.run([bash,'-n'],input=block,text=True,capture_output=True,check=True)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_text_x3*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    success=result.wasSuccessful() and not (args.require_tensors and result.skipped)
    report=dict(status=('PASS_LOCAL_WITH_SKIPS' if result.skipped else 'PASS_LOCAL') if success else 'FAIL',time_utc=datetime.now(timezone.utc).isoformat(),
        run=result.testsRun,passed=result.testsRun-len(result.skipped)-len(result.errors)-len(result.failures),
        skipped=[{'test':test.id(),'reason':reason} for test,reason in result.skipped],
        errors=[{'test':test.id(),'traceback':message} for test,message in result.errors+result.failures],
        production_assets_read=False,remote_conversion_verified=False,gpu_training_verified=False,eight_rank_resume_verified=False,
        python_syntax_target='3.10',
        source_sha256={path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f"{report['status']}: passed={report['passed']}, skipped={len(result.skipped)}; report={args.output}")
    return 0 if success else 1


if __name__=='__main__': raise SystemExit(main())
