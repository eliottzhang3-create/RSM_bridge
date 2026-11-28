"""Pinned repository-layout snapshot, including every transitive S3 code_digest input."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from .registry import PACKAGE,ROOT,SOURCE_CACHE,variant_spec
from mizar_s1_ced_reasonaqa5000_eval.storage import output_lock


def profile(target):
    variant_spec(target)
    return json.loads((PACKAGE/'source_manifest.json').read_text(encoding='utf-8'))[target]


def snapshot_root(target,cache=SOURCE_CACHE):
    return (Path(cache)/profile(target)['code_sha256']).resolve()


def route_root(target,cache=SOURCE_CACHE):
    return snapshot_root(target,cache)/'code/RSmol'/variant_spec(target)['source_relative']


def verify_snapshot(target,snapshot):
    wanted=profile(target)['files']
    for name in wanted:
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or not name.startswith('code/RSmol/'):
            raise ValueError('invalid pinned repository path: '+name)
    actual={p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*')
            if p.is_file() and p.suffix in ('.py','.sh') and '__pycache__' not in p.parts}
    if actual!={n for n in wanted if Path(n).suffix in ('.py','.sh')}:
        raise ValueError('S3 executable snapshot inventory differs')
    for name,digest in wanted.items():
        p=snapshot/name
        if (not p.is_file() or p.is_symlink() or any(parent.is_symlink() for parent in p.parents if parent!=snapshot.parent)
                or hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest):
            raise ValueError('S3 source snapshot changed: '+name)
    return snapshot


def verify(target,root):
    snapshot=root.parents[2]
    if root!=snapshot/'code/RSmol'/variant_spec(target)['source_relative']:
        raise ValueError('unexpected S3 snapshot layout')
    verify_snapshot(target,snapshot)
    return root


def prepare(target,cache=SOURCE_CACHE,revision=None):
    destination=snapshot_root(target,cache)
    if destination.exists():return verify(target,route_root(target,cache))
    cache=Path(cache).resolve();cache.mkdir(parents=True,exist_ok=True)
    with output_lock(cache/('lock_'+profile(target)['code_sha256'])):
        if destination.exists():return verify(target,route_root(target,cache))
        values={}
        for name,digest in profile(target)['files'].items():
            value=subprocess.check_output(['git','show',(revision or profile(target)['revision'])+':'+name],cwd=ROOT.parents[1]).replace(b'\r\n',b'\n')
            if hashlib.sha256(value).hexdigest()!=digest:raise ValueError('pinned source mismatch: '+name)
            values[name]=value
        temporary=Path(tempfile.mkdtemp(prefix='.s3-eval-',dir=cache))
        for name,value in values.items():
            path=temporary/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
        verify_snapshot(target,temporary)
        os.replace(temporary,destination)
    return verify(target,route_root(target,cache))


def load_contract(target,cache=SOURCE_CACHE,revision=None):
    root=prepare(target,cache,revision)
    spec=importlib.util.spec_from_file_location('_s3_eval_contract',root/'contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.code_digest()!=profile(target)['code_sha256']:raise ValueError('S3 transitive code digest differs')
    return module


def import_training(target,cache=SOURCE_CACHE,revision=None):
    root=prepare(target,cache,revision)
    prefixes={'contract','model','assets','training_helpers','models','data','utils'}
    for name in list(sys.modules):
        if name.split('.')[0] in prefixes:del sys.modules[name]
    sys.path.insert(0,str(root))
    modules={name:importlib.import_module(name) for name in
             ('contract','assets','model','data.fixed20','data.audiotext_dataset','training_helpers')}
    if modules['contract'].code_digest()!=profile(target)['code_sha256']:raise ValueError('loaded S3 code digest differs')
    for name,module in list(sys.modules.items()):
        location=getattr(module,'__file__',None)
        if name.split('.')[0] in prefixes and location and not Path(location).resolve().is_relative_to(root):
            raise ValueError('S3 transitive import escaped snapshot: '+name)
    return modules
