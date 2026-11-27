"""Immutable source snapshots bound to the unchanged training code digest."""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from .registry import PACKAGE, ROOT, SOURCE_CACHE, variant_spec
from mizar_s1_ced_reasonaqa5000_eval.storage import output_lock


def profile(target):
    return json.loads((PACKAGE/'source_manifest.json').read_text(encoding='utf-8'))[target]


def route_root(target, cache=SOURCE_CACHE):
    return (Path(cache)/profile(target)['code_sha256']/variant_spec(target)['source_relative']).resolve()


def verify(target, root):
    wanted = profile(target)['files']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p.suffix in ('.py','.sh') and '__pycache__' not in p.parts}
    if actual != {n for n in wanted if Path(n).suffix in ('.py','.sh')}:
        raise ValueError('training snapshot executable inventory differs')
    for name, sha in wanted.items():
        path = root/name
        if (not path.is_file() or path.is_symlink() or
                hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest() != sha):
            raise ValueError(f'training source differs from registered content: {path}')
    return root


def prepare(target, cache=SOURCE_CACHE, revision=None):
    variant_spec(target)
    destination = route_root(target,cache)
    if destination.exists():
        return verify(target,destination)
    cache = Path(cache).resolve()
    cache.mkdir(parents=True,exist_ok=True)
    with output_lock(cache/('lock_'+profile(target)['code_sha256'])):
        if destination.exists():
            return verify(target,destination)
        original = ROOT/variant_spec(target)['source_relative']
        revision = revision or profile(target)['revision']
        values = {}
        for name,sha in profile(target)['files'].items():
            if revision:
                repo_path = 'code/RSmol/'+variant_spec(target)['source_relative']+'/'+name
                value = subprocess.check_output(['git','show',revision+':'+repo_path],cwd=ROOT.parents[1])
            else:
                value = (original/name).read_bytes()
            value = value.replace(b'\r\n',b'\n')
            if hashlib.sha256(value).hexdigest() != sha:
                raise ValueError('source hash mismatch; supply matching --source-revision: '+name)
            values[name] = value
        temporary = Path(tempfile.mkdtemp(prefix='.l1014-',dir=cache))
        for name,value in values.items():
            path = temporary/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(value)
        verify(target,temporary)
        destination.parent.mkdir(parents=True,exist_ok=True)
        os.replace(temporary,destination)
    return verify(target,destination)


def import_training(target, cache=SOURCE_CACHE, revision=None):
    root = prepare(target,cache,revision)
    for name in list(sys.modules):
        if name in ('contract','model','assets','llama_compat','training_helpers') or name.split('.')[0] in ('models','data','utils'):
            del sys.modules[name]
    sys.path.insert(0,str(root))
    modules = {name:importlib.import_module(name) for name in
               ('contract','assets','model','data.fixed20','data.audiotext_dataset','training_helpers')}
    for name,module in modules.items():
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError('model import escaped selected source snapshot: '+name)
    if modules['contract'].code_digest() != profile(target)['code_sha256']:
        raise ValueError('loaded training code digest differs')
    for name,module in list(sys.modules.items()):
        if name.split('.')[0] in ('contract','model','assets','llama_compat','training_helpers','models','data','utils'):
            location = getattr(module,'__file__',None)
            if location and not Path(location).resolve().is_relative_to(root):
                raise ValueError('transitive training import escaped selected snapshot: '+name)
    return modules


def load_contract(target,cache=SOURCE_CACHE,revision=None):
    import importlib.util
    root = prepare(target,cache,revision)
    definition = importlib.util.spec_from_file_location('_l1014_contract_'+target,root/'contract.py')
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module
