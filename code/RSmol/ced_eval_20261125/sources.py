"""Immutable, content-verified training snapshots; current evaluators stay current."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile

from .registry import PACKAGE, ROOT, variant_spec
from mizar_s1_ced_reasonaqa5000_eval.storage import output_lock, write_json


def profile(target):
    revision = variant_spec(target)['revision']
    manifest = json.loads((PACKAGE/'source_manifest.json').read_text(encoding='utf-8'))
    return revision, manifest['profiles'][revision]


def route_root(target, cache):
    spec = variant_spec(target)
    return Path(cache)/spec['revision']/spec['source_relative']


def verify(target, cache):
    revision, expected = profile(target)
    snapshot = Path(cache)/revision
    for name, wanted in expected['files'].items():
        path = snapshot/name
        if (not path.is_file() or path.is_symlink() or
            hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=wanted):
            raise ValueError(f'training snapshot content mismatch: {path}')
    actual = {p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*')
              if p.is_file() and p.suffix in ('.py','.sh')}
    wanted = {p for p in expected['files'] if p.endswith(('.py','.sh'))}
    if actual != wanted:
        raise ValueError('training snapshot has missing/extra executable source files')
    return route_root(target, cache).resolve()


def prepare(target, cache):
    revision, expected = profile(target)
    cache = Path(cache).resolve()
    destination = cache/revision
    cache.mkdir(parents=True,exist_ok=True)
    # Prepared snapshots are immutable. Concurrent GPU readers need no write lock.
    if destination.exists():
        return verify(target,cache)
    with output_lock(cache/('lock_'+revision)):
        if destination.exists():
            return verify(target,cache)
        # Git archives source only. No checkout/branch switch, network or weights.
        result = subprocess.run(['git','archive','--format=tar',revision,*expected['archive_paths']],
            cwd=ROOT.parents[1],capture_output=True,check=True)
        staging = Path(tempfile.mkdtemp(prefix='.preparing_',dir=cache))
        written = set()
        with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:') as archive:
            for member in archive:
                if member.name not in expected['files']:
                    continue
                name = PurePosixPath(member.name)
                if (not member.isfile() or name.is_absolute() or '..' in name.parts
                        or member.name in written):
                    raise ValueError(f'invalid source archive member: {member.name}')
                value = archive.extractfile(member).read().replace(b'\r\n',b'\n')
                if hashlib.sha256(value).hexdigest()!=expected['files'][member.name]:
                    raise ValueError(f'Git source differs from pinned manifest: {member.name}')
                path = staging/member.name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(value)
                written.add(member.name)
        if written != set(expected['files']):
            raise ValueError('incomplete pinned source archive')
        write_json(staging/'snapshot_identity.json',{'revision':revision,'files':expected['files']})
        os.replace(staging,destination)
        return verify(target,cache)


def activate(target, cache):
    root = verify(target,cache)
    shared = root.parents[1]
    namespaces = ('models','data','utils','contract','mizar_text_7_14x2_7_mesh',
                  'recursive_model_mizar_text_7_14x2_7_mesh','mizar_text_7_14x3_7_mesh',
                  'recursive_model_mizar_text_7_14x3_7_mesh','text_hrm_h2l3',
                  'text_hrm_reinit3600_bp5','audit_mizar_s1_raw_audio')
    for name in list(sys.modules):
        if any(name==prefix or name.startswith(prefix+'.') for prefix in namespaces):
            del sys.modules[name]
    sys.path[:0] = [str(root),str(shared),str(shared/'scripts')]
    importlib.invalidate_caches()
    return root


def load_contract(target, cache):
    root = activate(target,cache)
    definition = importlib.util.spec_from_file_location('_ced20261125_contract_'+target,root/'contract.py')
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    for package in variant_spec(target)['shared_packages']:
        shared = importlib.import_module(package+'.contract')
        if Path(shared.__file__).resolve()!=root.parents[1]/package/'contract.py':
            raise ValueError('shared text contract escaped the pinned snapshot')
    return module


def import_training(target, cache):
    from . import runtime
    root = activate(target,cache)
    modules = runtime.reference.import_training_route(root)
    packages=variant_spec(target)['shared_packages']
    for name,module in list(sys.modules.items()):
        if any(name==package or name.startswith(package+'.') or name=='recursive_model_'+package
               for package in packages) or name=='audit_mizar_s1_raw_audio':
            if getattr(module,'__file__',None) and not Path(module.__file__).resolve().is_relative_to(root.parents[1]):
                raise ValueError(f'shared model import escaped pinned source: {name}')
    return modules
