"""Source/recipe, real Bash argument chain and streaming staging regression."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
ROUTE=ROOT/'code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small'
BASE=ROUTE.with_name('mizar_s1_gbs128_text71427_mesh_ced_small')
ENTRY=ROOT/'code/RSmol/run_mizar_s1_gbs128_hrm_bp5_prefixlm_ced_small_3090.sh'
sys.path.insert(0,str(ROUTE))
import contract as C
import prepare_index,stage_data,write_config
from data.staging import StagingGate
from data.audio_paths import AudioPaths


def tree(path):return ast.parse(path.read_text(encoding='utf-8-sig'))
def definition(path,name):return next(n for n in tree(path).body if getattr(n,'name',None)==name)


class ContractTests(unittest.TestCase):
    def test_all_existing_tracked_sources_preserved(self):
        guard=json.loads((ROUTE/'source_guard.json').read_text())
        self.assertGreater(len(guard),3600)
        for name,value in guard.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),value,name)

    def test_unmodified_data_ced_pipeline_and_original_loss_gradnorm(self):
        files=[p.relative_to(BASE) for p in BASE.rglob('*.py')
            if p.parent.name in ('data','utils') or p.name in ('stage_data.py','prepare_index.py','ced.py','ced_offline.py','audio.py')]
        self.assertGreater(len(files),12)
        for name in files:self.assertEqual((ROUTE/name).read_bytes().replace(b'\r\n',b'\n'),(BASE/name).read_bytes().replace(b'\r\n',b'\n'),str(name))
        for file,name in (('train.py','answer_loss'),('optim.py','GradNormTracker')):
            self.assertEqual(ast.dump(definition(ROUTE/file,name)),ast.dump(definition(BASE/file,name)))
        for name in ('Mellow','AudioEncoder'):
            self.assertEqual(ast.dump(definition(ROUTE/'models/mellow.py',name)),ast.dump(definition(BASE/'models/mellow.py',name)))

    def test_bp_recurrence_ast_identical_except_input_mask(self):
        old=next(n for n in tree(ROOT/'code/RSmol/text_hrm_h2l3/model.py').body if isinstance(n,ast.ClassDef) and n.name=='SmolHRMModel')
        new=definition(ROUTE/'models/hrm_prefix.py','AudioPrefixHRMModel')
        def recurrence(cls):
            forward=next(n for n in cls.body if getattr(n,'name',None)=='forward')
            start=next(i for i,n in enumerate(forward.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='cos_sin')
            return [ast.dump(n) for n in forward.body[start:]]
        self.assertEqual(recurrence(old),recurrence(new))

    def test_fixed_recipe_source_paths_geometry_and_groups(self):
        geo=C.geometry(1477492)
        self.assertEqual((geo['steps_per_epoch'],geo['total_steps'],geo['warmup_steps'],geo['dropped_rows_per_epoch']),(11542,34626,1731,116))
        self.assertEqual([round(34626*i/6) for i in range(1,7)],[5771,11542,17313,23084,28855,34626])
        self.assertIn('formal_20261122_200554_2644345/checkpoint-step-009244',C.DEFAULT_TEXT_MODEL)
        self.assertEqual((C.LM_MAX_LR,C.LM_MIN_LR,C.MAP_MAX_LR,C.MAP_MIN_LR),(6e-4,3e-5,8e-4,6e-5))
        self.assertEqual(C.BP_STEPS,5);self.assertEqual(len(C.TEXT_MODEL_SCHEDULE),50)
        self.assertNotIn('--init-checkpoint',ENTRY.read_text())
        self.assertNotIn('audio_initialization', (ROUTE/'train.py').read_text().split('def train(')[1].split('source_tokenizer_audit')[0])

    def test_python310_shell_syntax_and_no_shared_model_changes(self):
        for path in list(ROUTE.rglob('*.py'))+[Path(__file__),ROOT/'tests/test_audio_hrm_prefixlm_tensors.py',ROOT/'tests/validate_audio_hrm_prefixlm.py']:
            ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path),feature_version=(3,10))
        bash='C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else shutil.which('bash')
        if not bash:self.skipTest('Bash unavailable')
        for path in (ENTRY,ROUTE/'run.sh'):subprocess.run([bash,'-n',str(path)],check=True,capture_output=True)

    def test_text_seed_and_epoch_sampler_rules_unchanged(self):
        for name in ('data_order','rank_seed','raw_references'):
            self.assertEqual(ast.dump(definition(ROUTE/'contract.py',name)),ast.dump(definition(BASE/'contract.py',name)))
        a=C.data_order(256);self.assertEqual(a,C.data_order(256));self.assertNotEqual(a,list(range(256)))


class StagingTests(unittest.TestCase):
    def test_shared_training_ready_before_copy_and_safe_local_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'audio').mkdir();source=root/'source.wav';source.write_bytes(b'raw audio bytes')
            info=dict(token='test',producer_pid=os.getpid(),epoch=0,start_batch=0,status_path=str(root/'STAGING.json'))
            C.write_json(root/'.owner.json',dict(root=str(root),token='test'))
            state={**info,'phase':'COPYING','heartbeat':time.time(),'ready_through_batch':0,'copied_files':0,'copied_bytes':0}
            prepared=dict(status='TRAINING_READY',read_policy='shared_until_complete',stage_root=str(root),async_staging=info,
                planned_files=1,planned_bytes=len(source.read_bytes()),initial_steps=0,staging_epoch_steps=11542,phase_seconds={},precopy_seconds=0.)
            C.write_json(root/'STAGING.json',state)
            started,release=threading.Event(),threading.Event();errors=[];original=stage_data.copy_one
            def blocked(*args):
                started.set()
                if not release.wait(5):raise TimeoutError('test did not release copy')
                return original(*args)
            def producer():
                try:stage_data.copy_background(root,prepared,[('audio/x.wav',dict(source=str(source),bytes=source.stat().st_size))],root/'report.json',state,workers=1)
                except BaseException as exc:errors.append(exc)
            with patch.object(stage_data,'copy_one',side_effect=blocked):
                thread=threading.Thread(target=producer);thread.start()
                try:
                    self.assertTrue(started.wait(3))
                    gate=StagingGate(json.loads((root/'report.json').read_text()),timeout=.02,poll=.001)
                    gate.wait_for_batch(0,5000);gate.wait_for_batch(1,0);self.assertFalse(gate.complete)
                finally:release.set();thread.join(5)
            self.assertFalse(errors);gate.wait_complete();self.assertTrue(gate.complete)
            self.assertEqual((root/'audio/x.wav').read_bytes(),source.read_bytes())

    def test_copy_failure_and_owner_path_guard_rejected(self):
        for name in ('/tmp/mizar_s1_gbs256_bad','/dev/shm/other','/dev/shm/mizar_s1_gbs256_bad/child'):
            with self.assertRaises(ValueError):stage_data.validate_stage_root(Path(name))
        if os.name!='nt':
            self.assertEqual(stage_data.validate_stage_root(Path('/dev/shm/mizar_s1_gbs256_hrm_prefixlm_test')),Path('/dev/shm/mizar_s1_gbs256_hrm_prefixlm_test'))


class ShellTests(unittest.TestCase):
    def test_actual_runtime_torchrun_failure_stops_and_cleans_background(self):
        bash='C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else shutil.which('bash')
        if not bash:self.skipTest('Bash unavailable')
        with tempfile.TemporaryDirectory(prefix='audio hrm runtime ') as tmp:
            root=Path(tmp);route=root/'route';route.mkdir();(route/'run.sh').write_bytes((ROUTE/'run.sh').read_bytes())
            bindir=root/'bin';bindir.mkdir();conda=root/'conda';(conda/'etc/profile.d').mkdir(parents=True)
            (conda/'etc/profile.d/conda.sh').write_text('conda() { return 0; }\n',encoding='utf-8',newline='\n')
            (bindir/'python').write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$HRM_TEST_CALLS"\nexit 0\n',encoding='utf-8',newline='\n')
            (bindir/'torchrun').write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$HRM_TEST_TORCH"\nexit 27\n',encoding='utf-8',newline='\n')
            for p in bindir.iterdir():p.chmod(0o755)
            resolved=root/'resolved.tsv';resolved.touch();text=root/'text';text.mkdir();ced=root/'ced';ced.mkdir();index=root/'index';index.mkdir()
            env=os.environ.copy();env.update(PATH=str(bindir)+os.pathsep+env['PATH'],MSYS_NO_PATHCONV='1',MSYS2_ARG_CONV_EXCL='*',MELLOW_CONDA_BASE=conda.as_posix(),HRM_TEST_CALLS=str(root/'calls'),HRM_TEST_TORCH=str(root/'torch'))
            result=subprocess.run([bash,str(route/'run.sh'),'formal',resolved.as_posix(),(root/'output with spaces').as_posix(),'',text.as_posix(),ced.as_posix(),index.as_posix(),'4'],env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,27,result.stderr)
            calls=(root/'calls').read_text();self.assertIn('--cleanup',calls)
            self.assertIn('/dev/shm/mizar_s1_gbs256_hrm_prefixlm_',calls)
            self.assertNotIn('--init-checkpoint',calls)
            args=(root/'torch').read_text().splitlines();self.assertIn('--nproc_per_node=8',args)

    def test_real_bash_submission_resources_and_quoted_argument_chain(self):
        bash='C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else shutil.which('bash')
        if not bash:self.skipTest('Bash unavailable')
        with tempfile.TemporaryDirectory(prefix='audio hrm shell ') as tmp:
            root=Path(tmp);bindir=root/'bin';bindir.mkdir();index=root/'index';index.mkdir()
            text=root/'text model';text.mkdir();(text/'config.json').write_text('{}')
            ced=root/'ced model';ced.mkdir()
            manifest=root/'manifest.json';manifest.write_text('{}');resolved=root/'resolved.tsv';resolved.write_text('fixture')
            # Minimal copied layout lets the actual launcher run without remote assets.
            code=root/'checkout';code.mkdir();route=code/'mellow_official_training_c8204d8_adamw_cosine'/ROUTE.name;route.mkdir(parents=True)
            entry=code/ENTRY.name;entry.write_bytes(ENTRY.read_bytes())
            # Replace only hard-coded source-data locations in this isolated fixture.
            content=entry.read_text().replace('MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json','MANIFEST='+shlex.quote(manifest.as_posix()))
            entry.write_text(content,encoding='utf-8',newline='\n')
            (bindir/'python').write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$HRM_TEST_PY"\n'
                'if [[ "$1" == *prepare_index.py ]]; then\nwhile [[ "$1" != --path-output ]]; do shift; done\nshift\nprintf "%s" "$HRM_TEST_INDEX" > "$1"\nfi\n',encoding='utf-8',newline='\n')
            (bindir/'vc').write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$HRM_TEST_VC"\n',encoding='utf-8',newline='\n')
            for path in bindir.iterdir():path.chmod(0o755)
            env=os.environ.copy();env.update(PATH=str(bindir)+os.pathsep+env['PATH'],MSYS_NO_PATHCONV='1',MSYS2_ARG_CONV_EXCL='*',HRM_TEST_PY=str(root/'python.txt'),HRM_TEST_VC=str(root/'vc.txt'),HRM_TEST_INDEX=index.as_posix())
            output=root/'output with spaces'
            cmd=[bash,str(entry),'--text-model-dir',text.as_posix(),'--ced-model-dir',ced.as_posix(),
                '--resolved-paths',resolved.as_posix(),'--output-dir',output.as_posix(),'--index-cache-root',(root/'cache with spaces').as_posix()]
            result=subprocess.run(cmd,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            args=(root/'vc.txt').read_text().splitlines()
            for key,value in (('-p','pdgpu-3090'),('-c','64'),('-m','256G'),('-g','8'),('-n','1')):self.assertEqual(args[args.index(key)+1],value)
            self.assertLess(len(args[args.index('-j')+1]),60)
            command=args[args.index('--cmd')+1]
            # Have Bash parse the exact submitted command and capture argv.
            (route/'run.sh').write_text('#!/bin/bash\nprintf "%s\\n" "$0" "$@" > "$HRM_TEST_RUNTIME_ARGS"\n',encoding='utf-8',newline='\n')
            env['HRM_TEST_RUNTIME_ARGS']=str(root/'runtime.txt')
            r=subprocess.run([bash,'-c',command],cwd=code,env=env,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
            runtime=(root/'runtime.txt').read_text().splitlines()
            self.assertEqual(runtime[1],'formal');self.assertEqual(runtime[3],output.as_posix());self.assertEqual(runtime[4],'')
            self.assertEqual(runtime[5],text.as_posix());self.assertEqual(runtime[-1],'4')
            py=(root/'python.txt').read_text();self.assertIn('check_text_assets.py',py);self.assertNotIn('audio_initialization.py',py)
            (root/'vc.txt').unlink()
            bad=subprocess.run(cmd+['--init-checkpoint','old.ckpt'],env=env,capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0);self.assertFalse((root/'vc.txt').exists())


if __name__=='__main__':unittest.main(verbosity=2)
